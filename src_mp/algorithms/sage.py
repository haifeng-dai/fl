import math

import torch
import torch.nn.functional as F
from torch.nn.functional import one_hot

from .core import BaseClientExecutor, BaseServer, ClientResult, ce_loss, clone_state
from .core.augment import strong_augment, weak_augment
from .core.model import build_model


class Client(BaseClientExecutor):
    def __init__(self, args, device, num_class) -> None:
        super().__init__(args, device, num_class)
        self.dataset = args.dataset
        self.conf = args.conf
        self.kappa: float = math.log(2.0) / 0.05
        self.lambda_u = args.lambda_u

        self.model_g = build_model(self.model_param).to(self.device)
        self.model_g.eval()
        for parameter in self.model_g.parameters():
            parameter.requires_grad_(False)

    def train(self):
        self.model.train()
        self.model_g.load_state_dict(clone_state(self.model.state_dict()))
        total, batches = 0.0, 0
        for _ in range(self.epochs):
            total, batches = self.run_epoch(total, batches)
        return ClientResult(
            self.client_id,
            total / max(1, batches),
            clone_state(self.model.state_dict()),
        )

    def run_epoch(self, total: float, batches: int) -> tuple[float, int]:
        for x, y, _, is_labeled in self.loader:
            x = x.to(self.device)
            y = y.to(self.device)
            is_labeled = is_labeled.to(self.device).bool()

            x_l, y_l = x[is_labeled], y[is_labeled]
            x_u = x[~is_labeled]

            # 与模型计算图连接的 float 标量；避免 L/U 均无有效样本时
            # uint8 零值无法执行 backward。
            loss = self.model.classifier.weight.sum() * 0.0

            if x_l.size(0) > 0:
                # 有标签样本使用弱增强，并在增强后完成归一化。
                x_l_w = weak_augment(x_l, self.dataset)
                logits_l = self.model(x_l_w)
                loss = loss + ce_loss(logits_l, y_l)

            if x_u.size(0) > 0:
                x_u_w = weak_augment(x_u, self.dataset)
                valid_indices, confidence_l, confidence_g, targets_l, targets_g = (
                    self.confident_samples(x_u_w)
                )

                if valid_indices.numel() > 0:
                    targets = self.build_pseudo_label(
                        confidence_l,
                        confidence_g,
                        targets_l,
                        targets_g,
                    )
                    x_u_s = strong_augment(x_u[valid_indices], self.dataset)
                    logits_u = self.model(x_u_s)
                    loss_u = F.kl_div(
                        F.log_softmax(logits_u, dim=1),
                        targets,
                        reduction="none",
                    ).sum(dim=1)
                    loss = loss + self.lambda_u * loss_u.mean()

            self.check_nan(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total += loss.item()
            batches += 1

        return total, batches

    @torch.no_grad()
    def confident_samples(self, x_u_w: torch.Tensor):
        """根据 weak view 的 local/global 置信度筛选无标签样本。"""
        confidence_g, targets_g = torch.softmax(self.model_g(x_u_w), dim=1).max(dim=1)
        confidence_l, targets_l = torch.softmax(self.model(x_u_w), dim=1).max(dim=1)
        valid_indices = torch.where(
            confidence_l.ge(self.conf) | confidence_g.ge(self.conf)
        )[0]
        return (
            valid_indices,
            confidence_l[valid_indices],
            confidence_g[valid_indices],
            targets_l[valid_indices],
            targets_g[valid_indices],
        )

    def build_pseudo_label(
        self,
        confidence_l: torch.Tensor,
        confidence_g: torch.Tensor,
        targets_l: torch.Tensor,
        targets_g: torch.Tensor,
    ) -> torch.Tensor:
        """仅为已经筛选出的高置信度样本构造修正伪标签。"""
        mask_l = confidence_l.ge(self.conf)
        mask_g = confidence_g.ge(self.conf)
        # Eq. (2)：local/global 置信差
        delta_c = (confidence_l - confidence_g).abs()
        # Eq. (3)：基于置信差的修正权重
        correction = torch.exp(-self.kappa * delta_c)
        # Eq. (4)/(5)：local/global 硬伪标签
        delta_l = one_hot(targets_l, self.num_class).float()
        delta_g = one_hot(targets_g, self.num_class).float()
        # Eqs. (6) & (7-1)：local/global 软伪标签修正
        targets = correction.unsqueeze(1) * delta_l
        targets += (1.0 - correction).unsqueeze(1) * delta_g
        # Eq. (7-2)：仅 global 达到阈值时使用 global 硬伪标签
        g_only = ~mask_l.bool() & mask_g.bool()
        targets[g_only] = delta_g[g_only]
        return targets


class Server(BaseServer):
    client_cls = Client
    supports_ssl = True

    def apply_result(self, results):
        self.aggregate_model(results)
