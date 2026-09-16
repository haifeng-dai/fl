from __future__ import annotations

import torch
import torch.nn.functional as F

from .core import (
    BaseClientExecutor,
    BaseServer,
    ClientResult,
    ce_loss,
    clone_state,
)
from .core.augment import strong_augment, weak_augment


class Client(BaseClientExecutor):
    """FedAvg-FlexMatch 客户端 (NeurIPS 2021)。

    核心机制：
    课程伪标签 (Curriculum Pseudo-Labeling, CPL)：
    动态统计各类别已越过置信度阈值的样本数量，为困难类与少见类动态下调门控阈值，
    解决 Non-IID 下长尾类别伪标签难以被采纳的问题。
    """

    def __init__(self, args, device, num_class, **kwargs) -> None:
        super().__init__(args, device, num_class, **kwargs)
        self.conf: float = args.conf
        self.lambda_u: float = args.lambda_u
        self.class_counts = torch.zeros(self.num_class, device=self.device)

    def train(self):
        self.model.train()
        total, batches = 0.0, 0
        for _ in range(self.epochs):
            total, batches = self.run_epoch(total, batches)
        return ClientResult(
            self.client_id,
            total / max(1, batches),
            clone_state(self.model.state_dict()),
        )

    def run_epoch(self, total: float, batches: int) -> tuple[float, int]:
        for (x_l, y_l), (x_u, *_) in self.get_ssl_loaders():
            x_l, y_l = x_l.to(self.device), y_l.to(self.device)
            x_u = x_u.to(self.device)

            x_l_w = weak_augment(x_l, self.dataset)
            x_u_w = weak_augment(x_u, self.dataset)
            x_u_s = strong_augment(x_u, self.dataset)

            # 1. 有标签数据监督损失
            logits_l = self.model(x_l_w)
            loss_l = ce_loss(logits_l, y_l)

            # 2. 弱增强预测分布 & 动态类别阈值 (CPL)
            with torch.no_grad():
                probs_u = torch.softmax(self.model(x_u_w), dim=-1)
                max_probs, pseudo_targets = probs_u.max(dim=-1)

                # 计算当前各类别成熟度 beta_c 与动态阈值 tau_c
                max_count = self.class_counts.max().clamp(min=1.0)
                beta = self.class_counts / max_count
                dynamic_thresholds = self.conf * (beta / (2.0 - beta))  # [NumClasses]

                # 样本级动态门控判定
                sample_thresholds = dynamic_thresholds[pseudo_targets]
                mask = max_probs.ge(sample_thresholds)

                # 更新类别计数器
                if mask.any():
                    accepted_targets = pseudo_targets[mask]
                    self.class_counts.scatter_add_(
                        0,
                        accepted_targets,
                        torch.ones_like(accepted_targets, dtype=torch.float32),
                    )

            # 3. 强增强数据伪标签损失
            logits_u_s = self.model(x_u_s)
            if mask.any():
                loss_u = F.cross_entropy(logits_u_s[mask], pseudo_targets[mask])
            else:
                loss_u = torch.tensor(0.0, device=self.device)

            loss = loss_l + self.lambda_u * loss_u
            self.check_nan(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total += loss.item()
            batches += 1
        return total, batches


class Server(BaseServer):
    """FedAvg-FlexMatch Server，使用标准 FedAvg 聚合。"""

    client_cls = Client
    supports_ssl = True

    def apply_result(self, results):
        self.aggregate_model(results)
