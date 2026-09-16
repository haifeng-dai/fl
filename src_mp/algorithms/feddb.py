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
    """FedDB 客户端 (IJCAI 2024)。

    核心机制：
    1. 统计无标签数据的平均预测分布 (APP-U) p_s，捕捉本地类别先验偏斜；
    2. 基于贝叶斯法则 P(Y|X) / P(Y) 调整预测概率分布，消除 Non-IID 先验偏差；
    3. 在去偏后的概率分布上进行置信度门控与硬伪标签计算。
    """

    def __init__(self, args, device, num_class, **kwargs) -> None:
        super().__init__(args, device, num_class, **kwargs)
        self.conf: float = args.conf
        self.lambda_u: float = args.lambda_u
        self.p_s = torch.ones(self.num_class, device=self.device) / self.num_class

    def train(self):
        self.model.train()
        total, batches = 0.0, 0
        for _ in range(self.epochs):
            total, batches = self.run_epoch(total, batches)
        return ClientResult(
            self.client_id,
            total / max(1, batches),
            clone_state(self.model.state_dict()),
            {"p_s": self.p_s.detach().cpu().clone()},
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

            # 2. 弱增强预测分布与动量 APP-U 先验估计
            with torch.no_grad():
                logits_u_w = self.model(x_u_w)
                probs_u_w = torch.softmax(logits_u_w, dim=-1)
                batch_p_s = probs_u_w.mean(dim=0)
                self.p_s = 0.9 * self.p_s + 0.1 * batch_p_s

                # 3. 贝叶斯去偏后验修正: P_debiased(Y=c|X) ∝ P(Y=c|X) / P(Y=c)
                p_s_clamped = self.p_s.clamp(min=1e-6)
                debiased_probs = probs_u_w / p_s_clamped.unsqueeze(0)
                debiased_probs = debiased_probs / debiased_probs.sum(
                    dim=-1, keepdim=True
                ).clamp(min=1e-6)

                max_probs, pseudo_targets = debiased_probs.max(dim=-1)
                mask = max_probs.ge(self.conf)

            # 4. 强增强数据去偏伪标签交叉熵
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
    """FedDB Server，通过各客户端 APP-U 求解使全局混合先验趋向均匀分布的去偏聚合权重。"""

    client_cls = Client
    supports_ssl = True

    def apply_result(self, results):
        client_ps_list = [results[cid].payload["p_s"] for cid in self.selected]
        client_ps = torch.stack(client_ps_list)  # [K, NumClasses]
        num_selected = len(self.selected)

        # 优化聚合权重 alpha 使全局混合分布逼近均匀分布
        weight = torch.ones(num_selected, 1, dtype=torch.float32, requires_grad=True)
        target_ps = torch.ones(self.num_class, dtype=torch.float32) / self.num_class

        optimizer = torch.optim.SGD([weight], lr=1.0, momentum=0.9)
        for _ in range(50):
            norm_weight = torch.softmax(weight, dim=0)
            pred_global_p = (client_ps * norm_weight).sum(dim=0)
            loss = F.mse_loss(pred_global_p, target_ps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        debiased_weights = torch.softmax(weight.detach(), dim=0).view(-1).tolist()
        self.aggregate_model(results, weights=debiased_weights)
