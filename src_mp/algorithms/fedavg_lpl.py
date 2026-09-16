from __future__ import annotations

from .core import (
    BaseClientExecutor,
    BaseServer,
    ClientResult,
    ce_loss,
    clone_state,
    pseudo_label_loss,
)
from .core.augment import strong_augment, weak_augment


class Client(BaseClientExecutor):
    """FedAvg-LPL (Local Pseudo-Labeling) 客户端。

    使用本地模型当前的弱增强预测作为 Teacher 产生伪标签（仅对高置信度样本），
    并在强增强视图上监督本地模型迭代更新。
    """

    def __init__(self, args, device, num_class, **kwargs) -> None:
        super().__init__(args, device, num_class, **kwargs)
        self.conf: float = args.conf
        self.lambda_u: float = args.lambda_u

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

            # 2. 本地 Teacher 伪标签损失 (弱增强生成伪标签，强增强计算损失)
            loss_u, _ = pseudo_label_loss(
                logits_student=self.model(x_u_s),
                logits_teacher=self.model(x_u_w),
                conf=self.conf,
            )

            loss = loss_l + self.lambda_u * loss_u
            self.check_nan(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total += loss.item()
            batches += 1
        return total, batches


class Server(BaseServer):
    """FedAvg-LPL Server，使用标准 FedAvg 聚合。"""

    client_cls = Client
    supports_ssl = True

    def apply_result(self, results):
        self.aggregate_model(results)
