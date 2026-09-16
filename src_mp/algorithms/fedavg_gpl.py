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
from .core.model import build_model


class Client(BaseClientExecutor):
    """FedAvg-GPL (Global Pseudo-Labeling) 客户端。

    使用每轮下发并冻结的全局模型作为 Teacher，为无标签弱增强视图生成高置信度伪标签，
    并在强增强视图上监督本地模型迭代更新。
    """

    def __init__(self, args, device, num_class, **kwargs) -> None:
        super().__init__(args, device, num_class, **kwargs)
        self.conf: float = args.conf
        self.lambda_u: float = args.lambda_u
        self.model_g = build_model(self.model_param).to(self.device)
        self.model_g.eval()
        for parameter in self.model_g.parameters():
            parameter.requires_grad_(False)

    def train(self):
        self.model.train()
        # 同步每轮任务传入的全局模型状态到 model_g 作为冻结 Teacher
        self.model_g.load_state_dict(clone_state(self.model.state_dict()))
        self.model_g.eval()

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

            # 2. 全局 Teacher 伪标签损失 (弱增强生成伪标签，强增强计算损失)
            loss_u, _ = pseudo_label_loss(
                logits_student=self.model(x_u_s),
                logits_teacher=self.model_g(x_u_w),
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
    """FedAvg-GPL Server，使用标准 FedAvg 聚合。"""

    client_cls = Client
    supports_ssl = True

    def apply_result(self, results):
        self.aggregate_model(results)
