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
from .core.model import build_model


class Client(BaseClientExecutor):
    """FedDure 客户端 (AAAI 2024)。

    核心机制：
    1. 双调节器 (Dual Regulators) 与元伪标签 (Meta Pseudo-Labeling, MPL)：
       - 细粒度调节器 (F-reg)：Teacher 模型在弱增强无标签数据上生成自适应软/硬伪标签；
       - Student 学习：Student 模型利用 Teacher 伪标签进行参数更新；
       - 粗粒度调节器 (C-reg)：计算 Student 更新前后在本地有标签数据上的验证损失差作为反馈梯度，
         反向指导 Teacher 模型优化其伪标签生成策略。
    """

    def __init__(self, args, device, num_class, **kwargs) -> None:
        super().__init__(args, device, num_class, **kwargs)
        self.conf: float = args.conf
        self.lambda_u: float = args.lambda_u
        self.teacher_model = build_model(self.model_param).to(self.device)
        self.student_model = self.model
        self.optimizer_teacher = torch.optim.SGD(
            self.teacher_model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.optimizer_student = self.optimizer

    def reset_teacher_optimizer(self):
        for group in self.optimizer_teacher.param_groups:
            group.update(
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        self.optimizer_teacher.state.clear()

    def train(self):
        self.teacher_model.train()
        self.student_model.train()
        # 同步每轮下发的全局模型状态
        self.teacher_model.load_state_dict(clone_state(self.model.state_dict()))
        self.student_model.load_state_dict(clone_state(self.model.state_dict()))
        self.reset_optimizer()
        self.reset_teacher_optimizer()

        total, batches = 0.0, 0
        for _ in range(self.epochs):
            total, batches = self.run_epoch(total, batches)
        return ClientResult(
            self.client_id,
            total / max(1, batches),
            clone_state(self.teacher_model.state_dict()),
        )

    def run_epoch(self, total: float, batches: int) -> tuple[float, int]:
        for (x_l, y_l), (x_u, *_) in self.get_ssl_loaders():
            x_l, y_l = x_l.to(self.device), y_l.to(self.device)
            x_u = x_u.to(self.device)

            x_l_w = weak_augment(x_l, self.dataset)
            x_u_w = weak_augment(x_u, self.dataset)
            x_u_s = strong_augment(x_u, self.dataset)

            # -------------------------------------------------------------
            # 1. Teacher 前向传播与 UDA 损失 (有监督 + 无监督伪标签)
            # -------------------------------------------------------------
            t_logits_l = self.teacher_model(x_l_w)
            t_loss_l = ce_loss(t_logits_l, y_l)

            t_logits_uw = self.teacher_model(x_u_w)
            t_logits_us = self.teacher_model(x_u_s)

            soft_pseudo_label = torch.softmax(t_logits_uw.detach(), dim=-1)
            max_probs, targets_u = soft_pseudo_label.max(dim=-1)
            mask = max_probs.ge(self.conf)

            if mask.any():
                t_loss_u = -(
                    soft_pseudo_label[mask]
                    * F.log_softmax(t_logits_us[mask], dim=-1)
                ).sum(dim=-1).mean()
            else:
                t_loss_u = torch.tensor(0.0, device=self.device)

            t_loss_uda = t_loss_l + self.lambda_u * t_loss_u

            # -------------------------------------------------------------
            # 2. Student 依据 Teacher 伪标签进行一步参数更新
            # -------------------------------------------------------------
            s_logits_l_old = self.student_model(x_l_w)
            s_loss_l_old = ce_loss(s_logits_l_old.detach(), y_l)

            s_logits_u = self.student_model(x_u_w)
            if mask.any():
                s_loss = F.cross_entropy(s_logits_u[mask], targets_u[mask])
                self.optimizer_student.zero_grad()
                s_loss.backward()
                self.optimizer_student.step()

            # -------------------------------------------------------------
            # 3. 粗粒度反馈 (C-reg) 与 Teacher 元更新 (MPL Step)
            # -------------------------------------------------------------
            with torch.no_grad():
                s_logits_l_new = self.student_model(x_l_w)
                s_loss_l_new = ce_loss(s_logits_l_new, y_l)

            # 评估 Student 在有标签数据上的质量变化
            dot_product = (s_loss_l_new - s_loss_l_old).detach()

            if mask.any():
                t_loss_mpl = dot_product * F.cross_entropy(
                    t_logits_us[mask], targets_u[mask]
                )
            else:
                t_loss_mpl = torch.tensor(0.0, device=self.device)

            t_loss = t_loss_uda + t_loss_mpl
            self.check_nan(t_loss)

            self.optimizer_teacher.zero_grad()
            t_loss.backward()
            self.optimizer_teacher.step()

            total += t_loss.item()
            batches += 1
        return total, batches


class Server(BaseServer):
    """FedDure Server，使用标准 FedAvg 聚合 Teacher 模型。"""

    client_cls = Client
    supports_ssl = True

    def apply_result(self, results):
        self.aggregate_model(results)
