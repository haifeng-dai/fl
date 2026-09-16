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
    """FedLoKe 客户端。

    核心机制：
    1. 本地常驻 EMA 模型 (Local Knowledge Reservoir)：
       每轮在接收到的全局模型与客户端历史 local_model 之间进行平滑 EMA 融合；
    2. 双向互助蒸馏与熵门控 (Bidirectional Cross-Distillation with Entropy Gating)：
       - 当 Global 预测熵 < threshold 时，Global 指导 Local 学习（全局规范化）；
       - 当 Local 预测熵 < threshold 时，Local 指导 Global 学习（局部特异性知识反哺全局！）。
    """

    def __init__(self, args, device, num_class, **kwargs) -> None:
        super().__init__(args, device, num_class, **kwargs)
        self.lambda_u: float = args.lambda_u
        self.ema_weight: float = getattr(args, "ema_weight", 0.95)
        self.entropy_threshold: float = getattr(args, "entropy_threshold", 0.5)

        self.global_model = self.model
        self.local_model = build_model(self.model_param).to(self.device)
        self.local_optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.global_optimizer = self.optimizer

    def reset_local_optimizer(self):
        for group in self.local_optimizer.param_groups:
            group.update(
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        self.local_optimizer.state.clear()

    def train(self):
        self.global_model.train()
        self.local_model.train()

        # 1. 本地模型 EMA 融合
        prev_local_state = self.payload
        curr_g_state = self.global_model.state_dict()
        if prev_local_state is None:
            self.local_model.load_state_dict(clone_state(curr_g_state))
        else:
            ema_state = {}
            for k, v_prev in prev_local_state.items():
                if v_prev.dtype.is_floating_point:
                    ema_state[k] = (
                        self.ema_weight * v_prev.to(self.device)
                        + (1.0 - self.ema_weight) * curr_g_state[k]
                    )
                else:
                    ema_state[k] = curr_g_state[k].clone()
            self.local_model.load_state_dict(ema_state)

        self.reset_optimizer()
        self.reset_local_optimizer()

        total, batches = 0.0, 0
        for _ in range(self.epochs):
            total, batches = self.run_epoch(total, batches)

        return ClientResult(
            self.client_id,
            total / max(1, batches),
            clone_state(self.global_model.state_dict()),
            {"local_state": clone_state(self.local_model.state_dict())},
        )

    def run_epoch(self, total: float, batches: int) -> tuple[float, int]:
        for (x_l, y_l), (x_u, *_) in self.get_ssl_loaders():
            x_l, y_l = x_l.to(self.device), y_l.to(self.device)
            x_u = x_u.to(self.device)

            x_l_w = weak_augment(x_l, self.dataset)
            x_l_s = strong_augment(x_l, self.dataset)
            x_u_w = weak_augment(x_u, self.dataset)
            x_u_s = strong_augment(x_u, self.dataset)

            # 1. 双模型有监督损失
            local_logits_l = self.local_model(x_l_w)
            global_logits_l = self.global_model(x_l_s)
            loss_label = ce_loss(local_logits_l, y_l) + ce_loss(global_logits_l, y_l)

            # 2. 弱增强预测分布与香农熵门控
            with torch.no_grad():
                p_local_w = torch.softmax(self.local_model(x_u_w), dim=-1)
                p_global_w = torch.softmax(self.global_model(x_u_w), dim=-1)
                entropy_local = -(
                    p_local_w * torch.log(p_local_w + 1e-8)
                ).sum(dim=-1)
                entropy_global = -(
                    p_global_w * torch.log(p_global_w + 1e-8)
                ).sum(dim=-1)
                mask_local = entropy_local < self.entropy_threshold
                mask_global = entropy_global < self.entropy_threshold

            # 3. 强增强前向与双向互助蒸馏
            logits_local_s = self.local_model(x_u_s)
            logits_global_s = self.global_model(x_u_s)

            # Global -> Local 蒸馏 (全局知识规范化)
            if mask_global.any():
                loss_g2l = -(
                    p_global_w[mask_global]
                    * F.log_softmax(logits_local_s[mask_global], dim=-1)
                ).sum(dim=-1).mean()
            else:
                loss_g2l = torch.tensor(0.0, device=self.device)

            # Local -> Global 蒸馏 (局部特异性知识反哺全局)
            if mask_local.any():
                loss_l2g = -(
                    p_local_w[mask_local]
                    * F.log_softmax(logits_global_s[mask_local], dim=-1)
                ).sum(dim=-1).mean()
            else:
                loss_l2g = torch.tensor(0.0, device=self.device)

            loss_unlabel = loss_g2l + loss_l2g
            loss = loss_label + self.lambda_u * loss_unlabel
            self.check_nan(loss)

            self.global_optimizer.zero_grad()
            self.local_optimizer.zero_grad()
            loss.backward()
            self.global_optimizer.step()
            self.local_optimizer.step()

            total += loss.item()
            batches += 1
        return total, batches


class Server(BaseServer):
    """FedLoKe Server，维护客户端专属 local_model EMA 状态并聚合全局模型。"""

    client_cls = Client
    supports_ssl = True

    def __init__(self, args, devices):
        super().__init__(args, devices)
        self.local_client_states: dict[int, dict[str, torch.Tensor]] = {}

    def train_payloads(self):
        return {cid: self.local_client_states.get(cid) for cid in self.selected}

    def apply_result(self, results):
        for cid in self.selected:
            self.local_client_states[cid] = results[cid].payload["local_state"]
        self.aggregate_model(results)
