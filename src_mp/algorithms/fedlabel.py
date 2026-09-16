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
    """FedLabel 客户端 (ICCV 2023)。

    核心机制：
    1. 自适应专家选择：同时使用本地模型与全局模型作为 Teacher，动态选取对当前样本置信度更高的模型作为胜出专家；
    2. 阈值过滤：仅对胜出模型置信度 >= conf 的样本赋予硬伪标签；
    3. 全局-局部一致性正则化：当落选专家与胜出专家预测分类一致时（达成共识），
       通过加权 KL 散度吸纳落选专家的软分布知识。
    """

    def __init__(self, args, device, num_class, **kwargs) -> None:
        super().__init__(args, device, num_class, **kwargs)
        self.conf: float = args.conf
        self.lambda_u: float = args.lambda_u
        self.lambda_0: float = getattr(args, "lambda_0", 1.0)
        self.model_g = build_model(self.model_param).to(self.device)
        self.model_g.eval()
        for parameter in self.model_g.parameters():
            parameter.requires_grad_(False)

    def train(self):
        self.model.train()
        # 同步每轮任务传入的全局模型状态到 model_g 作为冻结 Global Teacher
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

            # 2. Local 与 Global Teacher 预测与置信度评估
            with torch.no_grad():
                probs_l = torch.softmax(self.model(x_u_w), dim=1)
                probs_g = torch.softmax(self.model_g(x_u_w), dim=1)
                max_p_l, pred_l = probs_l.max(dim=1)
                max_p_g, pred_g = probs_g.max(dim=1)

                # 自适应选择置信度更高的模型作为胜出专家
                local_is_better = max_p_l >= max_p_g
                selected_max_p = torch.where(local_is_better, max_p_l, max_p_g)
                selected_targets = torch.where(local_is_better, pred_l, pred_g)
                discarded_max_p = torch.where(local_is_better, max_p_g, max_p_l)
                discarded_preds = torch.where(local_is_better, pred_g, pred_l)
                discarded_probs = torch.where(
                    local_is_better.unsqueeze(1), probs_g, probs_l
                )

                # 置信度阈值过滤
                mask_conf = selected_max_p.ge(self.conf)
                # 一致性共识掩码：高置信且落选专家预测分类一致
                mask_agree = mask_conf & (discarded_preds == selected_targets)

            # 3. 强增强数据前向与伪标签交叉熵损失
            logits_u_s = self.model(x_u_s)
            if mask_conf.any():
                loss_ce = F.cross_entropy(
                    logits_u_s[mask_conf], selected_targets[mask_conf]
                )
            else:
                loss_ce = torch.tensor(0.0, device=self.device)

            # 4. 全局-局部一致性正则化 (Global-Local Consistency Regularization)
            if mask_agree.any():
                # 动态权重：lambda_0 * (discarded_score / selected_score)
                lambda_weight = (
                    self.lambda_0
                    * (
                        discarded_max_p[mask_agree]
                        / selected_max_p[mask_agree].clamp(min=1e-6)
                    )
                ).unsqueeze(1)

                log_probs_s = F.log_softmax(logits_u_s[mask_agree], dim=1)
                loss_reg_sample = F.kl_div(
                    log_probs_s,
                    discarded_probs[mask_agree],
                    reduction="none",
                ).sum(dim=1, keepdim=True)
                loss_reg = (lambda_weight * loss_reg_sample).mean()
            else:
                loss_reg = torch.tensor(0.0, device=self.device)

            loss_u = loss_ce + loss_reg
            loss = loss_l + self.lambda_u * loss_u
            self.check_nan(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total += loss.item()
            batches += 1
        return total, batches


class Server(BaseServer):
    """FedLabel Server，使用标准 FedAvg 聚合。"""

    client_cls = Client
    supports_ssl = True

    def apply_result(self, results):
        self.aggregate_model(results)
