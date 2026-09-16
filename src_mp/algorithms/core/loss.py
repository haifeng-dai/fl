"""src_mp 公共损失函数。"""

import torch
import torch.nn.functional as F


def ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """分类交叉熵。"""
    return F.cross_entropy(logits, targets)


def kl_loss(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """计算 teacher 分布到 student 分布的 KL 散度。"""
    if tau != 1.0:
        logits_student = logits_student / tau
        logits_teacher = logits_teacher / tau
    return F.kl_div(
        F.log_softmax(logits_student, dim=1),
        F.softmax(logits_teacher, dim=1),
        reduction="batchmean",
    ) * (tau**2)
