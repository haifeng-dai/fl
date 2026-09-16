"""src_mp 公共损失函数库。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """分类交叉熵。支持硬标签 (class indices) 与软标签 (probability distribution)。"""
    return F.cross_entropy(logits, targets)


def soft_ce_loss(
    logits: torch.Tensor,
    targets_prob: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """软标签交叉熵损失。

    数学性质：当目标分布 P 冻结（即无梯度）时，目标分布与模型预测分布之间的
    KL 散度 D_KL(P || Q) 与交叉熵 H(P, Q) 对模型参数的梯度完全等价：
        ∇_θ D_KL(P || Q_θ) = ∇_θ H(P, Q_θ)
    相比 PyTorch 的 kl_div，直接调用 soft_ce_loss 运算更简洁且无需额外计算目标分布的负熵。
    """
    if mask is not None:
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)
        logits = logits[mask]
        targets_prob = targets_prob[mask]
    return F.cross_entropy(logits, targets_prob)


def kl_loss(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """计算 Teacher 分布到 Student 分布的带温度系数 KL 散度（知识蒸馏/分布对齐）。"""
    if tau != 1.0:
        logits_student = logits_student / tau
        logits_teacher = logits_teacher / tau
    return F.kl_div(
        F.log_softmax(logits_student, dim=1),
        F.softmax(logits_teacher.detach(), dim=1),
        reduction="batchmean",
    ) * (tau**2)


def pseudo_label_loss(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    conf: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """标准 FixMatch 风格无标签伪标签损失。

    逻辑：
    1. 使用 Teacher 预测的弱增强 logits 生成概率分布；
    2. 提取最大概率并进行 conf 阈值过滤 (max_probs >= conf)；
    3. 在高置信度子集上计算 Student 强增强 logits 的交叉熵。

    参数:
        logits_student: Student 模型在强增强视图下的 Logits [B, C]。
        logits_teacher: Teacher 模型（本地或全局）在弱增强视图下的 Logits [B, C]。
        conf: 置信度阈值 τ (如 0.95)。

    返回:
        (loss_u, mask): 伪标签损失 (Tensor) 与高置信度样本掩码 (Bool Tensor)。
    """
    with torch.no_grad():
        probs_u = torch.softmax(logits_teacher.detach(), dim=-1)
        max_probs, pseudo_targets = probs_u.max(dim=-1)
        mask = max_probs.ge(conf)

    if mask.any():
        loss_u = F.cross_entropy(logits_student[mask], pseudo_targets[mask])
    else:
        loss_u = torch.tensor(0.0, device=logits_student.device)
    return loss_u, mask


def cos_similarity(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """向量化优化的余弦对比损失 (Cosine Contrastive Loss / InfoNCE)。

    利用原型作为 Anchor，拉近同类样本，推开异类样本。
    """
    features_norm = F.normalize(features, p=2, dim=1)
    protos_norm = F.normalize(prototypes.to(features.device), p=2, dim=1)
    sim_matrix = torch.matmul(features_norm, protos_norm.T) / tau
    return F.cross_entropy(sim_matrix, labels)


def dist_contrastive_loss(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """基于欧氏距离的对比损失 (Distance-based Contrastive Loss)。

    支持可选的 Margin 参数以增强类间边界。
    """
    device = features.device
    num_classes = prototypes.shape[0]
    dist = torch.cdist(features, prototypes.to(device), p=2.0)
    if margin > 0:
        one_hot = F.one_hot(labels, num_classes).to(device)
        dist = dist + one_hot * margin
    return F.cross_entropy(-dist, labels)


def orthogonality_loss(
    features: torch.Tensor,
    prototypes: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """特征正交性约束损失。

    1. 若未传入 prototypes: 确保 features 内部各样本特征互相正交，防止表示坍缩；
    2. 若传入 prototypes & labels: 确保样本特征仅与其类别原型对齐，与其他原型正交。
    """
    device = features.device
    if prototypes is None:
        logits = torch.matmul(features, features.T)
        target_labels = torch.arange(features.shape[0], device=device)
    else:
        logits = torch.matmul(features, prototypes.to(device).T)
        target_labels = labels
    return F.cross_entropy(logits, target_labels)
