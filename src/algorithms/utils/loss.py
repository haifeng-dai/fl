import torch
import torch.nn.functional as F


def kl_loss(logits_s, logits_t, tau=1.0):
    """计算两个 Logits 之间的 KL 散度损失。"""
    if tau != 1.0:
        logits_s = logits_s / tau
        logits_t = logits_t / tau

    log_p_s = F.log_softmax(logits_s, dim=1)
    p_t = F.softmax(logits_t, dim=1)

    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau**2)


def masked_kl_loss(logits, targets, mask=None):
    """计算逐样本 KL 散度，并可按样本掩码后求 batch 平均。"""
    losses = F.kl_div(
        F.log_softmax(logits, dim=1),
        targets,
        reduction="none",
    ).sum(dim=1)
    if mask is not None:
        losses = losses * mask
    return losses.mean()


def cos_similarity(features, prototypes, labels, tau=0.1):
    """
    向量化优化的余弦对比损失 (Cosine Contrastive Loss)。
    利用原型作为 Anchor，拉近同类样本，推开异类样本。
    """
    device = features.device
    prototypes = prototypes.to(device)

    # 1. 特征与原型标准化 (L2 Normalize)
    features_norm = F.normalize(features, p=2, dim=1)
    protos_norm = F.normalize(prototypes, p=2, dim=1)

    # 2. 计算余弦相似度矩阵 [BatchSize, NumClasses]
    sim_matrix = torch.matmul(features_norm, protos_norm.T) / tau

    # 3. 使用交叉熵计算 InfoNCE Loss
    return F.cross_entropy(sim_matrix, labels)


def dist_contrastive_loss(features, prototypes, labels, margin=0.0):
    """
    基于欧式距离的对比损失 (Distance-based Contrastive Loss)。
    支持可选的 Margin 参数以增强类间边界。
    """
    device = features.device
    num_classes = prototypes.shape[0]

    # 1. 计算欧式距离矩阵 [BatchSize, NumClasses]
    dist = torch.cdist(features, prototypes.to(device), p=2.0)

    # 2. 如果有 Margin，则在正样本（Label 对应项）上加上 Margin 以增加距离（推开）
    if margin > 0:
        one_hot = F.one_hot(labels, num_classes).to(device)
        dist = dist + one_hot * margin

    # 3. 距离越小 Logits 越高，因此取负号
    return F.cross_entropy(-dist, labels)


def orthogonality_loss(features, prototypes=None, labels=None):
    """
    通用正交性损失。
    1. 如果仅传入 features: 确保 features 内部各向量互相正交（防止特征坍缩）。
    2. 如果传入 features, prototypes 和 labels: 确保每个 feature 仅与其对应类别的 prototype 对齐，与其他 prototype 正交。
    """
    device = features.device
    if prototypes is None:
        # Case 1: 内部正交 (原逻辑)
        logits = torch.matmul(features, features.T)
        target_labels = torch.arange(features.shape[0], device=device)
    else:
        # Case 2: 特征-原型正交
        prototypes = prototypes.to(device)
        # 计算特征与原型的点积矩阵 [BatchSize, NumClasses]
        logits = torch.matmul(features, prototypes.T)
        target_labels = labels

    return F.cross_entropy(logits, target_labels)
