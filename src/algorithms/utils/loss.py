import torch
import torch.nn.functional as F


def ce_loss(predictions, targets):
    """计算交叉熵损失"""
    return F.cross_entropy(predictions, targets)


def mse_loss(predictions, targets):
    """计算均方误差损失"""
    return F.mse_loss(predictions, targets)


def kl_loss(student_logits, teacher_logits, temperature=1.0):
    """
    计算两个 Logits 之间的 KL 散度损失。
    """
    if temperature != 1.0:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

    student_soft = F.log_softmax(student_logits, dim=1)
    teacher_soft = F.softmax(teacher_logits, dim=1)

    loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (
        temperature**2
    )

    return loss


def cos_contrastive_loss(features, prototypes, labels, temperature=0.1):
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
    sim_matrix = torch.matmul(features_norm, protos_norm.T) / temperature

    # 3. 使用交叉熵计算 InfoNCE Loss
    return ce_loss(sim_matrix, labels)


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
    return ce_loss(-dist, labels)


def orthogonality_loss(vectors):
    """
    正交性损失 (Orthogonality Loss)。
    确保向量组（如原型）在特征空间中互不重叠，防止特征坍缩。
    """
    device = vectors.device
    num_vectors = vectors.shape[0]

    # 计算自相关矩阵 [N, N]
    logits = torch.matmul(vectors, vectors.T)
    # 目标是使对角线元素（自相关）最大
    labels = torch.arange(num_vectors, device=device)

    return ce_loss(logits, labels)
