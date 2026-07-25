import numpy
import torch

from .aggregate import flattened_matrix_aggregate, param_aggregate, proto_aggregate
from .augment import strong_augment, weak_augment
from .evaluate import evaluate_model, evaluate_prototype
from .fed_utils import BaseServer, evaluate, get_model
from .loss import (
    ce_loss,
    cos_contrastive_loss,
    dist_contrastive_loss,
    kl_loss,
    mse_loss,
    orthogonality_loss,
)
from .topology import compute_mh_weights, generate_adjacency_matrix, sinkhorn_knopp

__all__ = [
    "flattened_matrix_aggregate",
    "param_aggregate",
    "proto_aggregate",
    "strong_augment",
    "weak_augment",
    "compare_model_parameters",
    "evaluate_model",
    "evaluate_prototype",
    "BaseServer",
    "evaluate",
    "get_model",
    "compute_mh_weights",
    "generate_adjacency_matrix",
    "sinkhorn_knopp",
    "cos_contrastive_loss",
    "dist_contrastive_loss",
    "ce_loss",
    "mse_loss",
    "kl_loss",
    "extract_prototypes",
    "orthogonality_loss",
    "fmt_num",
    "mixup",
]


def fmt_num(x):
    """数值统一转字符串：整数不保留 .0，浮点数保留原样"""
    if isinstance(x, float) and x == int(x):
        return str(int(x))
    return str(x)


def compare_model_parameters(params1: dict, params2: dict) -> bool:
    """对比两个结构相同的模型参数字典是否完全相同"""
    if params1.keys() != params2.keys():
        return False
    return all(torch.equal(params1[k].cpu(), params2[k].cpu()) for k in params1)


def extract_prototypes(
    model, loader, num_classes, feature_dim, device, return_counts=False
):
    """
    统一提取本地数据集的类别表征原型。
    返回:
        protos: [num_classes, feature_dim] 的 CPU 张量
        counts: [num_classes] 的 CPU 张量 (仅当 return_counts=True 时返回)
    """
    model.eval()
    proto_sum = torch.zeros((num_classes, feature_dim), device=device)
    proto_count = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            features = model.extractor(x)
            proto_sum.index_add_(0, y, features)
            proto_count += torch.bincount(y, minlength=num_classes)

    active_mask = proto_count > 0
    safe_count = torch.where(active_mask, proto_count, torch.ones_like(proto_count))
    avg_protos = proto_sum / safe_count.unsqueeze(1)
    avg_protos[~active_mask] = 0.0

    protos_cpu = avg_protos.cpu().detach().clone()
    if return_counts:
        return protos_cpu, proto_count.cpu().detach().clone()
    return protos_cpu


def mixup(x1, y1, x2, y2, alpha=1.0):
    """
    对两个输入样本执行 mixup 数据增强。

    从 Beta(alpha, alpha) 分布中采样混合系数 lam，
    对 (x1, y1) 和 (x2, y2) 进行线性插值。

    返回:
        mixed_x: 混合后的特征
        mixed_y1, mixed_y2: 混合前的标签（供损失函数分别加权）
        lam: 混合系数
    """
    lam = numpy.random.beta(alpha, alpha)
    device = x1.device
    lam = torch.tensor(lam, device=device)

    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y1 = y1
    mixed_y2 = y2

    return mixed_x, mixed_y1, mixed_y2, lam
