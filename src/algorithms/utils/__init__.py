import torch

from .aggregate import flattened_matrix_aggregate, param_aggregate, proto_aggregate
from .augment import strong_augment, weak_augment
from .evaluate import evaluate_model, evaluate_prototype
from .fed_utils import BaseParams, BaseServer, evaluate, get_model
from .input import (
    DATASET_SPECS,
    normalize_dataset,
    normalize_image_tensor,
    prepare_input_batch,
)
from .loss import (
    cos_similarity,
    dist_contrastive_loss,
    kl_loss,
    masked_kl_loss,
    orthogonality_loss,
)
from .topology import compute_mh_weights, generate_adjacency_matrix, sinkhorn_knopp

__all__ = [
    "DATASET_SPECS",
    "BaseParams",
    "BaseServer",
    "clone_cpu_state",
    "compare_model_parameters",
    "compute_mh_weights",
    "cos_similarity",
    "dist_contrastive_loss",
    "evaluate",
    "evaluate_model",
    "evaluate_prototype",
    "extract_prototypes",
    "flattened_matrix_aggregate",
    "fmt_num",
    "generate_adjacency_matrix",
    "get_model",
    "kl_loss",
    "masked_kl_loss",
    "normalize_dataset",
    "normalize_image_tensor",
    "orthogonality_loss",
    "param_aggregate",
    "prepare_input_batch",
    "proto_aggregate",
    "sinkhorn_knopp",
    "strong_augment",
    "weak_augment",
]


def clone_cpu_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """返回完全在 CPU、无梯度且独立存储的 state_dict 副本。

    用于客户端训练结束后的模型状态导出，防止 Ray 对象存储出现悬空引用。
    """
    return {k: v.cpu().detach().clone() for k, v in state.items()}


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
