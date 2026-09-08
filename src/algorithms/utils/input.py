import torch
from torchvision.transforms import v2

DATASET_SPECS: dict[str, dict] = {
    "cifar10": {
        "kind": "image",
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "cifar10_dg": {
        "kind": "image",
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "cifar100": {
        "kind": "image",
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "cinic10": {
        "kind": "image",
        "mean": (0.47889522, 0.47227842, 0.43047404),
        "std": (0.24205776, 0.23828046, 0.25874835),
    },
    "svhn": {
        "kind": "image",
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
    },
    "mnist": {
        "kind": "image",
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "fashionmnist": {
        "kind": "image",
        "mean": (0.2860,),
        "std": (0.3530,),
    },
    "femnist": {
        "kind": "image",
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "emnist": {
        "kind": "image",
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "gtsrb": {
        "kind": "image",
        "mean": (0.3337, 0.3064, 0.3171),
        "std": (0.2672, 0.2564, 0.2629),
    },
    "tiny_imagenet": {
        "kind": "image",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "cars": {
        "kind": "image",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "flowers102": {
        "kind": "image",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "pacs": {
        "kind": "image",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "officehome": {
        "kind": "image",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "domainnet": {
        "kind": "image",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "vlcs": {
        "kind": "image",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "har": {
        "kind": "signal",
    },
    "har_feat": {
        "kind": "feature",
    },
}


def normalize_image_tensor(x_float: torch.Tensor, dataset_name: str) -> torch.Tensor:
    """针对已经是 [0, 1] 浮点区间的图像 Tensor 执行标准化：(x - mean) / std。"""
    try:
        spec = DATASET_SPECS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"未知数据集: {dataset_name}") from exc

    if spec["kind"] != "image":
        return x_float

    mean = spec["mean"]
    std = spec["std"]
    channels = x_float.shape[-3]
    if len(mean) != channels or len(std) != channels:
        raise ValueError(
            f"数据集 {dataset_name} 的归一化通道数与输入不一致: "
            f"配置={len(mean)}, 输入={channels}"
        )
    mean_tensor = x_float.new_tensor(mean).view(1, channels, 1, 1)
    std_tensor = x_float.new_tensor(std).view(1, channels, 1, 1)
    return (x_float - mean_tensor) / std_tensor


def prepare_input_batch(x: torch.Tensor, dataset_name: str) -> torch.Tensor:
    """统一批量输入转换。

    - 图像数据：
      - 若 x 为 uint8，将其转为 [0, 1] 的 float32 并完成 (x - mean) / std 归一化。
      - 若 x 已经为 float32（如服务端已预归一化），直接返回，实现幂等保护。
    - 非图像数据（如 har、har_feat）：原样返回，不改变类型与数值。
    """
    try:
        spec = DATASET_SPECS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"未知数据集: {dataset_name}") from exc

    if spec["kind"] != "image":
        return x

    if x.dtype != torch.uint8:
        # 已完成预归一化（float32）直接返回，幂等保护
        return x

    if x.ndim != 4:
        raise ValueError(
            f"{dataset_name} 图像批量输入必须为 4 维 [B, C, H, W]，实际为 {x.shape}"
        )

    x_float = v2.functional.to_dtype(x, torch.float32, scale=True)
    return normalize_image_tensor(x_float, dataset_name)


def normalize_dataset(meta_dataset, dataset_name: str):
    """就地对 MetaDataset 的样本数据 x 进行一次性归一化。

    - 针对图像数据集：uint8 -> float32 -> (x - mean) / std。
    - 针对非图像数据集或已归一化数据：原样保留，不重复处理。
    """
    spec = DATASET_SPECS.get(dataset_name)
    if spec is None or spec.get("kind") != "image":
        return meta_dataset

    if meta_dataset.x.dtype != torch.uint8:
        return meta_dataset

    x_float = v2.functional.to_dtype(meta_dataset.x, torch.float32, scale=True)
    meta_dataset.x = normalize_image_tensor(x_float, dataset_name)
    return meta_dataset
