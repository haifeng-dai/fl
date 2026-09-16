from functools import cache

import torch
from torchvision.transforms import InterpolationMode, v2

from src.algorithms.utils.input import DATASET_SPECS, normalize_image_tensor


def _validate_image_batch(x: torch.Tensor, dataset_name: str) -> None:
    if x.ndim != 4:
        raise ValueError(
            f"{dataset_name} 图像增强输入必须是 [B, C, H, W]，实际为 {tuple(x.shape)}"
        )
    if x.dtype != torch.uint8:
        raise TypeError(
            f"{dataset_name} 图像增强要求原始 uint8 输入，实际为 {x.dtype}"
        )


@cache
def _weak_transform(height: int, width: int):
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomCrop((height, width), padding=4, padding_mode="reflect"),
        ]
    )


@cache
def _strong_transform(height: int, width: int):
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomCrop((height, width), padding=4, padding_mode="reflect"),
            v2.RandAugment(
                num_ops=2,
                magnitude=9,
                num_magnitude_bins=31,
                interpolation=InterpolationMode.BILINEAR,
                fill=127,
            ),
            v2.RandomErasing(
                p=1.0,
                scale=(0.25, 0.25),
                ratio=(1.0, 1.0),
                value=127,
            ),
        ]
    )


def _normalize_after(transform, x: torch.Tensor, dataset_name: str) -> torch.Tensor:
    spec = DATASET_SPECS[dataset_name]
    if spec["kind"] != "image":
        return x
    _validate_image_batch(x, dataset_name)
    transformed = transform(x)
    transformed = v2.functional.to_dtype(transformed, torch.float32, scale=True)
    return normalize_image_tensor(transformed, dataset_name)


def weak_augment(x: torch.Tensor, dataset_name: str) -> torch.Tensor:
    if DATASET_SPECS[dataset_name]["kind"] != "image":
        return x
    height, width = x.shape[-2:]
    return _normalize_after(_weak_transform(height, width), x, dataset_name)


def strong_augment(x: torch.Tensor, dataset_name: str) -> torch.Tensor:
    if DATASET_SPECS[dataset_name]["kind"] != "image":
        return x
    height, width = x.shape[-2:]
    return _normalize_after(_strong_transform(height, width), x, dataset_name)
