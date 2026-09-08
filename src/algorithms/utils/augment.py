from functools import cache

import torch
from torchvision.transforms import InterpolationMode, v2

from .input import DATASET_SPECS, normalize_image_tensor


def _validate_image_batch(x, dataset_name):
    if x.ndim != 4:
        raise ValueError("图像增强输入必须是 [B, C, H, W] Tensor")
    if x.dtype != torch.uint8:
        raise TypeError(
            f"数据集 {dataset_name} 的图像增强要求输入为 uint8 Tensor，实际为 {x.dtype}"
        )


@cache
def _weak_transform(height, width):
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomCrop(
                size=(height, width),
                padding=4,
                padding_mode="reflect",
            ),
        ]
    )


@cache
def _strong_transform(height, width):
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomCrop(
                size=(height, width),
                padding=4,
                padding_mode="reflect",
            ),
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


def weak_augment(x, dataset_name):
    """使用 Torchvision Tensor 后端在 GPU 上执行批量弱增强，输出归一化 float32。"""
    spec = DATASET_SPECS[dataset_name]
    if spec["kind"] != "image":
        return x

    _validate_image_batch(x, dataset_name)
    height, width = x.shape[-2:]
    transformed = _weak_transform(height, width)(x)
    x_float = v2.functional.to_dtype(transformed, torch.float32, scale=True)
    return normalize_image_tensor(x_float, dataset_name)


def strong_augment(x, dataset_name):
    """使用 Torchvision Tensor 后端在 GPU 上执行批量强增强，输出归一化 float32。"""
    spec = DATASET_SPECS[dataset_name]
    if spec["kind"] != "image":
        return x

    _validate_image_batch(x, dataset_name)
    height, width = x.shape[-2:]
    transformed = _strong_transform(height, width)(x)
    x_float = v2.functional.to_dtype(transformed, torch.float32, scale=True)
    return normalize_image_tensor(x_float, dataset_name)
