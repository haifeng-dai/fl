import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from torchvision import transforms
from torchvision.transforms import functional as TF


def weak_augment(x):
    """弱增强：随机水平翻转 + 随机裁剪（Reflect Padding）"""
    if torch.rand(1) < 0.5:
        x = x.flip(-1)
    pad = 4
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    _, _, H, W = x.shape
    h_start = torch.randint(0, 2 * pad + 1, (1,)).item()
    w_start = torch.randint(0, 2 * pad + 1, (1,)).item()
    x = x[:, :, h_start : h_start + H - 2 * pad, w_start : w_start + W - 2 * pad]
    return x


def strong_augment(x):
    """强增强：弱增强 + 随机仿射 + 高斯模糊 + 随机噪声"""
    x = weak_augment(x)
    if torch.rand(1) < 0.8:
        theta = (torch.rand(1).item() - 0.5) * 30
        tx = (torch.rand(1).item() - 0.5) * 0.2
        ty = (torch.rand(1).item() - 0.5) * 0.2
        cos, sin = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        affine = torch.tensor(
            [[[cos, -sin, tx], [sin, cos, ty]]], dtype=x.dtype, device=x.device
        ).repeat(x.shape[0], 1, 1)
        grid = F.affine_grid(affine, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
    if torch.rand(1) < 0.5:
        k = 3
        kernel = torch.randn(1, 1, k, k, device=x.device) * 0.1
        kernel[0, 0, k // 2, k // 2] += 1.0
        kernel = (kernel / kernel.sum()).expand(x.shape[1], 1, k, k)
        x = F.conv2d(x, kernel, padding=k // 2, groups=x.shape[1])
    if torch.rand(1) < 0.5:
        x = x + torch.randn_like(x) * 0.05
    return x


def weak_augment_1d(x):
    """1D 弱增强：小幅高斯噪声"""
    return x + torch.randn_like(x) * 0.01


def strong_augment_1d(x):
    """1D 强增强：随机幅度缩放 + 通道掩码 + 噪声"""
    scale = 0.8 + torch.rand(1, device=x.device).item() * 0.4
    x = x * scale
    mask = torch.rand(x.shape[1], device=x.device) > 0.2
    if x.dim() == 3:
        x = x * mask.view(1, -1, 1)
    else:
        x = x * mask
    return x + torch.randn_like(x) * 0.05


_SAGE_NORMALIZATION = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar10_dg": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "cinic10": (
        (0.47889522, 0.47227842, 0.43047404),
        (0.24205776, 0.23828046, 0.25874835),
    ),
    "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    "mnist": ((0.1307,), (0.3081,)),
    "fashionmnist": ((0.2860,), (0.3530,)),
    "femnist": ((0.1307,), (0.3081,)),
    "emnist": ((0.1307,), (0.3081,)),
}


PARAMETER_MAX = 10


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def AutoContrast(img, **kwargs):
    return ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    return ImageEnhance.Brightness(img).enhance(_float_parameter(v, max_v) + bias)


def Color(img, v, max_v, bias=0):
    return ImageEnhance.Color(img).enhance(_float_parameter(v, max_v) + bias)


def Contrast(img, v, max_v, bias=0):
    return ImageEnhance.Contrast(img).enhance(_float_parameter(v, max_v) + bias)


def Equalize(img, **kwargs):
    return ImageOps.equalize(img)


def Identity(img, **kwargs):
    return img


def Posterize(img, v, max_v, bias=0):
    return ImageOps.posterize(img, _int_parameter(v, max_v) + bias)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return img.rotate(-v if random.random() < 0.5 else v)


def Sharpness(img, v, max_v, bias=0):
    return ImageEnhance.Sharpness(img).enhance(_float_parameter(v, max_v) + bias)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    v = -v if random.random() < 0.5 else v
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    v = -v if random.random() < 0.5 else v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return ImageOps.solarize(img, 256 - v)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    v = -v if random.random() < 0.5 else v
    return img.transform(img.size, Image.AFFINE, (1, 0, int(v * img.size[0]), 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    v = -v if random.random() < 0.5 else v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, int(v * img.size[1])))


def fixmatch_augment_pool():
    return [
        (AutoContrast, None, None),
        (Brightness, 0.9, 0.05),
        (Color, 0.9, 0.05),
        (Contrast, 0.9, 0.05),
        (Equalize, None, None),
        (Identity, None, None),
        (Posterize, 4, 4),
        (Rotate, 30, 0),
        (Sharpness, 0.9, 0.05),
        (ShearX, 0.3, 0),
        (ShearY, 0.3, 0),
        (Solarize, 256, 0),
        (TranslateX, 0.3, 0),
        (TranslateY, 0.3, 0),
    ]


def CutoutAbs(img, v, **kwargs):
    w, h = img.size
    x0 = int(max(0, np.random.uniform(0, w) - v / 2.0))
    y0 = int(max(0, np.random.uniform(0, h) - v / 2.0))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    img = img.copy()
    ImageDraw.Draw(img).rectangle((x0, y0, x1, y1), (127, 127, 127))
    return img


class RandAugmentMC:
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        for op, max_v, bias in random.choices(self.augment_pool, k=self.n):
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        return CutoutAbs(img, int(32 * 0.5))


def _sage_transform_sample(sample, dataset_name, strong):
    if sample.ndim != 3:
        raise ValueError("SAGE 图像增强输入必须是 [C, H, W] Tensor")
    try:
        mean, std = _SAGE_NORMALIZATION[dataset_name]
    except KeyError as exc:
        raise ValueError(f"SAGE 增强暂不支持数据集: {dataset_name}") from exc

    channels = sample.shape[0]
    mean_tensor = sample.new_tensor(mean).view(channels, 1, 1)
    std_tensor = sample.new_tensor(std).view(channels, 1, 1)
    image_tensor = (sample * std_tensor + mean_tensor).clamp(0, 1)
    image = TF.to_pil_image(image_tensor.cpu())
    height, width = sample.shape[-2:]
    crop_size = (height, width)
    transform_ops = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=crop_size, padding=4, padding_mode="reflect"),
    ]
    if strong:
        transform_ops.append(RandAugmentMC(n=2, m=10))
    image = transforms.Compose(transform_ops)(image)
    image_tensor = TF.to_tensor(image)
    normalized = (image_tensor - mean_tensor.cpu()) / std_tensor.cpu()
    return normalized.to(device=sample.device, dtype=sample.dtype)


def sage_weak_augment(x, dataset_name):
    """SAGE 风格弱增强，输入为已归一化的图像 Tensor 批次。"""
    return torch.stack(
        [_sage_transform_sample(sample, dataset_name, False) for sample in x]
    )


def sage_strong_augment(x, dataset_name):
    """SAGE 风格强增强，包含 RandAugmentMC 操作池和 Cutout。"""
    return torch.stack(
        [_sage_transform_sample(sample, dataset_name, True) for sample in x]
    )
