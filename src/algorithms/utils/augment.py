import numpy as np
import torch
import torch.nn.functional as F


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
    if x.dim() == 3:
        x = x * scale
        mask = torch.rand(x.shape[1], device=x.device) > 0.2
        x = x * mask.view(1, -1, 1)
    else:
        x = x * scale
        mask = torch.rand(x.shape[1], device=x.device) > 0.2
        x = x * mask
    return x + torch.randn_like(x) * 0.05
