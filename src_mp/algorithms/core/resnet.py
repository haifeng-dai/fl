from __future__ import annotations

import math
import torch
from torch import nn


def _norm2d(group_norm_num_groups: int | None, planes: int) -> nn.Module:
    if group_norm_num_groups is not None and group_norm_num_groups > 0:
        return nn.GroupNorm(group_norm_num_groups, planes)
    return nn.BatchNorm2d(planes)


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        group_norm_num_groups: int | None = None,
    ):
        super().__init__()
        self.conv1 = _conv3x3(in_planes, out_planes, stride)
        self.bn1 = _norm2d(group_norm_num_groups, out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_planes, out_planes)
        self.bn2 = _norm2d(group_norm_num_groups, out_planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNet_PC(nn.Module):
    """SAGE 与 ProxyFL 官方代码所用的轻量 ResNet (ResNet-PC / ResNet-8)。

    结构：3 个 Stage BasicBlock，通道数为 64, 128, 256（scaling=4）。
    暴露 self.extractor 与 self.classifier，标准 forward(x) 返回 Logits。
    """

    def __init__(
        self,
        num_classes: int = 10,
        feature_dim: int = 256,
        dataset_name: str = "cifar10",
        resnet_size: int = 8,
        scaling: int = 4,
        group_norm_num_groups: int | None = None,
    ):
        super().__init__()
        rgb = {
            "tiny_imagenet",
            "flowers102",
            "cars",
            "gtsrb",
            "cinic10",
            "svhn",
            "pacs",
            "officehome",
            "vlcs",
            "domainnet",
        }
        in_channels = 3 if "cifar" in dataset_name or dataset_name in rgb else 1

        if resnet_size % 6 != 2:
            raise ValueError("resnet_size must be 6n + 2:", resnet_size)
        block_nums = (resnet_size - 2) // 6

        planes1 = int(16 * scaling)  # 64
        planes2 = int(32 * scaling)  # 128
        planes3 = int(64 * scaling)  # 256

        self.inplanes = planes1
        self.conv1 = nn.Conv2d(
            in_channels, planes1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = _norm2d(group_norm_num_groups, planes1)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_block(
            planes1, block_nums, group_norm_num_groups=group_norm_num_groups
        )
        self.layer2 = self._make_block(
            planes2, block_nums, stride=2, group_norm_num_groups=group_norm_num_groups
        )
        self.layer3 = self._make_block(
            planes3, block_nums, stride=2, group_norm_num_groups=group_norm_num_groups
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dim = planes3
        self.classifier = nn.Linear(planes3, num_classes)

        self._init_weights()

    def _make_block(
        self,
        planes: int,
        block_num: int,
        stride: int = 1,
        group_norm_num_groups: int | None = None,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes, kernel_size=1, stride=stride, bias=False
                ),
                _norm2d(group_norm_num_groups, planes),
            )
        layers = [
            BasicBlock(
                self.inplanes, planes, stride, downsample, group_norm_num_groups
            )
        ]
        self.inplanes = planes
        for _ in range(1, block_num):
            layers.append(
                BasicBlock(
                    self.inplanes, planes, group_norm_num_groups=group_norm_num_groups
                )
            )
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def extractor(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.avgpool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.extractor(x)
        return self.classifier(feature)
