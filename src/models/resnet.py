import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, feature_dim=64):
        super(ResNet18, self).__init__()
        # 使用预训练的ResNet18作为基础
        self.resnet = models.resnet18(weights=None)

        # 修改第一层卷积以适应CIFAR10的3通道输入
        # CIFAR10图片是32x32，比ImageNet的224x224小很多
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # 移除自适应平均池化和全连接层，替换为适合CIFAR10的版本
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 特征提取器
        self.features = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            self.resnet.avgpool,
            nn.Flatten(),
        )

        # 获取特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            feature_size = self.features(dummy_input).shape[1]

        # 投影头（用于MOON等算法）
        self.proj = nn.Sequential(
            nn.Linear(feature_size, 128), nn.ReLU(), nn.Linear(128, feature_dim)
        )

        # 分类头
        self.fc = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        h = self.features(x)
        # MOON使用投影头，其他算法直接使用特征
        z = self.proj(h)
        y = self.fc(h)  # 注意：这里直接使用特征h而不是z进行分类
        return y, z
