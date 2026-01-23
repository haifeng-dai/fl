import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512):
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
        self.extractor = nn.Sequential(
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
            if feature_dim is not None:
                self.feature_dim = feature_dim
            else:
                dummy_input = torch.randn(1, 3, 32, 32)
                self.feature_dim = self.extractor(dummy_input).shape[1]

        # 分类头
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        feature = self.extractor(x)
        logits = self.classifier(feature)
        return logits, feature