import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, feature_dim=None, dataset_name="cifar10"):
        super(ResNet18, self).__init__()
        # 使用预训练的ResNet18作为基础
        self.resnet = models.resnet18(weights=None)

        if dataset_name in ["cifar10", "cifar100", "svhn"]:
            # 小图模式 (32x32): 3x3卷积, stride=1, 无maxpool
            self.resnet.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            # 移除maxpool (替换为Identity)以保留特征图尺寸
            self.resnet.maxpool = nn.Identity()  # type: ignore

        elif dataset_name in ["mnist", "fashionmnist", "femnist"]:
            # 单通道小图模式 (28x28): 1通道输入, 其余同上
            self.resnet.conv1 = nn.Conv2d(
                1, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.resnet.maxpool = nn.Identity()  # type: ignore

        else:
            # 默认 ImageNet 模式 (224x224):
            # 保持原始的 7x7 conv, stride=2 和 3x3 maxpool, stride=2
            # 这里的 self.resnet.maxpool 已经是 MaxPool2d(kernel_size=3, stride=2, padding=1)
            pass

        # 移除自适应平均池化和全连接层
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 特征提取器
        # 我们按照 ResNet 的标准顺序重新组织 layer
        self.extractor = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
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
                # 根据数据集创建对应的 dummy input
                if dataset_name in ["mnist", "fashionmnist", "femnist"]:
                    dummy_input = torch.randn(1, 1, 28, 28)
                elif dataset_name in ["cifar10", "cifar100", "svhn"]:
                    dummy_input = torch.randn(1, 3, 32, 32)
                else:
                    # Default large image
                    dummy_input = torch.randn(1, 3, 224, 224)

                self.feature_dim = self.extractor(dummy_input).shape[1]

        # 分类头
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        feature = self.extractor(x)
        logits = self.classifier(feature)
        return logits, feature


class ResNet50(nn.Module):
    def __init__(self, num_classes=10, feature_dim=None, dataset_name="cifar10"):
        super(ResNet50, self).__init__()
        # 使用预训练的ResNet50作为基础
        self.resnet = models.resnet50(weights=None)

        if dataset_name in ["cifar10", "cifar100", "svhn"]:
            # 小图模式 (32x32): 3x3卷积, stride=1, 无maxpool
            self.resnet.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            # 移除maxpool (替换为Identity)以保留特征图尺寸
            self.resnet.maxpool = nn.Identity()  # type: ignore

        elif dataset_name in ["mnist", "fashionmnist", "femnist"]:
            # 单通道小图模式 (28x28): 1通道输入, 其余同上
            self.resnet.conv1 = nn.Conv2d(
                1, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.resnet.maxpool = nn.Identity()  # type: ignore

        else:
            # 默认 ImageNet 模式 (224x224):
            # 保持原始的 7x7 conv, stride=2 和 3x3 maxpool, stride=2
            pass

        # 移除自适应平均池化和全连接层
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 特征提取器
        # ResNet50 的层命名与 ResNet18 相同 (layer1...layer4)
        self.extractor = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
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
                # 根据数据集创建对应的 dummy input
                if dataset_name in ["mnist", "fashionmnist", "femnist"]:
                    dummy_input = torch.randn(1, 1, 28, 28)
                elif dataset_name in ["cifar10", "cifar100", "svhn"]:
                    dummy_input = torch.randn(1, 3, 32, 32)
                else:
                    # Default large image
                    dummy_input = torch.randn(1, 3, 224, 224)

                self.feature_dim = self.extractor(dummy_input).shape[1]

        # 分类头
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        feature = self.extractor(x)
        logits = self.classifier(feature)
        return logits, feature
