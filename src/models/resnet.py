import torch
import torch.nn as nn
import torchvision.models as models


def _get_flatten_dim(extractor, dataset_name):
    if dataset_name in ["mnist", "fashionmnist", "femnist"]:
        dummy_input = torch.randn(1, 1, 28, 28)
    elif dataset_name in ["cifar10", "cifar100", "svhn", "gtsrb"]:
        dummy_input = torch.randn(1, 3, 32, 32)
    elif dataset_name == "tiny_imagenet":
        dummy_input = torch.randn(1, 3, 64, 64)
    else:
        # 默认大图模式
        dummy_input = torch.randn(1, 3, 224, 224)

    return extractor(dummy_input).shape[1]


def _adapt_resnet_input_layer(resnet, dataset_name):
    """
    修改 ResNet 最前端的输入层，以适配小分辨率数据集 (如 CIFAR/MNIST)
    避免图层下采样过快导致后续层获取不到有效内容
    """
    if dataset_name in ["cifar10", "cifar100", "svhn", "gtsrb"]:
        # 小图模式 (32x32): 3x3卷积, stride=1, 无maxpool
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 移除maxpool (替换为Identity)以保留特征图尺寸
        resnet.maxpool = nn.Identity()

    elif dataset_name in ["mnist", "fashionmnist", "femnist"]:
        # 单通道小图模式 (28x28): 1通道输入, 其余同上
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()

    elif dataset_name == "tiny_imagenet":
        # Tiny-ImageNet (64x64): 使用小尺寸卷积核并移除池化层以保持分辨率
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()

    else:
        # 默认 ImageNet 模式 (224x224):
        # 保持原始的 7x7 conv, stride=2 和 3x3 maxpool, stride=2
        pass

    # 统一移除自适应平均池化，因为后续用 Flatten
    resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    return resnet


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512, dataset_name="cifar10"):
        super(ResNet18, self).__init__()
        # 使用预训练的ResNet18作为基础
        base_resnet = models.resnet18(weights=None)
        base_resnet = _adapt_resnet_input_layer(base_resnet, dataset_name)

        # 获取特征维度
        # 根据数据集创建对应的 dummy input
        extractor = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool,
            base_resnet.layer1,
            base_resnet.layer2,
            base_resnet.layer3,
            base_resnet.layer4,
            base_resnet.avgpool,
            nn.Flatten(),
        )
        dim = _get_flatten_dim(extractor, dataset_name)

        # 投影层
        self.extractor = nn.Sequential(
            extractor,
            nn.Linear(dim, feature_dim),
            nn.ReLU(inplace=True),
        )

        # 分类头
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feature = self.extractor(x)
        logits = self.classifier(feature)
        return logits


class ResNet50(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512, dataset_name="cifar10"):
        super(ResNet50, self).__init__()
        # 使用预训练的ResNet50作为基础
        base_resnet = models.resnet50(weights=None)
        base_resnet = _adapt_resnet_input_layer(base_resnet, dataset_name)

        # 获取特征维度
        extractor = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool,
            base_resnet.layer1,
            base_resnet.layer2,
            base_resnet.layer3,
            base_resnet.layer4,
            base_resnet.avgpool,
            nn.Flatten(),
        )
        dim = _get_flatten_dim(extractor, dataset_name)

        # 特征提取器
        # ResNet50 的层命名与 ResNet18 相同 (layer1...layer4)
        self.extractor = nn.Sequential(
            extractor,
            nn.Linear(dim, feature_dim),
            nn.ReLU(inplace=True),
        )

        # 分类头
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feature = self.extractor(x)
        logits = self.classifier(feature)
        return logits
