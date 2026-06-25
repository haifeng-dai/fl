import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self, input_channels=1, num_classes=10, feature_dim=512, dataset_name="mnist"
    ):
        super(CNN, self).__init__()
        # 计算展平后的特征维度
        if dataset_name in ["mnist", "fashionmnist", "femnist", "emnist"]:
            # 28x28 -> MaxPool(2x2) -> 14x14 -> MaxPool(2x2) -> 7x7
            dim = 64 * 7 * 7
        elif dataset_name == "tiny_imagenet":
            # 64x64 -> MaxPool(2x2) -> 32x32 -> MaxPool(2x2) -> 16x16
            dim = 64 * 16 * 16
        elif dataset_name in ["cars", "flowers102"]:
            # 224x224 -> MaxPool(2x2) -> 112x112 -> MaxPool(2x2) -> 56x56
            dim = 64 * 56 * 56
        else:
            # CIFAR10/100, GTSRB (32x32) -> MaxPool(2x2) -> 16x16 -> MaxPool(2x2) -> 8x8
            dim = 64 * 8 * 8

        # 特征提取器 (Feature extractor)
        self.extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(dim, feature_dim),
            nn.ReLU(inplace=True),
        )

        # 分类头 (Classification head)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feature = self.extractor(x)
        logits = self.classifier(feature)
        return logits
