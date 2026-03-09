import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, feature_dim=512):
        super(CNN, self).__init__()
        # 计算展平后的特征维度
        if input_channels == 1:
            # MNIST (28x28)
            dim = 64 * 7 * 7
        else:
            # CIFAR10 (32x32)
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
