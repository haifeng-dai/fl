import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=10, feature_dim=None, input_channels=1):
        super(CNN, self).__init__()
        # Feature extractor
        self.extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        # 计算展平后的特征维度
        if feature_dim is not None:
            self.feature_dim = feature_dim
        else:
            if input_channels == 1:
                # MNIST (28x28)
                self.feature_dim = 64 * 7 * 7
            else:
                # CIFAR10 (32x32)
                self.feature_dim = 64 * 8 * 8

        # Classification head
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        feature = self.extractor(x)
        logits = self.classifier(feature)
        return logits, feature