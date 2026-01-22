import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=10, feature_dim=64, input_channels=1):
        super(CNN, self).__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        # 计算展平后的特征维度
        if input_channels == 1:
            # MNIST (28x28)
            flattened_dim = 64 * 7 * 7
        else:
            # CIFAR10 (32x32)
            flattened_dim = 64 * 8 * 8

        # Projection head (for MOON)
        self.proj = nn.Sequential(
            nn.Linear(flattened_dim, 128), nn.ReLU(), nn.Linear(128, feature_dim)
        )
        # Classification head
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        h = self.features(x)
        # MOON uses projection head, others use features directly
        z = self.proj(h)
        y = self.fc(z)
        return y, z
