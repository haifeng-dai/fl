import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Feature extractor
        self.features = nn.Sequential(
            self.conv1, nn.ReLU(), self.pool,
            self.conv2, nn.ReLU(), self.pool,
            nn.Flatten()
        )
        # Projection head (for MOON)
        self.proj = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        # Classification head
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        h = self.features(x)
        z = self.proj(h)
        y = self.fc(z)
        return y, z
