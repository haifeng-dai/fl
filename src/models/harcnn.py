import torch
import torch.nn as nn


class HARCNN(nn.Module):
    def __init__(self, in_channels=9, num_classes=6, feature_dim=512):
        super(HARCNN, self).__init__()

        # Feature extractor
        self.extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, feature_dim),
            nn.ReLU(inplace=True),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        # Input shape: (Batch, 9, 128)
        feature = self.extractor(x)
        logits = self.classifier(feature)
        return logits, feature
