import torch
import torch.nn as nn


class HARMLP(nn.Module):
    def __init__(self, input_dim=561, num_classes=6, hidden_dim=64):
        super(HARMLP, self).__init__()

        self.feature_dim = hidden_dim

        # Feature extractor (Body)
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, hidden_dim), nn.ReLU()
        )

        # Classifier head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Input shape: (Batch, 561)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        feature = self.extractor(x)
        logits = self.classifier(feature)
        return logits, feature
