import torch.nn as nn


class HARMLP(nn.Module):
    def __init__(self, input_dim=561, num_classes=6, feature_dim=512):
        super(HARMLP, self).__init__()

        # 特征提取器 (主体)
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
        )

        # 分类头
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # 输入形状: (Batch, 561)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        feature = self.extractor(x)
        logits = self.classifier(feature)
        return logits
