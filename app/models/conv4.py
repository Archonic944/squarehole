import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class Conv4Backbone(nn.Module):
    """4-block convolutional backbone for few-shot learning.

    Input: 84x84x3
    Output: flattened feature vector.
      - hidden=64  → 1600-dim  (~113K params)
      - hidden=128 → 6400-dim  (~430K params)
      - hidden=256 → 25600-dim (~1.7M params)
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.block1 = conv_block(3, hidden)
        self.block2 = conv_block(hidden, hidden)
        self.block3 = conv_block(hidden, hidden)
        self.block4 = conv_block(hidden, hidden)
        # 84 -> 42 -> 21 -> 10 -> 5, so feature_dim = hidden * 5 * 5
        self.feature_dim = hidden * 5 * 5

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


class Conv4WithHead(nn.Module):
    """Conv4 backbone with a detachable linear classification head."""

    def __init__(self, num_classes, hidden=64):
        super().__init__()
        self.backbone = Conv4Backbone(hidden=hidden)
        self.head = nn.Linear(self.backbone.feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def rebuild_head(self, num_classes):
        """Replace the linear head for a different number of classes."""
        self.head = nn.Linear(self.backbone.feature_dim, num_classes)
