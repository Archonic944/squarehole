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
    Input: 84x84x3 -> output: 1600-dim feature vector.
    """

    def __init__(self):
        super().__init__()
        self.block1 = conv_block(3, 64)
        self.block2 = conv_block(64, 64)
        self.block3 = conv_block(64, 64)
        self.block4 = conv_block(64, 64)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


class Conv4WithHead(nn.Module):
    """Conv4 backbone with a detachable linear classification head."""

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Conv4Backbone()
        self.head = nn.Linear(1600, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def rebuild_head(self, num_classes):
        """Replace the linear head with a new one for a different number of classes."""
        self.head = nn.Linear(1600, num_classes)
