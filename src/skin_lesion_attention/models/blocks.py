from __future__ import annotations

import torch
from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
    ) -> None:
        if padding is None:
            padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class MultiScaleFeatureBlock(nn.Module):
    def __init__(self, in_channels: int, branch_channels: int) -> None:
        super().__init__()
        self.local_path = ConvBNReLU(in_channels, branch_channels, kernel_size=3)
        self.border_path = ConvBNReLU(in_channels, branch_channels, kernel_size=5)
        self.context_path = ConvBNReLU(
            in_channels,
            branch_channels,
            kernel_size=3,
            dilation=2,
        )
        self.out_channels = branch_channels * 3

    def forward(self, x):
        return torch.cat(
            [self.local_path(x), self.border_path(x), self.context_path(x)],
            dim=1,
        )
