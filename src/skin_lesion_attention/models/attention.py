from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.activation(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.activation(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attention, attention


class CBAMBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.channel_attention(x)
        return self.spatial_attention(x)

