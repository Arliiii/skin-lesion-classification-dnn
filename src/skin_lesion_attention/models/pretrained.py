from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from .attention import CBAMBlock
from .blocks import ConvBNReLU, MultiScaleFeatureBlock


class ResNet50MultiScaleAttentionClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        head_channels: int = 256,
        fusion_channels: int = 512,
        dropout: float = 0.2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.multi_scale = MultiScaleFeatureBlock(
            in_channels=2048,
            branch_channels=head_channels,
        )
        fused_channels = self.multi_scale.out_channels
        self.attention = CBAMBlock(fused_channels)
        self.fusion = ConvBNReLU(fused_channels, fusion_channels, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(fusion_channels, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        x = self.backbone(x)
        x = self.multi_scale(x)
        x, attention = self.attention(x)
        x = self.fusion(x)
        x = self.pool(x)
        logits = self.classifier(x)
        if return_attention:
            return logits, attention
        return logits
