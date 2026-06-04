from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        counts: torch.Tensor,
        beta: float = 0.999,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= beta < 1.0:
            raise ValueError("class-balanced focal beta must be in [0, 1).")
        if gamma < 0.0:
            raise ValueError("focal gamma must be non-negative.")

        counts = torch.clamp(counts.float(), min=1.0)
        beta_tensor = torch.tensor(beta, dtype=torch.float32, device=counts.device)
        effective_number = 1.0 - torch.pow(beta_tensor, counts)
        weights = (1.0 - beta) / effective_number
        weights = weights / weights.sum() * counts.numel()
        self.register_buffer("weights", weights)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)
        true_class_probability = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = torch.pow(1.0 - true_class_probability.clamp(min=1e-6, max=1.0), self.gamma)
        cross_entropy = F.cross_entropy(
            logits,
            targets,
            weight=self.weights.to(logits.device),
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        return (focal_factor * cross_entropy).mean()


def build_loss(config: dict, labels: list[int] | None, device: torch.device) -> nn.Module:
    loss_name = str(config["training"].get("loss", "cross_entropy")).lower()
    label_smoothing = float(config["training"].get("label_smoothing", 0.0))
    if loss_name in {"cross_entropy", "ce"}:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if loss_name in {"class_balanced_focal", "cb_focal", "class_balanced_focal_loss"}:
        num_classes = int(config["model"]["num_classes"])
        if labels:
            counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes)
        else:
            counts = torch.ones(num_classes, dtype=torch.long)
        return ClassBalancedFocalLoss(
            counts=counts.to(device),
            beta=float(config["training"].get("class_balanced_beta", 0.999)),
            gamma=float(config["training"].get("focal_gamma", 2.0)),
            label_smoothing=label_smoothing,
        )

    raise ValueError(f"Unsupported loss: {loss_name}")
