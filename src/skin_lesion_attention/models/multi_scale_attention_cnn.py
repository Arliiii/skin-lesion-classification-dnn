from __future__ import annotations

from .pretrained import ResNet50MultiScaleAttentionClassifier


def build_model(config: dict) -> ResNet50MultiScaleAttentionClassifier:
    model_cfg = config["model"]
    model_name = str(model_cfg.get("name", "resnet50_multiscale_attention")).lower()
    if model_name not in {"resnet50_multiscale_attention", "resnet50_msa"}:
        raise ValueError(
            "Only the current ResNet-50 multi-scale attention architecture is supported. "
            f"Received model.name={model_name!r}."
        )

    return ResNet50MultiScaleAttentionClassifier(
        num_classes=int(model_cfg["num_classes"]),
        head_channels=int(model_cfg.get("head_channels", 256)),
        fusion_channels=int(model_cfg.get("fusion_channels", 512)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        pretrained=bool(model_cfg.get("pretrained", True)),
    )
