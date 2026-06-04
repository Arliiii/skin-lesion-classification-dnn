from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"seed": 42, "device": "auto"},
    "data": {
        "synthetic": False,
        "raw_dir": "data/raw/archive_2",
        "processed_dir": "data/processed_lesion",
        "metadata_file": "HAM10000_metadata.csv",
        "image_dirs": ["HAM10000_images_part_1", "HAM10000_images_part_2"],
        "image_size": 299,
        "batch_size": 16,
        "num_workers": 0,
        "val_size": 0.15,
        "test_size": 0.15,
        "regenerate_splits": True,
        "group_by": "lesion_id",
        "balanced_sampler": True,
        "sampler_weight_power": 0.5,
        "classes": ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"],
    },
    "model": {
        "name": "resnet50_multiscale_attention",
        "num_classes": 7,
        "pretrained": True,
        "head_channels": 128,
        "fusion_channels": 256,
        "dropout": 0.3,
    },
    "training": {
        "epochs": 40,
        "learning_rate": 1e-4,
        "weight_decay": 2e-4,
        "loss": "class_balanced_focal",
        "class_balanced_beta": 0.999,
        "focal_gamma": 2.0,
        "label_smoothing": 0.05,
        "backbone_learning_rate": 1e-5,
        "head_learning_rate": 1e-4,
        "freeze_backbone_epochs": 2,
        "scheduler": "reduce_on_plateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 2,
        "early_stopping": True,
        "early_stopping_patience": 7,
        "early_stopping_min_delta": 5e-4,
    },
    "outputs": {
        "checkpoint_dir": "outputs/checkpoints",
        "figure_dir": "outputs/figures",
        "log_dir": "outputs/logs",
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    config = deep_update(deepcopy(DEFAULT_CONFIG), user_config)
    config["_config_path"] = str(path)
    return config
