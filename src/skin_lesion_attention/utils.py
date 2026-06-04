from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_history_csv(history: list[dict[str, float]], path: str | Path) -> None:
    if not history:
        return
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
    classes: list[str],
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
            "classes": classes,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)

