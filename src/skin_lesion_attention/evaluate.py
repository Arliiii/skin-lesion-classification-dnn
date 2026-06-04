from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import load_config
from .data import create_dataloaders
from .metrics import classification_metrics
from .models import build_model
from .utils import ensure_dir, get_device, load_checkpoint, save_json


def save_confusion_matrix(matrix: list[list[int]], classes: list[str], path: Path) -> None:
    ensure_dir(path.parent)
    values = np.asarray(matrix)
    plt.figure(figsize=(8, 7))
    plt.imshow(values, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    plt.colorbar()
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            plt.text(col, row, str(values[row, col]), ha="center", va="center", color="#111827")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def evaluate(config_path: str, checkpoint_path: str) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint.get("config") or load_config(config_path)
    device = get_device(str(config["project"].get("device", "auto")))
    dataloaders, _, classes = create_dataloaders(config)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    targets: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for batch in dataloaders["test"]:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            targets.extend(labels.cpu().tolist())
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())

    metrics = classification_metrics(targets, predictions, int(config["model"]["num_classes"]), classes)
    log_dir = ensure_dir(config["outputs"]["log_dir"])
    figure_dir = ensure_dir(config["outputs"]["figure_dir"])
    save_json(metrics, log_dir / "test_metrics.json")
    save_confusion_matrix(metrics["confusion_matrix"], classes, figure_dir / "confusion_matrix.png")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    metrics = evaluate(args.config, args.checkpoint)
    print(metrics)


if __name__ == "__main__":
    main()

