from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .config import load_config
from .data import create_dataloaders
from .losses import build_loss
from .metrics import classification_metrics
from .models import build_model
from .utils import ensure_dir, get_device, save_checkpoint, save_history_csv, save_json, set_seed


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return
    for parameter in backbone.parameters():
        parameter.requires_grad = trainable


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    default_lr = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg["weight_decay"])
    backbone_lr = train_cfg.get("backbone_learning_rate")
    head_lr = train_cfg.get("head_learning_rate")
    backbone = getattr(model, "backbone", None)

    if backbone is None or backbone_lr is None or head_lr is None:
        return torch.optim.AdamW(model.parameters(), lr=default_lr, weight_decay=weight_decay)

    backbone_parameter_ids = {id(parameter) for parameter in backbone.parameters()}
    head_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in backbone_parameter_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": backbone.parameters(), "lr": float(backbone_lr), "name": "backbone"},
            {"params": head_parameters, "lr": float(head_lr), "name": "head"},
        ],
        weight_decay=weight_decay,
    )


def learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def format_learning_rates(optimizer: torch.optim.Optimizer) -> str:
    return " ".join(
        f"lr_{name}={value:.2e}" for name, value in learning_rates(optimizer).items()
    )


def run_epoch(model, dataloader, criterion, device, optimizer=None) -> tuple[float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())

    average_loss = total_loss / max(1, len(dataloader.dataset))
    return average_loss, targets, predictions


def plot_history(history: list[dict[str, float]], figure_dir: Path) -> None:
    if not history:
        return
    ensure_dir(figure_dir)
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_macro_f1"] for row in history], label="train macro F1")
    plt.plot(epochs, [row["val_macro_f1"] for row in history], label="val macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "macro_f1_curve.png", dpi=160)
    plt.close()


def train(config_path: str) -> dict:
    config = load_config(config_path)
    set_seed(int(config["project"]["seed"]))
    device = get_device(str(config["project"].get("device", "auto")))

    dataloaders, _, classes = create_dataloaders(config)
    model = build_model(config).to(device)
    freeze_backbone_epochs = int(config["training"].get("freeze_backbone_epochs", 0))
    if freeze_backbone_epochs > 0:
        set_backbone_trainable(model, False)
    train_labels = getattr(dataloaders["train"].dataset, "labels", None)
    criterion = build_loss(config, train_labels, device)
    optimizer = build_optimizer(model, config)
    scheduler = None
    scheduler_name = str(config["training"].get("scheduler", "none")).lower()
    if scheduler_name in {"reduce_on_plateau", "plateau"}:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(config["training"].get("scheduler_factor", 0.5)),
            patience=int(config["training"].get("scheduler_patience", 2)),
        )
    elif scheduler_name not in {"none", ""}:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    checkpoint_dir = ensure_dir(config["outputs"]["checkpoint_dir"])
    figure_dir = ensure_dir(config["outputs"]["figure_dir"])
    log_dir = ensure_dir(config["outputs"]["log_dir"])

    history: list[dict[str, float]] = []
    best_macro_f1 = -1.0
    best_metrics: dict = {}
    epochs = int(config["training"]["epochs"])
    early_stopping = bool(config["training"].get("early_stopping", False))
    patience = int(config["training"].get("early_stopping_patience", 5))
    min_delta = float(config["training"].get("early_stopping_min_delta", 0.0))
    epochs_without_improvement = 0
    early_stopped = False

    print(
        f"training_start model={config['model']['name']} device={device} "
        f"epochs={epochs} early_stopping={early_stopping} patience={patience} "
        f"freeze_backbone_epochs={freeze_backbone_epochs} {format_learning_rates(optimizer)}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
            set_backbone_trainable(model, True)
            print(f"backbone_unfrozen epoch={epoch}", flush=True)

        print(f"epoch={epoch}/{epochs} started {format_learning_rates(optimizer)}", flush=True)
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        train_loss, train_targets, train_preds = run_epoch(
            model, dataloaders["train"], criterion, device, optimizer
        )
        train_seconds = time.perf_counter() - train_start
        val_start = time.perf_counter()
        val_loss, val_targets, val_preds = run_epoch(model, dataloaders["val"], criterion, device)
        val_seconds = time.perf_counter() - val_start
        epoch_seconds = time.perf_counter() - epoch_start
        train_metrics = classification_metrics(
            train_targets, train_preds, int(config["model"]["num_classes"]), classes
        )
        val_metrics = classification_metrics(
            val_targets, val_preds, int(config["model"]["num_classes"]), classes
        )
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            "epoch_seconds": epoch_seconds,
        }
        row.update({f"lr_{name}": value for name, value in learning_rates(optimizer).items()})
        history.append(row)

        improved = False
        if val_metrics["macro_f1"] > best_macro_f1 + min_delta:
            best_macro_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics
            epochs_without_improvement = 0
            improved = True
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                config,
                epoch,
                val_metrics,
                classes,
            )
        else:
            epochs_without_improvement += 1

        save_history_csv(history, log_dir / "training_history.csv")
        print(
            f"epoch={epoch}/{epochs} done "
            f"train_loss={train_loss:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"best_val_macro_f1={best_macro_f1:.4f} improved={improved} "
            f"no_improve={epochs_without_improvement}/{patience} "
            f"train_seconds={train_seconds:.2f} val_seconds={val_seconds:.2f} "
            f"epoch_seconds={epoch_seconds:.2f} {format_learning_rates(optimizer)}",
            flush=True,
        )

        if early_stopping and epochs_without_improvement >= patience:
            early_stopped = True
            print(
                f"early_stopping_triggered epoch={epoch} "
                f"best_val_macro_f1={best_macro_f1:.4f}",
                flush=True,
            )
            break

        if scheduler is not None:
            scheduler.step(val_metrics["macro_f1"])

    save_history_csv(history, log_dir / "training_history.csv")
    save_json(
        {
            "best_val_metrics": best_metrics,
            "history": history,
            "early_stopped": early_stopped,
            "epochs_ran": len(history),
        },
        log_dir / "training_summary.json",
    )
    plot_history(history, figure_dir)
    return {
        "best_val_macro_f1": best_macro_f1,
        "checkpoint": str(checkpoint_dir / "best.pt"),
        "early_stopped": early_stopped,
        "epochs_ran": len(history),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Multi-Scale Attention CNN")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    result = train(args.config)
    print(result)


if __name__ == "__main__":
    main()
