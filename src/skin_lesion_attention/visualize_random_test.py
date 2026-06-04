from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from .config import load_config
from .data import build_transforms, prepare_splits
from .models import build_model
from .utils import ensure_dir, get_device, load_checkpoint


def normalize_attention(attention: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    resized = F.interpolate(
        attention,
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    return (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)


def plot_probability_bars(
    axis: plt.Axes,
    probabilities: torch.Tensor,
    classes: list[str],
    true_index: int,
    predicted_index: int,
    top_k: int,
) -> None:
    k = min(top_k, len(classes))
    top_probabilities, top_indexes = torch.topk(probabilities, k=k)
    labels = [classes[int(index)] for index in top_indexes]
    values = [float(value) for value in top_probabilities]
    colors = [
        "#238636" if int(index) == true_index else "#d29922" if int(index) == predicted_index else "#8b949e"
        for index in top_indexes
    ]

    axis.barh(labels[::-1], values[::-1], color=colors[::-1])
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Probability")
    axis.grid(axis="x", alpha=0.25)
    for spine in axis.spines.values():
        spine.set_visible(False)


def visualize_random_test_predictions(
    checkpoint_path: str,
    config_path: str,
    output_path: str | None = None,
    count: int = 6,
    seed: int = 42,
    top_k: int = 3,
    attention_cmap: str = "jet_r",
) -> str:
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint.get("config") or load_config(config_path)
    classes = checkpoint.get("classes", config["data"]["classes"])
    class_to_index = {name: index for index, name in enumerate(classes)}
    device = get_device(str(config["project"].get("device", "auto")))

    test_rows = prepare_splits(config)["test"]
    if not test_rows:
        raise RuntimeError("The test split is empty.")

    rng = random.Random(seed)
    rows_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in test_rows:
        rows_by_class[row["dx"]].append(row)

    selected_rows: list[dict[str, str]] = []
    for class_name in classes:
        class_rows = rows_by_class.get(class_name, [])
        if class_rows and len(selected_rows) < count:
            selected_rows.append(rng.choice(class_rows))

    remaining_rows = [
        row
        for row in test_rows
        if row["image_id"] not in {selected["image_id"] for selected in selected_rows}
    ]
    rng.shuffle(remaining_rows)
    selected_rows.extend(remaining_rows[: max(0, count - len(selected_rows))])

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = build_transforms(int(config["data"]["image_size"]), train=False)
    figure, axes = plt.subplots(
        len(selected_rows),
        3,
        figsize=(13, 3.2 * len(selected_rows)),
        squeeze=False,
    )
    correct = 0

    with torch.no_grad():
        for row_index, row in enumerate(selected_rows):
            image = Image.open(Path(row["image_path"])).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            logits, attention = model(tensor, return_attention=True)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
            predicted_index = int(torch.argmax(probabilities).item())
            true_index = class_to_index[row["dx"]]
            is_correct = predicted_index == true_index
            correct += int(is_correct)

            axes[row_index, 0].imshow(image)
            axes[row_index, 0].set_title(f"{row['image_id']} | true: {row['dx']}")
            axes[row_index, 0].axis("off")

            axes[row_index, 1].imshow(image)
            if attention is not None:
                attention_map = normalize_attention(attention.detach().cpu(), image.size[::-1])
                axes[row_index, 1].imshow(attention_map, cmap=attention_cmap, alpha=0.45)
            predicted_class = classes[predicted_index]
            confidence = float(probabilities[predicted_index])
            status = "correct" if is_correct else "wrong"
            title_color = "#238636" if is_correct else "#cf222e"
            axes[row_index, 1].set_title(
                f"pred: {predicted_class} ({confidence:.2f}) | {status}",
                color=title_color,
            )
            axes[row_index, 1].axis("off")

            plot_probability_bars(
                axes[row_index, 2],
                probabilities,
                classes,
                true_index,
                predicted_index,
                top_k,
            )

    figure.suptitle(
        f"Random HAM10000 Test Predictions | {correct}/{len(selected_rows)} correct",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))

    if output_path is None:
        figure_dir = ensure_dir(config["outputs"]["figure_dir"])
        output_path = str(figure_dir / "random_test_predictions.png")
    else:
        ensure_dir(Path(output_path).parent)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a report-ready figure from random test-set predictions."
    )
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--attention-cmap",
        default="jet_r",
        help="Matplotlib colormap for the attention overlay. Default reverses jet colors.",
    )
    args = parser.parse_args()

    output_path = visualize_random_test_predictions(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_path=args.output,
        count=args.count,
        seed=args.seed,
        top_k=args.top_k,
        attention_cmap=args.attention_cmap,
    )
    print(output_path)


if __name__ == "__main__":
    main()
