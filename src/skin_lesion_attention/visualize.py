from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import build_transforms
from .models import build_model
from .utils import ensure_dir, get_device, load_checkpoint


def visualize_attention(
    checkpoint_path: str,
    image_path: str,
    output_path: str | None = None,
    attention_cmap: str = "jet_r",
) -> str:
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    classes = checkpoint.get("classes", config["data"]["classes"])
    device = get_device(str(config["project"].get("device", "auto")))

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image = Image.open(Path(image_path)).convert("RGB")
    transform = build_transforms(int(config["data"]["image_size"]), train=False)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, attention = model(tensor, return_attention=True)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        predicted_index = int(torch.argmax(probabilities).item())

    if attention is None:
        raise RuntimeError("This checkpoint was trained with attention disabled.")

    attention = F.interpolate(
        attention,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    ).squeeze().detach().cpu().numpy()
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

    if output_path is None:
        figure_dir = ensure_dir(config["outputs"]["figure_dir"])
        output_path = str(figure_dir / f"{Path(image_path).stem}_attention.png")

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(attention, cmap=attention_cmap, alpha=0.45)
    plt.title(f"{classes[predicted_index]} ({float(probabilities[predicted_index]):.2f})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an attention visualization")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--attention-cmap",
        default="jet_r",
        help="Matplotlib colormap for the attention overlay. Default reverses jet colors.",
    )
    args = parser.parse_args()
    print(visualize_attention(args.checkpoint, args.image, args.output, args.attention_cmap))


if __name__ == "__main__":
    main()
