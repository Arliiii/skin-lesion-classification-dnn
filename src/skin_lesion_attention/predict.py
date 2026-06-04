from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .data import build_transforms
from .models import build_model
from .utils import get_device, load_checkpoint


def predict(checkpoint_path: str, image_path: str) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    classes = checkpoint.get("classes", config["data"]["classes"])
    device = get_device(str(config["project"].get("device", "auto")))
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = build_transforms(int(config["data"]["image_size"]), train=False)
    image = Image.open(Path(image_path)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()
    top_index = int(torch.argmax(probabilities).item())
    return {
        "image": str(image_path),
        "predicted_class": classes[top_index],
        "predicted_index": top_index,
        "probabilities": {classes[i]: float(probabilities[i]) for i in range(len(classes))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    print(json.dumps(predict(args.checkpoint, args.image), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

