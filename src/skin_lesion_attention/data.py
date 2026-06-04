from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import transforms

from .utils import ensure_dir

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class SyntheticSkinLesionDataset(Dataset):
    def __init__(self, samples: int, image_size: int, num_classes: int) -> None:
        self.samples = samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image = torch.rand(3, self.image_size, self.image_size)
        label = torch.tensor(index % self.num_classes, dtype=torch.long)
        return {"image": image, "label": label, "image_id": f"synthetic_{index}"}


class HAM10000Dataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        class_to_idx: dict[str, int],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.rows = rows
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.class_to_idx[row["dx"]], dtype=torch.long)
        return {
            "image": image,
            "label": label,
            "image_id": row["image_id"],
            "path": row["image_path"],
        }

    @property
    def labels(self) -> list[int]:
        return [self.class_to_idx[row["dx"]] for row in self.rows]


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.15,
                    hue=0.03,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(
                    p=0.15,
                    scale=(0.02, 0.12),
                    ratio=(0.3, 3.3),
                    value="random",
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_image_index(raw_dir: Path, image_dirs: list[str]) -> dict[str, str]:
    roots = [raw_dir / name for name in image_dirs if (raw_dir / name).exists()]
    if not roots:
        roots = [raw_dir]
    index: dict[str, str] = {}
    for root in roots:
        for image_path in root.rglob("*"):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                index[image_path.stem] = str(image_path)
    return index


def read_metadata(raw_dir: Path, metadata_file: str, image_dirs: list[str]) -> list[dict[str, str]]:
    metadata_path = raw_dir / metadata_file
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not find {metadata_path}. Place HAM10000_metadata.csv under data/raw."
        )

    image_index = build_image_index(raw_dir, image_dirs)
    rows: list[dict[str, str]] = []
    with metadata_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row.get("image_id", "").strip()
            label = row.get("dx", "").strip()
            if not image_id or not label:
                continue
            image_path = image_index.get(image_id)
            if image_path is None:
                continue
            cleaned = dict(row)
            cleaned["image_id"] = image_id
            cleaned["dx"] = label
            cleaned["image_path"] = image_path
            rows.append(cleaned)

    if not rows:
        raise RuntimeError(
            "No metadata rows could be matched to image files. Check data/raw image folders."
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stratified_split(
    rows: list[dict[str, str]],
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["dx"]].append(row)

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    for class_rows in by_class.values():
        class_rows = class_rows[:]
        rng.shuffle(class_rows)
        n = len(class_rows)
        n_test = max(1, round(n * test_size)) if n >= 3 else 0
        n_val = max(1, round(n * val_size)) if n - n_test >= 3 else 0
        test_rows.extend(class_rows[:n_test])
        val_rows.extend(class_rows[n_test : n_test + n_val])
        train_rows.extend(class_rows[n_test + n_val :])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def stratified_group_split(
    rows: list[dict[str, str]],
    group_key: str,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = row.get(group_key) or row.get("image_id") or str(index)
        groups[key].append(row)

    by_class: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for group_rows in groups.values():
        by_class[group_rows[0]["dx"]].append(group_rows)

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    for class_groups in by_class.values():
        class_groups = class_groups[:]
        rng.shuffle(class_groups)
        n = len(class_groups)
        n_test = max(1, round(n * test_size)) if n >= 3 else 0
        n_val = max(1, round(n * val_size)) if n - n_test >= 3 else 0
        for group_rows in class_groups[:n_test]:
            test_rows.extend(group_rows)
        for group_rows in class_groups[n_test : n_test + n_val]:
            val_rows.extend(group_rows)
        for group_rows in class_groups[n_test + n_val :]:
            train_rows.extend(group_rows)

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def prepare_splits(config: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    data_cfg = config["data"]
    processed_dir = Path(data_cfg["processed_dir"])
    split_paths = {
        "train": processed_dir / "train.csv",
        "val": processed_dir / "val.csv",
        "test": processed_dir / "test.csv",
    }

    if not data_cfg.get("regenerate_splits", False) and all(path.exists() for path in split_paths.values()):
        return {name: read_rows(path) for name, path in split_paths.items()}

    rows = read_metadata(
        Path(data_cfg["raw_dir"]),
        data_cfg["metadata_file"],
        data_cfg.get("image_dirs", []),
    )
    group_by = str(data_cfg.get("group_by", "")).strip()
    if group_by:
        train_rows, val_rows, test_rows = stratified_group_split(
            rows,
            group_by,
            float(data_cfg.get("val_size", 0.15)),
            float(data_cfg.get("test_size", 0.15)),
            int(config["project"]["seed"]),
        )
    else:
        train_rows, val_rows, test_rows = stratified_split(
            rows,
            float(data_cfg.get("val_size", 0.15)),
            float(data_cfg.get("test_size", 0.15)),
            int(config["project"]["seed"]),
        )
    for name, split_rows in {"train": train_rows, "val": val_rows, "test": test_rows}.items():
        write_rows(split_paths[name], split_rows)
    return {"train": train_rows, "val": val_rows, "test": test_rows}


def create_dataloaders(config: dict[str, Any]) -> tuple[dict[str, DataLoader], dict[str, int], list[str]]:
    data_cfg = config["data"]
    classes = data_cfg["classes"]
    class_to_idx = {name: index for index, name in enumerate(classes)}
    image_size = int(data_cfg["image_size"])
    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 0))

    if data_cfg.get("synthetic", False):
        full = SyntheticSkinLesionDataset(
            samples=int(data_cfg.get("synthetic_samples", 28)),
            image_size=image_size,
            num_classes=len(classes),
        )
        train_len = max(1, int(len(full) * 0.7))
        val_len = max(1, int(len(full) * 0.15))
        test_len = len(full) - train_len - val_len
        if test_len <= 0:
            test_len = 1
            train_len -= 1
        generator = torch.Generator().manual_seed(int(config["project"]["seed"]))
        train_set, val_set, test_set = random_split(full, [train_len, val_len, test_len], generator)
    else:
        splits = prepare_splits(config)
        train_set = HAM10000Dataset(splits["train"], class_to_idx, build_transforms(image_size, train=True))
        val_set = HAM10000Dataset(splits["val"], class_to_idx, build_transforms(image_size, train=False))
        test_set = HAM10000Dataset(splits["test"], class_to_idx, build_transforms(image_size, train=False))

    train_sampler = None
    train_shuffle = True
    if not data_cfg.get("synthetic", False) and data_cfg.get("balanced_sampler", False):
        labels = getattr(train_set, "labels", [])
        if labels:
            counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=len(classes)).float()
            counts = torch.clamp(counts, min=1.0)
            power = float(data_cfg.get("sampler_weight_power", 1.0))
            class_sample_weights = torch.pow(1.0 / counts, power)
            sample_weights = class_sample_weights[torch.tensor(labels, dtype=torch.long)]
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_shuffle = False

    dataloaders = {
        "train": DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
        ),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return dataloaders, class_to_idx, classes
