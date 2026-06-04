from __future__ import annotations

import numpy as np


def confusion_matrix(
    targets: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(targets, predictions):
        matrix[int(target), int(pred)] += 1
    return matrix


def classification_metrics(
    targets: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    num_classes: int,
    classes: list[str] | None = None,
) -> dict:
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    matrix = confusion_matrix(targets, predictions, num_classes)
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0

    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    for index in range(num_classes):
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - tp)
        fn = float(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        name = classes[index] if classes else str(index)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matrix[index, :].sum()),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }

