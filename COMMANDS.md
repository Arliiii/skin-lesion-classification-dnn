# Project Command Guide

Run all commands from the project root:

```powershell
cd D:\arlinda
```

## Install Project

```powershell
python -m pip install -e .
```

## Dataset Location

HAM10000 should be placed here:

```text
data/raw/archive_2/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/
```

The training code creates split files here:

```text
data/processed_lesion/train.csv
data/processed_lesion/val.csv
data/processed_lesion/test.csv
```

## Current Training Configuration

The active config file is:

```text
configs/default.yaml
```

Current important settings:

```yaml
image_size: 299
batch_size: 16
loss: class_balanced_focal
class_balanced_beta: 0.999
focal_gamma: 2.0
balanced_sampler: true
sampler_weight_power: 0.5
early_stopping: true
early_stopping_patience: 7
early_stopping_min_delta: 0.0005
```

## Train Model

```powershell
python -m skin_lesion_attention.train --config configs/default.yaml
```

Training writes:

```text
outputs/checkpoints/best.pt
outputs/logs/training_history.csv
outputs/logs/training_summary.json
outputs/figures/loss_curve.png
outputs/figures/macro_f1_curve.png
```

The command prints each epoch to the terminal, including loss, accuracy, macro
F1, learning rate, and epoch time.

## Evaluate Best Checkpoint

```powershell
python -m skin_lesion_attention.evaluate --checkpoint outputs/checkpoints/best.pt --config configs/default.yaml
```

Evaluation writes:

```text
outputs/logs/test_metrics.json
outputs/figures/confusion_matrix.png
```

## Predict One Image

Replace `path\to\image.jpg` with an image path.

```powershell
python -m skin_lesion_attention.predict --checkpoint outputs/checkpoints/best.pt --image path\to\image.jpg
```

This prints the predicted class and class probabilities to the terminal.

## Visualize Attention For One Image

```powershell
python -m skin_lesion_attention.visualize --checkpoint outputs/checkpoints/best.pt --image path\to\image.jpg --attention-cmap jet_r
```

Optional output path:

```powershell
python -m skin_lesion_attention.visualize --checkpoint outputs/checkpoints/best.pt --image path\to\image.jpg --output outputs\figures\single_attention.png --attention-cmap jet_r
```

The default reversed colormap is `jet_r`.

## Create Report Figure From Random Test Images

Create a report-ready figure with one image per class:

```powershell
python -m skin_lesion_attention.visualize_random_test --checkpoint outputs/checkpoints/best.pt --config configs/default.yaml --count 7 --seed 7 --output outputs\figures\random_test_predictions.png --attention-cmap jet_r
```

The figure includes:

- original test image
- true label
- predicted label
- confidence
- correctness
- CBAM attention overlay
- top class probabilities

Output:

```text
outputs/figures/random_test_predictions.png
```

## Change Random Samples

Use a different seed:

```powershell
python -m skin_lesion_attention.visualize_random_test --checkpoint outputs/checkpoints/best.pt --config configs/default.yaml --count 7 --seed 21 --output outputs\figures\random_test_predictions_seed21.png --attention-cmap jet_r
```

## Clean Generated Artifacts

Use this when you want to remove old checkpoints, logs, and figures before a
fresh run. It keeps `.gitkeep` files.

```powershell
Get-ChildItem outputs\checkpoints,outputs\logs,outputs\figures -File -Force |
  Where-Object { $_.Name -ne ".gitkeep" } |
  Remove-Item -Force
```

## Important Output Files

```text
outputs/checkpoints/best.pt
outputs/logs/training_history.csv
outputs/logs/training_summary.json
outputs/logs/test_metrics.json
outputs/figures/loss_curve.png
outputs/figures/macro_f1_curve.png
outputs/figures/confusion_matrix.png
outputs/figures/random_test_predictions.png
```

## Useful Current Result

Latest completed run:

```text
best validation epoch: 23
validation accuracy: 0.8541
validation macro F1: 0.7367
test accuracy: 0.8246
test macro F1: 0.6814
```
