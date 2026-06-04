# Current Project Report

## Active Architecture

The project now keeps only the current architecture:

```text
ResNet-50 backbone + custom multi-scale CBAM attention head
```

The ResNet-50 backbone is ImageNet-pretrained and produces a 2048-channel
feature map. The custom head applies three parallel convolution branches:

| Branch | Purpose | Output |
| --- | --- | --- |
| 3x3 convolution | Local lesion texture | 128 channels |
| 5x5 convolution | Wider lesion morphology | 128 channels |
| Dilated 3x3 convolution | Larger contextual patterns | 128 channels |

The branch outputs are concatenated into 384 channels, refined with CBAM
channel and spatial attention, fused with a 1x1 convolution to 256 channels,
then classified with global average pooling, dropout, and a dense seven-class
classifier.

## Training Setup

The current training setup is:

```yaml
image_size: 299
batch_size: 16
loss: class_balanced_focal
class_balanced_beta: 0.999
focal_gamma: 2.0
label_smoothing: 0.05
optimizer: AdamW
freeze_backbone_epochs: 2
backbone_learning_rate: 0.00001
head_learning_rate: 0.0001
dropout: 0.3
weight_decay: 0.0002
balanced_sampler: true
sampler_weight_power: 0.5
scheduler: ReduceLROnPlateau
early_stopping_metric: validation macro F1
early_stopping_patience: 7
early_stopping_min_delta: 0.0005
```

## Output Classes

```text
nv, mel, bkl, bcc, akiec, vasc, df
```

## Full Training Run

The active configuration was updated to a maximum of 40 epochs with early
stopping enabled.

```text
max epochs: 40
early stopping: true
early stopping patience: 5
early stopping min delta: 0.001
```

Training stopped early at epoch 14. The best validation checkpoint was saved
from epoch 9.

| Metric | Value |
| --- | ---: |
| Best validation accuracy | 0.8428 |
| Best validation macro precision | 0.7396 |
| Best validation macro recall | 0.7056 |
| Best validation macro F1 | 0.7162 |
| Epochs run | 14 |
| Early stopped | true |

Test evaluation from `outputs/checkpoints/best.pt`:

| Metric | Value |
| --- | ---: |
| Test accuracy | 0.8187 |
| Test macro precision | 0.6975 |
| Test macro recall | 0.6666 |
| Test macro F1 | 0.6803 |

Per-class test F1:

| Class | F1 |
| --- | ---: |
| nv | 0.9124 |
| mel | 0.6281 |
| bkl | 0.6319 |
| bcc | 0.7093 |
| akiec | 0.4375 |
| vasc | 0.8182 |
| df | 0.6250 |

## Anti-Overfitting Regularization Run

After the first full run, the main problem was overfitting. The best checkpoint
from the first full run had a strong validation result, but the training macro
F1 was much higher than the validation macro F1. To reduce this gap, the next
run used a more conservative setup:

```yaml
image_size: 224
loss: cross_entropy
label_smoothing: 0.05
sampler_weight_power: 0.25
dropout: 0.4
weight_decay: 0.0005
RandomErasing p: 0.25
early_stopping_patience: 4
early_stopping_min_delta: 0.001
```

This run stopped early at epoch 10. The best checkpoint was from epoch 6.

| Metric | Value |
| --- | ---: |
| Best validation accuracy | 0.8434 |
| Best validation macro F1 | 0.7008 |
| Train macro F1 at best epoch | 0.7753 |
| Epochs run | 10 |
| Early stopped | true |

Test evaluation:

| Metric | Value |
| --- | ---: |
| Test accuracy | 0.8101 |
| Test macro precision | 0.6737 |
| Test macro recall | 0.6325 |
| Test macro F1 | 0.6475 |

Per-class test F1:

| Class | F1 |
| --- | ---: |
| nv | 0.9144 |
| mel | 0.5548 |
| bkl | 0.6184 |
| bcc | 0.6627 |
| akiec | 0.4037 |
| vasc | 0.7727 |
| df | 0.6061 |

This run reduced overfitting, but it underperformed the previous checkpoint on
the held-out test set. The conclusion was that regularization alone was not
enough; it made the model more stable but reduced useful learning capacity.

## Class-Balanced Focal Loss and Larger Image Size

The next change targeted the two practical weaknesses that remained:

1. The model was still weak on minority and clinically important classes.
2. The `224x224` input size may have been losing lesion detail.

The setup was changed to:

```yaml
image_size: 299
loss: class_balanced_focal
class_balanced_beta: 0.999
focal_gamma: 2.0
label_smoothing: 0.05
dropout: 0.3
weight_decay: 0.0002
sampler_weight_power: 0.5
early_stopping_patience: 7
early_stopping_min_delta: 0.0005
```

An initial attempt used `batch_size: 32`, but this was too heavy for the 6GB
RTX 3060 laptop GPU. GPU memory reached about `5975 / 6144 MiB`, and epoch 4
ran for more than 15 minutes after epoch 3 had taken about 5 minutes. That run
was stopped and the batch size was reduced to 16.

The final run used:

```yaml
image_size: 299
batch_size: 16
loss: class_balanced_focal
class_balanced_beta: 0.999
focal_gamma: 2.0
sampler_weight_power: 0.5
```

Training stopped early at epoch 30. The best checkpoint was saved from epoch
23.

| Metric | Value |
| --- | ---: |
| Best validation accuracy | 0.8541 |
| Best validation macro precision | 0.7268 |
| Best validation macro recall | 0.7522 |
| Best validation macro F1 | 0.7367 |
| Train macro F1 at best epoch | 0.9393 |
| Epochs run | 30 |
| Early stopped | true |

Per-class validation F1 at the best checkpoint:

| Class | F1 |
| --- | ---: |
| nv | 0.9352 |
| mel | 0.5575 |
| bkl | 0.7375 |
| bcc | 0.8047 |
| akiec | 0.5905 |
| vasc | 0.8889 |
| df | 0.6429 |

Test evaluation from `outputs/checkpoints/best.pt`:

| Metric | Value |
| --- | ---: |
| Test accuracy | 0.8246 |
| Test macro precision | 0.6882 |
| Test macro recall | 0.6834 |
| Test macro F1 | 0.6814 |

Per-class test F1:

| Class | F1 |
| --- | ---: |
| nv | 0.9149 |
| mel | 0.6048 |
| bkl | 0.6667 |
| bcc | 0.7474 |
| akiec | 0.4898 |
| vasc | 0.7907 |
| df | 0.5556 |

Compared with the first full run, this setup improved validation macro F1 from
`0.7162` to `0.7367`. However, the test macro F1 only changed from `0.6803` to
`0.6814`. This means the new setup learned a better validation checkpoint but
did not produce a meaningful held-out test improvement.

## Current Evaluation

The strongest validation result so far is the class-balanced focal loss run at
epoch 23:

```text
validation macro F1: 0.7367
validation accuracy: 0.8541
```

The strongest held-out test result so far is still effectively tied with the
first full run:

```text
first full run test macro F1: 0.6803
latest run test macro F1:    0.6814
```

The project therefore improved validation performance but has not yet achieved
a major test-set gain. The main remaining issue is generalization. In the latest
best checkpoint, train macro F1 was `0.9393` while validation macro F1 was
`0.7367`, a gap of about `0.2026`.

The most practical next experiment is to keep class-balanced focal loss and
`299x299` input size, but reduce the sampler pressure:

```yaml
sampler_weight_power: 0.3
```

The reason is that focal loss already handles class imbalance. Combining it
with `sampler_weight_power: 0.5` may be over-correcting minority classes and
causing unstable validation behavior.
