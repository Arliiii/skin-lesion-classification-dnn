<img width="889" height="415" alt="Capture" src="https://github.com/user-attachments/assets/23eb4df6-59e2-488d-b52c-214c72c08f9b" /># ResNet-50 Multi-Scale Attention Skin Lesion Classifier

This project keeps one active architecture: an ImageNet-pretrained ResNet-50
backbone with a custom multi-scale CBAM attention head for seven-class
HAM10000 skin lesion classification.

## Architecture

```text
Input image
  -> ImageNet-pretrained ResNet-50 backbone
  -> 2048-channel feature map
  -> 3x3 convolution branch, 128 channels
  -> 5x5 convolution branch, 128 channels
  -> dilated 3x3 convolution branch, 128 channels
  -> concatenate to 384 channels
  -> CBAM channel + spatial attention
  -> 1x1 fusion to 256 channels
  -> global average pooling
  -> dropout
  -> dense 7-class classifier
```

Training uses class-balanced focal loss, AdamW, frozen-backbone warmup, a lower
backbone learning rate, a higher head learning rate, ReduceLROnPlateau, and
early stopping by validation macro F1.

## Setup

```powershell
python -m pip install -e .
```

## Dataset Layout

Place HAM10000 under:

```text
data/raw/archive_2/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/
```

## Train

```powershell
python -m skin_lesion_attention.train --config configs/default.yaml
```

## Evaluate

```powershell
python -m skin_lesion_attention.evaluate --checkpoint outputs/checkpoints/best.pt --config configs/default.yaml
```

## Predict One Image

```powershell
python -m skin_lesion_attention.predict --checkpoint outputs/checkpoints/best.pt --image path/to/image.jpg
```

## Visualize Attention

```powershell
python -m skin_lesion_attention.visualize --checkpoint outputs/checkpoints/best.pt --image path/to/image.jpg --attention-cmap jet_r
```
<img width="849" height="414" alt="Captfure" src="https://github.com/user-attachments/assets/2445cbb0-0fec-45a3-a841-d87dfda8a9d2" />


## Report Figure From Random Test Images

```powershell
python -m skin_lesion_attention.visualize_random_test --checkpoint outputs/checkpoints/best.pt --config configs/default.yaml --count 6 --attention-cmap jet_r
```
"# skin-lesion-classification-dnn"  
