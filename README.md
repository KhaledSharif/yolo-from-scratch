# YOLO from Scratch

A minimal PyTorch implementation of YOLO (You Only Look Once) object detection, built from scratch for educational purposes and practical single-class detection tasks.

## Features

- **Anchor-based detection** with 3 anchors per grid cell (13×13 grid = 507 possible detections)
- **CIoU loss** for accurate bounding box regression
- **Multi-class support** with configurable number of classes
- **NMS (Non-Maximum Suppression)** for removing duplicate detections
- **Interactive evaluation viewer** with OpenCV for visual inspection
- **Comprehensive metrics**: Precision, Recall, F1 Score

## Requirements

```bash
pip install torch torchvision pillow opencv-python numpy pyyaml tqdm
```

Tested with Python 3.8+ and PyTorch 1.10+.

## Quick Start

### 1. Prepare Your Dataset

Organize your dataset in YOLO format:

```
dataset/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── labels/
│       ├── img1.txt
│       └── img2.txt
└── val/
    ├── images/
    └── labels/
```

**Label format** (one line per object):
```
class_id x_center y_center width height
```
All coordinates are normalized to [0, 1].

**Dataset configuration** (`dataset.yaml`):
```yaml
nc: 1  # number of classes
names: ['cone']  # class names
train: /path/to/dataset/train/images
val: /path/to/dataset/val/images
```

### 2. Train

```bash
python train.py dataset.yaml
```

Training runs for 100 epochs and saves the model as `yolo_YYYYMMDD_HHMMSS.pt`.

**Training output:**
```
Epoch 1: Loss: 12.3456 (bbox: 2.34, obj: 8.90, cls: 1.11) | Val: Loss 11.23, P 45.2%, R 38.7%, F1 41.7%
Epoch 2: Loss: 10.2345 (bbox: 1.89, obj: 7.45, cls: 0.90) | Val: Loss 9.87, P 52.3%, R 46.8%, F1 49.4%
...
```

### 3. Evaluate

```bash
python train.py dataset.yaml model.pt
```

**Evaluation output:**
```
Training Set:
  Loss: 2.3456
  Precision: 87.34%
  Recall: 82.15%
  F1 Score: 84.66%

Validation Set:
  Loss: 3.1234
  Precision: 79.21%
  Recall: 74.38%
  F1 Score: 76.72%
```

### 4. Run Inference

```bash
python train.py image.jpg model.pt
```

**Inference output:**
```
Detected 3 object(s):
  1. Box: (145.2, 230.5, 198.7, 310.2), Confidence: 0.876, Class: 0
  2. Box: (320.1, 180.3, 365.4, 250.8), Confidence: 0.823, Class: 0
  3. Box: (510.5, 290.7, 548.2, 350.1), Confidence: 0.791, Class: 0
```

### 5. Interactive Viewer

```bash
python eval.py model.pt dataset.yaml
```

Visualize predictions vs. ground truth:
- **Green boxes**: Ground truth
- **Red boxes**: Predictions
- **Arrow keys / A / D**: Navigate images
- **S**: Save screenshot
- **Q / ESC**: Quit

## Architecture

```
Input (416×416×3)
    ↓
Conv2d(3→32, stride=2) + BN + LeakyReLU
    ↓ (208×208)
Conv2d(32→64, stride=2) + BN + LeakyReLU
    ↓ (104×104)
Conv2d(64→128, stride=2) + BN + LeakyReLU
    ↓ (52×52)
Conv2d(128→256, stride=2) + BN + LeakyReLU
    ↓ (26×26)
Conv2d(256→512, stride=2) + BN + LeakyReLU
    ↓ (13×13)
Conv2d(512→3×(5+nc), kernel=1)
    ↓
Output (13×13×3×(5+nc))
```

**Each anchor predicts:**
- `x, y, w, h`: Bounding box coordinates (normalized)
- `objectness`: Confidence score
- `class_probs`: Class probabilities (nc values)

**Default anchors (optimized for cones):**
```python
[[10, 13], [16, 30], [33, 23]]  # [width, height] in pixels at 416×416
```

## Loss Function

**Composite loss with three components:**

1. **Bounding Box Loss (CIoU)** - weight: 5.0
   - Complete IoU considering overlap + center distance + aspect ratio
   - Applied only to cells with objects

2. **Objectness Loss (BCE)** - weight: 1.0
   - Binary cross-entropy for object presence
   - Applied to all grid cells

3. **Classification Loss (BCE)** - weight: 1.0
   - Binary cross-entropy for class probabilities
   - Applied only to cells with objects

## Advanced Usage

### Inspect Model

```bash
python train.py model.pt
```

Displays architecture and parameter counts.

### Custom Anchors

Modify anchor boxes in the code or pass custom anchors:

```python
model = YOLO(num_classes=3, anchors=[[15, 20], [25, 40], [45, 60]])
dataset = YOLODataset(img_dir, num_classes=3, anchors=[[15, 20], [25, 40], [45, 60]])
```

For best results, compute optimal anchors using k-means clustering on your training set bounding boxes.

### Multi-class Detection

Update `dataset.yaml`:
```yaml
nc: 3
names: ['car', 'person', 'bike']
train: /path/to/train/images
val: /path/to/val/images
```

The model automatically adapts to the number of classes.

## Project Structure

```
yolo-from-scratch/
├── train.py        # Main script: training, evaluation, inference
├── eval.py         # Interactive OpenCV viewer
├── CLAUDE.md       # Technical documentation for Claude Code
└── README.md       # This file
```

**train.py contains:**
- `YOLODataset`: Dataset loader with anchor matching
- `YOLO`: Model architecture
- `ciou_loss`: Complete IoU loss function
- `yolo_loss`: Composite loss function
- `train_epoch`: Training loop
- `eval_epoch`: Evaluation with metrics
- `predict`: Inference with NMS
- CLI interface for all operations

## Implementation Notes

### What This Implementation Has

✅ Anchor-based detection (3 anchors per grid cell)
✅ CIoU loss for bbox regression
✅ BCE loss for objectness and classification
✅ Multi-class support
✅ Non-Maximum Suppression (NMS)
✅ Proper anchor matching during training
✅ Grid-based detection (13×13)

### What This Implementation Lacks (vs. YOLOv5+)

❌ Multi-scale detection (only 13×13 grid)
❌ Feature Pyramid Network (FPN) / PANet
❌ CSPDarknet backbone
❌ Data augmentation (Mosaic, MixUp, HSV)
❌ Advanced training techniques (warmup, cosine LR, label smoothing)
❌ True mAP evaluation (uses grid-based precision/recall)
❌ Automatic anchor optimization (k-means)

This is a **minimal educational implementation** suitable for learning YOLO concepts and simple detection tasks. For production use, consider YOLOv5, YOLOv8, or other established frameworks.

## Tips for Best Results

1. **Balance your dataset**: Ensure roughly equal numbers of examples per class
2. **Use appropriate anchors**: Tune anchors to match your object sizes
3. **Monitor F1 score**: Best indicator of overall detection quality
4. **Adjust confidence threshold**: Lower for more detections, higher for fewer false positives
5. **Check with eval.py**: Visually inspect predictions to diagnose issues
6. **Train longer**: 100 epochs may not be enough for complex datasets

## Troubleshooting

**No objects detected:**
- Lower confidence threshold in `predict()` (default: 0.5)
- Check if model is trained (loss should decrease over epochs)
- Verify label format is correct

**Poor accuracy:**
- Train for more epochs
- Verify anchors match your object sizes
- Check dataset quality and balance
- Ensure bbox coordinates are normalized [0, 1]

**High loss:**
- Check label files match images (same filename)
- Verify coordinates are in correct format (x_center, y_center, w, h)
- Ensure images are in train/images and labels in train/labels
