# YOLO from Scratch

A YOLOv5-inspired PyTorch implementation of YOLO (You Only Look Once) object detection, built from scratch for educational purposes and practical applications.

## Features

- **YOLOv5 Architecture**: SPPF module + SiLU activation + offset prediction
- **Configurable Resolution**: Default 640×640, supports 416, 512, 1280, etc.
- **Anchor-based detection** with 3 anchors per grid cell
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
# Default 640×640 resolution
python train.py dataset.yaml

# Custom resolution (must be divisible by 32)
python train.py dataset.yaml --img-size 512   # Faster, lower memory
python train.py dataset.yaml --img-size 1280  # Higher accuracy, more memory
```

Training runs for 100 epochs and saves the model as `yolo_YYYYMMDD_HHMMSS.pt`.
The checkpoint includes img_size, so it will be loaded automatically during inference/evaluation.

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

## Architecture (YOLOv5-style)

```
Input (configurable, default 640×640×3)
    ↓
━━━ Backbone (5 stride-2 Conv layers) ━━━
Conv2d(3→32, stride=2) + BN + SiLU
    ↓ (320×320)
Conv2d(32→64, stride=2) + BN + SiLU
    ↓ (160×160)
Conv2d(64→128, stride=2) + BN + SiLU
    ↓ (80×80)
Conv2d(128→256, stride=2) + BN + SiLU
    ↓ (40×40)
Conv2d(256→512, stride=2) + BN + SiLU
    ↓ (20×20)
━━━ SPPF Module (Multi-scale pooling) ━━━
Conv 512→256 + MaxPool×3 sequential + Conv 1024→512
    ↓ (20×20×512)
━━━ Detection Head ━━━
Conv2d(512→3×(5+nc), kernel=1)
    ↓
Output (20×20×3×(5+nc))  [grid_size = img_size // 32]
```

**Key Architecture Features:**
- **SiLU Activation**: YOLOv5's standard activation (smoother gradients than LeakyReLU)
- **SPPF Module**: Spatial Pyramid Pooling - Fast (2× faster than SPP, same output)
- **Offset Prediction**: Outputs encoded offsets (t_x, t_y, t_w, t_h), decoded via YOLOv5 formulas
- **Dynamic Grid Size**: Automatically calculated as img_size // 32

**Each anchor predicts:**
- `t_x, t_y, t_w, t_h`: **Encoded offsets** (constrained predictions)
- `objectness`: Confidence logit
- `class_probs`: Class logits (nc values)

**Decoding (YOLOv5 formulas):**
```python
b_x = ((σ(t_x) * 2 - 0.5) + c_x) / grid_size
b_y = ((σ(t_y) * 2 - 0.5) + c_y) / grid_size
b_w = (anchor_w * (2σ(t_w))²) / img_size
b_h = (anchor_h * (2σ(t_h))²) / img_size
```

**Default anchors:**
```python
[[10, 13], [16, 30], [33, 23]]  # [width, height] in pixels
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

Displays architecture, parameter counts, and img_size.

### Custom Resolution

```bash
# Training with different resolutions
python train.py dataset.yaml --img-size 416   # Faster, ~30% less memory
python train.py dataset.yaml --img-size 512   # Good balance
python train.py dataset.yaml --img-size 640   # Default (best for most tasks)
python train.py dataset.yaml --img-size 1280  # High accuracy, 4× memory

# Resolution must be divisible by 32 (due to 5 stride-2 layers)
```

**Note**: Model automatically uses saved img_size during inference/eval. Override with `--img-size` if needed.

### Custom Anchors

Modify anchor boxes in the code or pass custom anchors:

```python
model = YOLO(num_classes=3, anchors=[[15, 20], [25, 40], [45, 60]], img_size=640)
dataset = YOLODataset(img_dir, num_classes=3, anchors=[[15, 20], [25, 40], [45, 60]], img_size=640)
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
- `SPPF`: Spatial Pyramid Pooling - Fast module
- `YOLO`: Model architecture with backbone, SPPF, and detection head
- `decode_predictions`: YOLOv5-style offset decoding
- `ciou_loss`: Complete IoU loss function
- `yolo_loss`: Composite loss function with decoding
- `train_epoch`: Training loop
- `eval_epoch`: Evaluation with metrics
- `predict`: Inference with NMS
- CLI interface for all operations

## Implementation Notes

### YOLOv5 Features Implemented

✅ **SPPF Module**: Spatial Pyramid Pooling - Fast (2× faster than SPP)
✅ **SiLU Activation**: YOLOv5's standard activation function
✅ **Offset Prediction**: YOLOv5-style encoded predictions with decoding
✅ **CIoU Loss**: Complete IoU for bbox regression
✅ **Configurable Resolution**: 416, 512, 640, 1280, etc.
✅ **Anchor-based Detection**: 3 anchors per grid cell with IoU matching
✅ **Multi-class Support**: Configurable number of classes
✅ **NMS**: Non-Maximum Suppression for duplicate removal

### What This Implementation Lacks (vs. Full YOLOv5)

❌ Multi-scale detection (only single grid size, not P3/P4/P5)
❌ Feature Pyramid Network (FPN) / PANet neck
❌ Full CSPDarknet backbone with C3 blocks
❌ Data augmentation (Mosaic, MixUp, HSV)
❌ Advanced training (warmup, cosine LR, label smoothing)
❌ True mAP evaluation (uses grid-based precision/recall)
❌ Automatic anchor optimization (k-means clustering)

## Tips for Best Results

1. **Start with default 640×640**: Good balance between speed and accuracy for most tasks
2. **Adjust resolution based on object sizes**:
   - Small objects: Use 1280×1280 (more GPU memory required)
   - Large objects or speed priority: Use 512×512 or 416×416
3. **Balance your dataset**: Ensure roughly equal numbers of examples per class
4. **Use appropriate anchors**: Tune anchors to match your object sizes (k-means on training set)
5. **Monitor F1 score**: Best indicator of overall detection quality
6. **Adjust confidence threshold**: Lower for more detections, higher for fewer false positives
7. **Check with eval.py**: Visually inspect predictions to diagnose issues
8. **Train longer**: 100 epochs may not be enough for complex datasets

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
