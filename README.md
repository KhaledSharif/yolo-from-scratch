# YOLO from Scratch

A YOLOv5-inspired PyTorch implementation of YOLO (You Only Look Once) object 
detection with **Feature Pyramid Network (FPN) + PANet** for multi-scale 
detection. Built from scratch for educational purposes and practical applications, 
with excellent small object detection capabilities.

## Features

- **YOLOv5 FPN Architecture**: Multi-scale detection with Feature Pyramid Network + PANet
- **Three Detection Heads**: P3 (stride 8), P4 (stride 16), P5 (stride 32) for small/medium/large objects
- **C3 Modules**: CSP Bottleneck with residual connections (YOLOv5 building blocks)
- **SPPF Module**: Spatial Pyramid Pooling - Fast for multi-scale receptive fields
- **Scale-specific Anchors**: 9 anchors optimized for different object sizes
- **Configurable Resolution**: Default 640×640, supports 512, 1024, 1280, etc.
- **CIoU loss** for accurate bounding box regression
- **Multi-class support** with configurable number of classes
- **Global NMS**: Cross-scale Non-Maximum Suppression for removing duplicates
- **Interactive evaluation viewer** with OpenCV for visual inspection
- **Comprehensive metrics**: Precision, Recall, F1 Score

## Requirements

```bash
pip install torch torchvision pillow opencv-python numpy pyyaml tqdm

# For testing (optional)
pip install pytest pytest-cov
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
# Default 640×640 resolution with YOLOv5-style LR scheduling
python train.py dataset.yaml

# Custom resolution (must be divisible by 32)
python train.py dataset.yaml --img-size 512   # Faster, lower memory
python train.py dataset.yaml --img-size 1280  # Higher accuracy, more memory

# Customize learning rate schedule
python train.py dataset.yaml --lr 0.02          # Higher initial LR
python train.py dataset.yaml --min-lr 0.0001    # Minimum LR at end
python train.py dataset.yaml --warmup-epochs 5  # Longer warmup
python train.py dataset.yaml --epochs 50        # Shorter training

# Combined options
python train.py dataset.yaml --img-size 1024 --lr 0.015 --epochs 150
```

**Learning Rate Schedule (YOLOv5-style):**
- **Warmup**: Linear ramp from 1e-6 → 0.01 over 3 epochs (prevents early instability)
- **Cosine Annealing**: Smooth decay from 0.01 → 0.0001 over remaining epochs
- **Gradient Clipping**: max_norm=10.0 to prevent exploding gradients
- **10× higher peak LR** than previous fixed-rate implementation for faster convergence

Training runs for 100 epochs by default and saves the model as `yolo_YYYYMMDD_HHMMSS.pt`.
The checkpoint includes img_size, so it will be loaded automatically during inference/evaluation.

**Training output:**
```
Epoch 1: Loss: 12.3456 (bbox: 2.34, obj: 8.90, cls: 1.11) | Val: Loss 11.23, P 45.2%, R 38.7%, F1 41.7% | LR: 0.000001
Epoch 2: Loss: 10.2345 (bbox: 1.89, obj: 7.45, cls: 0.90) | Val: Loss 9.87, P 52.3%, R 46.8%, F1 49.4% | LR: 0.003334
Epoch 4: Loss: 8.1234 (bbox: 1.45, obj: 5.89, cls: 0.78) | Val: Loss 8.12, P 61.7%, R 54.2%, F1 57.7% | LR: 0.010000
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

## Testing

The project includes > 80 comprehensive unit tests covering all major components.

### Running Tests

```bash
# Run all tests
python3 -m pytest tests

# Run tests with coverage report
python3 -m pytest tests --cov=train --cov-report=term

# Show which lines are missing coverage
python3 -m pytest tests --cov=train --cov-report=term-missing

# Generate HTML coverage report
python3 -m pytest tests --cov=train --cov-report=html
# Then open htmlcov/index.html in your browser
```

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
