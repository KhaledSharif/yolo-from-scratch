# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a YOLOv5-inspired object detection implementation from scratch in PyTorch. The implementation uses a single-scale detection head with configurable input resolution, anchor boxes, and YOLOv5 features (SPPF, SiLU, offset prediction). Suitable for single-class or multi-class object detection tasks.

## Architecture

**Model Structure (YOLOv5-style):**
- Input: Configurable square images (default 640×640, supports 416, 512, 1280, etc.)
- **Backbone**: 5 stride-2 conv layers (3→32→64→128→256→512) with BatchNorm + **SiLU activation**
- **SPPF**: Spatial Pyramid Pooling - Fast (512→512) for multi-scale receptive fields
- **Detection Head**: 1×1 conv to output channels
- Grid size: Dynamic, calculated as `img_size // 32` (e.g., 640→20, 512→16, 416→13)
- Output shape: `(batch, grid_size, grid_size, 3, 5+num_classes)` where each anchor predicts:
  - `t_x, t_y, t_w, t_h`: **Encoded offsets** (decoded via YOLOv5 formulas)
  - `objectness`: Confidence logit
  - `class_probs`: Class logits

**YOLOv5 Offset Prediction:**
- Model outputs encoded offsets, not direct coordinates
- Decoding formulas constrain predictions near responsible grid cell:
  - `b_x = ((σ(t_x) * 2 - 0.5) + c_x) / grid_size`
  - `b_y = ((σ(t_y) * 2 - 0.5) + c_y) / grid_size`
  - `b_w = (anchor_w * (2σ(t_w))²) / img_size`
  - `b_h = (anchor_h * (2σ(t_h))²) / img_size`
- Improves training stability by preventing extreme predictions

**Anchor Boxes:**
- Default anchors: `[[10, 13], [16, 30], [33, 23]]` (width, height in pixels)
- Each ground truth box is matched to the best anchor using shape-based IoU
- Automatically scaled to img_size during decoding
- Total detection capacity: `grid_size² × 3` predictions per image

**Loss Function:**
- Bounding box regression: CIoU (Complete IoU) loss
  - Considers IoU + center distance penalty + aspect ratio consistency
  - Applied only to cells with objects, weighted 5×
- Objectness: Binary cross-entropy with logits (all cells)
- Classification: Binary cross-entropy with logits (cells with objects only)

**Key Implementation Details:**
- YOLODataset performs anchor matching during target generation
- Model outputs raw encoded offsets; decoded via `decode_predictions()` function
- Sigmoid applied to objectness/class logits during evaluation/inference
- NMS (Non-Maximum Suppression) applied during inference with configurable IoU threshold
- Grid-based detection: Each cell is responsible for objects whose center falls within it
- SPPF uses sequential max pooling (2× faster than parallel SPP while maintaining same output)

## Commands

### Training
```bash
# Default 640×640 resolution
python train.py dataset.yaml

# Custom resolution (must be divisible by 32)
python train.py dataset.yaml --img-size 512
python train.py dataset.yaml --img-size 1280
```
Trains for 100 epochs, saves model as `yolo_YYYYMMDD_HHMMSS.pt`.
Checkpoint includes img_size, so model can be loaded correctly later.

**Dataset YAML format:**
```yaml
nc: 1  # number of classes
names: ['cone']  # class names
train: /path/to/train/images
val: /path/to/val/images
```

Expected directory structure:
```
dataset/
├── train/
│   ├── images/  # .jpg or .png files
│   └── labels/  # corresponding .txt files
└── val/
    ├── images/
    └── labels/
```

Label format (YOLO format, one box per line):
```
class_id x_center y_center width height
```
All coordinates normalized to [0, 1].

### Evaluation
```bash
# Automatically uses img_size from checkpoint
python train.py dataset.yaml model.pt

# Override with custom img_size (not recommended)
python train.py dataset.yaml model.pt --img-size 512
```
Computes loss, precision, recall, F1 score on both train and val sets.
Metrics use IoU threshold 0.5 and confidence threshold 0.5 to classify predictions as TP/FP/FN.

### Inference
```bash
# Automatically uses img_size from checkpoint
python train.py image.jpg model.pt

# Can override, but may affect accuracy
python train.py image.jpg model.pt --img-size 512
```
Runs detection on single image, outputs bounding boxes with confidence scores.

### Interactive Evaluation Viewer
```bash
python eval.py model.pt dataset.yaml
```
Automatically loads img_size from checkpoint.
Opens OpenCV window showing:
- Ground truth boxes (green)
- Predicted boxes (red)
- Navigate with arrow keys/A/D, quit with Q/ESC, save screenshot with S

### Model Inspection
```bash
python train.py model.pt
```
Displays model architecture, parameter counts, and img_size.

## Important Implementation Notes

**Target Generation (YOLODataset.__getitem__):**
- Iterates over all boxes in label file
- Assigns each box to grid cell based on center coordinates
- Matches box to best anchor using shape-based IoU (position-agnostic)
- Only assigns if cell+anchor pair is empty (first box wins if multiple overlap)

**Inference Pipeline (predict function):**
1. Resize image to model.img_size (e.g., 640×640)
2. Forward pass through model → `(1, grid_size, grid_size, 3, 5+nc)` RAW offsets
3. Decode predictions using YOLOv5 formulas (offsets → absolute coords)
4. Apply sigmoid to objectness and class predictions
5. Extract detections above confidence threshold
6. Convert normalized coords → pixel coords → original image scale
7. Apply NMS with IoU threshold 0.4

**Training Loop:**
- Adam optimizer, lr=1e-3
- Batch size 8
- Composite loss printed with breakdown: bbox, objectness, class
- Validation metrics: Loss, Precision, Recall, F1
- Checkpoint saved every epoch (overwrites), includes `img_size` and `num_classes`

**Decoding in Loss Function:**
- `yolo_loss()` internally calls `decode_predictions()` before computing CIoU loss
- Objectness and class losses use raw logits (not decoded)
- Gradients flow through decoding function to train offset prediction

## Known Limitations

This is a minimal implementation. Compared to production YOLO (YOLOv5+):
- Single-scale detection only (one grid size)
- No multi-scale architecture (FPN/PANet neck with P3/P4/P5 heads)
- No data augmentation (mosaic, mixup, HSV, etc.)
- Simple backbone (not full CSPDarknet with C3 blocks)
- No label smoothing, warmup, or cosine LR scheduling
- Evaluation metrics are grid-based (not true AP/mAP)
- No automatic anchor optimization (k-means clustering)

**YOLOv5 Features Implemented:**
✅ SPPF (Spatial Pyramid Pooling - Fast)
✅ SiLU activation function
✅ YOLOv5-style offset prediction with decoding
✅ CIoU loss for bbox regression
✅ Anchor-based detection with IoU matching
✅ Configurable input resolution

Despite remaining limitations, the implementation correctly handles anchor-based detection, multi-class classification, and produces reasonable results for simple detection tasks.

## Extending the Implementation

**To add more classes:**
1. Update `nc` in dataset.yaml
2. Ensure label files use correct class IDs (0 to nc-1)
3. Model automatically adapts output dimensions

**To modify anchors:**
Pass custom anchors to constructors:
```python
model = YOLO(num_classes=3, anchors=[[w1,h1], [w2,h2], [w3,h3]], img_size=640)
dataset = YOLODataset(img_dir, num_classes=3, anchors=[[w1,h1], [w2,h2], [w3,h3]], img_size=640)
```
Anchors should be tuned to your dataset using k-means clustering on training box dimensions.

**To change image resolution:**
Use `--img-size` CLI argument (must be divisible by 32):
```bash
python train.py dataset.yaml --img-size 1280  # High resolution
python train.py dataset.yaml --img-size 512   # Lower resolution, faster training
```
Grid size is automatically calculated as `img_size // 32`.
