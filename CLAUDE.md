# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a minimal YOLO (You Only Look Once) object detection implementation from scratch in PyTorch. The implementation uses a single-scale detection head operating on a 13×13 grid with anchor boxes, suitable for single-class or multi-class object detection tasks (particularly optimized for cone detection).

## Architecture

**Model Structure:**
- Input: 416×416 RGB images
- Backbone: 5 stride-2 conv layers (3→32→64→128→256→512 channels) reducing spatial dimensions from 416→13
- Detection Head: Predicts on 13×13 grid with 3 anchor boxes per cell
- Output shape: `(batch, 13, 13, 3, 5+num_classes)` where each anchor predicts:
  - `x, y, w, h`: Bounding box (normalized coordinates)
  - `objectness`: Confidence that cell contains object
  - `class_probs`: Class probabilities (1+ values)

**Anchor Boxes:**
- Default anchors: `[[10, 13], [16, 30], [33, 23]]` (width, height in pixels at 416×416)
- Each ground truth box is matched to the best anchor using IoU
- Total detection capacity: 13×13×3 = 507 possible predictions per image

**Loss Function:**
- Bounding box regression: CIoU (Complete IoU) loss
  - Considers IoU + center distance penalty + aspect ratio consistency
  - Applied only to cells with objects, weighted 5×
- Objectness: Binary cross-entropy with logits (all cells)
- Classification: Binary cross-entropy with logits (cells with objects only)

**Key Implementation Details:**
- YOLODataset performs anchor matching during target generation
- Model outputs raw logits; sigmoid applied during evaluation/inference
- NMS (Non-Maximum Suppression) applied during inference with configurable IoU threshold
- Grid-based detection: Each cell is responsible for objects whose center falls within it

## Commands

### Training
```bash
python train.py dataset.yaml
```
Trains for 100 epochs, saves model as `yolo_YYYYMMDD_HHMMSS.pt`.

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
python train.py dataset.yaml model.pt
```
Computes loss, precision, recall, F1 score on both train and val sets.

Metrics use IoU threshold 0.5 and confidence threshold 0.5 to classify predictions as TP/FP/FN.

### Inference
```bash
python train.py image.jpg model.pt
```
Runs detection on single image, outputs bounding boxes with confidence scores.

### Interactive Evaluation Viewer
```bash
python eval.py model.pt dataset.yaml
```
Opens OpenCV window showing:
- Ground truth boxes (green)
- Predicted boxes (red)
- Navigate with arrow keys/A/D, quit with Q/ESC, save screenshot with S

### Model Inspection
```bash
python train.py model.pt
```
Displays model architecture and parameter counts.

## Important Implementation Notes

**Target Generation (YOLODataset.__getitem__):**
- Iterates over all boxes in label file
- Assigns each box to grid cell based on center coordinates
- Matches box to best anchor using shape-based IoU (position-agnostic)
- Only assigns if cell+anchor pair is empty (first box wins if multiple overlap)

**Inference Pipeline (predict function):**
1. Resize image to 416×416
2. Forward pass through model → `(1, 13, 13, 3, 5+nc)`
3. Apply sigmoid to objectness and class predictions
4. Extract detections above confidence threshold
5. Convert normalized coords → pixel coords → original image scale
6. Apply NMS with IoU threshold 0.4

**Training Loop:**
- Adam optimizer, lr=1e-3
- Batch size 8
- Composite loss printed with breakdown: bbox, objectness, class
- Validation metrics: Loss, Precision, Recall, F1
- Checkpoint saved every epoch (overwrites)

## Known Limitations

This is a minimal implementation. Compared to production YOLO (YOLOv5+):
- Single-scale detection only (13×13 grid)
- No multi-scale architecture (FPN/PANet)
- No data augmentation (mosaic, mixup, HSV, etc.)
- Simple backbone (not CSPDarknet)
- No label smoothing, warmup, or cosine LR scheduling
- Evaluation metrics are grid-based (not true AP/mAP)
- No automatic anchor optimization (k-means clustering)

Despite these limitations, the implementation correctly handles anchor-based detection, multi-class classification, and produces reasonable results for simple detection tasks.

## Extending the Implementation

**To add more classes:**
1. Update `nc` in dataset.yaml
2. Ensure label files use correct class IDs (0 to nc-1)
3. Model automatically adapts output dimensions

**To modify anchors:**
Pass custom anchors to `YOLO(anchors=[[w1,h1], [w2,h2], ...])` and `YOLODataset(anchors=...)`.
Anchors should be tuned to your dataset (use k-means clustering on training box dimensions).

**To change image resolution:**
Currently hardcoded to 416×416. Changing requires modifying `YOLODataset.img_size` and grid_size calculations (grid_size = img_size / 32 due to 5 stride-2 layers).
