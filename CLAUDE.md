# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a minimal YOLO (You Only Look Once) object detection implementation from scratch using PyTorch. The entire implementation is contained in a single file (`yolo.py`) for simplicity and educational purposes.

## Architecture

### Model Structure (YOLO class)
- Simple convolutional architecture with 5 downsampling blocks
- Each block: Conv2d → BatchNorm2d → LeakyReLU
- Input: 416x416 RGB images
- Output: 13x13 grid with 5 values per cell (x, y, w, h, objectness)
- Uses MSE loss for training (bounding box regression)

### Dataset (YOLODataset class)
- Expects YOLO format: images in `/images/` directory, labels in `/labels/` directory
- Label format: class x_center y_center width height (normalized 0-1)
- Images resized to 416x416
- Target tensor shape: (13, 13, 5)

### Training Pipeline
- `train_epoch()`: Training loop with gradient updates
- `eval_epoch()`: Validation loop without gradients
- Optimizer: Adam with lr=1e-3
- Batch size: 8
- Default epochs: 100

## Commands

### Training
```bash
python yolo.py data.yaml
```
Requires a YAML config file with `train` and `val` keys pointing to image directories.

Example `data.yaml`:
```yaml
train: /path/to/train/images
val: /path/to/val/images
```

Model is saved as `yolo_YYYYMMDD_HHMMSS.pt` with model state dict and epoch number.

### Evaluation
```bash
python yolo.py data.yaml model.pt
```
Loads a trained model and evaluates on train/val sets, printing accuracies.

### Model Inspection
```bash
python yolo.py model.pt
```
Loads and prints model architecture with parameter shapes and counts.

## Key Implementation Details

- The model operates on a fixed 13x13 grid due to 5 stride-2 convolutions (416 → 208 → 104 → 52 → 26 → 13)
- Target tensor currently uses simplified logic: broadcasts first box to entire grid (see yolo.py:30)
- Accuracy metric: objectness prediction threshold at 0.5
- No anchor boxes or multi-scale predictions (unlike full YOLO implementations)
- No NMS (Non-Maximum Suppression) in current implementation

## Dependencies

- PyTorch (torch, torch.nn, torch.optim)
- PyYAML
- PIL (Pillow)
- tqdm
- glob, pathlib (stdlib)
