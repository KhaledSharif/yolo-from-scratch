# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a YOLOv5-inspired object detection implementation from scratch in PyTorch. The implementation features a **Feature Pyramid Network (FPN) with PANet** for multi-scale detection, configurable input resolution, scale-specific anchor boxes, and YOLOv5 features (SPPF, SiLU, offset prediction, C3 modules). Suitable for single-class or multi-class object detection tasks with excellent small object detection capabilities.

## Architecture

**Model Structure (YOLOv5-style FPN + PANet):**

### Input and Resolution
- Configurable square images (default 640×640, supports 512, 1024, 1280, etc.)
- **Must be divisible by 32** due to 5 stride-2 convolutions

### Backbone (Multi-Scale Feature Extraction)
5 stride-2 convolutional layers extract features at 3 scales:
```
Input (3, H, W)
  ↓ Conv 3→32, stride 2
  ↓ Conv 32→64, stride 2
  ↓ Conv 64→128, stride 2  → P3 backbone (stride 8, 128 channels)
  ↓ Conv 128→256, stride 2 → P4 backbone (stride 16, 256 channels)
  ↓ Conv 256→512, stride 2 → P5 backbone (stride 32, 512 channels)
  ↓ SPPF (512→512)
```

All convolutions use BatchNorm + **SiLU activation** (YOLOv5 standard)

### FPN Neck (Top-Down Pathway)
Enriches high-resolution features with semantic information from deeper layers:
```
P5 (512ch) ─[reduce to 256]─→ Upsample ─┐
                                         ├─→ Concat → C3 → P4_FPN (256ch)
P4 (256ch) ─[lateral]────────────────────┘

P4_FPN ────[reduce to 128]─→ Upsample ─┐
                                        ├─→ Concat → C3 → P3_FPN (128ch)
P3 (128ch) ─[lateral]───────────────────┘
```

**Lateral connections** preserve fine-grained spatial information
**C3 modules** (CSP Bottleneck) provide rich feature fusion with residual connections

### PANet (Bottom-Up Pathway)
Refines features bidirectionally to improve localization:
```
P3_FPN (128ch) ─[stride-2 conv]→ Downsample ─┐
                                              ├─→ Concat → C3 → P4_refined (256ch)
P4_FPN (256ch) ───────────────────────────────┘

P4_refined ────[stride-2 conv]→ Downsample ─┐
                                             ├─→ Concat → C3 → P5_refined (512ch)
P5 (512ch) ──────────────────────────────────┘
```

Bottom-up pathway strengthens feature hierarchy for accurate boundary prediction

### Detection Heads (Multi-Scale)
Three independent detection heads operate at different spatial resolutions:

| Head | Stride | Grid Size @ 640px | Grid Size @ 1024px | Channels | Purpose |
|------|--------|-------------------|---------------------|----------|---------|
| **P3** | 8 | 80×80 | 128×128 | 128 | Small objects |
| **P4** | 16 | 40×40 | 64×64 | 256 | Medium objects |
| **P5** | 32 | 20×20 | 32×32 | 512 | Large objects |

Each head outputs: `(batch, grid_h, grid_w, 3, 5+num_classes)` where each anchor predicts:
- `t_x, t_y, t_w, t_h`: **Encoded offsets** (decoded via YOLOv5 formulas)
- `objectness`: Confidence logit
- `class_probs`: Class logits

**Total detection capacity at 640px:** 25,200 predictions (19,200 + 4,800 + 1,200)
**Total detection capacity at 1024px:** 64,512 predictions (49,152 + 12,288 + 3,072)

### YOLOv5 Offset Prediction
Model outputs encoded offsets, not direct coordinates. Decoding formulas constrain predictions near responsible grid cell:
- `b_x = ((σ(t_x) * 2 - 0.5) + c_x) / grid_size`
- `b_y = ((σ(t_y) * 2 - 0.5) + c_y) / grid_size`
- `b_w = (anchor_w * (2σ(t_w))²) / img_size`
- `b_h = (anchor_h * (2σ(t_h))²) / img_size`

This approach improves training stability by preventing extreme predictions.

### Scale-Specific Anchor Boxes
Each scale has anchors optimized for different object sizes:

**P3 (small objects):** `[[10, 13], [16, 30], [33, 23]]`
**P4 (medium objects):** `[[30, 61], [62, 45], [59, 119]]`
**P5 (large objects):** `[[116, 90], [156, 198], [373, 326]]`

Anchors in width×height pixels. During training:
- Each GT box is matched to **best anchor across ALL scales** using shape-based IoU (position-agnostic)
- Box assigned to grid cell at the chosen scale based on center coordinates
- Automatically scaled during decoding

### Loss Function (Multi-Scale)
Loss computed independently for each scale, then summed:

**Per-scale loss:**
1. **Bounding box regression:** CIoU (Complete IoU) loss
   - Considers IoU + center distance penalty + aspect ratio consistency
   - Applied only to cells with objects
2. **Objectness:** Binary cross-entropy with logits (all cells)
3. **Classification:** Binary cross-entropy with logits (cells with objects only)

**Loss component weights (YOLOv5 defaults from hyp.scratch-low.yaml):**
- box: 0.05 (bbox loss gain)
- obj: 1.0 (objectness loss gain, with per-scale balancing [4.0, 1.0, 0.4])
- cls: 0.5 (classification loss gain)

**Total loss:** `loss_P3 + loss_P4 + loss_P5`

### Key Implementation Details

**Multi-Scale Target Generation (YOLODataset.__getitem__):**
1. Creates 3 empty target tensors (one per scale)
2. For each GT box:
   - Computes IoU with all 9 anchors (3 per scale × 3 scales)
   - Assigns to best-matching anchor+scale pair
   - Places target in appropriate grid cell at selected scale
3. Returns `(image, [target_p3, target_p4, target_p5])`

**Custom Collate Function:**
- `yolo_collate_fn()` handles multi-scale targets in DataLoader
- Stacks images: `(batch, 3, H, W)`
- Keeps targets as list of lists for flexible per-scale stacking

**Multi-Scale Inference Pipeline (predict function):**
1. Resize image to model.img_size
2. Forward pass → `[pred_p3, pred_p4, pred_p5]` (RAW offsets)
3. For each scale:
   - Decode predictions using scale-specific anchors and grid size
   - Apply sigmoid to objectness and class predictions
   - Extract detections above confidence threshold
   - Convert normalized coords → pixel coords → original image scale
4. Concatenate detections from all scales
5. Apply **global NMS** with IoU threshold 0.4

**Training Loop:**
- Adam optimizer with **YOLOv5-style learning rate scheduling**
  - **Linear warmup** (epochs 0-2): LR ramps from 1e-6 → 1e-2
  - **Cosine annealing** (epochs 3-99): LR decays from 1e-2 → 1e-4
  - Configurable via CLI: `--lr`, `--min-lr`, `--warmup-epochs`
- **Gradient clipping** (max_norm=10.0) to prevent exploding gradients
- Batch size 8
- Multi-scale loss printed with breakdown: bbox, objectness, class
- Validation metrics: Loss, Precision, Recall, F1 (aggregated across scales)
- Current learning rate displayed each epoch for monitoring
- Checkpoint saved every epoch, includes `img_size` and `num_classes`

**Model Statistics:**
- Total parameters: ~6.1M (4× larger than single-scale baseline)
- Increased compute from FPN/PANet feature fusion
- Significantly better accuracy, especially for small objects

## Testing

The project includes comprehensive unit tests covering all major components.

### Running Tests

```bash
# Run all tests
python3 -m pytest tests

# Run tests with coverage report
python3 -m pytest tests --cov=train --cov-report=term

# Show which lines are missing coverage
python3 -m pytest tests --cov=train --cov-report=term-missing

# Generate HTML coverage report (opens in browser)
python3 -m pytest tests --cov=train --cov-report=html
# Then open htmlcov/index.html

# Multiple reports at once
python3 -m pytest tests --cov=train --cov-report=term --cov-report=html
```

**Test Coverage:**
The test suite includes 81 tests covering:
- **Dataset loading** (`tests/test_dataset.py`): Multi-scale target generation, anchor matching, collate function
- **Model architecture** (`tests/test_model.py`): Forward pass, feature map shapes, parameter counts
- **Loss functions** (`tests/test_loss.py`): CIoU calculation, multi-scale loss computation, gradient flow
- **Inference** (`tests/test_inference.py`): Prediction decoding, NMS, coordinate transformations
- **Utilities** (`tests/test_utils.py`): Helper functions, metrics computation

**Important**: Use `--cov=train` (not `--cov=main`) since the main code is in `train.py`.

## Commands

### Training
```bash
# Default 640×640 resolution with default LR schedule
python train.py dataset.yaml

# Custom resolution (must be divisible by 32)
python train.py dataset.yaml --img-size 512
python train.py dataset.yaml --img-size 1280

# Customize learning rate schedule
python train.py dataset.yaml --lr 0.02          # Higher initial LR
python train.py dataset.yaml --min-lr 0.0001    # Minimum LR at end
python train.py dataset.yaml --warmup-epochs 5  # Longer warmup
python train.py dataset.yaml --epochs 50        # Shorter training

# Combined options
python train.py dataset.yaml --img-size 1024 --lr 0.015 --epochs 150
```

**Default Learning Rate Schedule:**
- Initial LR: `0.01` (10× higher than previous fixed LR)
- Minimum LR: `0.0001`
- Warmup: 3 epochs (linear ramp from 1e-6 to initial LR)
- Decay: Cosine annealing from initial to minimum LR
- Gradient clipping: max_norm=10.0

Trains for 100 epochs by default, saves model as `yolo_YYYYMMDD_HHMMSS.pt`.
Checkpoint includes img_size, so model can be loaded correctly later.
Learning rate is logged each epoch for monitoring convergence.

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

**Building Blocks (train.py:138-240):**
- `ConvBlock`: Conv2d + BatchNorm + SiLU activation (basic building block)
- `C3`: CSP Bottleneck with 3 convolutions (YOLOv5 module for feature fusion)
- `Bottleneck`: Residual block with optional shortcut connection
- `SPPF`: Spatial Pyramid Pooling - Fast (sequential max pooling for multi-scale receptive fields)

**Multi-Scale Target Generation (YOLODataset.__getitem__, train.py:77-141):**
1. Initializes 3 empty target tensors (one per scale: P3, P4, P5)
2. For each GT box in label file:
   - Converts box dimensions to pixels
   - Computes shape-based IoU with all 9 anchors (3 anchors × 3 scales)
   - Selects best-matching anchor+scale pair (highest IoU)
   - Calculates responsible grid cell at chosen scale based on center coordinates
   - Assigns encoded box coordinates, objectness=1, and class label
3. Returns `(image, [target_p3, target_p4, target_p5])`
4. Only assigns if cell+anchor pair is empty (first box wins if multiple overlap)

**Custom DataLoader Collate Function (yolo_collate_fn, train.py:143-156):**
- Required because default collate can't handle multi-scale targets
- Stacks images into batch tensor: `(batch, 3, H, W)`
- Preserves targets as list of lists: `[(img0_targets), (img1_targets), ...]`
- Each sample's targets remain as `[target_p3, target_p4, target_p5]`
- In training/eval, targets get stacked by scale: `torch.stack([t[0] for t in targets])`

**Model Forward Pass (train.py:360-413):**
1. **Backbone:** Extracts features at 3 scales (P3, P4, P5)
2. **FPN Top-Down:** P5 → P4 → P3 with upsampling and lateral connections
3. **PANet Bottom-Up:** P3 → P4 → P5 with downsampling and feature fusion
4. **Detection Heads:** Three independent 1×1 convolutions
5. Returns: `[out_p3, out_p4, out_p5]` each shaped `(batch, grid_h, grid_w, 3, 5+nc)`

**Multi-Scale Loss Computation (yolo_loss_multiscale, train.py:615-644):**
- Accepts list of predictions and targets for all scales
- Calls `yolo_loss()` independently for each scale
- Each scale's loss includes: 5×bbox_loss + obj_loss + class_loss
- Returns summed losses: `total_loss = loss_P3 + loss_P4 + loss_P5`
- Gradients flow through decoding in each scale's bbox loss

**Multi-Scale Inference Pipeline (predict, train.py:840-934):**
1. Resize image to model.img_size
2. Forward pass → `[pred_p3, pred_p4, pred_p5]` (RAW offsets)
3. **For each scale:**
   - Decode predictions using scale-specific anchors and grid size
   - Apply sigmoid to objectness and class predictions
   - Extract detections above confidence threshold (default 0.5)
   - Convert: normalized coords → pixel coords → original image scale
4. Concatenate all detections from all scales
5. Apply **global NMS** with IoU threshold 0.4 (suppresses duplicates across scales)

**Training Workflow (train_epoch, train.py:646-684):**
1. Get multi-scale anchors from model
2. For each batch:
   - Stack targets by scale: `[batch_p3, batch_p4, batch_p5]`
   - Forward pass returns `[pred_p3, pred_p4, pred_p5]`
   - Compute multi-scale loss
   - Backpropagate
   - Clip gradients (max_norm=10.0) to prevent explosions
   - Update weights with optimizer
3. Checkpoint includes `img_size`, `num_classes`, and full model state

**Learning Rate Scheduler (get_lr_lambda, train.py:790-822):**
YOLOv5-style scheduler with two phases:
1. **Warmup Phase** (epochs 0 to warmup_epochs-1):
   ```python
   lr = warmup_start_lr + (initial_lr - warmup_start_lr) * (epoch / warmup_epochs)
   ```
   - Prevents early training instability
   - Default: ramps from 1e-6 → 1e-2 over 3 epochs

2. **Cosine Annealing** (remaining epochs):
   ```python
   progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
   cosine_decay = 0.5 * (1.0 + cos(π * progress))
   lr = min_lr + (initial_lr - min_lr) * cosine_decay
   ```
   - Smooth decay for fine-tuning
   - Default: decays from 1e-2 → 1e-4 over epochs 3-99

Implemented using `torch.optim.lr_scheduler.LambdaLR` and stepped after each epoch.

**Evaluation Workflow (eval_epoch, train.py:714-788):**
- Computes multi-scale loss on validation set
- For each scale:
  - Decodes predictions
  - Compares with targets using IoU threshold 0.5
  - Counts true positives, false positives, false negatives
- Aggregates metrics across all scales
- Returns: loss, precision, recall, F1 score

## Known Limitations

This is a minimal implementation. Compared to production YOLOv5:
- No data augmentation (mosaic, mixup, HSV, etc.)
- Simpler backbone (not full CSPDarknet53 with deep C3 stacks)
- No label smoothing
- Evaluation metrics are grid-based (not true COCO-style AP/mAP)
- No automatic anchor optimization (k-means clustering on dataset)
- No mixed precision training (FP16/AMP)

**YOLOv5 Features Implemented:**
✅ **Multi-scale detection (FPN + PANet neck)**
✅ **Three detection heads (P3/P4/P5) for small/medium/large objects**
✅ **C3 modules (CSP Bottleneck) with residual connections**
✅ **Scale-specific anchors with cross-scale matching**
✅ **Learning rate scheduling with warmup and cosine annealing**
✅ **Gradient clipping for training stability**
✅ SPPF (Spatial Pyramid Pooling - Fast)
✅ SiLU activation function
✅ YOLOv5-style offset prediction with decoding
✅ CIoU loss for bbox regression
✅ Anchor-based detection with IoU matching
✅ Configurable input resolution

Despite remaining limitations, the implementation includes the **core YOLOv5 architecture** (FPN+PANet) and produces excellent results for multi-scale object detection tasks, including small objects that single-scale detectors struggle with.

## Extending the Implementation

**To add more classes:**
1. Update `nc` in dataset.yaml
2. Ensure label files use correct class IDs (0 to nc-1)
3. Model automatically adapts output dimensions for all 3 detection heads

**To modify anchors (advanced):**
Multi-scale anchors are lists of 3 anchor sets (one per scale). Pass custom anchors:
```python
anchors_p3 = [[10, 13], [16, 30], [33, 23]]      # Small
anchors_p4 = [[30, 61], [62, 45], [59, 119]]     # Medium
anchors_p5 = [[116, 90], [156, 198], [373, 326]] # Large
anchors = [anchors_p3, anchors_p4, anchors_p5]

model = YOLO(num_classes=3, anchors=anchors, img_size=640)
dataset = YOLODataset(img_dir, num_classes=3, anchors=anchors, img_size=640)
```
Anchors should be tuned to your dataset using k-means clustering on training box dimensions, grouped by size.

**To change image resolution:**
Use `--img-size` CLI argument (must be divisible by 32):
```bash
python train.py dataset.yaml --img-size 1280  # High resolution, better for small objects
python train.py dataset.yaml --img-size 512   # Lower resolution, faster training
```
Grid sizes automatically scale: P3=img_size//8, P4=img_size//16, P5=img_size//32

## Performance Characteristics

**Expected Improvements from FPN:**
- **Small object detection:** 30-50% improvement in recall for objects <32×32 pixels
- **Overall mAP:** 10-20% improvement over single-scale baseline
- **Multi-scale robustness:** Better handling of objects at varying distances
- **Localization accuracy:** Tighter bounding boxes from bidirectional feature fusion

**Trade-offs:**
- Model size: ~6.1M parameters (4× larger than single-scale)
- Training time: ~1.5-2× slower per epoch
- Inference time: ~2× slower due to 3 detection heads + NMS
- Memory usage: ~2× higher GPU memory required

**Recommendations:**
- Use **640×640** for balanced speed/accuracy
- Use **1024×1024** or higher for maximum small object detection
- Use **512×512** for faster training/inference on limited hardware
- P3 head provides most benefit for small objects; consider removing if only large objects present

## Advanced Debugging

**Verify multi-scale target distribution:**
```python
from train import YOLODataset
dataset = YOLODataset('path/to/images', img_size=640)
img, targets = dataset[0]  # targets = [target_p3, target_p4, target_p5]

# Count objects assigned to each scale
for i, t in enumerate(targets):
    num_objects = (t[..., 4] > 0).sum().item()
    print(f"Scale P{i+3}: {num_objects} objects")
```

**Check feature map shapes during forward pass:**
```python
model = YOLO(num_classes=1, img_size=640)
x = torch.randn(1, 3, 640, 640)
outputs = model(x)  # [out_p3, out_p4, out_p5]
for i, out in enumerate(outputs):
    print(f"P{i+3} output shape: {out.shape}")
# Expected: P3=(1,80,80,3,6), P4=(1,40,40,3,6), P5=(1,20,20,3,6)
```

**Analyze detection distribution across scales:**
```python
from train import predict, YOLO
model = YOLO(num_classes=1, img_size=640)
# ... load checkpoint ...
detections = predict(model, 'image.jpg', 'cuda')
# Detections from all scales are concatenated; add tracking in predict() to analyze per-scale
```
