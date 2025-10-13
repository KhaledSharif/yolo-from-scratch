import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import yaml
import sys
from tqdm import tqdm
from PIL import Image
import numpy as np
import glob
from datetime import datetime


def letterbox_resize(image, target_size=640, pad_color=(114, 114, 114)):
    """
    Resize image with aspect ratio preservation (letterbox).

    Instead of distorting the image, this function:
    1. Scales the image to fit within target_size while preserving aspect ratio
    2. Pads the remaining space with pad_color to create a square image

    This prevents geometry distortion that breaks object detection accuracy.

    Args:
        image: PIL Image object
        target_size: Target size for both width and height (creates square output)
        pad_color: RGB tuple for padding color (default: gray 114,114,114)

    Returns:
        resized_image: PIL Image of size (target_size, target_size)
        scale: Scale factor applied (for coordinate adjustment)
        pad_top: Top padding in pixels
        pad_left: Left padding in pixels
    """
    orig_w, orig_h = image.size

    # Calculate scale to fit image in target_size while preserving aspect ratio
    scale = min(target_size / orig_w, target_size / orig_h)

    # New dimensions after scaling
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Resize image (using high-quality resampling)
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else 2)

    # Create new image with padding
    new_image = Image.new('RGB', (target_size, target_size), pad_color)

    # Calculate padding to center the image
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2

    # Paste resized image onto padded canvas
    new_image.paste(resized, (pad_left, pad_top))

    return new_image, scale, pad_top, pad_left

class YOLODataset(Dataset):
    def __init__(self, img_dir, num_classes=1, anchors=None, img_size=640):
        self.imgs = sorted(glob.glob(f"{img_dir}/*.jpg") + glob.glob(f"{img_dir}/*.png"))
        self.labels = [p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt' for p in self.imgs]
        self.num_classes = num_classes
        self.img_size = img_size

        # Multi-scale grid sizes
        self.grid_size_p3 = img_size // 8   # Stride 8
        self.grid_size_p4 = img_size // 16  # Stride 16
        self.grid_size_p5 = img_size // 32  # Stride 32
        self.grid_sizes = [self.grid_size_p3, self.grid_size_p4, self.grid_size_p5]
        self.strides = [8, 16, 32]

        # Multi-scale anchors: [P3, P4, P5]
        if anchors is None:
            anchors_p3 = [[10, 13], [16, 30], [33, 23]]      # Small objects
            anchors_p4 = [[30, 61], [62, 45], [59, 119]]     # Medium objects
            anchors_p5 = [[116, 90], [156, 198], [373, 326]] # Large objects
            self.anchors = [
                torch.tensor(anchors_p3, dtype=torch.float32),
                torch.tensor(anchors_p4, dtype=torch.float32),
                torch.tensor(anchors_p5, dtype=torch.float32)
            ]
        else:
            # If custom anchors provided, expect list of 3 anchor sets
            if isinstance(anchors[0][0], list):
                self.anchors = [torch.tensor(a, dtype=torch.float32) for a in anchors]
            else:
                # Single anchor set provided (backward compatibility)
                self.anchors = [torch.tensor(anchors, dtype=torch.float32)] * 3

        self.num_anchors_per_scale = 3  # 3 anchors per scale
        self.output_dim = 5 + num_classes  # x, y, w, h, objectness + class probs

    def __len__(self):
        return len(self.imgs)

    def compute_anchor_iou(self, box_wh, anchors):
        """
        Compute IoU between a box and all anchors (without considering position).

        Args:
            box_wh: (width, height) in pixels
            anchors: (num_anchors, 2) tensor of anchor dimensions

        Returns:
            iou: (num_anchors,) tensor of IoU values
        """
        box_area = box_wh[0] * box_wh[1]
        anchor_area = anchors[:, 0] * anchors[:, 1]

        # Intersection (assuming both centered at origin)
        inter_w = torch.min(box_wh[0], anchors[:, 0])
        inter_h = torch.min(box_wh[1], anchors[:, 1])
        inter_area = inter_w * inter_h

        # Union
        union_area = box_area + anchor_area - inter_area

        iou = inter_area / (union_area + 1e-16)
        return iou

    def __getitem__(self, idx):
        # Load image and apply letterbox resize (preserves aspect ratio)
        pil_img = Image.open(self.imgs[idx]).convert('RGB')
        orig_w, orig_h = pil_img.size
        pil_img, scale, pad_top, pad_left = letterbox_resize(pil_img, self.img_size)
        img = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0

        # Initialize empty targets for all three scales
        targets = [
            torch.zeros((self.grid_sizes[0], self.grid_sizes[0], self.num_anchors_per_scale, self.output_dim)),  # P3
            torch.zeros((self.grid_sizes[1], self.grid_sizes[1], self.num_anchors_per_scale, self.output_dim)),  # P4
            torch.zeros((self.grid_sizes[2], self.grid_sizes[2], self.num_anchors_per_scale, self.output_dim))   # P5
        ]

        # Load all boxes from label file
        if Path(self.labels[idx]).exists():
            with open(self.labels[idx], encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(float(parts[0]))
                        x_center, y_center, width, height = [float(x) for x in parts[1:]]

                        # Adjust coordinates for letterbox resize
                        # Convert from normalized [0,1] relative to original image
                        # to normalized [0,1] relative to padded/scaled image
                        x_center = (x_center * orig_w * scale + pad_left) / self.img_size
                        y_center = (y_center * orig_h * scale + pad_top) / self.img_size
                        width = (width * orig_w * scale) / self.img_size
                        height = (height * orig_h * scale) / self.img_size

                        # Convert box dimensions to pixels (relative to img_size after letterbox)
                        box_w_px = width * self.img_size
                        box_h_px = height * self.img_size
                        box_wh = torch.tensor([box_w_px, box_h_px])

                        # Find best matching anchor across ALL scales
                        best_iou = -1
                        best_scale_idx = 0
                        best_anchor_idx = 0

                        for scale_idx in range(3):
                            ious = self.compute_anchor_iou(box_wh, self.anchors[scale_idx])
                            max_iou = ious.max().item()
                            if max_iou > best_iou:
                                best_iou = max_iou
                                best_scale_idx = scale_idx
                                best_anchor_idx = ious.argmax().item()

                        # Determine which grid cell is responsible at the selected scale
                        grid_size = self.grid_sizes[best_scale_idx]
                        grid_x = int(x_center * grid_size)
                        grid_y = int(y_center * grid_size)

                        # Clamp to valid range
                        grid_x = min(grid_x, grid_size - 1)
                        grid_y = min(grid_y, grid_size - 1)

                        # Only assign if this cell+anchor doesn't already have an object
                        target = targets[best_scale_idx]
                        if target[grid_y, grid_x, best_anchor_idx, 4] == 0:
                            # Set bounding box coordinates (normalized to [0,1])
                            target[grid_y, grid_x, best_anchor_idx, 0:4] = torch.tensor(
                                [x_center, y_center, width, height]
                            )
                            # Set objectness
                            target[grid_y, grid_x, best_anchor_idx, 4] = 1.0
                            # Set class probability
                            if self.num_classes == 1:
                                target[grid_y, grid_x, best_anchor_idx, 5] = 1.0
                            else:
                                # For multi-class, use one-hot encoding
                                target[grid_y, grid_x, best_anchor_idx, 5 + class_id] = 1.0

        return img, targets

def yolo_collate_fn(batch):
    """
    Custom collate function for DataLoader to handle multi-scale targets.

    Args:
        batch: List of (image, targets) tuples where targets is [target_p3, target_p4, target_p5]

    Returns:
        images: Batched images tensor (B, C, H, W)
        targets: List of target lists (one per sample in batch)
    """
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]  # Keep as list of lists
    return images, targets

class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer.

    YOLOv5's efficient implementation of SPP that applies sequential max pooling
    instead of parallel pooling, which is 2× faster while producing the same output.

    Structure: Conv → MaxPool → MaxPool → MaxPool → Concat all → Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, 1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act = nn.SiLU()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(hidden_channels * 4, out_channels, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        # Apply maxpool 3 times sequentially and concatenate with original
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        # Concatenate along channel dimension: [x, y1, y2, y3]
        out = torch.cat([x, y1, y2, y3], dim=1)
        return self.act(self.bn2(self.conv2(out)))

class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv2d + BatchNorm + SiLU activation.
    Used as building block throughout the FPN architecture.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class C3(nn.Module):
    """
    CSP Bottleneck with 3 convolutions (YOLOv5 C3 module).
    Provides richer feature extraction with residual connections.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        n: Number of bottleneck blocks (default 1)
        shortcut: Whether to use shortcut connection (default True)
    """
    def __init__(self, in_channels, out_channels, n=1, shortcut=True):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1, 1, 0)
        self.conv2 = ConvBlock(in_channels, hidden_channels, 1, 1, 0)
        self.conv3 = ConvBlock(2 * hidden_channels, out_channels, 1, 1, 0)
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(hidden_channels, hidden_channels, shortcut) for _ in range(n)]
        )

    def forward(self, x):
        # Split into two paths
        x1 = self.bottlenecks(self.conv1(x))
        x2 = self.conv2(x)
        # Concatenate and project
        return self.conv3(torch.cat([x1, x2], dim=1))

class Bottleneck(nn.Module):
    """
    Standard bottleneck block used in C3.
    """
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 3, 1, 1)
        self.conv2 = ConvBlock(out_channels, out_channels, 3, 1, 1)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.shortcut else self.conv2(self.conv1(x))

class YOLO(nn.Module):
    """
    YOLO object detection model with YOLOv5-style FPN multi-scale detection.

    The model outputs ENCODED predictions (t_x, t_y, t_w, t_h) at three scales
    that must be decoded to absolute coordinates. See decode_predictions() function.

    Architecture:
    - Backbone: 5 stride-2 conv layers extracting features at 3 scales
    - FPN Neck: Top-down pathway with lateral connections + PANet bottom-up
    - 3 Detection Heads: P3 (stride 8), P4 (stride 16), P5 (stride 32)
    - Each head outputs: num_anchors * (5 + num_classes) channels per grid cell
    - Each anchor predicts: t_x, t_y, t_w, t_h (offsets), objectness, class_probs

    Multi-scale detection:
    - P3 (stride 8): 80×80 grid @ 640px - small objects
    - P4 (stride 16): 40×40 grid @ 640px - medium objects
    - P5 (stride 32): 20×20 grid @ 640px - large objects

    Decoding formulas (applied in decode_predictions):
    - b_x = ((σ(t_x) * 2 - 0.5) + c_x) / grid_size
    - b_y = ((σ(t_y) * 2 - 0.5) + c_y) / grid_size
    - b_w = (anchor_w * (2 * σ(t_w))²) / img_size
    - b_h = (anchor_h * (2 * σ(t_h))²) / img_size

    This approach constrains predictions to be near their responsible grid cell
    and scales dimensions relative to anchor size, improving training stability.
    """
    def __init__(self, num_classes=1, anchors=None, img_size=640,
                 width_mult=0.50, depth_mult=0.33):  # Default to 's' model
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.width_mult = width_mult  # Store for checkpoint
        self.depth_mult = depth_mult  # Store for checkpoint

        # --- Helper functions for scaling ---
        def make_divisible(x, divisor=8):
            """Ensures channel counts are multiples of 8 for GPU efficiency."""
            return int(np.ceil(x * width_mult / divisor) * divisor)

        def make_repeats(n):
            """Calculates number of C3 block repeats based on depth_mult."""
            return max(round(n * depth_mult), 1) if n > 1 else n

        # --- Scalable Channel Counts ---
        c_stem = make_divisible(64)   # Stem output: 64 base
        c3_p3 = make_divisible(128)   # P3: 128 base
        c3_p4 = make_divisible(256)   # P4: 256 base
        c3_p5 = make_divisible(512)   # P5: 512 base

        # Grid sizes for three scales
        self.grid_size_p3 = img_size // 8   # Stride 8 (e.g., 640 -> 80)
        self.grid_size_p4 = img_size // 16  # Stride 16 (e.g., 640 -> 40)
        self.grid_size_p5 = img_size // 32  # Stride 32 (e.g., 640 -> 20)
        # For backward compatibility
        self.grid_size = self.grid_size_p5

        # Multi-scale anchors: small, medium, large
        if anchors is None:
            anchors_p3 = [[10, 13], [16, 30], [33, 23]]      # Small objects
            anchors_p4 = [[30, 61], [62, 45], [59, 119]]     # Medium objects
            anchors_p5 = [[116, 90], [156, 198], [373, 326]] # Large objects
            self.anchors = [
                torch.tensor(anchors_p3, dtype=torch.float32),
                torch.tensor(anchors_p4, dtype=torch.float32),
                torch.tensor(anchors_p5, dtype=torch.float32)
            ]
        else:
            # If custom anchors provided, expect list of 3 anchor sets
            if isinstance(anchors[0][0], list):
                self.anchors = [torch.tensor(a, dtype=torch.float32) for a in anchors]
            else:
                # Single anchor set provided (backward compatibility)
                self.anchors = [torch.tensor(anchors, dtype=torch.float32)] * 3

        self.num_anchors = 3  # 3 anchors per scale

        # Output channels per detection head: num_anchors * (5 + num_classes)
        self.output_channels = self.num_anchors * (5 + num_classes)

        # ===== BACKBONE (Scalable CSPDarknet) =====
        # Stem: 3 -> 32 -> 64 (with scaling)
        self.stem = nn.Sequential(
            nn.Conv2d(3, c_stem // 2, 3, 2, 1), nn.BatchNorm2d(c_stem // 2), nn.SiLU(),
            nn.Conv2d(c_stem // 2, c_stem, 3, 2, 1), nn.BatchNorm2d(c_stem), nn.SiLU()
        )
        # P3 backbone: stride 4 -> stride 8 (128 channels base)
        self.backbone_p3 = nn.Sequential(
            C3(c_stem, c_stem, n=make_repeats(1)),
            nn.Conv2d(c_stem, c3_p3, 3, 2, 1), nn.BatchNorm2d(c3_p3), nn.SiLU(),
            C3(c3_p3, c3_p3, n=make_repeats(2))
        )
        # P4 backbone: stride 8 -> stride 16 (256 channels base)
        self.backbone_p4 = nn.Sequential(
            nn.Conv2d(c3_p3, c3_p4, 3, 2, 1), nn.BatchNorm2d(c3_p4), nn.SiLU(),
            C3(c3_p4, c3_p4, n=make_repeats(2))
        )
        # P5 backbone: stride 16 -> stride 32 (512 channels base)
        self.backbone_p5 = nn.Sequential(
            nn.Conv2d(c3_p4, c3_p5, 3, 2, 1), nn.BatchNorm2d(c3_p5), nn.SiLU(),
            C3(c3_p5, c3_p5, n=make_repeats(1))
        )
        # SPPF on P5
        self.sppf = SPPF(c3_p5, c3_p5)

        # ===== FPN NECK (Top-Down) =====
        # Lateral connections
        self.lateral_p4 = ConvBlock(c3_p4, c3_p4, 1, 1, 0)
        self.lateral_p3 = ConvBlock(c3_p3, c3_p3, 1, 1, 0)

        # Top-down pathway
        # P5 -> P4: Upsample + merge
        self.upsample_p5_to_p4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce_p5_for_p4 = ConvBlock(c3_p5, c3_p4, 1, 1, 0)
        self.merge_p4 = C3(c3_p4 * 2, c3_p4, n=make_repeats(1))

        # P4 -> P3: Upsample + merge
        self.upsample_p4_to_p3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce_p4_for_p3 = ConvBlock(c3_p4, c3_p3, 1, 1, 0)
        self.merge_p3 = C3(c3_p3 * 2, c3_p3, n=make_repeats(1))

        # ===== PANet (Bottom-Up) =====
        # P3 -> P4: Downsample + merge
        self.downsample_p3_to_p4 = ConvBlock(c3_p3, c3_p3, 3, 2, 1)
        self.panet_merge_p4 = C3(c3_p3 + c3_p4, c3_p4, n=make_repeats(1))

        # P4 -> P5: Downsample + merge
        self.downsample_p4_to_p5 = ConvBlock(c3_p4, c3_p4, 3, 2, 1)
        self.panet_merge_p5 = C3(c3_p4 + c3_p5, c3_p5, n=make_repeats(1))

        # ===== DETECTION HEADS =====
        # Stronger heads with 2 Conv blocks before final prediction (YOLOv5 style)
        self.head_p3 = nn.Sequential(
            ConvBlock(c3_p3, c3_p3, 3, 1, 1),
            ConvBlock(c3_p3, c3_p3, 3, 1, 1),
            nn.Conv2d(c3_p3, self.output_channels, 1)
        )
        self.head_p4 = nn.Sequential(
            ConvBlock(c3_p4, c3_p4, 3, 1, 1),
            ConvBlock(c3_p4, c3_p4, 3, 1, 1),
            nn.Conv2d(c3_p4, self.output_channels, 1)
        )
        self.head_p5 = nn.Sequential(
            ConvBlock(c3_p5, c3_p5, 3, 1, 1),
            ConvBlock(c3_p5, c3_p5, 3, 1, 1),
            nn.Conv2d(c3_p5, self.output_channels, 1)
        )

        # Initialize detection head biases for training stability
        self.initialize_detection_biases()

    def initialize_detection_biases(self, prior=0.01):
        """
        Initialize detection head biases for stable training.

        YOLOv5-style bias initialization:
        - Objectness bias: Set to -log((1-prior)/prior) to start with low objectness predictions
        - Class bias: Set to 0.0 (or can be based on class frequencies)

        This prevents the model from predicting too many false positives early in training,
        leading to faster convergence and better final accuracy.

        Args:
            prior: Initial objectness probability (default 0.01 = 1%)
        """
        import math

        # Calculate objectness bias: -log((1-prior)/prior)
        # For prior=0.01: -log(99) ≈ -4.595
        # This makes sigmoid(bias) ≈ prior
        obj_bias = -math.log((1 - prior) / prior)

        # Initialize biases for all three detection heads
        for head in [self.head_p3, self.head_p4, self.head_p5]:
            # Each head is now an nn.Sequential; get the final Conv2d layer
            final_conv = head[-1]  # Last layer in Sequential

            # Each head outputs: num_anchors * (5 + num_classes) channels
            # Layout per anchor: [tx, ty, tw, th, objectness, class1, class2, ...]
            output_per_anchor = 5 + self.num_classes

            with torch.no_grad():
                bias = final_conv.bias.view(self.num_anchors, output_per_anchor)

                # Initialize objectness bias (index 4) for each anchor
                bias[:, 4].fill_(obj_bias)

                # Initialize class biases (indices 5:) to 0.0
                if self.num_classes > 0:
                    bias[:, 5:].fill_(0.0)

                # tx, ty, tw, th biases can stay at default initialization (near 0)
                # This is fine as the decoding formulas have built-in offsets

    def forward(self, x):
        batch_size = x.shape[0]

        # ===== BACKBONE: Extract multi-scale features =====
        x_stem = self.stem(x)                       # (B, c_stem, H/4, W/4)
        p3_backbone = self.backbone_p3(x_stem)      # (B, c3_p3, H/8, W/8) - stride 8
        p4_backbone = self.backbone_p4(p3_backbone) # (B, c3_p4, H/16, W/16) - stride 16
        p5_backbone = self.backbone_p5(p4_backbone) # (B, c3_p5, H/32, W/32) - stride 32
        p5_backbone = self.sppf(p5_backbone)        # (B, c3_p5, H/32, W/32) with SPPF

        # ===== FPN: Top-down pathway =====
        # Lateral connections
        p4_lateral = self.lateral_p4(p4_backbone)  # (B, 256, H/16, W/16)
        p3_lateral = self.lateral_p3(p3_backbone)  # (B, 128, H/8, W/8)

        # P5 -> P4
        p5_up = self.upsample_p5_to_p4(self.reduce_p5_for_p4(p5_backbone))  # (B, 256, H/16, W/16)
        p4_fpn = self.merge_p4(torch.cat([p5_up, p4_lateral], dim=1))  # (B, 256, H/16, W/16)

        # P4 -> P3
        p4_up = self.upsample_p4_to_p3(self.reduce_p4_for_p3(p4_fpn))  # (B, 128, H/8, W/8)
        p3_fpn = self.merge_p3(torch.cat([p4_up, p3_lateral], dim=1))  # (B, 128, H/8, W/8)

        # ===== PANet: Bottom-up pathway =====
        # P3 -> P4
        p3_down = self.downsample_p3_to_p4(p3_fpn)  # (B, 128, H/16, W/16)
        p4_panet = self.panet_merge_p4(torch.cat([p3_down, p4_fpn], dim=1))  # (B, 256, H/16, W/16)

        # P4 -> P5
        p4_down = self.downsample_p4_to_p5(p4_panet)  # (B, 256, H/32, W/32)
        p5_panet = self.panet_merge_p5(torch.cat([p4_down, p5_backbone], dim=1))  # (B, 512, H/32, W/32)

        # ===== DETECTION HEADS =====
        # P3 head (stride 8)
        out_p3 = self.head_p3(p3_fpn)  # (B, num_anchors*(5+nc), H/8, W/8)
        # Derive grid size dynamically from tensor shape for robustness
        _, _, h_p3, w_p3 = out_p3.shape
        out_p3 = out_p3.view(batch_size, self.num_anchors, 5 + self.num_classes, h_p3, w_p3)
        out_p3 = out_p3.permute(0, 3, 4, 1, 2).contiguous()  # (B, H/8, W/8, num_anchors, 5+nc)

        # P4 head (stride 16)
        out_p4 = self.head_p4(p4_panet)  # (B, num_anchors*(5+nc), H/16, W/16)
        # Derive grid size dynamically from tensor shape
        _, _, h_p4, w_p4 = out_p4.shape
        out_p4 = out_p4.view(batch_size, self.num_anchors, 5 + self.num_classes, h_p4, w_p4)
        out_p4 = out_p4.permute(0, 3, 4, 1, 2).contiguous()  # (B, H/16, W/16, num_anchors, 5+nc)

        # P5 head (stride 32)
        out_p5 = self.head_p5(p5_panet)  # (B, num_anchors*(5+nc), H/32, W/32)
        # Derive grid size dynamically from tensor shape
        _, _, h_p5, w_p5 = out_p5.shape
        out_p5 = out_p5.view(batch_size, self.num_anchors, 5 + self.num_classes, h_p5, w_p5)
        out_p5 = out_p5.permute(0, 3, 4, 1, 2).contiguous()  # (B, H/32, W/32, num_anchors, 5+nc)

        # Return list of predictions at three scales
        return [out_p3, out_p4, out_p5]

def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Complete IoU loss.

    Args:
        pred_boxes: (N, 4) - predicted boxes (x, y, w, h) in normalized coords
        target_boxes: (N, 4) - target boxes (x, y, w, h) in normalized coords

    Returns:
        CIoU loss value
    """
    # Extract coordinates
    pred_x, pred_y = pred_boxes[:, 0], pred_boxes[:, 1]
    pred_w, pred_h = pred_boxes[:, 2], pred_boxes[:, 3]
    target_x, target_y = target_boxes[:, 0], target_boxes[:, 1]
    target_w, target_h = target_boxes[:, 2], target_boxes[:, 3]

    # Convert to corner format
    pred_x1 = pred_x - pred_w / 2
    pred_y1 = pred_y - pred_h / 2
    pred_x2 = pred_x + pred_w / 2
    pred_y2 = pred_y + pred_h / 2

    target_x1 = target_x - target_w / 2
    target_y1 = target_y - target_h / 2
    target_x2 = target_x + target_w / 2
    target_y2 = target_y + target_h / 2

    # Intersection area
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    # Union area
    pred_area = pred_w * pred_h
    target_area = target_w * target_h
    union_area = pred_area + target_area - inter_area

    # IoU
    iou = inter_area / (union_area + eps)

    # Center distance
    center_dist = (pred_x - target_x) ** 2 + (pred_y - target_y) ** 2

    # Diagonal length of smallest enclosing box
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)

    enclose_w = enclose_x2 - enclose_x1
    enclose_h = enclose_y2 - enclose_y1
    enclose_diagonal = enclose_w ** 2 + enclose_h ** 2 + eps

    # Distance penalty
    distance_penalty = center_dist / enclose_diagonal

    # Aspect ratio consistency
    arctan_pred = torch.atan(pred_w / (pred_h + eps))
    arctan_target = torch.atan(target_w / (target_h + eps))
    v = (4 / (torch.pi ** 2)) * torch.pow(arctan_pred - arctan_target, 2)

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    # CIoU
    ciou = iou - distance_penalty - alpha * v

    # Loss is 1 - CIoU
    loss = 1 - ciou

    return loss.mean()

def decode_predictions(raw_preds, anchors, img_size=640):
    """
    Decode raw YOLO predictions from offset format to absolute coordinates.

    Implements YOLOv5-style decoding where the network outputs offsets (t_x, t_y, t_w, t_h)
    that are transformed into absolute bounding box coordinates:
    - Centers: b_x = ((σ(t_x) * 2 - 0.5) + c_x) / grid_width
    - Dims: b_w = (anchor_w * (2 * σ(t_w))²) / img_size

    This constrains predictions to be near their responsible grid cell and scales
    dimensions relative to anchor size, improving training stability.

    Args:
        raw_preds: (batch, grid_h, grid_w, num_anchors, 5+nc)
                   Raw network outputs where first 4 values are t_x, t_y, t_w, t_h
        anchors: (num_anchors, 2) tensor of anchor dimensions [width, height] in pixels
        img_size: Input image size in pixels (default 640)

    Returns:
        decoded: Same shape as raw_preds but with first 4 values as absolute coordinates
                (b_x, b_y, b_w, b_h) in normalized [0,1] range.
                Objectness and class predictions remain as logits (unchanged).
    """
    # Always derive grid dimensions from tensor shape (robust to any input size)
    _, h, w, num_anchors, _ = raw_preds.shape
    decoded = raw_preds.clone()

    # Create grid coordinate meshes
    # grid_x[i,j] = j (column index), grid_y[i,j] = i (row index)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=raw_preds.device, dtype=raw_preds.dtype),
        torch.arange(w, device=raw_preds.device, dtype=raw_preds.dtype),
        indexing='ij'
    )
    # Reshape for broadcasting: (h, w) -> (1, h, w, 1)
    grid_x = grid_x.view(1, h, w, 1)
    grid_y = grid_y.view(1, h, w, 1)

    # Move anchors to same device as predictions
    anchors = anchors.to(raw_preds.device)

    # Decode center coordinates (x, y)
    # Sigmoid constrains offset to [0, 1], multiply by 2 and subtract 0.5 gives [-0.5, 1.5]
    # This allows the center to be up to 0.5 cells outside the responsible cell
    # Formula: b_x = ((σ(t_x) * 2 - 0.5) + c_x) / grid_width
    # Use actual grid dimensions (w for x, h for y) instead of assuming square
    decoded[..., 0] = ((torch.sigmoid(raw_preds[..., 0]) * 2.0 - 0.5) + grid_x) / w
    decoded[..., 1] = ((torch.sigmoid(raw_preds[..., 1]) * 2.0 - 0.5) + grid_y) / h

    # Decode dimensions (w, h)
    # Sigmoid constrains to [0, 1], multiply by 2 gives [0, 2], square gives [0, 4]
    # This means predicted dimension can be up to 4x the anchor size
    # Formula: b_w = (anchor_w * (2 * σ(t_w))²) / img_size
    # Need to apply per-anchor since each anchor has different base dimensions
    for a in range(num_anchors):
        anchor_w, anchor_h = anchors[a]
        decoded[:, :, :, a, 2] = (anchor_w / img_size) * torch.pow(2.0 * torch.sigmoid(raw_preds[:, :, :, a, 2]), 2)
        decoded[:, :, :, a, 3] = (anchor_h / img_size) * torch.pow(2.0 * torch.sigmoid(raw_preds[:, :, :, a, 3]), 2)

    # Objectness and class predictions remain as logits (will be passed to BCEWithLogitsLoss)
    # No modification needed: decoded[..., 4:] already copied via clone()

    return decoded

def yolo_loss(predictions, targets, anchors, num_classes=1):
    """
    Composite YOLO loss function with anchor support and CIoU.

    Args:
        predictions: (batch, 13, 13, num_anchors, 5+nc) - RAW model output (t_x, t_y, t_w, t_h, ...)
        targets: (batch, 13, 13, num_anchors, 5+nc) - ground truth (x, y, w, h, ...)
        anchors: (num_anchors, 2) - anchor dimensions [width, height] in pixels
        num_classes: number of classes

    Returns:
        total_loss, bbox_loss, obj_loss, class_loss
    """
    # Decode predictions from offset format to absolute coordinates
    # This transforms t_x, t_y, t_w, t_h -> b_x, b_y, b_w, b_h
    decoded_preds = decode_predictions(predictions, anchors)

    # Extract components from DECODED predictions for bbox loss
    pred_boxes = decoded_preds[..., 0:4]     # x, y, w, h (now absolute coordinates)
    # Use RAW predictions for objectness and class (need logits for BCEWithLogitsLoss)
    pred_obj = predictions[..., 4:5]         # objectness (logits)
    pred_class = predictions[..., 5:]        # class probs (logits)

    target_boxes = targets[..., 0:4]
    target_obj = targets[..., 4:5]
    target_class = targets[..., 5:]

    # Create mask for cells+anchors that contain objects
    obj_mask = target_obj > 0.5  # (batch, 13, 13, num_anchors, 1)

    # 1. Bounding Box Loss (CIoU, only for cells with objects)
    if obj_mask.sum() > 0:
        # Extract boxes that have objects
        pred_boxes_obj = pred_boxes[obj_mask.squeeze(-1)]  # (N, 4)
        target_boxes_obj = target_boxes[obj_mask.squeeze(-1)]  # (N, 4)

        # Use CIoU loss
        bbox_loss = ciou_loss(pred_boxes_obj, target_boxes_obj)
    else:
        bbox_loss = torch.tensor(0.0, device=predictions.device)

    # 2. Objectness Loss (BCEWithLogitsLoss, all cells+anchors)
    obj_loss = nn.BCEWithLogitsLoss()(pred_obj, target_obj)

    # 3. Class Loss (BCEWithLogitsLoss, only for cells with objects)
    if obj_mask.sum() > 0 and num_classes > 0:
        class_loss = nn.BCEWithLogitsLoss()(
            pred_class[obj_mask.expand_as(pred_class)].view(-1),
            target_class[obj_mask.expand_as(target_class)].view(-1)
        )
    else:
        class_loss = torch.tensor(0.0, device=predictions.device)

    # Combined loss with weights (bbox weighted 5x higher)
    total_loss = 5.0 * bbox_loss + 1.0 * obj_loss + 1.0 * class_loss

    return total_loss, bbox_loss, obj_loss, class_loss

def yolo_loss_multiscale(predictions, targets, anchors_list, num_classes=1):
    """
    Multi-scale YOLO loss function for FPN architecture with per-scale objectness weighting.

    YOLOv5-style per-scale loss balancing:
    - P3 has many cells (80×80 = 6400) → needs higher objectness weight to balance gradients
    - P5 has few cells (20×20 = 400) → needs lower objectness weight
    - Without balancing, P3's objectness loss would dominate training

    Args:
        predictions: List of [pred_p3, pred_p4, pred_p5]
                    Each pred has shape (batch, grid_size, grid_size, num_anchors, 5+nc)
        targets: List of [target_p3, target_p4, target_p5]
                Each target has shape (batch, grid_size, grid_size, num_anchors, 5+nc)
        anchors_list: List of [anchors_p3, anchors_p4, anchors_p5]
                     Each anchors is (num_anchors, 2) tensor
        num_classes: Number of classes

    Returns:
        total_loss, total_bbox_loss, total_obj_loss, total_class_loss (weighted sum across scales)
    """
    # Per-scale objectness weights (balances gradient contributions)
    # P3: 4.0× (many cells, small objects)
    # P4: 1.0× (baseline)
    # P5: 0.4× (few cells, large objects)
    obj_weights = [4.0, 1.0, 0.4]

    total_loss = 0.0
    total_bbox_loss = 0.0
    total_obj_loss = 0.0
    total_class_loss = 0.0

    # Compute loss for each scale with appropriate objectness weighting
    for pred, target, anchors, obj_weight in zip(predictions, targets, anchors_list, obj_weights):
        _, bbox_loss, obj_loss, class_loss = yolo_loss(pred, target, anchors, num_classes)

        # Apply per-scale objectness weighting
        # Decompose total loss: total_loss = 5.0*bbox + 1.0*obj + 1.0*cls
        # Re-weight: total_loss = 5.0*bbox + obj_weight*obj + 1.0*cls
        weighted_loss = 5.0 * bbox_loss + obj_weight * obj_loss + 1.0 * class_loss

        total_loss += weighted_loss
        total_bbox_loss += bbox_loss
        total_obj_loss += obj_loss  # Track unweighted for logging
        total_class_loss += class_loss

    return total_loss, total_bbox_loss, total_obj_loss, total_class_loss

def train_epoch(model, loader, optimizer, device, num_classes=1):
    model.train()
    total_loss, total_bbox, total_obj, total_cls = 0, 0, 0, 0

    # Get multi-scale anchors from model
    anchors_list = model.anchors  # List of [anchors_p3, anchors_p4, anchors_p5]

    for imgs, targets in loader:
        imgs = imgs.to(device)

        # Targets is a list of [target_p3, target_p4, target_p5] per image
        # Need to stack into batch dimension: [batch of p3, batch of p4, batch of p5]
        targets_p3 = torch.stack([t[0] for t in targets]).to(device)  # (B, H/8, W/8, 3, 5+nc)
        targets_p4 = torch.stack([t[1] for t in targets]).to(device)  # (B, H/16, W/16, 3, 5+nc)
        targets_p5 = torch.stack([t[2] for t in targets]).to(device)  # (B, H/32, W/32, 3, 5+nc)
        targets_batch = [targets_p3, targets_p4, targets_p5]

        optimizer.zero_grad()
        preds = model(imgs)  # Returns [pred_p3, pred_p4, pred_p5]

        # Use multi-scale loss function
        loss, bbox_loss, obj_loss, cls_loss = yolo_loss_multiscale(
            preds, targets_batch, anchors_list, num_classes
        )

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        total_loss += loss.item()
        total_bbox += bbox_loss.item()
        total_obj += obj_loss.item()
        total_cls += cls_loss.item()

    n = len(loader)
    return total_loss/n, total_bbox/n, total_obj/n, total_cls/n

def compute_box_iou(box1, box2):
    """
    Compute IoU between two boxes in center format (x, y, w, h).
    Boxes are normalized coordinates (0-1).
    """
    # Convert to corner format
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2

    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    # Intersection
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou

def eval_epoch(model, loader, device, num_classes=1, iou_threshold=0.5, conf_threshold=0.5):
    """
    Evaluate with IoU-based detection metrics (precision, recall, F1) for multi-scale FPN.
    """
    model.eval()
    total_loss = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Get multi-scale anchors from model
    anchors_list = model.anchors  # List of [anchors_p3, anchors_p4, anchors_p5]
    grid_sizes = [model.grid_size_p3, model.grid_size_p4, model.grid_size_p5]

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)

            # Stack targets for all scales
            targets_p3 = torch.stack([t[0] for t in targets]).to(device)
            targets_p4 = torch.stack([t[1] for t in targets]).to(device)
            targets_p5 = torch.stack([t[2] for t in targets]).to(device)
            targets_batch = [targets_p3, targets_p4, targets_p5]

            preds = model(imgs)  # Returns [pred_p3, pred_p4, pred_p5]

            # Calculate loss
            loss, _, _, _ = yolo_loss_multiscale(preds, targets_batch, anchors_list, num_classes)
            total_loss += loss.item()

            # Evaluate predictions from all three scales
            for pred, target, anchors, grid_size in zip(preds, targets_batch, anchors_list, grid_sizes):
                # Decode predictions from offset format to absolute coordinates
                preds_decoded = decode_predictions(pred, anchors)

                # Apply sigmoid to objectness and class predictions for evaluation
                preds_eval = preds_decoded.clone()
                preds_eval[..., 4] = torch.sigmoid(pred[..., 4])
                if num_classes > 0:
                    preds_eval[..., 5:] = torch.sigmoid(pred[..., 5:])

                # Evaluate each image in batch
                for b in range(pred.shape[0]):
                    for i in range(grid_size):
                        for j in range(grid_size):
                            for a in range(pred.shape[3]):  # Iterate over anchors
                                pred_obj = preds_eval[b, i, j, a, 4].item()
                                target_obj = target[b, i, j, a, 4].item()

                                if pred_obj > conf_threshold and target_obj > conf_threshold:
                                    # Both predict and target have object - check IoU
                                    pred_box = preds_eval[b, i, j, a, 0:4]
                                    target_box = target[b, i, j, a, 0:4]
                                    iou = compute_box_iou(pred_box, target_box)

                                    if iou > iou_threshold:
                                        true_positives += 1
                                    else:
                                        false_positives += 1
                                elif pred_obj > conf_threshold and target_obj <= conf_threshold:
                                    # False positive: predicted object but no ground truth
                                    false_positives += 1
                                elif pred_obj <= conf_threshold and target_obj > conf_threshold:
                                    # False negative: missed detection
                                    false_negatives += 1

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    avg_loss = total_loss / len(loader)
    return avg_loss, precision * 100, recall * 100, f1 * 100

def get_lr_lambda(warmup_epochs=3, total_epochs=100, initial_lr=1e-2, min_lr=1e-4, warmup_start_lr=1e-6):
    """
    Create a learning rate scheduler function with warmup + cosine annealing.

    YOLOv5-style learning rate schedule:
    1. Linear warmup: Gradually increase LR from warmup_start_lr to initial_lr over warmup_epochs
    2. Cosine annealing: Smoothly decay LR from initial_lr to min_lr using cosine curve

    Args:
        warmup_epochs: Number of epochs for warmup phase (default 3)
        total_epochs: Total training epochs (default 100)
        initial_lr: Peak learning rate after warmup (default 1e-2)
        min_lr: Minimum learning rate at end of training (default 1e-4)
        warmup_start_lr: Starting learning rate for warmup (default 1e-6)

    Returns:
        Lambda function that takes epoch number and returns LR multiplier
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup: scale from warmup_start_lr to initial_lr
            return (warmup_start_lr + (initial_lr - warmup_start_lr) * epoch / warmup_epochs) / initial_lr
        else:
            # Cosine annealing: smoothly decay from initial_lr to min_lr
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return (min_lr + (initial_lr - min_lr) * cosine_decay) / initial_lr

    return lr_lambda

def compute_iou_corners(box1, box2):
    """
    Compute IoU between two boxes in corner format (x1, y1, x2, y2, conf, class_id).
    """
    x1_1, y1_1, x2_1, y2_1 = box1[0:4]
    x1_2, y1_2, x2_2, y2_2 = box2[0:4]

    # Intersection area
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def nms(detections, iou_threshold):
    """
    Non-Maximum Suppression.

    Args:
        detections: List of (x1, y1, x2, y2, conf, class_id)
        iou_threshold: IoU threshold for suppression

    Returns:
        List of kept detections
    """
    if len(detections) == 0:
        return []

    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x[4], reverse=True)

    keep = []
    while len(detections) > 0:
        # Keep highest confidence detection
        keep.append(detections[0])

        # Remove detections with high IoU overlap
        detections = [det for det in detections[1:]
                      if compute_iou_corners(keep[-1], det) < iou_threshold]

    return keep

def predict(model, image_path, device, num_classes=1, conf_threshold=0.5, iou_threshold=0.4):
    """
    Run multi-scale inference on a single image with global NMS.

    Args:
        model: YOLO FPN model
        image_path: Path to input image
        device: torch device
        num_classes: Number of classes
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for global NMS

    Returns:
        List of detections [(x1, y1, x2, y2, conf, class_id), ...]
        where coordinates are in original image scale
    """
    model.eval()

    # Load and preprocess image with letterbox resize
    pil_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = pil_img.size
    img_size = model.img_size
    pil_img, scale, pad_top, pad_left = letterbox_resize(pil_img, img_size)
    img = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        preds = model(img)  # Returns [pred_p3, pred_p4, pred_p5]

    # Multi-scale detection heads info
    anchors_list = model.anchors  # [anchors_p3, anchors_p4, anchors_p5]
    grid_sizes = [model.grid_size_p3, model.grid_size_p4, model.grid_size_p5]

    detections = []

    # Process each scale separately
    for pred, anchors, grid_size in zip(preds, anchors_list, grid_sizes):
        # Decode predictions from offset format to absolute coordinates
        preds_decoded = decode_predictions(pred, anchors, img_size)

        # Apply sigmoid to objectness and class predictions
        preds_decoded[..., 4] = torch.sigmoid(pred[..., 4])
        if num_classes > 0:
            preds_decoded[..., 5:] = torch.sigmoid(pred[..., 5:])

        num_anchors = pred.shape[3]

        # Convert grid predictions to image coordinates
        for i in range(grid_size):
            for j in range(grid_size):
                for a in range(num_anchors):
                    obj_conf = preds_decoded[0, i, j, a, 4].item()

                    if obj_conf > conf_threshold:
                        # Extract DECODED coordinates (already in normalized [0,1] range)
                        x_center = preds_decoded[0, i, j, a, 0].item()
                        y_center = preds_decoded[0, i, j, a, 1].item()
                        width = preds_decoded[0, i, j, a, 2].item()
                        height = preds_decoded[0, i, j, a, 3].item()

                        # Get class prediction
                        if num_classes == 1:
                            class_prob = preds_decoded[0, i, j, a, 5].item()
                            class_id = 0
                        else:
                            class_probs = preds_decoded[0, i, j, a, 5:].cpu().numpy()
                            class_id = int(class_probs.argmax())
                            class_prob = class_probs[class_id]

                        # Convert to pixel coordinates (0-img_size)
                        x_center_px = x_center * img_size
                        y_center_px = y_center * img_size
                        width_px = width * img_size
                        height_px = height * img_size

                        # Convert to corner format (in letterbox coordinates)
                        x1 = x_center_px - width_px / 2
                        y1 = y_center_px - height_px / 2
                        x2 = x_center_px + width_px / 2
                        y2 = y_center_px + height_px / 2

                        # Reverse letterbox transformation to get original image coordinates
                        # 1. Remove padding
                        x1 = x1 - pad_left
                        y1 = y1 - pad_top
                        x2 = x2 - pad_left
                        y2 = y2 - pad_top
                        # 2. Undo scaling
                        x1 = x1 / scale
                        y1 = y1 / scale
                        x2 = x2 / scale
                        y2 = y2 / scale

                        # Combined confidence
                        conf = obj_conf * class_prob
                        detections.append((x1, y1, x2, y2, conf, class_id))

    # Apply global NMS across all scales
    detections = nms(detections, iou_threshold)

    return detections

def compute_optimal_anchors(dataset_yaml, img_size=640, num_anchors=9):
    """
    Compute optimal anchors for a dataset using k-means clustering.

    This analyzes all bounding boxes in the training set and finds anchor sizes
    that best represent the distribution of object sizes. Better anchors lead to
    faster convergence and higher accuracy.

    Args:
        dataset_yaml: Path to dataset YAML config file
        img_size: Image size used for training (anchors scale with this)
        num_anchors: Total number of anchors to generate (default 9, will be split 3/3/3 across scales)

    Returns:
        List of 3 anchor sets: [anchors_p3, anchors_p4, anchors_p5]
        Each anchor set contains 3 anchors as [[w1, h1], [w2, h2], [w3, h3]]
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("ERROR: scikit-learn is required for anchor clustering.")
        print("Install with: pip install scikit-learn")
        return None

    # Load dataset configuration
    with open(dataset_yaml, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Collect all box dimensions from training set
    img_dir = config['train']
    label_files = sorted(glob.glob(f"{img_dir.replace('/images/', '/labels/')}/*.txt"))

    all_boxes = []
    for label_file in label_files:
        if Path(label_file).exists():
            with open(label_file, encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        # Extract width and height (normalized [0, 1])
                        width = float(parts[3])
                        height = float(parts[4])
                        # Convert to pixels relative to img_size
                        width_px = width * img_size
                        height_px = height * img_size
                        all_boxes.append([width_px, height_px])

    if len(all_boxes) == 0:
        print(f"ERROR: No boxes found in {img_dir}")
        return None

    all_boxes = np.array(all_boxes)
    print(f"Loaded {len(all_boxes)} boxes from {len(label_files)} images")
    print(f"Box size range: width [{all_boxes[:, 0].min():.1f}, {all_boxes[:, 0].max():.1f}], "
          f"height [{all_boxes[:, 1].min():.1f}, {all_boxes[:, 1].max():.1f}]")

    # Run k-means clustering
    print(f"\nRunning k-means clustering with k={num_anchors}...")
    kmeans = KMeans(n_clusters=num_anchors, random_state=0, n_init=10)
    kmeans.fit(all_boxes)

    # Get cluster centers (these are our anchors)
    anchors = kmeans.cluster_centers_
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]  # Sort by area

    print("\nOptimal anchors (sorted by area):")
    for i, (w, h) in enumerate(anchors):
        area = w * h
        print(f"  Anchor {i+1}: [{w:.1f}, {h:.1f}] (area: {area:.0f})")

    # Split into 3 scales (P3, P4, P5)
    # Small objects (P3): first 3 anchors
    # Medium objects (P4): middle 3 anchors
    # Large objects (P5): last 3 anchors
    anchors_p3 = anchors[0:3].round().astype(int).tolist()
    anchors_p4 = anchors[3:6].round().astype(int).tolist()
    anchors_p5 = anchors[6:9].round().astype(int).tolist()

    print("\n" + "="*60)
    print("Recommended anchor configuration:")
    print("="*60)
    print(f"P3 (small objects):  {anchors_p3}")
    print(f"P4 (medium objects): {anchors_p4}")
    print(f"P5 (large objects):  {anchors_p5}")
    print("\nTo use these anchors, pass them to YOLO() and YOLODataset():")
    print(f"  anchors = [{anchors_p3}, {anchors_p4}, {anchors_p5}]")
    print(f"  model = YOLO(num_classes=..., anchors=anchors, img_size={img_size})")
    print("="*60)

    return [anchors_p3, anchors_p4, anchors_p5]

# YOLOv5 model size variants
YOLO_SIZES = {
    'n': {'width_mult': 0.25, 'depth_mult': 0.33},  # Nano: ~1.9M params
    's': {'width_mult': 0.50, 'depth_mult': 0.33},  # Small: ~7.2M params (default)
    'm': {'width_mult': 0.75, 'depth_mult': 0.67},  # Medium: ~21.2M params
    'l': {'width_mult': 1.00, 'depth_mult': 1.00},  # Large: ~46.5M params
    'x': {'width_mult': 1.25, 'depth_mult': 1.33},  # XLarge: ~86.7M params
}

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLO Training/Inference')
    parser.add_argument('files', nargs='*', help='YAML config, .pt model, or image file')
    parser.add_argument('--size', type=str, default='s', choices=list(YOLO_SIZES.keys()),
                        help='Model size: n(ano), s(mall), m(edium), l(arge), x(large) (default: s)')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size (default: 640)')
    parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate (default: 0.01)')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='Number of warmup epochs (default: 3)')
    parser.add_argument('--min-lr', type=float, default=1e-4, help='Minimum learning rate (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs (default: 100)')
    parser.add_argument('--compute-anchors', action='store_true',
                        help='Compute optimal anchors for dataset using k-means')
    parsed_args = parser.parse_args()

    # Extract file types from positional arguments
    yaml_file = next((a for a in parsed_args.files if a.endswith('.yaml') or a.endswith('.yml')), None)
    pt_file = next((a for a in parsed_args.files if a.endswith('.pt')), None)
    image_file = next((a for a in parsed_args.files if a.endswith(('.jpg', '.png', '.jpeg'))), None)

    img_size = parsed_args.img_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Handle --compute-anchors mode
    if parsed_args.compute_anchors:
        if not yaml_file:
            print("ERROR: --compute-anchors requires a dataset YAML file")
            print("Usage: python train.py dataset.yaml --compute-anchors [--img-size SIZE]")
            sys.exit(1)
        print(f"Computing optimal anchors for {yaml_file} at img_size={img_size}...")
        compute_optimal_anchors(yaml_file, img_size=img_size)
        sys.exit(0)

    # Determine num_classes from config if available
    num_classes = 1
    config = None
    if yaml_file:
        with open(yaml_file, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        num_classes = config.get('nc', 1)

    # Get model size configuration
    size_config = YOLO_SIZES[parsed_args.size]
    width_mult = size_config['width_mult']
    depth_mult = size_config['depth_mult']

    # Create model with appropriate num_classes, img_size, and size multipliers
    print(f"Creating YOLOv5{parsed_args.size.upper()} "
          f"(width={width_mult}, depth={depth_mult})")
    model = YOLO(num_classes=num_classes, img_size=img_size,
                 width_mult=width_mult, depth_mult=depth_mult).to(device)

    if pt_file and not yaml_file and not image_file:
        # Inspect mode: python train.py model.pt
        checkpoint = torch.load(pt_file, map_location=device)
        # Use saved img_size and multipliers if available, otherwise use defaults
        img_size_ckpt = checkpoint.get('img_size', img_size)
        width_mult_ckpt = checkpoint.get('width_mult', 0.5)  # Default to 's'
        depth_mult_ckpt = checkpoint.get('depth_mult', 0.33)
        model = YOLO(num_classes=num_classes, img_size=img_size_ckpt,
                     width_mult=width_mult_ckpt, depth_mult=depth_mult_ckpt).to(device)
        model.load_state_dict(checkpoint['model'])
        print(f"Model loaded from {pt_file}")
        print(f"Number of classes: {num_classes}")
        print(f"Image size: {model.img_size}")
        print(f"Width multiplier: {model.width_mult}")
        print(f"Depth multiplier: {model.depth_mult}")
        print("\nModel architecture:")
        for name, param in model.named_parameters():
            print(f"  {name}: {list(param.shape)}, {param.numel()} parameters")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {total_params:,}")

    elif image_file and pt_file:
        # Inference mode: python train.py image.jpg model.pt
        checkpoint = torch.load(pt_file, map_location=device)
        # Use saved img_size and multipliers if available
        img_size_ckpt = checkpoint.get('img_size', img_size)
        width_mult_ckpt = checkpoint.get('width_mult', 0.5)
        depth_mult_ckpt = checkpoint.get('depth_mult', 0.33)
        model = YOLO(num_classes=num_classes, img_size=img_size_ckpt,
                     width_mult=width_mult_ckpt, depth_mult=depth_mult_ckpt).to(device)
        model.load_state_dict(checkpoint['model'])
        print(f"Running inference on {image_file}")
        print(f"Model: {pt_file}, Classes: {num_classes}, Image size: {model.img_size}")

        detections = predict(model, image_file, device, num_classes=num_classes)

        if len(detections) == 0:
            print("No objects detected.")
        else:
            print(f"\nDetected {len(detections)} object(s):")
            for i, (x1, y1, x2, y2, conf, class_id) in enumerate(detections):
                print(f"  {i+1}. Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), "
                      f"Confidence: {conf:.3f}, Class: {int(class_id)}")

    elif yaml_file and config is not None:
        # Training or evaluation mode
        if pt_file:
            # Eval mode: python train.py data.yaml model.pt
            checkpoint = torch.load(pt_file, map_location=device)
            # Use saved img_size and multipliers if available
            img_size = checkpoint.get('img_size', img_size)
            width_mult_ckpt = checkpoint.get('width_mult', 0.5)
            depth_mult_ckpt = checkpoint.get('depth_mult', 0.33)
            model = YOLO(num_classes=num_classes, img_size=img_size,
                         width_mult=width_mult_ckpt, depth_mult=depth_mult_ckpt).to(device)
            model.load_state_dict(checkpoint['model'])
            print(f"Evaluating model from {pt_file}")
            print(f"Number of classes: {num_classes}")
            print(f"Image size: {model.img_size}")
            print(f"Width multiplier: {model.width_mult}")
            print(f"Depth multiplier: {model.depth_mult}")

        # Create dataloaders with correct img_size and custom collate function
        train_loader = DataLoader(YOLODataset(config['train'], num_classes=num_classes, img_size=img_size),
                                   batch_size=8, shuffle=True, collate_fn=yolo_collate_fn)
        val_loader = DataLoader(YOLODataset(config['val'], num_classes=num_classes, img_size=img_size),
                                batch_size=8, collate_fn=yolo_collate_fn)

        if pt_file:

            train_loss, train_prec, train_rec, train_f1 = eval_epoch(model, train_loader, device, num_classes)
            val_loss, val_prec, val_rec, val_f1 = eval_epoch(model, val_loader, device, num_classes)

            print("\nTraining Set:")
            print(f"  Loss: {train_loss:.4f}")
            print(f"  Precision: {train_prec:.2f}%")
            print(f"  Recall: {train_rec:.2f}%")
            print(f"  F1 Score: {train_f1:.2f}%")

            print("\nValidation Set:")
            print(f"  Loss: {val_loss:.4f}")
            print(f"  Precision: {val_prec:.2f}%")
            print(f"  Recall: {val_rec:.2f}%")
            print(f"  F1 Score: {val_f1:.2f}%")
        else:
            # Train mode: python train.py data.yaml
            print("Training YOLO model")
            print(f"Number of classes: {num_classes}")
            print(f"Training images: {len(train_loader.dataset)}")
            print(f"Validation images: {len(val_loader.dataset)}")
            print(f"Device: {device}")
            print("\nLearning Rate Schedule:")
            print(f"  Initial LR: {parsed_args.lr}")
            print(f"  Minimum LR: {parsed_args.min_lr}")
            print(f"  Warmup epochs: {parsed_args.warmup_epochs}")
            print(f"  Total epochs: {parsed_args.epochs}")

            # Optimizer with initial learning rate (will be scaled by scheduler)
            optimizer = optim.Adam(model.parameters(), lr=parsed_args.lr)

            # Learning rate scheduler with warmup + cosine annealing
            lr_lambda_fn = get_lr_lambda(
                warmup_epochs=parsed_args.warmup_epochs,
                total_epochs=parsed_args.epochs,
                initial_lr=parsed_args.lr,
                min_lr=parsed_args.min_lr
            )
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fn)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"yolo_{timestamp}.pt"

            for epoch in tqdm(range(parsed_args.epochs), desc="Training"):
                train_loss, bbox_loss, obj_loss, cls_loss = train_epoch(
                    model, train_loader, optimizer, device, num_classes)
                val_loss, val_prec, val_rec, val_f1 = eval_epoch(model, val_loader, device, num_classes)

                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']

                tqdm.write(f"Epoch {epoch+1}: "
                          f"Loss: {train_loss:.4f} (bbox: {bbox_loss:.4f}, obj: {obj_loss:.4f}, cls: {cls_loss:.4f}) | "
                          f"Val: Loss {val_loss:.4f}, P {val_prec:.1f}%, R {val_rec:.1f}%, F1 {val_f1:.1f}% | "
                          f"LR: {current_lr:.6f}")

                torch.save({
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'num_classes': num_classes,
                    'img_size': img_size,
                    'width_mult': model.width_mult,
                    'depth_mult': model.depth_mult
                }, save_path)

                # Update learning rate
                scheduler.step()

            print(f"\nTraining complete. Model saved to {save_path}")
    else:
        print("Usage:")
        print("  Training:        python train.py data.yaml [OPTIONS]")
        print("  Evaluation:      python train.py data.yaml model.pt [--img-size SIZE]")
        print("  Inference:       python train.py image.jpg model.pt [--img-size SIZE]")
        print("  Inspect:         python train.py model.pt")
        print("  Compute Anchors: python train.py data.yaml --compute-anchors [--img-size SIZE]")
        print("")
        print("Options:")
        print("  --size {n,s,m,l,x}     Model size variant (default: s)")
        print("                         n=nano (~1.9M), s=small (~7.2M), m=medium (~21M),")
        print("                         l=large (~47M), x=xlarge (~87M) parameters")
        print("  --img-size SIZE        Input image size (default: 640)")
        print("                         Must be divisible by 32 (e.g., 416, 512, 640, 1280)")
        print("  --lr LR                Initial learning rate (default: 0.01)")
        print("  --min-lr LR            Minimum learning rate (default: 0.0001)")
        print("  --warmup-epochs N      Number of warmup epochs (default: 3)")
        print("  --epochs N             Total training epochs (default: 100)")
        print("  --compute-anchors      Run k-means clustering to find optimal anchors for dataset")
