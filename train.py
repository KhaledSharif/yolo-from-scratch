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

class YOLODataset(Dataset):
    def __init__(self, img_dir, num_classes=1, anchors=None, img_size=640):
        self.imgs = sorted(glob.glob(f"{img_dir}/*.jpg") + glob.glob(f"{img_dir}/*.png"))
        self.labels = [p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt' for p in self.imgs]
        self.num_classes = num_classes
        self.img_size = img_size
        self.grid_size = img_size // 32  # 5 stride-2 convs = 32× downsample

        # Default anchors (width, height in pixels)
        if anchors is None:
            anchors = [[10, 13], [16, 30], [33, 23]]

        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.num_anchors = len(anchors)
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
        # Load and preprocess image
        pil_img = Image.open(self.imgs[idx]).convert('RGB').resize((self.img_size, self.img_size))
        img = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0

        # Initialize empty target (grid_size x grid_size x num_anchors x output_dim)
        target = torch.zeros((self.grid_size, self.grid_size, self.num_anchors, self.output_dim))

        # Load all boxes from label file
        if Path(self.labels[idx]).exists():
            with open(self.labels[idx]) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(float(parts[0]))
                        x_center, y_center, width, height = [float(x) for x in parts[1:]]

                        # Determine which grid cell is responsible
                        grid_x = int(x_center * self.grid_size)
                        grid_y = int(y_center * self.grid_size)

                        # Clamp to valid range
                        grid_x = min(grid_x, self.grid_size - 1)
                        grid_y = min(grid_y, self.grid_size - 1)

                        # Convert box dimensions to pixels
                        box_w_px = width * self.img_size
                        box_h_px = height * self.img_size
                        box_wh = torch.tensor([box_w_px, box_h_px])

                        # Find best matching anchor
                        ious = self.compute_anchor_iou(box_wh, self.anchors)
                        best_anchor_idx = torch.argmax(ious).item()

                        # Only assign if this cell+anchor doesn't already have an object
                        if target[grid_y, grid_x, best_anchor_idx, 4] == 0:
                            # Set bounding box coordinates
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

        return img, target

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

class YOLO(nn.Module):
    """
    YOLO object detection model with YOLOv5-style offset prediction.

    The model outputs ENCODED predictions (t_x, t_y, t_w, t_h) that must be decoded
    to absolute coordinates before use. See decode_predictions() function.

    Architecture:
    - 5 stride-2 conv layers reduce 416x416 input to 13x13 feature map
    - Detection head outputs: num_anchors * (5 + num_classes) channels per grid cell
    - Each anchor predicts: t_x, t_y, t_w, t_h (offsets), objectness, class_probs

    Decoding formulas (applied in decode_predictions):
    - b_x = ((σ(t_x) * 2 - 0.5) + c_x) / grid_size
    - b_y = ((σ(t_y) * 2 - 0.5) + c_y) / grid_size
    - b_w = (anchor_w * (2 * σ(t_w))²) / img_size
    - b_h = (anchor_h * (2 * σ(t_h))²) / img_size

    This approach constrains predictions to be near their responsible grid cell
    and scales dimensions relative to anchor size, improving training stability.
    """
    def __init__(self, num_classes=1, anchors=None, img_size=640):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.grid_size = img_size // 32  # 5 stride-2 convs = 2^5 = 32 downsample

        # Default anchors (width, height in pixels)
        # Optimized for cone-like objects at different scales
        if anchors is None:
            anchors = [[10, 13], [16, 30], [33, 23]]

        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.num_anchors = len(anchors)

        # Output: num_anchors * (5 + num_classes) per grid cell
        # Each anchor predicts: t_x, t_y, t_w, t_h (OFFSETS), objectness_logit, class_logits
        self.output_channels = self.num_anchors * (5 + num_classes)

        # Backbone: 5 stride-2 conv layers with SiLU activation (YOLOv5 standard)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.SiLU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.SiLU(),
        )

        # SPPF: Spatial Pyramid Pooling - Fast (YOLOv5 feature)
        self.sppf = SPPF(512, 512)

        # Detection head: 1x1 conv to output channels
        self.head = nn.Conv2d(512, self.output_channels, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        # Backbone: feature extraction
        x = self.backbone(x)  # (B, 512, grid_size, grid_size)
        # SPPF: multi-scale pooling
        x = self.sppf(x)  # (B, 512, grid_size, grid_size)
        # Detection head
        out = self.head(x)  # (B, num_anchors*(5+nc), grid_size, grid_size)

        # Reshape to (B, grid_size, grid_size, num_anchors, 5+nc)
        out = out.view(batch_size, self.num_anchors, 5 + self.num_classes,
                      self.grid_size, self.grid_size)
        out = out.permute(0, 3, 4, 1, 2).contiguous()

        return out

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
    pred_x, pred_y, pred_w, pred_h = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    target_x, target_y, target_w, target_h = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

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

def decode_predictions(raw_preds, anchors, grid_size=None, img_size=640):
    """
    Decode raw YOLO predictions from offset format to absolute coordinates.

    Implements YOLOv5-style decoding where the network outputs offsets (t_x, t_y, t_w, t_h)
    that are transformed into absolute bounding box coordinates:
    - Centers: b_x = ((σ(t_x) * 2 - 0.5) + c_x) / grid_size
    - Dims: b_w = (anchor_w * (2 * σ(t_w))²) / img_size

    This constrains predictions to be near their responsible grid cell and scales
    dimensions relative to anchor size, improving training stability.

    Args:
        raw_preds: (batch, grid_size, grid_size, num_anchors, 5+nc)
                   Raw network outputs where first 4 values are t_x, t_y, t_w, t_h
        anchors: (num_anchors, 2) tensor of anchor dimensions [width, height] in pixels
        grid_size: Size of detection grid (default 13)
        img_size: Input image size in pixels (default 416)

    Returns:
        decoded: Same shape as raw_preds but with first 4 values as absolute coordinates
                (b_x, b_y, b_w, b_h) in normalized [0,1] range.
                Objectness and class predictions remain as logits (unchanged).
    """
    _, h, w, num_anchors, _ = raw_preds.shape
    # Infer grid_size from shape if not provided
    if grid_size is None:
        grid_size = h  # Assume square grid
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
    # Formula: b_x = ((σ(t_x) * 2 - 0.5) + c_x) / grid_size
    decoded[..., 0] = ((torch.sigmoid(raw_preds[..., 0]) * 2.0 - 0.5) + grid_x) / grid_size
    decoded[..., 1] = ((torch.sigmoid(raw_preds[..., 1]) * 2.0 - 0.5) + grid_y) / grid_size

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

def train_epoch(model, loader, optimizer, device, num_classes=1):
    model.train()
    total_loss, total_bbox, total_obj, total_cls = 0, 0, 0, 0

    # Get anchors from model for decoding predictions
    anchors = model.anchors

    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(imgs)

        # Use composite loss function (will decode predictions internally)
        loss, bbox_loss, obj_loss, cls_loss = yolo_loss(preds, targets, anchors, num_classes)

        loss.backward()
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
    Evaluate with IoU-based detection metrics (precision, recall, F1).
    """
    model.eval()
    total_loss = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Get anchors from model for decoding predictions
    anchors = model.anchors

    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)

            # Calculate loss (will decode predictions internally)
            loss, _, _, _ = yolo_loss(preds, targets, anchors, num_classes)
            total_loss += loss.item()

            # Decode predictions from offset format to absolute coordinates
            preds_decoded = decode_predictions(preds, anchors)

            # Apply sigmoid to objectness and class predictions for evaluation
            # Use RAW predictions for objectness/class (need to apply sigmoid)
            preds_eval = preds_decoded.clone()
            preds_eval[..., 4] = torch.sigmoid(preds[..., 4])
            if num_classes > 0:
                preds_eval[..., 5:] = torch.sigmoid(preds[..., 5:])

            # Evaluate each image in batch (now with anchor dimension)
            grid_size = model.grid_size
            for b in range(preds.shape[0]):
                for i in range(grid_size):
                    for j in range(grid_size):
                        for a in range(preds.shape[3]):  # Iterate over anchors
                            pred_obj = preds_eval[b, i, j, a, 4].item()
                            target_obj = targets[b, i, j, a, 4].item()

                            if pred_obj > conf_threshold and target_obj > conf_threshold:
                                # Both predict and target have object - check IoU
                                pred_box = preds_eval[b, i, j, a, 0:4]
                                target_box = targets[b, i, j, a, 0:4]
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
    Run inference on a single image with NMS.

    Args:
        model: YOLO model
        image_path: Path to input image
        device: torch device
        num_classes: Number of classes
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS

    Returns:
        List of detections [(x1, y1, x2, y2, conf, class_id), ...]
        where coordinates are in original image scale
    """
    model.eval()

    # Load and preprocess image
    pil_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = pil_img.size
    img_size = model.img_size
    pil_img = pil_img.resize((img_size, img_size))
    img = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        preds = model(img)  # (1, grid_size, grid_size, num_anchors, 5+nc) - RAW outputs

    # Decode predictions from offset format to absolute coordinates
    preds_decoded = decode_predictions(preds, model.anchors, model.grid_size, img_size)

    # Apply sigmoid to objectness and class predictions
    # Use RAW predictions for sigmoid (not decoded)
    preds_decoded[..., 4] = torch.sigmoid(preds[..., 4])
    if num_classes > 0:
        preds_decoded[..., 5:] = torch.sigmoid(preds[..., 5:])

    detections = []
    num_anchors = preds.shape[3]
    grid_size = model.grid_size

    # Convert grid predictions to image coordinates (iterate over anchors too)
    for i in range(grid_size):
        for j in range(grid_size):
            for a in range(num_anchors):  # Iterate over all anchors
                obj_conf = preds_decoded[0, i, j, a, 4].item()

                if obj_conf > conf_threshold:
                    # Extract DECODED coordinates (already in normalized [0,1] range)
                    x_center = preds_decoded[0, i, j, a, 0].item()
                    y_center = preds_decoded[0, i, j, a, 1].item()
                    width = preds_decoded[0, i, j, a, 2].item()
                    height = preds_decoded[0, i, j, a, 3].item()

                    # Get class prediction (use decoded predictions with sigmoid already applied)
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

                    # Convert to corner format
                    x1 = x_center_px - width_px / 2
                    y1 = y_center_px - height_px / 2
                    x2 = x_center_px + width_px / 2
                    y2 = y_center_px + height_px / 2

                    # Scale back to original image size
                    x1 = (x1 / img_size) * orig_w
                    y1 = (y1 / img_size) * orig_h
                    x2 = (x2 / img_size) * orig_w
                    y2 = (y2 / img_size) * orig_h

                    # Combined confidence
                    conf = obj_conf * class_prob
                    detections.append((x1, y1, x2, y2, conf, class_id))

    # Apply NMS
    detections = nms(detections, iou_threshold)

    return detections

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLO Training/Inference')
    parser.add_argument('files', nargs='*', help='YAML config, .pt model, or image file')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size (default: 640)')
    parsed_args = parser.parse_args()

    # Extract file types from positional arguments
    yaml_file = next((a for a in parsed_args.files if a.endswith('.yaml') or a.endswith('.yml')), None)
    pt_file = next((a for a in parsed_args.files if a.endswith('.pt')), None)
    image_file = next((a for a in parsed_args.files if a.endswith(('.jpg', '.png', '.jpeg'))), None)

    img_size = parsed_args.img_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine num_classes from config if available
    num_classes = 1
    config = None
    if yaml_file:
        with open(yaml_file) as f:
            config = yaml.safe_load(f)
        num_classes = config.get('nc', 1)

    # Create model with appropriate num_classes and img_size
    model = YOLO(num_classes=num_classes, img_size=img_size).to(device)

    if pt_file and not yaml_file and not image_file:
        # Inspect mode: python train.py model.pt
        checkpoint = torch.load(pt_file, map_location=device)
        # Use saved img_size if available, otherwise use CLI arg
        if 'img_size' in checkpoint:
            model = YOLO(num_classes=num_classes, img_size=checkpoint['img_size']).to(device)
        model.load_state_dict(checkpoint['model'])
        print(f"Model loaded from {pt_file}")
        print(f"Number of classes: {num_classes}")
        print(f"Image size: {model.img_size}")
        print("\nModel architecture:")
        for name, param in model.named_parameters():
            print(f"  {name}: {list(param.shape)}, {param.numel()} parameters")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {total_params:,}")

    elif image_file and pt_file:
        # Inference mode: python train.py image.jpg model.pt
        checkpoint = torch.load(pt_file, map_location=device)
        # Use saved img_size if available, otherwise use CLI arg
        if 'img_size' in checkpoint:
            model = YOLO(num_classes=num_classes, img_size=checkpoint['img_size']).to(device)
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
            # Use saved img_size if available, otherwise use CLI arg
            if 'img_size' in checkpoint:
                img_size = checkpoint['img_size']
                model = YOLO(num_classes=num_classes, img_size=img_size).to(device)
            model.load_state_dict(checkpoint['model'])
            print(f"Evaluating model from {pt_file}")
            print(f"Number of classes: {num_classes}")
            print(f"Image size: {model.img_size}")

        # Create dataloaders with correct img_size
        train_loader = DataLoader(YOLODataset(config['train'], num_classes=num_classes, img_size=img_size),
                                   batch_size=8, shuffle=True)
        val_loader = DataLoader(YOLODataset(config['val'], num_classes=num_classes, img_size=img_size),
                                batch_size=8)

        if pt_file:

            train_loss, train_prec, train_rec, train_f1 = eval_epoch(model, train_loader, device, num_classes)
            val_loss, val_prec, val_rec, val_f1 = eval_epoch(model, val_loader, device, num_classes)

            print(f"\nTraining Set:")
            print(f"  Loss: {train_loss:.4f}")
            print(f"  Precision: {train_prec:.2f}%")
            print(f"  Recall: {train_rec:.2f}%")
            print(f"  F1 Score: {train_f1:.2f}%")

            print(f"\nValidation Set:")
            print(f"  Loss: {val_loss:.4f}")
            print(f"  Precision: {val_prec:.2f}%")
            print(f"  Recall: {val_rec:.2f}%")
            print(f"  F1 Score: {val_f1:.2f}%")
        else:
            # Train mode: python train.py data.yaml
            print(f"Training YOLO model")
            print(f"Number of classes: {num_classes}")
            print(f"Training images: {len(train_loader.dataset)}")
            print(f"Validation images: {len(val_loader.dataset)}")
            print(f"Device: {device}")

            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"yolo_{timestamp}.pt"

            for epoch in tqdm(range(100), desc="Training"):
                train_loss, bbox_loss, obj_loss, cls_loss = train_epoch(model, train_loader, optimizer, device, num_classes)
                val_loss, val_prec, val_rec, val_f1 = eval_epoch(model, val_loader, device, num_classes)

                tqdm.write(f"Epoch {epoch+1}: "
                          f"Loss: {train_loss:.4f} (bbox: {bbox_loss:.4f}, obj: {obj_loss:.4f}, cls: {cls_loss:.4f}) | "
                          f"Val: Loss {val_loss:.4f}, P {val_prec:.1f}%, R {val_rec:.1f}%, F1 {val_f1:.1f}%")

                torch.save({'model': model.state_dict(), 'epoch': epoch, 'num_classes': num_classes, 'img_size': img_size}, save_path)

            print(f"\nTraining complete. Model saved to {save_path}")
    else:
        print("Usage:")
        print("  Training:     python train.py data.yaml [--img-size SIZE]")
        print("  Evaluation:   python train.py data.yaml model.pt [--img-size SIZE]")
        print("  Inference:    python train.py image.jpg model.pt [--img-size SIZE]")
        print("  Inspect:      python train.py model.pt")
        print("")
        print("Options:")
        print("  --img-size SIZE  Input image size (default: 640)")
        print("                   Must be divisible by 32 (e.g., 416, 512, 640, 1280)")