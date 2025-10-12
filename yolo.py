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
    def __init__(self, img_dir, num_classes=1):
        self.imgs = sorted(glob.glob(f"{img_dir}/*.jpg") + glob.glob(f"{img_dir}/*.png"))
        self.labels = [p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt' for p in self.imgs]
        self.num_classes = num_classes
        self.grid_size = 13
        self.output_dim = 5 + num_classes  # x, y, w, h, objectness + class probs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Load and preprocess image
        pil_img = Image.open(self.imgs[idx]).convert('RGB').resize((416, 416))
        img = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0

        # Initialize empty target (grid_size x grid_size x output_dim)
        target = torch.zeros((self.grid_size, self.grid_size, self.output_dim))

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

                        # Only assign if this cell doesn't already have an object
                        # (Simple handling: one object per grid cell)
                        if target[grid_y, grid_x, 4] == 0:
                            # Set bounding box coordinates
                            target[grid_y, grid_x, 0:4] = torch.tensor([x_center, y_center, width, height])
                            # Set objectness
                            target[grid_y, grid_x, 4] = 1.0
                            # Set class probability (for single class, just set to 1.0)
                            if self.num_classes == 1:
                                target[grid_y, grid_x, 5] = 1.0
                            else:
                                # For multi-class, use one-hot encoding
                                target[grid_y, grid_x, 5 + class_id] = 1.0

        return img, target

class YOLO(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.output_channels = 5 + num_classes  # x, y, w, h, objectness + class probs
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.1),
            nn.Conv2d(512, self.output_channels, 1)
        )

    def forward(self, x):
        return self.net(x).permute(0, 2, 3, 1)

def yolo_loss(predictions, targets, num_classes=1):
    """
    Composite YOLO loss function.

    Args:
        predictions: (batch, 13, 13, 5+nc) - model output
        targets: (batch, 13, 13, 5+nc) - ground truth
        num_classes: number of classes

    Returns:
        total_loss, bbox_loss, obj_loss, class_loss
    """
    # Separate components
    pred_boxes = predictions[..., 0:4]      # x, y, w, h
    pred_obj = predictions[..., 4:5]         # objectness (logits)
    pred_class = predictions[..., 5:]        # class probs (logits)

    target_boxes = targets[..., 0:4]
    target_obj = targets[..., 4:5]
    target_class = targets[..., 5:]

    # Create mask for cells that contain objects
    obj_mask = target_obj > 0.5  # (batch, 13, 13, 1)

    # 1. Bounding Box Loss (MSE, only for cells with objects)
    if obj_mask.sum() > 0:
        bbox_loss = nn.MSELoss()(
            pred_boxes[obj_mask.expand_as(pred_boxes)].view(-1),
            target_boxes[obj_mask.expand_as(target_boxes)].view(-1)
        )
    else:
        bbox_loss = torch.tensor(0.0, device=predictions.device)

    # 2. Objectness Loss (BCEWithLogitsLoss, all cells)
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

    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(imgs)

        # Use composite loss function
        loss, bbox_loss, obj_loss, cls_loss = yolo_loss(preds, targets, num_classes)

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

    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)

            # Calculate loss
            loss, _, _, _ = yolo_loss(preds, targets, num_classes)
            total_loss += loss.item()

            # Apply sigmoid to objectness and class predictions for evaluation
            preds_eval = preds.clone()
            preds_eval[..., 4] = torch.sigmoid(preds_eval[..., 4])
            if num_classes > 0:
                preds_eval[..., 5:] = torch.sigmoid(preds_eval[..., 5:])

            # Evaluate each image in batch
            for b in range(preds.shape[0]):
                for i in range(13):
                    for j in range(13):
                        pred_obj = preds_eval[b, i, j, 4].item()
                        target_obj = targets[b, i, j, 4].item()

                        if pred_obj > conf_threshold and target_obj > conf_threshold:
                            # Both predict and target have object - check IoU
                            pred_box = preds_eval[b, i, j, 0:4]
                            target_box = targets[b, i, j, 0:4]
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
    pil_img = pil_img.resize((416, 416))
    img = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        preds = model(img)  # (1, 13, 13, 5+nc)

    # Apply sigmoid to objectness and class predictions
    preds[..., 4] = torch.sigmoid(preds[..., 4])
    if num_classes > 0:
        preds[..., 5:] = torch.sigmoid(preds[..., 5:])

    detections = []

    # Convert grid predictions to image coordinates
    for i in range(13):
        for j in range(13):
            obj_conf = preds[0, i, j, 4].item()

            if obj_conf > conf_threshold:
                x_center = preds[0, i, j, 0].item()
                y_center = preds[0, i, j, 1].item()
                width = preds[0, i, j, 2].item()
                height = preds[0, i, j, 3].item()

                # Get class prediction
                if num_classes == 1:
                    class_prob = preds[0, i, j, 5].item()
                    class_id = 0
                else:
                    class_probs = preds[0, i, j, 5:].cpu().numpy()
                    class_id = int(class_probs.argmax())
                    class_prob = class_probs[class_id]

                # Convert to pixel coordinates (0-416)
                x_center_px = x_center * 416
                y_center_px = y_center * 416
                width_px = width * 416
                height_px = height * 416

                # Convert to corner format
                x1 = x_center_px - width_px / 2
                y1 = y_center_px - height_px / 2
                x2 = x_center_px + width_px / 2
                y2 = y_center_px + height_px / 2

                # Scale back to original image size
                x1 = (x1 / 416) * orig_w
                y1 = (y1 / 416) * orig_h
                x2 = (x2 / 416) * orig_w
                y2 = (y2 / 416) * orig_h

                # Combined confidence
                conf = obj_conf * class_prob
                detections.append((x1, y1, x2, y2, conf, class_id))

    # Apply NMS
    detections = nms(detections, iou_threshold)

    return detections

if __name__ == "__main__":
    args = sys.argv[1:]
    yaml_file = next((a for a in args if a.endswith('.yaml') or a.endswith('.yml')), None)
    pt_file = next((a for a in args if a.endswith('.pt')), None)
    image_file = next((a for a in args if a.endswith(('.jpg', '.png', '.jpeg'))), None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine num_classes from config if available
    num_classes = 1
    config = None
    if yaml_file:
        with open(yaml_file) as f:
            config = yaml.safe_load(f)
        num_classes = config.get('nc', 1)

    # Create model with appropriate num_classes
    model = YOLO(num_classes=num_classes).to(device)

    if pt_file and not yaml_file and not image_file:
        # Inspect mode: python yolo.py model.pt
        checkpoint = torch.load(pt_file, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"Model loaded from {pt_file}")
        print(f"Number of classes: {num_classes}")
        print("\nModel architecture:")
        for name, param in model.named_parameters():
            print(f"  {name}: {list(param.shape)}, {param.numel()} parameters")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {total_params:,}")

    elif image_file and pt_file:
        # Inference mode: python yolo.py image.jpg model.pt
        checkpoint = torch.load(pt_file, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"Running inference on {image_file}")
        print(f"Model: {pt_file}, Classes: {num_classes}")

        detections = predict(model, image_file, device, num_classes=num_classes)

        if len(detections) == 0:
            print("No objects detected.")
        else:
            print(f"\nDetected {len(detections)} object(s):")
            for i, (x1, y1, x2, y2, conf, class_id) in enumerate(detections):
                print(f"  {i+1}. Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), "
                      f"Confidence: {conf:.3f}, Class: {int(class_id)}")

    elif yaml_file:
        # Training or evaluation mode
        train_loader = DataLoader(YOLODataset(config['train'], num_classes=num_classes),
                                   batch_size=8, shuffle=True)
        val_loader = DataLoader(YOLODataset(config['val'], num_classes=num_classes),
                                batch_size=8)

        if pt_file:
            # Eval mode: python yolo.py data.yaml model.pt
            checkpoint = torch.load(pt_file, map_location=device)
            model.load_state_dict(checkpoint['model'])
            print(f"Evaluating model from {pt_file}")
            print(f"Number of classes: {num_classes}")

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
            # Train mode: python yolo.py data.yaml
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

                torch.save({'model': model.state_dict(), 'epoch': epoch, 'num_classes': num_classes}, save_path)

            print(f"\nTraining complete. Model saved to {save_path}")
    else:
        print("Usage:")
        print("  Training:     python yolo.py data.yaml")
        print("  Evaluation:   python yolo.py data.yaml model.pt")
        print("  Inference:    python yolo.py image.jpg model.pt")
        print("  Inspect:      python yolo.py model.pt")