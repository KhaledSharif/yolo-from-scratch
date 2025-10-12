#!/usr/bin/env python3
"""
Interactive evaluation viewer for YOLO model.

Usage:
    python eval.py model.pt dataset.yaml

Controls:
    Right Arrow / D: Next image
    Left Arrow / A:  Previous image
    Q / ESC:         Quit
"""

import torch
import cv2
import numpy as np
import yaml
import sys
import glob
from pathlib import Path
from yolo import YOLO, predict


def load_ground_truth(label_path):
    """
    Load ground truth boxes from YOLO format label file.

    Returns:
        List of (class_id, x_center, y_center, width, height) in normalized coords
    """
    boxes = []
    if Path(label_path).exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(float(parts[0]))
                    x_center, y_center, width, height = [float(x) for x in parts[1:]]
                    boxes.append((class_id, x_center, y_center, width, height))
    return boxes


def draw_boxes(img, gt_boxes, pred_boxes, class_names):
    """
    Draw ground truth and prediction boxes on image.

    Args:
        img: OpenCV image (BGR)
        gt_boxes: List of ground truth (class_id, x_center, y_center, w, h) in normalized coords
        pred_boxes: List of predictions (x1, y1, x2, y2, conf, class_id) in pixel coords
        class_names: List of class names

    Returns:
        Image with boxes drawn
    """
    img_display = img.copy()
    h, w = img.shape[:2]

    # Draw ground truth boxes (GREEN)
    for class_id, x_center, y_center, box_w, box_h in gt_boxes:
        # Convert normalized to pixel coordinates
        x_center_px = int(x_center * w)
        y_center_px = int(y_center * h)
        box_w_px = int(box_w * w)
        box_h_px = int(box_h * h)

        x1 = int(x_center_px - box_w_px / 2)
        y1 = int(y_center_px - box_h_px / 2)
        x2 = int(x_center_px + box_w_px / 2)
        y2 = int(y_center_px + box_h_px / 2)

        # Draw box
        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        label = f"GT: {class_name}"

        # Calculate text size and draw background
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_display, (x1, y1 - text_h - baseline - 5),
                     (x1 + text_w, y1), (0, 255, 0), -1)
        cv2.putText(img_display, label, (x1, y1 - baseline - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Draw prediction boxes (RED)
    for x1, y1, x2, y2, conf, class_id in pred_boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(class_id)

        # Draw box
        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw label
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        label = f"Pred: {class_name} {conf:.2f}"

        # Calculate text size and draw background
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_display, (x1, y2 + baseline),
                     (x1 + text_w, y2 + text_h + baseline + 5), (0, 0, 255), -1)
        cv2.putText(img_display, label, (x1, y2 + text_h + baseline + 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img_display


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    model_path = sys.argv[1]
    dataset_yaml = sys.argv[2]

    # Load dataset configuration
    with open(dataset_yaml) as f:
        config = yaml.safe_load(f)

    num_classes = config.get('nc', 1)
    class_names = config.get('names', [f'class_{i}' for i in range(num_classes)])

    # Collect all images from both train and val sets
    all_images = []
    for split in ['train', 'val']:
        if split in config:
            img_dir = config[split]
            images = sorted(glob.glob(f"{img_dir}/*.jpg") + glob.glob(f"{img_dir}/*.png"))
            all_images.extend([(img, split) for img in images])

    if len(all_images) == 0:
        print("No images found in dataset!")
        sys.exit(1)

    print(f"Found {len(all_images)} images")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = YOLO(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"Loaded model from {model_path}")

    # Interactive viewer
    current_idx = 0
    window_name = "YOLO Evaluation Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n" + "="*60)
    print("Controls:")
    print("  Right Arrow / D: Next image")
    print("  Left Arrow / A:  Previous image")
    print("  Q / ESC:         Quit")
    print("="*60 + "\n")

    while True:
        # Get current image
        img_path, split = all_images[current_idx]

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            current_idx = (current_idx + 1) % len(all_images)
            continue

        # Load ground truth
        label_path = img_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        gt_boxes = load_ground_truth(label_path)

        # Run inference
        detections = predict(model, img_path, device, num_classes=num_classes,
                           conf_threshold=0.25, iou_threshold=0.4)

        # Draw boxes
        img_display = draw_boxes(img, gt_boxes, detections, class_names)

        # Add info text
        info_text = [
            f"Image {current_idx + 1}/{len(all_images)} ({split} set)",
            f"File: {Path(img_path).name}",
            f"GT boxes: {len(gt_boxes)}, Predictions: {len(detections)}"
        ]

        # Draw info panel at top
        panel_height = 80
        panel = np.zeros((panel_height, img_display.shape[1], 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)

        y_offset = 20
        for text in info_text:
            cv2.putText(panel, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25

        # Combine panel and image
        img_with_panel = np.vstack([panel, img_display])

        # Add legend at bottom
        legend_height = 60
        legend = np.zeros((legend_height, img_display.shape[1], 3), dtype=np.uint8)
        legend[:] = (40, 40, 40)

        cv2.rectangle(legend, (10, 15), (30, 35), (0, 255, 0), 2)
        cv2.putText(legend, "Ground Truth", (40, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.rectangle(legend, (200, 15), (220, 35), (0, 0, 255), 2)
        cv2.putText(legend, "Prediction", (230, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Combine with legend
        final_img = np.vstack([img_with_panel, legend])

        # Display
        cv2.imshow(window_name, final_img)

        # Handle keyboard input
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == 83 or key == ord('d'):  # Right arrow or D
            current_idx = (current_idx + 1) % len(all_images)
        elif key == 81 or key == ord('a'):  # Left arrow or A
            current_idx = (current_idx - 1) % len(all_images)
        elif key == ord('s'):  # S - save screenshot
            save_path = f"eval_screenshot_{Path(img_path).stem}.png"
            cv2.imwrite(save_path, final_img)
            print(f"Saved screenshot: {save_path}")

    cv2.destroyAllWindows()
    print("\nViewer closed.")


if __name__ == "__main__":
    main()
