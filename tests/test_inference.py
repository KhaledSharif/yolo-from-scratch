"""
Tests for inference functions (predict, NMS).
"""
import pytest
import torch
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
from train import predict, nms, compute_iou_corners, YOLO


class TestNMS:
    """Test Non-Maximum Suppression."""

    def test_nms_empty_input(self):
        """Test NMS with empty detections."""
        detections = []
        result = nms(detections, iou_threshold=0.5)
        assert result == []

    def test_nms_single_detection(self):
        """Test NMS with single detection."""
        detections = [(10, 10, 50, 50, 0.9, 0)]
        result = nms(detections, iou_threshold=0.5)
        assert len(result) == 1
        assert result[0] == detections[0]

    def test_nms_removes_overlapping_boxes(self):
        """Test NMS removes highly overlapping boxes."""
        detections = [
            (10, 10, 50, 50, 0.9, 0),  # High confidence
            (12, 12, 52, 52, 0.8, 0),  # Overlapping, lower confidence
            (100, 100, 150, 150, 0.85, 0)  # Non-overlapping
        ]
        result = nms(detections, iou_threshold=0.5)
        assert len(result) == 2
        # Should keep first (highest conf) and third (non-overlapping)
        assert result[0] == detections[0]
        assert result[1] == detections[2]

    def test_nms_keeps_non_overlapping_boxes(self):
        """Test NMS keeps non-overlapping boxes."""
        detections = [
            (10, 10, 50, 50, 0.9, 0),
            (100, 100, 150, 150, 0.8, 0),
            (200, 200, 250, 250, 0.85, 0)
        ]
        result = nms(detections, iou_threshold=0.5)
        assert len(result) == 3  # All should be kept

    def test_nms_confidence_ordering(self):
        """Test NMS processes boxes in confidence order."""
        detections = [
            (10, 10, 50, 50, 0.6, 0),  # Lower confidence
            (12, 12, 52, 52, 0.9, 0),  # Higher confidence, overlapping
        ]
        result = nms(detections, iou_threshold=0.5)
        # Should keep the higher confidence one
        assert len(result) == 1
        assert result[0][4] == 0.9

    def test_nms_with_different_thresholds(self):
        """Test NMS with different IoU thresholds."""
        detections = [
            (10, 10, 50, 50, 0.9, 0),
            (20, 20, 60, 60, 0.8, 0),  # Moderate overlap
        ]

        # Strict threshold (0.3) - should remove second
        result_strict = nms(detections, iou_threshold=0.3)
        assert len(result_strict) == 1

        # Lenient threshold (0.7) - should keep both
        result_lenient = nms(detections, iou_threshold=0.7)
        assert len(result_lenient) == 2


class TestComputeIoUCorners:
    """Test IoU computation with corner format boxes."""

    def test_iou_perfect_overlap(self):
        """Test IoU with identical boxes."""
        box1 = (10, 10, 50, 50, 0.9, 0)
        box2 = (10, 10, 50, 50, 0.8, 0)
        iou = compute_iou_corners(box1, box2)
        assert abs(iou - 1.0) < 1e-6

    def test_iou_no_overlap(self):
        """Test IoU with non-overlapping boxes."""
        box1 = (10, 10, 50, 50, 0.9, 0)
        box2 = (100, 100, 150, 150, 0.8, 0)
        iou = compute_iou_corners(box1, box2)
        assert iou == 0.0

    def test_iou_partial_overlap(self):
        """Test IoU with partial overlap."""
        box1 = (10, 10, 50, 50, 0.9, 0)
        box2 = (30, 30, 70, 70, 0.8, 0)
        iou = compute_iou_corners(box1, box2)
        assert 0.0 < iou < 1.0

    def test_iou_symmetry(self):
        """Test that IoU is symmetric."""
        box1 = (10, 10, 50, 50, 0.9, 0)
        box2 = (20, 20, 60, 60, 0.8, 0)
        iou1 = compute_iou_corners(box1, box2)
        iou2 = compute_iou_corners(box2, box1)
        assert abs(iou1 - iou2) < 1e-6


class TestPredict:
    """Test multi-scale prediction function."""

    def test_predict_returns_list(self, temp_dataset_dir, device):
        """Test that predict returns a list of detections."""
        model = YOLO(num_classes=1, img_size=640)
        model.eval()

        # Get first image from temp dataset
        from glob import glob
        image_path = glob(f"{temp_dataset_dir}/*.jpg")[0]

        detections = predict(model, image_path, device, num_classes=1, conf_threshold=0.5)

        assert isinstance(detections, list)

    def test_predict_detection_format(self, temp_dataset_dir, device):
        """Test that detections have correct format."""
        model = YOLO(num_classes=1, img_size=640)
        model.eval()

        from glob import glob
        image_path = glob(f"{temp_dataset_dir}/*.jpg")[0]

        # Use low confidence to get some detections
        detections = predict(model, image_path, device, num_classes=1, conf_threshold=0.3)

        if len(detections) > 0:
            # Each detection should be (x1, y1, x2, y2, conf, class_id)
            for det in detections:
                assert len(det) == 6
                x1, y1, x2, y2, conf, class_id = det
                assert x2 > x1
                assert y2 > y1
                assert 0.0 <= conf <= 1.0
                assert class_id >= 0

    def test_predict_confidence_filtering(self, temp_dataset_dir, device):
        """Test that confidence threshold filters detections."""
        model = YOLO(num_classes=1, img_size=640)
        model.eval()

        from glob import glob
        image_path = glob(f"{temp_dataset_dir}/*.jpg")[0]

        # Low threshold - more detections
        dets_low = predict(model, image_path, device, num_classes=1, conf_threshold=0.3)
        # High threshold - fewer detections
        dets_high = predict(model, image_path, device, num_classes=1, conf_threshold=0.95)

        assert len(dets_low) >= len(dets_high)

    def test_predict_nms_applied(self, temp_dataset_dir, device):
        """Test that NMS is applied during prediction."""
        model = YOLO(num_classes=1, img_size=640)
        model.eval()

        from glob import glob
        image_path = glob(f"{temp_dataset_dir}/*.jpg")[0]

        # Get detections (should have NMS applied)
        detections = predict(model, image_path, device, num_classes=1, conf_threshold=0.3)

        # All remaining detections should have low IoU with each other
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                iou = compute_iou_corners(detections[i], detections[j])
                assert iou < 0.4  # NMS threshold in predict()

    def test_predict_multi_scale(self, temp_dataset_dir, device):
        """Test that prediction processes all 3 scales."""
        model = YOLO(num_classes=1, img_size=640)
        model.eval()

        from glob import glob
        image_path = glob(f"{temp_dataset_dir}/*.jpg")[0]

        # Verify model has 3 scales
        assert len(model.anchors) == 3
        assert model.grid_size_p3 == 80
        assert model.grid_size_p4 == 40
        assert model.grid_size_p5 == 20

        # Run prediction (should process all scales)
        detections = predict(model, image_path, device, num_classes=1, conf_threshold=0.3)

        # Detections can come from any scale
        assert isinstance(detections, list)
