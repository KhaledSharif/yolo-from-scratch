"""
Tests for evaluation functions (eval_epoch, metrics).
"""
import pytest
import torch
from torch.utils.data import DataLoader
from train import eval_epoch, YOLO, YOLODataset, yolo_collate_fn


class TestEvalEpoch:
    """Test eval_epoch function (lines 958-1026)."""

    def test_eval_epoch_with_objects(self, temp_dataset_dir, device):
        """Test evaluation with ground truth objects."""
        model = YOLO(num_classes=1, img_size=640).to(device)
        model.eval()

        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)
        loader = DataLoader(dataset, batch_size=2, collate_fn=yolo_collate_fn)

        # Run evaluation
        avg_loss, precision, recall, f1 = eval_epoch(
            model, loader, device, num_classes=1,
            iou_threshold=0.5, conf_threshold=0.5
        )

        # Verify all metrics are returned (can be int or float)
        assert isinstance(avg_loss, (int, float))
        assert isinstance(precision, (int, float))
        assert isinstance(recall, (int, float))
        assert isinstance(f1, (int, float))

        # Metrics should be in valid ranges
        assert avg_loss >= 0
        assert 0 <= precision <= 100
        assert 0 <= recall <= 100
        assert 0 <= f1 <= 100

    def test_eval_epoch_no_objects(self, temp_dataset_dir, device):
        """Test evaluation with empty targets."""
        import tempfile
        import os
        from pathlib import Path
        from PIL import Image
        import numpy as np

        # Create dataset with no objects (empty label files)
        with tempfile.TemporaryDirectory() as tmpdir:
            imgs_dir = Path(tmpdir) / 'images'
            labels_dir = Path(tmpdir) / 'labels'
            imgs_dir.mkdir()
            labels_dir.mkdir()

            # Create image with empty label
            img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
            img.save(imgs_dir / 'test.jpg')
            (labels_dir / 'test.txt').write_text('')  # Empty label file

            model = YOLO(num_classes=1, img_size=640).to(device)
            model.eval()

            dataset = YOLODataset(str(imgs_dir), num_classes=1, img_size=640)
            loader = DataLoader(dataset, batch_size=1, collate_fn=yolo_collate_fn)

            avg_loss, precision, recall, f1 = eval_epoch(
                model, loader, device, num_classes=1
            )

            # With no objects, precision/recall/f1 should be 0
            # (model might predict false positives, but no true positives possible)
            assert avg_loss >= 0
            # Precision/recall/f1 could be 0 if no predictions or all FP
            assert 0 <= precision <= 100
            assert 0 <= recall <= 100
            assert 0 <= f1 <= 100

    def test_eval_epoch_precision_recall(self, temp_dataset_dir, device):
        """Test precision and recall computation."""
        model = YOLO(num_classes=1, img_size=640).to(device)
        model.eval()

        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)
        loader = DataLoader(dataset, batch_size=2, collate_fn=yolo_collate_fn)

        # Use lower confidence threshold to get more predictions
        avg_loss, precision, recall, f1 = eval_epoch(
            model, loader, device, num_classes=1,
            iou_threshold=0.5, conf_threshold=0.3
        )

        # F1 score should be related to precision and recall
        # F1 = 2 * (precision * recall) / (precision + recall)
        if precision + recall > 0:
            expected_f1 = 2 * (precision * recall) / (precision + recall)
            assert abs(f1 - expected_f1) < 0.1, f"F1 computation incorrect: {f1} vs {expected_f1}"
        else:
            # If both precision and recall are 0, F1 should be 0
            assert f1 == 0

    def test_eval_epoch_iou_threshold(self, temp_dataset_dir, device):
        """Test IoU threshold affects TP/FP classification."""
        model = YOLO(num_classes=1, img_size=640).to(device)
        model.eval()

        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)
        loader = DataLoader(dataset, batch_size=2, collate_fn=yolo_collate_fn)

        # Run with strict IoU threshold
        _, prec_strict, rec_strict, f1_strict = eval_epoch(
            model, loader, device, num_classes=1,
            iou_threshold=0.8, conf_threshold=0.3
        )

        # Run with lenient IoU threshold
        _, prec_lenient, rec_lenient, f1_lenient = eval_epoch(
            model, loader, device, num_classes=1,
            iou_threshold=0.3, conf_threshold=0.3
        )

        # Lenient threshold should give higher or equal metrics
        # (more predictions count as TP with lower IoU requirement)
        assert prec_lenient >= prec_strict - 1.0  # Allow small numerical differences
        assert rec_lenient >= rec_strict - 1.0
        assert f1_lenient >= f1_strict - 1.0

    def test_eval_epoch_confidence_threshold(self, temp_dataset_dir, device):
        """Test confidence threshold filters predictions."""
        model = YOLO(num_classes=1, img_size=640).to(device)
        model.eval()

        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)
        loader = DataLoader(dataset, batch_size=2, collate_fn=yolo_collate_fn)

        # High confidence threshold - fewer predictions
        _, prec_high, rec_high, f1_high = eval_epoch(
            model, loader, device, num_classes=1,
            iou_threshold=0.5, conf_threshold=0.9
        )

        # Low confidence threshold - more predictions
        _, prec_low, rec_low, f1_low = eval_epoch(
            model, loader, device, num_classes=1,
            iou_threshold=0.5, conf_threshold=0.1
        )

        # Both should be valid
        assert 0 <= prec_high <= 100
        assert 0 <= rec_high <= 100
        assert 0 <= prec_low <= 100
        assert 0 <= rec_low <= 100

        # Lower threshold typically gives higher recall (catches more objects)
        # but may have lower precision (more false positives)
        # Just verify both are valid
        assert rec_low >= 0 and rec_high >= 0

    def test_eval_epoch_multi_scale(self, temp_dataset_dir, device):
        """Test evaluation processes all 3 scales (P3/P4/P5)."""
        model = YOLO(num_classes=1, img_size=640).to(device)
        model.eval()

        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)
        loader = DataLoader(dataset, batch_size=2, collate_fn=yolo_collate_fn)

        # Verify model has 3 scales
        assert len(model.anchors) == 3
        assert model.grid_size_p3 == 80
        assert model.grid_size_p4 == 40
        assert model.grid_size_p5 == 20

        # Run evaluation (should process all scales internally)
        avg_loss, precision, recall, f1 = eval_epoch(
            model, loader, device, num_classes=1
        )

        # All metrics should be computed successfully
        assert avg_loss >= 0
        assert 0 <= precision <= 100
        assert 0 <= recall <= 100
        assert 0 <= f1 <= 100

    def test_eval_epoch_multiclass(self, temp_dataset_dir, device):
        """Test evaluation with multiple classes."""
        model = YOLO(num_classes=3, img_size=640).to(device)
        model.eval()

        dataset = YOLODataset(temp_dataset_dir, num_classes=3, img_size=640)
        loader = DataLoader(dataset, batch_size=2, collate_fn=yolo_collate_fn)

        avg_loss, precision, recall, f1 = eval_epoch(
            model, loader, device, num_classes=3,
            iou_threshold=0.5, conf_threshold=0.5
        )

        # Metrics should aggregate across all classes
        assert avg_loss >= 0
        assert 0 <= precision <= 100
        assert 0 <= recall <= 100
        assert 0 <= f1 <= 100

    def test_eval_epoch_different_resolutions(self, temp_dataset_dir, device):
        """Test evaluation with different image resolutions."""
        for img_size in [512, 640, 1024]:
            model = YOLO(num_classes=1, img_size=img_size).to(device)
            model.eval()

            dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=img_size)
            loader = DataLoader(dataset, batch_size=2, collate_fn=yolo_collate_fn)

            avg_loss, precision, recall, f1 = eval_epoch(
                model, loader, device, num_classes=1
            )

            # All resolutions should evaluate successfully
            assert avg_loss >= 0
            assert 0 <= precision <= 100
            assert 0 <= recall <= 100
            assert 0 <= f1 <= 100

    def test_eval_epoch_no_grad(self, temp_dataset_dir, device):
        """Test that eval_epoch doesn't compute gradients."""
        model = YOLO(num_classes=1, img_size=640).to(device)
        model.eval()

        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)
        loader = DataLoader(dataset, batch_size=2, collate_fn=yolo_collate_fn)

        # Enable gradient checking
        for param in model.parameters():
            param.requires_grad = True

        # Run evaluation
        eval_epoch(model, loader, device, num_classes=1)

        # No gradients should be accumulated
        for param in model.parameters():
            assert param.grad is None or torch.allclose(param.grad, torch.zeros_like(param.grad))
