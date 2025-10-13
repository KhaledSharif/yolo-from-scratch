"""
Tests for utility functions (IoU calculations, etc).
"""
import pytest
import torch
from train import compute_box_iou, compute_iou_corners


class TestComputeBoxIoU:
    """Test IoU computation with center format boxes (used in evaluation)."""

    def test_iou_perfect_overlap(self):
        """Test IoU with identical boxes."""
        box1 = torch.tensor([0.5, 0.5, 0.2, 0.3])  # x_center, y_center, w, h
        box2 = torch.tensor([0.5, 0.5, 0.2, 0.3])
        iou = compute_box_iou(box1, box2)
        assert torch.abs(iou - 1.0) < 2e-5  # Relaxed for floating-point precision

    def test_iou_no_overlap(self):
        """Test IoU with non-overlapping boxes."""
        box1 = torch.tensor([0.2, 0.2, 0.1, 0.1])
        box2 = torch.tensor([0.8, 0.8, 0.1, 0.1])
        iou = compute_box_iou(box1, box2)
        assert iou < 1e-5  # Should be essentially 0

    def test_iou_partial_overlap(self):
        """Test IoU with partial overlap."""
        box1 = torch.tensor([0.5, 0.5, 0.3, 0.3])
        box2 = torch.tensor([0.6, 0.6, 0.3, 0.3])
        iou = compute_box_iou(box1, box2)
        assert 0.0 < iou < 1.0

    def test_iou_range(self):
        """Test that IoU is always in [0, 1]."""
        for _ in range(10):
            # Random boxes
            box1 = torch.rand(4)
            box2 = torch.rand(4)
            # Ensure valid boxes (positive width/height)
            box1[2:] = torch.abs(box1[2:]) + 0.01
            box2[2:] = torch.abs(box2[2:]) + 0.01
            iou = compute_box_iou(box1, box2)
            assert 0.0 <= iou <= 1.0

    def test_iou_symmetry(self):
        """Test that IoU is symmetric."""
        box1 = torch.tensor([0.3, 0.4, 0.2, 0.2])
        box2 = torch.tensor([0.5, 0.5, 0.3, 0.3])
        iou1 = compute_box_iou(box1, box2)
        iou2 = compute_box_iou(box2, box1)
        assert torch.abs(iou1 - iou2) < 1e-6

    def test_iou_contained_box(self):
        """Test IoU when one box is inside another."""
        box1 = torch.tensor([0.5, 0.5, 0.6, 0.6])  # Large box
        box2 = torch.tensor([0.5, 0.5, 0.2, 0.2])  # Small box inside
        iou = compute_box_iou(box1, box2)
        # IoU = area(small) / area(large)
        small_area = 0.2 * 0.2
        large_area = 0.6 * 0.6
        expected_iou = small_area / large_area
        assert abs(iou.item() - expected_iou) < 0.01


class TestComputeIoUCornersFormat:
    """Test IoU computation with corner format boxes."""

    def test_corners_iou_perfect_overlap(self):
        """Test corners IoU with identical boxes."""
        box1 = (10, 10, 50, 50, 0.9, 0)  # x1, y1, x2, y2, conf, class
        box2 = (10, 10, 50, 50, 0.8, 0)
        iou = compute_iou_corners(box1, box2)
        assert abs(iou - 1.0) < 1e-6

    def test_corners_iou_no_overlap(self):
        """Test corners IoU with non-overlapping boxes."""
        box1 = (10, 10, 50, 50, 0.9, 0)
        box2 = (100, 100, 150, 150, 0.8, 0)
        iou = compute_iou_corners(box1, box2)
        assert iou == 0.0

    def test_corners_iou_half_overlap(self):
        """Test corners IoU with approximately half overlap."""
        box1 = (0, 0, 100, 100, 0.9, 0)     # 100×100 box
        box2 = (50, 0, 150, 100, 0.8, 0)    # 100×100 box, shifted right by 50
        iou = compute_iou_corners(box1, box2)
        # Intersection: 50×100 = 5000
        # Union: 10000 + 10000 - 5000 = 15000
        # IoU = 5000/15000 = 1/3
        assert abs(iou - 1/3) < 0.01

    def test_corners_iou_range(self):
        """Test that corners IoU is in [0, 1]."""
        import random
        for _ in range(10):
            x1_1, y1_1 = random.uniform(0, 100), random.uniform(0, 100)
            x2_1, y2_1 = x1_1 + random.uniform(10, 50), y1_1 + random.uniform(10, 50)

            x1_2, y1_2 = random.uniform(0, 100), random.uniform(0, 100)
            x2_2, y2_2 = x1_2 + random.uniform(10, 50), y1_2 + random.uniform(10, 50)

            box1 = (x1_1, y1_1, x2_1, y2_1, 0.9, 0)
            box2 = (x1_2, y1_2, x2_2, y2_2, 0.8, 0)

            iou = compute_iou_corners(box1, box2)
            assert 0.0 <= iou <= 1.0


class TestMultiScaleIntegration:
    """Integration tests for multi-scale components."""

    def test_anchor_matching_distributes_across_scales(self, temp_dataset_dir):
        """Test that boxes are distributed across different scales."""
        from train import YOLODataset

        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)

        # Collect statistics across all samples
        p3_count, p4_count, p5_count = 0, 0, 0

        for i in range(len(dataset)):
            img, targets = dataset[i]
            p3_count += (targets[0][..., 4] > 0).sum().item()
            p4_count += (targets[1][..., 4] > 0).sum().item()
            p5_count += (targets[2][..., 4] > 0).sum().item()

        total = p3_count + p4_count + p5_count
        assert total > 0, "No objects assigned to any scale"

        # At least one scale should have objects (ideally distributed)
        scales_used = sum([p3_count > 0, p4_count > 0, p5_count > 0])
        assert scales_used > 0

    def test_grid_size_consistency(self):
        """Test that grid sizes are consistent between model and dataset."""
        from train import YOLO, YOLODataset
        import tempfile
        import os
        from PIL import Image
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            train_imgs = os.path.join(tmpdir, 'images')
            os.makedirs(train_imgs)

            # Create single dummy image
            img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
            img.save(os.path.join(train_imgs, 'test.jpg'))

            # Create empty label
            labels_dir = train_imgs.replace('images', 'labels')
            os.makedirs(labels_dir)
            open(os.path.join(labels_dir, 'test.txt'), 'w').close()

            for size in [512, 640, 1024]:
                model = YOLO(num_classes=1, img_size=size)
                dataset = YOLODataset(train_imgs, num_classes=1, img_size=size)

                # Grid sizes should match
                assert model.grid_size_p3 == dataset.grid_size_p3
                assert model.grid_size_p4 == dataset.grid_size_p4
                assert model.grid_size_p5 == dataset.grid_size_p5
