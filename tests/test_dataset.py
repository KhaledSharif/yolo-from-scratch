"""
Tests for YOLODataset and multi-scale target generation.
"""
import pytest
import torch
from torch.utils.data import DataLoader
from train import YOLODataset, yolo_collate_fn


class TestYOLODataset:
    """Test YOLODataset class."""

    def test_dataset_initialization(self, temp_dataset_dir, num_classes, img_size):
        """Test dataset can be initialized."""
        dataset = YOLODataset(temp_dataset_dir, num_classes=num_classes, img_size=img_size)
        assert len(dataset) == 5  # Created 5 images in fixture
        assert dataset.num_classes == num_classes
        assert dataset.img_size == img_size

    def test_grid_sizes(self, temp_dataset_dir):
        """Test grid sizes are calculated correctly for all scales."""
        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)
        assert dataset.grid_size_p3 == 80
        assert dataset.grid_size_p4 == 40
        assert dataset.grid_size_p5 == 20
        assert dataset.grid_sizes == [80, 40, 20]
        assert dataset.strides == [8, 16, 32]

    def test_multi_scale_anchors(self, temp_dataset_dir):
        """Test multi-scale anchors are initialized."""
        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)
        assert len(dataset.anchors) == 3
        for anchors in dataset.anchors:
            assert anchors.shape == (3, 2)  # 3 anchors per scale

    def test_getitem_returns_correct_format(self, temp_dataset_dir, num_classes, img_size):
        """Test __getitem__ returns image and multi-scale targets."""
        dataset = YOLODataset(temp_dataset_dir, num_classes=num_classes, img_size=img_size)
        img, targets = dataset[0]

        # Check image
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, img_size, img_size)
        assert img.min() >= 0 and img.max() <= 1  # Normalized

        # Check targets (should be list of 3 tensors)
        assert isinstance(targets, list)
        assert len(targets) == 3

    def test_target_shapes(self, temp_dataset_dir):
        """Test target tensor shapes for all scales."""
        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)
        img, targets = dataset[0]

        # P3: 80×80 grid, 3 anchors, 6 outputs (5+1 class)
        assert targets[0].shape == (80, 80, 3, 6)
        # P4: 40×40 grid
        assert targets[1].shape == (40, 40, 3, 6)
        # P5: 20×20 grid
        assert targets[2].shape == (20, 20, 3, 6)

    def test_target_shapes_multiclass(self, temp_dataset_dir):
        """Test target shapes with multiple classes."""
        dataset = YOLODataset(temp_dataset_dir, num_classes=3, img_size=640)
        img, targets = dataset[0]

        # 5 + 3 classes = 8 output dims
        assert targets[0].shape == (80, 80, 3, 8)
        assert targets[1].shape == (40, 40, 3, 8)
        assert targets[2].shape == (20, 20, 3, 8)

    def test_target_contains_objects(self, temp_dataset_dir):
        """Test that targets contain assigned objects."""
        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)
        img, targets = dataset[0]

        # At least one scale should have objects (objectness > 0)
        total_objects = 0
        for target in targets:
            num_objects = (target[..., 4] > 0).sum().item()
            total_objects += num_objects

        assert total_objects > 0, "No objects assigned to any scale"

    def test_anchor_iou_computation(self, temp_dataset_dir):
        """Test anchor IoU computation."""
        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)
        box_wh = torch.tensor([50.0, 60.0])  # 50×60 pixel box
        anchors = torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32)

        ious = dataset.compute_anchor_iou(box_wh, anchors)

        assert ious.shape == (3,)
        assert (ious >= 0).all() and (ious <= 1).all()
        # Larger anchor should have higher IoU with larger box
        assert ious[2] > ious[0]

    def test_different_img_sizes(self, temp_dataset_dir):
        """Test dataset with different image sizes."""
        for size in [416, 512, 640, 1024]:
            dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=size)
            img, targets = dataset[0]

            assert img.shape == (3, size, size)
            assert targets[0].shape[0] == size // 8  # P3 grid size
            assert targets[1].shape[0] == size // 16  # P4 grid size
            assert targets[2].shape[0] == size // 32  # P5 grid size


class TestCollateFn:
    """Test custom collate function."""

    def test_collate_batches_images(self, temp_dataset_dir):
        """Test collate function properly batches images."""
        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)
        loader = DataLoader(dataset, batch_size=3, collate_fn=yolo_collate_fn)

        imgs, targets = next(iter(loader))

        # Images should be stacked into a batch tensor
        assert isinstance(imgs, torch.Tensor)
        assert imgs.shape == (3, 3, 640, 640)

    def test_collate_preserves_target_structure(self, temp_dataset_dir):
        """Test collate function preserves multi-scale target structure."""
        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)
        loader = DataLoader(dataset, batch_size=3, collate_fn=yolo_collate_fn)

        imgs, targets = next(iter(loader))

        # Targets should be list of lists
        assert isinstance(targets, list)
        assert len(targets) == 3  # Batch size

        # Each sample should have 3 scale targets
        for sample_targets in targets:
            assert isinstance(sample_targets, list)
            assert len(sample_targets) == 3

    def test_collate_allows_stacking_by_scale(self, temp_dataset_dir):
        """Test that collated targets can be stacked by scale."""
        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)
        loader = DataLoader(dataset, batch_size=4, collate_fn=yolo_collate_fn)

        imgs, targets = next(iter(loader))

        # Should be able to stack targets by scale
        targets_p3 = torch.stack([t[0] for t in targets])
        targets_p4 = torch.stack([t[1] for t in targets])
        targets_p5 = torch.stack([t[2] for t in targets])

        assert targets_p3.shape == (4, 80, 80, 3, 6)
        assert targets_p4.shape == (4, 40, 40, 3, 6)
        assert targets_p5.shape == (4, 20, 20, 3, 6)

    def test_collate_with_different_resolutions(self, temp_dataset_dir):
        """Test collate function with different image resolutions."""
        for size in [512, 1024]:
            dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=size)
            loader = DataLoader(dataset, batch_size=2, collate_fn=yolo_collate_fn)

            imgs, targets = next(iter(loader))

            assert imgs.shape == (2, 3, size, size)
            targets_p3 = torch.stack([t[0] for t in targets])
            assert targets_p3.shape[1] == size // 8
