"""
Pytest configuration and shared fixtures for YOLO tests.
"""
import pytest
import torch
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np
import sys

# Add parent directory to path to import train module
sys.path.insert(0, str(Path(__file__).parent.parent))

from train import YOLO, YOLODataset


@pytest.fixture
def device():
    """Get torch device (CPU for tests)."""
    return torch.device('cpu')


@pytest.fixture
def img_size():
    """Default image size for testing."""
    return 640


@pytest.fixture
def num_classes():
    """Default number of classes for testing."""
    return 1


@pytest.fixture
def batch_size():
    """Default batch size for testing."""
    return 2


@pytest.fixture
def dummy_model(num_classes, img_size):
    """Create a YOLO model for testing."""
    return YOLO(num_classes=num_classes, img_size=img_size)


@pytest.fixture
def dummy_input(batch_size, img_size):
    """Create dummy input tensor."""
    return torch.randn(batch_size, 3, img_size, img_size)


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary dataset directory with dummy images and labels."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directory structure
        train_imgs = Path(tmpdir) / 'train' / 'images'
        train_labels = Path(tmpdir) / 'train' / 'labels'
        train_imgs.mkdir(parents=True)
        train_labels.mkdir(parents=True)

        # Create 5 dummy images with labels
        for i in range(5):
            # Create random image
            img = Image.fromarray(
                np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            )
            img_path = train_imgs / f'img{i}.jpg'
            img.save(img_path)

            # Create label file with random boxes
            label_path = train_labels / f'img{i}.txt'
            with open(label_path, 'w') as f:
                # Add 1-3 random boxes per image
                num_boxes = np.random.randint(1, 4)
                for _ in range(num_boxes):
                    class_id = 0
                    x_center = np.random.uniform(0.2, 0.8)
                    y_center = np.random.uniform(0.2, 0.8)
                    width = np.random.uniform(0.1, 0.3)
                    height = np.random.uniform(0.1, 0.3)
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        yield str(train_imgs)


@pytest.fixture
def anchors_p3():
    """Default P3 anchors."""
    return [[10, 13], [16, 30], [33, 23]]


@pytest.fixture
def anchors_p4():
    """Default P4 anchors."""
    return [[30, 61], [62, 45], [59, 119]]


@pytest.fixture
def anchors_p5():
    """Default P5 anchors."""
    return [[116, 90], [156, 198], [373, 326]]


@pytest.fixture
def all_anchors(anchors_p3, anchors_p4, anchors_p5):
    """All multi-scale anchors as list."""
    return [anchors_p3, anchors_p4, anchors_p5]
