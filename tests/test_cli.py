"""
Tests for command-line interface (lines 1347-1556).
"""
import pytest
import sys
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np
import yaml
import torch
from unittest.mock import patch, MagicMock
import subprocess


@pytest.fixture
def temp_cli_dataset():
    """Create temporary dataset for CLI tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create train directory
        train_imgs = Path(tmpdir) / 'train' / 'images'
        train_labels = Path(tmpdir) / 'train' / 'labels'
        train_imgs.mkdir(parents=True)
        train_labels.mkdir(parents=True)

        # Create 10 training images (need at least 9 for k-means with 9 anchors)
        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
            img.save(train_imgs / f'train{i}.jpg')
            with open(train_labels / f'train{i}.txt', 'w') as f:
                # Create varied box sizes for better anchor clustering
                f.write(f"0 0.5 0.5 {0.1 + i*0.02} {0.1 + i*0.02}\n")

        # Create val directory
        val_imgs = Path(tmpdir) / 'val' / 'images'
        val_labels = Path(tmpdir) / 'val' / 'labels'
        val_imgs.mkdir(parents=True)
        val_labels.mkdir(parents=True)

        # Create 1 validation image
        img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        img.save(val_imgs / 'val0.jpg')
        with open(val_labels / 'val0.txt', 'w') as f:
            f.write(f"0 0.5 0.5 0.2 0.2\n")

        # Create YAML config
        yaml_path = Path(tmpdir) / 'dataset.yaml'
        config = {
            'nc': 1,
            'names': ['object'],
            'train': str(train_imgs),
            'val': str(val_imgs)
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)

        yield {
            'yaml': str(yaml_path),
            'train_imgs': str(train_imgs),
            'val_imgs': str(val_imgs),
            'tmpdir': tmpdir
        }


class TestCLI:
    """Test command-line interface."""

    def test_cli_usage_message(self, capsys):
        """Test help/usage output (lines 1539-1556)."""
        # Run train.py with no arguments - should print usage
        result = subprocess.run(
            [sys.executable, 'train.py'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent  # Run from repo root
        )

        # Should print usage information
        assert "Usage:" in result.stdout
        assert "Training:" in result.stdout
        assert "Evaluation:" in result.stdout
        assert "Inference:" in result.stdout
        assert "--img-size" in result.stdout
        assert "--lr" in result.stdout
        assert "--epochs" in result.stdout

    def test_cli_training_mode(self, temp_cli_dataset):
        """Test training mode: python train.py data.yaml (lines 1484-1537)."""
        # Run minimal training (1 epoch)
        result = subprocess.run(
            [sys.executable, 'train.py', temp_cli_dataset['yaml'], '--epochs', '1'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=120  # 2 minute timeout
        )

        # Should complete successfully
        assert result.returncode == 0, f"Training failed: {result.stderr}"

        # Should print training info
        assert "Training YOLO model" in result.stdout
        assert "Number of classes: 1" in result.stdout
        assert "Training images:" in result.stdout
        assert "Validation images:" in result.stdout

        # Should show epoch progress
        assert "Epoch 1:" in result.stdout or "Epoch 0:" in result.stdout
        assert "Loss:" in result.stdout

        # Should save model
        assert "Training complete" in result.stdout
        assert "Model saved to" in result.stdout

    def test_cli_evaluation_mode(self, temp_cli_dataset):
        """Test evaluation mode: python train.py data.yaml model.pt (lines 1446-1483)."""
        # First create a minimal model checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
            from train import YOLO
            model = YOLO(num_classes=1, img_size=640)
            torch.save({
                'model': model.state_dict(),
                'epoch': 0,
                'num_classes': 1,
                'img_size': 640,
                'width_mult': 0.5,
                'depth_mult': 0.33
            }, model_path)

        try:
            # Run evaluation
            result = subprocess.run(
                [sys.executable, 'train.py', temp_cli_dataset['yaml'], model_path],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                timeout=60
            )

            # Should complete successfully
            assert result.returncode == 0, f"Evaluation failed: {result.stderr}"

            # Should print evaluation results
            assert "Evaluating model from" in result.stdout
            assert "Training Set:" in result.stdout
            assert "Validation Set:" in result.stdout
            assert "Precision:" in result.stdout
            assert "Recall:" in result.stdout
            assert "F1 Score:" in result.stdout

        finally:
            os.unlink(model_path)

    def test_cli_inference_mode(self, temp_cli_dataset):
        """Test inference mode: python train.py image.jpg model.pt (lines 1421-1442)."""
        # Create minimal checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
            from train import YOLO
            model = YOLO(num_classes=1, img_size=640)
            torch.save({
                'model': model.state_dict(),
                'epoch': 0,
                'num_classes': 1,
                'img_size': 640,
                'width_mult': 0.5,
                'depth_mult': 0.33
            }, model_path)

        # Get test image
        test_image = list(Path(temp_cli_dataset['train_imgs']).glob('*.jpg'))[0]

        try:
            # Run inference
            result = subprocess.run(
                [sys.executable, 'train.py', str(test_image), model_path],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                timeout=30
            )

            # Should complete successfully
            assert result.returncode == 0, f"Inference failed: {result.stderr}"

            # Should print inference info
            assert "Running inference on" in result.stdout
            assert "Detected" in result.stdout or "No objects detected" in result.stdout

        finally:
            os.unlink(model_path)

    def test_cli_inspect_mode(self):
        """Test inspect mode: python train.py model.pt (lines 1400-1419)."""
        # Create minimal checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
            from train import YOLO
            model = YOLO(num_classes=1, img_size=640)
            torch.save({
                'model': model.state_dict(),
                'epoch': 0,
                'num_classes': 1,
                'img_size': 640,
                'width_mult': 0.5,
                'depth_mult': 0.33
            }, model_path)

        try:
            # Run inspection
            result = subprocess.run(
                [sys.executable, 'train.py', model_path],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                timeout=30
            )

            # Should complete successfully
            assert result.returncode == 0, f"Inspection failed: {result.stderr}"

            # Should print model info
            assert "Model loaded from" in result.stdout
            assert "Number of classes:" in result.stdout
            assert "Image size:" in result.stdout
            assert "Model architecture:" in result.stdout
            assert "Total parameters:" in result.stdout

        finally:
            os.unlink(model_path)

    def test_cli_compute_anchors_mode(self, temp_cli_dataset):
        """Test --compute-anchors mode (lines 1371-1379)."""
        # Run anchor computation
        result = subprocess.run(
            [sys.executable, 'train.py', temp_cli_dataset['yaml'], '--compute-anchors'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=60
        )

        # Should complete successfully
        assert result.returncode == 0, f"Anchor computation failed: {result.stderr}"

        # Should print anchor computation info
        assert "Computing optimal anchors" in result.stdout
        assert "Loaded" in result.stdout
        assert "boxes" in result.stdout
        assert "Running k-means" in result.stdout
        assert "Optimal anchors" in result.stdout
        assert "P3" in result.stdout
        assert "P4" in result.stdout
        assert "P5" in result.stdout

    def test_cli_custom_img_size(self, temp_cli_dataset):
        """Test --img-size flag."""
        # Run training with custom img_size
        result = subprocess.run(
            [sys.executable, 'train.py', temp_cli_dataset['yaml'],
             '--img-size', '512', '--epochs', '1'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=120
        )

        # Should complete successfully
        assert result.returncode == 0, f"Custom img_size failed: {result.stderr}"
        assert "Training YOLO model" in result.stdout

    def test_cli_custom_lr_params(self, temp_cli_dataset):
        """Test custom LR parameters (--lr, --min-lr, --warmup-epochs)."""
        # Run training with custom LR schedule
        result = subprocess.run(
            [sys.executable, 'train.py', temp_cli_dataset['yaml'],
             '--lr', '0.02', '--min-lr', '0.0001', '--warmup-epochs', '2',
             '--epochs', '1'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=120
        )

        # Should complete successfully
        assert result.returncode == 0, f"Custom LR params failed: {result.stderr}"

        # Should show LR schedule info
        assert "Learning Rate Schedule:" in result.stdout
        assert "Initial LR: 0.02" in result.stdout
        assert "Minimum LR: 0.0001" in result.stdout
        assert "Warmup epochs: 2" in result.stdout

    def test_cli_model_size_variants(self, temp_cli_dataset):
        """Test --size flag for different model variants (n/s/m/l/x)."""
        # Test with nano model
        result = subprocess.run(
            [sys.executable, 'train.py', temp_cli_dataset['yaml'],
             '--size', 'n', '--epochs', '1'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=120
        )

        # Should complete successfully
        assert result.returncode == 0, f"Model size variant failed: {result.stderr}"
        assert "Creating YOLOv5N" in result.stdout

    def test_cli_compute_anchors_no_yaml_error(self):
        """Test --compute-anchors requires YAML file."""
        # Run without YAML file
        result = subprocess.run(
            [sys.executable, 'train.py', '--compute-anchors'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=10
        )

        # Should fail with error message
        assert result.returncode == 1
        assert "ERROR" in result.stderr or "ERROR" in result.stdout
        assert "requires a dataset YAML file" in result.stderr or "requires a dataset YAML file" in result.stdout


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_full_training_pipeline(self, temp_cli_dataset):
        """Test full train → eval → inference pipeline."""
        # 1. Train model (1 epoch)
        result_train = subprocess.run(
            [sys.executable, 'train.py', temp_cli_dataset['yaml'], '--epochs', '1'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=120
        )

        assert result_train.returncode == 0

        # Find saved model file
        model_files = list(Path.cwd().glob('yolo_*.pt'))
        assert len(model_files) > 0, "No model file saved"
        model_path = str(model_files[-1])  # Get most recent

        try:
            # 2. Evaluate model
            result_eval = subprocess.run(
                [sys.executable, 'train.py', temp_cli_dataset['yaml'], model_path],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                timeout=60
            )

            assert result_eval.returncode == 0
            assert "Validation Set:" in result_eval.stdout

            # 3. Run inference
            test_image = list(Path(temp_cli_dataset['train_imgs']).glob('*.jpg'))[0]
            result_infer = subprocess.run(
                [sys.executable, 'train.py', str(test_image), model_path],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                timeout=30
            )

            assert result_infer.returncode == 0
            assert "Running inference" in result_infer.stdout

        finally:
            # Cleanup
            if Path(model_path).exists():
                Path(model_path).unlink()
