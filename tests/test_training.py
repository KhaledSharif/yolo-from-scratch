"""
Tests for training functions (train_epoch, LR scheduler).
"""
import pytest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from train import train_epoch, get_lr_lambda, YOLO, YOLODataset, yolo_collate_fn
import numpy as np


class TestTrainEpoch:
    """Test train_epoch function."""

    def test_train_epoch_basic(self, temp_dataset_dir, device):
        """Test basic train_epoch execution (lines 883-920)."""
        model = YOLO(num_classes=1, img_size=640).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Create small dataset
        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)
        loader = DataLoader(dataset, batch_size=2, collate_fn=yolo_collate_fn)

        # Run one training epoch
        loss, bbox_loss, obj_loss, cls_loss = train_epoch(
            model, loader, optimizer, device, num_classes=1
        )

        # Verify losses are computed
        assert isinstance(loss, float)
        assert isinstance(bbox_loss, float)
        assert isinstance(obj_loss, float)
        assert isinstance(cls_loss, float)

        # Losses should be non-negative
        assert loss >= 0
        assert bbox_loss >= 0
        assert obj_loss >= 0
        assert cls_loss >= 0

    def test_train_epoch_loss_computation(self, temp_dataset_dir, device):
        """Test multi-scale loss aggregation."""
        model = YOLO(num_classes=1, img_size=640).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)
        loader = DataLoader(dataset, batch_size=2, collate_fn=yolo_collate_fn)

        loss, bbox_loss, obj_loss, cls_loss = train_epoch(
            model, loader, optimizer, device, num_classes=1
        )

        # Total loss should be aggregated from all scales
        # Expected: loss = 5*bbox + weighted_obj + cls
        # All components should contribute
        assert loss > 0
        assert obj_loss > 0  # Objectness loss always present

    def test_train_epoch_gradient_flow(self, temp_dataset_dir, device):
        """Test that gradients flow and optimizer updates parameters."""
        model = YOLO(num_classes=1, img_size=640).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Store initial parameter values
        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)
        loader = DataLoader(dataset, batch_size=2, collate_fn=yolo_collate_fn)

        # Run training
        train_epoch(model, loader, optimizer, device, num_classes=1)

        # Verify at least some parameters changed
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_params[name], atol=1e-7):
                params_changed = True
                break

        assert params_changed, "No parameters were updated during training"

    def test_train_epoch_gradient_clipping(self, temp_dataset_dir, device):
        """Test gradient clipping with max_norm=10.0."""
        model = YOLO(num_classes=1, img_size=640).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        dataset = YOLODataset(temp_dataset_dir, num_classes=1, img_size=640)
        loader = DataLoader(dataset, batch_size=2, collate_fn=yolo_collate_fn)

        # Run training (gradient clipping happens inside train_epoch)
        loss, _, _, _ = train_epoch(model, loader, optimizer, device, num_classes=1)

        # If clipping works, training should complete without exploding gradients
        assert not np.isnan(loss)
        assert not np.isinf(loss)
        assert loss < 1e6  # Loss shouldn't explode

    def test_train_epoch_multiclass(self, temp_dataset_dir, device):
        """Test training with multiple classes."""
        model = YOLO(num_classes=3, img_size=640).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        dataset = YOLODataset(temp_dataset_dir, num_classes=3, img_size=640)
        loader = DataLoader(dataset, batch_size=2, collate_fn=yolo_collate_fn)

        loss, bbox_loss, obj_loss, cls_loss = train_epoch(
            model, loader, optimizer, device, num_classes=3
        )

        # All losses should be valid for multi-class
        assert loss >= 0
        assert bbox_loss >= 0
        assert obj_loss >= 0
        assert cls_loss >= 0


class TestLRScheduler:
    """Test learning rate scheduler function."""

    def test_lr_scheduler_warmup(self):
        """Test linear warmup phase (lines 1046-1056)."""
        warmup_epochs = 3
        total_epochs = 100
        initial_lr = 1e-2
        min_lr = 1e-4
        warmup_start_lr = 1e-6

        lr_lambda_fn = get_lr_lambda(
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            initial_lr=initial_lr,
            min_lr=min_lr,
            warmup_start_lr=warmup_start_lr
        )

        # Test warmup phase (epochs 0, 1, 2)
        for epoch in range(warmup_epochs):
            multiplier = lr_lambda_fn(epoch)
            expected_lr = (warmup_start_lr + (initial_lr - warmup_start_lr) * epoch / warmup_epochs) / initial_lr
            assert abs(multiplier - expected_lr) < 1e-6, f"Epoch {epoch}: expected {expected_lr}, got {multiplier}"

        # At end of warmup (epoch 3), should reach initial_lr
        epoch3_multiplier = lr_lambda_fn(3)
        assert abs(epoch3_multiplier - 1.0) < 0.01, "Should reach 1.0 multiplier after warmup"

    def test_lr_scheduler_cosine(self):
        """Test cosine annealing phase."""
        warmup_epochs = 3
        total_epochs = 100
        initial_lr = 1e-2
        min_lr = 1e-4

        lr_lambda_fn = get_lr_lambda(
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            initial_lr=initial_lr,
            min_lr=min_lr
        )

        # Test cosine phase (after warmup)
        # At epoch 50 (middle), should be somewhere between initial and min
        epoch50_multiplier = lr_lambda_fn(50)
        epoch50_lr = epoch50_multiplier * initial_lr
        assert min_lr < epoch50_lr < initial_lr, "Mid-training LR should be between min and initial"

        # At final epoch, should approach min_lr
        final_epoch = total_epochs - 1
        final_multiplier = lr_lambda_fn(final_epoch)
        final_lr = final_multiplier * initial_lr
        assert abs(final_lr - min_lr) < 1e-3, f"Final LR should be close to min_lr: {final_lr} vs {min_lr}"

    def test_lr_scheduler_full_schedule(self):
        """Test complete schedule over 100 epochs."""
        warmup_epochs = 3
        total_epochs = 100
        initial_lr = 1e-2
        min_lr = 1e-4

        lr_lambda_fn = get_lr_lambda(
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            initial_lr=initial_lr,
            min_lr=min_lr
        )

        # Collect LR values over all epochs
        lr_values = [lr_lambda_fn(e) * initial_lr for e in range(total_epochs)]

        # Verify warmup: LR increases
        for i in range(warmup_epochs - 1):
            assert lr_values[i] < lr_values[i+1], f"Warmup: LR should increase at epoch {i}"

        # Verify cosine decay: LR generally decreases (with some tolerance for numerical issues)
        # Check that final LR < mid LR < post-warmup LR
        post_warmup_lr = lr_values[warmup_epochs]
        mid_lr = lr_values[total_epochs // 2]
        final_lr = lr_values[-1]

        assert final_lr < mid_lr < post_warmup_lr, \
            f"Cosine decay should monotonically decrease: {post_warmup_lr} > {mid_lr} > {final_lr}"

    def test_lr_scheduler_with_optimizer(self):
        """Test scheduler integration with PyTorch optimizer."""
        model = YOLO(num_classes=1, img_size=640)
        optimizer = optim.Adam(model.parameters(), lr=1e-2)

        lr_lambda_fn = get_lr_lambda(
            warmup_epochs=3,
            total_epochs=100,
            initial_lr=1e-2,
            min_lr=1e-4
        )

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fn)

        # Verify initial LR
        initial_lr = optimizer.param_groups[0]['lr']
        assert abs(initial_lr - 1e-6) < 1e-7, f"Initial LR should be ~1e-6 (warmup start), got {initial_lr}"

        # Step through a few epochs
        # Note: In PyTorch 1.1+, optimizer.step() should be called before scheduler.step()
        for epoch in range(10):
            optimizer.step()  # Dummy step to satisfy PyTorch scheduler requirements
            scheduler.step()

        # After 10 epochs, should be in cosine phase with LR < initial
        current_lr = optimizer.param_groups[0]['lr']
        assert current_lr > 1e-6, "LR should increase from warmup start"
        assert current_lr <= 1e-2, "LR shouldn't exceed initial LR"

    def test_lr_scheduler_custom_params(self):
        """Test scheduler with custom parameters."""
        # Custom: shorter warmup, different LR range
        warmup_epochs = 5
        total_epochs = 50
        initial_lr = 2e-2
        min_lr = 5e-5

        lr_lambda_fn = get_lr_lambda(
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            initial_lr=initial_lr,
            min_lr=min_lr
        )

        # Test warmup end
        epoch5_multiplier = lr_lambda_fn(5)
        epoch5_lr = epoch5_multiplier * initial_lr
        assert abs(epoch5_lr - initial_lr) < 1e-3, "Should reach initial_lr after custom warmup"

        # Test final epoch
        final_multiplier = lr_lambda_fn(total_epochs - 1)
        final_lr = final_multiplier * initial_lr
        assert abs(final_lr - min_lr) < 1e-3, "Should reach custom min_lr at end"
