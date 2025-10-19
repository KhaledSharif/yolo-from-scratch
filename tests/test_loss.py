"""
Tests for loss functions (CIoU, YOLO loss, multi-scale loss).
"""
import pytest
import torch
from train import ciou_loss, yolo_loss, yolo_loss_multiscale, decode_predictions


class TestCIoULoss:
    """Test Complete IoU loss function."""

    def test_ciou_perfect_overlap(self):
        """Test CIoU loss with perfect overlap (should be 0)."""
        pred_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
        target_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
        loss = ciou_loss(pred_boxes, target_boxes)
        assert loss.item() < 0.01  # Should be very close to 0

    def test_ciou_no_overlap(self):
        """Test CIoU loss with no overlap (should be high)."""
        pred_boxes = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
        target_boxes = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
        loss = ciou_loss(pred_boxes, target_boxes)
        assert loss.item() > 1.0  # Should be high

    def test_ciou_partial_overlap(self):
        """Test CIoU loss with partial overlap."""
        pred_boxes = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
        target_boxes = torch.tensor([[0.6, 0.6, 0.3, 0.3]])
        loss = ciou_loss(pred_boxes, target_boxes)
        assert 0.0 < loss.item() < 1.0

    def test_ciou_batch(self):
        """Test CIoU loss with multiple boxes."""
        pred_boxes = torch.tensor([
            [0.5, 0.5, 0.2, 0.3],
            [0.3, 0.4, 0.15, 0.25],
            [0.7, 0.6, 0.25, 0.2]
        ])
        target_boxes = torch.tensor([
            [0.5, 0.5, 0.2, 0.3],
            [0.35, 0.45, 0.15, 0.25],
            [0.7, 0.6, 0.25, 0.2]
        ])
        loss = ciou_loss(pred_boxes, target_boxes)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_ciou_aspect_ratio_penalty(self):
        """Test that CIoU penalizes aspect ratio differences."""
        pred_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4]])  # tall box
        target_boxes = torch.tensor([[0.5, 0.5, 0.4, 0.2]])  # wide box
        loss = ciou_loss(pred_boxes, target_boxes)
        assert loss.item() > 0.5  # Should have significant penalty


class TestYOLOLoss:
    """Test single-scale YOLO loss function."""

    def test_yolo_loss_with_objects(self):
        """Test YOLO loss with objects present."""
        batch_size, grid_size, num_anchors = 2, 20, 3
        num_classes = 1

        # Create random predictions and targets
        predictions = torch.randn(batch_size, grid_size, grid_size, num_anchors, 5 + num_classes)
        targets = torch.zeros(batch_size, grid_size, grid_size, num_anchors, 5 + num_classes)

        # Add one object to targets
        targets[0, 10, 10, 0, :5] = torch.tensor([0.5, 0.5, 0.2, 0.3, 1.0])
        targets[0, 10, 10, 0, 5] = 1.0  # Class probability

        anchors = torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32)

        total_loss, bbox_loss, obj_loss, class_loss = yolo_loss(
            predictions, targets, anchors, num_classes
        )

        # Check all losses are valid tensors
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(bbox_loss, torch.Tensor)
        assert isinstance(obj_loss, torch.Tensor)
        assert isinstance(class_loss, torch.Tensor)

        # Check no NaN or Inf
        assert not torch.isnan(total_loss)
        assert not torch.isnan(bbox_loss)
        assert not torch.isnan(obj_loss)
        assert not torch.isnan(class_loss)

    def test_yolo_loss_no_objects(self):
        """Test YOLO loss with no objects (all background)."""
        batch_size, grid_size, num_anchors = 2, 20, 3
        num_classes = 1

        predictions = torch.randn(batch_size, grid_size, grid_size, num_anchors, 5 + num_classes)
        targets = torch.zeros(batch_size, grid_size, grid_size, num_anchors, 5 + num_classes)

        anchors = torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32)

        total_loss, bbox_loss, obj_loss, class_loss = yolo_loss(
            predictions, targets, anchors, num_classes
        )

        # Bbox and class loss should be 0 (no objects)
        assert bbox_loss.item() == 0.0
        assert class_loss.item() == 0.0
        # Only objectness loss should be non-zero
        assert obj_loss.item() > 0.0

    def test_yolo_loss_weights(self):
        """Test that bbox loss is weighted 5× in total loss."""
        batch_size, grid_size, num_anchors = 2, 20, 3
        num_classes = 1

        predictions = torch.randn(batch_size, grid_size, grid_size, num_anchors, 5 + num_classes)
        targets = torch.zeros(batch_size, grid_size, grid_size, num_anchors, 5 + num_classes)
        targets[0, 10, 10, 0, :5] = torch.tensor([0.5, 0.5, 0.2, 0.3, 1.0])
        targets[0, 10, 10, 0, 5] = 1.0

        anchors = torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32)

        total_loss, bbox_loss, obj_loss, class_loss = yolo_loss(
            predictions, targets, anchors, num_classes
        )

        # Total loss = 0.05*bbox + 1.0*obj + 0.5*class (YOLOv5 defaults)
        expected = 0.05 * bbox_loss + 1.0 * obj_loss + 0.5 * class_loss
        assert torch.allclose(total_loss, expected, atol=1e-5)


class TestMultiScaleLoss:
    """Test multi-scale YOLO loss function."""

    def test_multiscale_loss_aggregation(self):
        """Test that multi-scale loss properly aggregates across scales."""
        batch_size, num_classes = 2, 1

        # Create predictions and targets for 3 scales
        preds_p3 = torch.randn(batch_size, 80, 80, 3, 6)
        preds_p4 = torch.randn(batch_size, 40, 40, 3, 6)
        preds_p5 = torch.randn(batch_size, 20, 20, 3, 6)
        predictions = [preds_p3, preds_p4, preds_p5]

        targets_p3 = torch.zeros(batch_size, 80, 80, 3, 6)
        targets_p4 = torch.zeros(batch_size, 40, 40, 3, 6)
        targets_p5 = torch.zeros(batch_size, 20, 20, 3, 6)
        targets = [targets_p3, targets_p4, targets_p5]

        anchors_list = [
            torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32),
            torch.tensor([[30, 61], [62, 45], [59, 119]], dtype=torch.float32),
            torch.tensor([[116, 90], [156, 198], [373, 326]], dtype=torch.float32)
        ]

        total_loss, bbox_loss, obj_loss, class_loss = yolo_loss_multiscale(
            predictions, targets, anchors_list, num_classes
        )

        # All losses should be valid
        assert isinstance(total_loss, (torch.Tensor, float))
        # Check for NaN without copying tensor
        if isinstance(total_loss, torch.Tensor):
            assert not torch.isnan(total_loss)
        else:
            import math
            assert not math.isnan(total_loss)

    def test_multiscale_loss_with_objects_at_different_scales(self):
        """Test multi-scale loss with objects at different scales."""
        batch_size, num_classes = 2, 1

        # Create predictions and targets
        preds_p3 = torch.randn(batch_size, 80, 80, 3, 6)
        preds_p4 = torch.randn(batch_size, 40, 40, 3, 6)
        preds_p5 = torch.randn(batch_size, 20, 20, 3, 6)
        predictions = [preds_p3, preds_p4, preds_p5]

        targets_p3 = torch.zeros(batch_size, 80, 80, 3, 6)
        targets_p4 = torch.zeros(batch_size, 40, 40, 3, 6)
        targets_p5 = torch.zeros(batch_size, 20, 20, 3, 6)

        # Add objects to different scales
        targets_p3[0, 40, 40, 0, :5] = torch.tensor([0.5, 0.5, 0.1, 0.1, 1.0])  # Small object
        targets_p3[0, 40, 40, 0, 5] = 1.0
        targets_p4[0, 20, 20, 1, :5] = torch.tensor([0.5, 0.5, 0.3, 0.3, 1.0])  # Medium object
        targets_p4[0, 20, 20, 1, 5] = 1.0
        targets_p5[1, 10, 10, 2, :5] = torch.tensor([0.5, 0.5, 0.5, 0.5, 1.0])  # Large object
        targets_p5[1, 10, 10, 2, 5] = 1.0

        targets = [targets_p3, targets_p4, targets_p5]

        anchors_list = [
            torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32),
            torch.tensor([[30, 61], [62, 45], [59, 119]], dtype=torch.float32),
            torch.tensor([[116, 90], [156, 198], [373, 326]], dtype=torch.float32)
        ]

        total_loss, bbox_loss, obj_loss, class_loss = yolo_loss_multiscale(
            predictions, targets, anchors_list, num_classes
        )

        # Bbox loss should be non-zero (we have objects)
        assert bbox_loss > 0
        assert obj_loss > 0
        assert class_loss >= 0

    def test_multiscale_loss_gradient_flow(self):
        """Test that gradients flow through multi-scale loss."""
        batch_size, num_classes = 2, 1

        # Create predictions with requires_grad
        preds_p3 = torch.randn(batch_size, 80, 80, 3, 6, requires_grad=True)
        preds_p4 = torch.randn(batch_size, 40, 40, 3, 6, requires_grad=True)
        preds_p5 = torch.randn(batch_size, 20, 20, 3, 6, requires_grad=True)
        predictions = [preds_p3, preds_p4, preds_p5]

        # Create targets with objects
        targets_p3 = torch.zeros(batch_size, 80, 80, 3, 6)
        targets_p4 = torch.zeros(batch_size, 40, 40, 3, 6)
        targets_p5 = torch.zeros(batch_size, 20, 20, 3, 6)
        targets_p3[0, 40, 40, 0, :5] = torch.tensor([0.5, 0.5, 0.2, 0.3, 1.0])
        targets_p3[0, 40, 40, 0, 5] = 1.0
        targets = [targets_p3, targets_p4, targets_p5]

        anchors_list = [
            torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32),
            torch.tensor([[30, 61], [62, 45], [59, 119]], dtype=torch.float32),
            torch.tensor([[116, 90], [156, 198], [373, 326]], dtype=torch.float32)
        ]

        total_loss, _, _, _ = yolo_loss_multiscale(
            predictions, targets, anchors_list, num_classes
        )

        # Backward pass should work
        total_loss.backward()

        # Gradients should exist for all predictions
        assert preds_p3.grad is not None
        assert preds_p4.grad is not None
        assert preds_p5.grad is not None


class TestDecodePredictions:
    """Test prediction decoding function."""

    def test_decode_predictions_shape(self):
        """Test that decoding preserves shape."""
        raw_preds = torch.randn(2, 20, 20, 3, 6)
        anchors = torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32)

        decoded = decode_predictions(raw_preds, anchors, img_size=640)

        assert decoded.shape == raw_preds.shape

    def test_decode_predictions_center_range(self):
        """Test that decoded centers are in valid range."""
        # Create predictions with zeros (should decode to grid cell centers)
        raw_preds = torch.zeros(1, 20, 20, 3, 6)
        anchors = torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32)

        decoded = decode_predictions(raw_preds, anchors, img_size=640)

        # Decoded x and y should be in [0, 1]
        assert (decoded[..., 0] >= 0).all() and (decoded[..., 0] <= 1).all()
        assert (decoded[..., 1] >= 0).all() and (decoded[..., 1] <= 1).all()

    def test_decode_predictions_dimension_range(self):
        """Test that decoded dimensions are positive."""
        raw_preds = torch.randn(1, 20, 20, 3, 6)
        anchors = torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32)

        decoded = decode_predictions(raw_preds, anchors, img_size=640)

        # Width and height should be positive
        assert (decoded[..., 2] > 0).all()
        assert (decoded[..., 3] > 0).all()

    def test_decode_predictions_objectness_unchanged(self):
        """Test that objectness and class logits remain unchanged."""
        raw_preds = torch.randn(1, 20, 20, 3, 6)
        anchors = torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32)

        decoded = decode_predictions(raw_preds, anchors, img_size=640)

        # Objectness and class predictions should be unchanged (still logits)
        assert torch.allclose(decoded[..., 4:], raw_preds[..., 4:])

    def test_decode_predictions_different_grid_sizes(self):
        """Test decoding with different grid sizes."""
        anchors = torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32)

        for grid_size in [20, 40, 80]:
            raw_preds = torch.randn(1, grid_size, grid_size, 3, 6)
            decoded = decode_predictions(raw_preds, anchors, img_size=640)

            assert decoded.shape == raw_preds.shape
            # YOLOv5 decoding allows centers to extend ~0.5 cells outside [0,1] for better boundary predictions
            assert (decoded[..., 0] >= -0.1).all() and (decoded[..., 0] <= 1.1).all()
            assert (decoded[..., 1] >= -0.1).all() and (decoded[..., 1] <= 1.1).all()

    def test_decode_predictions_gradient_flow(self):
        """Test that gradients flow through decoding."""
        raw_preds = torch.randn(1, 20, 20, 3, 6, requires_grad=True)
        anchors = torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32)

        decoded = decode_predictions(raw_preds, anchors, img_size=640)

        # Take mean and backward
        loss = decoded[..., :4].mean()
        loss.backward()

        # Gradient should exist
        assert raw_preds.grad is not None
