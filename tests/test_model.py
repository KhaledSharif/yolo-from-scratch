"""
Tests for YOLO FPN model architecture and forward pass.
"""
import pytest
import torch
from train import YOLO, ConvBlock, C3, Bottleneck, SPPF


class TestConvBlock:
    """Test ConvBlock building block."""

    def test_conv_block_forward(self):
        """Test ConvBlock forward pass."""
        block = ConvBlock(64, 128, kernel_size=3, stride=1, padding=1)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 128, 32, 32)

    def test_conv_block_stride(self):
        """Test ConvBlock with stride=2."""
        block = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 128, 16, 16)


class TestBottleneck:
    """Test Bottleneck residual block."""

    def test_bottleneck_with_shortcut(self):
        """Test Bottleneck with shortcut connection."""
        block = Bottleneck(128, 128, shortcut=True)
        x = torch.randn(2, 128, 32, 32)
        out = block(x)
        assert out.shape == (2, 128, 32, 32)

    def test_bottleneck_without_shortcut(self):
        """Test Bottleneck without shortcut."""
        block = Bottleneck(128, 128, shortcut=False)
        x = torch.randn(2, 128, 32, 32)
        out = block(x)
        assert out.shape == (2, 128, 32, 32)

    def test_bottleneck_different_channels(self):
        """Test Bottleneck with different input/output channels."""
        block = Bottleneck(128, 256, shortcut=False)
        x = torch.randn(2, 128, 32, 32)
        out = block(x)
        assert out.shape == (2, 256, 32, 32)


class TestC3:
    """Test C3 CSP Bottleneck module."""

    def test_c3_forward(self):
        """Test C3 forward pass."""
        block = C3(256, 128, n=1)
        x = torch.randn(2, 256, 32, 32)
        out = block(x)
        assert out.shape == (2, 128, 32, 32)

    def test_c3_multiple_bottlenecks(self):
        """Test C3 with multiple bottleneck blocks."""
        block = C3(256, 256, n=3)
        x = torch.randn(2, 256, 32, 32)
        out = block(x)
        assert out.shape == (2, 256, 32, 32)


class TestSPPF:
    """Test SPPF module."""

    def test_sppf_forward(self):
        """Test SPPF forward pass."""
        sppf = SPPF(512, 512, kernel_size=5)
        x = torch.randn(2, 512, 20, 20)
        out = sppf(x)
        assert out.shape == (2, 512, 20, 20)

    def test_sppf_channel_reduction(self):
        """Test SPPF with channel reduction."""
        sppf = SPPF(512, 256, kernel_size=5)
        x = torch.randn(2, 512, 20, 20)
        out = sppf(x)
        assert out.shape == (2, 256, 20, 20)


class TestYOLOModel:
    """Test YOLO FPN model."""

    def test_model_initialization(self, num_classes, img_size):
        """Test model can be initialized."""
        model = YOLO(num_classes=num_classes, img_size=img_size)
        assert model.num_classes == num_classes
        assert model.img_size == img_size
        assert len(model.anchors) == 3  # Three scales

    def test_grid_sizes(self, img_size):
        """Test grid sizes are calculated correctly."""
        model = YOLO(num_classes=1, img_size=img_size)
        assert model.grid_size_p3 == img_size // 8
        assert model.grid_size_p4 == img_size // 16
        assert model.grid_size_p5 == img_size // 32

    def test_default_anchors(self):
        """Test default anchors are set correctly."""
        model = YOLO(num_classes=1, img_size=640)
        assert len(model.anchors) == 3
        assert model.anchors[0].shape == (3, 2)  # P3: 3 anchors, 2 dims
        assert model.anchors[1].shape == (3, 2)  # P4: 3 anchors, 2 dims
        assert model.anchors[2].shape == (3, 2)  # P5: 3 anchors, 2 dims

    def test_custom_anchors(self):
        """Test model with custom multi-scale anchors."""
        custom_anchors = [
            [[5, 7], [10, 14], [20, 28]],      # P3
            [[25, 35], [40, 50], [60, 70]],    # P4
            [[80, 100], [120, 140], [200, 220]] # P5
        ]
        model = YOLO(num_classes=1, anchors=custom_anchors, img_size=640)
        assert len(model.anchors) == 3
        assert torch.allclose(model.anchors[0], torch.tensor(custom_anchors[0], dtype=torch.float32))

    def test_forward_pass_640(self, dummy_model, dummy_input):
        """Test forward pass with 640×640 input."""
        outputs = dummy_model(dummy_input)

        # Should return list of 3 outputs
        assert isinstance(outputs, list)
        assert len(outputs) == 3

        # Check shapes for each scale
        batch_size = dummy_input.shape[0]
        assert outputs[0].shape == (batch_size, 80, 80, 3, 6)   # P3: 80×80 grid
        assert outputs[1].shape == (batch_size, 40, 40, 3, 6)   # P4: 40×40 grid
        assert outputs[2].shape == (batch_size, 20, 20, 3, 6)   # P5: 20×20 grid

    def test_forward_pass_1024(self, num_classes):
        """Test forward pass with 1024×1024 input."""
        model = YOLO(num_classes=num_classes, img_size=1024)
        x = torch.randn(2, 3, 1024, 1024)
        outputs = model(x)

        assert len(outputs) == 3
        assert outputs[0].shape == (2, 128, 128, 3, 6)  # P3: 1024/8 = 128
        assert outputs[1].shape == (2, 64, 64, 3, 6)    # P4: 1024/16 = 64
        assert outputs[2].shape == (2, 32, 32, 3, 6)    # P5: 1024/32 = 32

    def test_forward_pass_512(self):
        """Test forward pass with 512×512 input."""
        model = YOLO(num_classes=1, img_size=512)
        x = torch.randn(2, 3, 512, 512)
        outputs = model(x)

        assert len(outputs) == 3
        assert outputs[0].shape == (2, 64, 64, 3, 6)   # P3: 512/8 = 64
        assert outputs[1].shape == (2, 32, 32, 3, 6)   # P4: 512/16 = 32
        assert outputs[2].shape == (2, 16, 16, 3, 6)   # P5: 512/32 = 16

    def test_multi_class_output_shape(self):
        """Test output shape with multiple classes."""
        model = YOLO(num_classes=3, img_size=640)
        x = torch.randn(2, 3, 640, 640)
        outputs = model(x)

        # 5 + 3 classes = 8 output channels per anchor
        assert outputs[0].shape == (2, 80, 80, 3, 8)
        assert outputs[1].shape == (2, 40, 40, 3, 8)
        assert outputs[2].shape == (2, 20, 20, 3, 8)

    def test_output_contains_valid_values(self, dummy_model, dummy_input):
        """Test that outputs contain valid (non-NaN, non-inf) values."""
        outputs = dummy_model(dummy_input)

        for i, out in enumerate(outputs):
            assert not torch.isnan(out).any(), f"P{i+3} output contains NaN"
            assert not torch.isinf(out).any(), f"P{i+3} output contains Inf"

    def test_model_parameters_trainable(self, dummy_model):
        """Test that model parameters require gradients."""
        trainable_params = [p for p in dummy_model.parameters() if p.requires_grad]
        total_params = list(dummy_model.parameters())
        assert len(trainable_params) == len(total_params)
        assert len(trainable_params) > 0

    def test_model_parameter_count(self):
        """Test approximate parameter count."""
        model = YOLO(num_classes=1, img_size=640)
        total_params = sum(p.numel() for p in model.parameters())
        # YOLOv5-s with width_mult=0.5, depth_mult=0.33 has ~3.66M parameters
        assert 3_000_000 < total_params < 4_500_000

    def test_backward_compatibility_single_scale_anchors(self):
        """Test backward compatibility with single anchor set."""
        single_anchors = [[10, 13], [16, 30], [33, 23]]
        model = YOLO(num_classes=1, anchors=single_anchors, img_size=640)
        # Should replicate anchors across all 3 scales
        assert len(model.anchors) == 3
        for i in range(3):
            assert torch.allclose(model.anchors[i], torch.tensor(single_anchors, dtype=torch.float32))

    def test_model_eval_mode(self, dummy_model, dummy_input):
        """Test model in eval mode."""
        dummy_model.eval()
        with torch.no_grad():
            outputs = dummy_model(dummy_input)
        assert len(outputs) == 3
        assert outputs[0].shape[0] == dummy_input.shape[0]

    def test_batch_size_one(self, dummy_model):
        """Test forward pass with batch size 1."""
        x = torch.randn(1, 3, 640, 640)
        outputs = dummy_model(x)
        assert outputs[0].shape == (1, 80, 80, 3, 6)
        assert outputs[1].shape == (1, 40, 40, 3, 6)
        assert outputs[2].shape == (1, 20, 20, 3, 6)

    def test_large_batch_size(self, dummy_model):
        """Test forward pass with larger batch size."""
        x = torch.randn(16, 3, 640, 640)
        outputs = dummy_model(x)
        assert outputs[0].shape == (16, 80, 80, 3, 6)
        assert outputs[1].shape == (16, 40, 40, 3, 6)
        assert outputs[2].shape == (16, 20, 20, 3, 6)

    def test_initialize_biases_with_none(self, capsys):
        """Test bias initialization handles None bias (lines 543-544)."""
        import torch.nn as nn

        model = YOLO(num_classes=1, img_size=640)

        # Manually set one detection head's bias to None to trigger warning
        # Detection heads are nn.Sequential, last layer is Conv2d
        model.head_p3[-1].bias = None

        # Now call initialize_detection_biases - should create bias and print warning
        model.initialize_detection_biases()

        # Capture stdout to verify warning was printed
        captured = capsys.readouterr()
        assert "Warning: Detection head bias was None" in captured.out

        # Verify bias was created
        assert model.head_p3[-1].bias is not None
        assert isinstance(model.head_p3[-1].bias, nn.Parameter)
        assert model.head_p3[-1].bias.shape == (model.output_channels,)
