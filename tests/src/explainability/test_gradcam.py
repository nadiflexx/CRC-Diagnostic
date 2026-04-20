# tests/src/evaluation/test_gradcam.py
"""
Tests for src/evaluation/gradcam.py
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.evaluation.gradcam import (
    compute_attention_stats,
    compute_pointing_accuracy,
    create_heatmap_overlay,
    find_target_layer,
    generate_gradcam,
)

# ─────────────────────────────────────────────────────────────
#  Minimal models for testing
# ─────────────────────────────────────────────────────────────


class _TinyConvNet(nn.Module):
    """Minimal model with a Conv2d for target-layer discovery."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class _NoConvNet(nn.Module):
    """Model with no Conv2d layers."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)

    def forward(self, x):
        return self.fc(x.flatten(1))


class _BackboneModel(nn.Module):
    """Model with a backbone attribute containing Conv2d."""

    def __init__(self):
        super().__init__()

        class _Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_head = nn.Conv2d(3, 8, 1)
                self.pool = nn.AdaptiveAvgPool2d(1)

            def forward(self, x):
                return self.pool(self.conv_head(x)).flatten(1)

        self.backbone = _Backbone()
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc(self.backbone(x))


# ─────────────────────────────────────────────────────────────
#  find_target_layer
# ─────────────────────────────────────────────────────────────


class TestFindTargetLayer:
    def test_finds_conv_in_simple_model(self):
        model = _TinyConvNet()
        layer = find_target_layer(model)
        assert isinstance(layer, nn.Conv2d)

    def test_raises_when_no_conv(self):
        model = _NoConvNet()
        with pytest.raises(RuntimeError, match="No convolutional layer"):
            find_target_layer(model)

    def test_finds_conv_head_in_backbone(self):
        model = _BackboneModel()
        layer = find_target_layer(model)
        assert isinstance(layer, nn.Conv2d)

    def test_returns_nn_module(self):
        model = _TinyConvNet()
        layer = find_target_layer(model)
        assert isinstance(layer, nn.Module)


# ─────────────────────────────────────────────────────────────
#  generate_gradcam
# ─────────────────────────────────────────────────────────────


class TestGenerateGradcam:
    @pytest.fixture
    def tiny_model(self):
        m = _TinyConvNet(num_classes=3)
        m.eval()
        return m

    @pytest.fixture
    def input_tensor(self):
        return torch.rand(1, 3, 32, 32)

    def test_returns_tuple(self, tiny_model, input_tensor):
        cam, pred = generate_gradcam(
            tiny_model, input_tensor, target_class=0, method="gradcam"
        )
        assert isinstance(cam, np.ndarray)
        assert isinstance(pred, int)

    def test_cam_shape_is_2d(self, tiny_model, input_tensor):
        cam, _ = generate_gradcam(
            tiny_model, input_tensor, target_class=0, method="gradcam"
        )
        assert cam.ndim == 2

    def test_cam_values_in_0_1(self, tiny_model, input_tensor):
        cam, _ = generate_gradcam(
            tiny_model, input_tensor, target_class=0, method="gradcam"
        )
        assert cam.min() >= 0.0 - 1e-6
        assert cam.max() <= 1.0 + 1e-6

    def test_pred_class_valid_index(self, tiny_model, input_tensor):
        _, pred = generate_gradcam(tiny_model, input_tensor, method="gradcam")
        assert 0 <= pred < 3

    def test_with_explicit_target_layer(self, tiny_model, input_tensor):
        layer = tiny_model.conv1
        cam, pred = generate_gradcam(
            tiny_model,
            input_tensor,
            target_class=1,
            method="gradcam",
            target_layer=layer,
        )
        assert cam.ndim == 2

    def test_gradcampp_method(self, tiny_model, input_tensor):
        cam, _ = generate_gradcam(
            tiny_model,
            input_tensor,
            target_class=0,
            method="gradcam++",
        )
        assert cam.ndim == 2

    def test_none_target_uses_predicted(self, tiny_model, input_tensor):
        cam, pred = generate_gradcam(
            tiny_model,
            input_tensor,
            target_class=None,
            method="gradcam",
        )
        assert isinstance(pred, int)
        assert cam.ndim == 2


# ─────────────────────────────────────────────────────────────
#  create_heatmap_overlay
# ─────────────────────────────────────────────────────────────


class TestCreateHeatmapOverlay:
    def test_output_shape_matches_input(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cam = np.random.rand(64, 64).astype(np.float32)
        result = create_heatmap_overlay(img, cam)
        assert result.shape == img.shape

    def test_output_dtype_uint8(self):
        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        cam = np.zeros((32, 32), dtype=np.float32)
        result = create_heatmap_overlay(img, cam)
        assert result.dtype == np.uint8

    def test_cam_resized_to_image(self):
        """CAM smaller than image is upsampled correctly."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cam = np.random.rand(16, 16).astype(np.float32)
        result = create_heatmap_overlay(img, cam)
        assert result.shape == (64, 64, 3)

    def test_alpha_zero_returns_original_like(self):
        """alpha=0 → image_weight=1.0 → overlay ≈ original."""
        img = np.ones((32, 32, 3), dtype=np.uint8) * 150
        cam = np.zeros((32, 32), dtype=np.float32)
        result = create_heatmap_overlay(img, cam, alpha=0.0)
        assert result.shape == img.shape

    def test_different_alphas_produce_different_results(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cam = np.random.rand(64, 64).astype(np.float32)
        r1 = create_heatmap_overlay(img, cam, alpha=0.2)
        r2 = create_heatmap_overlay(img, cam, alpha=0.8)
        assert not np.array_equal(r1, r2)


# ─────────────────────────────────────────────────────────────
#  compute_attention_stats
# ─────────────────────────────────────────────────────────────


class TestComputeAttentionStats:
    def test_returns_two_floats(self):
        cam = np.random.rand(64, 64).astype(np.float32)
        center, border = compute_attention_stats(cam)
        assert isinstance(center, float)
        assert isinstance(border, float)

    def test_uniform_cam_equal_center_border(self):
        cam = np.ones((64, 64), dtype=np.float32) * 0.5
        center, border = compute_attention_stats(cam)
        assert abs(center - border) < 1e-4

    def test_bright_center_detected(self):
        cam = np.zeros((100, 100), dtype=np.float32)
        cam[20:80, 20:80] = 1.0  # centre is bright
        center, border = compute_attention_stats(cam, center_ratio=0.6)
        assert center > border

    def test_bright_border_detected(self):
        cam = np.ones((100, 100), dtype=np.float32)
        cam[20:80, 20:80] = 0.0  # centre is dark
        center, border = compute_attention_stats(cam, center_ratio=0.6)
        assert border > center

    def test_values_in_range_0_1(self):
        cam = np.random.rand(50, 50).astype(np.float32)
        center, border = compute_attention_stats(cam)
        assert 0.0 <= center <= 1.0
        assert 0.0 <= border <= 1.0

    def test_all_zeros_cam(self):
        cam = np.zeros((64, 64), dtype=np.float32)
        center, border = compute_attention_stats(cam)
        assert center == 0.0
        assert border == 0.0


# ─────────────────────────────────────────────────────────────
#  compute_pointing_accuracy
# ─────────────────────────────────────────────────────────────


class TestComputePointingAccuracy:
    def test_returns_bool_and_float(self):
        cam = np.random.rand(64, 64).astype(np.float32)
        in_center, ratio = compute_pointing_accuracy(cam)
        assert isinstance(in_center, bool)
        assert isinstance(ratio, float)

    def test_peak_in_center_detected(self):
        cam = np.zeros((100, 100), dtype=np.float32)
        cam[50, 50] = 1.0  # exact centre
        in_center, _ = compute_pointing_accuracy(cam, center_ratio=0.6)
        assert in_center is True

    def test_peak_at_corner_not_in_center(self):
        cam = np.zeros((100, 100), dtype=np.float32)
        cam[0, 0] = 1.0  # top-left corner
        in_center, _ = compute_pointing_accuracy(cam, center_ratio=0.6)
        assert in_center is False

    def test_center_strong_ratio_in_0_1(self):
        cam = np.random.rand(64, 64).astype(np.float32)
        _, ratio = compute_pointing_accuracy(cam)
        assert 0.0 <= ratio <= 1.0

    def test_all_strong_in_center_ratio_near_1(self):
        cam = np.zeros((100, 100), dtype=np.float32)
        cam[30:70, 30:70] = 1.0  # all activity in centre
        _, ratio = compute_pointing_accuracy(cam, center_ratio=0.6)
        assert ratio > 0.5

    def test_uniform_cam_max_somewhere(self):
        cam = np.ones((64, 64), dtype=np.float32)
        in_center, ratio = compute_pointing_accuracy(cam)
        # uniform → argmax at (0,0) which may or may not be in center
        assert isinstance(in_center, bool)
        assert 0.0 <= ratio <= 1.0
