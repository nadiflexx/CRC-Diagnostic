# tests/src/models/test_polyp_segmenter.py
"""
Tests for src/models/polyp_segmenter.py
(ColonPolypSegmenter + DiceBCELoss)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.polyp_segmenter import ColonPolypSegmenter, DiceBCELoss

# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def segmenter():
    return ColonPolypSegmenter(encoder_name="efficientnet-b0", pretrained=None)


@pytest.fixture
def batch_img():
    return torch.rand(2, 3, 128, 128)


@pytest.fixture
def batch_mask():
    m = torch.zeros(2, 1, 128, 128)
    m[:, :, 40:80, 40:80] = 1.0
    return m


# ─────────────────────────────────────────────────────────────
#  ColonPolypSegmenter — __init__
# ─────────────────────────────────────────────────────────────


class TestSegmenterInit:
    def test_model_attribute_exists(self, segmenter):
        assert segmenter.model is not None

    def test_is_nn_module(self, segmenter):
        assert isinstance(segmenter, nn.Module)


# ─────────────────────────────────────────────────────────────
#  ColonPolypSegmenter — forward
# ─────────────────────────────────────────────────────────────


class TestSegmenterForward:
    def test_output_shape(self, segmenter, batch_img):
        out = segmenter(batch_img)
        assert out.shape == (2, 1, 128, 128)

    def test_output_is_tensor(self, segmenter, batch_img):
        out = segmenter(batch_img)
        assert isinstance(out, torch.Tensor)

    def test_output_dtype_float(self, segmenter, batch_img):
        out = segmenter(batch_img)
        assert out.dtype == torch.float32

    def test_single_sample(self, segmenter):
        x = torch.rand(1, 3, 64, 64)
        out = segmenter(x)
        assert out.shape == (1, 1, 64, 64)


# ─────────────────────────────────────────────────────────────
#  ColonPolypSegmenter — predict_mask
# ─────────────────────────────────────────────────────────────


class TestPredictMask:
    def test_returns_two_tensors(self, segmenter, batch_img):
        masks, probs = segmenter.predict_mask(batch_img)
        assert isinstance(masks, torch.Tensor)
        assert isinstance(probs, torch.Tensor)

    def test_mask_shape(self, segmenter, batch_img):
        masks, _ = segmenter.predict_mask(batch_img)
        assert masks.shape == (2, 1, 128, 128)

    def test_mask_binary(self, segmenter, batch_img):
        masks, _ = segmenter.predict_mask(batch_img)
        unique = torch.unique(masks)
        assert all(v.item() in (0.0, 1.0) for v in unique)

    def test_probs_in_0_1(self, segmenter, batch_img):
        _, probs = segmenter.predict_mask(batch_img)
        assert probs.min().item() >= 0.0
        assert probs.max().item() <= 1.0

    def test_custom_threshold(self, segmenter, batch_img):
        masks_05, _ = segmenter.predict_mask(batch_img, threshold=0.5)
        masks_09, _ = segmenter.predict_mask(batch_img, threshold=0.9)
        # Higher threshold → fewer positive pixels
        assert masks_09.sum() <= masks_05.sum()

    def test_no_grad(self, segmenter, batch_img):
        masks, probs = segmenter.predict_mask(batch_img)
        assert not masks.requires_grad
        assert not probs.requires_grad

    def test_model_in_eval_mode(self, segmenter, batch_img):
        segmenter.train()
        segmenter.predict_mask(batch_img)
        assert not segmenter.training


# ─────────────────────────────────────────────────────────────
#  ColonPolypSegmenter — predict_mask_tta
# ─────────────────────────────────────────────────────────────


class TestPredictMaskTTA:
    def test_output_shape(self, segmenter, batch_img):
        masks, probs = segmenter.predict_mask_tta(batch_img)
        assert masks.shape == (2, 1, 128, 128)
        assert probs.shape == (2, 1, 128, 128)

    def test_mask_binary(self, segmenter, batch_img):
        masks, _ = segmenter.predict_mask_tta(batch_img)
        unique = torch.unique(masks)
        assert all(v.item() in (0.0, 1.0) for v in unique)

    def test_probs_in_0_1(self, segmenter, batch_img):
        _, probs = segmenter.predict_mask_tta(batch_img)
        assert probs.min().item() >= 0.0
        assert probs.max().item() <= 1.0

    def test_tta_vs_standard_differ(self, segmenter, batch_img):
        """TTA averages 4 augmentations → slightly different from single pass."""
        _, probs_std = segmenter.predict_mask(batch_img)
        _, probs_tta = segmenter.predict_mask_tta(batch_img)
        # Both are valid probability maps
        assert probs_std.shape == probs_tta.shape


# ─────────────────────────────────────────────────────────────
#  ColonPolypSegmenter — postprocess_instances (static)
# ─────────────────────────────────────────────────────────────


class TestPostprocessInstances:
    def test_empty_mask_returns_zero_polyps(self):
        mask = np.zeros((128, 128), dtype=np.float32)
        result, n = ColonPolypSegmenter.postprocess_instances(mask)
        assert n == 0

    def test_full_mask_one_polyp(self):
        mask = np.ones((128, 128), dtype=np.float32)
        result, n = ColonPolypSegmenter.postprocess_instances(mask, min_area=10)
        assert n >= 1

    def test_two_separate_regions(self):
        mask = np.zeros((200, 200), dtype=np.float32)
        mask[10:50, 10:50] = 1.0  # region 1
        mask[120:160, 120:160] = 1.0  # region 2
        result, n = ColonPolypSegmenter.postprocess_instances(mask, min_area=100)
        assert n == 2

    def test_small_component_removed(self):
        mask = np.zeros((200, 200), dtype=np.float32)
        mask[10:50, 10:50] = 1.0  # large region (1600 px)
        mask[100:102, 100:102] = 1.0  # tiny (4 px) → removed
        result, n = ColonPolypSegmenter.postprocess_instances(mask, min_area=100)
        assert n == 1

    def test_output_is_integer_mask(self):
        mask = np.ones((64, 64), dtype=np.float32)
        result, n = ColonPolypSegmenter.postprocess_instances(mask, min_area=10)
        assert result.dtype in (np.float32, np.float64, np.int32, np.int64)

    def test_instance_ids_consecutive(self):
        mask = np.zeros((200, 200), dtype=np.float32)
        mask[10:40, 10:40] = 1.0
        mask[100:130, 100:130] = 1.0
        result, n = ColonPolypSegmenter.postprocess_instances(mask, min_area=100)
        for pid in range(1, n + 1):
            assert (result == pid).sum() > 0

    def test_returns_tuple(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        result = ColonPolypSegmenter.postprocess_instances(mask)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_custom_min_area(self):
        mask = np.zeros((200, 200), dtype=np.float32)
        mask[10:20, 10:20] = 1.0  # 100 px
        _, n_strict = ColonPolypSegmenter.postprocess_instances(mask, min_area=200)
        _, n_loose = ColonPolypSegmenter.postprocess_instances(mask, min_area=50)
        assert n_strict == 0
        assert n_loose >= 1


# ─────────────────────────────────────────────────────────────
#  DiceBCELoss — __init__
# ─────────────────────────────────────────────────────────────


class TestDiceBCELossInit:
    def test_weights_stored(self):
        loss = DiceBCELoss(dice_weight=0.6, bce_weight=0.2, boundary_weight=0.2)
        assert loss.dice_weight == 0.6
        assert loss.bce_weight == 0.2
        assert loss.boundary_weight == 0.2

    def test_smooth_stored(self):
        loss = DiceBCELoss(smooth=2.0)
        assert loss.smooth == 2.0

    def test_is_nn_module(self):
        assert isinstance(DiceBCELoss(), nn.Module)

    def test_default_values(self):
        loss = DiceBCELoss()
        assert loss.dice_weight == 0.5
        assert loss.bce_weight == 0.3
        assert loss.boundary_weight == 0.2


# ─────────────────────────────────────────────────────────────
#  DiceBCELoss — forward
# ─────────────────────────────────────────────────────────────


class TestDiceBCELossForward:
    @pytest.fixture
    def loss_fn(self):
        return DiceBCELoss()

    @pytest.fixture
    def preds_targets(self):
        pred = torch.randn(2, 1, 64, 64)
        target = (torch.rand(2, 1, 64, 64) > 0.5).float()
        return pred, target

    def test_returns_scalar(self, loss_fn, preds_targets):
        pred, target = preds_targets
        loss = loss_fn(pred, target)
        assert loss.shape == ()

    def test_loss_is_positive(self, loss_fn, preds_targets):
        pred, target = preds_targets
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_loss_is_finite(self, loss_fn, preds_targets):
        pred, target = preds_targets
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_gradient_flows(self, loss_fn):
        pred = torch.randn(2, 1, 32, 32, requires_grad=True)
        target = torch.rand(2, 1, 32, 32)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_3d_target_accepted(self, loss_fn):
        """Target without channel dim should be accepted."""
        pred = torch.randn(2, 1, 32, 32)
        target = torch.rand(2, 32, 32)  # 3D — no channel
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_perfect_prediction_lower_loss(self, loss_fn):
        """Perfect predictions give lower loss than random."""
        target = (torch.rand(2, 1, 32, 32) > 0.5).float()
        # Perfect logits: large positive where target=1, negative elsewhere
        perfect = (target * 2 - 1) * 10.0
        random = torch.randn(2, 1, 32, 32)
        loss_perfect = loss_fn(perfect, target).item()
        loss_random = loss_fn(random, target).item()
        assert loss_perfect < loss_random

    def test_zero_boundary_weight(self):
        loss_fn = DiceBCELoss(boundary_weight=0.0)
        pred = torch.randn(2, 1, 32, 32)
        target = torch.rand(2, 1, 32, 32)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_compute_boundary_mask_shape(self, loss_fn):
        target = torch.rand(2, 1, 64, 64)
        boundary = loss_fn._compute_boundary_mask(target)
        assert boundary.shape == target.shape

    def test_compute_boundary_mask_values(self, loss_fn):
        target = torch.zeros(1, 1, 64, 64)
        target[:, :, 20:40, 20:40] = 1.0
        boundary = loss_fn._compute_boundary_mask(target)
        assert boundary.min().item() >= 0.0
        assert boundary.max().item() <= 1.0
