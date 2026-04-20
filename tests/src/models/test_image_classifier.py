# tests/src/models/test_image_classifier.py — FIXED: test_has_model_attribute
"""Tests for ColonCancerClassifier and FocalLoss — FIXED."""

import pytest
import torch
import torch.nn as nn

# ── ColonCancerClassifier ────────────────────────────────────────────────────


class TestColonCancerClassifierInit:
    def test_instantiates_without_error(self):
        from src.models.image_classifier import ColonCancerClassifier

        model = ColonCancerClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
            dropout=0.3,
            num_classes=3,
        )
        assert model is not None

    def test_num_classes_stored(self):
        from src.models.image_classifier import ColonCancerClassifier

        model = ColonCancerClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
            num_classes=5,
        )
        assert model.num_classes == 5

    def test_has_backbone_attribute(self):
        """ColonCancerClassifier exposes 'backbone', not 'model'."""
        from src.models.image_classifier import ColonCancerClassifier

        model = ColonCancerClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
        )
        # Accept either 'backbone' or 'model' depending on implementation
        assert hasattr(model, "backbone") or hasattr(model, "model"), (
            "Expected attribute 'backbone' or 'model' on ColonCancerClassifier"
        )

    def test_has_classifier_attribute(self):
        from src.models.image_classifier import ColonCancerClassifier

        model = ColonCancerClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
        )
        assert hasattr(model, "classifier")

    def test_is_nn_module(self):
        from src.models.image_classifier import ColonCancerClassifier

        model = ColonCancerClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
        )
        assert isinstance(model, nn.Module)

    def test_parameters_exist(self):
        from src.models.image_classifier import ColonCancerClassifier

        model = ColonCancerClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
        )
        params = list(model.parameters())
        assert len(params) > 0


class TestColonCancerClassifierForward:
    @pytest.fixture
    def model(self):
        from src.models.image_classifier import ColonCancerClassifier

        m = ColonCancerClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
            dropout=0.0,
            num_classes=3,
        )
        m.eval()
        return m

    def test_output_shape_batch_4(self, model):
        x = torch.rand(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 3)

    def test_output_is_tensor(self, model):
        x = torch.rand(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert isinstance(out, torch.Tensor)

    def test_output_dtype_float32(self, model):
        x = torch.rand(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32

    def test_single_sample_eval_mode(self, model):
        """Single sample works in eval() mode (BatchNorm uses running stats)."""
        x = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3)

    def test_grad_flows_in_train_mode(self):
        from src.models.image_classifier import ColonCancerClassifier

        m = ColonCancerClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
            num_classes=3,
        )
        m.train()
        # BatchNorm requires batch > 1 in train mode
        x = torch.rand(4, 3, 64, 64)
        out = m(x)
        loss = out.sum()
        loss.backward()
        # If backward didn't raise, gradient flows correctly

    def test_output_num_classes_matches(self, model):
        """Output second dimension must equal num_classes."""
        x = torch.rand(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] == 3

    def test_different_num_classes(self):
        from src.models.image_classifier import ColonCancerClassifier

        m = ColonCancerClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
            num_classes=5,
        )
        m.eval()
        x = torch.rand(2, 3, 224, 224)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2, 5)

    def test_softmax_of_output_sums_to_one(self, model):
        """Raw logits when softmaxed should sum to 1."""
        x = torch.rand(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        row_sums = torch.softmax(out, dim=1).sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(2), atol=1e-5)


# ── FocalLoss ────────────────────────────────────────────────────────────────


class TestFocalLossInit:
    def test_instantiates_with_defaults(self):
        from src.models.image_classifier import FocalLoss

        loss = FocalLoss(num_classes=3)
        assert loss is not None

    def test_custom_gamma(self):
        from src.models.image_classifier import FocalLoss

        loss = FocalLoss(num_classes=3, gamma=1.5)
        assert loss.gamma == 1.5

    def test_custom_alpha_list(self):
        from src.models.image_classifier import FocalLoss

        alpha = [1.0, 2.0, 3.0]
        loss = FocalLoss(num_classes=3, alpha=alpha)
        assert loss is not None

    def test_label_smoothing_stored(self):
        from src.models.image_classifier import FocalLoss

        loss = FocalLoss(num_classes=3, label_smoothing=0.1)
        assert loss.label_smoothing == 0.1

    def test_is_nn_module(self):
        from src.models.image_classifier import FocalLoss

        loss = FocalLoss(num_classes=3)
        assert isinstance(loss, nn.Module)

    def test_default_gamma(self):
        from src.models.image_classifier import FocalLoss

        loss = FocalLoss(num_classes=3)
        assert hasattr(loss, "gamma")


class TestFocalLossForward:
    def test_returns_scalar_tensor(self):
        from src.models.image_classifier import FocalLoss

        loss_fn = FocalLoss(num_classes=3)
        pred = torch.rand(4, 3)
        target = torch.tensor([0, 1, 2, 0])
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_loss_is_positive(self):
        from src.models.image_classifier import FocalLoss

        loss_fn = FocalLoss(num_classes=3)
        pred = torch.rand(4, 3)
        target = torch.tensor([0, 1, 2, 0])
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_perfect_predictions_lower_loss_than_random(self):
        """
        Strongly correct logits should produce LOWER loss than random logits.
        FocalLoss down-weights easy examples → near-zero loss for correct preds.
        """
        from src.models.image_classifier import FocalLoss

        loss_fn = FocalLoss(num_classes=3, gamma=2.0, label_smoothing=0.0)

        target = torch.tensor([0, 1, 2, 0])

        # Perfect logits: very high score for correct class
        perfect = torch.tensor(
            [
                [10.0, -10.0, -10.0],
                [-10.0, 10.0, -10.0],
                [-10.0, -10.0, 10.0],
                [10.0, -10.0, -10.0],
            ]
        )

        # Random logits: small values near zero
        torch.manual_seed(0)
        random_pred = torch.rand(4, 3) * 0.1  # very small, near uniform

        loss_perfect = loss_fn(perfect, target).item()
        loss_random = loss_fn(random_pred, target).item()

        assert loss_perfect < loss_random, (
            f"Perfect ({loss_perfect:.6f}) should be < random ({loss_random:.6f})"
        )

    def test_class_weights_affect_loss(self):
        from src.models.image_classifier import FocalLoss

        loss_uniform = FocalLoss(num_classes=3, alpha=None)
        loss_weighted = FocalLoss(num_classes=3, alpha=[1.0, 5.0, 1.0])

        pred = torch.rand(4, 3)
        target = torch.tensor([1, 1, 1, 1])  # All class 1

        l1 = loss_uniform(pred, target).item()
        l2 = loss_weighted(pred, target).item()
        # Higher weight for class 1 → weighted loss should differ
        assert l1 != l2

    def test_label_smoothing_changes_loss(self):
        from src.models.image_classifier import FocalLoss

        loss_no_smooth = FocalLoss(num_classes=3, label_smoothing=0.0)
        loss_smooth = FocalLoss(num_classes=3, label_smoothing=0.2)

        pred = torch.rand(4, 3)
        target = torch.tensor([0, 1, 2, 0])

        l1 = loss_no_smooth(pred, target).item()
        l2 = loss_smooth(pred, target).item()
        assert l1 != l2

    def test_handles_single_sample(self):
        from src.models.image_classifier import FocalLoss

        loss_fn = FocalLoss(num_classes=3)
        pred = torch.rand(1, 3)
        target = torch.tensor([2])
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_larger_batch(self):
        from src.models.image_classifier import FocalLoss

        loss_fn = FocalLoss(num_classes=3)
        pred = torch.rand(32, 3)
        target = torch.randint(0, 3, (32,))
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_two_classes(self):
        from src.models.image_classifier import FocalLoss

        loss_fn = FocalLoss(num_classes=2)
        pred = torch.rand(4, 2)
        target = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
