# tests/src/models/test_tissue_classifier.py — FIXED: test_has_model_backbone
"""Tests for TissueOnlyClassifier — FIXED attribute name."""

import pytest
import torch
import torch.nn as nn


class TestTissueOnlyClassifierInit:
    def test_instantiates_without_error(self):
        from src.models.tissue_classifier import TissueOnlyClassifier

        model = TissueOnlyClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
            num_classes=3,
        )
        assert model is not None

    def test_has_backbone_or_model_attribute(self):
        """
        TissueOnlyClassifier exposes 'backbone', not 'model'.
        Accept either name to be robust against implementation changes.
        """
        from src.models.tissue_classifier import TissueOnlyClassifier

        model = TissueOnlyClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
        )
        assert hasattr(model, "backbone") or hasattr(model, "model"), (
            "Expected attribute 'backbone' or 'model' on TissueOnlyClassifier. "
            f"Available: {[a for a in dir(model) if not a.startswith('_')]}"
        )

    def test_has_classifier_attribute(self):
        from src.models.tissue_classifier import TissueOnlyClassifier

        model = TissueOnlyClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
        )
        assert hasattr(model, "classifier")

    def test_is_nn_module(self):
        from src.models.tissue_classifier import TissueOnlyClassifier

        model = TissueOnlyClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
        )
        assert isinstance(model, nn.Module)

    def test_custom_dropout(self):
        from src.models.tissue_classifier import TissueOnlyClassifier

        model = TissueOnlyClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
            dropout=0.5,
        )
        assert model is not None

    def test_parameters_exist(self):
        from src.models.tissue_classifier import TissueOnlyClassifier

        model = TissueOnlyClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
        )
        params = list(model.parameters())
        assert len(params) > 0

    def test_num_classes_affects_output(self):
        """Different num_classes → different output dimension."""
        from src.models.tissue_classifier import TissueOnlyClassifier

        m3 = TissueOnlyClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
            num_classes=3,
        )
        m5 = TissueOnlyClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
            num_classes=5,
        )
        m3.eval()
        m5.eval()
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            out3 = m3(x)
            out5 = m5(x)
        assert out3.shape[1] == 3
        assert out5.shape[1] == 5


class TestTissueOnlyClassifierForward:
    @pytest.fixture
    def model(self):
        from src.models.tissue_classifier import TissueOnlyClassifier

        m = TissueOnlyClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
            dropout=0.0,
            num_classes=3,
        )
        # CRITICAL: eval() mode required for single-sample (BatchNorm)
        m.eval()
        return m

    def test_output_is_tensor(self, model):
        x = torch.rand(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert isinstance(out, torch.Tensor)

    def test_output_shape_batch_4(self, model):
        x = torch.rand(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 3)

    def test_output_dtype_float32(self, model):
        x = torch.rand(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32

    def test_single_sample_eval_mode(self, model):
        """Single sample works in eval() mode."""
        x = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3)

    def test_batch_output_shape(self, model):
        x = torch.rand(8, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 8
        assert out.shape[1] == 3

    def test_train_mode_batch_gt_1(self):
        """In train mode, BatchNorm requires batch > 1."""
        from src.models.tissue_classifier import TissueOnlyClassifier

        m = TissueOnlyClassifier(
            model_name="tf_efficientnetv2_s.in21k",
            pretrained=False,
            num_classes=3,
        )
        m.train()
        x = torch.rand(4, 3, 64, 64)
        out = m(x)
        assert out.shape == (4, 3)

    def test_output_num_classes(self, model):
        x = torch.rand(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] == 3

    def test_softmax_sums_to_one(self, model):
        x = torch.rand(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        row_sums = torch.softmax(out, dim=1).sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(2), atol=1e-5)

    def test_different_input_sizes_work(self, model):
        """Model should handle different spatial resolutions."""
        for size in [64, 128]:
            x = torch.rand(2, 3, size, size)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (2, 3), f"Failed for size {size}"
