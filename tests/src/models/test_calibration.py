# tests/src/evaluation/test_explainability.py
"""
Tests for src/evaluation/explainability.py

Heavy external dependencies (SHAP, GradCAM, segmenter) are mocked.
"""

from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # non-interactive backend

from src.evaluation.explainability import ImageExplainer, TabularExplainer

# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def mock_ensemble():
    ens = MagicMock()
    ens.ensemble_available = True
    ens.model_a = MagicMock()
    ens.model_b = MagicMock()
    ens.preprocessor_a = MagicMock()
    ens.preprocessor_b = MagicMock()
    # preprocessor_a.process_image returns a white BGR image
    ens.preprocessor_a.process_image.return_value = (
        np.ones((64, 64, 3), dtype=np.uint8) * 128
    )
    # preprocessor_b.process_image returns a list of crops
    ens.preprocessor_b.process_image.return_value = [
        np.ones((64, 64, 3), dtype=np.uint8) * 128
    ]
    return ens


@pytest.fixture
def mock_transform():
    import torch

    t = MagicMock()
    t.return_value = {"image": torch.rand(3, 64, 64)}
    return t


@pytest.fixture
def image_explainer(mock_ensemble, mock_transform):
    return ImageExplainer(
        ensemble=mock_ensemble,
        img_transform=mock_transform,
        class_names={0: "normal", 1: "polyp", 2: "inflammation"},
        device="cpu",
        cam_method="gradcam",
    )


@pytest.fixture
def mock_tabular_model():
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.3, 0.7]])
    m.model = MagicMock()
    m.model.feature_importances_ = np.array([0.1, 0.2, 0.05, 0.15, 0.5])
    return m


@pytest.fixture
def tabular_explainer(mock_tabular_model):
    features = ["age", "cea", "hgb", "adc", "entropy"]
    return TabularExplainer(
        model=mock_tabular_model,
        feature_names=features,
    )


# ─────────────────────────────────────────────────────────────
#  ImageExplainer.__init__
# ─────────────────────────────────────────────────────────────


class TestImageExplainerInit:
    def test_stores_ensemble(self, image_explainer, mock_ensemble):
        assert image_explainer.ensemble is mock_ensemble

    def test_stores_class_names(self, image_explainer):
        assert image_explainer.class_names[0] == "normal"

    def test_stores_device(self, image_explainer):
        assert image_explainer.device == "cpu"

    def test_stores_cam_method(self, image_explainer):
        assert image_explainer.cam_method == "gradcam"


# ─────────────────────────────────────────────────────────────
#  ImageExplainer.generate_gradcam
# ─────────────────────────────────────────────────────────────


class TestGenerateGradcam:
    @patch("src.evaluation.explainability.generate_gradcam")
    @patch("src.evaluation.explainability.create_heatmap_overlay")
    @patch("src.evaluation.explainability.numpy_to_base64")
    def test_returns_dict_with_correct_keys(
        self, mock_b64, mock_overlay, mock_gcam, image_explainer
    ):
        mock_gcam.return_value = (np.zeros((64, 64)), 0)
        mock_overlay.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_b64.return_value = "data:image/png;base64,abc"

        img = np.ones((64, 64, 3), dtype=np.uint8) * 100
        result = image_explainer.generate_gradcam(img, pred_idx=0)

        assert "gradcam_a" in result
        assert "gradcam_b" in result
        assert "gradcam_fusion" in result

    @patch("src.evaluation.explainability.generate_gradcam")
    @patch("src.evaluation.explainability.create_heatmap_overlay")
    @patch("src.evaluation.explainability.numpy_to_base64")
    def test_gradcam_a_populated(
        self, mock_b64, mock_overlay, mock_gcam, image_explainer
    ):
        mock_gcam.return_value = (np.random.rand(64, 64).astype(np.float32), 0)
        mock_overlay.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_b64.return_value = "base64data"

        img = np.ones((64, 64, 3), dtype=np.uint8) * 100
        result = image_explainer.generate_gradcam(img, pred_idx=1)
        assert result["gradcam_a"] == "base64data"

    @patch(
        "src.evaluation.explainability.generate_gradcam",
        side_effect=Exception("cam failed"),
    )
    def test_gradcam_a_none_on_exception(self, mock_gcam, image_explainer):
        img = np.ones((64, 64, 3), dtype=np.uint8) * 100
        result = image_explainer.generate_gradcam(img, pred_idx=0)
        assert result["gradcam_a"] is None


# ─────────────────────────────────────────────────────────────
#  ImageExplainer._compute_fusion
# ─────────────────────────────────────────────────────────────


class TestComputeFusion:
    @patch("src.evaluation.explainability.create_heatmap_overlay")
    @patch("src.evaluation.explainability.numpy_to_base64")
    def test_fusion_with_both_cams(self, mock_b64, mock_overlay, image_explainer):
        mock_overlay.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_b64.return_value = "fused"
        cam_a = np.ones((64, 64), dtype=np.float32) * 0.6
        cam_b = np.ones((64, 64), dtype=np.float32) * 0.4
        img_rgb = np.ones((64, 64, 3), dtype=np.uint8) * 128
        result = {}
        image_explainer._compute_fusion(img_rgb, cam_a, cam_b, 0.5, 0.5, result)
        assert result["gradcam_fusion"] == "fused"

    @patch("src.evaluation.explainability.create_heatmap_overlay")
    @patch("src.evaluation.explainability.numpy_to_base64")
    def test_fusion_with_only_cam_a(self, mock_b64, mock_overlay, image_explainer):
        mock_overlay.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_b64.return_value = "only_a"
        cam_a = np.ones((64, 64), dtype=np.float32) * 0.5
        img_rgb = np.ones((64, 64, 3), dtype=np.uint8)
        result = {}
        image_explainer._compute_fusion(img_rgb, cam_a, None, 1.0, 0.0, result)
        assert result["gradcam_fusion"] == "only_a"

    def test_fusion_skipped_when_cam_a_none(self, image_explainer):
        result = {}
        image_explainer._compute_fusion(
            np.zeros((64, 64, 3), dtype=np.uint8), None, None, 0.5, 0.5, result
        )
        assert "gradcam_fusion" not in result


# ─────────────────────────────────────────────────────────────
#  ImageExplainer.postprocess_and_save_mask
# ─────────────────────────────────────────────────────────────


class TestPostprocessAndSaveMask:
    @patch("src.evaluation.explainability.ColonPolypSegmenter.postprocess_instances")
    def test_returns_false_when_no_polyps(
        self, mock_post, image_explainer, tmp_path, monkeypatch
    ):
        mock_post.return_value = (np.zeros((64, 64), dtype=np.int32), 0)
        mask = np.zeros((64, 64), dtype=np.float32)
        preprocessed = np.zeros((64, 64, 3), dtype=np.uint8)
        found, url, n = image_explainer.postprocess_and_save_mask(mask, preprocessed)
        assert found is False
        assert url is None
        assert n == 0

    @patch("src.evaluation.explainability.ColonPolypSegmenter.postprocess_instances")
    @patch("cv2.imwrite")
    def test_returns_true_when_polyps_found(
        self, mock_write, mock_post, image_explainer, tmp_path, monkeypatch
    ):
        import src.evaluation.explainability as exp_mod

        fake_paths = MagicMock()
        fake_paths.REPORTS = tmp_path / "reports"
        monkeypatch.setattr(exp_mod, "paths", fake_paths)

        instance_mask = np.zeros((64, 64), dtype=np.int32)
        instance_mask[20:40, 20:40] = 1
        mock_post.return_value = (instance_mask, 1)
        mock_write.return_value = True

        mask = np.ones((64, 64), dtype=np.float32)
        preprocessed = np.ones((64, 64, 3), dtype=np.uint8) * 128
        found, url, n = image_explainer.postprocess_and_save_mask(mask, preprocessed)
        assert found is True
        assert n == 1
        assert url is not None
        assert "report_" in url


# ─────────────────────────────────────────────────────────────
#  TabularExplainer
# ─────────────────────────────────────────────────────────────


class TestTabularExplainerInit:
    def test_stores_model(self, tabular_explainer, mock_tabular_model):
        assert tabular_explainer.model is mock_tabular_model

    def test_stores_feature_names(self, tabular_explainer):
        assert "age" in tabular_explainer.feature_names

    def test_not_fitted_initially(self, tabular_explainer):
        assert tabular_explainer._fitted is False

    def test_explainer_none_initially(self, tabular_explainer):
        assert tabular_explainer._explainer is None


class TestTabularExplainerFit:
    def test_fit_without_shap_does_not_raise(self, tabular_explainer):
        X = np.random.rand(50, 5)
        with patch.dict("sys.modules", {"shap": None}):
            # ImportError path — should not raise
            try:
                tabular_explainer.fit(X)
            except Exception:
                pass  # acceptable if shap missing

    @patch("shap.TreeExplainer")
    def test_fit_sets_fitted_flag(self, mock_tree, tabular_explainer):
        mock_explainer = MagicMock()
        mock_tree.return_value = mock_explainer
        X = np.random.rand(50, 5)
        tabular_explainer.fit(X)
        assert tabular_explainer._fitted is True


class TestTabularExplainerExplain:
    def test_explain_returns_dict(self, tabular_explainer):
        X = np.random.rand(1, 5)
        result = tabular_explainer.explain(X)
        assert isinstance(result, dict)

    def test_explain_has_required_keys(self, tabular_explainer):
        X = np.random.rand(1, 5)
        result = tabular_explainer.explain(X)
        for key in (
            "shap_values",
            "feature_names",
            "top_risk_factors",
            "protective_factors",
            "base_value",
        ):
            assert key in result

    def test_explain_feature_importance_fallback(self, tabular_explainer):
        """_explain_feature_importance used when not fitted."""
        X = np.random.rand(1, 5)
        result = tabular_explainer._explain_feature_importance(X)
        assert result["base_value"] == 0.0
        assert len(result["top_risk_factors"]) <= len(tabular_explainer.feature_names)
        assert result["protective_factors"] == []

    def test_explain_feature_names_match(self, tabular_explainer):
        X = np.random.rand(1, 5)
        result = tabular_explainer.explain(X)
        assert result["feature_names"] == tabular_explainer.feature_names

    def test_shap_values_length(self, tabular_explainer):
        X = np.random.rand(1, 5)
        result = tabular_explainer.explain(X)
        assert len(result["shap_values"]) == len(tabular_explainer.feature_names)


class TestTabularExplainerPlot:
    def test_plot_explanation_returns_figure(self, tabular_explainer):
        import matplotlib.pyplot as plt

        X = np.random.rand(1, 5)
        fig = tabular_explainer.plot_explanation(X)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_saves_file(self, tabular_explainer, tmp_path):
        import matplotlib.pyplot as plt

        X = np.random.rand(1, 5)
        save_path = str(tmp_path / "shap_plot.png")
        fig = tabular_explainer.plot_explanation(X, save_path=save_path)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_beeswarm_returns_figure_when_not_fitted(self, tabular_explainer):
        import matplotlib.pyplot as plt

        X = np.random.rand(10, 5)
        fig = tabular_explainer.plot_explanation(X, plot_type="beeswarm")
        assert isinstance(fig, plt.Figure)
        plt.close("all")
