# tests/src/models/test_ensemble_predictor.py
"""
Tests for EnsemblePredictor — focusing on pure-logic methods
that do not require model checkpoints.
"""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.models.ensemble_predictor import EnsemblePredictor

# ─────────────────────────────────────────────────────────────
#  Fixture
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def predictor():
    ep = EnsemblePredictor(device="cpu")
    ep.num_classes = 3
    ep.class_names = {0: "normal", 1: "polyp", 2: "inflammation"}
    return ep


def _fake_model(num_classes=3):
    """Return a mock that acts like a loaded PyTorch model."""
    import torch

    m = MagicMock()
    m.eval.return_value = m
    m.return_value = torch.rand(1, num_classes)
    return m


# ─────────────────────────────────────────────────────────────
#  __init__
# ─────────────────────────────────────────────────────────────


class TestInit:
    def test_device_stored(self):
        ep = EnsemblePredictor(device="cpu")
        assert ep.device == "cpu"

    def test_models_start_none(self):
        ep = EnsemblePredictor(device="cpu")
        assert ep.model_a is None
        assert ep.model_b is None

    def test_loaded_false_initially(self):
        ep = EnsemblePredictor(device="cpu")
        assert ep._loaded is False

    def test_alpha_beta_default_half(self):
        ep = EnsemblePredictor(device="cpu")
        assert ep.alpha_base == 0.5
        assert ep.beta_base == 0.5

    def test_ensemble_available_false_initially(self):
        ep = EnsemblePredictor(device="cpu")
        assert ep.ensemble_available is False

    def test_use_adaptive_true_by_default(self):
        ep = EnsemblePredictor(device="cpu")
        assert ep.use_adaptive is True


# ─────────────────────────────────────────────────────────────
#  ensemble_available property
# ─────────────────────────────────────────────────────────────


class TestEnsembleAvailable:
    def test_false_when_both_none(self, predictor):
        predictor.model_a = None
        predictor.model_b = None
        assert predictor.ensemble_available is False

    def test_false_when_only_a_loaded(self, predictor):
        predictor.model_a = _fake_model()
        predictor.model_b = None
        assert predictor.ensemble_available is False

    def test_true_when_both_loaded(self, predictor):
        predictor.model_a = _fake_model()
        predictor.model_b = _fake_model()
        assert predictor.ensemble_available is True


# ─────────────────────────────────────────────────────────────
#  compute_adaptive_weights
# ─────────────────────────────────────────────────────────────


class TestComputeAdaptiveWeights:
    def test_returns_two_floats(self, predictor):
        alpha, beta = predictor.compute_adaptive_weights(1.5)
        assert isinstance(alpha, float)
        assert isinstance(beta, float)

    def test_sum_to_one(self, predictor):
        for ratio in [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]:
            a, b = predictor.compute_adaptive_weights(ratio)
            assert abs(a + b - 1.0) < 1e-6

    def test_high_ratio_favours_model_a(self, predictor):
        a_high, b_high = predictor.compute_adaptive_weights(3.0)
        a_low, b_low = predictor.compute_adaptive_weights(0.1)
        assert a_high > a_low

    def test_low_ratio_favours_model_b(self, predictor):
        _, b_low = predictor.compute_adaptive_weights(0.1)
        _, b_high = predictor.compute_adaptive_weights(3.0)
        assert b_low > b_high

    def test_values_within_alpha_min_max(self, predictor):
        for ratio in np.linspace(0, 4, 20):
            a, _ = predictor.compute_adaptive_weights(float(ratio))
            assert predictor.alpha_min <= a <= predictor.alpha_max

    def test_neutral_ratio_gives_balanced_weights(self, predictor):
        """At sigmoid_center the weights should be ~equal."""
        a, b = predictor.compute_adaptive_weights(predictor.sigmoid_center)
        assert abs(a - b) < 0.15  # within 15% of each other


# ─────────────────────────────────────────────────────────────
#  combine_predictions
# ─────────────────────────────────────────────────────────────


class TestCombinePredictions:
    def test_returns_three_elements(self, predictor):
        probs_a = np.array([0.6, 0.3, 0.1])
        probs_b = np.array([0.2, 0.5, 0.3])
        result = predictor.combine_predictions(probs_a, probs_b)
        assert len(result) == 3

    def test_combined_shape_matches(self, predictor):
        probs_a = np.array([0.6, 0.3, 0.1])
        probs_b = np.array([0.2, 0.5, 0.3])
        combined, _, _ = predictor.combine_predictions(probs_a, probs_b)
        assert combined.shape == probs_a.shape

    def test_fixed_weights_used_when_adaptive_off(self, predictor):
        predictor.use_adaptive = False
        predictor.alpha_base = 0.7
        predictor.beta_base = 0.3
        probs_a = np.array([1.0, 0.0, 0.0])
        probs_b = np.array([0.0, 1.0, 0.0])
        combined, alpha, beta = predictor.combine_predictions(
            probs_a, probs_b, attention_ratio=3.0
        )
        assert alpha == 0.7
        assert beta == 0.3
        np.testing.assert_allclose(combined, [0.7, 0.3, 0.0], atol=1e-6)

    def test_adaptive_weights_when_ratio_given(self, predictor):
        predictor.use_adaptive = True
        probs_a = np.array([0.8, 0.1, 0.1])
        probs_b = np.array([0.2, 0.6, 0.2])
        _, alpha, beta = predictor.combine_predictions(
            probs_a, probs_b, attention_ratio=3.0
        )
        a_expected, _ = predictor.compute_adaptive_weights(3.0)
        assert abs(alpha - a_expected) < 1e-6

    def test_fixed_weights_when_ratio_none(self, predictor):
        predictor.use_adaptive = True
        predictor.alpha_base = 0.6
        predictor.beta_base = 0.4
        probs_a = np.array([0.9, 0.05, 0.05])
        probs_b = np.array([0.1, 0.8, 0.1])
        _, alpha, beta = predictor.combine_predictions(
            probs_a, probs_b, attention_ratio=None
        )
        assert alpha == 0.6
        assert beta == 0.4

    def test_combined_values_are_weighted_average(self, predictor):
        predictor.use_adaptive = False
        predictor.alpha_base = 0.5
        predictor.beta_base = 0.5
        probs_a = np.array([0.4, 0.4, 0.2])
        probs_b = np.array([0.2, 0.6, 0.2])
        combined, _, _ = predictor.combine_predictions(probs_a, probs_b)
        expected = 0.5 * probs_a + 0.5 * probs_b
        np.testing.assert_allclose(combined, expected, atol=1e-6)


# ─────────────────────────────────────────────────────────────
#  _load_ensemble_config
# ─────────────────────────────────────────────────────────────


class TestLoadEnsembleConfig:
    def test_missing_config_file_does_not_raise(self, predictor, tmp_path, monkeypatch):
        import src.models.ensemble_predictor as ep_mod

        fake_paths = MagicMock()
        fake_paths.ENSEMBLE_CONFIG_PATH = tmp_path / "nonexistent.json"
        monkeypatch.setattr(ep_mod, "paths", fake_paths)
        predictor._load_ensemble_config()  # must not raise

    def test_loads_alpha_from_config(self, predictor, tmp_path, monkeypatch):
        import src.models.ensemble_predictor as ep_mod

        config = {"alpha": 0.7, "beta": 0.3, "use_adaptive": False}
        cfg_path = tmp_path / "ensemble_config.json"
        cfg_path.write_text(json.dumps(config))
        fake_paths = MagicMock()
        fake_paths.ENSEMBLE_CONFIG_PATH = cfg_path
        monkeypatch.setattr(ep_mod, "paths", fake_paths)
        predictor._load_ensemble_config()
        assert predictor.alpha_base == 0.7

    def test_loads_use_adaptive_from_config(self, predictor, tmp_path, monkeypatch):
        import src.models.ensemble_predictor as ep_mod

        config = {"alpha": 0.5, "beta": 0.5, "use_adaptive": False}
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(config))
        fake_paths = MagicMock()
        fake_paths.ENSEMBLE_CONFIG_PATH = cfg_path
        monkeypatch.setattr(ep_mod, "paths", fake_paths)
        predictor._load_ensemble_config()
        assert predictor.use_adaptive is False


# ─────────────────────────────────────────────────────────────
#  _save_ensemble_config
# ─────────────────────────────────────────────────────────────


class TestSaveEnsembleConfig:
    def test_creates_json_file(self, predictor, tmp_path, monkeypatch):
        import src.models.ensemble_predictor as ep_mod

        cfg_path = tmp_path / "models" / "ensemble_config.json"
        fake_paths = MagicMock()
        fake_paths.ENSEMBLE_CONFIG_PATH = cfg_path
        monkeypatch.setattr(ep_mod, "paths", fake_paths)
        predictor._save_ensemble_config()
        assert cfg_path.exists()

    def test_saved_json_has_alpha_beta(self, predictor, tmp_path, monkeypatch):
        import src.models.ensemble_predictor as ep_mod

        cfg_path = tmp_path / "models" / "ensemble_config.json"
        fake_paths = MagicMock()
        fake_paths.ENSEMBLE_CONFIG_PATH = cfg_path
        monkeypatch.setattr(ep_mod, "paths", fake_paths)
        predictor.alpha_base = 0.65
        predictor.beta_base = 0.35
        predictor._save_ensemble_config()
        with open(cfg_path) as f:
            data = json.load(f)
        assert data["alpha"] == 0.65
        assert data["beta"] == 0.35

    def test_saved_config_includes_sigmoid_params(
        self, predictor, tmp_path, monkeypatch
    ):
        import src.models.ensemble_predictor as ep_mod

        cfg_path = tmp_path / "models" / "cfg.json"
        fake_paths = MagicMock()
        fake_paths.ENSEMBLE_CONFIG_PATH = cfg_path
        monkeypatch.setattr(ep_mod, "paths", fake_paths)
        predictor._save_ensemble_config()
        with open(cfg_path) as f:
            data = json.load(f)
        assert "sigmoid_center" in data
        assert "sigmoid_slope" in data


# ─────────────────────────────────────────────────────────────
#  predict — error cases
# ─────────────────────────────────────────────────────────────


class TestPredictErrors:
    def test_raises_if_not_loaded(self, predictor):
        with pytest.raises(RuntimeError, match="load_models"):
            predictor.predict(np.zeros((64, 64, 3), dtype=np.uint8))

    def test_raises_on_invalid_image_path(self, predictor, tmp_path):
        predictor._loaded = True
        predictor.model_a = _fake_model()
        with pytest.raises((ValueError, Exception)):
            predictor.predict(str(tmp_path / "nonexistent_image.jpg"))
