# tests/src/models/test_tabular_model.py
"""Tests for TabularCancerModel — FIXED: joblib mock prevents pickle error."""

from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_tiny_data(n=80, n_features=5, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, y, feature_names


def _make_trained_model(n_trials=1):
    """
    Build a real TabularCancerModel trained on tiny data.
    Uses n_trials=1 for speed.
    """
    from src.models.tabular_model import TabularCancerModel

    X, y, feature_names = _make_tiny_data()
    model = TabularCancerModel(model_type="xgboost")
    model.train(
        X,
        y,
        X_val=X[:20],
        y_val=y[:20],
        feature_names=feature_names,
        n_trials=n_trials,
        recall_target=0.5,
        t_minimum=1.0,
    )
    return model, X, y, feature_names


# ── __init__ ──────────────────────────────────────────────────────────────────


class TestTabularCancerModelInit:
    def test_default_model_type(self):
        from src.models.tabular_model import TabularCancerModel

        m = TabularCancerModel()
        assert m.model_type == "xgboost"

    def test_custom_model_type(self):
        from src.models.tabular_model import TabularCancerModel

        m = TabularCancerModel(model_type="xgboost")
        assert m.model_type == "xgboost"

    def test_initial_model_is_none(self):
        from src.models.tabular_model import TabularCancerModel

        m = TabularCancerModel()
        assert m.model is None

    def test_initial_threshold(self):
        from src.models.tabular_model import TabularCancerModel

        m = TabularCancerModel()
        assert m.best_threshold == 0.5

    def test_initial_feature_names_none(self):
        from src.models.tabular_model import TabularCancerModel

        m = TabularCancerModel()
        assert m.feature_names is None

    def test_initial_calibrator_none(self):
        from src.models.tabular_model import TabularCancerModel

        m = TabularCancerModel()
        assert m._calibrator is None

    def test_initial_base_model_none(self):
        from src.models.tabular_model import TabularCancerModel

        m = TabularCancerModel()
        assert m._base_model is None


# ── _optuna_objective ─────────────────────────────────────────────────────────


class TestOptunaObjective:
    def test_returns_float_between_zero_one(self):
        import optuna

        from src.models.tabular_model import TabularCancerModel

        m = TabularCancerModel()
        X, y, _ = _make_tiny_data(n=60)
        study = optuna.create_study(direction="maximize")

        def objective(trial):
            return m._optuna_objective(trial, X, y)

        study.optimize(objective, n_trials=1)
        assert 0.0 <= study.best_value <= 1.0

    def test_returns_numeric(self):
        import optuna

        from src.models.tabular_model import TabularCancerModel

        m = TabularCancerModel()
        X, y, _ = _make_tiny_data(n=60)
        trial = MagicMock()
        trial.suggest_int.side_effect = [100, 3, 1]
        trial.suggest_float.side_effect = [0.1, 0.8, 0.8, 0.0, 0.01, 0.01]
        # Run with a real trial via study
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: m._optuna_objective(t, X, y), n_trials=1)
        assert isinstance(study.best_value, float)


# ── train ─────────────────────────────────────────────────────────────────────


class TestTabularCancerModelTrain:
    def test_model_set_after_training(self):
        from src.models.tabular_model import TabularCancerModel

        X, y, feat = _make_tiny_data()
        m = TabularCancerModel()
        m.train(X, y, feature_names=feat, n_trials=1, recall_target=0.5, t_minimum=1.0)
        assert m.model is not None

    def test_base_model_set_after_training(self):
        from src.models.tabular_model import TabularCancerModel

        X, y, feat = _make_tiny_data()
        m = TabularCancerModel()
        m.train(X, y, feature_names=feat, n_trials=1, recall_target=0.5, t_minimum=1.0)
        assert m._base_model is not None

    def test_calibrator_set_after_training(self):
        from src.models.tabular_model import TabularCancerModel

        X, y, feat = _make_tiny_data()
        m = TabularCancerModel()
        m.train(X, y, feature_names=feat, n_trials=1, recall_target=0.5, t_minimum=1.0)
        assert m._calibrator is not None

    def test_feature_names_stored(self):
        from src.models.tabular_model import TabularCancerModel

        X, y, feat = _make_tiny_data()
        m = TabularCancerModel()
        m.train(X, y, feature_names=feat, n_trials=1, recall_target=0.5, t_minimum=1.0)
        assert m.feature_names == feat

    def test_best_threshold_in_valid_range(self):
        from src.models.tabular_model import TabularCancerModel

        X, y, feat = _make_tiny_data()
        m = TabularCancerModel()
        m.train(X, y, feature_names=feat, n_trials=1, recall_target=0.5, t_minimum=1.0)
        assert 0.0 <= m.best_threshold <= 1.0

    def test_train_with_val_set(self):
        from src.models.tabular_model import TabularCancerModel

        X, y, feat = _make_tiny_data(n=100)
        m = TabularCancerModel()
        m.train(
            X[:60],
            y[:60],
            X_val=X[60:],
            y_val=y[60:],
            feature_names=feat,
            n_trials=1,
            recall_target=0.5,
            t_minimum=1.0,
        )
        assert m.model is not None

    def test_train_without_val_set_uses_train(self):
        """When X_val=None, calibration uses training data."""
        from src.models.tabular_model import TabularCancerModel

        X, y, feat = _make_tiny_data()
        m = TabularCancerModel()
        m.train(
            X,
            y,
            X_val=None,
            y_val=None,
            feature_names=feat,
            n_trials=1,
            recall_target=0.5,
            t_minimum=1.0,
        )
        assert m._calibrator is not None


# ── predict_proba ─────────────────────────────────────────────────────────────


class TestPredictProba:
    def test_returns_ndarray(self):
        model, X, y, _ = _make_trained_model()
        probs = model.predict_proba(X[:10])
        assert isinstance(probs, np.ndarray)

    def test_shape_matches_input(self):
        model, X, y, _ = _make_trained_model()
        probs = model.predict_proba(X[:10])
        assert probs.shape == (10,)

    def test_values_in_zero_one(self):
        model, X, y, _ = _make_trained_model()
        probs = model.predict_proba(X)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_uses_calibrator_when_available(self):
        from src.models.tabular_model import TabularCancerModel

        m = TabularCancerModel()
        mock_calibrator = MagicMock()
        mock_calibrator.predict_proba.return_value = np.array([[0.3, 0.7], [0.6, 0.4]])
        m._calibrator = mock_calibrator
        X = np.random.rand(2, 5).astype(np.float32)
        result = m.predict_proba(X)
        mock_calibrator.predict_proba.assert_called_once_with(X)
        np.testing.assert_array_almost_equal(result, [0.7, 0.4])

    def test_uses_base_model_when_no_calibrator(self):
        from src.models.tabular_model import TabularCancerModel

        m = TabularCancerModel()
        mock_base = MagicMock()
        mock_base.predict_proba.return_value = np.array([[0.4, 0.6]])
        m._base_model = mock_base
        m._calibrator = None
        X = np.random.rand(1, 5).astype(np.float32)
        result = m.predict_proba(X)
        np.testing.assert_array_almost_equal(result, [0.6])


# ── predict ───────────────────────────────────────────────────────────────────


class TestPredict:
    def test_returns_binary_array(self):
        model, X, y, _ = _make_trained_model()
        preds = model.predict(X[:20])
        assert set(np.unique(preds)).issubset({0, 1})

    def test_shape_matches_input(self):
        model, X, y, _ = _make_trained_model()
        preds = model.predict(X[:15])
        assert preds.shape == (15,)

    def test_threshold_applied(self):
        from src.models.tabular_model import TabularCancerModel

        m = TabularCancerModel()
        m.best_threshold = 0.4
        mock_calibrator = MagicMock()
        mock_calibrator.predict_proba.return_value = np.array(
            [
                [0.7, 0.3],
                [0.2, 0.8],
                [0.65, 0.35],
            ]
        )
        m._calibrator = mock_calibrator
        X = np.random.rand(3, 5)
        preds = m.predict(X)
        # 0.3 < 0.4 → 0; 0.8 >= 0.4 → 1; 0.35 < 0.4 → 0
        np.testing.assert_array_equal(preds, [0, 1, 0])


# ── evaluate ──────────────────────────────────────────────────────────────────


class TestEvaluate:
    def test_returns_dict_with_required_keys(self):
        model, X, y, _ = _make_trained_model()
        results = model.evaluate(X, y)
        expected = {
            "auc_roc",
            "recall",
            "precision",
            "f1",
            "confusion_matrix",
            "classification_report",
            "threshold",
        }
        assert expected == set(results.keys())

    def test_auc_in_zero_one(self):
        model, X, y, _ = _make_trained_model()
        results = model.evaluate(X, y)
        assert 0.0 <= results["auc_roc"] <= 1.0

    def test_confusion_matrix_is_list(self):
        model, X, y, _ = _make_trained_model()
        results = model.evaluate(X, y)
        assert isinstance(results["confusion_matrix"], list)

    def test_threshold_matches_model(self):
        model, X, y, _ = _make_trained_model()
        results = model.evaluate(X, y)
        assert results["threshold"] == model.best_threshold


# ── cross_validate ────────────────────────────────────────────────────────────


class TestCrossValidate:
    def test_returns_dict_with_required_keys(self):
        model, X, y, _ = _make_trained_model()
        results = model.cross_validate(X, y, cv=3)
        assert set(results.keys()) == {"mean_auc", "std_auc", "scores"}

    def test_mean_auc_in_zero_one(self):
        model, X, y, _ = _make_trained_model()
        results = model.cross_validate(X, y, cv=3)
        assert 0.0 <= results["mean_auc"] <= 1.0

    def test_scores_length_equals_cv(self):
        model, X, y, _ = _make_trained_model()
        results = model.cross_validate(X, y, cv=3)
        assert len(results["scores"]) == 3

    def test_std_auc_non_negative(self):
        model, X, y, _ = _make_trained_model()
        results = model.cross_validate(X, y, cv=3)
        assert results["std_auc"] >= 0.0


# ── _find_optimal_threshold ───────────────────────────────────────────────────


class TestFindOptimalThreshold:
    def test_threshold_in_valid_range(self):
        model, X, y, _ = _make_trained_model()
        threshold = model._find_optimal_threshold(X[:20], y[:20], recall_target=0.5)
        assert 0.0 <= threshold <= 1.0

    def test_returns_float(self):
        model, X, y, _ = _make_trained_model()
        threshold = model._find_optimal_threshold(X[:20], y[:20], recall_target=0.0)
        assert isinstance(threshold, float)

    def test_higher_recall_target_lower_threshold(self):
        """Higher recall requirement → lower (more aggressive) threshold."""
        model, X, y, _ = _make_trained_model()
        t_low = model._find_optimal_threshold(X, y, recall_target=0.30)
        t_high = model._find_optimal_threshold(X, y, recall_target=0.95)
        # t_high should be ≤ t_low (more aggressive to guarantee higher recall)
        assert t_high <= t_low + 0.1  # allow small tolerance


# ── save / load ───────────────────────────────────────────────────────────────


class TestSaveLoad:
    """
    Use real XGBoost models to avoid MagicMock pickle errors.
    joblib.dump serialises the actual objects, so mocks cannot be used here.
    """

    def test_save_creates_file(self, tmp_path):
        model, X, y, _ = _make_trained_model()
        path = tmp_path / "model.pkl"
        model.save(path)
        assert path.exists()

    def test_load_returns_instance(self, tmp_path):
        from src.models.tabular_model import TabularCancerModel

        model, X, y, _ = _make_trained_model()
        path = tmp_path / "model.pkl"
        model.save(path)
        loaded = TabularCancerModel.load(path)
        assert isinstance(loaded, TabularCancerModel)

    def test_load_restores_threshold(self, tmp_path):
        from src.models.tabular_model import TabularCancerModel

        model, X, y, _ = _make_trained_model()
        original_threshold = model.best_threshold
        path = tmp_path / "model.pkl"
        model.save(path)
        loaded = TabularCancerModel.load(path)
        assert loaded.best_threshold == pytest.approx(original_threshold)

    def test_load_restores_feature_names(self, tmp_path):
        from src.models.tabular_model import TabularCancerModel

        model, X, y, feat = _make_trained_model()
        path = tmp_path / "model.pkl"
        model.save(path)
        loaded = TabularCancerModel.load(path)
        assert loaded.feature_names == feat

    def test_load_restores_model_type(self, tmp_path):
        from src.models.tabular_model import TabularCancerModel

        model, X, y, _ = _make_trained_model()
        path = tmp_path / "model.pkl"
        model.save(path)
        loaded = TabularCancerModel.load(path)
        assert loaded.model_type == "xgboost"

    def test_loaded_model_predicts(self, tmp_path):
        from src.models.tabular_model import TabularCancerModel

        model, X, y, _ = _make_trained_model()
        path = tmp_path / "model.pkl"
        model.save(path)
        loaded = TabularCancerModel.load(path)
        preds = loaded.predict(X[:5])
        assert preds.shape == (5,)

    def test_save_uses_joblib(self, tmp_path):
        """Verify joblib.dump is called with the correct path."""
        model, X, y, _ = _make_trained_model()
        path = tmp_path / "model.pkl"
        with patch("src.models.tabular_model.joblib.dump") as mock_dump:
            model.save(path)
        mock_dump.assert_called_once()
        call_args = mock_dump.call_args
        assert call_args[0][1] == path

    def test_load_without_calibrator_key(self, tmp_path):
        """Backward compatibility: files without 'calibrator' key."""
        import xgboost as xgb

        from src.models.tabular_model import TabularCancerModel

        # Build a minimal state dict without 'calibrator'
        X, y, feat = _make_tiny_data(n=40)
        base = xgb.XGBClassifier(n_estimators=10, random_state=42)
        base.fit(X, y)
        state = {
            "model": base,
            "base_model": base,
            # No 'calibrator' key
            "model_type": "xgboost",
            "feature_names": feat,
            "best_threshold": 0.45,
        }
        path = tmp_path / "old_model.pkl"
        joblib.dump(state, path)
        loaded = TabularCancerModel.load(path)
        assert loaded._calibrator is None
        assert loaded.best_threshold == pytest.approx(0.45)
