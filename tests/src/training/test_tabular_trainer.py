# tests/src/training/test_tabular_trainer.py
"""Tests for train_tabular.py — FIXED"""

from unittest.mock import MagicMock, patch

import numpy as np

# ── validate_data_quality ─────────────────────────────────────────────────────


class TestValidateDataQuality:
    def _import(self):
        from src.training.train_tabular import validate_data_quality

        return validate_data_quality

    def test_returns_true_when_no_perfect_separation(self):
        """Features overlap between classes → no leakage."""
        validate_data_quality = self._import()
        # Both classes share overlapping ranges in all features
        X = np.array(
            [
                [1.0, 5.0],
                [2.0, 6.0],
                [1.5, 5.5],  # label=1 overlaps with label=0
                [1.8, 5.8],
            ]
        )
        y = np.array([0, 0, 1, 1])
        result = validate_data_quality(X, y, ["feat_a", "feat_b"])
        assert result is True

    def test_returns_false_when_perfect_separation_exists(self):
        """Non-overlapping ranges → leakage detected."""
        validate_data_quality = self._import()
        X = np.array(
            [
                [0.0, 5.0],
                [1.0, 5.0],
                [10.0, 5.0],
                [11.0, 5.0],
            ]
        )
        y = np.array([0, 0, 1, 1])
        result = validate_data_quality(X, y, ["sep_feat", "ok_feat"])
        assert result is False

    def test_identifies_single_perfectly_separated_feature(self):
        validate_data_quality = self._import()
        X = np.array([[0.0], [1.0], [100.0], [200.0]])
        y = np.array([0, 0, 1, 1])
        result = validate_data_quality(X, y, ["bad_feat"])
        assert result is False

    def test_all_features_overlapping_returns_true(self):
        validate_data_quality = self._import()
        rng = np.random.RandomState(42)
        # Generate data with guaranteed overlap: both classes centered at 0
        X = rng.randn(100, 5)
        y = np.array([0] * 50 + [1] * 50)
        result = validate_data_quality(X, y, [f"f{i}" for i in range(5)])
        assert result is True

    def test_logs_warning_on_leakage(self):
        validate_data_quality = self._import()
        X = np.array([[0.0], [1.0], [100.0], [200.0]])
        y = np.array([0, 0, 1, 1])
        with patch("src.training.train_tabular.logger") as mock_log:
            validate_data_quality(X, y, ["bad"])
        mock_log.warning.assert_called_once()

    def test_logs_info_when_clean(self):
        """When no separation is found, logger.info should be called."""
        validate_data_quality = self._import()
        # Perfect overlap: same value for both classes in each feature
        X = np.array(
            [
                [5.0, 5.0],
                [5.0, 5.0],
            ]
        )
        y = np.array([0, 1])
        # max_healthy = min_healthy = 5.0, max_cancer = min_cancer = 5.0
        # healthy_max (5) < cancer_min (5)? No → no separation
        with patch("src.training.train_tabular.logger") as mock_log:
            result = validate_data_quality(X, y, ["a", "b"])
        assert result is True
        mock_log.info.assert_called_once()

    def test_single_sample_per_class_no_separation(self):
        """Same value for both classes → ranges overlap at a point."""
        validate_data_quality = self._import()
        X = np.array([[5.0], [5.0]])
        y = np.array([0, 1])
        result = validate_data_quality(X, y, ["feat"])
        assert result is True

    def test_returns_bool(self):
        validate_data_quality = self._import()
        X = np.array([[1.0, 2.0], [1.5, 1.5]])
        y = np.array([0, 1])
        result = validate_data_quality(X, y, ["f1", "f2"])
        assert isinstance(result, bool)

    def test_cancer_max_less_than_healthy_min_is_separation(self):
        """cancer range entirely below healthy range → separated."""
        validate_data_quality = self._import()
        X = np.array(
            [
                [100.0],  # healthy
                [110.0],  # healthy
                [0.0],  # cancer
                [5.0],  # cancer
            ]
        )
        y = np.array([0, 0, 1, 1])
        result = validate_data_quality(X, y, ["feat"])
        assert result is False

    def test_multiple_features_partial_separation(self):
        """Only one of two features is separated → still returns False."""
        validate_data_quality = self._import()
        X = np.array(
            [
                [1.0, 50.0],  # healthy
                [2.0, 60.0],  # healthy
                [100.0, 55.0],  # cancer — feat0 separated, feat1 overlaps
                [110.0, 52.0],  # cancer
            ]
        )
        y = np.array([0, 0, 1, 1])
        result = validate_data_quality(X, y, ["sep", "ok"])
        assert result is False


# ── train_tabular_pipeline (smoke tests) ─────────────────────────────────────


class TestTrainTabularPipeline:
    def _make_mocks(self):
        X = np.random.rand(100, 5).astype(np.float32)
        y = np.array([0] * 50 + [1] * 50)
        feature_names = [f"feat_{i}" for i in range(5)]

        mock_gen = MagicMock()
        mock_gen.generate_and_validate.return_value = MagicMock()

        mock_prep = MagicMock()
        mock_prep.fit_transform.return_value = (X, y, feature_names)
        mock_prep.save = MagicMock()

        mock_model = MagicMock()
        mock_model.train = MagicMock()
        mock_model.evaluate.return_value = {"accuracy": 0.9, "f1": 0.88}
        mock_model.cross_validate = MagicMock()
        mock_model.save = MagicMock()

        mock_explainer = MagicMock()
        mock_explainer.fit = MagicMock()
        mock_explainer.explain.return_value = {
            "top_risk_factors": [("f1", 0.5), ("f2", 0.3), ("f3", 0.1)]
        }
        mock_explainer.plot_explanation = MagicMock()

        return mock_gen, mock_prep, mock_model, mock_explainer, X, y, feature_names

    def _run_pipeline(self, mock_gen, mock_prep, mock_model, mock_explainer):
        from pathlib import Path

        with (
            patch(
                "src.training.train_tabular.ClinicalDataGenerator",
                return_value=mock_gen,
            ),
            patch(
                "src.training.train_tabular.TabularPreprocessor", return_value=mock_prep
            ),
            patch(
                "src.training.train_tabular.TabularCancerModel", return_value=mock_model
            ),
            patch(
                "src.training.train_tabular.TabularExplainer",
                return_value=mock_explainer,
            ),
            patch(
                "src.training.train_tabular.validate_data_quality", return_value=True
            ),
            patch("src.training.train_tabular.paths") as mock_paths,
        ):
            mock_paths.MODELS = MagicMock()
            mock_paths.MODELS.mkdir = MagicMock()
            mock_paths.MODELS.__str__ = lambda _: "/tmp/models"
            mock_paths.MODELS.__truediv__ = lambda s, o: Path("/tmp") / o
            mock_paths.TABULAR_MODEL_PATH = Path("/tmp/model.pkl")
            mock_paths.TABULAR_PREPROCESSOR_PATH = Path("/tmp/prep.pkl")

            from src.training.train_tabular import train_tabular_pipeline

            return train_tabular_pipeline()

    def test_returns_three_tuple(self):
        mocks = self._make_mocks()
        result = self._run_pipeline(*mocks[:4])
        assert len(result) == 3

    def test_returns_model_instance(self):
        mocks = self._make_mocks()
        model, _, _ = self._run_pipeline(*mocks[:4])
        assert model is mocks[2]

    def test_returns_preprocessor_instance(self):
        mocks = self._make_mocks()
        _, prep, _ = self._run_pipeline(*mocks[:4])
        assert prep is mocks[1]

    def test_model_train_called(self):
        mocks = self._make_mocks()
        self._run_pipeline(*mocks[:4])
        mocks[2].train.assert_called_once()

    def test_model_evaluate_called(self):
        mocks = self._make_mocks()
        self._run_pipeline(*mocks[:4])
        mocks[2].evaluate.assert_called_once()

    def test_model_save_called(self):
        mocks = self._make_mocks()
        self._run_pipeline(*mocks[:4])
        mocks[2].save.assert_called_once()

    def test_preprocessor_save_called(self):
        mocks = self._make_mocks()
        self._run_pipeline(*mocks[:4])
        mocks[1].save.assert_called_once()

    def test_explainer_fit_called(self):
        mocks = self._make_mocks()
        self._run_pipeline(*mocks[:4])
        mocks[3].fit.assert_called_once()

    def test_cross_validate_called(self):
        mocks = self._make_mocks()
        self._run_pipeline(*mocks[:4])
        mocks[2].cross_validate.assert_called_once()

    def test_results_dict_returned(self):
        mocks = self._make_mocks()
        _, _, results = self._run_pipeline(*mocks[:4])
        assert results == {"accuracy": 0.9, "f1": 0.88}

    def test_generator_called(self):
        mocks = self._make_mocks()
        self._run_pipeline(*mocks[:4])
        mocks[0].generate_and_validate.assert_called_once()

    def test_preprocessor_fit_transform_called(self):
        mocks = self._make_mocks()
        self._run_pipeline(*mocks[:4])
        mocks[1].fit_transform.assert_called_once()

    def test_explainer_plot_called(self):
        mocks = self._make_mocks()
        self._run_pipeline(*mocks[:4])
        mocks[3].plot_explanation.assert_called_once()
