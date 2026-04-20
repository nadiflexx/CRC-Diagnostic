# tests/test_tabular_preprocessor.py
"""
Tests for src/data/processing/tabular_preprocessor.py
"""

import numpy as np
import pandas as pd
import pytest

from src.config.constants import CLINICAL_NUMERIC_FEATURES, TABULAR_TARGET
from src.data.processing.tabular_preprocessor import TabularPreprocessor

# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df():
    """DataFrame with all expected numeric features + Diagnosis target."""
    n = 80
    rng = np.random.default_rng(0)
    data = {col: rng.random(n) for col in CLINICAL_NUMERIC_FEATURES}
    data[TABULAR_TARGET] = rng.integers(0, 2, size=n)
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor():
    return TabularPreprocessor()


@pytest.fixture
def fitted_preprocessor(preprocessor, sample_df):
    preprocessor.fit_transform(sample_df)
    return preprocessor


# ─────────────────────────────────────────────────────────────
#  __init__
# ─────────────────────────────────────────────────────────────


class TestInit:
    def test_not_fitted_initially(self, preprocessor):
        assert preprocessor._fitted is False

    def test_feature_names_empty_initially(self, preprocessor):
        assert preprocessor.feature_names == []

    def test_scaler_is_standard_scaler(self, preprocessor):
        from sklearn.preprocessing import StandardScaler

        assert isinstance(preprocessor.scaler, StandardScaler)


# ─────────────────────────────────────────────────────────────
#  fit_transform
# ─────────────────────────────────────────────────────────────


class TestFitTransform:
    def test_returns_three_elements(self, preprocessor, sample_df):
        result = preprocessor.fit_transform(sample_df)
        assert len(result) == 3

    def test_x_is_ndarray(self, preprocessor, sample_df):
        X, y, names = preprocessor.fit_transform(sample_df)
        assert isinstance(X, np.ndarray)

    def test_y_is_ndarray(self, preprocessor, sample_df):
        X, y, names = preprocessor.fit_transform(sample_df)
        assert isinstance(y, np.ndarray)

    def test_y_shape_matches_rows(self, preprocessor, sample_df):
        X, y, _ = preprocessor.fit_transform(sample_df)
        assert len(y) == len(sample_df)

    def test_x_shape_rows_match(self, preprocessor, sample_df):
        X, y, _ = preprocessor.fit_transform(sample_df)
        assert X.shape[0] == len(sample_df)

    def test_x_shape_cols_match_features(self, preprocessor, sample_df):
        X, y, names = preprocessor.fit_transform(sample_df)
        assert X.shape[1] == len(names)

    def test_sets_fitted_flag(self, preprocessor, sample_df):
        preprocessor.fit_transform(sample_df)
        assert preprocessor._fitted is True

    def test_feature_names_populated(self, preprocessor, sample_df):
        preprocessor.fit_transform(sample_df)
        assert len(preprocessor.feature_names) > 0

    def test_y_is_none_without_target(self, preprocessor, sample_df):
        df_no_target = sample_df.drop(columns=[TABULAR_TARGET])
        X, y, _ = preprocessor.fit_transform(df_no_target)
        assert y is None

    def test_scaled_mean_near_zero(self, preprocessor, sample_df):
        X, _, _ = preprocessor.fit_transform(sample_df)
        np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-6)

    def test_scaled_std_near_one(self, preprocessor, sample_df):
        X, _, _ = preprocessor.fit_transform(sample_df)
        np.testing.assert_allclose(X.std(axis=0), 1.0, atol=1e-6)

    def test_handles_missing_features_gracefully(self, preprocessor):
        """Only available feature columns are used."""
        df = pd.DataFrame(
            {
                CLINICAL_NUMERIC_FEATURES[0]: [1.0, 2.0],
                TABULAR_TARGET: [0, 1],
            }
        )
        X, y, names = preprocessor.fit_transform(df)
        assert X.shape[1] == 1

    def test_fills_nan_with_zero_before_scaling(self, preprocessor):
        n = 50
        data = {col: np.random.rand(n) for col in CLINICAL_NUMERIC_FEATURES[:3]}
        data[CLINICAL_NUMERIC_FEATURES[0]][0] = np.nan
        data[TABULAR_TARGET] = np.zeros(n, dtype=int)
        df = pd.DataFrame(data)
        X, _, _ = preprocessor.fit_transform(df)
        assert not np.isnan(X).any()


# ─────────────────────────────────────────────────────────────
#  transform
# ─────────────────────────────────────────────────────────────


class TestTransform:
    def test_raises_if_not_fitted(self, preprocessor, sample_df):
        with pytest.raises(RuntimeError, match="not fitted"):
            preprocessor.transform(sample_df)

    def test_returns_ndarray(self, fitted_preprocessor, sample_df):
        X = fitted_preprocessor.transform(sample_df)
        assert isinstance(X, np.ndarray)

    def test_shape_matches_fit_transform(self, fitted_preprocessor, sample_df):
        X_fit, _, _ = fitted_preprocessor.fit_transform(sample_df)
        X_tr = fitted_preprocessor.transform(sample_df)
        assert X_fit.shape == X_tr.shape

    def test_consistent_with_fit_transform(self, sample_df):
        """transform on training data ≈ fit_transform output."""
        pp = TabularPreprocessor()
        X_fit, _, _ = pp.fit_transform(sample_df)
        X_tr = pp.transform(sample_df)
        np.testing.assert_allclose(X_fit, X_tr, atol=1e-10)

    def test_handles_nan_in_new_data(self, fitted_preprocessor, sample_df):
        new_df = sample_df.copy()
        new_df.iloc[0, 0] = np.nan
        X = fitted_preprocessor.transform(new_df)
        assert not np.isnan(X).any()


# ─────────────────────────────────────────────────────────────
#  save / load
# ─────────────────────────────────────────────────────────────


class TestSaveLoad:
    def test_save_creates_file(self, fitted_preprocessor, tmp_path):
        out = tmp_path / "pp.joblib"
        fitted_preprocessor.save(out)
        assert out.exists()

    def test_load_restores_feature_names(self, fitted_preprocessor, tmp_path):
        out = tmp_path / "pp.joblib"
        fitted_preprocessor.save(out)
        loaded = TabularPreprocessor.load(out)
        assert loaded.feature_names == fitted_preprocessor.feature_names

    def test_load_is_fitted(self, fitted_preprocessor, tmp_path):
        out = tmp_path / "pp.joblib"
        fitted_preprocessor.save(out)
        loaded = TabularPreprocessor.load(out)
        assert loaded._fitted is True

    def test_loaded_transform_matches_original(
        self, fitted_preprocessor, sample_df, tmp_path
    ):
        out = tmp_path / "pp.joblib"
        fitted_preprocessor.save(out)
        loaded = TabularPreprocessor.load(out)
        X_orig = fitted_preprocessor.transform(sample_df)
        X_load = loaded.transform(sample_df)
        np.testing.assert_allclose(X_orig, X_load, atol=1e-10)

    def test_load_returns_tabular_preprocessor_instance(
        self, fitted_preprocessor, tmp_path
    ):
        out = tmp_path / "pp.joblib"
        fitted_preprocessor.save(out)
        loaded = TabularPreprocessor.load(out)
        assert isinstance(loaded, TabularPreprocessor)

    def test_save_and_load_roundtrip_on_new_data(self, fitted_preprocessor, tmp_path):
        out = tmp_path / "pp.joblib"
        fitted_preprocessor.save(out)
        loaded = TabularPreprocessor.load(out)
        n = 10
        new_data = {col: np.random.rand(n) for col in fitted_preprocessor.feature_names}
        new_df = pd.DataFrame(new_data)
        X = loaded.transform(new_df)
        assert X.shape == (n, len(fitted_preprocessor.feature_names))
