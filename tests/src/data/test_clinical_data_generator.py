# tests/test_clinical_data_generator.py
"""
Tests for src/data/processing/clinical_data_generator.py
"""

import numpy as np
import pandas as pd
import pytest

from src.data.processing.clinical_data_generator import (
    ClinicalDataGenerator,
    _corr_to_cov,
    _generate_features,
    validate_distributions,
)

# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def base_df():
    """Minimal base DataFrame for feature generation."""
    n = 100
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Age": rng.integers(30, 80, size=n),
            "Diagnosis": rng.integers(0, 2, size=n),
            "Gender": rng.integers(0, 2, size=n),
            "Smoking_History": rng.integers(0, 2, size=n),
            "Inflammatory_Bowel_Disease": rng.integers(0, 2, size=n),
            "Genetic_Mutation": rng.integers(0, 2, size=n),
        }
    )


@pytest.fixture
def sample_csv(tmp_path, base_df):
    """Write base_df to a temporary CSV and return its path."""
    csv = tmp_path / "colorectal_cancer_cleaned.csv"
    base_df.to_csv(csv, index=False)
    return csv


@pytest.fixture
def generator(sample_csv):
    return ClinicalDataGenerator(seed=42, csv_path=sample_csv)


# ─────────────────────────────────────────────────────────────
#  _corr_to_cov
# ─────────────────────────────────────────────────────────────


class TestCorrToCov:
    def test_output_shape(self):
        corr = np.eye(3)
        stds = np.array([1.0, 2.0, 3.0])
        cov = _corr_to_cov(corr, stds)
        assert cov.shape == (3, 3)

    def test_identity_corr_gives_diagonal_cov(self):
        corr = np.eye(4)
        stds = np.array([1.0, 2.0, 3.0, 4.0])
        cov = _corr_to_cov(corr, stds)
        expected = np.diag(stds**2)
        np.testing.assert_allclose(cov, expected)

    def test_symmetry(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        stds = np.array([2.0, 3.0])
        cov = _corr_to_cov(corr, stds)
        np.testing.assert_allclose(cov, cov.T)

    def test_positive_semidefinite(self):
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        stds = np.array([1.5, 2.5])
        cov = _corr_to_cov(corr, stds)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)


# ─────────────────────────────────────────────────────────────
#  _generate_features
# ─────────────────────────────────────────────────────────────


class TestGenerateFeatures:
    def test_output_columns(self, base_df):
        rng = np.random.default_rng(42)
        df = _generate_features(base_df, rng)
        expected_cols = {
            "Patient_ID",
            "Age",
            "Smoking_History",
            "CEA_Level_ng_mL",
            "Hemoglobin_g_dL",
            "PyRad_ADC_Mean",
            "PyRad_ADC_Std",
            "PyRad_Entropy",
            "PyRad_GLCM_Contrast",
            "PyRad_GLCM_Homogeneity",
            "PyRad_Shape_Sphericity",
            "PyRad_FirstOrder_Skewness",
            "Diagnosis",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_output_row_count_matches_input(self, base_df):
        rng = np.random.default_rng(42)
        df = _generate_features(base_df, rng)
        assert len(df) == len(base_df)

    def test_cea_within_clip_bounds(self, base_df):
        rng = np.random.default_rng(42)
        df = _generate_features(base_df, rng)
        assert df["CEA_Level_ng_mL"].min() >= 0.10
        assert df["CEA_Level_ng_mL"].max() <= 5000.0

    def test_hemoglobin_within_clip_bounds(self, base_df):
        rng = np.random.default_rng(42)
        df = _generate_features(base_df, rng)
        assert df["Hemoglobin_g_dL"].min() >= 5.0
        assert df["Hemoglobin_g_dL"].max() <= 20.0

    def test_adc_mean_within_clip_bounds(self, base_df):
        rng = np.random.default_rng(42)
        df = _generate_features(base_df, rng)
        assert df["PyRad_ADC_Mean"].min() >= 200.0
        assert df["PyRad_ADC_Mean"].max() <= 2500.0

    def test_homogeneity_within_bounds(self, base_df):
        rng = np.random.default_rng(42)
        df = _generate_features(base_df, rng)
        assert df["PyRad_GLCM_Homogeneity"].min() >= 0.01
        assert df["PyRad_GLCM_Homogeneity"].max() <= 1.0

    def test_sphericity_within_bounds(self, base_df):
        rng = np.random.default_rng(42)
        df = _generate_features(base_df, rng)
        assert df["PyRad_Shape_Sphericity"].min() >= 0.25
        assert df["PyRad_Shape_Sphericity"].max() <= 1.0

    def test_patient_id_auto_generated(self):
        df_base = pd.DataFrame({"Age": [50, 60], "Diagnosis": [0, 1]})
        rng = np.random.default_rng(1)
        df = _generate_features(df_base, rng)
        assert "Patient_ID" in df.columns
        assert all(df["Patient_ID"].str.startswith("PT-"))

    def test_existing_patient_id_preserved(self):
        df_base = pd.DataFrame(
            {"Patient_ID": ["P001", "P002"], "Age": [50, 60], "Diagnosis": [0, 1]}
        )
        rng = np.random.default_rng(1)
        df = _generate_features(df_base, rng)
        assert list(df["Patient_ID"]) == ["P001", "P002"]

    def test_diagnosis_preserved(self, base_df):
        rng = np.random.default_rng(42)
        df = _generate_features(base_df, rng)
        np.testing.assert_array_equal(
            df["Diagnosis"].values, base_df["Diagnosis"].values
        )

    def test_adc_std_within_bounds(self, base_df):
        rng = np.random.default_rng(42)
        df = _generate_features(base_df, rng)
        assert df["PyRad_ADC_Std"].min() >= 5.0
        assert df["PyRad_ADC_Std"].max() <= 400.0

    def test_entropy_within_bounds(self, base_df):
        rng = np.random.default_rng(42)
        df = _generate_features(base_df, rng)
        assert df["PyRad_Entropy"].min() >= 0.1
        assert df["PyRad_Entropy"].max() <= 10.0

    def test_no_nan_in_output(self, base_df):
        rng = np.random.default_rng(42)
        df = _generate_features(base_df, rng)
        assert not df.isnull().any().any()

    def test_reproducible_with_same_seed(self, base_df):
        df1 = _generate_features(base_df, np.random.default_rng(99))
        df2 = _generate_features(base_df, np.random.default_rng(99))
        pd.testing.assert_frame_equal(df1, df2)

    def test_only_cancer_group_has_elevated_cea(self, base_df):
        """Cancer group median CEA should exceed healthy median."""
        rng = np.random.default_rng(42)
        df = _generate_features(base_df, rng)
        cancer_median = df[df["Diagnosis"] == 1]["CEA_Level_ng_mL"].median()
        healthy_median = df[df["Diagnosis"] == 0]["CEA_Level_ng_mL"].median()
        assert cancer_median > healthy_median


# ─────────────────────────────────────────────────────────────
#  validate_distributions
# ─────────────────────────────────────────────────────────────


class TestValidateDistributions:
    def test_runs_without_error(self, base_df):
        rng = np.random.default_rng(42)
        df = _generate_features(base_df, rng)
        validate_distributions(df)  # should not raise

    def test_handles_all_zeros(self):
        """Should not crash even with degenerate data."""
        df = pd.DataFrame(
            {
                "Diagnosis": [0, 1],
                "CEA_Level_ng_mL": [0.0, 0.0],
                "Hemoglobin_g_dL": [0.0, 0.0],
                "PyRad_ADC_Mean": [0.0, 0.0],
                "PyRad_ADC_Std": [0.0, 0.0],
                "PyRad_Entropy": [0.0, 0.0],
                "PyRad_GLCM_Contrast": [0.0, 0.0],
                "PyRad_GLCM_Homogeneity": [0.0, 0.0],
                "PyRad_Shape_Sphericity": [0.0, 0.0],
                "PyRad_FirstOrder_Skewness": [0.0, 0.0],
            }
        )
        validate_distributions(df)  # should not raise


# ─────────────────────────────────────────────────────────────
#  ClinicalDataGenerator
# ─────────────────────────────────────────────────────────────


class TestClinicalDataGenerator:
    def test_init_default_seed(self, sample_csv):
        gen = ClinicalDataGenerator(csv_path=sample_csv)
        assert gen.seed == 42

    def test_init_custom_seed(self, sample_csv):
        gen = ClinicalDataGenerator(seed=7, csv_path=sample_csv)
        assert gen.seed == 7

    def test_generate_returns_dataframe(self, generator):
        df = generator.generate()
        assert isinstance(df, pd.DataFrame)

    def test_generate_raises_if_csv_missing(self, tmp_path):
        gen = ClinicalDataGenerator(csv_path=tmp_path / "missing.csv")
        with pytest.raises(FileNotFoundError):
            gen.generate()

    def test_generate_has_correct_columns(self, generator):
        df = generator.generate()
        assert "Diagnosis" in df.columns
        assert "CEA_Level_ng_mL" in df.columns

    def test_generate_row_count_matches_csv(self, generator, base_df):
        df = generator.generate()
        assert len(df) == len(base_df)

    def test_generate_and_validate_returns_dataframe(self, generator):
        df = generator.generate_and_validate()
        assert isinstance(df, pd.DataFrame)

    def test_reproducibility_same_seed(self, sample_csv):
        gen1 = ClinicalDataGenerator(seed=0, csv_path=sample_csv)
        gen2 = ClinicalDataGenerator(seed=0, csv_path=sample_csv)
        df1 = gen1.generate()
        df2 = gen2.generate()
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self, sample_csv):
        gen1 = ClinicalDataGenerator(seed=1, csv_path=sample_csv)
        gen2 = ClinicalDataGenerator(seed=2, csv_path=sample_csv)
        df1 = gen1.generate()
        df2 = gen2.generate()
        # CEA values should differ
        assert not np.allclose(
            df1["CEA_Level_ng_mL"].values,
            df2["CEA_Level_ng_mL"].values,
        )

    def test_generate_no_nans(self, generator):
        df = generator.generate()
        assert not df.isnull().any().any()

    def test_diagnosis_values_are_binary(self, generator):
        df = generator.generate()
        assert set(df["Diagnosis"].unique()).issubset({0, 1})
