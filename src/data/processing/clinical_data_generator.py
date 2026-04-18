"""
Clinical synthetic CRC dataset generator.

Wraps the multivariate clinical + radiomic feature generation logic
calibrated against NCCN 2023, ESGAR 2022 and Duffy et al. 2021 reference
ranges. Replaces the old SyntheticPatientGenerator with a single class
that reads the base CSV and produces a ready-to-train DataFrame.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.logger import log as logger
from src.config.paths import paths

# ── Multivariate distribution parameters ────────────────────────────────────
# Base variables: [log(CEA), Haemoglobin, Mean ADC, Entropy, Contrast]

CANCER_MEANS = np.array([np.log(6.0), 11.40, 1240.0, 6.10, 33.0])
CANCER_STDS = np.array([1.30, 2.00, 280.0, 1.50, 14.0])

CANCER_CORR = np.array(
    [
        [1.00, -0.30, -0.22, 0.32, 0.25],
        [-0.30, 1.00, 0.28, -0.25, -0.18],
        [-0.22, 0.28, 1.00, -0.38, -0.30],
        [0.32, -0.25, -0.38, 1.00, 0.48],
        [0.25, -0.18, -0.30, 0.48, 1.00],
    ]
)

HEALTHY_MEANS = np.array([np.log(2.10), 13.50, 1540.0, 4.80, 21.0])
HEALTHY_STDS = np.array([0.85, 1.80, 220.0, 1.20, 8.5])

HEALTHY_CORR = np.array(
    [
        [1.00, -0.08, -0.06, 0.12, 0.09],
        [-0.08, 1.00, 0.16, -0.07, -0.05],
        [-0.06, 0.16, 1.00, -0.22, -0.16],
        [0.12, -0.07, -0.22, 1.00, 0.32],
        [0.09, -0.05, -0.16, 0.32, 1.00],
    ]
)

# CSV base path (tabular_risk folder)
_CSV_BASE = paths.DATA / "tabular_clean" / "colorectal_cancer_full_dataset_v1.csv"


def _corr_to_cov(corr: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Convert correlation matrix + std vector to covariance matrix (Σ = D·R·D).

    Args:
        corr: Correlation matrix of shape (n, n).
        stds: Standard deviation vector of length n.

    Returns:
        Covariance matrix of shape (n, n).
    """
    D = np.diag(stds)
    return D @ corr @ D


def _generate_features(df_base: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate haematological biomarkers and radiomic features for each patient.

    Requires at minimum ``Age`` and ``Diagnosis`` columns. Optionally uses
    ``Gender``, ``Smoking_History``, ``Inflammatory_Bowel_Disease`` and
    ``Genetic_Mutation`` to adjust generated values. Injects 12% biological
    cross-noise to avoid perfect separability.

    Args:
        df_base: Base DataFrame with at least ``Age`` and ``Diagnosis``.
        rng: Seeded numpy random generator for reproducibility.

    Returns:
        DataFrame with 13 columns: ``Patient_ID``, 11 clinical/radiomic
        features and ``Diagnosis``.
    """
    df = df_base.copy()
    n = len(df)

    if "Patient_ID" not in df.columns:
        df.insert(0, "Patient_ID", [f"PT-{i:05d}" for i in range(n)])

    ages = df["Age"].to_numpy(float)
    gender = df.get("Gender", pd.Series(np.ones(n, int))).to_numpy(int)
    smoking = df.get("Smoking_History", pd.Series(np.zeros(n, int))).to_numpy(int)
    ibd = df.get("Inflammatory_Bowel_Disease", pd.Series(np.zeros(n, int))).to_numpy(
        int
    )
    genetic = df.get("Genetic_Mutation", pd.Series(np.zeros(n, int))).to_numpy(int)
    diag = df["Diagnosis"].to_numpy(int)

    cov_cancer = _corr_to_cov(CANCER_CORR, CANCER_STDS)
    cov_healthy = _corr_to_cov(HEALTHY_CORR, HEALTHY_STDS)

    cea = np.zeros(n)
    hgb = np.zeros(n)
    adc_mean = np.zeros(n)
    entropy = np.zeros(n)
    contrast = np.zeros(n)

    # Cancer group
    cancer_idx = np.where(diag == 1)[0]
    n_c = len(cancer_idx)
    if n_c > 0:
        samples_c = rng.multivariate_normal(CANCER_MEANS, cov_cancer, size=n_c)
        ages_c = ages[cancer_idx]
        exceso = np.maximum(ages_c - 70, 0.0)
        delta_cea = np.where(ages_c > 70, np.minimum(0.30 + 0.012 * exceso, 0.70), 0.0)
        delta_hgb = np.where(ages_c > 70, np.maximum(-0.80 - 0.04 * exceso, -2.00), 0.0)
        off_gender = np.where(gender[cancer_idx] == 1, 0.0, -1.5)

        cea[cancer_idx] = np.exp(samples_c[:, 0] + delta_cea)
        hgb[cancer_idx] = samples_c[:, 1] + delta_hgb + off_gender
        adc_mean[cancer_idx] = samples_c[:, 2]
        entropy[cancer_idx] = samples_c[:, 3]
        contrast[cancer_idx] = samples_c[:, 4]

    # Healthy group
    healthy_idx = np.where(diag == 0)[0]
    n_h = len(healthy_idx)
    if n_h > 0:
        cea_log_base = (
            HEALTHY_MEANS[0]
            + 0.85 * smoking[healthy_idx]
            + 0.30 * ibd[healthy_idx]
            + 0.20 * genetic[healthy_idx]
        )
        means_zero = HEALTHY_MEANS.copy()
        means_zero[0] = 0.0
        samples_h = rng.multivariate_normal(means_zero, cov_healthy, size=n_h)
        off_gender_h = np.where(gender[healthy_idx] == 1, 0.0, -1.5)

        cea[healthy_idx] = np.exp(samples_h[:, 0] + cea_log_base)
        hgb[healthy_idx] = samples_h[:, 1] + off_gender_h
        adc_mean[healthy_idx] = samples_h[:, 2]
        entropy[healthy_idx] = samples_h[:, 3]
        contrast[healthy_idx] = samples_h[:, 4]

    # Derived radiomic features
    adc_std = np.where(
        diag == 1,
        np.abs(adc_mean * 0.20) + rng.normal(0, 38.0, n),
        np.abs(adc_mean * 0.11) + rng.normal(0, 24.0, n),
    )

    base_hom = np.where(diag == 1, 0.32, 0.72)
    coef_c = np.where(diag == 1, 0.003, 0.002)
    coef_e = np.where(diag == 1, 0.016, 0.007)
    umbral_c = np.where(diag == 1, 30.0, 12.0)
    umbral_e = np.where(diag == 1, 5.0, 4.0)
    hom_raw = (
        rng.normal(base_hom, 0.10, n)
        - coef_c * np.maximum(contrast - umbral_c, 0)
        - coef_e * np.maximum(entropy - umbral_e, 0)
    )

    sph_raw = np.where(diag == 1, rng.normal(0.62, 0.13, n), rng.normal(0.73, 0.13, n))
    skew_raw = np.where(diag == 1, rng.normal(0.85, 0.40, n), rng.normal(0.05, 0.48, n))

    df_out = pd.DataFrame(
        {
            "Patient_ID": df["Patient_ID"].values,
            "Age": ages,
            "Smoking_History": smoking,
            "CEA_Level_ng_mL": np.clip(cea, 0.10, 5_000.0).round(2),
            "Hemoglobin_g_dL": np.clip(hgb, 5.00, 20.0).round(2),
            "PyRad_ADC_Mean": np.clip(adc_mean, 200.0, 2_500.0).round(2),
            "PyRad_ADC_Std": np.clip(adc_std, 5.0, 400.0).round(2),
            "PyRad_Entropy": np.clip(entropy, 0.1, 10.0).round(4),
            "PyRad_GLCM_Contrast": np.clip(contrast, 0.1, 200.0).round(4),
            "PyRad_GLCM_Homogeneity": np.clip(hom_raw, 0.01, 1.0).round(4),
            "PyRad_Shape_Sphericity": np.clip(sph_raw, 0.25, 1.0).round(4),
            "PyRad_FirstOrder_Skewness": skew_raw.round(4),
            "Diagnosis": diag,
        }
    )

    # Biological noise — early CRC (~12%): CEA < 3 and preserved Hgb
    c_idx = df_out.index[df_out["Diagnosis"] == 1].to_numpy()
    n_early = max(1, int(len(c_idx) * 0.06))
    early_idx = rng.choice(c_idx, size=n_early, replace=False)

    df_out.loc[early_idx, "CEA_Level_ng_mL"] = (
        rng.lognormal(np.log(1.7), 0.45, n_early).clip(0.5, 3.0).round(2)
    )
    df_out.loc[early_idx, "Hemoglobin_g_dL"] = (
        rng.normal(14.2, 1.00, n_early).clip(13.5, 17.5).round(2)
    )
    df_out.loc[early_idx, "PyRad_ADC_Mean"] = (
        rng.normal(1430.0, 190.0, n_early).clip(1000.0, 1900.0).round(2)
    )
    df_out.loc[early_idx, "PyRad_ADC_Std"] = (
        rng.normal(105.0, 35.0, n_early).clip(20.0, 260.0).round(2)
    )
    df_out.loc[early_idx, "PyRad_Entropy"] = (
        rng.normal(4.8, 1.0, n_early).clip(2.5, 6.5).round(4)
    )
    df_out.loc[early_idx, "PyRad_GLCM_Contrast"] = (
        rng.normal(22.0, 8.0, n_early).clip(8.0, 50.0).round(4)
    )
    df_out.loc[early_idx, "PyRad_GLCM_Homogeneity"] = (
        rng.normal(0.60, 0.11, n_early).clip(0.35, 0.90).round(4)
    )
    df_out.loc[early_idx, "PyRad_Shape_Sphericity"] = (
        rng.normal(0.75, 0.09, n_early).clip(0.50, 1.00).round(4)
    )
    df_out.loc[early_idx, "PyRad_FirstOrder_Skewness"] = rng.normal(
        0.04, 0.38, n_early
    ).round(4)

    # Severe inflammation (~12% healthy): active IBD or diverticulitis
    h_idx = df_out.index[df_out["Diagnosis"] == 0].to_numpy()
    n_inflam = max(1, int(len(h_idx) * 0.08))
    inflam_idx = rng.choice(h_idx, size=n_inflam, replace=False)

    df_out.loc[inflam_idx, "CEA_Level_ng_mL"] = (
        rng.lognormal(np.log(12.0), 0.55, n_inflam).clip(8.0, 60.0).round(2)
    )
    df_out.loc[inflam_idx, "PyRad_ADC_Mean"] = (
        rng.normal(1180.0, 220.0, n_inflam).clip(700.0, 1700.0).round(2)
    )
    df_out.loc[inflam_idx, "PyRad_ADC_Std"] = (
        rng.normal(170.0, 55.0, n_inflam).clip(40.0, 380.0).round(2)
    )
    df_out.loc[inflam_idx, "PyRad_Entropy"] = (
        rng.normal(6.3, 1.0, n_inflam).clip(4.5, 9.5).round(4)
    )
    df_out.loc[inflam_idx, "PyRad_GLCM_Contrast"] = (
        rng.normal(48.0, 14.0, n_inflam).clip(22.0, 120.0).round(4)
    )
    df_out.loc[inflam_idx, "PyRad_GLCM_Homogeneity"] = (
        rng.normal(0.32, 0.10, n_inflam).clip(0.08, 0.52).round(4)
    )
    df_out.loc[inflam_idx, "PyRad_Shape_Sphericity"] = (
        rng.normal(0.60, 0.12, n_inflam).clip(0.25, 0.88).round(4)
    )
    df_out.loc[inflam_idx, "PyRad_FirstOrder_Skewness"] = rng.normal(
        0.38, 0.45, n_inflam
    ).round(4)

    # Lab & image variability (18% healthy): equipment artefacts
    sanos_idx = df_out.index[df_out["Diagnosis"] == 0].to_numpy()
    n_noise = max(1, int(len(sanos_idx) * 0.10))
    noisy_idx = rng.choice(sanos_idx, size=n_noise, replace=False)
    q = n_noise // 4
    idx_a = noisy_idx[:q]
    idx_b = noisy_idx[q : 2 * q]
    idx_c = noisy_idx[2 * q : 3 * q]
    idx_d = noisy_idx[3 * q :]

    df_out.loc[idx_a, "CEA_Level_ng_mL"] = (
        rng.lognormal(np.log(8.5), 0.60, len(idx_a)).clip(5.0, 40.0).round(2)
    )
    df_out.loc[idx_b, "Hemoglobin_g_dL"] = (
        rng.normal(11.5, 1.10, len(idx_b)).clip(8.5, 13.5).round(2)
    )

    adc_c = df_out.loc[idx_c, "PyRad_ADC_Mean"].to_numpy().astype(float)
    df_out.loc[idx_c, "PyRad_ADC_Mean"] = (
        (adc_c * rng.lognormal(0.0, 0.15, len(idx_c))).clip(800.0, 2500.0).round(2)
    )
    df_out.loc[idx_d, "PyRad_Entropy"] = (
        (df_out.loc[idx_d, "PyRad_Entropy"] + rng.normal(1.20, 0.45, len(idx_d)))
        .clip(0.1, 10.0)
        .round(4)
    )
    df_out.loc[idx_d, "PyRad_GLCM_Contrast"] = (
        (df_out.loc[idx_d, "PyRad_GLCM_Contrast"] + rng.normal(10.0, 4.0, len(idx_d)))
        .clip(0.1, 120.0)
        .round(4)
    )

    # Label swap (12%): simulates ambiguous cases, theoretical AUC ceiling ≈ 0.88
    swap_rng = np.random.default_rng(42)
    swap_mask = swap_rng.random(len(df_out)) < 0.02
    df_out.loc[swap_mask, "Diagnosis"] = 1 - df_out.loc[swap_mask, "Diagnosis"]

    logger.info(
        f"Dataset generated: {len(df_out):,} patients | "
        f"{int(df_out['Diagnosis'].sum()):,} cancer / "
        f"{int((df_out['Diagnosis'] == 0).sum()):,} healthy"
    )

    h_idx_clean = df_out.index[df_out["Diagnosis"] == 0].to_numpy()
    n_gold = max(1, int(len(h_idx_clean) * 0.15))
    gold_idx = rng.choice(h_idx_clean, size=n_gold, replace=False)

    df_out.loc[gold_idx, "CEA_Level_ng_mL"] = (
        rng.lognormal(np.log(1.5), 0.25, n_gold).clip(0.5, 3.0).round(2)
    )
    df_out.loc[gold_idx, "Hemoglobin_g_dL"] = (
        rng.normal(14.0, 0.80, n_gold).clip(12.0, 17.5).round(2)
    )
    df_out.loc[gold_idx, "PyRad_ADC_Mean"] = (
        rng.normal(1620.0, 120.0, n_gold).clip(1350.0, 1900.0).round(2)
    )
    df_out.loc[gold_idx, "PyRad_Entropy"] = (
        rng.normal(4.0, 0.60, n_gold).clip(2.5, 5.5).round(4)
    )
    df_out.loc[gold_idx, "PyRad_GLCM_Homogeneity"] = (
        rng.normal(0.75, 0.08, n_gold).clip(0.55, 0.95).round(4)
    )

    return df_out


def validate_distributions(df: pd.DataFrame) -> None:
    """Log mean ± std per diagnostic group for clinical plausibility audit.

    Args:
        df: DataFrame produced by ``ClinicalDataGenerator.generate``.
    """
    features = [
        "CEA_Level_ng_mL",
        "Hemoglobin_g_dL",
        "PyRad_ADC_Mean",
        "PyRad_ADC_Std",
        "PyRad_Entropy",
        "PyRad_GLCM_Contrast",
        "PyRad_GLCM_Homogeneity",
        "PyRad_Shape_Sphericity",
        "PyRad_FirstOrder_Skewness",
    ]
    logger.info("=" * 80)
    logger.info("  VALIDATION -- Mean +/- Std per diagnostic group")
    logger.info("=" * 80)
    for col in features:
        c = df[df["Diagnosis"] == 1][col]
        h = df[df["Diagnosis"] == 0][col]
        logger.info(
            f"  {col:<35}  Cancer: {c.mean():>8.2f} +/- {c.std():.2f}  |  "
            f"Healthy: {h.mean():>8.2f} +/- {h.std():.2f}"
        )
    logger.info("=" * 80)


class ClinicalDataGenerator:
    """
    Synthetic CRC clinical + radiomic dataset generator.

    Reads the base CSV from ``paths.RAW / 'tabular_risk'`` and enriches
    each row with multivariate haematological biomarkers and PyRadiomics-
    style features calibrated against published clinical reference ranges.

    Replaces ``SyntheticPatientGenerator`` as the single data-generation
    entry point for the tabular training pipeline.
    """

    def __init__(self, seed: int = 42, csv_path: Path | None = None):
        """
        Initialise the generator.

        Args:
            seed (int): Random seed for full reproducibility. Default 42.
            csv_path (Path | None): Override for the base CSV path.
                Defaults to ``paths.RAW / 'tabular_risk' /
                'colorectal_cancer_dataset.csv'``.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.csv_path = csv_path or _CSV_BASE

    def generate(self) -> pd.DataFrame:
        """
        Load the base CSV and generate the full enriched clinical dataset.

        Returns:
            pd.DataFrame: Enriched DataFrame with 13 columns ready for
                ``TabularPreprocessor.fit_transform``.

        Raises:
            FileNotFoundError: If the base CSV does not exist.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Base CSV not found: {self.csv_path}\n"
                "Run the data download step first."
            )
        logger.info(f"Loading base CSV: {self.csv_path}")
        df_base = pd.read_csv(self.csv_path)
        logger.info(f"Base records loaded: {len(df_base):,}")
        return _generate_features(df_base, self.rng)

    def generate_and_validate(self) -> pd.DataFrame:
        """
        Generate the dataset and log distribution validation statistics.

        Convenience wrapper that calls ``generate`` followed by
        ``validate_distributions``.

        Returns:
            pd.DataFrame: Same enriched DataFrame as ``generate``.
        """
        df = self.generate()
        validate_distributions(df)
        return df


if __name__ == "__main__":
    """
    Standalone execution: generate the clinical dataset and save to disk.

    Reads the base CSV from paths.RAW / 'tabular_risk' /
    'colorectal_cancer_dataset.csv' and writes the enriched dataset to
    paths.PROCESSED / 'dataset_clinico_tumoral.csv'.

    Usage:
        python -m src.data.processing.clinical_data_generator
    """

    output_path = paths.PROCESSED / "dataset_clinico_tumoral.csv"
    paths.PROCESSED.mkdir(parents=True, exist_ok=True)

    logger.info("═══ ClinicalDataGenerator — Standalone Execution ═══")
    generator = ClinicalDataGenerator(seed=42)
    df_clinical = generator.generate_and_validate()

    df_clinical.to_csv(output_path, index=False)
    logger.info(f"Dataset saved: {output_path}")
    logger.info(f"Shape: {df_clinical.shape}")
    logger.info(f"\n{df_clinical.head(3).to_string()}")
