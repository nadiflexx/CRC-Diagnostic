"""
Clinical synthetic CRC dataset generator.

Wraps the multivariate clinical + radiomic feature generation logic
calibrated against NCCN 2023, ESGAR 2022 and Duffy et al. 2021 reference
ranges. Replaces the old SyntheticPatientGenerator with a single class
that reads the base CSV and produces a ready-to-train DataFrame.

Key improvement over v1: tumour staging (T1→T4/M1) drives all cancer-group
distributions, producing realistic probability overlap with healthy cases
(target AUC ceiling ~0.82-0.86) instead of artificial separability.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.constants import (
    CANCER_CORR,
    HEALTHY_CORR,
    HEALTHY_MEANS,
    HEALTHY_STDS,
    STAGE_PARAMS,
)
from src.config.logger import log as logger
from src.config.paths import paths

_CSV_BASE = paths.TABULAR_PROCESSED / "colorectal_cancer_cleaned.csv"


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
    ``Genetic_Mutation`` to adjust generated values.

    Cancer group is stratified by T-stage (T1 18%, T2 27%, T3 30%, T4/M1 25%)
    using Cholesky-decomposed correlated residuals so that each stage respects
    the inter-feature clinical correlations defined in ``CANCER_CORR``.

    Healthy group injects:
    - 14% severe multi-feature inflammation noise (IBD/diverticulitis).
    - 18% lab/imaging artefact noise split across four artefact types.

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

    cov_healthy = _corr_to_cov(HEALTHY_CORR, HEALTHY_STDS)

    cea = np.zeros(n)
    hgb = np.zeros(n)
    adc_mean = np.zeros(n)
    entropy = np.zeros(n)
    contrast = np.zeros(n)

    # ── Cancer group — T-stage stratified ────────────────────────────────────
    cancer_idx = np.where(diag == 1)[0]
    n_c = len(cancer_idx)

    hom_base_c = np.zeros(n_c)
    sph_mu_c = np.zeros(n_c)
    skew_mu_c = np.zeros(n_c)

    if n_c > 0:
        L_cancer = np.linalg.cholesky(CANCER_CORR)
        z_c = rng.standard_normal((n_c, 5))
        corr_res_c = z_c @ L_cancer.T

        stages = rng.choice([1, 2, 3, 4], size=n_c, p=[0.18, 0.27, 0.30, 0.25])

        for stage, (means_s, stds_s, hom_b, sph_m, skew_m) in STAGE_PARAMS.items():
            mask = stages == stage
            if not mask.any():
                continue
            r = corr_res_c[mask]
            ages_s = ages[cancer_idx[mask]]
            exceso = np.maximum(ages_s - 70, 0.0)
            d_cea = np.where(ages_s > 70, np.minimum(0.30 + 0.012 * exceso, 0.70), 0.0)
            d_hgb = np.where(ages_s > 70, np.maximum(-0.80 - 0.04 * exceso, -2.00), 0.0)
            g_off = np.where(gender[cancer_idx[mask]] == 1, 0.0, -1.5)

            cea[cancer_idx[mask]] = np.exp(means_s[0] + stds_s[0] * r[:, 0] + d_cea)
            hgb[cancer_idx[mask]] = means_s[1] + stds_s[1] * r[:, 1] + d_hgb + g_off
            adc_mean[cancer_idx[mask]] = means_s[2] + stds_s[2] * r[:, 2]
            entropy[cancer_idx[mask]] = means_s[3] + stds_s[3] * r[:, 3]
            contrast[cancer_idx[mask]] = means_s[4] + stds_s[4] * r[:, 4]
            hom_base_c[mask] = hom_b
            sph_mu_c[mask] = sph_m
            skew_mu_c[mask] = skew_m

    # ── Healthy group ─────────────────────────────────────────────────────────
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

    adc_std = np.where(
        diag == 1,
        np.clip(rng.lognormal(np.log(200), 0.55, n), 5.0, 400.0),
        np.clip(rng.lognormal(np.log(82), 0.48, n), 5.0, 400.0),
    )

    # Homogeneity: T1(0.67)→T4(0.18) gradient for cancer; 0.72 for healthy.
    hom_base_full = np.full(n, 0.72)
    hom_base_full[cancer_idx] = hom_base_c

    coef_c = np.where(diag == 1, 0.003, 0.002)
    coef_e = np.where(diag == 1, 0.016, 0.007)
    umbral_c = np.where(diag == 1, 30.0, 12.0)
    umbral_e = np.where(diag == 1, 5.0, 4.0)
    hom_raw = (
        rng.normal(hom_base_full, 0.10, n)
        - coef_c * np.maximum(contrast - umbral_c, 0)
        - coef_e * np.maximum(entropy - umbral_e, 0)
    )

    # Sphericity: T1(regular)→T4(very irregular) gradient; 0.73 for healthy.
    sph_mu_full = np.full(n, 0.73)
    sph_mu_full[cancer_idx] = sph_mu_c
    sph_raw = rng.normal(sph_mu_full, 0.12, n)

    # Skewness: T1(mild)→T4(extreme due to massive necrosis) gradient;
    skew_mu_full = np.full(n, 0.05)
    skew_mu_full[cancer_idx] = skew_mu_c
    skew_raw = rng.normal(skew_mu_full, 0.42, n)

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

    # ── Biological noise — clinical grey zone ─────────────────────────────────
    h_idx = df_out.index[df_out["Diagnosis"] == 0].to_numpy()
    n_inflam = max(1, int(len(h_idx) * 0.14))
    inflam_idx = rng.choice(h_idx, size=n_inflam, replace=False)

    df_out.loc[inflam_idx, "CEA_Level_ng_mL"] = (
        rng.lognormal(np.log(11.0), 0.60, n_inflam).clip(5.0, 55.0).round(2)
    )
    df_out.loc[inflam_idx, "Hemoglobin_g_dL"] = (
        rng.normal(11.2, 1.50, n_inflam).clip(7.5, 13.8).round(2)
    )
    df_out.loc[inflam_idx, "PyRad_ADC_Mean"] = (
        rng.normal(1200.0, 240.0, n_inflam).clip(700.0, 1750.0).round(2)
    )
    df_out.loc[inflam_idx, "PyRad_ADC_Std"] = (
        rng.normal(175.0, 60.0, n_inflam).clip(40.0, 380.0).round(2)
    )
    df_out.loc[inflam_idx, "PyRad_Entropy"] = (
        rng.normal(6.1, 1.1, n_inflam).clip(4.0, 9.5).round(4)
    )
    df_out.loc[inflam_idx, "PyRad_GLCM_Contrast"] = (
        rng.normal(46.0, 15.0, n_inflam).clip(20.0, 120.0).round(4)
    )
    df_out.loc[inflam_idx, "PyRad_GLCM_Homogeneity"] = (
        rng.normal(0.34, 0.11, n_inflam).clip(0.08, 0.56).round(4)
    )
    df_out.loc[inflam_idx, "PyRad_Shape_Sphericity"] = (
        rng.normal(0.61, 0.13, n_inflam).clip(0.25, 0.88).round(4)
    )
    df_out.loc[inflam_idx, "PyRad_FirstOrder_Skewness"] = rng.normal(
        0.35, 0.48, n_inflam
    ).round(4)

    # Lab & imaging variability (18% healthy)
    sanos_idx = df_out.index[df_out["Diagnosis"] == 0].to_numpy()
    n_noise = max(1, int(len(sanos_idx) * 0.18))
    noisy_idx = rng.choice(sanos_idx, size=n_noise, replace=False)
    q = n_noise // 4
    idx_a = noisy_idx[:q]
    idx_b = noisy_idx[q : 2 * q]
    idx_c = noisy_idx[2 * q : 3 * q]
    idx_d = noisy_idx[3 * q :]

    # Benign CEA peak: active smoking or acute inflammation
    df_out.loc[idx_a, "CEA_Level_ng_mL"] = (
        rng.lognormal(np.log(8.5), 0.60, len(idx_a)).clip(5.0, 40.0).round(2)
    )
    # Mild non-oncological anaemia (iron deficiency or B12 deficit)
    df_out.loc[idx_b, "Hemoglobin_g_dL"] = (
        rng.normal(11.5, 1.10, len(idx_b)).clip(8.5, 13.5).round(2)
    )
    # ADC variability due to acquisition protocol differences
    adc_c = df_out.loc[idx_c, "PyRad_ADC_Mean"].to_numpy().astype(float)
    df_out.loc[idx_c, "PyRad_ADC_Mean"] = (
        (adc_c * rng.lognormal(0.0, 0.15, len(idx_c))).clip(800.0, 2500.0).round(2)
    )
    # Texture artefacts: motion or B0 distortion
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

    logger.info(
        f"Dataset generated: {len(df_out):,} patients | "
        f"{int(df_out['Diagnosis'].sum()):,} cancer / "
        f"{int((df_out['Diagnosis'] == 0).sum()):,} healthy"
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

    Reads the base CSV from ``paths.DATA / 'tabular_clean'`` and enriches
    each row with multivariate haematological biomarkers and PyRadiomics-
    style features calibrated against published clinical reference ranges.

    Cancer cases are stratified by T-stage (T1 18%, T2 27%, T3 30%,
    T4/M1 25%) so that early-stage tumours produce realistic probability
    overlap with healthy tissue (target AUC ceiling ~0.82-0.86).

    Replaces ``SyntheticPatientGenerator`` as the single data-generation
    entry point for the tabular training pipeline.
    """

    def __init__(self, seed: int = 42, csv_path: Path | None = None):
        """
        Initialise the generator.

        Args:
            seed (int): Random seed for full reproducibility. Default 42.
            csv_path (Path | None): Override for the base CSV path.
                Defaults to ``paths.DATA / 'tabular_clean' /
                'colorectal_cancer_full_dataset_v1.csv'``.
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
