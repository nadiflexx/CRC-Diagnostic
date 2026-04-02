"""
Synthetic patient data generation with clinical overlap.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from src.config.constants import (
    CSV_ACTIVITY_MAP,
    CSV_AGE_RISK_GROUP_MAP,
    CSV_BINARY_MAP,
    CSV_DIET_MAP,
    CSV_FINAL_COLUMNS,
    CSV_GENDER_MAP,
    CSV_LIFESTYLE_CLUSTER_OPTIONS,
    CSV_OBESITY_MAP,
    CSV_RANDOM_SEED,
    CSV_SCREENING_MAP,
    CSV_URBAN_MAP,
    PHYSIOLOGICAL_RANGES,
)
from src.config.logger import log as logger


# ═══════════════════════════════════════════════════════════
#  EXISTING: clinical lab synthetic generator
# ═══════════════════════════════════════════════════════════

class SyntheticPatientGenerator:
    """
    Synthetic patient data generator.
    """

    CONTINUOUS_FEATURES = [
        "age",
        "bmi",
        "hemoglobin",
        "hematocrit",
        "wbc_count",
        "platelet_count",
        "albumin",
        "iron_serum",
        "ferritin",
        "crp",
        "cea",
        "ca19_9",
        "pack_years_smoked",
        "previous_polyps_count",
    ]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_healthy_patients(self, n: int) -> pd.DataFrame:
        """
        Generate healthy patients.

        :param n: Number of healthy patients to generate.
        :return: DataFrame containing generated healthy patients.
        """
        logger.info(f"Generating {n} healthy patients...")
        n_clean = int(n * 0.55)
        n_risk = int(n * 0.20)
        n_altered = int(n * 0.17)
        n_tricky = n - n_clean - n_risk - n_altered

        dfs = [
            self._healthy_clean(n_clean),
            self._healthy_with_risk_factors(n_risk),
            self._healthy_with_altered_values(n_altered),
            self._healthy_tricky(n_tricky),
        ]
        df = pd.concat(dfs, ignore_index=True)
        df["has_cancer"] = False
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"✅ Generated {len(df)} healthy patients")
        return df

    def _healthy_clean(self, n):
        """
        Generate clean healthy patients.

        :param n: Number of clean healthy patients to generate.
        :return: DataFrame containing generated clean healthy patients.
        """
        smoked = self.rng.binomial(1, 0.28, n)
        pack_years = smoked * self.rng.exponential(4.0, n)
        has_polyp_history = self.rng.binomial(1, 0.12, n)
        polyp_count = has_polyp_history * self.rng.poisson(1.0, n)
        return pd.DataFrame(
            {
                "age": self.rng.normal(45, 14, n).clip(18, 85),
                "bmi": self.rng.normal(25.0, 4.0, n).clip(17, 40),
                "hemoglobin": self.rng.normal(14.5, 1.3, n).clip(11.0, 17.5),
                "hematocrit": self.rng.normal(43, 3.5, n).clip(33, 52),
                "wbc_count": self.rng.normal(6.8, 1.6, n).clip(3.8, 11.0),
                "platelet_count": self.rng.normal(248, 52, n).clip(150, 400),
                "albumin": self.rng.normal(4.3, 0.35, n).clip(3.5, 5.2),
                "iron_serum": self.rng.normal(105, 28, n).clip(55, 170),
                "ferritin": self.rng.normal(115, 55, n).clip(20, 280),
                "crp": self.rng.exponential(0.3, n).clip(0.01, 2.0),
                "cea": (
                    self.rng.exponential(0.9, n) + smoked * self.rng.exponential(1.2, n)
                ).clip(0.1, 6.0),
                "ca19_9": self.rng.exponential(7.0, n).clip(0.5, 33),
                "pack_years_smoked": pack_years.clip(0, 30),
                "previous_polyps_count": polyp_count.clip(0, 4),
                **self._healthy_categorical(n, "low"),
                **self._healthy_binary(n, "low"),
            }
        )

    def _healthy_with_risk_factors(self, n):
        """
        Generate patients with risk factors.

        :param n: Number of patients with risk factors to generate.
        :return: DataFrame containing generated patients with risk factors.
        """
        smoked = self.rng.binomial(1, 0.55, n)
        pack_years = smoked * self.rng.exponential(8.0, n)
        has_polyps = self.rng.binomial(1, 0.22, n)
        polyp_count = has_polyps * self.rng.poisson(1.2, n)
        return pd.DataFrame(
            {
                "age": self.rng.normal(55, 12, n).clip(30, 85),
                "bmi": self.rng.normal(29, 5, n).clip(20, 45),
                "hemoglobin": self.rng.normal(14.0, 1.4, n).clip(10.5, 17.0),
                "hematocrit": self.rng.normal(42, 3.8, n).clip(32, 51),
                "wbc_count": self.rng.normal(7.2, 1.8, n).clip(4.0, 12.0),
                "platelet_count": self.rng.normal(262, 58, n).clip(150, 420),
                "albumin": self.rng.normal(4.1, 0.38, n).clip(3.3, 5.0),
                "iron_serum": self.rng.normal(88, 28, n).clip(45, 165),
                "ferritin": self.rng.normal(100, 55, n).clip(15, 270),
                "crp": self.rng.exponential(0.7, n).clip(0.03, 5.0),
                "cea": (
                    self.rng.exponential(1.2, n) + smoked * self.rng.exponential(1.8, n)
                ).clip(0.2, 9.0),
                "ca19_9": self.rng.exponential(9.0, n).clip(1.0, 38),
                "pack_years_smoked": pack_years.clip(0, 45),
                "previous_polyps_count": polyp_count.clip(0, 5),
                **self._healthy_categorical(n, "medium"),
                **self._healthy_binary(n, "medium"),
            }
        )

    def _healthy_with_altered_values(self, n):
        """
        Generate patients with altered values.

        :param n: Number of patients with altered values to generate.
        :return: DataFrame containing generated patients with altered values.
        """
        smoked = self.rng.binomial(1, 0.45, n)
        pack_years = smoked * self.rng.exponential(6.0, n)
        has_polyps = self.rng.binomial(1, 0.28, n)
        polyp_count = has_polyps * self.rng.poisson(1.5, n)
        return pd.DataFrame(
            {
                "age": self.rng.normal(53, 14, n).clip(22, 88),
                "bmi": self.rng.normal(27.5, 5, n).clip(17, 44),
                "hemoglobin": self.rng.normal(11.5, 1.6, n).clip(8.0, 15.0),
                "hematocrit": self.rng.normal(36, 4.2, n).clip(26, 46),
                "wbc_count": self.rng.normal(7.5, 2.2, n).clip(3.5, 13.0),
                "platelet_count": self.rng.normal(280, 72, n).clip(140, 460),
                "albumin": self.rng.normal(3.8, 0.5, n).clip(2.7, 4.9),
                "iron_serum": self.rng.normal(52, 22, n).clip(18, 130),
                "ferritin": self.rng.normal(32, 22, n).clip(5, 110),
                "crp": self.rng.exponential(1.8, n).clip(0.1, 10.0),
                "cea": (
                    self.rng.exponential(2.0, n) + smoked * self.rng.exponential(2.0, n)
                ).clip(0.3, 12.0),
                "ca19_9": self.rng.exponential(11.0, n).clip(1.5, 50),
                "pack_years_smoked": pack_years.clip(0, 40),
                "previous_polyps_count": polyp_count.clip(0, 6),
                **self._healthy_categorical(n, "high"),
                **self._healthy_binary(n, "high"),
            }
        )

    def _healthy_tricky(self, n):
        """
        Generate patients with tricky profiles (high-risk but healthy).
        """
        smoked = self.rng.binomial(1, 0.55, n)
        pack_years = smoked * self.rng.exponential(10.0, n)
        has_polyps = self.rng.binomial(1, 0.35, n)
        polyp_count = has_polyps * self.rng.poisson(2.0, n)
        return pd.DataFrame(
            {
                "age": self.rng.normal(60, 10, n).clip(35, 88),
                "bmi": self.rng.normal(30, 5.5, n).clip(20, 48),
                "hemoglobin": self.rng.normal(10.8, 1.8, n).clip(7.0, 14.5),
                "hematocrit": self.rng.normal(34, 4.5, n).clip(24, 44),
                "wbc_count": self.rng.normal(8.8, 2.8, n).clip(3.5, 16.0),
                "platelet_count": self.rng.normal(315, 85, n).clip(150, 520),
                "albumin": self.rng.normal(3.4, 0.55, n).clip(2.3, 4.6),
                "iron_serum": self.rng.normal(42, 20, n).clip(12, 115),
                "ferritin": self.rng.normal(22, 15, n).clip(3, 85),
                "crp": self.rng.exponential(3.0, n).clip(0.2, 14.0),
                "cea": (
                    self.rng.exponential(2.8, n) + smoked * self.rng.exponential(2.5, n)
                ).clip(0.5, 15.0),
                "ca19_9": self.rng.exponential(14.0, n).clip(2.0, 60),
                "pack_years_smoked": pack_years.clip(0, 55),
                "previous_polyps_count": polyp_count.clip(0, 8),
                **self._healthy_categorical(n, "high"),
                **self._healthy_binary(n, "high"),
            }
        )

    def _healthy_categorical(self, n, severity):
        if severity == "low":
            return {
                "gender": self.rng.choice(["male", "female"], n),
                "ethnicity": self.rng.choice(
                    ["white", "hispanic", "black", "asian", "other"],
                    n,
                    p=[0.60, 0.18, 0.13, 0.06, 0.03],
                ),
                "smoking_status": self.rng.choice(
                    ["never", "former", "current"], n, p=[0.60, 0.28, 0.12]
                ),
                "alcohol_consumption": self.rng.choice(
                    ["none", "moderate", "heavy"], n, p=[0.35, 0.55, 0.10]
                ),
                "physical_activity": self.rng.choice(
                    ["active", "moderate", "sedentary"], n, p=[0.45, 0.40, 0.15]
                ),
                "diet_type": self.rng.choice(
                    ["high_fiber", "balanced", "high_fat_low_fiber"],
                    n,
                    p=[0.35, 0.45, 0.20],
                ),
            }
        elif severity == "medium":
            return {
                "gender": self.rng.choice(["male", "female"], n),
                "ethnicity": self.rng.choice(
                    ["white", "hispanic", "black", "asian", "other"],
                    n,
                    p=[0.55, 0.20, 0.15, 0.07, 0.03],
                ),
                "smoking_status": self.rng.choice(
                    ["never", "former", "current"], n, p=[0.40, 0.35, 0.25]
                ),
                "alcohol_consumption": self.rng.choice(
                    ["none", "moderate", "heavy"], n, p=[0.25, 0.50, 0.25]
                ),
                "physical_activity": self.rng.choice(
                    ["active", "moderate", "sedentary"], n, p=[0.30, 0.40, 0.30]
                ),
                "diet_type": self.rng.choice(
                    ["high_fiber", "balanced", "high_fat_low_fiber"],
                    n,
                    p=[0.20, 0.40, 0.40],
                ),
            }
        else:
            return {
                "gender": self.rng.choice(["male", "female"], n),
                "ethnicity": self.rng.choice(
                    ["white", "hispanic", "black", "asian", "other"],
                    n,
                    p=[0.50, 0.22, 0.18, 0.07, 0.03],
                ),
                "smoking_status": self.rng.choice(
                    ["never", "former", "current"], n, p=[0.25, 0.35, 0.40]
                ),
                "alcohol_consumption": self.rng.choice(
                    ["none", "moderate", "heavy"], n, p=[0.15, 0.45, 0.40]
                ),
                "physical_activity": self.rng.choice(
                    ["active", "moderate", "sedentary"], n, p=[0.15, 0.35, 0.50]
                ),
                "diet_type": self.rng.choice(
                    ["high_fiber", "balanced", "high_fat_low_fiber"],
                    n,
                    p=[0.10, 0.30, 0.60],
                ),
            }

    def _healthy_binary(self, n, severity):
        probs = {
            "low": {
                "family_history_ccr": 0.05,
                "family_history_polyps": 0.08,
                "family_history_lynch": 0.01,
                "family_history_fap": 0.005,
                "has_ibd": 0.02,
                "has_diabetes_t2": 0.06,
                "previous_polyps": 0.08,
                "previous_cancer": 0.01,
                "fobt_positive": 0.05,
                "fit_positive": 0.06,
            },
            "medium": {
                "family_history_ccr": 0.12,
                "family_history_polyps": 0.18,
                "family_history_lynch": 0.025,
                "family_history_fap": 0.010,
                "has_ibd": 0.06,
                "has_diabetes_t2": 0.11,
                "previous_polyps": 0.22,
                "previous_cancer": 0.04,
                "fobt_positive": 0.18,
                "fit_positive": 0.20,
            },
            "high": {
                "family_history_ccr": 0.20,
                "family_history_polyps": 0.26,
                "family_history_lynch": 0.045,
                "family_history_fap": 0.018,
                "has_ibd": 0.12,
                "has_diabetes_t2": 0.17,
                "previous_polyps": 0.42,
                "previous_cancer": 0.10,
                "fobt_positive": 0.52,
                "fit_positive": 0.58,
            },
        }
        p = probs[severity]
        return {k: self.rng.binomial(1, v, n).astype(bool) for k, v in p.items()}

    def _cancer_binary(self, n, severity):
        probs = {
            "early": {
                "family_history_ccr": 0.26,
                "family_history_polyps": 0.32,
                "family_history_lynch": 0.055,
                "family_history_fap": 0.022,
                "has_ibd": 0.16,
                "has_diabetes_t2": 0.20,
                "previous_polyps": 0.55,
                "previous_cancer": 0.16,
                "fobt_positive": 0.72,
                "fit_positive": 0.78,
            },
            "advanced": {
                "family_history_ccr": 0.30,
                "family_history_polyps": 0.38,
                "family_history_lynch": 0.070,
                "family_history_fap": 0.028,
                "has_ibd": 0.22,
                "has_diabetes_t2": 0.26,
                "previous_polyps": 0.65,
                "previous_cancer": 0.22,
                "fobt_positive": 0.88,
                "fit_positive": 0.90,
            },
        }
        p = probs[severity]
        return {k: self.rng.binomial(1, v, n).astype(bool) for k, v in p.items()}

    def generate_cancer_patients(self, n: int) -> pd.DataFrame:
        """Generate cancer patients with realistic clinical profiles."""
        logger.info(f"Generating {n} cancer patients...")
        n_early = int(n * 0.40)
        n_advanced = n - n_early

        dfs = [
            self._cancer_early(n_early),
            self._cancer_advanced(n_advanced),
        ]
        df = pd.concat(dfs, ignore_index=True)
        df["has_cancer"] = True
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"✅ Generated {len(df)} cancer patients")
        return df

    def _cancer_early(self, n):
        smoked = self.rng.binomial(1, 0.52, n)
        pack_years = smoked * self.rng.exponential(12.0, n)
        has_polyps = self.rng.binomial(1, 0.55, n)
        polyp_count = has_polyps * self.rng.poisson(2.5, n)
        return pd.DataFrame(
            {
                "age": self.rng.normal(62, 11, n).clip(35, 88),
                "bmi": self.rng.normal(27.5, 5.5, n).clip(18, 46),
                "hemoglobin": self.rng.normal(12.2, 1.8, n).clip(8.0, 16.0),
                "hematocrit": self.rng.normal(37, 4.5, n).clip(26, 48),
                "wbc_count": self.rng.normal(8.2, 2.4, n).clip(3.5, 15.0),
                "platelet_count": self.rng.normal(295, 78, n).clip(140, 500),
                "albumin": self.rng.normal(3.8, 0.55, n).clip(2.5, 4.8),
                "iron_serum": self.rng.normal(60, 22, n).clip(18, 140),
                "ferritin": self.rng.normal(42, 28, n).clip(5, 160),
                "crp": self.rng.exponential(2.2, n).clip(0.1, 12.0),
                "cea": (
                    self.rng.exponential(4.5, n) + smoked * self.rng.exponential(3.0, n)
                ).clip(0.5, 45.0),
                "ca19_9": self.rng.exponential(22.0, n).clip(2.0, 120),
                "pack_years_smoked": pack_years.clip(0, 60),
                "previous_polyps_count": polyp_count.clip(0, 10),
                **self._healthy_categorical(n, "high"),
                **self._cancer_binary(n, "early"),
            }
        )

    def _cancer_advanced(self, n):
        smoked = self.rng.binomial(1, 0.60, n)
        pack_years = smoked * self.rng.exponential(18.0, n)
        has_polyps = self.rng.binomial(1, 0.65, n)
        polyp_count = has_polyps * self.rng.poisson(3.0, n)
        return pd.DataFrame(
            {
                "age": self.rng.normal(66, 10, n).clip(38, 90),
                "bmi": self.rng.normal(25.5, 5.8, n).clip(16, 44),
                "hemoglobin": self.rng.normal(10.2, 2.0, n).clip(6.5, 14.5),
                "hematocrit": self.rng.normal(32, 5.0, n).clip(22, 44),
                "wbc_count": self.rng.normal(9.8, 3.2, n).clip(3.5, 18.0),
                "platelet_count": self.rng.normal(335, 95, n).clip(130, 580),
                "albumin": self.rng.normal(3.2, 0.60, n).clip(1.8, 4.4),
                "iron_serum": self.rng.normal(38, 18, n).clip(8, 110),
                "ferritin": self.rng.normal(28, 20, n).clip(3, 120),
                "crp": self.rng.exponential(4.5, n).clip(0.2, 22.0),
                "cea": (
                    self.rng.exponential(18.0, n) + smoked * self.rng.exponential(8.0, n)
                ).clip(1.0, 200.0),
                "ca19_9": self.rng.exponential(55.0, n).clip(5.0, 550),
                "pack_years_smoked": pack_years.clip(0, 75),
                "previous_polyps_count": polyp_count.clip(0, 15),
                **self._healthy_categorical(n, "high"),
                **self._cancer_binary(n, "advanced"),
            }
        )

    def _clip_physiological(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clips all columns to physiological ranges using centralized constants."""
        for col, (lo, hi) in PHYSIOLOGICAL_RANGES.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)
        return df

    def _add_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds correlations between physiological features."""
        df = df.copy()
        n = len(df)
        hb_std = max(df["hemoglobin"].std(), 0.01)
        hb_z = (df["hemoglobin"] - df["hemoglobin"].mean()) / hb_std
        df["hematocrit"] += hb_z * 1.5 + self.rng.normal(0, 0.5, n)
        df["iron_serum"] += hb_z * 8.0 + self.rng.normal(0, 2.0, n)
        df["ferritin"] += hb_z * 12.0 + self.rng.normal(0, 4.0, n)

        crp_std = max(df["crp"].std(), 0.01)
        crp_z = (df["crp"] - df["crp"].mean()) / crp_std
        df["wbc_count"] += crp_z * 0.5 + self.rng.normal(0, 0.3, n)
        df["platelet_count"] += crp_z * 15.0 + self.rng.normal(0, 5.0, n)

        alb_std = max(df["albumin"].std(), 0.01)
        alb_z = (df["albumin"] - df["albumin"].mean()) / alb_std
        df["bmi"] += alb_z * 0.8 + self.rng.normal(0, 0.5, n)

        return self._clip_physiological(df)

    def _add_measurement_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds measurement noise to physiological features."""
        df = df.copy()
        n = len(df)
        cv_by_feature = {
            "hemoglobin": 0.03,
            "hematocrit": 0.03,
            "wbc_count": 0.06,
            "platelet_count": 0.05,
            "albumin": 0.04,
            "iron_serum": 0.08,
            "ferritin": 0.08,
            "crp": 0.10,
            "cea": 0.08,
            "ca19_9": 0.10,
            "bmi": 0.02,
            "age": 0.0,
            "pack_years_smoked": 0.15,
        }
        for col, cv in cv_by_feature.items():
            if col in df.columns and cv > 0:
                df[col] = df[col] * self.rng.normal(1.0, cv, n)
        return self._clip_physiological(df)

    def generate_balanced_dataset(self, n_per_class: int = 5000) -> pd.DataFrame:
        """Generate a balanced dataset of healthy and cancer patients."""
        logger.info(
            f"Generating balanced dataset with {n_per_class} patients per class..."
        )
        healthy = self.generate_healthy_patients(n_per_class)
        cancer = self.generate_cancer_patients(n_per_class)
        combined = pd.concat([healthy, cancer], ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        combined = self._add_correlations(combined)
        combined = self._add_measurement_noise(combined)
        self._verify_overlap(combined)
        logger.info(f"Balanced dataset: {len(combined)} patients")
        return combined

    def generate_healthy_patients_standalone(self, n):
        """Generate healthy patients with correlations and noise applied."""
        df = self.generate_healthy_patients(n)
        df = self._add_correlations(df)
        return self._add_measurement_noise(df)

    def generate_cancer_patients_standalone(self, n):
        """Generate cancer patients with correlations and noise applied."""
        df = self.generate_cancer_patients(n)
        df = self._add_correlations(df)
        return self._add_measurement_noise(df)

    def _verify_overlap(self, df):
        """Verify overlap between healthy and cancer patients."""
        cancer = df[df["has_cancer"]]
        healthy = df[~df["has_cancer"]]
        logger.info("═══ Overlap Verification ═══")
        for feat in [
            "hemoglobin", "cea", "albumin", "iron_serum",
            "age", "pack_years_smoked", "ferritin", "crp",
        ]:
            if feat not in df.columns:
                continue
            c_mean, c_std = cancer[feat].mean(), cancer[feat].std()
            h_mean, h_std = healthy[feat].mean(), healthy[feat].std()
            pooled_std = np.sqrt((c_std**2 + h_std**2) / 2)
            cohens_d = abs(c_mean - h_mean) / max(pooled_std, 0.01)
            status = (
                "✅ Low" if cohens_d < 0.5
                else "✅ Medium" if cohens_d < 0.8
                else "⚠️ High" if cohens_d < 1.2
                else "🔴 VERY High"
            )
            logger.info(
                f"  {feat:>22s}: healthy={h_mean:.1f}±{h_std:.1f} | "
                f"cancer={c_mean:.1f}±{c_std:.1f} | d={cohens_d:.2f} {status}"
            )
        for feat in ["fobt_positive", "fit_positive", "previous_polyps"]:
            if feat not in df.columns:
                continue
            h_rate = healthy[feat].mean()
            c_rate = cancer[feat].mean()
            gap = abs(c_rate - h_rate)
            status = "✅" if gap < 0.30 else "⚠️" if gap < 0.45 else "🔴"
            logger.info(
                f"  {feat:>22s}: healthy={h_rate:.1%} | cancer={c_rate:.1%} | "
                f"gap={gap:.1%} {status}"
            )


class SyntheticMultimodalLinker:
    """Links medical images with tabular patient data."""

    def __init__(self, generator: SyntheticPatientGenerator):
        self.generator = generator

    def link_images_with_tabular(self, image_paths, image_labels):
        n = len(image_paths)
        polyp_mask = np.array(image_labels) == 1
        n_polyp = int(polyp_mask.sum())
        n_normal = n - n_polyp
        healthy_profiles = self.generator.generate_healthy_patients_standalone(n_normal)
        cancer_profiles = self.generator.generate_cancer_patients_standalone(n_polyp)
        healthy_idx, cancer_idx = 0, 0
        rows = []
        for i in range(n):
            if image_labels[i] == 0:
                row = healthy_profiles.iloc[healthy_idx].to_dict()
                healthy_idx += 1
            else:
                row = cancer_profiles.iloc[cancer_idx].to_dict()
                cancer_idx += 1
            row["image_path"] = image_paths[i]
            row["image_label"] = image_labels[i]
            rows.append(row)
        result = pd.DataFrame(rows)
        logger.info(
            f"✅ Linked {n} images with tabular profiles "
            f"({n_normal} normal, {n_polyp} polyps)"
        )
        return result


# ═══════════════════════════════════════════════════════════
#  NEW: CSV-based epidemiological synthetic generator
#  (colorectal_cancer_dataset.csv pipeline)
# ═══════════════════════════════════════════════════════════

def _csv_calc_risk_score(
    obesity: str, diet: str, activity: str, smoking: str, alcohol: str,
    diabetes: str, ibd: str, genetic: str, family_history: str, age: int,
) -> float:
    """Risk_Score [0-10]: composite risk index from the CSV pipeline."""
    score = 0.0
    if family_history == "Yes": score += 1.5
    if genetic == "Yes":        score += 2.0
    if age >= 60:               score += 1.0
    elif age >= 45:             score += 0.5
    if diabetes == "Yes":       score += 1.0
    if ibd == "Yes":            score += 1.0
    if obesity == "Obese":          score += 0.75
    elif obesity == "Overweight":   score += 0.25
    if diet == "High":              score += 0.75
    elif diet == "Moderate":        score += 0.25
    if activity == "Low":           score += 0.75
    if smoking == "Yes":            score += 0.75
    if alcohol == "Yes":            score += 0.50
    return round(min(score, 10.0), 2)


def _csv_calc_prevention_index(
    screening: str, early_detection: str, activity: str, diet: str,
) -> float:
    """Prevention_Index [0-10]: active preventive behaviours (CSV pipeline)."""
    score = 0.0
    if screening == "Regular":     score += 3.5
    elif screening == "Irregular": score += 1.5
    if early_detection == "Yes":   score += 2.5
    if activity == "High":         score += 2.0
    elif activity == "Moderate":   score += 1.0
    if diet == "Low":              score += 2.0
    elif diet == "Moderate":       score += 1.0
    return round(min(score, 10.0), 2)


def _csv_calc_access_score(urban_or_rural: str) -> float:
    """Access_Score [0|1]: urban/rural proxy for healthcare access."""
    return 1.0 if urban_or_rural == "Urban" else 0.0


def _csv_calc_age_risk_group(age: int) -> str:
    """Age_Risk_Group ordinal category."""
    if age < 40: return "Low"
    if age < 60: return "Medium"
    if age < 75: return "High"
    return "Very_High"


def _csv_calc_lifestyle_cluster(
    obesity: str, diet: str, activity: str, smoking: str, alcohol: str,
) -> str:
    """Lifestyle_Cluster: 4 modifiable-risk profiles."""
    risk_count = sum([
        obesity in ("Obese", "Overweight"),
        diet == "High",
        activity == "Low",
        smoking == "Yes",
        alcohol == "Yes",
    ])
    if risk_count == 0:                        return "Healthy"
    if risk_count >= 3:                        return "High_Risk"
    if activity == "Low" and diet != "High":   return "Sedentary"
    return "Dietary"


class CsvSyntheticGenerator:
    """
    Generates healthy (Diagnosis=0) synthetic patients that mirror the
    epidemiological feature space of the Kaggle colorectal cancer CSV.

    Used exclusively by the CSV tabular pipeline to balance the dataset
    against real cancer patients (Diagnosis=1).

    Usage
    -----
    gen = CsvSyntheticGenerator(seed=42)
    df_healthy = gen.generate_healthy_patients(n=5000)
    """

    def __init__(self, seed: int = CSV_RANDOM_SEED) -> None:
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    # ── Individual feature generators ──────────────────────────────────────

    def _gen_age(self) -> int:
        if random.random() < 0.05:
            return random.randint(30, 50)
        while True:
            age = int(np.random.normal(loc=69, scale=10))
            if 50 <= age <= 89:
                return age

    def _gen_gender(self) -> str:
        return np.random.choice(["M", "F"], p=[0.55, 0.45])

    def _gen_family_history(self) -> str:
        return "Yes" if random.random() < 0.28 else "No"

    def _gen_smoking(self, age: int) -> str:
        p = 0.20 if age < 30 else 0.35 if age < 50 else 0.40 if age < 65 else 0.28
        return "Yes" if random.random() < p else "No"

    def _gen_alcohol(self, age: int, gender: str) -> str:
        base = 0.40 if gender == "M" else 0.28
        if 30 <= age <= 60:
            base += 0.08
        return "Yes" if random.random() < base else "No"

    def _gen_obesity(self) -> str:
        return np.random.choice(["Normal", "Overweight", "Obese"], p=[0.40, 0.35, 0.25])

    def _gen_diet_risk(self, obesity: str) -> str:
        if obesity == "Obese":
            return np.random.choice(["High", "Moderate", "Low"], p=[0.65, 0.30, 0.05])
        if obesity == "Overweight":
            return np.random.choice(["High", "Moderate", "Low"], p=[0.25, 0.50, 0.25])
        return np.random.choice(["High", "Moderate", "Low"], p=[0.05, 0.35, 0.60])

    def _gen_physical_activity(self, obesity: str) -> str:
        if obesity == "Obese":
            return np.random.choice(["Low", "Moderate", "High"], p=[0.70, 0.25, 0.05])
        if obesity == "Overweight":
            return np.random.choice(["Low", "Moderate", "High"], p=[0.30, 0.45, 0.25])
        return np.random.choice(["Low", "Moderate", "High"], p=[0.10, 0.35, 0.55])

    def _gen_diabetes(self, obesity: str, diet: str, activity: str, age: int) -> str:
        score = 0.0
        if obesity == "Obese":      score += 0.35
        elif obesity == "Overweight": score += 0.15
        if diet == "High":          score += 0.20
        elif diet == "Moderate":    score += 0.08
        if activity == "Low":       score += 0.20
        elif activity == "Moderate": score += 0.08
        if age >= 60:               score += 0.15
        elif age >= 45:             score += 0.07
        return "Yes" if random.random() < min(score, 0.90) else "No"

    def _gen_ibd(self, age: int, family_history: str) -> str:
        base = 0.05
        if family_history == "Yes": base += 0.10
        if 20 <= age <= 40:         base += 0.05
        return "Yes" if random.random() < base else "No"

    def _gen_genetic(self, family_history: str) -> str:
        p = 0.35 if family_history == "Yes" else 0.05
        return "Yes" if random.random() < p else "No"

    def _gen_screening(self, age: int) -> str:
        if age < 30:
            return "Never"
        base_never = 0.85 if age < 45 else 0.40 if age < 60 else 0.25
        p_regular = (1 - base_never) * 0.55
        p_irregular = 1 - base_never - p_regular
        return np.random.choice(
            ["Never", "Regular", "Irregular"],
            p=[base_never, p_regular, p_irregular],
        )

    def _gen_early_detection(self, screening: str) -> str:
        p = 0.72 if screening == "Regular" else 0.30 if screening == "Irregular" else 0.08
        return "Yes" if random.random() < p else "No"

    # ── Main generation method ──────────────────────────────────────────────

    def generate_healthy_patients(self, n: int) -> pd.DataFrame:
        """
        Generate `n` healthy patients (Diagnosis=0) already encoded as
        numeric values, with all derived features computed.

        Returns a DataFrame with the canonical CSV_FINAL_COLUMNS column order.
        """
        records = []
        for _ in range(n):
            age       = self._gen_age()
            gender    = self._gen_gender()
            fam_hist  = self._gen_family_history()
            obesity   = self._gen_obesity()
            diet      = self._gen_diet_risk(obesity)
            activity  = self._gen_physical_activity(obesity)
            smoking   = self._gen_smoking(age)
            alcohol   = self._gen_alcohol(age, gender)
            diabetes  = self._gen_diabetes(obesity, diet, activity, age)
            ibd       = self._gen_ibd(age, fam_hist)
            genetic   = self._gen_genetic(fam_hist)
            screening = self._gen_screening(age)
            early_det = self._gen_early_detection(screening)
            urban     = np.random.choice(["Urban", "Rural"], p=[0.70, 0.30])

            lifestyle = _csv_calc_lifestyle_cluster(obesity, diet, activity, smoking, alcohol)
            age_group = _csv_calc_age_risk_group(age)

            record = {
                "Age":                        age,
                "Gender":                     CSV_GENDER_MAP[gender],
                "Family_History":             CSV_BINARY_MAP[fam_hist],
                "Smoking_History":            CSV_BINARY_MAP[smoking],
                "Alcohol_Consumption":        CSV_BINARY_MAP[alcohol],
                "Obesity_BMI":                CSV_OBESITY_MAP[obesity],
                "Diet_Risk":                  CSV_DIET_MAP[diet],
                "Physical_Activity":          CSV_ACTIVITY_MAP[activity],
                "Diabetes":                   CSV_BINARY_MAP[diabetes],
                "Inflammatory_Bowel_Disease": CSV_BINARY_MAP[ibd],
                "Genetic_Mutation":           CSV_BINARY_MAP[genetic],
                "Screening_History":          CSV_SCREENING_MAP[screening],
                "Early_Detection":            CSV_BINARY_MAP[early_det],
                "Incidence_Rate_per_100K":    round(random.uniform(5, 50), 2),
                "Mortality_Rate_per_100K":    round(random.uniform(2, 20), 2),
                "Urban_or_Rural":             CSV_URBAN_MAP[urban],
                "Risk_Score": _csv_calc_risk_score(
                    obesity, diet, activity, smoking, alcohol,
                    diabetes, ibd, genetic, fam_hist, age,
                ),
                "Prevention_Index": _csv_calc_prevention_index(
                    screening, early_det, activity, diet,
                ),
                "Access_Score":    _csv_calc_access_score(urban),
                "Age_Risk_Group":  CSV_AGE_RISK_GROUP_MAP[age_group],
                "Diagnosis":       0,
                **{
                    f"LC_{opt}": int(lifestyle == opt)
                    for opt in CSV_LIFESTYLE_CLUSTER_OPTIONS
                },
            }
            records.append(record)

        df = pd.DataFrame(records)
        existing = [c for c in CSV_FINAL_COLUMNS if c in df.columns]
        return df[existing]