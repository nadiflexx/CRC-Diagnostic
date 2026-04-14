"""
Synthetic patient data generation with clinical overlap.
"""

import numpy as np
import pandas as pd

from src.config.constants import PHYSIOLOGICAL_RANGES
from src.config.logger import log as logger


class SyntheticPatientGenerator:
    """
    Generator for synthetic clinical patient records with realistic
    physiological distributions and inter-feature correlations.

    Produces both healthy and cancer patient cohorts across multiple
    severity sub-groups so that the resulting dataset has controlled
    class overlap, making it suitable for training and evaluating
    tabular cancer risk models without using real patient data.
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
        """
        Initialise the generator with a fixed random seed.

        Args:
            seed (int): Seed for ``numpy.random.default_rng`` to ensure
                reproducibility across runs. Default is 42.
        """
        self.rng = np.random.default_rng(seed)

    def generate_healthy_patients(self, n: int) -> pd.DataFrame:
        """
        Generate a stratified cohort of healthy (non-cancer) patients.

        The cohort is composed of four sub-groups that differ in their
        baseline risk profiles:
            - 55%: Clean healthy (low risk factors).
            - 20%: Healthy with risk factors (e.g. obesity, smoking).
            - 17%: Healthy with altered lab values (borderline anaemia,
              elevated inflammation).
            - 8%: Tricky cases that resemble early cancer profiles.

        After concatenation, all rows have ``has_cancer = False``.

        Args:
            n (int): Total number of healthy patients to generate.

        Returns:
            pd.DataFrame: Shuffled DataFrame with ``n`` rows and all
                clinical feature columns plus ``has_cancer = False``.
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
        Generate clean healthy patients with low baseline risk.

        All continuous features are sampled from normal or exponential
        distributions parameterised to reflect typical values for a
        healthy population without significant risk factors.

        Args:
            n (int): Number of patients to generate.

        Returns:
            pd.DataFrame: DataFrame with ``n`` rows and all feature
                columns (continuous, categorical, binary).
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
        Generate healthy patients with moderate lifestyle risk factors.

        Elevated smoking rates, higher BMI, and slightly lower albumin
        reflect a population at increased colorectal cancer risk but
        without a current diagnosis.

        Args:
            n (int): Number of patients to generate.

        Returns:
            pd.DataFrame: DataFrame with ``n`` rows and all feature
                columns.
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
        Generate healthy patients with borderline or altered laboratory values.

        Simulates common benign conditions such as mild anaemia, elevated
        CRP from non-malignant inflammation, and low ferritin from
        nutritional deficiency. These patients may be misclassified by
        simple threshold-based rules.

        Args:
            n (int): Number of patients to generate.

        Returns:
            pd.DataFrame: DataFrame with ``n`` rows and all feature
                columns.
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
        Generate healthy patients whose profiles closely resemble early cancer.

        These records intentionally overlap with the cancer distribution
        to ensure the model cannot rely on simple decision boundaries.
        All patients remain ``has_cancer = False``.

        Args:
            n (int): Number of patients to generate.

        Returns:
            pd.DataFrame: DataFrame with ``n`` rows and all feature
                columns.
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

    def generate_cancer_patients(self, n: int) -> pd.DataFrame:
        """
        Generate a stratified cohort of cancer patients.

        The cohort is composed of four sub-groups:
            - 30%: Early-stage cancer (mild lab deviations).
            - 35%: Moderate-stage cancer.
            - 25%: Advanced-stage cancer (pronounced deviations).
            - 10%: Atypical cancer (young, few risk factors).

        After concatenation, all rows have ``has_cancer = True``.

        Args:
            n (int): Total number of cancer patients to generate.

        Returns:
            pd.DataFrame: Shuffled DataFrame with ``n`` rows and all
                clinical feature columns plus ``has_cancer = True``.
        """
        logger.info(f"Generating {n} cancer patients...")
        n_early = int(n * 0.30)
        n_moderate = int(n * 0.35)
        n_advanced = int(n * 0.25)
        n_atypical = n - n_early - n_moderate - n_advanced
        dfs = [
            self._cancer_early_stage(n_early),
            self._cancer_moderate(n_moderate),
            self._cancer_advanced(n_advanced),
            self._cancer_atypical(n_atypical),
        ]
        df = pd.concat(dfs, ignore_index=True)
        df["has_cancer"] = True
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"✅ Generated {len(df)} cancer patients")
        return df

    def _cancer_early_stage(self, n):
        """
        Generate early-stage cancer patients with mild lab abnormalities.

        Lab values overlap substantially with healthy ranges, making these
        records the hardest cases for the classifier.

        Args:
            n (int): Number of patients to generate.

        Returns:
            pd.DataFrame: DataFrame with ``n`` rows and all feature
                columns.
        """
        smoked = self.rng.binomial(1, 0.38, n)
        pack_years = smoked * self.rng.exponential(5.5, n)
        has_polyps = self.rng.binomial(1, 0.32, n)
        polyp_count = has_polyps * self.rng.poisson(1.5, n)
        return pd.DataFrame(
            {
                "age": self.rng.normal(58, 14, n).clip(25, 90),
                "bmi": self.rng.normal(27, 4.5, n).clip(18, 42),
                "hemoglobin": self.rng.normal(13.2, 1.8, n).clip(9.0, 16.5),
                "hematocrit": self.rng.normal(40, 4.2, n).clip(29, 50),
                "wbc_count": self.rng.normal(7.3, 1.8, n).clip(3.8, 12.5),
                "platelet_count": self.rng.normal(268, 62, n).clip(150, 430),
                "albumin": self.rng.normal(4.0, 0.42, n).clip(3.0, 5.0),
                "iron_serum": self.rng.normal(78, 28, n).clip(30, 160),
                "ferritin": self.rng.normal(68, 42, n).clip(10, 230),
                "crp": self.rng.exponential(0.6, n).clip(0.03, 5.0),
                "cea": (
                    self.rng.exponential(1.3, n) + smoked * self.rng.exponential(1.0, n)
                ).clip(0.2, 9.0),
                "ca19_9": self.rng.exponential(9.0, n).clip(1.0, 42),
                "pack_years_smoked": pack_years.clip(0, 38),
                "previous_polyps_count": polyp_count.clip(0, 7),
                **self._cancer_categorical(n, "low"),
                **self._cancer_binary(n, "low"),
            }
        )

    def _cancer_moderate(self, n):
        """
        Generate moderate-stage cancer patients.

        CEA and CA19-9 follow log-normal distributions to capture the
        right-skewed marker elevations typical at this stage.

        Args:
            n (int): Number of patients to generate.

        Returns:
            pd.DataFrame: DataFrame with ``n`` rows and all feature
                columns.
        """
        smoked = self.rng.binomial(1, 0.48, n)
        pack_years = smoked * self.rng.exponential(7.0, n)
        has_polyps = self.rng.binomial(1, 0.45, n)
        polyp_count = has_polyps * self.rng.poisson(2.0, n)
        return pd.DataFrame(
            {
                "age": self.rng.normal(63, 11, n).clip(30, 92),
                "bmi": self.rng.normal(28, 5, n).clip(18, 45),
                "hemoglobin": self.rng.normal(11.8, 1.9, n).clip(7.5, 15.5),
                "hematocrit": self.rng.normal(37, 4.5, n).clip(25, 48),
                "wbc_count": self.rng.normal(8.0, 2.3, n).clip(3.5, 14.0),
                "platelet_count": self.rng.normal(298, 78, n).clip(150, 510),
                "albumin": self.rng.normal(3.6, 0.52, n).clip(2.4, 4.8),
                "iron_serum": self.rng.normal(58, 25, n).clip(18, 140),
                "ferritin": self.rng.normal(42, 30, n).clip(5, 185),
                "crp": self.rng.exponential(1.8, n).clip(0.1, 10.0),
                "cea": (
                    self.rng.lognormal(1.3, 0.9, n)
                    + smoked * self.rng.exponential(1.0, n)
                ).clip(0.5, 55),
                "ca19_9": self.rng.lognormal(2.8, 0.9, n).clip(2.0, 130),
                "pack_years_smoked": pack_years.clip(0, 48),
                "previous_polyps_count": polyp_count.clip(0, 10),
                **self._cancer_categorical(n, "medium"),
                **self._cancer_binary(n, "medium"),
            }
        )

    def _cancer_advanced(self, n):
        """
        Generate advanced-stage cancer patients with pronounced deviations.

        Severe anaemia, hypoalbuminaemia, markedly elevated tumour
        markers, and high CRP characterise this sub-group.

        Args:
            n (int): Number of patients to generate.

        Returns:
            pd.DataFrame: DataFrame with ``n`` rows and all feature
                columns.
        """
        smoked = self.rng.binomial(1, 0.55, n)
        pack_years = smoked * self.rng.exponential(9.0, n)
        has_polyps = self.rng.binomial(1, 0.55, n)
        polyp_count = has_polyps * self.rng.poisson(3.0, n)
        return pd.DataFrame(
            {
                "age": self.rng.normal(67, 10, n).clip(35, 95),
                "bmi": self.rng.normal(26, 5.5, n).clip(16, 45),
                "hemoglobin": self.rng.normal(10.2, 1.9, n).clip(5.5, 14.0),
                "hematocrit": self.rng.normal(33, 5, n).clip(19, 44),
                "wbc_count": self.rng.normal(9.5, 3.0, n).clip(3.5, 18.0),
                "platelet_count": self.rng.normal(345, 100, n).clip(150, 600),
                "albumin": self.rng.normal(3.1, 0.6, n).clip(1.7, 4.5),
                "iron_serum": self.rng.normal(38, 20, n).clip(8, 105),
                "ferritin": self.rng.normal(22, 18, n).clip(2, 105),
                "crp": self.rng.exponential(3.5, n).clip(0.3, 18.0),
                "cea": (
                    self.rng.lognormal(2.3, 1.1, n)
                    + smoked * self.rng.exponential(1.0, n)
                ).clip(1.5, 200),
                "ca19_9": self.rng.lognormal(3.5, 1.1, n).clip(4.0, 500),
                "pack_years_smoked": pack_years.clip(0, 60),
                "previous_polyps_count": polyp_count.clip(0, 15),
                **self._cancer_categorical(n, "high"),
                **self._cancer_binary(n, "high"),
            }
        )

    def _cancer_atypical(self, n):
        """
        Generate atypical cancer patients with near-normal lab profiles.

        Represents young patients with few conventional risk factors and
        lab values close to the healthy range, simulating hereditary or
        Lynch syndrome-related cancers.

        Args:
            n (int): Number of patients to generate.

        Returns:
            pd.DataFrame: DataFrame with ``n`` rows and all feature
                columns.
        """
        smoked = self.rng.binomial(1, 0.22, n)
        pack_years = smoked * self.rng.exponential(3.0, n)
        has_polyps = self.rng.binomial(1, 0.18, n)
        polyp_count = has_polyps * self.rng.poisson(1.0, n)
        return pd.DataFrame(
            {
                "age": self.rng.normal(40, 8, n).clip(20, 55),
                "bmi": self.rng.normal(25.5, 4, n).clip(18, 38),
                "hemoglobin": self.rng.normal(13.8, 1.4, n).clip(10, 17),
                "hematocrit": self.rng.normal(41.5, 3.5, n).clip(32, 50),
                "wbc_count": self.rng.normal(7.0, 1.5, n).clip(4.0, 11.0),
                "platelet_count": self.rng.normal(252, 55, n).clip(150, 400),
                "albumin": self.rng.normal(4.2, 0.35, n).clip(3.4, 5.1),
                "iron_serum": self.rng.normal(90, 26, n).clip(42, 160),
                "ferritin": self.rng.normal(85, 48, n).clip(12, 230),
                "crp": self.rng.exponential(0.4, n).clip(0.02, 3.5),
                "cea": (
                    self.rng.exponential(1.0, n) + smoked * self.rng.exponential(0.8, n)
                ).clip(0.1, 6.0),
                "ca19_9": self.rng.exponential(7.5, n).clip(0.5, 35),
                "pack_years_smoked": pack_years.clip(0, 18),
                "previous_polyps_count": polyp_count.clip(0, 4),
                **self._cancer_categorical(n, "low"),
                **self._cancer_binary(n, "low"),
            }
        )

    def _healthy_categorical(self, n, risk_level):
        """
        Sample categorical lifestyle features for healthy patients.

        Probabilities for gender, ethnicity, smoking status, alcohol
        consumption, physical activity, and diet type are drawn from
        risk-level-specific distributions (``"low"``, ``"medium"``, or
        ``"high"``). Higher risk levels shift distributions towards
        sedentary behaviour, heavy smoking, and poor diet.

        Args:
            n (int): Number of samples.
            risk_level (str): One of ``"low"``, ``"medium"``, or
                ``"high"``.

        Returns:
            dict[str, np.ndarray]: Dictionary mapping feature names to
                arrays of sampled categorical values, each of length ``n``.
        """
        configs = {
            "low": {
                "gender": (["male", "female"], [0.48, 0.52]),
                "ethnicity": (
                    ["white", "black", "hispanic", "asian", "other"],
                    [0.60, 0.12, 0.18, 0.07, 0.03],
                ),
                "smoking_status": (["never", "former", "current"], [0.68, 0.22, 0.10]),
                "alcohol_consumption": (
                    ["none", "moderate", "heavy"],
                    [0.35, 0.52, 0.13],
                ),
                "physical_activity": (
                    ["active", "moderate", "sedentary"],
                    [0.38, 0.40, 0.22],
                ),
                "diet_type": (
                    ["high_fiber", "balanced", "high_fat_low_fiber"],
                    [0.30, 0.46, 0.24],
                ),
            },
            "medium": {
                "gender": (["male", "female"], [0.51, 0.49]),
                "ethnicity": (
                    ["white", "black", "hispanic", "asian", "other"],
                    [0.58, 0.14, 0.17, 0.07, 0.04],
                ),
                "smoking_status": (["never", "former", "current"], [0.50, 0.28, 0.22]),
                "alcohol_consumption": (
                    ["none", "moderate", "heavy"],
                    [0.28, 0.45, 0.27],
                ),
                "physical_activity": (
                    ["active", "moderate", "sedentary"],
                    [0.28, 0.38, 0.34],
                ),
                "diet_type": (
                    ["high_fiber", "balanced", "high_fat_low_fiber"],
                    [0.24, 0.42, 0.34],
                ),
            },
            "high": {
                "gender": (["male", "female"], [0.53, 0.47]),
                "ethnicity": (
                    ["white", "black", "hispanic", "asian", "other"],
                    [0.56, 0.16, 0.16, 0.07, 0.05],
                ),
                "smoking_status": (["never", "former", "current"], [0.42, 0.30, 0.28]),
                "alcohol_consumption": (
                    ["none", "moderate", "heavy"],
                    [0.22, 0.42, 0.36],
                ),
                "physical_activity": (
                    ["active", "moderate", "sedentary"],
                    [0.18, 0.35, 0.47],
                ),
                "diet_type": (
                    ["high_fiber", "balanced", "high_fat_low_fiber"],
                    [0.15, 0.38, 0.47],
                ),
            },
        }
        cfg = configs[risk_level]
        return {k: self.rng.choice(v[0], n, p=v[1]) for k, v in cfg.items()}

    def _cancer_categorical(self, n, severity):
        """
        Sample categorical lifestyle features for cancer patients.

        Mirrors ``_healthy_categorical`` but uses severity-specific
        distributions (``"low"``, ``"medium"``, or ``"high"``) that
        shift incrementally towards higher-risk profiles as severity
        increases.

        Args:
            n (int): Number of samples.
            severity (str): One of ``"low"``, ``"medium"``, or
                ``"high"``.

        Returns:
            dict[str, np.ndarray]: Dictionary mapping feature names to
                arrays of sampled categorical values, each of length ``n``.
        """
        configs = {
            "low": {
                "gender": (["male", "female"], [0.51, 0.49]),
                "ethnicity": (
                    ["white", "black", "hispanic", "asian", "other"],
                    [0.57, 0.15, 0.17, 0.07, 0.04],
                ),
                "smoking_status": (["never", "former", "current"], [0.52, 0.26, 0.22]),
                "alcohol_consumption": (
                    ["none", "moderate", "heavy"],
                    [0.30, 0.44, 0.26],
                ),
                "physical_activity": (
                    ["active", "moderate", "sedentary"],
                    [0.28, 0.38, 0.34],
                ),
                "diet_type": (
                    ["high_fiber", "balanced", "high_fat_low_fiber"],
                    [0.22, 0.40, 0.38],
                ),
            },
            "medium": {
                "gender": (["male", "female"], [0.53, 0.47]),
                "ethnicity": (
                    ["white", "black", "hispanic", "asian", "other"],
                    [0.55, 0.17, 0.16, 0.07, 0.05],
                ),
                "smoking_status": (["never", "former", "current"], [0.44, 0.30, 0.26]),
                "alcohol_consumption": (
                    ["none", "moderate", "heavy"],
                    [0.24, 0.42, 0.34],
                ),
                "physical_activity": (
                    ["active", "moderate", "sedentary"],
                    [0.20, 0.36, 0.44],
                ),
                "diet_type": (
                    ["high_fiber", "balanced", "high_fat_low_fiber"],
                    [0.15, 0.35, 0.50],
                ),
            },
            "high": {
                "gender": (["male", "female"], [0.55, 0.45]),
                "ethnicity": (
                    ["white", "black", "hispanic", "asian", "other"],
                    [0.53, 0.19, 0.15, 0.07, 0.06],
                ),
                "smoking_status": (["never", "former", "current"], [0.38, 0.30, 0.32]),
                "alcohol_consumption": (
                    ["none", "moderate", "heavy"],
                    [0.20, 0.38, 0.42],
                ),
                "physical_activity": (
                    ["active", "moderate", "sedentary"],
                    [0.14, 0.32, 0.54],
                ),
                "diet_type": (
                    ["high_fiber", "balanced", "high_fat_low_fiber"],
                    [0.10, 0.30, 0.60],
                ),
            },
        }
        cfg = configs[severity]
        return {k: self.rng.choice(v[0], n, p=v[1]) for k, v in cfg.items()}

    def _healthy_binary(self, n, risk_level):
        """
        Sample binary clinical risk indicators for healthy patients.

        Probabilities cover family history, comorbidities (IBD, T2DM),
        prior polyp or cancer history, and stool-based screening test
        results. All probabilities increase progressively from ``"low"``
        to ``"high"`` risk level.

        Args:
            n (int): Number of samples.
            risk_level (str): One of ``"low"``, ``"medium"``, or
                ``"high"``.

        Returns:
            dict[str, np.ndarray]: Dictionary mapping binary feature names
                to boolean arrays of length ``n``.
        """
        probs = {
            "low": {
                "family_history_ccr": 0.05,
                "family_history_polyps": 0.07,
                "family_history_lynch": 0.01,
                "family_history_fap": 0.004,
                "has_ibd": 0.02,
                "has_diabetes_t2": 0.07,
                "previous_polyps": 0.12,
                "previous_cancer": 0.015,
                "fobt_positive": 0.05,
                "fit_positive": 0.06,
            },
            "medium": {
                "family_history_ccr": 0.10,
                "family_history_polyps": 0.14,
                "family_history_lynch": 0.025,
                "family_history_fap": 0.01,
                "has_ibd": 0.06,
                "has_diabetes_t2": 0.13,
                "previous_polyps": 0.22,
                "previous_cancer": 0.04,
                "fobt_positive": 0.09,
                "fit_positive": 0.10,
            },
            "high": {
                "family_history_ccr": 0.16,
                "family_history_polyps": 0.22,
                "family_history_lynch": 0.04,
                "family_history_fap": 0.015,
                "has_ibd": 0.10,
                "has_diabetes_t2": 0.18,
                "previous_polyps": 0.32,
                "previous_cancer": 0.08,
                "fobt_positive": 0.15,
                "fit_positive": 0.14,
            },
        }
        p = probs[risk_level]
        return {k: self.rng.binomial(1, v, n).astype(bool) for k, v in p.items()}

    def _cancer_binary(self, n, severity):
        """
        Sample binary clinical risk indicators for cancer patients.

        Same features as ``_healthy_binary`` but with higher base
        probabilities and a much higher rate of positive screening tests
        (FOBT, FIT) to reflect the malignant phenotype.

        Args:
            n (int): Number of samples.
            severity (str): One of ``"low"``, ``"medium"``, or
                ``"high"``.

        Returns:
            dict[str, np.ndarray]: Dictionary mapping binary feature names
                to boolean arrays of length ``n``.
        """
        probs = {
            "low": {
                "family_history_ccr": 0.14,
                "family_history_polyps": 0.18,
                "family_history_lynch": 0.03,
                "family_history_fap": 0.012,
                "has_ibd": 0.08,
                "has_diabetes_t2": 0.14,
                "previous_polyps": 0.30,
                "previous_cancer": 0.06,
                "fobt_positive": 0.32,
                "fit_positive": 0.38,
            },
            "medium": {
                "family_history_ccr": 0.20,
                "family_history_polyps": 0.25,
                "family_history_lynch": 0.045,
                "family_history_fap": 0.018,
                "has_ibd": 0.12,
                "has_diabetes_t2": 0.17,
                "previous_polyps": 0.42,
                "previous_cancer": 0.10,
                "fobt_positive": 0.52,
                "fit_positive": 0.58,
            },
            "high": {
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
        }
        p = probs[severity]
        return {k: self.rng.binomial(1, v, n).astype(bool) for k, v in p.items()}

    def _clip_physiological(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clip all feature columns to their valid physiological ranges.

        Ranges are defined in ``PHYSIOLOGICAL_RANGES`` from
        ``src.config.constants``. Columns not present in the constant
        are left unchanged.

        Args:
            df (pd.DataFrame): DataFrame to clip in place.

        Returns:
            pd.DataFrame: DataFrame with all covered columns clipped to
                ``[lo, hi]`` as defined by ``PHYSIOLOGICAL_RANGES``.
        """
        for col, (lo, hi) in PHYSIOLOGICAL_RANGES.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)
        return df

    def _add_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Introduce physiologically motivated correlations between features.

        Adjustments applied:
            - Haematocrit, serum iron, and ferritin co-vary with
              haemoglobin (positive correlation).
            - WBC count and platelet count co-vary with CRP (positive
              correlation reflecting systemic inflammation).
            - BMI co-varies with albumin (slight positive correlation).

        All adjustments include additive Gaussian noise to avoid perfect
        linear relationships. Results are clipped to physiological ranges
        via ``_clip_physiological``.

        Args:
            df (pd.DataFrame): DataFrame of generated patient records.

        Returns:
            pd.DataFrame: Copy of ``df`` with correlated adjustments
                applied and values clipped.
        """
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
        """
        Simulate analytical measurement variability in laboratory results.

        Multiplies each feature by a factor drawn from
        ``Normal(1.0, CV)`` where ``CV`` is the per-feature coefficient
        of variation defined in the method body. Age is left unchanged
        (CV = 0). Results are clipped to physiological ranges.

        Args:
            df (pd.DataFrame): DataFrame of patient records after
                correlation adjustment.

        Returns:
            pd.DataFrame: Copy of ``df`` with multiplicative noise
                applied to each covered feature and values re-clipped.
        """
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
        """
        Generate a balanced dataset with equal numbers of healthy and cancer patients.

        Calls ``generate_healthy_patients`` and ``generate_cancer_patients``
        independently, concatenates the results, shuffles the combined
        DataFrame, applies correlation adjustments and measurement noise,
        and runs ``_verify_overlap`` to log distributional statistics.

        Args:
            n_per_class (int): Number of patients per class. The resulting
                dataset will have ``2 * n_per_class`` rows in total.
                Default is 5000.

        Returns:
            pd.DataFrame: Fully processed, shuffled DataFrame with
                ``2 * n_per_class`` rows and a reset integer index.
        """
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
        """
        Generate healthy patients and apply correlation and noise post-processing.

        Convenience wrapper that combines ``generate_healthy_patients``,
        ``_add_correlations``, and ``_add_measurement_noise`` in a single
        call, suitable for use in multimodal linking workflows.

        Args:
            n (int): Number of healthy patients to generate.

        Returns:
            pd.DataFrame: Processed healthy patient DataFrame.
        """
        df = self.generate_healthy_patients(n)
        df = self._add_correlations(df)
        return self._add_measurement_noise(df)

    def generate_cancer_patients_standalone(self, n):
        """
        Generate cancer patients and apply correlation and noise post-processing.

        Convenience wrapper that combines ``generate_cancer_patients``,
        ``_add_correlations``, and ``_add_measurement_noise`` in a single
        call, suitable for use in multimodal linking workflows.

        Args:
            n (int): Number of cancer patients to generate.

        Returns:
            pd.DataFrame: Processed cancer patient DataFrame.
        """
        df = self.generate_cancer_patients(n)
        df = self._add_correlations(df)
        return self._add_measurement_noise(df)

    def _verify_overlap(self, df):
        """
        Log Cohen's d effect sizes and binary feature gap statistics.

        For each continuous feature, computes Cohen's d between the
        healthy and cancer groups and labels the effect size as Low
        (d < 0.5), Medium (< 0.8), High (< 1.2), or Very High (≥ 1.2).
        For binary features, logs the positive rate per group and the
        absolute gap. This provides a quick sanity check that the
        generated dataset has realistic overlap rather than trivially
        separable classes.

        Args:
            df (pd.DataFrame): Combined healthy + cancer DataFrame with
                the ``has_cancer`` column.
        """
        cancer = df[df["has_cancer"]]
        healthy = df[~df["has_cancer"]]
        logger.info("═══ Overlap Verification ═══")
        for feat in [
            "hemoglobin",
            "cea",
            "albumin",
            "iron_serum",
            "age",
            "pack_years_smoked",
            "ferritin",
            "crp",
        ]:
            if feat not in df.columns:
                continue
            c_mean, c_std = cancer[feat].mean(), cancer[feat].std()
            h_mean, h_std = healthy[feat].mean(), healthy[feat].std()
            pooled_std = np.sqrt((c_std**2 + h_std**2) / 2)
            cohens_d = abs(c_mean - h_mean) / max(pooled_std, 0.01)
            status = (
                "✅ Low"
                if cohens_d < 0.5
                else "✅ Medium"
                if cohens_d < 0.8
                else "⚠️ High"
                if cohens_d < 1.2
                else "🔴 VERY High"
            )
            logger.info(
                f"  {feat:>22s}: healthy={h_mean:.1f}±{h_std:.1f} | "
                f"cancer={c_mean:.1f}±{c_std:.1f} | "
                f"d={cohens_d:.2f} {status}"
            )
        for feat in ["fobt_positive", "fit_positive", "previous_polyps"]:
            if feat not in df.columns:
                continue
            h_rate = healthy[feat].mean()
            c_rate = cancer[feat].mean()
            gap = abs(c_rate - h_rate)
            status = "✅" if gap < 0.30 else "⚠️" if gap < 0.45 else "🔴"
            logger.info(
                f"  {feat:>22s}: healthy={h_rate:.1%} | "
                f"cancer={c_rate:.1%} | gap={gap:.1%} {status}"
            )


class SyntheticMultimodalLinker:
    """
    Links colonoscopy image paths with synthetic tabular patient profiles.

    Assigns a generated healthy or cancer patient profile to each image
    based on its label, creating a paired multimodal dataset for
    training or evaluation purposes.
    """

    def __init__(self, generator: SyntheticPatientGenerator):
        """
        Initialise the linker with a configured patient generator.

        Args:
            generator (SyntheticPatientGenerator): Pre-initialised
                generator instance used to produce healthy and cancer
                patient profiles.
        """
        self.generator = generator

    def link_images_with_tabular(self, image_paths, image_labels):
        """
        Pair each image path with a synthetic patient profile.

        Generates one healthy profile per normal image and one cancer
        profile per polyp image, then assembles a flat DataFrame with
        one row per image containing both the tabular features and the
        image metadata.

        Args:
            image_paths (list[str]): Ordered list of absolute image file
                paths.
            image_labels (list[int]): Ordered list of integer labels
                aligned with ``image_paths``. Label 1 is treated as
                polyp/cancer; label 0 as normal/healthy.

        Returns:
            pd.DataFrame: DataFrame with one row per image, containing
                all generated tabular features plus ``image_path`` and
                ``image_label`` columns.
        """
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
