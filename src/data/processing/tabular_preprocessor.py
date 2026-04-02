"""
Tabular clinical data preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config.constants import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CSV_ACTIVITY_MAP,
    CSV_AGE_RISK_GROUP_MAP,
    CSV_BINARY_MAP,
    CSV_COLUMNS_TO_DROP,
    CSV_DIET_MAP,
    CSV_FINAL_COLUMNS,
    CSV_GENDER_MAP,
    CSV_LIFESTYLE_CLUSTER_OPTIONS,
    CSV_OBESITY_MAP,
    CSV_SCREENING_MAP,
    CSV_URBAN_MAP,
    GDC_ALCOHOL_MAP,
    GDC_TOBACCO_MAP,
    KAGGLE_ACTIVITY_MAP,
    KAGGLE_ALCOHOL_MAP,
    KAGGLE_DIET_MAP,
    KAGGLE_RACE_MAP,
    KAGGLE_SMOKING_MAP,
    KAGGLE_STAGE_MAP,
    NUMERIC_FEATURES,
    TABULAR_TARGET,
)
from src.config.logger import log as logger


# ═══════════════════════════════════════════════════════════
#  EXISTING: clinical lab preprocessor
# ═══════════════════════════════════════════════════════════

class TabularPreprocessor:
    NUMERIC_FEATURES = NUMERIC_FEATURES
    CATEGORICAL_FEATURES = CATEGORICAL_FEATURES
    BINARY_FEATURES = BINARY_FEATURES
    TARGET = TABULAR_TARGET

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self._fitted = False

    def clean_gdc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean GDC clinical data."""
        cleaned = pd.DataFrame()

        if "age_at_index" in df.columns:
            cleaned["age"] = pd.to_numeric(df["age_at_index"], errors="coerce")
        elif "age_at_diagnosis" in df.columns:
            cleaned["age"] = (
                pd.to_numeric(df["age_at_diagnosis"], errors="coerce") / 365.25
            )

        cleaned["gender"] = (
            df["gender"].fillna("unknown").astype(str).str.lower()
            if "gender" in df.columns
            else pd.Series(["unknown"] * len(df))
        )
        cleaned["bmi"] = (
            pd.to_numeric(df["bmi"], errors="coerce")
            if "bmi" in df.columns
            else pd.Series([np.nan] * len(df))
        )
        cleaned["ethnicity"] = (
            df["race"].fillna("unknown").astype(str).str.lower()
            if "race" in df.columns
            else pd.Series(["unknown"] * len(df))
        )

        if "tobacco_smoking_status" in df.columns:
            cleaned["smoking_status"] = (
                df["tobacco_smoking_status"]
                .fillna("unknown")
                .astype(str)
                .str.lower()
                .map(GDC_TOBACCO_MAP)
                .fillna("unknown")
            )
        else:
            cleaned["smoking_status"] = "unknown"

        if "alcohol_history" in df.columns:
            cleaned["alcohol_consumption"] = (
                df["alcohol_history"]
                .fillna("unknown")
                .astype(str)
                .str.lower()
                .map(GDC_ALCOHOL_MAP)
                .fillna("unknown")
            )
        else:
            cleaned["alcohol_consumption"] = "unknown"

        cleaned["pack_years_smoked"] = (
            pd.to_numeric(df["pack_years_smoked"], errors="coerce").fillna(0)
            if "pack_years_smoked" in df.columns
            else 0.0
        )
        cleaned["ajcc_stage"] = (
            df["ajcc_stage"].fillna("unknown").astype(str)
            if "ajcc_stage" in df.columns
            else "unknown"
        )
        cleaned["has_cancer"] = True
        return cleaned

    def clean_kaggle_risk_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean Kaggle risk factor data."""
        cleaned = pd.DataFrame()

        cleaned["age"] = (
            pd.to_numeric(df["Age"], errors="coerce") if "Age" in df.columns else np.nan
        )
        cleaned["gender"] = (
            df["Gender"].str.lower() if "Gender" in df.columns else "unknown"
        )
        cleaned["ethnicity"] = (
            df["Race"].map(KAGGLE_RACE_MAP).fillna("other")
            if "Race" in df.columns
            else "unknown"
        )
        cleaned["bmi"] = (
            pd.to_numeric(df["BMI"], errors="coerce") if "BMI" in df.columns else np.nan
        )
        cleaned["smoking_status"] = (
            df["Smoking_Status"].map(KAGGLE_SMOKING_MAP).fillna("unknown")
            if "Smoking_Status" in df.columns
            else "unknown"
        )
        cleaned["alcohol_consumption"] = (
            df["Alcohol_Consumption"].map(KAGGLE_ALCOHOL_MAP).fillna("unknown")
            if "Alcohol_Consumption" in df.columns
            else "unknown"
        )
        cleaned["physical_activity"] = (
            df["Physical_Activity_Level"].map(KAGGLE_ACTIVITY_MAP).fillna("unknown")
            if "Physical_Activity_Level" in df.columns
            else "unknown"
        )
        cleaned["diet_type"] = (
            df["Diet_Type"].map(KAGGLE_DIET_MAP).fillna("balanced")
            if "Diet_Type" in df.columns
            else "balanced"
        )
        cleaned["family_history_ccr"] = (
            df["Family_History"].map({"Yes": True, "No": False}).fillna(False)
            if "Family_History" in df.columns
            else False
        )
        cleaned["previous_cancer"] = (
            df["Previous_Cancer_History"].map({"Yes": True, "No": False}).fillna(False)
            if "Previous_Cancer_History" in df.columns
            else False
        )
        cleaned["has_cancer"] = True

        if "Stage_at_Diagnosis" in df.columns:
            cleaned["cancer_stage"] = df["Stage_at_Diagnosis"]
            cleaned["cancer_stage_numeric"] = df["Stage_at_Diagnosis"].map(KAGGLE_STAGE_MAP)
        if "Tumor_Aggressiveness" in df.columns:
            cleaned["tumor_aggressiveness"] = df["Tumor_Aggressiveness"].map(
                {"Low": 1, "Medium": 2, "High": 3}
            )
        if "Survival_Status" in df.columns:
            cleaned["survived"] = df["Survival_Status"].map(
                {"Survived": True, "Deceased": False}
            )
        return cleaned

    def merge_datasets(self, *dataframes: pd.DataFrame) -> pd.DataFrame:
        """Merge multiple DataFrames into a single DataFrame."""
        logger.info(f"Merging {len(dataframes)} datasets")
        combined = pd.concat(dataframes, ignore_index=True, sort=False)
        if "has_cancer" in combined.columns:
            combined["has_cancer"] = combined["has_cancer"].fillna(False).astype(bool)
        logger.info(f"Datasets merged: {len(combined)} rows")
        return combined

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features based on existing columns."""
        df = df.copy()
        if "hemoglobin" in df.columns and "hematocrit" in df.columns:
            df["hb_hct_ratio"] = df["hemoglobin"] / df["hematocrit"].clip(lower=1)
        if "cea" in df.columns and "albumin" in df.columns:
            df["cea_albumin_ratio"] = df["cea"] / df["albumin"].clip(lower=0.1)
        if "crp" in df.columns and "wbc_count" in df.columns:
            df["inflammation_index"] = np.log1p(df["crp"] * df["wbc_count"])
        if "ferritin" in df.columns and "iron_serum" in df.columns:
            df["iron_store_ratio"] = df["ferritin"] / df["iron_serum"].clip(lower=1)
        if "age" in df.columns and "bmi" in df.columns:
            df["age_bmi_interaction"] = (df["age"] / 100) * (df["bmi"] / 40)
        for col in ["cea", "ca19_9", "crp", "ferritin"]:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
        return df

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
        """Fit and transform the data."""
        df = self.engineer_features(df)
        y = (
            np.asarray(df[self.TARGET].astype(int).values)
            if self.TARGET in df.columns
            else None
        )

        num_cols = [c for c in self.NUMERIC_FEATURES if c in df.columns]
        eng_cols = [
            c for c in df.columns
            if c.endswith("_risk") or c.endswith("_elevated")
            or c in ["anemia", "low_albumin", "family_risk_score", "composite_risk_score"]
        ]
        num_cols = list(set(num_cols + eng_cols))
        cat_cols = [c for c in self.CATEGORICAL_FEATURES if c in df.columns]
        bin_cols = [c for c in self.BINARY_FEATURES if c in df.columns]

        if num_cols:
            num_data = df[num_cols].copy()
            imputer = KNNImputer(n_neighbors=5)
            num_imputed = imputer.fit_transform(num_data)
            num_scaled = self.scaler.fit_transform(num_imputed)
        else:
            num_scaled = np.empty((len(df), 0))

        cat_encoded_list = []
        for col in cat_cols:
            le = LabelEncoder()
            encoded = le.fit_transform(df[col].fillna("unknown").astype(str))
            self.label_encoders[col] = le
            cat_encoded_list.append(encoded.reshape(-1, 1))
        cat_encoded = (
            np.hstack(cat_encoded_list) if cat_encoded_list else np.empty((len(df), 0))
        )

        if bin_cols:
            bin_data = df[bin_cols].infer_objects(copy=False).fillna(0).astype(int).values
        else:
            bin_data = np.empty((len(df), 0))

        X = np.hstack([num_scaled, cat_encoded, bin_data])
        self.feature_names = num_cols + cat_cols + bin_cols
        self._fitted = True
        logger.info(f"Features: {len(self.feature_names)}, Samples: {len(X)}")
        return X, y, self.feature_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform the data."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        df = self.engineer_features(df)
        num_cols = [
            c for c in self.feature_names
            if c not in self.CATEGORICAL_FEATURES + self.BINARY_FEATURES
        ]
        cat_cols = [c for c in self.feature_names if c in self.CATEGORICAL_FEATURES]
        bin_cols = [c for c in self.feature_names if c in self.BINARY_FEATURES]

        num_scaled = (
            self.scaler.transform(df[num_cols].fillna(0).values)
            if num_cols else np.empty((len(df), 0))
        )
        cat_encoded_list = []
        for col in cat_cols:
            le = self.label_encoders.get(col)
            vals = df[col].fillna("unknown").astype(str)
            encoded = [le.transform([v])[0] if v in le.classes_ else -1 for v in vals]
            cat_encoded_list.append(np.array(encoded).reshape(-1, 1))
        cat_encoded = (
            np.hstack(cat_encoded_list) if cat_encoded_list else np.empty((len(df), 0))
        )
        bin_data = (
            df[bin_cols].infer_objects(copy=False).fillna(0).astype(int).values
            if bin_cols else np.empty((len(df), 0))
        )
        return np.hstack([num_scaled, cat_encoded, bin_data])

    def save(self, path: Path):
        """Save the preprocessor."""
        joblib.dump(
            {
                "scaler": self.scaler,
                "label_encoders": self.label_encoders,
                "feature_names": self.feature_names,
            },
            path,
        )
        logger.info(f"Preprocessor saved: {path}")

    @classmethod
    def load(cls, path: Path) -> "TabularPreprocessor":
        """Load the preprocessor."""
        state = joblib.load(path)
        p = cls()
        p.scaler = state["scaler"]
        p.label_encoders = state["label_encoders"]
        p.feature_names = state["feature_names"]
        p._fitted = True
        return p


# ═══════════════════════════════════════════════════════════
#  NEW: CSV-based preprocessor (colorectal_cancer_dataset.csv)
# ═══════════════════════════════════════════════════════════

# ── Epidemiological defaults (median values from the Kaggle dataset) ─────────
# These are applied automatically for fields not collected in the clinical form.
# They ensure the model receives a meaningful value without burdening the user.
#
# Field               Median     Rationale
# ────────────────────────────────────────────────────────────────────────────
# Incidence_Rate      20.0 /100k Global CRC median (GLOBOCAN 2022)
# Mortality_Rate       8.0 /100k Global CRC mortality median (GLOBOCAN 2022)
# Early_Detection      auto       Derived from Screening_History (see below)
#
# Early_Detection derivation rule:
#   Regular   → "Yes"   (regular screening catches early-stage disease)
#   Irregular → "No"    (inconsistent follow-up, likely no early detection)
#   Never     → "No"    (no screening, no early detection)

_DEFAULT_INCIDENCE_RATE: float = 20.0
_DEFAULT_MORTALITY_RATE: float = 8.0


def _derive_early_detection(screening: str) -> str:
    """Derive Early_Detection from Screening_History instead of asking the user."""
    return "Yes" if screening == "Regular" else "No"


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
    """Prevention_Index [0-10]: active preventive behaviours."""
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


class CsvTabularPreprocessor:
    """
    Preprocessor for the colorectal_cancer_dataset.csv pipeline.

    Responsibilities
    ----------------
    1. Load the raw Kaggle CSV and drop irrelevant columns.
    2. Compute derived features (Risk_Score, Prevention_Index, etc.).
    3. Encode categorical columns to numeric using constants.py maps.
    4. `process_raw_dataset`  — batch processing for training.
    5. `build_feature_row`    — single-row inference for the Streamlit form.
       The form only collects 13 clinically accessible fields; the remaining
       3 inputs (Early_Detection, Incidence_Rate, Mortality_Rate) are
       auto-filled with sensible defaults so the model always receives a
       complete 24-feature vector.
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self._fitted: bool = False
        self.feature_names: list[str] = []

    # ── Derived-feature static helpers ─────────────────────────────────────

    @staticmethod
    def calc_risk_score(
        obesity, diet, activity, smoking, alcohol,
        diabetes, ibd, genetic, family_history, age,
    ) -> float:
        return _csv_calc_risk_score(
            obesity, diet, activity, smoking, alcohol,
            diabetes, ibd, genetic, family_history, age,
        )

    @staticmethod
    def calc_prevention_index(screening, early_detection, activity, diet) -> float:
        return _csv_calc_prevention_index(screening, early_detection, activity, diet)

    @staticmethod
    def calc_access_score(urban_or_rural: str) -> float:
        return _csv_calc_access_score(urban_or_rural)

    @staticmethod
    def calc_age_risk_group(age: int) -> str:
        return _csv_calc_age_risk_group(age)

    @staticmethod
    def calc_lifestyle_cluster(obesity, diet, activity, smoking, alcohol) -> str:
        return _csv_calc_lifestyle_cluster(obesity, diet, activity, smoking, alcohol)

    # ── Dataset processing (training) ──────────────────────────────────────

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Risk_Score"] = df.apply(
            lambda r: _csv_calc_risk_score(
                r["Obesity_BMI"], r["Diet_Risk"], r["Physical_Activity"],
                r["Smoking_History"], r["Alcohol_Consumption"],
                r["Diabetes"], r["Inflammatory_Bowel_Disease"],
                r["Genetic_Mutation"], r["Family_History"], r["Age"],
            ), axis=1,
        )
        df["Prevention_Index"] = df.apply(
            lambda r: _csv_calc_prevention_index(
                r["Screening_History"], r["Early_Detection"],
                r["Physical_Activity"], r["Diet_Risk"],
            ), axis=1,
        )
        df["Access_Score"]      = df["Urban_or_Rural"].apply(_csv_calc_access_score)
        df["Age_Risk_Group"]    = df["Age"].apply(_csv_calc_age_risk_group)
        df["Lifestyle_Cluster"] = df.apply(
            lambda r: _csv_calc_lifestyle_cluster(
                r["Obesity_BMI"], r["Diet_Risk"], r["Physical_Activity"],
                r["Smoking_History"], r["Alcohol_Consumption"],
            ), axis=1,
        )
        return df

    def _encode_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        binary_cols = [
            "Family_History", "Smoking_History", "Alcohol_Consumption",
            "Diabetes", "Inflammatory_Bowel_Disease", "Genetic_Mutation", "Early_Detection",
        ]
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map(CSV_BINARY_MAP)

        ordinal_maps = {
            "Gender":            CSV_GENDER_MAP,
            "Obesity_BMI":       CSV_OBESITY_MAP,
            "Diet_Risk":         CSV_DIET_MAP,
            "Physical_Activity": CSV_ACTIVITY_MAP,
            "Screening_History": CSV_SCREENING_MAP,
            "Urban_or_Rural":    CSV_URBAN_MAP,
            "Age_Risk_Group":    CSV_AGE_RISK_GROUP_MAP,
        }
        for col, mapping in ordinal_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

        df = pd.get_dummies(
            df, columns=["Lifestyle_Cluster"], prefix="LC",
            drop_first=False, dtype=int,
        )
        for opt in CSV_LIFESTYLE_CLUSTER_OPTIONS:
            col_name = f"LC_{opt}"
            if col_name not in df.columns:
                df[col_name] = 0

        return df

    def process_raw_dataset(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        diagnosis_label: int = 1,
    ) -> pd.DataFrame:
        """
        Load the raw Kaggle CSV, apply feature engineering and encoding.

        Parameters
        ----------
        input_path     : path to colorectal_cancer_dataset.csv
        output_path    : if provided, saves the processed DataFrame here
        diagnosis_label: 1 for cancer patients (default), 0 for healthy

        Returns
        -------
        pd.DataFrame with CSV_FINAL_COLUMNS column order
        """
        logger.info(f"[CsvTabularPreprocessor] Loading raw CSV from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Raw shape: {df.shape}")

        cols_drop = [c for c in CSV_COLUMNS_TO_DROP if c in df.columns]
        df.drop(columns=cols_drop, inplace=True)
        df["Diagnosis"] = diagnosis_label

        df = self._add_derived_features(df)
        df = self._encode_columns(df)

        existing = [c for c in CSV_FINAL_COLUMNS if c in df.columns]
        df = df[existing]

        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Processed CSV saved → {output_path}")

        logger.info(f"Processed shape: {df.shape}")
        return df

    # ── Scaler helpers ──────────────────────────────────────────────────────

    def fit_scaler(self, X: np.ndarray) -> "CsvTabularPreprocessor":
        self.scaler.fit(X)
        self._fitted = True
        return self

    def scale(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit_scaler first.")
        return self.scaler.transform(X)

    def save(self, path: Path) -> None:
        joblib.dump({"scaler": self.scaler, "feature_names": self.feature_names}, path)
        logger.info(f"CsvTabularPreprocessor saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "CsvTabularPreprocessor":
        state = joblib.load(path)
        p = cls()
        p.scaler = state["scaler"]
        p.feature_names = state.get("feature_names", [])
        p._fitted = True
        return p

    # ── Single-row inference (Streamlit form → model) ───────────────────────

    def build_feature_row(
        self, inputs: dict
    ) -> tuple[pd.DataFrame, float, float, float, str, str]:
        """
        Convert the 13-field clinical form dict into the full 24-feature
        numeric vector expected by the MLP model.

        Fields collected in the form (13)
        ----------------------------------
        age, gender, family_history, smoking, alcohol, obesity,
        diet_risk, physical_activity, diabetes, ibd, genetic,
        screening, urban_rural

        Fields auto-filled from defaults or derivation rules (3)
        ----------------------------------------------------------
        early_detection  → derived from screening  (Regular → Yes, else No)
        incidence_rate   → 20.0 per 100k  (GLOBOCAN 2022 global median)
        mortality_rate   →  8.0 per 100k  (GLOBOCAN 2022 global median)

        Returns
        -------
        (df_row, risk_score, prevention_index, access_score,
         age_risk_group, lifestyle_cluster)
        """
        # ── Required fields (from form) ─────────────────────────────────────
        age       = inputs["age"]
        gender    = inputs["gender"]
        fam_hist  = inputs["family_history"]
        smoking   = inputs["smoking"]
        alcohol   = inputs["alcohol"]
        obesity   = inputs["obesity"]
        diet      = inputs["diet_risk"]
        activity  = inputs["physical_activity"]
        diabetes  = inputs["diabetes"]
        ibd       = inputs["ibd"]
        genetic   = inputs["genetic"]
        screening = inputs["screening"]
        urban     = inputs["urban_rural"]

        # ── Auto-filled fields ───────────────────────────────────────────────
        early_det = _derive_early_detection(screening)
        incidence = inputs.get("incidence_rate", _DEFAULT_INCIDENCE_RATE)
        mortality = inputs.get("mortality_rate", _DEFAULT_MORTALITY_RATE)

        # ── Derived features ─────────────────────────────────────────────────
        risk_score        = _csv_calc_risk_score(
            obesity, diet, activity, smoking, alcohol,
            diabetes, ibd, genetic, fam_hist, age,
        )
        prevention_index  = _csv_calc_prevention_index(screening, early_det, activity, diet)
        access_score      = _csv_calc_access_score(urban)
        age_risk_group    = _csv_calc_age_risk_group(age)
        lifestyle_cluster = _csv_calc_lifestyle_cluster(obesity, diet, activity, smoking, alcohol)

        # ── Numeric feature row ──────────────────────────────────────────────
        row = {
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
            "Incidence_Rate_per_100K":    incidence,
            "Mortality_Rate_per_100K":    mortality,
            "Urban_or_Rural":             CSV_URBAN_MAP[urban],
            "Risk_Score":                 risk_score,
            "Prevention_Index":           prevention_index,
            "Access_Score":               access_score,
            "Age_Risk_Group":             CSV_AGE_RISK_GROUP_MAP[age_risk_group],
            **{
                f"LC_{opt}": int(lifestyle_cluster == opt)
                for opt in CSV_LIFESTYLE_CLUSTER_OPTIONS
            },
        }
        df_row = pd.DataFrame([row])
        return df_row, risk_score, prevention_index, access_score, age_risk_group, lifestyle_cluster