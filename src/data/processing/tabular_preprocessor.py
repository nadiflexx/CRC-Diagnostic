"""
Tabular clinical data preprocessing.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config.constants import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
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


class TabularPreprocessor:
    """
    Clinical tabular data preprocessor.

    Handles cleaning of GDC and Kaggle source formats, feature
    engineering, fitting of scalers and label encoders, and
    serialisation/deserialisation of the fitted state. Supports both
    ``fit_transform`` (training) and ``transform`` (inference) workflows.
    """

    NUMERIC_FEATURES = NUMERIC_FEATURES
    CATEGORICAL_FEATURES = CATEGORICAL_FEATURES
    BINARY_FEATURES = BINARY_FEATURES
    TARGET = TABULAR_TARGET

    def __init__(self):
        """
        Initialise the preprocessor with unfitted scalers and encoders.

        All state (``scaler``, ``label_encoders``, ``feature_names``) is
        populated during ``fit_transform`` and can be persisted via
        ``save``.
        """
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self._fitted = False

    def clean_gdc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonise a raw GDC clinical DataFrame to the canonical schema.

        Maps GDC-specific column names and value vocabularies to the
        project's unified feature set. Missing columns are filled with
        sensible defaults (``np.nan`` for numeric, ``"unknown"`` for
        categorical). The ``has_cancer`` flag is set to ``True`` for all
        GDC rows (all records are cancer cases).

        Args:
            df (pd.DataFrame): Raw GDC clinical data. Expected columns
                include (but are not limited to): ``age_at_index``,
                ``gender``, ``bmi``, ``race``,
                ``tobacco_smoking_status``, ``alcohol_history``,
                ``pack_years_smoked``, and ``ajcc_stage``.

        Returns:
            pd.DataFrame: Cleaned DataFrame with the canonical columns:
                ``age``, ``gender``, ``bmi``, ``ethnicity``,
                ``smoking_status``, ``alcohol_consumption``,
                ``pack_years_smoked``, ``ajcc_stage``, and
                ``has_cancer``.
        """
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
        """
        Harmonise a raw Kaggle risk-factor DataFrame to the canonical schema.

        Maps Kaggle column names and categorical value strings to the
        unified vocabulary using the constants defined in
        ``src.config.constants``. Optionally extracts staging,
        aggressiveness, and survival information when present.

        Args:
            df (pd.DataFrame): Raw Kaggle risk-factor data. Expected
                columns include: ``Age``, ``Gender``, ``Race``, ``BMI``,
                ``Smoking_Status``, ``Alcohol_Consumption``,
                ``Physical_Activity_Level``, ``Diet_Type``,
                ``Family_History``, ``Previous_Cancer_History``, and
                optionally ``Stage_at_Diagnosis``,
                ``Tumor_Aggressiveness``, and ``Survival_Status``.

        Returns:
            pd.DataFrame: Cleaned DataFrame with the canonical columns
                plus optional columns ``cancer_stage``,
                ``cancer_stage_numeric``, ``tumor_aggressiveness``, and
                ``survived`` when the source columns are present.
        """
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
            cleaned["cancer_stage_numeric"] = df["Stage_at_Diagnosis"].map(
                KAGGLE_STAGE_MAP
            )
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
        """
        Concatenate multiple cleaned DataFrames into a single unified frame.

        Uses ``pd.concat`` with ``ignore_index=True`` and ``sort=False``
        to preserve column order. The ``has_cancer`` column is coerced to
        boolean after merging, with ``NaN`` treated as ``False``.

        Args:
            *dataframes (pd.DataFrame): One or more DataFrames to merge.
                All should share the canonical column schema produced by
                the cleaning methods.

        Returns:
            pd.DataFrame: Single concatenated DataFrame with a reset
                integer index.
        """
        logger.info(f"Merging {len(dataframes)} datasets")
        combined = pd.concat(dataframes, ignore_index=True, sort=False)
        if "has_cancer" in combined.columns:
            combined["has_cancer"] = combined["has_cancer"].fillna(False).astype(bool)
        logger.info(f"Datasets merged: {len(combined)} rows")
        return combined

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive compound clinical features from existing columns.

        Creates the following derived features when the required source
        columns are present:
            - ``hb_hct_ratio``: haemoglobin / haematocrit.
            - ``cea_albumin_ratio``: CEA / albumin.
            - ``inflammation_index``: log1p(CRP × WBC count).
            - ``iron_store_ratio``: ferritin / serum iron.
            - ``age_bmi_interaction``: (age / 100) × (BMI / 40).
            - ``{col}_log``: log1p transformation for ``cea``,
              ``ca19_9``, ``crp``, and ``ferritin``.

        Args:
            df (pd.DataFrame): DataFrame with canonical clinical columns.

        Returns:
            pd.DataFrame: Copy of ``df`` with additional derived feature
                columns appended. Original columns are unchanged.
        """
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
        """
        Fit all transformers on ``df`` and return the transformed feature matrix.

        Processing steps:
            1. Engineer derived features.
            2. Separate the target column ``has_cancer`` if present.
            3. Identify numeric, categorical, and binary feature columns.
            4. Impute numeric features with KNN (k=5) then scale with
               ``StandardScaler``.
            5. Encode categorical features with per-column
               ``LabelEncoder`` instances stored in
               ``self.label_encoders``.
            6. Cast binary features to int.
            7. Horizontally stack all feature groups.

        After this call ``self._fitted`` is ``True`` and ``transform``
        can be used for new data.

        Args:
            df (pd.DataFrame): Input DataFrame containing both features
                and the target column.

        Returns:
            tuple[np.ndarray, np.ndarray | None, list[str]]:
                - ``X``: Feature matrix of shape (n_samples, n_features).
                - ``y``: Integer target array of shape (n_samples,), or
                  ``None`` if ``has_cancer`` is not in ``df``.
                - ``feature_names``: Ordered list of feature names
                  corresponding to columns of ``X``.
        """
        df = self.engineer_features(df)
        y = (
            np.asarray(df[self.TARGET].astype(int).values)
            if self.TARGET in df.columns
            else None
        )

        num_cols = [c for c in self.NUMERIC_FEATURES if c in df.columns]
        eng_cols = [
            c
            for c in df.columns
            if c.endswith("_risk")
            or c.endswith("_elevated")
            or c
            in ["anemia", "low_albumin", "family_risk_score", "composite_risk_score"]
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
            bin_data = (
                df[bin_cols].infer_objects(copy=False).fillna(0).astype(int).values
            )
        else:
            bin_data = np.empty((len(df), 0))

        X = np.hstack([num_scaled, cat_encoded, bin_data])
        self.feature_names = num_cols + cat_cols + bin_cols
        self._fitted = True
        logger.info(f"Features: {len(self.feature_names)}, Samples: {len(X)}")
        return X, y, self.feature_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply the fitted transformers to new data without refitting.

        Uses the ``StandardScaler`` and ``LabelEncoder`` instances fitted
        during ``fit_transform``. Unseen categorical values are mapped to
        ``-1`` to avoid errors. Missing numeric values are filled with 0
        before scaling.

        Args:
            df (pd.DataFrame): Input DataFrame with the same column
                structure used during ``fit_transform``.

        Returns:
            np.ndarray: Transformed feature matrix of shape
                (n_samples, n_features) matching the column order of
                ``self.feature_names``.

        Raises:
            RuntimeError: If ``fit_transform`` has not been called before
                this method.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")

        df = self.engineer_features(df)
        num_cols = [
            c
            for c in self.feature_names
            if c not in self.CATEGORICAL_FEATURES + self.BINARY_FEATURES
        ]
        cat_cols = [c for c in self.feature_names if c in self.CATEGORICAL_FEATURES]
        bin_cols = [c for c in self.feature_names if c in self.BINARY_FEATURES]

        num_scaled = (
            self.scaler.transform(df[num_cols].fillna(0).values)
            if num_cols
            else np.empty((len(df), 0))
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
            if bin_cols
            else np.empty((len(df), 0))
        )
        return np.hstack([num_scaled, cat_encoded, bin_data])

    def save(self, path: Path):
        """
        Serialise the fitted preprocessor state to disk using joblib.

        Saves the ``StandardScaler``, per-column ``LabelEncoder``
        instances, and the ordered feature name list so the preprocessor
        can be fully restored via ``load`` without refitting.

        Args:
            path (Path): Destination file path for the serialised state.
        """
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
        """
        Deserialise a previously saved ``TabularPreprocessor``.

        Args:
            path (Path): Path to the joblib file produced by ``save``.

        Returns:
            TabularPreprocessor: Fully restored preprocessor instance
                with ``_fitted`` set to ``True``, ready to call
                ``transform`` on new data.
        """
        state = joblib.load(path)
        p = cls()
        p.scaler = state["scaler"]
        p.label_encoders = state["label_encoders"]
        p.feature_names = state["feature_names"]
        p._fitted = True
        return p
