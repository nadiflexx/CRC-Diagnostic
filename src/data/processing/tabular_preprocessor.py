"""
Tabular clinical data preprocessing.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config.constants import CLINICAL_NUMERIC_FEATURES, TABULAR_TARGET
from src.config.logger import log as logger


class TabularPreprocessor:
    """
    Clinical tabular data preprocessor for the CRC synthetic dataset.

    Handles scaling of the numeric clinical + radiomic feature set
    produced by ``ClinicalDataGenerator``, and serialisation /
    deserialisation of the fitted scaler state. Supports both
    ``fit_transform`` (training) and ``transform`` (inference) workflows.

    All features are purely numeric — no categorical encoding or KNN
    imputation is required because the synthetic dataset contains no
    missing values and no categorical columns.
    """

    NUMERIC_FEATURES = CLINICAL_NUMERIC_FEATURES
    TARGET = TABULAR_TARGET

    def __init__(self):
        """
        Initialise the preprocessor with an unfitted StandardScaler.

        State (``scaler``, ``feature_names``) is populated during
        ``fit_transform`` and can be persisted via ``save``.
        """
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self._fitted = False

    # ──────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
        """
        Fit the scaler on ``df`` and return the transformed feature matrix.

        Processing steps:
            1. Select the numeric feature columns present in ``df``
               from ``CLINICAL_NUMERIC_FEATURES``.
            2. Extract the target column ``Diagnosis`` when present.
            3. Fit and apply ``StandardScaler`` to the feature matrix.

        After this call ``self._fitted`` is ``True`` and ``transform``
        can be used for new data.

        Args:
            df (pd.DataFrame): Enriched DataFrame produced by
                ``ClinicalDataGenerator.generate``, containing both
                feature columns and the ``Diagnosis`` target.

        Returns:
            tuple[np.ndarray, np.ndarray | None, list[str]]:
                - ``X``: Scaled feature matrix of shape
                  (n_samples, n_features).
                - ``y``: Integer target array of shape (n_samples,),
                  or ``None`` if ``Diagnosis`` is not in ``df``.
                - ``feature_names``: Ordered list of feature names
                  corresponding to columns of ``X``.
        """
        y = (
            np.asarray(df[self.TARGET].astype(int).values)
            if self.TARGET in df.columns
            else None
        )

        num_cols = [c for c in self.NUMERIC_FEATURES if c in df.columns]
        X_raw = df[num_cols].fillna(0.0).values.astype(float)
        X_scaled = self.scaler.fit_transform(X_raw)

        self.feature_names = num_cols
        self._fitted = True
        logger.info(f"Features: {len(self.feature_names)}, Samples: {len(X_scaled)}")
        return X_scaled, y, self.feature_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply the fitted scaler to new data without refitting.

        Missing feature values are filled with 0 before scaling.

        Args:
            df (pd.DataFrame): Input DataFrame with the same column
                structure used during ``fit_transform``.

        Returns:
            np.ndarray: Scaled feature matrix of shape
                (n_samples, n_features).

        Raises:
            RuntimeError: If ``fit_transform`` has not been called
                before this method.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        X_raw = df[self.feature_names].fillna(0.0).values.astype(float)
        return self.scaler.transform(X_raw)

    def save(self, path: Path) -> None:
        """
        Serialise the fitted preprocessor state to disk using joblib.

        Args:
            path (Path): Destination file path for the serialised state.
        """
        joblib.dump(
            {
                "scaler": self.scaler,
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
            TabularPreprocessor: Fully restored instance with
                ``_fitted`` set to ``True``, ready to call
                ``transform`` on new data.
        """
        state = joblib.load(path)
        p = cls()
        p.scaler = state["scaler"]
        p.feature_names = state["feature_names"]
        p._fitted = True
        return p
