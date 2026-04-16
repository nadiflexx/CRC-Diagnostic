"""
XGBoost model for colon tabular analysis focused on country, smoking and alcohol.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config.constants import (
    TABULAR_ANALYSIS_CATEGORICAL_FEATURES,
    TABULAR_ANALYSIS_NUMERIC_FEATURES,
    TABULAR_ANALYSIS_RANDOM_SEED,
    TABULAR_ANALYSIS_XGB_CONFIG,
)
from src.config.paths import paths


class XGBoostColonAnalysisModel:
    """Binary XGBoost model that predicts advanced colorectal cancer stage."""

    def __init__(self, name: str = "xgboost_colon_analysis"):
        self.name = name
        self.pipeline = None
        self.feature_names = None
        self.metrics = {}

    def build_pipeline(self):
        """Creates preprocessing plus model pipeline."""
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), TABULAR_ANALYSIS_NUMERIC_FEATURES),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    TABULAR_ANALYSIS_CATEGORICAL_FEATURES,
                ),
            ]
        )

        model = XGBClassifier(
            random_state=TABULAR_ANALYSIS_RANDOM_SEED,
            **TABULAR_ANALYSIS_XGB_CONFIG,
        )

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        return self.pipeline

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fits the XGBoost model."""
        if self.pipeline is None:
            self.build_pipeline()

        self.pipeline.fit(X_train, y_train)
        self.feature_names = self.pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts the binary class."""
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities."""
        return self.pipeline.predict_proba(X)

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Returns the probability of advanced cancer."""
        return self.predict_proba(X)[:, 1]

    def predict_profile(self, profile: dict) -> float:
        """Predicts risk for a single user-defined profile."""
        input_df = pd.DataFrame([profile])
        return float(self.predict_risk(input_df)[0])

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluates the model on the test set."""
        y_pred = self.predict(X_test)
        y_prob = self.predict_risk(X_test)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "confusion_matrix": cm.tolist(),
        }
        return self.metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """Returns feature importance aligned with transformed features."""
        if self.pipeline is None or self.feature_names is None:
            return pd.DataFrame()

        importances = self.pipeline.named_steps["model"].feature_importances_
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importances,
            }
        )
        return importance_df.sort_values("importance", ascending=False)

    def save(self, filepath: Path = None):
        """Saves the trained artifact."""
        if filepath is None:
            filepath = paths.ANALYSIS_TABULAR_XGB_MODEL_PATH

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "pipeline": self.pipeline,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "model_name": self.name,
        }
        with open(filepath, "wb") as file:
            pickle.dump(artifact, file)
        return filepath
