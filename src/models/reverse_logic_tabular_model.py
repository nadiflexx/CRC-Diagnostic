"""
Reverse Logic Tabular Model: Predicts Smoking_History and Alcohol_Consumption
from clinical indicators (Age, Country, Gender, Cancer_Stage, Diet_Risk, etc.)

Supports: XGBoost, Random Forest
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


class FeatureWeightedXGBClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost classifier with per-feature weights applied after preprocessing."""
    
    _estimator_type = "classifier"
    
    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        random_seed: int = 42,
        model_config: Optional[Dict[str, Any]] = None,
        feature_weight_overrides: Optional[Dict[str, float]] = None,
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.random_seed = random_seed
        self.model_config = model_config
        self.feature_weight_overrides = feature_weight_overrides

    def _build_preprocessor(self) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features or []),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_features or [],
                ),
            ],
            remainder="drop",
        )

    def _build_feature_weights(self, feature_names: List[str]) -> np.ndarray:
        weights = np.ones(len(feature_names), dtype=float)
        overrides = self.feature_weight_overrides or {}
        
        for idx, feature_name in enumerate(feature_names):
            for raw_feature, weight in overrides.items():
                if (
                    feature_name == f"num__{raw_feature}"
                    or feature_name == f"cat__{raw_feature}"
                    or feature_name.startswith(f"cat__{raw_feature}_")
                ):
                    weights[idx] = float(weight)
                    break
        return weights

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.preprocessor_ = self._build_preprocessor()
        X_transformed = self.preprocessor_.fit_transform(X)
        self.feature_names_out_ = self.preprocessor_.get_feature_names_out().tolist()
        
        feature_weights = self._build_feature_weights(self.feature_names_out_)
        config = self.model_config or {}
        
        self.model_ = XGBClassifier(
            random_state=self.random_seed,
            **config,
        )
        
        self.model_.set_params(feature_weights=feature_weights)
        self.model_.fit(X_transformed, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_transformed = self.preprocessor_.transform(X)
        return self.model_.predict(X_transformed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_transformed = self.preprocessor_.transform(X)
        return self.model_.predict_proba(X_transformed)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model_.feature_importances_

    def __sklearn_tags__(self):
        try:
            tags = super().__sklearn_tags__()
            tags.estimator_type = "classifier"
            return tags
        except AttributeError:
            return {"estimator_type": "classifier"}


class ReverseLogicTabularModel:
    """
    Multi-model framework for predicting smoking/alcohol habits from clinical indicators.
    Supports model switching, batch prediction, and profile-based screening.
    """

    def __init__(
        self,
        target: str = "Smoking_History",
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        random_seed: int = 42,
    ):
        """
        Initialize the reverse logic model.
        
        Args:
            target: "Smoking_History" or "Alcohol_Consumption"
            numeric_features: List of numeric column names
            categorical_features: List of categorical column names
            random_seed: Random seed for reproducibility
        """
        self.target = target
        self.numeric_features = numeric_features or ["Age"]
        self.categorical_features = categorical_features or [
            "Country", "Gender", "Physical_Activity", "Cancer_Stage", "Diet_Risk", "Obesity_BMI"
        ]
        self.random_seed = random_seed

        self.pipelines: Dict[str, Pipeline] = {}
        self.feature_names: Dict[str, List[str]] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.cv_metrics: Dict[str, Dict[str, float]] = {}
        self.encoders: Dict[str, Any] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def _encode_target(
        self,
        y: pd.Series,
        model_type: str,
        fit_encoder: bool = False,
    ) -> pd.Series:
        """
        Encodes target values into a stable binary representation when needed.
        """
        if isinstance(y.iloc[0], str):
            if fit_encoder or model_type not in self.label_encoders:
                le = LabelEncoder()
                encoded = pd.Series(le.fit_transform(y), index=y.index)
                self.label_encoders[model_type] = le
                return encoded
            return pd.Series(self.label_encoders[model_type].transform(y), index=y.index)

        unique_values = sorted(pd.Series(y).dropna().unique().tolist())
        if unique_values == [0, 1]:
            return pd.Series(y, index=y.index)

        if fit_encoder or model_type not in self.label_encoders:
            le = LabelEncoder()
            encoded = pd.Series(le.fit_transform(y), index=y.index)
            self.label_encoders[model_type] = le
            return encoded

        return pd.Series(self.label_encoders[model_type].transform(y), index=y.index)

    # ─────────────────────────────────────────────────────────────────────────────
    # PIPELINE BUILDERS
    # ─────────────────────────────────────────────────────────────────────────────

    def build_xgboost_pipeline(self, config: Dict[str, Any]) -> FeatureWeightedXGBClassifier:
        """Builds XGBoost pipeline with preprocessing."""
        config_copy = config.copy()
        feature_weight_overrides = config_copy.pop("feature_weight_overrides", {})
        
        return FeatureWeightedXGBClassifier(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            random_seed=self.random_seed,
            model_config=config_copy,
            feature_weight_overrides=feature_weight_overrides,
        )

    def build_random_forest_pipeline(self, config: Dict[str, Any]) -> Pipeline:
        """Builds Random Forest pipeline."""
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_features,
                ),
            ],
            remainder="drop",
        )

        model = RandomForestClassifier(**config)

        return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # ─────────────────────────────────────────────────────────────────────────────
    # TRAINING METHODS
    # ─────────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = "xgboost",
        config: Optional[Dict[str, Any]] = None,
    ) -> "ReverseLogicTabularModel":
        """
        Trains a single model.
        
        Args:
            X_train: Training features
            y_train: Training target (binary: Yes/No)
            model_type: "xgboost" or "random_forest"
            config: Model hyperparameters
            
        Returns:
            self for chaining
        """
        # Encode target if needed
        y_train = self._encode_target(y_train, model_type, fit_encoder=True)

        # Build pipeline
        if model_type == "xgboost":
            pipeline = self.build_xgboost_pipeline((config or {}).copy())
        elif model_type == "random_forest":
            pipeline = self.build_random_forest_pipeline(config or {})
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Supported: xgboost, random_forest")

        # Train
        pipeline.fit(X_train, y_train)
        self.pipelines[model_type] = pipeline

        # Store feature names
        try:
            if hasattr(pipeline, "feature_names_out_"):
                self.feature_names[model_type] = pipeline.feature_names_out_
            else:
                self.feature_names[model_type] = (
                    pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
                )
        except (AttributeError, KeyError):
            self.feature_names[model_type] = []

        return self

    def cross_validate_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = "xgboost",
        config: Optional[Dict[str, Any]] = None,
        cv_folds: int = 5,
    ) -> Dict[str, float]:
        """
        Runs stratified cross-validation and stores mean/std metrics for a model.
        """
        y_encoded = self._encode_target(y_train, model_type, fit_encoder=True)

        if model_type == "xgboost":
            pipeline = self.build_xgboost_pipeline((config or {}).copy())
        elif model_type == "random_forest":
            pipeline = self.build_random_forest_pipeline(config or {})
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Supported: xgboost, random_forest")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_seed)
        scoring = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, zero_division=0),
            "recall": make_scorer(recall_score, zero_division=0),
            "f1": make_scorer(f1_score, zero_division=0),
            "roc_auc": "roc_auc",
        }
        scores = cross_validate(
            pipeline,
            X_train,
            y_encoded,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
            error_score=np.nan,
        )

        summary = {"cv_folds": cv_folds}
        for metric in scoring:
            metric_values = scores[f"test_{metric}"]
            summary[f"{metric}_mean"] = float(np.mean(metric_values))
            summary[f"{metric}_std"] = float(np.std(metric_values))

        self.cv_metrics[model_type] = summary
        return summary

    def fit_multiple(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_configs: Dict[str, Dict[str, Any]],
    ) -> "ReverseLogicTabularModel":
        """
        Trains multiple models in sequence.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_configs: Dict of {model_type: config}
        """
        for model_type, config in model_configs.items():
            self.fit(X_train, y_train, model_type=model_type, config=config)
        return self

    # ─────────────────────────────────────────────────────────────────────────────
    # PREDICTION METHODS
    # ─────────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        X: pd.DataFrame,
        model_type: str = "xgboost",
    ) -> np.ndarray:
        """Returns binary predictions (0 or 1)."""
        if model_type not in self.pipelines:
            raise ValueError(f"Model {model_type} not trained. Available: {list(self.pipelines.keys())}")
        return self.pipelines[model_type].predict(X)

    def predict_proba(
        self,
        X: pd.DataFrame,
        model_type: str = "xgboost",
    ) -> np.ndarray:
        """Returns probability predictions [P(No), P(Yes)]."""
        if model_type not in self.pipelines:
            raise ValueError(f"Model {model_type} not trained.")
        return self.pipelines[model_type].predict_proba(X)

    def predict_risk(
        self,
        X: pd.DataFrame,
        model_type: str = "xgboost",
    ) -> np.ndarray:
        """Returns risk probability P(habit=Yes)."""
        return self.predict_proba(X, model_type)[:, 1]

    def predict_profile(
        self,
        profile: Dict[str, any],
        model_type: str = "xgboost",
    ) -> float:
        """
        Predicts habit risk for a single profile.
        
        Args:
            profile: Dict with {feature_name: value}
            model_type: Model to use for prediction
            
        Returns:
            Probability of habit presence (0-1)
        """
        profile_df = pd.DataFrame([profile])
        return float(self.predict_risk(profile_df, model_type)[0])

    def predict_ensemble(
        self,
        X: pd.DataFrame,
        model_types: Optional[List[str]] = None,
        method: str = "mean",
    ) -> np.ndarray:
        """
        Ensemble prediction combining multiple models.
        
        Args:
            X: Features
            model_types: List of models to ensemble (default: all trained)
            method: "mean", "median", or "max"
            
        Returns:
            Ensemble probability predictions
        """
        if model_types is None:
            model_types = list(self.pipelines.keys())

        probabilities = [self.predict_risk(X, mt) for mt in model_types]

        if method == "mean":
            return np.mean(probabilities, axis=0)
        elif method == "median":
            return np.median(probabilities, axis=0)
        elif method == "max":
            return np.max(probabilities, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

    # ─────────────────────────────────────────────────────────────────────────────
    # EVALUATION METHODS
    # ─────────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_type: str = "xgboost",
    ) -> Dict[str, float]:
        """
        Evaluates model performance.
        
        Returns:
            Dict with accuracy, precision, recall, f1, roc_auc, sensitivity, specificity
        """
        # Encode target if needed
        y_test_encoded = self._encode_target(y_test, model_type, fit_encoder=False).values

        y_pred = self.predict(X_test, model_type)
        y_prob = self.predict_risk(X_test, model_type)

        cm = confusion_matrix(y_test_encoded, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        unique_classes = np.unique(y_test_encoded)
        roc_auc = (
            roc_auc_score(y_test_encoded, y_prob)
            if len(unique_classes) > 1
            else float("nan")
        )

        metrics = {
            "accuracy": accuracy_score(y_test_encoded, y_pred),
            "precision": precision_score(y_test_encoded, y_pred, zero_division=0),
            "recall": recall_score(y_test_encoded, y_pred, zero_division=0),
            "f1": f1_score(y_test_encoded, y_pred, zero_division=0),
            "roc_auc": roc_auc,
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "confusion_matrix": cm.tolist(),
        }

        self.metrics[model_type] = metrics
        return metrics

    def evaluate_multiple(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_types: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluates all trained models."""
        if model_types is None:
            model_types = list(self.pipelines.keys())

        results = {}
        for model_type in model_types:
            results[model_type] = self.evaluate(X_test, y_test, model_type)

        return results

    # ─────────────────────────────────────────────────────────────────────────────
    # FEATURE IMPORTANCE
    # ─────────────────────────────────────────────────────────────────────────────

    def get_feature_importance(
        self,
        model_type: str = "xgboost",
    ) -> pd.DataFrame:
        """Returns feature importance for tree-based models."""
        if model_type not in self.pipelines:
            return pd.DataFrame()

        pipeline = self.pipelines[model_type]
        if hasattr(pipeline, "named_steps"):
            model = pipeline.named_steps["model"]
        else:
            model = pipeline

        if not hasattr(model, "feature_importances_"):
            return pd.DataFrame()

        importances = model.feature_importances_
        feature_names = self.feature_names.get(model_type, [])

        if not feature_names:
            return pd.DataFrame()

        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        )
        return importance_df.sort_values("importance", ascending=False)

    def get_coef(self, model_type: str = "random_forest") -> pd.DataFrame:
        """Returns coefficients for linear models. Not applicable for XGBoost/Random Forest."""
        return pd.DataFrame()

    # ─────────────────────────────────────────────────────────────────────────────
    # SERIALIZATION
    # ─────────────────────────────────────────────────────────────────────────────

    def save(self, filepath: Path) -> Path:
        """Saves all trained models and metadata."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "target": self.target,
            "pipelines": self.pipelines,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "cv_metrics": self.cv_metrics,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "label_encoders": self.label_encoders,
        }

        with open(filepath, "wb") as f:
            pickle.dump(artifact, f)

        return filepath

    @staticmethod
    def load(filepath: Path) -> "ReverseLogicTabularModel":
        """Loads trained models from disk."""
        with open(filepath, "rb") as f:
            artifact = pickle.load(f)

        model = ReverseLogicTabularModel(
            target=artifact["target"],
            numeric_features=artifact["numeric_features"],
            categorical_features=artifact["categorical_features"],
        )

        model.pipelines = artifact["pipelines"]
        model.feature_names = artifact["feature_names"]
        model.metrics = artifact["metrics"]
        model.cv_metrics = artifact.get("cv_metrics", {})
        model.label_encoders = artifact["label_encoders"]

        return model
