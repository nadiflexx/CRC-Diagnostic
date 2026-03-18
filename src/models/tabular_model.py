"""
Tabular model for colon cancer risk classification.
"""

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb

from src.config.logger import log as logger


class TabularCancerModel:
    """Tabular classifier optimized for recall."""

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model: Any = None
        self.feature_names: list[str] | None = None
        self.best_threshold = 0.3

    def _create_model(self, class_weights: dict[Any, Any] | None = None):
        """
        Create the tabular model based on the specified type.

        :param class_weights: Dictionary of class weights for handling imbalanced data
        """
        scale_pos = (
            class_weights.get(1, 1.0) / class_weights.get(0, 1.0)
            if class_weights
            else 1.0
        )
        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.01,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight=5,
                scale_pos_weight=scale_pos,
                reg_alpha=1.0,
                reg_lambda=2.0,
                random_state=42,
                eval_metric="aucpr",
                early_stopping_rounds=30,
            )
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                scale_pos_weight=scale_pos,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
            )

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        feature_names=None,
        class_weights=None,
    ):
        """
        Train the tabular model.

        :param X_train: Training features
        :param y_train: Training labels
        :param X_val: Validation features
        :param y_val: Validation labels
        :param feature_names: List of feature names
        :param class_weights: Dictionary of class weights for handling imbalanced data
        """
        self.feature_names = feature_names
        self._create_model(class_weights)
        logger.info(f"Training {self.model_type} with {X_train.shape[0]} samples...")

        if self.model_type == "xgboost" and X_val is not None:
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
        elif self.model_type == "lightgbm" and X_val is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
            )
        else:
            self.model.fit(X_train, y_train)

        if X_val is not None:
            self._optimize_threshold(X_val, y_val)
        logger.info("✅ Tabular model trained")

    def _optimize_threshold(self, X_val, y_val):
        """
        Optimize the classification threshold using Youden's J statistic.

        :param X_val: Validation features
        :param y_val: Validation labels
        """
        probs = self.model.predict_proba(X_val)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val, probs)
        j_scores = tpr * 1.3 - fpr
        best_idx = np.argmax(j_scores)
        self.best_threshold = float(np.clip(thresholds[best_idx], 0.20, 0.60))

        preds = (probs >= self.best_threshold).astype(int)
        rec = recall_score(y_val, preds, zero_division=0)
        prec = precision_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        logger.info(
            f"Threshold (Youden's J): {self.best_threshold:.3f} | "
            f"Recall={rec:.3f} | Precision={prec:.3f} | F1={f1:.3f}"
        )

    def predict_proba(self, X):
        """
        Predict the probability of the positive class for the given input.
        :param X: Input features
        :return: Predicted probabilities
        """
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        """
        Predict the class labels for the given input.

        :param X: Input features
        :return: Predicted class labels
        """
        probs = self.predict_proba(X)
        return (probs >= self.best_threshold).astype(int)

    def evaluate(self, X_test, y_test) -> dict:
        """
        Evaluate the model on the given test data.
        :param X_test: Test features
        :param y_test: Test labels
        :return: Dictionary containing evaluation metrics
        """
        probs = self.predict_proba(X_test)
        preds = self.predict(X_test)
        results = {
            "auc_roc": roc_auc_score(y_test, probs),
            "recall": recall_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            "classification_report": classification_report(y_test, preds),
            "threshold": self.best_threshold,
        }
        logger.info(f"\n{results['classification_report']}")
        logger.info(f"AUC-ROC: {results['auc_roc']:.4f}")
        return results

    def cross_validate(self, X, y, cv=5) -> dict:
        """
        Perform cross-validation on the given data.

        :param X: Input features
        :param y: Target labels
        :param cv: Number of cross-validation folds
        :return: Dictionary containing cross-validation results
        """
        original = None
        if hasattr(self.model, "early_stopping_rounds"):
            original = self.model.early_stopping_rounds
            self.model.set_params(early_stopping_rounds=None)

        cv_scores = cross_val_score(
            self.model,
            X,
            y,
            cv=StratifiedKFold(cv, shuffle=True, random_state=42),
            scoring="recall",
        )

        if original is not None:
            self.model.set_params(early_stopping_rounds=original)

        logger.info(f"CV Recall: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return {
            "mean_recall": cv_scores.mean(),
            "std_recall": cv_scores.std(),
            "scores": cv_scores.tolist(),
        }

    def save(self, path: Path):
        """
        Save the trained model to the specified path.

        :param path: Path to save the model
        """
        joblib.dump(
            {
                "model": self.model,
                "model_type": self.model_type,
                "feature_names": self.feature_names,
                "best_threshold": self.best_threshold,
            },
            path,
        )
        logger.info(f"Tabular model saved: {path}")

    @classmethod
    def load(cls, path: Path) -> "TabularCancerModel":
        """
        Load a trained model from the specified path.

        :param path: Path to load the model from
        :return: Loaded model instance
        """
        state = joblib.load(path)
        instance = cls(model_type=state["model_type"])
        instance.model = state["model"]
        instance.feature_names = state["feature_names"]
        instance.best_threshold = state["best_threshold"]
        return instance
