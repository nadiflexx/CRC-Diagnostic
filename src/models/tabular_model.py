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
    """
    Gradient-boosted tabular classifier optimised for high recall.

    Supports two backend implementations selectable at construction time:
        - ``"xgboost"``: XGBoost with AUCPR evaluation and early stopping.
        - ``"lightgbm"``: LightGBM with early stopping callbacks.

    After training, a per-class decision threshold is optimised on the
    validation set using Youden's J statistic to maximise sensitivity
    (recall) while controlling false positives.
    """

    def __init__(self, model_type: str = "xgboost"):
        """
        Initialise the tabular cancer model.

        Args:
            model_type (str): Backend to use. Must be either
                ``"xgboost"`` or ``"lightgbm"``. Default is
                ``"xgboost"``.
        """
        self.model_type = model_type
        self.model: Any = None
        self.feature_names: list[str] | None = None
        self.best_threshold = 0.3

    def _create_model(self, class_weights: dict[Any, Any] | None = None):
        """
        Instantiate the underlying gradient-boosted model.

        Computes the ``scale_pos_weight`` ratio from ``class_weights`` and
        applies it to the model constructor so that the minority class
        receives proportionally higher influence during training.

        Args:
            class_weights (dict[Any, Any] | None): Dictionary mapping class
                indices to their weight values. If ``None``, equal weights
                are assumed (``scale_pos_weight = 1.0``).
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
        Train the tabular model on the provided data.

        If a validation set is supplied the model uses early stopping
        (XGBoost) or early-stopping callbacks (LightGBM) to prevent
        overfitting. After fitting, the decision threshold is optimised
        on the validation set via ``_optimize_threshold``.

        Args:
            X_train (array-like of shape (n_samples, n_features)):
                Training feature matrix.
            y_train (array-like of shape (n_samples,)): Training labels.
            X_val (array-like of shape (n_val, n_features) | None):
                Validation feature matrix used for early stopping and
                threshold optimisation. Pass ``None`` to skip both.
            y_val (array-like of shape (n_val,) | None): Validation labels
                aligned with ``X_val``.
            feature_names (list[str] | None): Human-readable names for
                each column of ``X_train``. Stored and serialised with the
                model. Default is ``None``.
            class_weights (dict | None): Class weight mapping forwarded to
                ``_create_model`` for ``scale_pos_weight`` computation.
                Default is ``None``.
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
        Find the decision threshold that maximises Youden's J statistic.

        Computes the ROC curve on the validation set and selects the
        threshold that maximises ``1.3 * TPR - FPR`` (a recall-weighted
        variant of Youden's J). The result is clipped to [0.20, 0.60] to
        avoid degenerate thresholds. The chosen threshold is stored in
        ``self.best_threshold`` and logged alongside recall, precision,
        and F1.

        Args:
            X_val (array-like of shape (n_val, n_features)): Validation
                feature matrix.
            y_val (array-like of shape (n_val,)): Validation labels.
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
        Return the predicted probability of the positive class.

        Args:
            X (array-like of shape (n_samples, n_features)): Input feature
                matrix.

        Returns:
            np.ndarray of shape (n_samples,): Predicted probability of
                class 1 for each sample.
        """
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        """
        Return binary class predictions using the optimised threshold.

        Probabilities from ``predict_proba`` are thresholded at
        ``self.best_threshold`` to produce integer class labels.

        Args:
            X (array-like of shape (n_samples, n_features)): Input feature
                matrix.

        Returns:
            np.ndarray of shape (n_samples,): Predicted class labels
                (0 or 1).
        """
        probs = self.predict_proba(X)
        return (probs >= self.best_threshold).astype(int)

    def evaluate(self, X_test, y_test) -> dict:
        """
        Compute and log classification metrics on the test set.

        Args:
            X_test (array-like of shape (n_samples, n_features)): Test
                feature matrix.
            y_test (array-like of shape (n_samples,)): Ground-truth test
                labels.

        Returns:
            dict: Evaluation results containing the following keys:
                - ``"auc_roc"`` (float): ROC AUC score.
                - ``"recall"`` (float): Recall for the positive class.
                - ``"precision"`` (float): Precision for the positive class.
                - ``"f1"`` (float): F1 score for the positive class.
                - ``"confusion_matrix"`` (list[list[int]]): Confusion matrix
                  as a nested list.
                - ``"classification_report"`` (str): Full scikit-learn
                  classification report string.
                - ``"threshold"`` (float): Decision threshold used to
                  produce predictions.
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
        Evaluate generalisation performance via stratified k-fold
        cross-validation.

        Early stopping is temporarily disabled during cross-validation
        because no separate validation set is available inside each fold.
        The original ``early_stopping_rounds`` setting is restored
        afterwards.

        Args:
            X (array-like of shape (n_samples, n_features)): Full feature
                matrix (train + val + test combined is acceptable here as
                the CV loop handles splitting internally).
            y (array-like of shape (n_samples,)): Target labels aligned
                with ``X``.
            cv (int): Number of stratified folds. Default is 5.

        Returns:
            dict: Cross-validation results containing:
                - ``"mean_recall"`` (float): Mean recall across all folds.
                - ``"std_recall"`` (float): Standard deviation of recall
                  across all folds.
                - ``"scores"`` (list[float]): Per-fold recall scores.
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
        Serialise the trained model and metadata to disk using joblib.

        The saved file contains the model object, model type string,
        feature names, and the optimised decision threshold so that the
        model can be fully restored via ``load``.

        Args:
            path (Path): Destination file path for the serialised model.
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
        Deserialise and restore a previously saved ``TabularCancerModel``.

        Args:
            path (Path): Path to the joblib file produced by ``save``.

        Returns:
            TabularCancerModel: Fully restored model instance with the
                original model object, model type, feature names, and
                decision threshold.
        """
        state = joblib.load(path)
        instance = cls(model_type=state["model_type"])
        instance.model = state["model"]
        instance.feature_names = state["feature_names"]
        instance.best_threshold = state["best_threshold"]
        return instance
