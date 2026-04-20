"""
Tabular model for colon cancer risk classification.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb

from src.config.logger import log as logger
from src.models.calibration import TemperatureScaledModel, find_temperature

optuna.logging.set_verbosity(optuna.logging.WARNING)


class TabularCancerModel:
    """
    XGBoost tabular classifier optimised for high recall in CRC screening.

    Uses Optuna Bayesian hyperparameter search and Temperature Scaling
    calibration (Guo et al., ICML 2017) to produce well-calibrated
    probabilities while guaranteeing a configurable minimum recall
    (default 0.90), reflecting the clinical priority of minimising false
    negatives.

    The decision threshold is searched on calibrated probabilities so
    that the inference-time threshold operates on the same probability
    space as the clinical display.
    """

    def __init__(self, model_type: str = "xgboost"):
        """
        Initialise the tabular cancer model.

        Args:
            model_type (str): Backend to use. Currently only
                ``"xgboost"`` is supported. Default is ``"xgboost"``.
        """
        self.model_type = model_type
        # Public alias kept for SHAP / cross_validate compatibility.
        self.model: Any = None
        self.feature_names: list[str] | None = None
        self.best_threshold: float = 0.5
        # Initialised to None so load() and predict_proba() are safe
        # even if called before train().
        self._base_model: Any = None
        self._calibrator: TemperatureScaledModel | None = None

    # ──────────────────────────────────────────────────────────
    #  Hyperparameter search
    # ──────────────────────────────────────────────────────────

    def _optuna_objective(self, trial, X, y) -> float:
        """
        Optuna objective: 5-fold stratified CV ROC-AUC.

        Args:
            trial: Optuna trial object.
            X: Feature matrix.
            y: Target labels.

        Returns:
            Mean ROC-AUC across 5 folds.
        """
        scale_pos = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": scale_pos,
            "eval_metric": "auc",
            "use_label_encoder": False,
            "random_state": 42,
            "n_jobs": -1,
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for tr_idx, val_idx in cv.split(X, y):
            clf = xgb.XGBClassifier(**params)
            clf.fit(X[tr_idx], y[tr_idx])
            prob = clf.predict_proba(X[val_idx])[:, 1]
            aucs.append(roc_auc_score(y[val_idx], prob))
        return float(np.mean(aucs))

    def _search_hyperparameters(self, X, y, n_trials: int = 30) -> dict:
        """
        Run Bayesian hyperparameter search with Optuna.

        Args:
            X: Training feature matrix.
            y: Training labels.
            n_trials (int): Number of Optuna trials. Default 30.

        Returns:
            dict: Best hyperparameters found.
        """
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        study.optimize(
            lambda trial: self._optuna_objective(trial, X, y),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        logger.info(f"Best CV ROC-AUC: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        return study.best_params

    # ──────────────────────────────────────────────────────────
    #  Threshold calibration
    # ──────────────────────────────────────────────────────────

    def _find_optimal_threshold(
        self, X_val, y_val, recall_target: float = 0.90
    ) -> float:
        """
        Find the highest threshold that guarantees recall >= recall_target.

        Uses **calibrated** probabilities (via ``predict_proba``) so that
        the threshold operates on the same probability space seen at
        inference time. In oncological screening a false negative implies
        a potentially serious diagnostic delay, hence the priority on
        recall.

        Args:
            X_val: Validation feature matrix.
            y_val: Validation labels.
            recall_target (float): Minimum recall required. Default 0.90.

        Returns:
            float: Optimal decision threshold rounded to two decimals.
        """
        probs = self.predict_proba(X_val)

        best_threshold = 0.50
        best_f1 = 0.0

        for t in np.arange(0.05, 0.95, 0.01):
            preds = (probs >= t).astype(int)
            rec = recall_score(y_val, preds, zero_division=0)
            f1 = f1_score(y_val, preds, zero_division=0)

            if rec >= recall_target and f1 > best_f1:
                best_f1 = f1
                best_threshold = round(float(t), 2)

        return best_threshold

    # ──────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        feature_names=None,
        class_weights=None,
        n_trials: int = 30,
        recall_target: float = 0.90,
        t_minimum: float = 1.5,
    ):
        """
        Train the XGBoost model with Optuna search and Temperature Scaling
        calibration.

        Steps:
            1. Optuna Bayesian search for best hyperparameters.
            2. Fit final XGBoost model on ``X_train``.
            3. Fit Temperature Scaling calibrator on ``X_val`` (or
               ``X_train`` as fallback).
            4. Calibrate decision threshold on calibrated probabilities.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
            X_val: Optional validation feature matrix for calibration
                and threshold search.
            y_val: Optional validation labels.
            feature_names: Ordered list of feature names.
            class_weights: Unused — kept for API compatibility.
            n_trials (int): Number of Optuna trials. Default 30.
            recall_target (float): Minimum recall for threshold search.
                Default 0.90.
            t_minimum (float): Minimum temperature for Temperature
                Scaling. Default 1.5.
        """
        self.feature_names = feature_names
        logger.info(f"Training {self.model_type} — {len(X_train)} samples")

        logger.info("Running Optuna hyperparameter search...")
        best_params = self._search_hyperparameters(X_train, y_train, n_trials)

        scale_pos = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
        final_params = {
            **best_params,
            "scale_pos_weight": scale_pos,
            "eval_metric": "auc",
            "use_label_encoder": False,
            "random_state": 42,
            "n_jobs": -1,
        }

        # 1. Fit base XGBoost model
        base_model = xgb.XGBClassifier(**final_params)
        base_model.fit(X_train, y_train)
        logger.info("Final model fitted.")

        # 2. Choose calibration set — prefer held-out val to avoid leakage
        if X_val is not None and y_val is not None:
            cal_X, cal_y = X_val, y_val
        else:
            cal_X, cal_y = X_train, y_train

        # 3. Fit Temperature Scaling calibrator
        logger.info("Fitting Temperature Scaling calibrator...")
        temperature = find_temperature(base_model, cal_X, cal_y, t_minimum)
        self._calibrator = TemperatureScaledModel(base_model, temperature)
        self._base_model = base_model
        self.model = base_model  # kept for SHAP / cross_validate compatibility

        logger.info(
            f"Temperature Scaling fitted (T={temperature:.4f}, "
            f"T>1 softens distribution)."
        )

        # 4. Calibrate threshold using calibrated probabilities
        self.best_threshold = self._find_optimal_threshold(cal_X, cal_y, recall_target)
        logger.info(f"Best threshold: {self.best_threshold:.2f}")
        logger.info("✅ Tabular model trained")

    def predict_proba(self, X) -> np.ndarray:
        """
        Return calibrated probability of the positive class.

        Probabilities are clipped to [PROB_MIN, PROB_MAX] = [0.05, 0.95]
        inside ``TemperatureScaledModel`` to prevent clinically
        implausible certainty claims.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray of shape (n_samples,): Calibrated P(cancer).
        """
        if self._calibrator is not None:
            return self._calibrator.predict_proba(X)[:, 1]
        return self._base_model.predict_proba(X)[:, 1]

    def predict(self, X) -> np.ndarray:
        """
        Return binary predictions using the calibrated threshold.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray of shape (n_samples,): Predicted labels (0 or 1).
        """
        return (self.predict_proba(X) >= self.best_threshold).astype(int)

    def evaluate(self, X_test, y_test) -> dict:
        """
        Compute and log classification metrics on the test set.

        Args:
            X_test: Test feature matrix.
            y_test: Ground-truth test labels.

        Returns:
            dict: Keys ``auc_roc``, ``recall``, ``precision``, ``f1``,
                ``confusion_matrix``, ``classification_report``,
                ``threshold``.
        """
        probs = self.predict_proba(X_test)
        preds = self.predict(X_test)
        results = {
            "auc_roc": roc_auc_score(y_test, probs),
            "recall": recall_score(y_test, preds, zero_division=0),
            "precision": precision_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            "classification_report": classification_report(y_test, preds),
            "threshold": self.best_threshold,
        }
        logger.info(f"\n{results['classification_report']}")
        logger.info(f"AUC-ROC: {results['auc_roc']:.4f}")
        return results

    def cross_validate(self, X, y, cv: int = 5) -> dict:
        """
        Stratified k-fold cross-validation on ROC-AUC.

        Uses the base XGBoost model directly because ``TemperatureScaledModel``
        does not implement the sklearn estimator API required by
        ``cross_val_score``.

        Args:
            X: Full feature matrix.
            y: Target labels.
            cv (int): Number of folds. Default 5.

        Returns:
            dict: Keys ``mean_auc``, ``std_auc``, ``scores``.
        """
        scores = cross_val_score(
            self._base_model,
            X,
            y,
            cv=StratifiedKFold(cv, shuffle=True, random_state=42),
            scoring="roc_auc",
        )
        logger.info(f"CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
        return {
            "mean_auc": float(scores.mean()),
            "std_auc": float(scores.std()),
            "scores": scores.tolist(),
        }

    def save(self, path: Path) -> None:
        """
        Serialise the trained model and metadata to disk using joblib.

        Args:
            path (Path): Destination file path.
        """
        joblib.dump(
            {
                "model": self.model,
                "base_model": self._base_model,
                "calibrator": self._calibrator,
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
        Deserialise a previously saved TabularCancerModel.

        Args:
            path (Path): Path to the joblib file produced by ``save``.

        Returns:
            TabularCancerModel: Fully restored model instance.
        """
        state = joblib.load(path)
        instance = cls(model_type=state["model_type"])
        instance.model = state["model"]
        instance._base_model = state.get("base_model", state["model"])
        instance._calibrator = state.get("calibrator", None)
        instance.feature_names = state["feature_names"]
        instance.best_threshold = state["best_threshold"]
        return instance
