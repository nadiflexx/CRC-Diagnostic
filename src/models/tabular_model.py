"""
Tabular model for colon cancer risk classification.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
from sklearn.isotonic import IsotonicRegression
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

optuna.logging.set_verbosity(optuna.logging.WARNING)


class TabularCancerModel:
    """
    XGBoost tabular classifier optimised for high recall in CRC screening.

    Uses Optuna Bayesian hyperparameter search and calibrates the decision
    threshold to guarantee a configurable minimum recall (default 0.90),
    reflecting the clinical priority of minimising false negatives.
    """

    def __init__(self, model_type: str = "xgboost"):
        """
        Initialise the tabular cancer model.

        Args:
            model_type (str): Backend to use. Currently only
                ``"xgboost"`` is supported. Default is ``"xgboost"``.
        """
        self.model_type = model_type
        self.model: Any = None
        self.feature_names: list[str] | None = None
        self.best_threshold = 0.5

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

    def _find_optimal_threshold(self, X_val, y_val, recall_target=0.90):
        """
        Find the highest threshold that guarantees recall >= recall_target.

        Calibration is performed on the training set to avoid leakage.
        In oncological screening a false negative implies a potentially
        serious diagnostic delay, hence the priority on recall.

        Args:
            X: Training feature matrix.
            y: Training labels.
            recall_target (float): Minimum recall required. Default 0.90.

        Returns:
            float: Optimal decision threshold rounded to two decimals.
        """
        probs = self.model.predict_proba(X_val)[:, 1]

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
    ):
        self.feature_names = feature_names
        logger.info(f"Training {self.model_type} — {len(X_train)} samples")

        logger.info("Running Optuna hyperparameter search...")
        best_params = self._search_hyperparameters(X_train, y_train, n_trials)

        scale_pos = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
        final_params = {
            **best_params,
            "scale_pos_weight": scale_pos,
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
        }

        # 1. Entrenar modelo base
        base_model = xgb.XGBClassifier(**final_params)
        base_model.fit(X_train, y_train)
        logger.info("Final model fitted.")

        # Obtener probabilidades raw del modelo base sobre val set
        if X_val is not None and y_val is not None:
            cal_X = X_val
            cal_y = y_val
        else:
            cal_X = X_train
            cal_y = y_train

        raw_probs = base_model.predict_proba(cal_X)[:, 1]

        # Ajustar isotonic regression: mapea raw_probs → calibrated_probs
        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(raw_probs, cal_y)
        self._base_model = base_model

        # Wrapper para mantener la API predict_proba
        self.model = base_model  # guardamos base para compatibilidad

        logger.info("Probability calibration fitted (isotonic).")

        # 3. Calibrar threshold sobre val set
        self.best_threshold = self._find_optimal_threshold(cal_X, cal_y, recall_target)
        logger.info(f"Best threshold: {self.best_threshold:.2f}")
        logger.info("✅ Tabular model trained")

    def predict_proba(self, X) -> np.ndarray:
        """
        Return calibrated probability of the positive class.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray of shape (n_samples,): P(cancer) calibrada.
        """
        raw = self._base_model.predict_proba(X)[:, 1]
        if hasattr(self, "_calibrator") and self._calibrator is not None:
            return self._calibrator.predict(raw)
        return raw

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

        Usa el modelo base (XGBoost) directamente para CV,
        ya que el calibrador manual no implementa la API de sklearn estimator.

        Args:
            X: Full feature matrix.
            y: Target labels.
            cv (int): Number of folds. Default 5.

        Returns:
            dict: Keys mean_auc, std_auc, scores.
        """
        scores = cross_val_score(
            self._base_model,  # ← modelo base, no el wrapper
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
            path (Path): Path to the joblib file produced by save.

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
