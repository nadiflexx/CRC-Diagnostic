"""
Tabular model for colon cancer risk classification.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.config.constants import (
    CSV_MIN_RECALL_TARGET,
    CSV_MLP_ACTIVATION,
    CSV_MLP_ALPHA,
    CSV_MLP_HIDDEN_LAYERS,
    CSV_MLP_LEARNING_RATE,
    CSV_MLP_LEARNING_RATE_INIT,
    CSV_MLP_MAX_ITER,
    CSV_MLP_N_ITER_NO_CHANGE,
    CSV_MLP_SOLVER,
    CSV_RANDOM_SEED,
    CSV_THRESHOLD_MAX,
    CSV_THRESHOLD_MIN,
    CSV_THRESHOLD_STEP,
    CSV_TRAIN_TEST_SIZE,
    CSV_TRAIN_VAL_SIZE,
)
from src.config.logger import log as logger

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════
#  EXISTING: XGBoost / LightGBM model
# ═══════════════════════════════════════════════════════════

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
        """Predict the probability of the positive class for the given input."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        """Predict the class labels for the given input."""
        probs = self.predict_proba(X)
        return (probs >= self.best_threshold).astype(int)

    def evaluate(self, X_test, y_test) -> dict:
        """Evaluate the model on the given test data."""
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
        """Perform cross-validation on the given data."""
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
        """Save the trained model to the specified path."""
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
        """Load a trained model from the specified path."""
        state = joblib.load(path)
        instance = cls(model_type=state["model_type"])
        instance.model = state["model"]
        instance.feature_names = state["feature_names"]
        instance.best_threshold = state["best_threshold"]
        return instance


# ═══════════════════════════════════════════════════════════
#  NEW: MLP model for the CSV pipeline
# ═══════════════════════════════════════════════════════════

@dataclass
class CsvMlpMetrics:
    """Evaluation metrics produced after training the CSV MLP model."""

    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    specificity: float = 0.0
    npv: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    threshold: float = 0.5
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    cv_mean: float = 0.0
    cv_std: float = 0.0
    confusion_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 2), int)
    )


class CsvMlpModel:
    """
    MLP classifier trained on the CSV-based epidemiological features.

    Architecture:  Input(24) → 256 → 128 → 64 → Output(1)
    Optimized for: Recall ≥ CSV_MIN_RECALL_TARGET via threshold search.

    Usage
    -----
    model = CsvMlpModel()
    model.fit(df)          # df must contain 'Diagnosis' column
    model.save(dest_dir)

    model = CsvMlpModel.load(dest_dir)
    proba = model.predict_proba(X_df)
    preds = model.predict(X_df)
    """

    def __init__(self, seed: int = CSV_RANDOM_SEED) -> None:
        self.seed = seed
        self.scaler = StandardScaler()
        self.model: MLPClassifier | None = None
        self.threshold: float = 0.50
        self.feature_names: list[str] = []
        self.metrics: CsvMlpMetrics | None = None

    # ── Internal helpers ────────────────────────────────────────────────────

    def _build_mlp(self, val_fraction: float) -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=CSV_MLP_HIDDEN_LAYERS,
            activation=CSV_MLP_ACTIVATION,
            solver=CSV_MLP_SOLVER,
            alpha=CSV_MLP_ALPHA,
            learning_rate=CSV_MLP_LEARNING_RATE,
            learning_rate_init=CSV_MLP_LEARNING_RATE_INIT,
            max_iter=CSV_MLP_MAX_ITER,
            early_stopping=True,
            validation_fraction=val_fraction,
            n_iter_no_change=CSV_MLP_N_ITER_NO_CHANGE,
            random_state=self.seed,
            verbose=False,
        )

    def _find_optimal_threshold(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> tuple[float, float, float]:
        """Maximize F1 subject to Recall ≥ CSV_MIN_RECALL_TARGET."""
        best_thresh, best_recall, best_f1 = 0.50, 0.0, 0.0
        for t in np.arange(CSV_THRESHOLD_MIN, CSV_THRESHOLD_MAX, CSV_THRESHOLD_STEP):
            y_pred = (y_prob >= t).astype(int)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1  = f1_score(y_true, y_pred, zero_division=0)
            if rec >= CSV_MIN_RECALL_TARGET and f1 > best_f1:
                best_f1, best_recall, best_thresh = f1, rec, round(t, 2)
        return best_thresh, best_recall, best_f1

    # ── Training ────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "CsvMlpModel":
        """
        Train the MLP on a pre-processed DataFrame that contains 'Diagnosis'.

        Steps
        -----
        1. Stratified train / val / test split
        2. StandardScaler (fit on train only)
        3. Early-stopping MLP on train+val
        4. Optimal threshold search on val (recall ≥ target)
        5. 5-fold CV on train for stability estimate
        6. Final evaluation on test → CsvMlpMetrics
        """
        import pandas as pd  # local to avoid circular at module level

        np.random.seed(self.seed)
        X = df.drop(columns=["Diagnosis"])
        y = df["Diagnosis"]
        self.feature_names = X.columns.tolist()

        val_ratio = CSV_TRAIN_VAL_SIZE / (1 - CSV_TRAIN_TEST_SIZE)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=CSV_TRAIN_TEST_SIZE, stratify=y, random_state=self.seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=self.seed
        )
        logger.info(
            f"[CsvMlpModel] Split — Train: {len(X_train):,} | "
            f"Val: {len(X_val):,} | Test: {len(X_test):,}"
        )

        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s   = self.scaler.transform(X_val)
        X_test_s  = self.scaler.transform(X_test)

        self.model = self._build_mlp(val_fraction=val_ratio)
        X_full = np.vstack([X_train_s, X_val_s])
        y_full = np.concatenate([y_train, y_val])
        t0 = time.time()
        self.model.fit(X_full, y_full)
        logger.info(
            f"[CsvMlpModel] Trained in {time.time()-t0:.1f}s | "
            f"Epochs: {self.model.n_iter_} | Loss: {self.model.loss_:.4f}"
        )

        y_prob_val = self.model.predict_proba(X_val_s)[:, 1]
        self.threshold, best_rec, best_f1 = self._find_optimal_threshold(
            y_val.to_numpy(), y_prob_val
        )
        logger.info(
            f"[CsvMlpModel] Threshold: {self.threshold} | "
            f"Recall_val: {best_rec:.4f} | F1_val: {best_f1:.4f}"
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        cv_scores = cross_val_score(
            self._build_mlp(val_fraction=val_ratio),
            X_train_s, y_train,
            cv=cv, scoring="recall", n_jobs=-1,
        )
        logger.info(
            f"[CsvMlpModel] CV Recall: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

        y_prob_test = self.model.predict_proba(X_test_s)[:, 1]
        y_pred_test = (y_prob_test >= self.threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_test)
        tn, fp, fn, tp = cm.ravel()

        self.metrics = CsvMlpMetrics(
            recall=recall_score(y_test, y_pred_test),
            precision=precision_score(y_test, y_pred_test),
            f1=f1_score(y_test, y_pred_test),
            specificity=tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            npv=tn / (tn + fn) if (tn + fn) > 0 else 0.0,
            roc_auc=roc_auc_score(y_test, y_prob_test),
            pr_auc=average_precision_score(y_test, y_prob_test),
            threshold=self.threshold,
            tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
            cv_mean=float(cv_scores.mean()),
            cv_std=float(cv_scores.std()),
            confusion_matrix=cm,
        )
        self._log_metrics()
        return self

    # ── Inference ───────────────────────────────────────────────────────────

    def predict_proba(self, X) -> np.ndarray:
        """Return cancer class probabilities."""
        assert self.model is not None, "Model not trained."
        import pandas as pd
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict_proba(self.scaler.transform(X_arr))[:, 1]

    def predict(self, X) -> np.ndarray:
        """Return binary predictions using the optimal threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, output_dir: str | Path) -> Path:
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model,  dest / "mlp_model.pkl")
        joblib.dump(self.scaler, dest / "scaler.pkl")
        joblib.dump(
            {"threshold": self.threshold, "features": self.feature_names},
            dest / "model_config.pkl",
        )
        logger.info(f"[CsvMlpModel] Saved to {dest}")
        return dest

    @classmethod
    def load(cls, model_dir: str | Path) -> "CsvMlpModel":
        src = Path(model_dir)
        instance = cls()
        instance.model   = joblib.load(src / "mlp_model.pkl")
        instance.scaler  = joblib.load(src / "scaler.pkl")
        config = joblib.load(src / "model_config.pkl")
        instance.threshold    = config["threshold"]
        instance.feature_names = config["features"]
        logger.info(f"[CsvMlpModel] Loaded from {src}")
        return instance

    # ── Logging ─────────────────────────────────────────────────────────────

    def _log_metrics(self) -> None:
        m = self.metrics
        if m is None:
            return
        logger.info("=" * 60)
        logger.info("  [CsvMlpModel] MÉTRICAS FINALES — TEST")
        logger.info("=" * 60)
        logger.info(f"  ★ RECALL      : {m.recall:.4f}  ← MÉTRICA PRINCIPAL")
        logger.info(f"    PRECISION   : {m.precision:.4f}")
        logger.info(f"    F1-SCORE    : {m.f1:.4f}")
        logger.info(f"    SPECIFICITY : {m.specificity:.4f}")
        logger.info(f"    NPV         : {m.npv:.4f}")
        logger.info(f"    ROC-AUC     : {m.roc_auc:.4f}")
        logger.info(f"    PR-AUC      : {m.pr_auc:.4f}")
        logger.info(f"    CV Recall   : {m.cv_mean:.4f} ± {m.cv_std:.4f}")
        logger.info(f"    Threshold   : {m.threshold}")
        logger.info(f"    TP / FN     : {m.tp:,} / {m.fn:,}")