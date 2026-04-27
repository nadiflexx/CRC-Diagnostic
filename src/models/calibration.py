"""
src/models/calibration.py

Temperature Scaling calibration wrapper for XGBoost classifiers.

Defined in its own module so that joblib can correctly
serialise/deserialise the class from any script (app.py, notebooks, …).

Reference: Guo et al., "On Calibration of Modern Neural Networks",
ICML 2017. The same technique applies to any classifier that emits
log-odds.

Clinical rationale
------------------
A diagnostic support system must never claim 0% or 100% certainty.
``TemperatureScaledModel`` enforces hard probability floors/ceilings
(``PROB_MIN = 0.05``, ``PROB_MAX = 0.95``) consistent with FDA AI/ML
Guidance 2023 and Jiang et al., Radiology 2012.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

# ── Internal helpers ─────────────────────────────────────────────────────────


def _logit(p: np.ndarray) -> np.ndarray:
    """Convert probabilities to log-odds (logit) with clipping for stability.

    Args:
        p: Probability array with values in (0, 1).

    Returns:
        Log-odds array of the same shape.
    """
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function.

    Args:
        x: Input array (log-odds).

    Returns:
        Probability array with values in (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-x))


# ── Public API ────────────────────────────────────────────────────────────────


def find_temperature(
    model,
    X_cal,
    y_cal,
    t_minimum: float = 1.5,
) -> float:
    """Find the optimal temperature by minimising log-loss on a calibration set.

    T > 1 softens the distribution (moves extremes toward the centre).
    T = 1 is equivalent to the uncalibrated model.
    A minimum ``t_minimum`` is enforced as a clinical safeguard: no
    diagnostic support system should emit probabilities so extreme that
    they simulate absolute certainty.

    Args:
        model: Fitted ``XGBClassifier`` (or any classifier with
            ``predict_proba``).
        X_cal: Feature matrix of the calibration set.
        y_cal: Labels of the calibration set.
        t_minimum (float): Minimum allowed temperature. Default 1.5.

    Returns:
        Effective temperature T (float), never smaller than ``t_minimum``.
    """
    probs_raw = model.predict_proba(X_cal)[:, 1]
    logits = _logit(probs_raw)
    y = np.asarray(y_cal, dtype=float)

    def nll(T: float) -> float:
        p = _sigmoid(logits / T)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    result = minimize_scalar(nll, bounds=(0.5, 20.0), method="bounded")
    t_optimal = float(result.x)
    t_effective = max(t_optimal, t_minimum)

    if t_effective > t_optimal:
        from src.config.logger import log as logger

        logger.info(
            f"[Temp] Optimal NLL T={t_optimal:.4f} < clinical minimum "
            f"{t_minimum} → applying T={t_effective}"
        )
    return t_effective


class TemperatureScaledModel:
    """Wrapper that applies Temperature Scaling over raw XGBoost probabilities.

    Divides the log-odds by a temperature T > 1 before the sigmoid,
    physically flattening the probability distribution toward the centre.
    Unlike isotonic regression, this does NOT preserve the output range:
    an overconfident model emitting 99.9% produces ~85-90% with T=3,
    giving a clinically interpretable probability space.

    Discrimination (AUC) is practically unchanged.

    Attributes:
        base_model: Fitted ``XGBClassifier``.
        temperature (float): Parameter T > 1 found during calibration.
        PROB_MIN (float): Hard floor for returned probabilities (0.05).
        PROB_MAX (float): Hard ceiling for returned probabilities (0.95).
    """

    PROB_MIN: float = 0.05
    PROB_MAX: float = 0.95

    def __init__(self, base_model, temperature: float) -> None:
        """
        Initialise the calibrated model wrapper.

        Args:
            base_model: Fitted ``XGBClassifier``.
            temperature (float): Temperature T > 1.
        """
        self.base_model = base_model
        self.temperature = float(temperature)

    def predict_proba(self, X) -> np.ndarray:
        """Return calibrated probabilities of shape (n_samples, 2).

        Pipeline:
            1. XGBoost emits raw probabilities.
            2. Convert to log-odds (logit).
            3. Divide by T (flattens the distribution).
            4. Apply sigmoid to return to [0, 1].
            5. Clip to [PROB_MIN, PROB_MAX].

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray of shape (n_samples, 2): columns are
            [P(healthy), P(cancer)].
        """
        raw = self.base_model.predict_proba(X)[:, 1]
        logits = _logit(raw)
        cal = _sigmoid(logits / self.temperature)
        cal = np.clip(cal, self.PROB_MIN, self.PROB_MAX)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X) -> np.ndarray:
        """Binary prediction with threshold 0.5.

        The operational threshold is managed externally by
        ``TabularCancerModel.best_threshold``.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray of shape (n_samples,): predicted labels (0 or 1).
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
