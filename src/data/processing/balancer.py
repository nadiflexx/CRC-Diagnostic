"""
Dataset balancing strategies.
"""

from typing import Any

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.config.logger import log as logger


class DataBalancer:
    """
    Collection of static methods for analysing and correcting class
    imbalance in tabular datasets.

    Supports oversampling (SMOTE, ADASYN, SMOTETomek), random
    undersampling, class-weight computation, and balance reporting.
    All methods are stateless and operate directly on NumPy arrays.
    """

    @staticmethod
    def check_balance(y: np.ndarray) -> dict:
        """
        Report the class distribution and balance ratio of a label array.

        Args:
            y (np.ndarray): 1-D integer array of class labels.

        Returns:
            dict: Dictionary with one entry per class (keyed by integer
                class index) plus two summary keys:
                    - Per-class entry: ``{"count": int, "percentage": float}``.
                    - ``"balance_ratio"`` (float): Ratio of the minority
                      to the majority class count, in [0, 1].
                    - ``"is_balanced"`` (bool): ``True`` if the ratio
                      exceeds 0.7.
        """
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        report: dict[int | str, Any] = {}
        for cls, cnt in zip(unique, counts, strict=True):
            report[int(cls)] = {
                "count": int(cnt),
                "percentage": round(cnt / total * 100, 2),
            }
        ratio = min(counts) / max(counts) if len(counts) > 1 else 1.0
        report["balance_ratio"] = round(ratio, 3)
        report["is_balanced"] = ratio > 0.7
        logger.info(f"Balance: {report}")
        return report

    @staticmethod
    def oversample_minority(X, y, strategy="smote"):
        """
        Oversample the minority class using a resampling technique.

        Args:
            X (array-like of shape (n_samples, n_features)): Feature
                matrix.
            y (array-like of shape (n_samples,)): Class labels.
            strategy (str): Resampling algorithm to apply. One of:
                - ``"smote"``: Synthetic Minority Over-sampling Technique.
                - ``"adasyn"``: Adaptive Synthetic Sampling.
                - ``"smote_tomek"``: SMOTE followed by Tomek link removal.
                Default is ``"smote"``.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - Resampled feature matrix.
                - Resampled label array.

        Raises:
            ValueError: If ``strategy`` is not one of the supported
                values.
        """
        if strategy == "smote":
            sampler = SMOTE(random_state=42, k_neighbors=5)
        elif strategy == "adasyn":
            sampler = ADASYN(random_state=42)
        elif strategy == "smote_tomek":
            sampler = SMOTETomek(random_state=42)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        X_res, y_res = sampler.fit_resample(X, y)
        logger.info(f"Oversampling ({strategy}): {len(y)} → {len(y_res)}")
        return X_res, y_res

    @staticmethod
    def undersample_majority(X, y, target_ratio=1.0):
        """
        Randomly undersample the majority class to achieve a target ratio.

        Args:
            X (np.ndarray of shape (n_samples, n_features)): Feature
                matrix.
            y (np.ndarray of shape (n_samples,)): Class labels.
            target_ratio (float): Desired ratio of minority to majority
                class samples after undersampling. A value of 1.0
                produces a perfectly balanced dataset. Default is 1.0.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - Undersampled feature matrix.
                - Undersampled label array, shuffled.
        """
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        n_minority = counts.min()
        target_majority = int(n_minority / target_ratio)

        minority_mask = y == minority_class
        majority_mask = y == majority_class
        majority_indices = np.where(majority_mask)[0]
        sampled_indices = np.random.choice(
            majority_indices, size=target_majority, replace=False
        )
        final_indices = np.concatenate([np.where(minority_mask)[0], sampled_indices])
        np.random.shuffle(final_indices)
        return X[final_indices], y[final_indices]

    @staticmethod
    def compute_class_weights(y: np.ndarray) -> dict:
        """
        Compute balanced class weights for use in a weighted loss function.

        Uses scikit-learn's ``compute_class_weight`` with
        ``class_weight="balanced"`` so that each class contributes
        equally to the total loss regardless of its sample count.

        Args:
            y (np.ndarray of shape (n_samples,)): Integer class labels
                from the training split.

        Returns:
            dict[int, float]: Dictionary mapping each integer class index
                to its corresponding weight value.
        """
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        weight_dict = {int(c): float(w) for c, w in zip(classes, weights, strict=True)}
        logger.info(f"Class weights: {weight_dict}")
        return weight_dict
