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
    @staticmethod
    def check_balance(y: np.ndarray) -> dict:
        """
        Check the balance of the dataset.

        :param y: Array of class labels.
        :return: Dictionary with balance information.
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
        Oversample the minority class.

        :param X: Array of features.
        :param y: Array of class labels.
        :param strategy: Oversampling strategy.
        :return: Oversampled features and labels.
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
        Undersample the majority class.

        :param X: Array of features.
        :param y: Array of class labels.
        :param target_ratio: Target ratio of minority to majority class sizes.
        :return: Undersampled features and labels.
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
        Compute class weights.

        :param y: Array of class labels.
        :return: Dictionary of class weights.
        """
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        weight_dict = {int(c): float(w) for c, w in zip(classes, weights, strict=True)}
        logger.info(f"Class weights: {weight_dict}")
        return weight_dict
