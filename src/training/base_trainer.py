"""
Base trainer with common patterns extracted from all trainers.
Subclasses implement _load_data, _train_epoch, _validate, _save_checkpoint.
"""

from collections import Counter
from pathlib import Path

from data.database.connection import get_db
from data.database.repositories import TrainingImageRepository
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from config.constants import detect_source_from_stem
from config.logger import log as logger
from core.interfaces import BaseTrainer
from training.tracking import MLflowTracker


class ImageTrainerBase(BaseTrainer):
    """
    Abstract base class for all image-based training pipelines.

    Provides reusable utilities for database loading, data-leakage
    verification, class-distribution logging, class-weight computation,
    and per-source accuracy analysis. Concrete subclasses (classifier,
    tissue classifier, segmenter) are expected to implement
    ``_load_data``, ``_train_epoch``, ``_validate``, and
    ``_save_checkpoint``.
    """

    def __init__(
        self,
        device: str | None = None,
        max_patience: int = 10,
    ):
        """
        Initialise the base trainer with device selection and MLflow tracker.

        Args:
            device (str | None): Target device identifier (e.g. ``"cuda"``
                or ``"cpu"``). Passed to the parent ``BaseTrainer.__init__``.
            max_patience (int): Number of consecutive epochs without
                improvement before early stopping is triggered. Default is 10.
        """
        super().__init__(device)
        self.max_patience = max_patience
        self.tracker = MLflowTracker()

    # ── Data loading from DB ──

    def _load_splits_from_db(self) -> dict:
        """
        Load train, validation, and test splits from the database.

        For each split, retrieves file paths, integer labels, and optional
        mask paths from ``TrainingImageRepository``.

        Returns:
            dict: Dictionary with keys ``"train"``, ``"val"``, and
                ``"test"``. Each value is a nested dictionary with keys:
                    - ``"paths"`` (list[str]): Absolute file paths.
                    - ``"labels"`` (list[int]): Integer class labels.
                    - ``"masks"`` (list[str | None]): Mask file paths, or
                      ``None`` when no mask is available.
        """
        with get_db() as db:
            repo = TrainingImageRepository(db)
            result = {}
            for split in ("train", "val", "test"):
                data = repo.get_by_split(split)
                result[split] = {
                    "paths": [str(d.file_path) for d in data],
                    "labels": [int(d.label) for d in data],
                    "masks": [str(d.mask_path) if d.mask_path else None for d in data],
                }
            return result

    def _verify_no_leakage(self, data: dict) -> None:
        """
        Assert that no image filename appears in more than one split.

        Checks all pairwise intersections of filename sets across train,
        val, and test.

        Args:
            data (dict): Data dictionary as returned by
                ``_load_splits_from_db``, where each split contains a
                ``"paths"`` list.

        Raises:
            ValueError: If any filenames are shared between splits,
                indicating a data-leakage risk.
        """
        names = {
            split: {Path(p).name for p in data[split]["paths"]}
            for split in ("train", "val", "test")
        }
        overlaps = (
            (names["train"] & names["val"])
            | (names["train"] & names["test"])
            | (names["val"] & names["test"])
        )
        if overlaps:
            raise ValueError(
                f"Data leakage: {len(overlaps)} images shared between splits"
            )
        logger.info("  ✅ No data leakage between splits")

    def _log_split_distribution(self, data: dict, class_names: dict[int, str]) -> None:
        """
        Log per-class and per-source sample counts for every split.

        Args:
            data (dict): Data dictionary as returned by
                ``_load_splits_from_db``.
            class_names (dict[int, str]): Mapping from integer label to
                human-readable class name used for display purposes.
        """
        for split in ("train", "val", "test"):
            labels = data[split]["paths"]
            label_list = data[split]["labels"]

            label_counts = Counter(label_list)
            detail = ", ".join(
                f"{class_names.get(k, '?')}={v}"
                for k, v in sorted(label_counts.items())
            )

            source_counts = Counter(
                detect_source_from_stem(Path(p).stem) for p in labels
            )
            source_detail = ", ".join(
                f"{k}={v}" for k, v in sorted(source_counts.items())
            )

            logger.info(f"  {split}: {len(label_list)} ({detail})")
            logger.info(f"         sources: {source_detail}")

    def _compute_class_weights(self, labels: list[int]) -> np.ndarray:
        """
        Compute balanced class weights for use in a weighted loss function.

        Uses scikit-learn's ``compute_class_weight`` with
        ``class_weight="balanced"`` so that minority classes receive
        proportionally higher weights.

        Args:
            labels (list[int]): Integer class labels from the training split.

        Returns:
            np.ndarray: 1-D array of class weights with length equal to the
                number of unique classes, ordered by class index.
        """
        classes = np.unique(labels)
        weights = compute_class_weight("balanced", classes=classes, y=np.array(labels))
        weight_dict = dict(
            zip(classes.tolist(), weights.round(3).tolist(), strict=True)
        )
        logger.info(f"  Class weights: {weight_dict}")
        return weights

    def _analyze_source_accuracy(
        self, paths_list: list[str], predictions: np.ndarray, labels: np.ndarray
    ) -> None:
        """
        Log per-source accuracy breakdown for the test or validation set.

        The source of each image is inferred from its filename stem using
        ``detect_source_from_stem``. Correct and total counts are
        accumulated per source and the accuracy ratio is logged.

        Args:
            paths_list (list[str]): Absolute file paths aligned with
                ``predictions`` and ``labels``.
            predictions (np.ndarray): Predicted class indices of shape (N,).
            labels (np.ndarray): Ground-truth class indices of shape (N,).
        """
        logger.info("\n  📊 ACCURACY BY SOURCE:")
        source_results: dict[str, dict] = {}
        for idx, p in enumerate(paths_list):
            source = detect_source_from_stem(Path(p).stem)
            if source not in source_results:
                source_results[source] = {"correct": 0, "total": 0}
            source_results[source]["total"] += 1
            if idx < len(predictions) and predictions[idx] == labels[idx]:
                source_results[source]["correct"] += 1

        for source, stats in sorted(source_results.items()):
            acc = stats["correct"] / max(stats["total"], 1)
            logger.info(
                f"    {source:15s}: {stats['correct']}/{stats['total']} ({acc:.1%})"
            )
