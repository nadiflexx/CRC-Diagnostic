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
    Common base for image-based trainers (classifier, tissue, segmenter).
    Provides: DB loading, source analysis, class weights, early stopping.
    """

    def __init__(
        self,
        device: str | None = None,
        max_patience: int = 10,
    ):
        super().__init__(device)
        self.max_patience = max_patience
        self.tracker = MLflowTracker()

    # ── Data loading from DB ──

    def _load_splits_from_db(self) -> dict:
        """Loads train/val/test splits from the database."""
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
        """Verifies no images are shared between splits."""
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
        """Logs class and source distribution per split."""
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
        """Computes balanced class weights."""
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
        """Logs accuracy breakdown by dataset source."""
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
