"""
Shared type definitions and lightweight data containers.
"""

from dataclasses import dataclass, field
from typing import TypedDict


class ImageRecord(TypedDict, total=False):
    """Standard record for a training image."""

    file_path: str
    mask_path: str | None
    label: int
    class_name: str
    source: str
    dataset_source: str
    split: str
    width: int
    height: int
    patient_id: str


class CollectedImage(TypedDict, total=False):
    """Record collected from a raw dataset source."""

    original_path: str
    mask_path: str | None
    source: str
    source_detail: str
    class_name: str
    patient_id: str


@dataclass
class TrainValTestSplit:
    """Container for data splits."""

    train_paths: list[str] = field(default_factory=list)
    train_labels: list[int] = field(default_factory=list)
    train_masks: list[str | None] = field(default_factory=list)
    val_paths: list[str] = field(default_factory=list)
    val_labels: list[int] = field(default_factory=list)
    val_masks: list[str | None] = field(default_factory=list)
    test_paths: list[str] = field(default_factory=list)
    test_labels: list[int] = field(default_factory=list)
    test_masks: list[str | None] = field(default_factory=list)


@dataclass
class ValidationMetrics:
    """Standard metrics container for all trainers."""

    loss: float = 0.0
    accuracy: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    auc: float = 0.0
    predictions: list | None = None
    labels: list | None = None
    probabilities: list | None = None

    def to_log_dict(self) -> dict[str, float]:
        """Returns only scalar metrics for logging."""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "auc": self.auc,
        }
