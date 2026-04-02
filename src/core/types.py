"""
Shared type definitions and lightweight data containers.
"""

from dataclasses import dataclass, field
from typing import TypedDict

import pandas as pd


# ═══════════════════════════════════════════════════════════
#  IMAGE / TRAINING TYPES  (existing)
# ═══════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════
#  CSV TABULAR PIPELINE TYPES  (colorectal_cancer_dataset.csv)
# ═══════════════════════════════════════════════════════════

class CsvPatientFormInput(TypedDict):
    """
    Raw values from the clinical form before numeric encoding.
    Used exclusively by the CSV-based tabular pipeline (MLP model).
    """

    age: int
    gender: str                   # 'M' | 'F'
    family_history: str           # 'Yes' | 'No'
    smoking: str                  # 'Yes' | 'No'
    alcohol: str                  # 'Yes' | 'No'
    obesity: str                  # 'Normal' | 'Overweight' | 'Obese'
    diet_risk: str                # 'Low' | 'Moderate' | 'High'
    physical_activity: str        # 'Low' | 'Moderate' | 'High'
    diabetes: str                 # 'Yes' | 'No'
    ibd: str                      # 'Yes' | 'No'
    genetic: str                  # 'Yes' | 'No'
    screening: str                # 'Never' | 'Irregular' | 'Regular'
    early_detection: str          # 'Yes' | 'No'
    incidence_rate: float
    mortality_rate: float
    urban_rural: str              # 'Urban' | 'Rural'


@dataclass
class CsvPatientFeatureResult:
    """
    Result of `build_csv_feature_row`: numeric feature row + human-readable
    labels for derived features to display in the dashboard.
    """

    df_row: pd.DataFrame          # 1 row × 24 numeric features
    risk_score: float             # [0-10]
    prevention_index: float       # [0-10]
    access_score: float           # [0 | 1]
    age_risk_group: str           # 'Low' | 'Medium' | 'High' | 'Very_High'
    lifestyle_cluster: str        # 'Healthy' | 'Dietary' | 'Sedentary' | 'High_Risk'


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
    confusion_matrix: "np.ndarray" = field(  # type: ignore[assignment]
        default_factory=lambda: __import__("numpy").zeros((2, 2), int)
    )