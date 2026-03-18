"""
Abstract base classes defining contracts for all major components.
Enforces consistency without altering any implementation logic.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# ═══════════════════════════════════════════════════════════
#  PREPROCESSING
# ═══════════════════════════════════════════════════════════


class BaseImagePreprocessor(ABC):
    """Contract for all image preprocessing pipelines."""

    @abstractmethod
    def process_image(self, img: np.ndarray) -> np.ndarray | list[np.ndarray]:
        """Process a single image (BGR)."""
        ...

    def suppress_green(self, img: np.ndarray) -> np.ndarray:
        """Optional: suppress green scope guides."""
        return img


class BaseTabularPreprocessor(ABC):
    """Contract for tabular data preprocessing."""

    @abstractmethod
    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray | None, list[str]]: ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseTabularPreprocessor": ...


# ═══════════════════════════════════════════════════════════
#  MODELS
# ═══════════════════════════════════════════════════════════


class BaseClassifier(ABC, torch.nn.Module):
    """Contract for all classifiers (image and tabular)."""

    @abstractmethod
    def forward(self, x: Any) -> Any: ...

    @abstractmethod
    def predict_proba(self, x: Any) -> Any: ...


class BaseSegmenter(ABC, torch.nn.Module):
    """Contract for segmentation models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def predict_mask(
        self, x: torch.Tensor, threshold: float
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


# ═══════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════


class BaseTrainer(ABC):
    """
    Contract for all trainers.
    Enforces consistent lifecycle: load_data → train → evaluate → save.
    """

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.best_metric: float = 0.0
        self.patience_counter: int = 0
        self.max_patience: int = 10
        self.history: dict[str, list] = {}

    @abstractmethod
    def train(self, **kwargs) -> None:
        """Run the full training pipeline."""
        ...

    @abstractmethod
    def _load_data(self) -> dict:
        """Load and prepare data splits."""
        ...

    @abstractmethod
    def _save_checkpoint(self, epoch: int, metrics: dict) -> None:
        """Save model checkpoint."""
        ...

    @abstractmethod
    def load_best(self) -> None:
        """Load best checkpoint."""
        ...

    def _check_early_stopping(self, current_metric: float) -> bool:
        """
        Common early stopping logic.
        Returns True if training should stop.
        """
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.max_patience


# ═══════════════════════════════════════════════════════════
#  DATA INGESTION
# ═══════════════════════════════════════════════════════════


class BaseDatasetOrganizer(ABC):
    """Contract for dataset organizers."""

    @abstractmethod
    def run(self) -> None:
        """Execute the full organization pipeline."""
        ...
