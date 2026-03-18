"""
MLflow tracking utilities — isolated from training logic.
All MLflow imports and calls are contained here.
"""

from contextlib import contextmanager
from typing import Any

import mlflow
import mlflow.pytorch

from src.config.logger import log as logger


class MLflowTracker:
    """Wrapper around MLflow for experiment tracking."""

    def __init__(
        self,
        tracking_uri: str = "http://127.0.0.1:5000",
        experiment_name: str = "Default",
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._mlflow = None
        self._available = False
        self._init_mlflow()

    def _init_mlflow(self) -> None:
        """Lazy import — training works even without MLflow."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
            self._available = True
        except ImportError:
            logger.warning("MLflow not installed — tracking disabled")
        except Exception as e:
            logger.warning(f"MLflow connection failed: {e} — tracking disabled")

    @contextmanager
    def start_run(self, run_name: str):
        """Context manager for an MLflow run. No-op if unavailable."""
        if self._available and self._mlflow is not None:
            with self._mlflow.start_run(run_name=run_name):
                yield True
        else:
            yield False

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        if self._available and self._mlflow is not None:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log evaluation metrics."""
        if self._available and self._mlflow is not None:
            self._mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric."""
        if self._available and self._mlflow is not None:
            self._mlflow.log_metric(key, value, step=step)

    def log_param(self, key: str, value: Any) -> None:
        """Log a single hyperparameter."""
        if self._available and self._mlflow is not None:
            self._mlflow.log_param(key, value)

    def log_model(self, model: Any, artifact_path: str) -> None:
        """Log the trained model."""
        if self._available and self._mlflow is not None:
            import mlflow.pytorch

            mlflow.pytorch.log_model(model, artifact_path)
