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
    """
    Lightweight wrapper around MLflow for experiment tracking.

    All MLflow interactions are encapsulated here so that training modules
    remain free of direct MLflow dependencies. If MLflow is unavailable or
    the tracking server cannot be reached, all methods become no-ops and
    training continues uninterrupted.
    """

    def __init__(
        self,
        tracking_uri: str = "http://127.0.0.1:5000",
        experiment_name: str = "Default",
    ):
        """
        Initialise the tracker and attempt to connect to the MLflow server.

        Args:
            tracking_uri (str): URI of the MLflow tracking server.
                Default is ``"http://127.0.0.1:5000"``.
            experiment_name (str): Name of the MLflow experiment to use or
                create. Default is ``"Default"``.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._mlflow = None
        self._available = False
        self._init_mlflow()

    def _init_mlflow(self) -> None:
        """
        Attempt to configure the MLflow tracking URI and experiment.

        Sets ``self._available`` to ``True`` on success. On ``ImportError``
        or any connection failure the flag remains ``False`` and a warning
        is logged so that training can proceed without tracking.
        """
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
        """
        Context manager that wraps code in an MLflow run.

        If MLflow is unavailable the context manager yields ``False`` and
        the enclosed code executes normally without any tracking.

        Args:
            run_name (str): Human-readable name assigned to the MLflow run.

        Yields:
            bool: ``True`` when an active MLflow run was successfully
                started, ``False`` otherwise.
        """
        if self._available and self._mlflow is not None:
            with self._mlflow.start_run(run_name=run_name):
                yield True
        else:
            yield False

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log a dictionary of hyperparameters to the active MLflow run.

        Args:
            params (dict[str, Any]): Key-value pairs where keys are
                parameter names and values are their settings.
        """
        if self._available and self._mlflow is not None:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """
        Log a dictionary of scalar metrics to the active MLflow run.

        Args:
            metrics (dict[str, float]): Key-value pairs where keys are
                metric names and values are their scalar measurements.
            step (int | None): Optional global step index associated with
                the metrics (e.g. epoch number). Default is ``None``.
        """
        if self._available and self._mlflow is not None:
            self._mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """
        Log a single scalar metric to the active MLflow run.

        Args:
            key (str): Metric name.
            value (float): Scalar metric value.
            step (int | None): Optional global step index. Default is
                ``None``.
        """
        if self._available and self._mlflow is not None:
            self._mlflow.log_metric(key, value, step=step)

    def log_param(self, key: str, value: Any) -> None:
        """
        Log a single hyperparameter to the active MLflow run.

        Args:
            key (str): Parameter name.
            value (Any): Parameter value (must be serialisable to string
                by MLflow).
        """
        if self._available and self._mlflow is not None:
            self._mlflow.log_param(key, value)

    def log_model(self, model: Any, artifact_path: str) -> None:
        """
        Log a PyTorch model as an MLflow artifact.

        Args:
            model (Any): PyTorch ``nn.Module`` instance to be saved.
            artifact_path (str): Relative path within the MLflow run's
                artifact store where the model will be saved.
        """
        if self._available and self._mlflow is not None:
            import mlflow.pytorch

            mlflow.pytorch.log_model(model, artifact_path)
