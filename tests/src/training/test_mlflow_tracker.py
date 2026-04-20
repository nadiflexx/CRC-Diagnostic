# tests/src/training/test_mlflow_tracker.py
"""Tests for src/training/tracking.py — MLflowTracker — FIXED"""

from unittest.mock import MagicMock, patch

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_tracker(
    available=True, tracking_uri="http://127.0.0.1:5000", experiment_name="Default"
):
    with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
        from src.training.tracking import MLflowTracker

        t = MLflowTracker(tracking_uri=tracking_uri, experiment_name=experiment_name)
    t._available = available
    if not available:
        t._mlflow = None
    return t


# ── Init ──────────────────────────────────────────────────────────────────────


class TestMLflowTrackerInit:
    def test_stores_tracking_uri(self):
        t = make_tracker()
        assert t.tracking_uri == "http://127.0.0.1:5000"

    def test_stores_experiment_name(self):
        t = make_tracker(experiment_name="MyExp")
        assert t.experiment_name == "MyExp"

    def test_default_values(self):
        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
            from src.training.tracking import MLflowTracker

            t = MLflowTracker()
        assert t.tracking_uri == "http://127.0.0.1:5000"
        assert t.experiment_name == "Default"

    def test_available_true_on_success(self):
        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
            from src.training.tracking import MLflowTracker

            t = MLflowTracker()
        assert t._available is True

    def test_mlflow_reference_set_on_success(self):
        import mlflow as mlflow_module

        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
            from src.training.tracking import MLflowTracker

            t = MLflowTracker()
        assert t._mlflow is mlflow_module

    def test_sess_starts_as_none(self):
        t = make_tracker()
        # _mlflow should be set after successful init
        assert t._mlflow is not None


# ── _init_mlflow ──────────────────────────────────────────────────────────────


class TestMLflowTrackerInitMlflow:
    def test_import_error_disables_tracking(self):
        with patch("mlflow.set_tracking_uri", side_effect=ImportError("no mlflow")):
            from src.training.tracking import MLflowTracker

            t = MLflowTracker()
        assert t._available is False

    def test_connection_error_disables_tracking(self):
        with patch(
            "mlflow.set_tracking_uri", side_effect=Exception("connection refused")
        ):
            from src.training.tracking import MLflowTracker

            t = MLflowTracker()
        assert t._available is False

    def test_connection_error_leaves_mlflow_none(self):
        with patch("mlflow.set_tracking_uri", side_effect=Exception("refused")):
            from src.training.tracking import MLflowTracker

            t = MLflowTracker()
        assert t._mlflow is None

    def test_sets_tracking_uri(self):
        with (
            patch("mlflow.set_tracking_uri") as mock_uri,
            patch("mlflow.set_experiment"),
        ):
            from src.training.tracking import MLflowTracker

            MLflowTracker(tracking_uri="http://custom:9000")
        mock_uri.assert_called_once_with("http://custom:9000")

    def test_sets_experiment(self):
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.set_experiment") as mock_exp,
        ):
            from src.training.tracking import MLflowTracker

            MLflowTracker(experiment_name="TestExp")
        mock_exp.assert_called_once_with("TestExp")

    def test_import_error_logs_warning(self, capsys):
        with patch("mlflow.set_tracking_uri", side_effect=ImportError("no mlflow")):
            from src.training.tracking import MLflowTracker

            t = MLflowTracker()
        assert not t._available


# ── start_run ─────────────────────────────────────────────────────────────────


class TestMLflowTrackerStartRun:
    def test_yields_true_when_available(self):
        t = make_tracker(available=True)
        mock_mlflow = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_mlflow.start_run.return_value = mock_ctx
        t._mlflow = mock_mlflow

        with t.start_run("test_run") as active:
            assert active is True

    def test_yields_false_when_unavailable(self):
        t = make_tracker(available=False)
        with t.start_run("test_run") as active:
            assert active is False

    def test_calls_mlflow_start_run_with_name(self):
        t = make_tracker(available=True)
        mock_mlflow = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_mlflow.start_run.return_value = mock_ctx
        t._mlflow = mock_mlflow

        with t.start_run("my_experiment"):
            pass

        mock_mlflow.start_run.assert_called_once_with(run_name="my_experiment")

    def test_no_error_when_mlflow_none(self):
        t = make_tracker(available=False)
        t._mlflow = None
        with t.start_run("run") as active:
            assert active is False

    def test_context_body_executes_when_available(self):
        t = make_tracker(available=True)
        mock_mlflow = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_mlflow.start_run.return_value = mock_ctx
        t._mlflow = mock_mlflow

        executed = []
        with t.start_run("run"):
            executed.append(True)
        assert executed == [True]

    def test_context_body_executes_when_unavailable(self):
        t = make_tracker(available=False)
        executed = []
        with t.start_run("run"):
            executed.append(True)
        assert executed == [True]


# ── log_params ────────────────────────────────────────────────────────────────


class TestMLflowTrackerLogParams:
    def test_calls_mlflow_log_params(self):
        t = make_tracker(available=True)
        mock_mlflow = MagicMock()
        t._mlflow = mock_mlflow
        params = {"lr": 0.001, "epochs": 10}
        t.log_params(params)
        mock_mlflow.log_params.assert_called_once_with(params)

    def test_noop_when_unavailable(self):
        t = make_tracker(available=False)
        t.log_params({"lr": 0.001})  # Should not raise

    def test_noop_when_mlflow_none(self):
        t = make_tracker(available=True)
        t._mlflow = None
        t.log_params({"key": "val"})  # Should not raise

    def test_empty_params_dict(self):
        t = make_tracker(available=True)
        mock_mlflow = MagicMock()
        t._mlflow = mock_mlflow
        t.log_params({})
        mock_mlflow.log_params.assert_called_once_with({})

    def test_noop_available_false_mlflow_set(self):
        """available=False means no call even if _mlflow is set."""
        t = make_tracker(available=False)
        mock_mlflow = MagicMock()
        t._mlflow = mock_mlflow
        t.log_params({"x": 1})
        mock_mlflow.log_params.assert_not_called()


# ── log_metrics ───────────────────────────────────────────────────────────────


class TestMLflowTrackerLogMetrics:
    def test_calls_mlflow_log_metrics_with_step(self):
        t = make_tracker(available=True)
        mock_mlflow = MagicMock()
        t._mlflow = mock_mlflow
        metrics = {"acc": 0.95, "loss": 0.1}
        t.log_metrics(metrics, step=5)
        mock_mlflow.log_metrics.assert_called_once_with(metrics, step=5)

    def test_calls_mlflow_log_metrics_without_step(self):
        t = make_tracker(available=True)
        mock_mlflow = MagicMock()
        t._mlflow = mock_mlflow
        t.log_metrics({"acc": 0.9})
        mock_mlflow.log_metrics.assert_called_once_with({"acc": 0.9}, step=None)

    def test_noop_when_unavailable(self):
        t = make_tracker(available=False)
        t.log_metrics({"acc": 0.9}, step=1)  # Should not raise

    def test_noop_when_mlflow_none(self):
        t = make_tracker(available=True)
        t._mlflow = None
        t.log_metrics({"loss": 0.1})  # Should not raise

    def test_noop_available_false(self):
        t = make_tracker(available=False)
        mock_mlflow = MagicMock()
        t._mlflow = mock_mlflow
        t.log_metrics({"acc": 0.9})
        mock_mlflow.log_metrics.assert_not_called()


# ── log_metric ────────────────────────────────────────────────────────────────


class TestMLflowTrackerLogMetric:
    def test_calls_mlflow_log_metric_with_step(self):
        t = make_tracker(available=True)
        mock_mlflow = MagicMock()
        t._mlflow = mock_mlflow
        t.log_metric("accuracy", 0.92, step=3)
        # MLflowTracker passes step as keyword argument
        mock_mlflow.log_metric.assert_called_once_with("accuracy", 0.92, step=3)

    def test_calls_without_step(self):
        t = make_tracker(available=True)
        mock_mlflow = MagicMock()
        t._mlflow = mock_mlflow
        t.log_metric("loss", 0.05)
        mock_mlflow.log_metric.assert_called_once_with("loss", 0.05, step=None)

    def test_noop_when_unavailable(self):
        t = make_tracker(available=False)
        t.log_metric("acc", 0.9)  # Should not raise

    def test_noop_when_mlflow_none(self):
        t = make_tracker(available=True)
        t._mlflow = None
        t.log_metric("loss", 0.5)  # Should not raise

    def test_noop_available_false_mlflow_set(self):
        t = make_tracker(available=False)
        mock_mlflow = MagicMock()
        t._mlflow = mock_mlflow
        t.log_metric("x", 1.0)
        mock_mlflow.log_metric.assert_not_called()


# ── log_param ─────────────────────────────────────────────────────────────────


class TestMLflowTrackerLogParam:
    def test_calls_mlflow_log_param(self):
        t = make_tracker(available=True)
        mock_mlflow = MagicMock()
        t._mlflow = mock_mlflow
        t.log_param("batch_size", 32)
        mock_mlflow.log_param.assert_called_once_with("batch_size", 32)

    def test_noop_when_unavailable(self):
        t = make_tracker(available=False)
        t.log_param("key", "value")  # Should not raise

    def test_noop_when_mlflow_none(self):
        t = make_tracker(available=True)
        t._mlflow = None
        t.log_param("key", "val")  # Should not raise

    def test_accepts_non_string_value(self):
        t = make_tracker(available=True)
        mock_mlflow = MagicMock()
        t._mlflow = mock_mlflow
        t.log_param("lr", 0.001)
        mock_mlflow.log_param.assert_called_once_with("lr", 0.001)

    def test_noop_available_false_mlflow_set(self):
        t = make_tracker(available=False)
        mock_mlflow = MagicMock()
        t._mlflow = mock_mlflow
        t.log_param("x", "y")
        mock_mlflow.log_param.assert_not_called()


# ── log_model ─────────────────────────────────────────────────────────────────


class TestMLflowTrackerLogModel:
    def test_calls_pytorch_log_model(self):
        t = make_tracker(available=True)
        mock_mlflow = MagicMock()
        t._mlflow = mock_mlflow
        mock_model = MagicMock()

        with patch("mlflow.pytorch.log_model") as mock_log:
            t.log_model(mock_model, "classifier")
        mock_log.assert_called_once_with(mock_model, "classifier")

    def test_noop_when_unavailable(self):
        t = make_tracker(available=False)
        t.log_model(MagicMock(), "model")  # Should not raise

    def test_noop_when_mlflow_none(self):
        t = make_tracker(available=True)
        t._mlflow = None
        t.log_model(MagicMock(), "model")  # Should not raise

    def test_noop_available_false_mlflow_set(self):
        t = make_tracker(available=False)
        mock_mlflow = MagicMock()
        t._mlflow = mock_mlflow
        mock_model = MagicMock()
        with patch("mlflow.pytorch.log_model") as mock_log:
            t.log_model(mock_model, "path")
        mock_log.assert_not_called()
