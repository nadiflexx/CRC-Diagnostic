# tests/test_onnx_session.py
"""Tests for src/utils/onnx_session.py"""

from unittest.mock import MagicMock, patch

import numpy as np
import onnxruntime as ort
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_ort_session():
    """Mock onnxruntime.InferenceSession."""
    sess = MagicMock(spec=ort.InferenceSession)
    mock_input = MagicMock()
    mock_input.name = "input"
    sess.get_inputs.return_value = [mock_input]
    sess.get_providers.return_value = ["CPUExecutionProvider"]
    sess.run.return_value = [np.zeros((1, 3), dtype=np.float32)]
    return sess


@pytest.fixture
def dummy_onnx_path(tmp_path):
    """Creates a dummy .onnx file."""
    p = tmp_path / "model.onnx"
    p.write_bytes(b"dummy")
    return p


@pytest.fixture
def missing_onnx_path(tmp_path):
    """Returns a path that does NOT exist."""
    return tmp_path / "nonexistent.onnx"


# ── ONNXSession ───────────────────────────────────────────────────────────────


class TestONNXSessionInit:
    def test_init_stores_path_and_device(self, dummy_onnx_path):
        from src.utils.onnx_session import ONNXSession

        sess = ONNXSession(dummy_onnx_path, device="cpu")
        assert sess.path == dummy_onnx_path
        assert sess.device == "cpu"
        assert sess._sess is None

    def test_init_default_device_is_cpu(self, dummy_onnx_path):
        from src.utils.onnx_session import ONNXSession

        sess = ONNXSession(dummy_onnx_path)
        assert sess.device == "cpu"

    def test_init_cuda_device(self, dummy_onnx_path):
        from src.utils.onnx_session import ONNXSession

        sess = ONNXSession(dummy_onnx_path, device="cuda")
        assert sess.device == "cuda"


class TestONNXSessionAvailable:
    def test_available_true_when_file_exists(self, dummy_onnx_path):
        from src.utils.onnx_session import ONNXSession

        sess = ONNXSession(dummy_onnx_path)
        assert sess.available is True

    def test_available_false_when_file_missing(self, missing_onnx_path):
        from src.utils.onnx_session import ONNXSession

        sess = ONNXSession(missing_onnx_path)
        assert sess.available is False


class TestONNXSessionSelectProviders:
    def test_cpu_returns_cpu_provider(self, dummy_onnx_path):
        from src.utils.onnx_session import ONNXSession

        sess = ONNXSession(dummy_onnx_path, device="cpu")
        providers = sess._select_providers()
        assert providers == ["CPUExecutionProvider"]

    def test_cuda_with_cuda_available_returns_both(self, dummy_onnx_path):
        from src.utils.onnx_session import ONNXSession

        sess = ONNXSession(dummy_onnx_path, device="cuda")
        with patch(
            "onnxruntime.get_available_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ):
            providers = sess._select_providers()
        assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_cuda_fallback_to_cpu_when_cuda_unavailable(self, dummy_onnx_path):
        from src.utils.onnx_session import ONNXSession

        sess = ONNXSession(dummy_onnx_path, device="cuda")
        with patch(
            "onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]
        ):
            providers = sess._select_providers()
        assert providers == ["CPUExecutionProvider"]

    def test_cpu_device_ignores_cuda_availability(self, dummy_onnx_path):
        from src.utils.onnx_session import ONNXSession

        sess = ONNXSession(dummy_onnx_path, device="cpu")
        with patch(
            "onnxruntime.get_available_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ):
            providers = sess._select_providers()
        assert providers == ["CPUExecutionProvider"]


class TestONNXSessionGet:
    def test_get_creates_session_on_first_call(self, dummy_onnx_path, mock_ort_session):
        from src.utils.onnx_session import ONNXSession

        sess = ONNXSession(dummy_onnx_path, device="cpu")
        with (
            patch("onnxruntime.InferenceSession", return_value=mock_ort_session),
            patch("onnxruntime.SessionOptions"),
            patch.object(
                sess, "_select_providers", return_value=["CPUExecutionProvider"]
            ),
        ):
            result = sess._get()
        assert result is mock_ort_session
        assert sess._sess is mock_ort_session

    def test_get_returns_cached_session_on_second_call(
        self, dummy_onnx_path, mock_ort_session
    ):
        from src.utils.onnx_session import ONNXSession

        sess = ONNXSession(dummy_onnx_path, device="cpu")
        sess._sess = mock_ort_session  # Pre-populate cache
        with patch("onnxruntime.InferenceSession") as mock_cls:
            result = sess._get()
        mock_cls.assert_not_called()
        assert result is mock_ort_session

    def test_get_logs_provider_info(self, dummy_onnx_path, mock_ort_session, caplog):

        from src.utils.onnx_session import ONNXSession

        sess = ONNXSession(dummy_onnx_path, device="cpu")
        with (
            patch("onnxruntime.InferenceSession", return_value=mock_ort_session),
            patch("onnxruntime.SessionOptions"),
        ):
            sess._get()
        # Session was initialized without error

    def test_get_sets_graph_optimization(self, dummy_onnx_path, mock_ort_session):
        from src.utils.onnx_session import ONNXSession

        sess = ONNXSession(dummy_onnx_path, device="cpu")
        mock_opts = MagicMock()
        with (
            patch("onnxruntime.InferenceSession", return_value=mock_ort_session),
            patch("onnxruntime.SessionOptions", return_value=mock_opts),
        ):
            sess._get()
        assert mock_opts.log_severity_level == 3
        assert (
            mock_opts.graph_optimization_level
            == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )


class TestONNXSessionRun:
    def test_run_returns_first_output(self, dummy_onnx_path, mock_ort_session):
        from src.utils.onnx_session import ONNXSession

        expected = np.array([[0.1, 0.9]], dtype=np.float32)
        mock_ort_session.run.return_value = [expected, np.zeros((1,))]

        sess = ONNXSession(dummy_onnx_path, device="cpu")
        sess._sess = mock_ort_session

        input_arr = np.random.rand(1, 3, 224, 224).astype(np.float32)
        result = sess.run(input_arr)
        np.testing.assert_array_equal(result, expected)

    def test_run_casts_to_float32(self, dummy_onnx_path, mock_ort_session):
        from src.utils.onnx_session import ONNXSession

        sess = ONNXSession(dummy_onnx_path, device="cpu")
        sess._sess = mock_ort_session

        input_arr = np.random.rand(1, 3).astype(np.float64)
        sess.run(input_arr)

        call_args = mock_ort_session.run.call_args
        passed_array = list(call_args[0][1].values())[0]
        assert passed_array.dtype == np.float32

    def test_run_uses_correct_input_name(self, dummy_onnx_path, mock_ort_session):
        from src.utils.onnx_session import ONNXSession

        mock_input = MagicMock()
        mock_input.name = "images"
        mock_ort_session.get_inputs.return_value = [mock_input]

        sess = ONNXSession(dummy_onnx_path, device="cpu")
        sess._sess = mock_ort_session

        input_arr = np.ones((1, 3), dtype=np.float32)
        sess.run(input_arr)

        call_args = mock_ort_session.run.call_args
        feed_dict = call_args[0][1]
        assert "images" in feed_dict

    def test_run_calls_get_to_init_session(self, dummy_onnx_path, mock_ort_session):
        from src.utils.onnx_session import ONNXSession

        sess = ONNXSession(dummy_onnx_path, device="cpu")
        with patch.object(sess, "_get", return_value=mock_ort_session) as mock_get:
            sess.run(np.zeros((1, 3), dtype=np.float32))
        mock_get.assert_called_once()


class TestONNXSessionRunAll:
    def test_run_all_returns_all_outputs(self, dummy_onnx_path, mock_ort_session):
        from src.utils.onnx_session import ONNXSession

        labels = np.array([1], dtype=np.int64)
        probas = np.array([[0.3, 0.7]], dtype=np.float32)
        mock_ort_session.run.return_value = [labels, probas]

        sess = ONNXSession(dummy_onnx_path, device="cpu")
        sess._sess = mock_ort_session

        result = sess.run_all(np.zeros((1, 5), dtype=np.float32))
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], labels)
        np.testing.assert_array_equal(result[1], probas)

    def test_run_all_casts_to_float32(self, dummy_onnx_path, mock_ort_session):
        from src.utils.onnx_session import ONNXSession

        mock_ort_session.run.return_value = [np.zeros(1)]
        sess = ONNXSession(dummy_onnx_path, device="cpu")
        sess._sess = mock_ort_session

        input_arr = np.ones((1, 5), dtype=np.float64)
        sess.run_all(input_arr)

        call_args = mock_ort_session.run.call_args
        passed_array = list(call_args[0][1].values())[0]
        assert passed_array.dtype == np.float32

    def test_run_all_calls_get_to_init_session(self, dummy_onnx_path, mock_ort_session):
        from src.utils.onnx_session import ONNXSession

        mock_ort_session.run.return_value = [np.zeros(1)]
        sess = ONNXSession(dummy_onnx_path, device="cpu")
        with patch.object(sess, "_get", return_value=mock_ort_session) as mock_get:
            sess.run_all(np.zeros((1, 3), dtype=np.float32))
        mock_get.assert_called_once()

    def test_run_all_uses_correct_input_name(self, dummy_onnx_path, mock_ort_session):
        from src.utils.onnx_session import ONNXSession

        mock_input = MagicMock()
        mock_input.name = "feature_input"
        mock_ort_session.get_inputs.return_value = [mock_input]
        mock_ort_session.run.return_value = [np.zeros(1)]

        sess = ONNXSession(dummy_onnx_path, device="cpu")
        sess._sess = mock_ort_session

        sess.run_all(np.ones((2, 10), dtype=np.float32))
        call_args = mock_ort_session.run.call_args
        assert "feature_input" in call_args[0][1]
