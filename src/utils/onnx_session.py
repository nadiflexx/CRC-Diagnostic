"""
Wrapper for ONNX Runtime.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.config.logger import log as logger


class ONNXSession:
    """
    Wrapper light over ONNX Runtime

    Characteristics:
      - **Lazy init**: the session is not created until the first call to ``run()``.
      - **Provider auto-selection**: CUDA if available and requested, CPU as fallback.
      - **Uniform interface**: ``run()`` and ``run_all()`` accept numpy arrays.

    Args:
        onnx_path: path to the ``.onnx`` file.
        device:    ``"cuda"`` or ``"cpu"``. If CUDA is requested but not available
                   in the runtime, CPU will be used automatically without error.
    """

    def __init__(self, onnx_path: Path, device: str = "cpu"):
        self.path = onnx_path
        self.device = device
        self._sess: ort.InferenceSession | None = None

    @property
    def available(self) -> bool:
        """``True`` if the ``.onnx`` file exists on disk."""
        return self.path.exists()

    def _get(self) -> ort.InferenceSession:
        """
        Initializes the session on the first access (lazy init).

        Returns:
            ONNX Runtime session ready for inference.

        Raises:
            RuntimeError: if the file does not exist or the session cannot be created.
        """
        if self._sess is None:
            providers = self._select_providers()

            opts = ort.SessionOptions()
            opts.log_severity_level = 3  # silenciar warnings internos de ORT
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._sess = ort.InferenceSession(
                str(self.path),
                sess_options=opts,
                providers=providers,
            )
            logger.info(
                f"  ✅ ONNX Session: {self.path.name} [{self._sess.get_providers()[0]}]"
            )
        return self._sess

    def _select_providers(self) -> list[str]:
        """
        Selects available providers based on the requested device.

        Returns:
            List of providers in order of preference.
        """
        available = ort.get_available_providers()
        if self.device == "cuda" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def run(self, input_np: np.ndarray) -> np.ndarray:
        """
        Executes inference and returns the first output.

        Args:
            input_np: array float32.
                      For image models: shape ``(B, 3, H, W)``.
                      For tabular models: shape ``(B, N_features)``.

        Returns:
            Primer output del modelo, shape dependiente de la arquitectura.
            For classifiers: ``(B, num_classes)`` logits.
            For segmenters: ``(B, 1, H, W)`` logits.
        """
        sess = self._get()
        input_name = sess.get_inputs()[0].name
        return sess.run(None, {input_name: input_np.astype(np.float32)})[0]

    def run_all(self, input_np: np.ndarray) -> list[np.ndarray]:
        """
        Executes inference and returns all outputs.

        Useful for ONNX-ML models (sklearn/xgboost) that produce
        múltiple exits.

        Args:
            input_np: array float32, shape ``(B, N_features)``.

        Returns:
            Arraylist, one for output of the model.
            Tipically ``[labels_array, probas_array]``.
        """
        sess = self._get()
        input_name = sess.get_inputs()[0].name
        return sess.run(None, {input_name: input_np.astype(np.float32)})
