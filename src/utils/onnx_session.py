"""
src/utils/onnx_session.py

Wrapper sobre onnxruntime.InferenceSession para uso en el engine
y cualquier otro módulo que necesite inferencia ONNX.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.config.logger import log as logger


class ONNXSession:
    """
    Wrapper ligero sobre ``onnxruntime.InferenceSession``.

    Características:
      - **Lazy init**: la sesión no se crea hasta la primera llamada a ``run()``.
      - **Provider auto-selección**: CUDA si disponible y solicitado, CPU como fallback.
      - **Interface uniforme**: ``run()`` y ``run_all()`` aceptan arrays numpy.

    Args:
        onnx_path: ruta al archivo ``.onnx``.
        device:    ``"cuda"`` o ``"cpu"``. Si se pide CUDA pero no está disponible
                   en el runtime, se usa CPU automáticamente sin error.
    """

    def __init__(self, onnx_path: Path, device: str = "cpu"):
        self.path = onnx_path
        self.device = device
        self._sess: ort.InferenceSession | None = None

    @property
    def available(self) -> bool:
        """``True`` si el archivo ``.onnx`` existe en disco."""
        return self.path.exists()

    def _get(self) -> ort.InferenceSession:
        """
        Inicializa la sesión en el primer acceso (lazy init).

        Returns:
            Sesión ONNX Runtime lista para inferencia.

        Raises:
            RuntimeError: si el archivo no existe o la sesión no se puede crear.
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
        Selecciona providers disponibles según el device solicitado.

        Returns:
            Lista de providers en orden de preferencia.
        """
        available = ort.get_available_providers()
        if self.device == "cuda" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def run(self, input_np: np.ndarray) -> np.ndarray:
        """
        Ejecuta inferencia y devuelve el primer output.

        Args:
            input_np: array float32.
                      Para modelos de imagen: shape ``(B, 3, H, W)``.
                      Para modelos tabulares: shape ``(B, N_features)``.

        Returns:
            Primer output del modelo, shape dependiente de la arquitectura.
            Para clasificadores: ``(B, num_classes)`` logits.
            Para segmentadores: ``(B, 1, H, W)`` logits.
        """
        sess = self._get()
        input_name = sess.get_inputs()[0].name
        return sess.run(None, {input_name: input_np.astype(np.float32)})[0]

    def run_all(self, input_np: np.ndarray) -> list[np.ndarray]:
        """
        Ejecuta inferencia y devuelve todos los outputs.

        Útil para modelos ONNX-ML (sklearn/xgboost) que producen
        múltiples salidas: ``[labels, probabilities]``.

        Args:
            input_np: array float32, shape ``(B, N_features)``.

        Returns:
            Lista de arrays, uno por output del modelo.
            Típicamente ``[labels_array, probas_array]``.
        """
        sess = self._get()
        input_name = sess.get_inputs()[0].name
        return sess.run(None, {input_name: input_np.astype(np.float32)})
