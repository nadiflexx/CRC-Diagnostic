"""
Utility functions for machine learning tasks.
"""

import base64

import cv2
import numpy as np

from src.config.constants import (
    BOOL_TRUE_STRINGS as _BOOL_TRUE_STRINGS,
    BOOLEAN_FEATURES as _BOOLEAN_FEATURES,
    CATEGORICAL_ENCODINGS as _CATEGORICAL_ENCODINGS,
    CLINICAL_DEFAULTS,
)
from src.config.logger import log as logger


def encode_patient_row(
    patient_data: dict,
    feature_names: list[str],
) -> np.ndarray:
    """
    Converts a dict of clinical data to a float32 vector for ONNX-ML.

    Handles three types of features:
      - **Boolean**: ``bool``, ``int`` or strings like ``"yes"``/``"no"`` → 0.0 / 1.0
      - **Categorical**: strings mapped to integers via ``_CATEGORICAL_ENCODINGS``
      - **Numerical**: direct conversion to ``float``

    Missing or ``None`` values are filled from ``CLINICAL_DEFAULTS`` or with 0.

    Args:
        patient_data:  dict with the patient's values, key = feature name.
        feature_names: ordered list of feature names as expected by the model.

    Returns:
        Array numpy of shape ``(1, len(feature_names))`` and dtype ``float32``.
    """
    row: list[float] = []

    for feat in feature_names:
        raw = patient_data.get(feat)

        if raw is None:
            raw = CLINICAL_DEFAULTS.get(feat, 0)
            logger.debug(f"  Feature '{feat}' es None → default={raw}")

        val = _convert_feature(feat, raw)
        row.append(val)

    return np.array([row], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _convert_feature(feat: str, raw) -> float:
    """
    Converts a single feature value to float.

    Args:
        feat: name of the feature.
        raw:  raw value (str, bool, int, float). Never None (already resolved upstream).

    Returns:
        Numeric value as float.
    """
    if feat in _BOOLEAN_FEATURES:
        return _to_bool_float(raw)
    if feat in _CATEGORICAL_ENCODINGS:
        return _to_categorical_float(feat, raw)
    return _to_numeric_float(feat, raw)


def _to_bool_float(raw) -> float:
    """Converts a boolean/string value to 0.0 or 1.0."""
    if raw is None:
        return 0.0
    if isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, str):
        return 1.0 if raw.lower().strip() in _BOOL_TRUE_STRINGS else 0.0
    try:
        return float(raw)
    except (ValueError, TypeError):
        return 0.0


def _to_categorical_float(feat: str, raw) -> float:
    """Converts a categorical string to its numeric code."""
    if raw is None:
        return 0.0
    if isinstance(raw, str):
        mapping = _CATEGORICAL_ENCODINGS[feat]
        code = mapping.get(raw.lower().strip())
        if code is None:
            logger.warning(f"  ⚠️  Feature '{feat}': valor '{raw}' no reconocido → 0")
            return 0.0
        return float(code)
    try:
        return float(raw)
    except (ValueError, TypeError):
        return 0.0


def _to_numeric_float(feat: str, raw) -> float:
    """Converts a numerical value to float, with warning if it fails."""
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (ValueError, TypeError):
        logger.warning(f"  ⚠️  Feature '{feat}': '{raw}' no convertible a float → 0")
        return 0.0


def numpy_to_base64(img_rgb: np.ndarray) -> str:
    """
    Converts a numpy RGB image to a base64 data URI PNG.

    Args:
        img_rgb: image in RGB format, shape (H, W, 3), dtype uint8.

    Returns:
        String with format ``data:image/png;base64,<datos>``,
        ready to use in HTML or as src of an image.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", img_bgr)
    return f"data:image/png;base64,{base64.b64encode(buf).decode()}"


def softmax_np(x: np.ndarray) -> np.ndarray:
    """
    Softmax stable implementation over a vector 1D.

    Args:
        x: vector of logits 1D, shape (C,).

    Returns:
        Vector of probabilities of the same shape as ``x``,
        with values in (0, 1) that sum to 1.
    """
    e = np.exp(x - np.max(x))
    return e / e.sum()
