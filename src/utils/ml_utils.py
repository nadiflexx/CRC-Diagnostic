"""
src/utils/ml_utils.py

Utilidades de ml reutilizables
"""

import base64

import cv2
import numpy as np

from src.config.constants import CLINICAL_DEFAULTS
from src.config.logger import log as logger

_CATEGORICAL_ENCODINGS: dict[str, dict[str, int]] = {
    "gender": {
        "male": 0,
        "m": 0,
        "masculino": 0,
        "female": 1,
        "f": 1,
        "femenino": 1,
    },
    "smoking_status": {
        "never": 0,
        "nunca": 0,
        "former": 1,
        "exfumador": 1,
        "current": 2,
        "fumador": 2,
    },
    "alcohol_consumption": {
        "none": 0,
        "ninguno": 0,
        "moderate": 1,
        "moderado": 1,
        "heavy": 2,
        "alto": 2,
    },
    "physical_activity": {
        "sedentary": 0,
        "sedentario": 0,
        "low": 1,
        "bajo": 1,
        "moderate": 2,
        "moderado": 2,
        "high": 3,
        "alto": 3,
    },
    "diet_type": {
        "western": 0,
        "occidental": 0,
        "mediterranean": 1,
        "mediterranea": 1,
        "vegetarian": 2,
        "vegetariana": 2,
    },
    "ethnicity": {
        "caucasian": 0,
        "caucasico": 0,
        "hispanic": 1,
        "hispanico": 1,
        "african": 2,
        "africano": 2,
        "asian": 3,
        "asiatico": 3,
        "other": 4,
        "otro": 4,
    },
}

_BOOLEAN_FEATURES: frozenset[str] = frozenset(
    {
        "family_history_ccr",
        "family_history_polyps",
        "family_history_lynch",
        "family_history_fap",
        "has_ibd",
        "has_diabetes_t2",
        "previous_polyps",
        "previous_cancer",
        "fobt_positive",
        "fit_positive",
    }
)

_BOOL_TRUE_STRINGS = frozenset({"yes", "true", "1", "sí", "si"})


def encode_patient_row(
    patient_data: dict,
    feature_names: list[str],
) -> np.ndarray:
    """
    Convierte un dict de datos clínicos a un vector float32 para ONNX-ML.

    Maneja tres tipos de features:
      - **Booleanas**: ``bool``, ``int`` o strings como ``"yes"``/``"no"`` → 0.0 / 1.0
      - **Categóricas**: strings mapeados a int via ``_CATEGORICAL_ENCODINGS``
      - **Numéricas**: conversión directa a ``float``

    Los valores ``None`` o ausentes se rellenan desde ``CLINICAL_DEFAULTS`` o con 0.

    Args:
        patient_data:  dict con los valores del paciente, clave = nombre de feature.
        feature_names: lista ordenada de nombres de features tal como espera el modelo.

    Returns:
        Array numpy de shape ``(1, len(feature_names))`` y dtype ``float32``.
    """
    row: list[float] = []

    for feat in feature_names:
        raw = patient_data.get(feat)

        # None explícito o clave ausente → usar default
        if raw is None:
            raw = CLINICAL_DEFAULTS.get(feat, 0)
            logger.debug(f"  Feature '{feat}' es None → default={raw}")

        val = _convert_feature(feat, raw)
        row.append(val)

    return np.array([row], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers internos
# ─────────────────────────────────────────────────────────────────────────────


def _convert_feature(feat: str, raw) -> float:
    """
    Convierte un único valor de feature a float.

    Args:
        feat: nombre de la feature.
        raw:  valor crudo (str, bool, int, float). Nunca None (ya resuelto upstream).

    Returns:
        Valor numérico como float.
    """
    if feat in _BOOLEAN_FEATURES:
        return _to_bool_float(raw)
    if feat in _CATEGORICAL_ENCODINGS:
        return _to_categorical_float(feat, raw)
    return _to_numeric_float(feat, raw)


def _to_bool_float(raw) -> float:
    """Convierte un valor booleano/string a 0.0 o 1.0."""
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
    """Convierte un string categórico a su código numérico."""
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
    """Convierte un valor numérico a float, con aviso si falla."""
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (ValueError, TypeError):
        logger.warning(f"  ⚠️  Feature '{feat}': '{raw}' no convertible a float → 0")
        return 0.0


def numpy_to_base64(img_rgb: np.ndarray) -> str:
    """
    Convierte una imagen RGB numpy a data URI base64 PNG.

    Args:
        img_rgb: imagen en formato RGB, shape (H, W, 3), dtype uint8.

    Returns:
        String con formato ``data:image/png;base64,<datos>``,
        listo para usar en HTML o como src de una imagen.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", img_bgr)
    return f"data:image/png;base64,{base64.b64encode(buf).decode()}"


def softmax_np(x: np.ndarray) -> np.ndarray:
    """
    Softmax numéricamente estable sobre un vector 1D.

    Resta el máximo antes de exponenciar para evitar overflow,
    lo que es equivalente matemáticamente pero más estable.

    Args:
        x: vector de logits 1D, shape (C,).

    Returns:
        Vector de probabilidades de la misma shape que ``x``,
        con valores en (0, 1) que suman 1.
    """
    e = np.exp(x - np.max(x))
    return e / e.sum()
