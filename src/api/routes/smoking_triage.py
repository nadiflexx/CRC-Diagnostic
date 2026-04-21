"""
POST /diagnosis/smoking-triage

Triaje de riesgo tabáquico mediante ReverseLogicTabularModel (ONNX-ML).
Preprocessing manual desde parámetros guardados en el JSON de metadata.
Fallback a .pkl si ONNX no está disponible.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd

from src.api.schemas import SmokingTriageRequest, SmokingTriageResponse
from src.config.logger import log as logger
from src.config.paths import paths

router = APIRouter(prefix="/diagnosis", tags=["Smoking Triage"])

_smoking_model: dict | None = None


def _get_smoking_model() -> dict | None:
    """
    Carga el modelo una sola vez (singleton).

    Estrategia:
      1. ONNX-ML + preprocessing manual desde JSON
         El JSON contiene scaler_mean, scaler_scale, scaler_feature_names
         y categorical_features. Con eso reconstruimos el input [1, 13]
         sin tocar ningún objeto sklearn.

      2. .pkl fallback → ReverseLogicTabularModel completo.
    """
    global _smoking_model
    if _smoking_model is not None:
        return _smoking_model

    onnx_path = paths.MODELS / "onnx" / "reverse_logic_smk_stat_type_cd.onnx"
    json_path = onnx_path.with_suffix(".json")

    if onnx_path.exists() and json_path.exists():
        try:
            import onnxruntime as ort

            with open(json_path) as f:
                meta: dict = json.load(f)

            required = ("scaler_mean", "scaler_scale", "scaler_feature_names")
            missing = [k for k in required if k not in meta]
            if missing:
                raise ValueError(
                    f"JSON sin parámetros de scaler: {missing}. "
                    f"Re-exporta el modelo con export_onnx.py"
                )

            scaler_mean = np.array(meta["scaler_mean"], dtype=np.float64)
            scaler_scale = np.array(meta["scaler_scale"], dtype=np.float64)
            scaler_feature_names: list[str] = meta["scaler_feature_names"]
            categorical_features: list[str] = meta.get("categorical_features", ["sex"])
            numeric_features: list[str] = meta.get(
                "numeric_features", scaler_feature_names
            )

            sess = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )

            _test_X = _build_onnx_input_from_params(
                row={**dict.fromkeys(numeric_features, 1.0), "sex": 0.0},
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                scaler_feature_names=scaler_feature_names,
                categorical_features=categorical_features,
            )
            _test_out = sess.run(None, {sess.get_inputs()[0].name: _test_X})
            logger.info(
                f"  ✅ ONNX test OK: "
                f"input={_test_X.shape} "
                f"outputs={[o.shape for o in _test_out]}"
            )

            _smoking_model = {
                "backend": "onnx",
                "session": sess,
                "scaler_mean": scaler_mean,
                "scaler_scale": scaler_scale,
                "scaler_feature_names": scaler_feature_names,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "n_features_transformed": int(meta.get("n_features_transformed", 13)),
                "label_classes": meta.get("label_encoders", {}).get(
                    "xgboost", ["No", "Yes"]
                ),
                "metrics": meta.get("metrics", {}),
                "exported_pipeline": meta.get("exported_pipeline", "xgboost"),
            }

            logger.info(
                f"✅ Smoking triage: ONNX-ML Runtime "
                f"(pipeline='{_smoking_model['exported_pipeline']}', "
                f"n_transformed={_smoking_model['n_features_transformed']})"
            )
            return _smoking_model

        except Exception as e:
            logger.warning(f"⚠️  ONNX smoking failed: {e} — intentando fallback .pkl")

    pkl_path = paths.REVERSE_LOGIC_MODEL_DIR / "reverse_logic_smk_stat_type_cd.pkl"
    if pkl_path.exists():
        try:
            from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

            model = ReverseLogicTabularModel.load(pkl_path)

            best_name = max(
                model.metrics,
                key=lambda k: model.metrics[k].get("f1", 0.0),
                default=next(iter(model.pipelines), "xgboost"),
            )

            _smoking_model = {
                "backend": "pkl",
                "model": model,
                "pipeline_name": best_name,
                "numeric_features": model.numeric_features,
                "categorical_features": model.categorical_features,
                "metrics": model.metrics,
            }
            logger.info(f"✅ Smoking triage: .pkl fallback (pipeline='{best_name}')")
            return _smoking_model

        except Exception as e:
            logger.error(f"❌ .pkl fallback failed: {e}")

    logger.error("❌ Smoking triage: ningún modelo disponible")
    return None


# ═════════════════════════════════════════════════════════════════════════════
# PREPROCESSING MANUAL
# ═════════════════════════════════════════════════════════════════════════════


def _build_onnx_input_from_params(
    row: dict,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    scaler_feature_names: list[str],
    categorical_features: list[str],
) -> np.ndarray:
    """
    Replica el ColumnTransformer manualmente usando solo numpy.

    El ColumnTransformer entrenado hace:
      1. StandardScaler sobre numeric_features  → n columnas escaladas
      2. OneHotEncoder sobre categorical_features → m columnas OHE

    Para este modelo concreto:
      - 11 numéricas → StandardScaler → 11 escaladas
      - 1 categórica (sex: 0=Female, 1=Male) → OHE → 2 cols [Female, Male]
      Total: 13 features transformadas

    Args:
        row:                   dict con nombres brutos y valores float
        scaler_mean:           mean_ del StandardScaler  shape (11,)
        scaler_scale:          scale_ del StandardScaler shape (11,)
        scaler_feature_names:  nombres en orden del StandardScaler
        categorical_features:  nombres de features categóricas (["sex"])

    Returns:
        np.ndarray float32 shape (1, 13)
    """
    numeric_vals = np.array(
        [row[f] for f in scaler_feature_names],
        dtype=np.float64,
    )
    scaled = (numeric_vals - scaler_mean) / scaler_scale

    ohe_parts: list[np.ndarray] = []
    for cat_feat in categorical_features:
        val = int(row.get(cat_feat, 0))
        if cat_feat == "sex":
            ohe = np.zeros(2, dtype=np.float64)
            ohe[val] = 1.0
            ohe_parts.append(ohe)
        else:
            ohe = np.array([1.0 - val, float(val)], dtype=np.float64)
            ohe_parts.append(ohe)

    ohe_array = (
        np.concatenate(ohe_parts) if ohe_parts else np.array([], dtype=np.float64)
    )

    X = np.concatenate([scaled, ohe_array]).astype(np.float32).reshape(1, -1)

    return X


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCIA
# ═════════════════════════════════════════════════════════════════════════════


def _infer_onnx(smoking_model: dict, row: dict) -> tuple[float, float]:
    """
    Inferencia ONNX con preprocessing manual puro numpy.
    No usa ningún objeto sklearn en runtime.
    """
    sess = smoking_model["session"]
    input_name = sess.get_inputs()[0].name

    X = _build_onnx_input_from_params(
        row=row,
        scaler_mean=smoking_model["scaler_mean"],
        scaler_scale=smoking_model["scaler_scale"],
        scaler_feature_names=smoking_model["scaler_feature_names"],
        categorical_features=smoking_model["categorical_features"],
    )

    logger.debug(f"  ONNX input shape={X.shape} dtype={X.dtype} values={X[0][:5]}...")

    outputs = sess.run(None, {input_name: X})

    if len(outputs) > 1:
        proba = outputs[1]
        if isinstance(proba, list):
            prob_yes = float(proba[0].get(1, 0.5))
        elif isinstance(proba, np.ndarray):
            prob_yes = float(proba[0]) if proba.ndim == 1 else float(proba[0, 1])
        else:
            prob_yes = 0.5
    else:
        pred_int = int(outputs[0][0])
        prob_yes = 0.75 if pred_int == 1 else 0.25

    prob_yes = float(np.clip(prob_yes, 0.0, 1.0))
    return prob_yes, 1.0 - prob_yes


def _infer_pkl(smoking_model: dict, row: dict) -> tuple[float, float]:
    """
    Inferencia con ReverseLogicTabularModel completo (.pkl).
    El pipeline incluye preprocessing → sin problemas de tipos.
    """
    model = smoking_model["model"]
    pipeline_name = smoking_model["pipeline_name"]
    numeric_features: list[str] = smoking_model["numeric_features"]
    categorical_features: list[str] = smoking_model["categorical_features"]
    all_features = numeric_features + categorical_features

    df = pd.DataFrame([{f: row[f] for f in all_features}])
    proba = model.predict_proba(df, model_type=pipeline_name)

    return float(proba[0, 1]), float(proba[0, 0])


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS DE NEGOCIO
# ═════════════════════════════════════════════════════════════════════════════


def _compute_row(req: SmokingTriageRequest) -> dict:
    """Construye el dict de entrada con features derivadas calculadas."""
    whr = (
        req.waist_height_ratio
        if req.waist_height_ratio > 0
        else req.waistline / req.height
    )
    hph = (
        req.hemoglobin_per_height
        if req.hemoglobin_per_height > 0
        else req.hemoglobin / req.height
    )
    return {
        "sex": float(req.sex),
        "age": float(req.age),
        "height": float(req.height),
        "weight": float(req.weight),
        "BMI": float(req.BMI),
        "waistline": float(req.waistline),
        "triglyceride": float(req.triglyceride),
        "HDL_chole": float(req.HDL_chole),
        "LDL_chole": float(req.LDL_chole),
        "hemoglobin": float(req.hemoglobin),
        "waist_height_ratio": whr,
        "hemoglobin_per_height": hph,
    }


def _risk_level(prob: float) -> tuple[str, str]:
    if prob >= 0.70:
        return "HIGH", "#D32F2F"
    if prob >= 0.45:
        return "MODERATE", "#F57C00"
    return "LOW", "#388E3C"


def _top_indicators(row: dict) -> list[dict]:
    return [
        {
            "name": "Hemoglobina / Altura",
            "value": f"{row['hemoglobin_per_height']:.4f}",
            "flag": row["hemoglobin_per_height"] > 0.088,
            "note": "Elevado en fumadores (policitemia relativa)",
        },
        {
            "name": "Ratio Cintura / Altura",
            "value": f"{row['waist_height_ratio']:.4f}",
            "flag": row["waist_height_ratio"] > 0.53,
            "note": "Distribución adiposa central asociada al tabaquismo",
        },
        {
            "name": "Hemoglobina",
            "value": f"{row['hemoglobin']:.2f} g/dL",
            "flag": row["hemoglobin"] > 16.0,
            "note": "Hemoconcentración frecuente en fumadores activos",
        },
        {
            "name": "Cintura",
            "value": f"{row['waistline']:.1f} cm",
            "flag": (
                row["waistline"] > 94 if row["sex"] == 1.0 else row["waistline"] > 80
            ),
            "note": "Obesidad abdominal ligada al tabaquismo crónico",
        },
        {
            "name": "Triglicéridos",
            "value": f"{row['triglyceride']:.1f} mg/dL",
            "flag": row["triglyceride"] > 150,
            "note": "Hipertrigliceridemia asociada al tabaquismo",
        },
        {
            "name": "HDL-Colesterol",
            "value": f"{row['HDL_chole']:.1f} mg/dL",
            "flag": row["HDL_chole"] < 40,
            "note": "HDL bajo: el tabaco reduce el colesterol protector",
        },
    ]


def _recommendation(prob: float) -> str:
    if prob >= 0.70:
        return (
            "El perfil biométrico presenta múltiples indicadores compatibles con "
            "tabaquismo activo. Se recomienda confirmación anamnésica prioritaria "
            "y valoración de intervención para deshabituación tabáquica antes de "
            "proceder con el cribado endoscópico."
        )
    if prob >= 0.45:
        return (
            "El perfil sugiere posible exposición tabáquica. Se recomienda "
            "indagación en la anamnesis sobre historial de consumo y valorar "
            "adelantar el intervalo de cribado endoscópico según protocolo "
            "institucional."
        )
    return (
        "El perfil biométrico no presenta indicadores significativos de "
        "tabaquismo. Continuar con el protocolo de cribado estándar según "
        "edad y factores de riesgo convencionales."
    )


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT
# ═════════════════════════════════════════════════════════════════════════════


@router.post("/smoking-triage", response_model=SmokingTriageResponse)
def smoking_triage(req: SmokingTriageRequest) -> SmokingTriageResponse:
    """
    Predice riesgo de tabaquismo desde biomarcadores clínicos.
    No requiere declaración del hábito tabáquico por parte del paciente.
    """
    smoking_model = _get_smoking_model()
    if smoking_model is None:
        raise HTTPException(503, "Smoking triage model not available")

    row = _compute_row(req)

    try:
        if smoking_model["backend"] == "onnx":
            prob_yes, prob_no = _infer_onnx(smoking_model, row)
        else:
            prob_yes, prob_no = _infer_pkl(smoking_model, row)
    except Exception as e:
        logger.error(f"Smoking triage inference error: {e}", exc_info=True)
        raise HTTPException(500, f"Inference failed: {e}")  # noqa: B904

    level, color = _risk_level(prob_yes)

    logger.info(
        f"[SmokingTriage] backend={smoking_model['backend']} "
        f"prob_smoke={prob_yes:.3f} level={level}"
    )

    return SmokingTriageResponse(
        smoking_probability=round(prob_yes, 4),
        non_smoking_probability=round(prob_no, 4),
        predicted_smoker=prob_yes >= 0.5,
        risk_level=level,
        risk_color=color,
        confidence=f"{max(prob_yes, prob_no):.1%}",
        backend=smoking_model["backend"].upper(),
        top_indicators=_top_indicators(row),
        recommendation=_recommendation(prob_yes),
    )
