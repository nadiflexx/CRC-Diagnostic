"""
POST /diagnosis/smoking-triage

Smoking risk triage via ReverseLogic tabular model.
Persists SmokingTriageResult + Visit(SMOKING_TRIAGE) linked to patient.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
import numpy as np
import onnxruntime as ort
import pandas as pd
from sqlalchemy.orm import Session

from src.api.schemas import SmokingTriageRequest, SmokingTriageResponse
from src.config.logger import log as logger
from src.config.paths import paths
from src.database.connection import get_db_dependency
from src.database.models import SmokingRiskEnum, VisitTypeEnum
from src.database.repositories import (
    PatientRepository,
    SmokingTriageRepository,
    VisitRepository,
)

router = APIRouter(prefix="/diagnosis", tags=["Smoking Triage"])

_smoking_model: dict | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────


def _get_smoking_model() -> dict | None:
    global _smoking_model
    if _smoking_model is not None:
        return _smoking_model
    models_path = paths.MODELS
    onnx_path = models_path / "onnx" / "reverse_logic_smk_stat_type_cd.onnx"
    json_path = models_path / "reverse_logic_smk_stat_type_cd.json"

    if onnx_path.exists() and json_path.exists():
        try:
            with open(json_path) as f:
                meta: dict = json.load(f)

            required = ("scaler_mean", "scaler_scale", "scaler_feature_names")
            missing = [k for k in required if k not in meta]
            if missing:
                raise ValueError(f"JSON missing scaler params: {missing}")

            scaler_mean = np.array(meta["scaler_mean"], dtype=np.float64)
            scaler_scale = np.array(meta["scaler_scale"], dtype=np.float64)
            scaler_feature_names = meta["scaler_feature_names"]
            categorical_features = meta.get("categorical_features", ["sex"])
            numeric_features = meta.get("numeric_features", scaler_feature_names)

            sess = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
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
            logger.info("✅ Smoking triage: ONNX-ML Runtime loaded")
            return _smoking_model

        except Exception as e:
            logger.warning(f"⚠️ ONNX smoking failed: {e} — trying .pkl fallback")

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
            logger.info("✅ Smoking triage: .pkl fallback loaded")
            return _smoking_model
        except Exception as e:
            logger.error(f"❌ .pkl fallback failed: {e}")

    logger.error("❌ Smoking triage: no model available")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────


def _build_onnx_input(
    row: dict,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    scaler_feature_names: list[str],
    categorical_features: list[str],
) -> np.ndarray:
    numeric_vals = np.array([row[f] for f in scaler_feature_names], dtype=np.float64)
    scaled = (numeric_vals - scaler_mean) / scaler_scale
    ohe_parts: list[np.ndarray] = []
    for cat_feat in categorical_features:
        val = int(row.get(cat_feat, 0))
        ohe = np.zeros(2, dtype=np.float64)
        ohe[val] = 1.0
        ohe_parts.append(ohe)
    ohe_array = (
        np.concatenate(ohe_parts) if ohe_parts else np.array([], dtype=np.float64)
    )
    return np.concatenate([scaled, ohe_array]).astype(np.float32).reshape(1, -1)


def _infer_onnx(m: dict, row: dict) -> tuple[float, float]:
    sess = m["session"]
    X = _build_onnx_input(
        row,
        m["scaler_mean"],
        m["scaler_scale"],
        m["scaler_feature_names"],
        m["categorical_features"],
    )
    outputs = sess.run(None, {sess.get_inputs()[0].name: X})
    if len(outputs) > 1:
        proba = outputs[1]
        if isinstance(proba, list):
            prob_yes = float(proba[0].get(1, 0.5))
        elif isinstance(proba, np.ndarray):
            prob_yes = float(proba[0]) if proba.ndim == 1 else float(proba[0, 1])
        else:
            prob_yes = 0.5
    else:
        prob_yes = 0.75 if int(outputs[0][0]) == 1 else 0.25
    return float(np.clip(prob_yes, 0.0, 1.0)), 1.0 - float(np.clip(prob_yes, 0.0, 1.0))


def _infer_pkl(m: dict, row: dict) -> tuple[float, float]:
    model = m["model"]
    all_feat = m["numeric_features"] + m["categorical_features"]
    df = pd.DataFrame([{f: row[f] for f in all_feat}])
    proba = model.predict_proba(df, model_type=m["pipeline_name"])
    return float(proba[0, 1]), float(proba[0, 0])


# ─────────────────────────────────────────────────────────────────────────────
# Business helpers
# ─────────────────────────────────────────────────────────────────────────────


def _compute_row(req: SmokingTriageRequest) -> dict:
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


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────────────────────────────────────


@router.post("/smoking-triage", response_model=SmokingTriageResponse)
def smoking_triage(
    req: SmokingTriageRequest,
    db: Session = Depends(get_db_dependency),  # noqa: B008
) -> SmokingTriageResponse:
    """
    Predict smoking risk from biometric markers.
    Creates SmokingTriageResult + Visit(SMOKING_TRIAGE) linked to patient.
    """
    patient = PatientRepository(db).get_by_id(req.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    smoking_model = _get_smoking_model()
    if smoking_model is None:
        raise HTTPException(
            status_code=503, detail="Smoking triage model not available"
        )

    row = _compute_row(req)
    try:
        if smoking_model["backend"] == "onnx":
            prob_yes, prob_no = _infer_onnx(smoking_model, row)
        else:
            prob_yes, prob_no = _infer_pkl(smoking_model, row)
    except Exception as e:
        logger.error(f"Smoking triage inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")  # noqa: B904

    level_str, color = _risk_level(prob_yes)
    indicators = _top_indicators(row)

    visit_id: int | None = None
    try:
        smoking_record = SmokingTriageRepository(db).create(
            sex=int(req.sex),
            age=float(req.age),
            height_cm=float(req.height),
            weight_kg=float(req.weight),
            bmi=float(req.BMI),
            waistline_cm=float(req.waistline),
            triglyceride_mg_dl=float(req.triglyceride),
            hdl_mg_dl=float(req.HDL_chole),
            ldl_mg_dl=float(req.LDL_chole),
            hemoglobin_g_dl=float(req.hemoglobin),
            waist_height_ratio=float(row["waist_height_ratio"]),
            hemoglobin_per_height=float(row["hemoglobin_per_height"]),
            smoking_probability=round(prob_yes, 4),
            non_smoking_probability=round(prob_no, 4),
            predicted_smoker=prob_yes >= 0.5,
            risk_level=SmokingRiskEnum(level_str),
            backend=smoking_model["backend"].upper(),
            top_indicators=indicators,
        )

        visit = VisitRepository(db).create(
            patient_id=req.patient_id,
            visit_type=VisitTypeEnum.SMOKING_TRIAGE,
            smoking_result_id=smoking_record.id,
            hemoglobin=float(req.hemoglobin),
        )
        visit_id = visit.id

        db.commit()
        logger.info(
            f"[SmokingTriage] patient={req.patient_id} "
            f"visit={visit_id} prob={prob_yes:.3f} level={level_str}"
        )
    except Exception as e:
        db.rollback()
        logger.error(f"[SmokingTriage] DB persist failed: {e}", exc_info=True)

    return SmokingTriageResponse(
        smoking_probability=round(prob_yes, 4),
        non_smoking_probability=round(prob_no, 4),
        predicted_smoker=prob_yes >= 0.5,
        risk_level=level_str,
        risk_color=color,
        confidence=f"{max(prob_yes, prob_no):.1%}",
        backend=smoking_model["backend"].upper(),
        top_indicators=indicators,
        recommendation=_recommendation(prob_yes),
        visit_id=visit_id,
    )
