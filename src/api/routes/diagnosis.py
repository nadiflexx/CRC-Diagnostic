"""
POST /diagnosis/run     — full AI pipeline (image + tumoral + fusion)
GET  /diagnosis/history/{patient_id}
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.schemas import DiagnosisRequest, DiagnosisResponse
from src.database.connection import get_db_dependency
from src.database.models import (
    DiagnosisResultEnum,
    ImageClassEnum,
    VisitTypeEnum,
)
from src.database.repositories import (
    ImageAnalysisRepository,
    PatientRepository,
    TumorAnalysisRepository,
    VisitRepository,
)
from src.diagnosis.engine import get_diagnosis_engine

router = APIRouter(prefix="/diagnosis", tags=["Diagnosis"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _score_to_diagnosis(score: float) -> DiagnosisResultEnum:
    if score >= 0.5:
        return DiagnosisResultEnum.POSITIVE
    if score >= 0.3:
        return DiagnosisResultEnum.SUSPICIOUS
    return DiagnosisResultEnum.NEGATIVE


def _safe_round(val, digits: int = 4):
    return round(val, digits) if val is not None else None


def _has_tumoral_inputs(clinical) -> bool:
    """
    Return True only when the user explicitly supplied tumoral model inputs.

    The engine may return a tabular_analysis even for image-only requests
    (using patient context data). We must NOT persist that as a
    TumorAnalysisResult — it would create an all-null row.

    A tumoral visit requires at least one of:
      • cea_level_ng_ml  (or the legacy cea alias)
      • any pyrad_* radiomic feature
    """
    has_cea = bool(clinical.cea_level_ng_ml or clinical.cea)
    has_radiomics = any(
        [
            clinical.pyrad_adc_mean,
            clinical.pyrad_adc_std,
            clinical.pyrad_entropy,
            clinical.pyrad_glcm_contrast,
            clinical.pyrad_glcm_homogeneity,
            clinical.pyrad_shape_sphericity,
            clinical.pyrad_firstorder_skewness,
        ]
    )
    return has_cea or has_radiomics


def _derive_visit_type(has_image: bool, has_tumor: bool) -> VisitTypeEnum:
    if has_image and has_tumor:
        return VisitTypeEnum.FULL
    if has_image:
        return VisitTypeEnum.ENDOSCOPY
    if has_tumor:
        return VisitTypeEnum.TUMOR_ANALYSIS
    return VisitTypeEnum.FULL


# ─────────────────────────────────────────────────────────────────────────────
# POST /diagnosis/run
# ─────────────────────────────────────────────────────────────────────────────


@router.post("/run", response_model=DiagnosisResponse)
def run_diagnosis(
    req: DiagnosisRequest,
    db: Session = Depends(get_db_dependency),  # noqa: B008
) -> DiagnosisResponse:
    """Execute full AI pipeline and persist results atomically."""

    patient = PatientRepository(db).get_by_id(req.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    age = (date.today() - patient.date_of_birth).days // 365

    patient_dict: dict = {
        "age": age,
        "gender": patient.gender.value if patient.gender else "unknown",
        "bmi": patient.bmi,
        "smoking_status": patient.smoking_status,
        "alcohol_consumption": patient.alcohol_consumption,
        "family_history_ccr": patient.family_history_ccr,
        "has_ibd": patient.has_ibd,
    }
    patient_dict.update(req.clinical_data.model_dump(exclude_none=True))

    if not patient_dict.get("age_value"):
        patient_dict["age_value"] = float(age)

    if patient_dict.get("smoking_history") is None:
        smoking_str = patient.smoking_status or "never"
        patient_dict["smoking_history"] = (
            1 if smoking_str in ("current", "former") else 0
        )

    engine = get_diagnosis_engine()
    result = engine.diagnose(
        patient_data=patient_dict,
        image_path=req.image_path,
        patient_id=req.patient_id,
    )

    clinical = req.clinical_data

    has_image = bool(result.get("image_analysis"))
    has_tumor = _has_tumoral_inputs(clinical) and bool(result.get("tabular_analysis"))

    image_result_id: int | None = None

    if has_image:
        ia = result["image_analysis"]
        raw_cls = (ia.get("prediction_class") or "normal").lower()
        try:
            img_class = ImageClassEnum(raw_cls)
        except ValueError:
            img_class = ImageClassEnum.NORMAL

        img_record = ImageAnalysisRepository(db).create(
            colonoscopy_image_path=req.image_path,
            prediction_class=img_class,
            prediction_score=_safe_round(ia.get("prediction_score", 0.0)),
            probabilities={
                k: _safe_round(v) for k, v in ia.get("probabilities", {}).items()
            },
            ensemble_mode=ia.get("ensemble_mode"),
            ensemble_used=bool(ia.get("ensemble_used", False)),
            alpha_used=_safe_round(ia.get("alpha_used", 0.0), 3),
            beta_used=_safe_round(ia.get("beta_used", 0.0), 3),
            attention_ratio=_safe_round(ia.get("attention_ratio"), 3),
            polyp_detected=bool(ia.get("polyp_detected", False)),
            lesion_count=int(ia.get("lesion_count", 0)),
            gradcam_a_path=ia.get("gradcam_a_path"),
            gradcam_b_path=ia.get("gradcam_b_path"),
            gradcam_fusion_path=ia.get("gradcam_fusion_path"),
            segmentation_mask_path=ia.get("segmentation_mask_path"),
            segmentation_report_path=ia.get("report_path"),
        )
        image_result_id = img_record.id

    tumor_result_id: int | None = None

    if has_tumor:
        ta = result["tabular_analysis"]
        c = clinical

        tumor_record = TumorAnalysisRepository(db).create(
            age_value=c.age_value,
            smoking_history=c.smoking_history,
            cea_level_ng_ml=c.cea_level_ng_ml or c.cea,
            hemoglobin_g_dl=c.hemoglobin_g_dl or c.hemoglobin,
            pyrad_adc_mean=c.pyrad_adc_mean,
            pyrad_adc_std=c.pyrad_adc_std,
            pyrad_entropy=c.pyrad_entropy,
            pyrad_glcm_contrast=c.pyrad_glcm_contrast,
            pyrad_glcm_homogeneity=c.pyrad_glcm_homogeneity,
            pyrad_shape_sphericity=c.pyrad_shape_sphericity,
            pyrad_firstorder_skewness=c.pyrad_firstorder_skewness,
            prediction_score=_safe_round(ta.get("prediction_score", 0.0)),
            high_risk=bool(ta.get("high_risk", False)),
            top_risk_factors=ta.get("top_risk_factors", []),
        )
        tumor_result_id = tumor_record.id

    risk_score: float = 0.0
    if result.get("risk_level"):
        risk_score = result["risk_level"].get("score", 0.0)

    multimodal_score: float | None = None
    if result.get("multimodal_result"):
        multimodal_score = _safe_round(
            result["multimodal_result"].get("combined_score")
        )

    VisitRepository(db).create(
        patient_id=req.patient_id,
        visit_type=_derive_visit_type(has_image, has_tumor),
        notes=clinical.notes,
        hemoglobin=clinical.hemoglobin or clinical.hemoglobin_g_dl,
        hematocrit=clinical.hematocrit,
        wbc_count=clinical.wbc_count,
        platelet_count=clinical.platelet_count,
        albumin=clinical.albumin,
        iron_serum=clinical.iron_serum,
        ferritin=clinical.ferritin,
        crp=clinical.crp,
        cea=clinical.cea or clinical.cea_level_ng_ml,
        ca19_9=clinical.ca19_9,
        fobt_positive=clinical.fobt_positive,
        fit_positive=clinical.fit_positive,
        multimodal_prediction_score=multimodal_score,
        diagnosis_result=_score_to_diagnosis(risk_score),
        confidence_score=result.get("confidence"),
        image_result_id=image_result_id,
        tumor_result_id=tumor_result_id,
        smoking_result_id=None,
    )

    db.commit()

    result["patient_id"] = req.patient_id
    return DiagnosisResponse(**result)


# ─────────────────────────────────────────────────────────────────────────────
# GET /diagnosis/history/{patient_id}
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/history/{patient_id}")
def get_history(
    patient_id: int,
    db: Session = Depends(get_db_dependency),  # noqa: B008
) -> list[dict]:
    visits = VisitRepository(db).get_patient_visits(patient_id)
    rows: list[dict] = []

    for v in visits:
        image_snap: dict | None = None
        if v.image_result is not None:
            ir = v.image_result
            image_snap = {
                "prediction_class": ir.prediction_class.value
                if ir.prediction_class
                else None,
                "prediction_score": ir.prediction_score,
                "probabilities": ir.probabilities or {},
                "ensemble_mode": ir.ensemble_mode,
                "ensemble_used": ir.ensemble_used,
                "alpha_used": ir.alpha_used,
                "beta_used": ir.beta_used,
                "attention_ratio": ir.attention_ratio,
                "polyp_detected": ir.polyp_detected,
                "lesion_count": ir.lesion_count,
                "has_gradcam": bool(ir.gradcam_fusion_path or ir.gradcam_a_path),
                "has_segmentation": bool(ir.segmentation_mask_path),
            }

        tumor_snap: dict | None = None
        if v.tumor_result is not None:
            tr = v.tumor_result
            tumor_snap = {
                "prediction_score": tr.prediction_score,
                "high_risk": tr.high_risk,
                "top_risk_factors": tr.top_risk_factors or [],
                "pyrad_adc_mean": tr.pyrad_adc_mean,
                "pyrad_entropy": tr.pyrad_entropy,
                "pyrad_glcm_homogeneity": tr.pyrad_glcm_homogeneity,
                "pyrad_shape_sphericity": tr.pyrad_shape_sphericity,
                "cea_level_ng_ml": tr.cea_level_ng_ml,
            }

        smoking_snap: dict | None = None
        if v.smoking_result is not None:
            sr = v.smoking_result
            smoking_snap = {
                "smoking_probability": sr.smoking_probability,
                "predicted_smoker": sr.predicted_smoker,
                "risk_level": sr.risk_level.value if sr.risk_level else None,
                "backend": sr.backend,
                "top_indicators": sr.top_indicators or [],
            }

        ai_snapshot: dict = {}
        if image_snap:
            ai_snapshot["image"] = image_snap
        if tumor_snap:
            ai_snapshot["tabular"] = tumor_snap
        if smoking_snap:
            ai_snapshot["smoking"] = smoking_snap
        if v.multimodal_prediction_score is not None:
            ai_snapshot["fusion"] = {
                "combined_score": v.multimodal_prediction_score,
                "final_diagnosis": v.diagnosis_result.value
                if v.diagnosis_result
                else None,
            }

        rows.append(
            {
                "id": v.id,
                "date": v.visit_date.isoformat(),
                "visit_type": v.visit_type.value if v.visit_type else None,
                "hemoglobin": v.hemoglobin,
                "hematocrit": v.hematocrit,
                "wbc_count": v.wbc_count,
                "platelet_count": v.platelet_count,
                "albumin": v.albumin,
                "iron_serum": v.iron_serum,
                "ferritin": v.ferritin,
                "crp": v.crp,
                "cea": v.cea,
                "ca19_9": v.ca19_9,
                "fobt_positive": v.fobt_positive,
                "fit_positive": v.fit_positive,
                "image_score": v.image_result.prediction_score
                if v.image_result
                else None,
                "tabular_score": v.tumor_result.prediction_score
                if v.tumor_result
                else None,
                "multimodal_score": v.multimodal_prediction_score,
                "diagnosis": v.diagnosis_result.value if v.diagnosis_result else None,
                "confidence": v.confidence_score,
                "colonoscopy_performed": v.image_result_id is not None,
                "polyps_found": v.image_result.polyp_detected
                if v.image_result
                else None,
                "smoking_result_id": v.smoking_result_id,
                "tumor_result_id": v.tumor_result_id,
                "image_result_id": v.image_result_id,
                "ai_snapshot": ai_snapshot if ai_snapshot else None,
            }
        )

    return rows
