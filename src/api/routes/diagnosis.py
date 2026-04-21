from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.schemas import DiagnosisRequest, DiagnosisResponse
from src.database.connection import get_db_dependency
from src.database.models import DiagnosisResultEnum
from src.database.repositories import PatientRepository, VisitRepository
from src.diagnosis.engine import get_diagnosis_engine

router = APIRouter(prefix="/diagnosis", tags=["Diagnosis"])


@router.post("/run", response_model=DiagnosisResponse)
def run_diagnosis(req: DiagnosisRequest, db: Session = Depends(get_db_dependency)):  # noqa: B008
    """Run a diagnosis for the given patient."""
    if db is None:
        db = Depends(get_db_dependency)

    patient = PatientRepository(db).get_by_id(req.patient_id)
    if not patient:
        raise HTTPException(404, "Patient not found")

    age = (date.today() - patient.date_of_birth).days // 365

    patient_dict = {
        "age": age,
        "gender": patient.gender.value if patient.gender else "unknown",
        "bmi": patient.bmi,
        "smoking_status": patient.smoking_status,
        "alcohol_consumption": patient.alcohol_consumption,
        "family_history_ccr": patient.family_history_ccr,
        "has_ibd": patient.has_ibd,
    }

    patient_dict.update(req.clinical_data.model_dump(exclude_none=True))

    if "age_value" not in patient_dict or patient_dict.get("age_value") is None:
        patient_dict["age_value"] = float(age)

    if (
        "smoking_history" not in patient_dict
        or patient_dict.get("smoking_history") is None
    ):
        smoking_status = patient.smoking_status or "never"
        patient_dict["smoking_history"] = (
            1 if smoking_status in ("current", "former") else 0
        )

    engine = get_diagnosis_engine()
    result = engine.diagnose(
        patient_data=patient_dict,
        image_path=req.image_path,
        patient_id=req.patient_id,
    )

    visit_repo = VisitRepository(db)
    clinical = req.clinical_data

    ai_snapshot: dict = {}

    if result.get("image_analysis"):
        ia = result["image_analysis"]
        ai_snapshot["image"] = {
            "prediction_class": ia.get("prediction_class"),
            "prediction_score": round(ia.get("prediction_score", 0), 4),
            "probabilities": {
                k: round(v, 4) for k, v in ia.get("probabilities", {}).items()
            },
            "ensemble_mode": ia.get("ensemble_mode"),
            "ensemble_used": ia.get("ensemble_used"),
            "alpha_used": round(ia.get("alpha_used", 0), 3),
            "beta_used": round(ia.get("beta_used", 0), 3),
            "attention_ratio": (
                round(ia["attention_ratio"], 3)
                if ia.get("attention_ratio") is not None
                else None
            ),
            "polyp_detected": ia.get("polyp_detected", False),
            "lesion_count": ia.get("lesion_count", 0),
        }

    if result.get("tabular_analysis"):
        ta = result["tabular_analysis"]
        ai_snapshot["tabular"] = {
            "prediction_score": round(ta.get("prediction_score", 0), 4),
            "high_risk": ta.get("high_risk", False),
            "top_risk_factors": ta.get("top_risk_factors", []),
        }

    if result.get("multimodal_result"):
        rl = result.get("risk_level", {})
        ai_snapshot["fusion"] = {
            "combined_score": round(
                result["multimodal_result"].get("combined_score", 0), 4
            ),
            "final_diagnosis": result.get("final_diagnosis", ""),
            "risk_level": rl.get("level", "LOW"),
        }

    visit_data = {
        "hemoglobin": clinical.hemoglobin or clinical.hemoglobin_g_dl,
        "hematocrit": clinical.hematocrit,
        "wbc_count": clinical.wbc_count,
        "platelet_count": clinical.platelet_count,
        "albumin": clinical.albumin,
        "iron_serum": clinical.iron_serum,
        "ferritin": clinical.ferritin,
        "crp": clinical.crp,
        "cea": clinical.cea or clinical.cea_level_ng_ml,
        "ca19_9": clinical.ca19_9,
        "fobt_positive": clinical.fobt_positive,
        "fit_positive": clinical.fit_positive,
        "notes": clinical.notes,
        "image_prediction_score": (
            result["image_analysis"]["prediction_score"]
            if result.get("image_analysis")
            else None
        ),
        "tabular_prediction_score": (
            result["tabular_analysis"]["prediction_score"]
            if result.get("tabular_analysis")
            else None
        ),
        "multimodal_prediction_score": (
            result["multimodal_result"]["combined_score"]
            if result.get("multimodal_result")
            else None
        ),
        "confidence_score": result.get("confidence"),
        "colonoscopy_performed": req.image_path is not None,
        "colonoscopy_image_path": req.image_path,
        "risk_factors_detected": (
            result["tabular_analysis"].get("top_risk_factors")
            if result.get("tabular_analysis")
            else None
        ),
        "history_comparison": result.get("history_analysis"),
        "ai_snapshot": ai_snapshot if ai_snapshot else None,
    }

    score = result["risk_level"]["score"] if result.get("risk_level") else 0
    if score >= 0.5:
        visit_data["diagnosis_result"] = DiagnosisResultEnum.POSITIVE
    elif score >= 0.3:
        visit_data["diagnosis_result"] = DiagnosisResultEnum.SUSPICIOUS
    else:
        visit_data["diagnosis_result"] = DiagnosisResultEnum.NEGATIVE

    visit_repo.create(patient_id=req.patient_id, **visit_data)
    db.commit()

    result["patient_id"] = req.patient_id
    return DiagnosisResponse(**result)


@router.get("/history/{patient_id}")
def get_history(patient_id: int, db: Session = Depends(get_db_dependency)):  # noqa: B008
    visits = VisitRepository(db).get_patient_visits(patient_id)
    return [
        {
            "id": v.id,
            "date": v.visit_date.isoformat(),
            "image_score": v.image_prediction_score,
            "tabular_score": v.tabular_prediction_score,
            "multimodal_score": v.multimodal_prediction_score,
            "diagnosis": v.diagnosis_result.value if v.diagnosis_result else None,
            "confidence": v.confidence_score,
            "cea": v.cea,
            "hemoglobin": v.hemoglobin,
            "crp": v.crp,
            "fobt_positive": v.fobt_positive,
            "fit_positive": v.fit_positive,
            "colonoscopy_performed": v.colonoscopy_performed,
            "polyps_found": v.polyps_found,
            "ai_snapshot": v.ai_snapshot,
        }
        for v in visits
    ]
