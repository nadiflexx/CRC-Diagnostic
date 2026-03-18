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
def run_diagnosis(req: DiagnosisRequest, db: Session = None):
    """Run a diagnosis for the given patient

    Params:
    - req: DiagnosisRequest
    - db: Session

    Returns:
    - DiagnosisResponse
    """

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
        "ethnicity": patient.ethnicity,
        "smoking_status": patient.smoking_status,
        "alcohol_consumption": patient.alcohol_consumption,
        "physical_activity": patient.physical_activity,
        "diet_type": patient.diet_type,
        "family_history_ccr": patient.family_history_ccr,
        "family_history_polyps": patient.family_history_polyps,
        "family_history_lynch": patient.family_history_lynch,
        "family_history_fap": patient.family_history_fap,
        "has_ibd": patient.has_ibd,
        "has_diabetes_t2": patient.has_diabetes_t2,
        "previous_polyps": patient.previous_polyps,
        "previous_polyps_count": patient.previous_polyps_count,
        "previous_cancer": patient.previous_cancer,
    }
    patient_dict.update(req.clinical_data.model_dump(exclude_none=True))

    engine = get_diagnosis_engine()
    result = engine.diagnose(
        patient_data=patient_dict, image_path=req.image_path, patient_id=req.patient_id
    )

    visit_repo = VisitRepository(db)
    visit_data = req.clinical_data.model_dump(exclude_none=True)
    visit_data.update(
        {
            "image_prediction_score": result["image_analysis"]["prediction_score"]
            if result.get("image_analysis")
            else None,
            "tabular_prediction_score": result["tabular_analysis"]["prediction_score"]
            if result.get("tabular_analysis")
            else None,
            "multimodal_prediction_score": result["multimodal_result"]["combined_score"]
            if result.get("multimodal_result")
            else None,
            "confidence_score": result.get("confidence"),
            "colonoscopy_performed": req.image_path is not None,
            "colonoscopy_image_path": req.image_path,
            "risk_factors_detected": result["tabular_analysis"].get("top_risk_factors")
            if result.get("tabular_analysis")
            else None,
            "history_comparison": result.get("history_analysis"),
        }
    )

    score = result["risk_level"]["score"] if result.get("risk_level") else 0
    if score >= 0.5:
        visit_data["diagnosis_result"] = DiagnosisResultEnum.POSITIVE
    elif score >= 0.3:
        visit_data["diagnosis_result"] = DiagnosisResultEnum.SUSPICIOUS
    else:
        visit_data["diagnosis_result"] = DiagnosisResultEnum.NEGATIVE

    visit_repo.create(patient_id=req.patient_id, **visit_data)
    db.commit()
    return DiagnosisResponse(**result)


@router.get("/history/{patient_id}")
def get_history(patient_id: int, db: Session = None):
    """Get the diagnosis history for the given patient
    Params:
    - patient_id: int
    - db: Session
    Returns:
    - list[dict]
    """
    if db is None:
        db = Depends(get_db_dependency)
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
        }
        for v in visits
    ]
