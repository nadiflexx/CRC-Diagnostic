from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.schemas import PatientCreate, PatientOut
from src.database.connection import get_db_dependency
from src.database.models import GenderEnum
from src.database.repositories import PatientRepository

router = APIRouter(prefix="/patients", tags=["Patients"])


@router.post("/", response_model=PatientOut)
def create_patient(
    patient_in: PatientCreate,
    db: Session = Depends(get_db_dependency),  # noqa: B008
) -> PatientOut:
    repo = PatientRepository(db)

    bmi: float | None = None
    if patient_in.height_cm and patient_in.weight_kg:
        h_m = patient_in.height_cm / 100.0
        bmi = round(patient_in.weight_kg / (h_m * h_m), 1)

    patient_data = patient_in.model_dump()
    try:
        patient_data["gender"] = GenderEnum[patient_data["gender"].upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid gender value.") from None

    patient_data["bmi"] = bmi

    try:
        new_patient = repo.create(**patient_data)
        db.commit()
        db.refresh(new_patient)
        return new_patient
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}") from e


@router.get("/", response_model=list[PatientOut])
def get_patients(
    db: Session = Depends(get_db_dependency),  # noqa: B008
) -> list[PatientOut]:
    return PatientRepository(db).get_all()


@router.get("/{patient_id}", response_model=PatientOut)
def get_patient(
    patient_id: int,
    db: Session = Depends(get_db_dependency),  # noqa: B008
) -> PatientOut:
    patient = PatientRepository(db).get_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient
