"""
Repository layer — Endo-AID.

PatientRepository
SmokingTriageRepository
TumorAnalysisRepository
ImageAnalysisRepository
VisitRepository
TrainingImageRepository
"""

from sqlalchemy import desc, func
from sqlalchemy.orm import Session, joinedload

from src.database.models import (
    ImageAnalysisResult,
    Patient,
    SmokingTriageResult,
    TrainingImage,
    TumorAnalysisResult,
    Visit,
)

# ─────────────────────────────────────────────────────────────────────────────
# Patient
# ─────────────────────────────────────────────────────────────────────────────


class PatientRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, **kwargs) -> Patient:
        if kwargs.get("height_cm") and kwargs.get("weight_kg"):
            h_m = kwargs["height_cm"] / 100.0
            kwargs["bmi"] = round(kwargs["weight_kg"] / (h_m**2), 1)
        patient = Patient(**kwargs)
        self.db.add(patient)
        self.db.flush()
        return patient

    def get_by_id(self, patient_id: int) -> Patient | None:
        return self.db.query(Patient).filter(Patient.id == patient_id).first()

    def get_all(self, skip: int = 0, limit: int = 100) -> list[Patient]:
        return self.db.query(Patient).offset(skip).limit(limit).all()

    def search(self, query: str) -> list[Patient]:
        pattern = f"%{query}%"
        return (
            self.db.query(Patient)
            .filter(
                Patient.first_name.ilike(pattern)
                | Patient.last_name.ilike(pattern)
                | Patient.external_id.ilike(pattern)
            )
            .all()
        )

    def update(self, patient_id: int, **kwargs) -> Patient | None:
        patient = self.get_by_id(patient_id)
        if patient:
            for k, v in kwargs.items():
                setattr(patient, k, v)
            self.db.flush()
        return patient

    def delete(self, patient_id: int) -> bool:
        patient = self.get_by_id(patient_id)
        if patient:
            self.db.delete(patient)
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# SmokingTriageResult
# ─────────────────────────────────────────────────────────────────────────────


class SmokingTriageRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, **kwargs) -> SmokingTriageResult:
        record = SmokingTriageResult(**kwargs)
        self.db.add(record)
        self.db.flush()
        return record

    def get_by_id(self, record_id: int) -> SmokingTriageResult | None:
        return (
            self.db.query(SmokingTriageResult)
            .filter(SmokingTriageResult.id == record_id)
            .first()
        )


# ─────────────────────────────────────────────────────────────────────────────
# TumorAnalysisResult
# ─────────────────────────────────────────────────────────────────────────────


class TumorAnalysisRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, **kwargs) -> TumorAnalysisResult:
        record = TumorAnalysisResult(**kwargs)
        self.db.add(record)
        self.db.flush()
        return record

    def get_by_id(self, record_id: int) -> TumorAnalysisResult | None:
        return (
            self.db.query(TumorAnalysisResult)
            .filter(TumorAnalysisResult.id == record_id)
            .first()
        )


# ─────────────────────────────────────────────────────────────────────────────
# ImageAnalysisResult
# ─────────────────────────────────────────────────────────────────────────────


class ImageAnalysisRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, **kwargs) -> ImageAnalysisResult:
        record = ImageAnalysisResult(**kwargs)
        self.db.add(record)
        self.db.flush()
        return record

    def get_by_id(self, record_id: int) -> ImageAnalysisResult | None:
        return (
            self.db.query(ImageAnalysisResult)
            .filter(ImageAnalysisResult.id == record_id)
            .first()
        )


# ─────────────────────────────────────────────────────────────────────────────
# Visit
# ─────────────────────────────────────────────────────────────────────────────


class VisitRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, patient_id: int, **kwargs) -> Visit:
        visit = Visit(patient_id=patient_id, **kwargs)
        self.db.add(visit)
        self.db.flush()
        return visit

    def get_by_id(self, visit_id: int) -> Visit | None:
        return self.db.query(Visit).filter(Visit.id == visit_id).first()

    def get_patient_visits(self, patient_id: int) -> list[Visit]:
        """
        Return all visits for a patient, newest first.
        Uses joinedload to eagerly fetch the three result sub-tables
        in a single round-trip, avoiding lazy-load issues.
        """
        return (
            self.db.query(Visit)
            .options(
                joinedload(Visit.smoking_result),
                joinedload(Visit.tumor_result),
                joinedload(Visit.image_result),
            )
            .filter(Visit.patient_id == patient_id)
            .order_by(desc(Visit.visit_date))
            .all()
        )

    def get_last_visit(self, patient_id: int) -> Visit | None:
        return (
            self.db.query(Visit)
            .options(
                joinedload(Visit.smoking_result),
                joinedload(Visit.tumor_result),
                joinedload(Visit.image_result),
            )
            .filter(Visit.patient_id == patient_id)
            .order_by(desc(Visit.visit_date))
            .first()
        )

    def update(self, visit_id: int, **kwargs) -> Visit | None:
        visit = self.get_by_id(visit_id)
        if visit:
            for k, v in kwargs.items():
                setattr(visit, k, v)
            self.db.flush()
        return visit


# ─────────────────────────────────────────────────────────────────────────────
# TrainingImage  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────


class TrainingImageRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def bulk_insert(self, images: list[dict]) -> int:
        objs = [TrainingImage(**img) for img in images]
        self.db.bulk_save_objects(objs)
        self.db.flush()
        return len(objs)

    def get_by_split(self, split: str) -> list[TrainingImage]:
        return self.db.query(TrainingImage).filter(TrainingImage.split == split).all()

    def get_stats(self):
        return (
            self.db.query(
                TrainingImage.label,
                TrainingImage.split,
                func.count(TrainingImage.id),
            )
            .group_by(TrainingImage.label, TrainingImage.split)
            .all()
        )

    def get_all(self) -> list[TrainingImage]:
        return self.db.query(TrainingImage).all()
