"""
Database models for the application.
"""

from datetime import datetime
import enum

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class GenderEnum(enum.Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class RiskLevelEnum(enum.Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class DiagnosisResultEnum(enum.Enum):
    NEGATIVE = "negative"
    SUSPICIOUS = "suspicious"
    POSITIVE = "positive"


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    external_id: Mapped[str | None] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100))
    date_of_birth: Mapped[datetime] = mapped_column(Date)
    gender: Mapped[GenderEnum] = mapped_column(SQLEnum(GenderEnum))
    ethnicity: Mapped[str | None] = mapped_column(String(50))
    height_cm: Mapped[float | None] = mapped_column(Float)
    weight_kg: Mapped[float | None] = mapped_column(Float)
    bmi: Mapped[float | None] = mapped_column(Float)
    smoking_status: Mapped[str | None] = mapped_column(String(20))
    alcohol_consumption: Mapped[str | None] = mapped_column(String(20))
    physical_activity: Mapped[str | None] = mapped_column(String(20))
    diet_type: Mapped[str | None] = mapped_column(String(30))
    family_history_ccr: Mapped[bool] = mapped_column(Boolean, default=False)
    family_history_polyps: Mapped[bool] = mapped_column(Boolean, default=False)
    family_history_lynch: Mapped[bool] = mapped_column(Boolean, default=False)
    family_history_fap: Mapped[bool] = mapped_column(Boolean, default=False)
    has_ibd: Mapped[bool] = mapped_column(Boolean, default=False)
    ibd_type: Mapped[str | None] = mapped_column(String(30))
    has_diabetes_t2: Mapped[bool] = mapped_column(Boolean, default=False)
    previous_polyps: Mapped[bool] = mapped_column(Boolean, default=False)
    previous_polyps_count: Mapped[int] = mapped_column(Integer, default=0)
    previous_cancer: Mapped[bool] = mapped_column(Boolean, default=False)
    computed_risk_level: Mapped[RiskLevelEnum | None] = mapped_column(
        SQLEnum(RiskLevelEnum)
    )
    visits: Mapped[list["Visit"]] = relationship(
        back_populates="patient", order_by="Visit.visit_date"
    )


class Visit(Base):
    __tablename__ = "visits"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    visit_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    notes: Mapped[str | None] = mapped_column(Text)

    # ── Analítica clínica ──────────────────────────────────────────────────
    hemoglobin: Mapped[float | None] = mapped_column(Float)
    hematocrit: Mapped[float | None] = mapped_column(Float)
    wbc_count: Mapped[float | None] = mapped_column(Float)
    platelet_count: Mapped[float | None] = mapped_column(Float)
    albumin: Mapped[float | None] = mapped_column(Float)
    iron_serum: Mapped[float | None] = mapped_column(Float)
    ferritin: Mapped[float | None] = mapped_column(Float)
    crp: Mapped[float | None] = mapped_column(Float)
    cea: Mapped[float | None] = mapped_column(Float)
    ca19_9: Mapped[float | None] = mapped_column(Float)
    fobt_positive: Mapped[bool | None] = mapped_column(Boolean)
    fit_positive: Mapped[bool | None] = mapped_column(Boolean)

    # ── Endoscopia ────────────────────────────────────────────────────────
    colonoscopy_performed: Mapped[bool] = mapped_column(Boolean, default=False)
    colonoscopy_image_path: Mapped[str | None] = mapped_column(String(500))
    polyps_found: Mapped[bool | None] = mapped_column(Boolean)
    polyps_count: Mapped[int | None] = mapped_column(Integer)
    polyp_max_size_mm: Mapped[float | None] = mapped_column(Float)
    polyp_location: Mapped[str | None] = mapped_column(String(100))
    polyp_morphology: Mapped[str | None] = mapped_column(String(100))

    # ── Scores IA ─────────────────────────────────────────────────────────
    image_prediction_score: Mapped[float | None] = mapped_column(Float)
    tabular_prediction_score: Mapped[float | None] = mapped_column(Float)
    multimodal_prediction_score: Mapped[float | None] = mapped_column(Float)
    diagnosis_result: Mapped[DiagnosisResultEnum | None] = mapped_column(
        SQLEnum(DiagnosisResultEnum)
    )
    confidence_score: Mapped[float | None] = mapped_column(Float)

    # ── Artefactos IA ─────────────────────────────────────────────────────
    gradcam_image_path: Mapped[str | None] = mapped_column(String(500))
    segmentation_mask_path: Mapped[str | None] = mapped_column(String(500))

    # ── JSON compacto con snapshot completo del análisis IA ───────────────
    ai_snapshot: Mapped[dict | None] = mapped_column(JSON)

    # ── Metadatos ─────────────────────────────────────────────────────────
    risk_factors_detected: Mapped[dict | None] = mapped_column(JSON)
    history_comparison: Mapped[dict | None] = mapped_column(JSON)
    alerts: Mapped[dict | None] = mapped_column(JSON)

    patient: Mapped["Patient"] = relationship(back_populates="visits")


class TrainingImage(Base):
    __tablename__ = "training_images"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    file_path: Mapped[str] = mapped_column(String(500))
    mask_path: Mapped[str | None] = mapped_column(String(500))
    dataset_source: Mapped[str] = mapped_column(String(100))
    label: Mapped[int] = mapped_column(Integer)
    split: Mapped[str | None] = mapped_column(String(10))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
