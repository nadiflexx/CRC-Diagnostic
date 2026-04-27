"""
Database models — Endo-AID.

Tables
──────
  patients               demographic & lifestyle profile (lean)
  smoking_triage_results output of the ReverseLogic tabular model
  tumor_analysis_results output of the XGBoost tumoral model + radiomics
  image_analysis_results output of ensemble classifier + GradCAM + U-Net
  visits                 one header row per clinical encounter
  training_images        training dataset registry (unchanged)

Relationship map
────────────────
  Patient ──< Visit
  Visit ──> SmokingTriageResult   (FK: visits.smoking_result_id)
  Visit ──> TumorAnalysisResult   (FK: visits.tumor_result_id)
  Visit ──> ImageAnalysisResult   (FK: visits.image_result_id)
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

# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────


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


class VisitTypeEnum(enum.Enum):
    """Which model(s) ran during this encounter."""

    SMOKING_TRIAGE = "smoking_triage"
    TUMOR_ANALYSIS = "tumor_analysis"
    ENDOSCOPY = "endoscopy"
    FULL = "full"


class SmokingRiskEnum(enum.Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class ImageClassEnum(enum.Enum):
    POLYP = "polyp"
    INFLAMMATION = "inflammation"
    NORMAL = "normal"


# ─────────────────────────────────────────────────────────────────────────────
# Patient  (lean — only fields that are actually written)
# ─────────────────────────────────────────────────────────────────────────────


class Patient(Base):
    """
    Core patient demographic and lifestyle profile.
    Only columns that are written by at least one router are kept.
    """

    __tablename__ = "patients"

    # ── Identity ──────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # ── Demographics ──────────────────────────────────────────────────────
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    date_of_birth: Mapped[datetime] = mapped_column(Date, nullable=False)
    gender: Mapped[GenderEnum] = mapped_column(SQLEnum(GenderEnum), nullable=False)

    # ── Anthropometrics ───────────────────────────────────────────────────
    height_cm: Mapped[float | None] = mapped_column(Float)
    weight_kg: Mapped[float | None] = mapped_column(Float)
    bmi: Mapped[float | None] = mapped_column(Float)

    # ── Lifestyle (written from registration form) ─────────────────────────
    smoking_status: Mapped[str | None] = mapped_column(String(20))
    alcohol_consumption: Mapped[str | None] = mapped_column(String(20))

    # ── Clinical risk flags (written from registration form) ───────────────
    family_history_ccr: Mapped[bool] = mapped_column(Boolean, default=False)
    has_ibd: Mapped[bool] = mapped_column(Boolean, default=False)
    previous_polyps: Mapped[bool] = mapped_column(Boolean, default=False)
    previous_cancer: Mapped[bool] = mapped_column(Boolean, default=False)

    # ── Relationships ─────────────────────────────────────────────────────
    visits: Mapped[list["Visit"]] = relationship(
        "Visit",
        back_populates="patient",
        order_by="Visit.visit_date",
        cascade="all, delete-orphan",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SmokingTriageResult
# ─────────────────────────────────────────────────────────────────────────────


class SmokingTriageResult(Base):
    """
    One row per execution of the ReverseLogic tabular model (smoking triage).
    Inputs  — all 12 features forwarded to the model.
    Outputs — probability, risk level, prediction flag, backend.
    """

    __tablename__ = "smoking_triage_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Model inputs
    sex: Mapped[int] = mapped_column(Integer, nullable=False)
    age: Mapped[float] = mapped_column(Float, nullable=False)
    height_cm: Mapped[float] = mapped_column(Float, nullable=False)
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False)
    bmi: Mapped[float] = mapped_column(Float, nullable=False)
    waistline_cm: Mapped[float] = mapped_column(Float, nullable=False)
    triglyceride_mg_dl: Mapped[float] = mapped_column(Float, nullable=False)
    hdl_mg_dl: Mapped[float] = mapped_column(Float, nullable=False)
    ldl_mg_dl: Mapped[float] = mapped_column(Float, nullable=False)
    hemoglobin_g_dl: Mapped[float] = mapped_column(Float, nullable=False)
    waist_height_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    hemoglobin_per_height: Mapped[float] = mapped_column(Float, nullable=False)

    # Model outputs
    smoking_probability: Mapped[float] = mapped_column(Float, nullable=False)
    non_smoking_probability: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_smoker: Mapped[bool] = mapped_column(Boolean, nullable=False)
    risk_level: Mapped[SmokingRiskEnum] = mapped_column(
        SQLEnum(SmokingRiskEnum), nullable=False
    )
    backend: Mapped[str] = mapped_column(String(10), nullable=False)

    # Structured indicator list
    top_indicators: Mapped[list | None] = mapped_column(JSON)

    # Back-ref
    visit: Mapped["Visit | None"] = relationship(
        "Visit",
        back_populates="smoking_result",
        foreign_keys="[Visit.smoking_result_id]",
        uselist=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TumorAnalysisResult
# ─────────────────────────────────────────────────────────────────────────────


class TumorAnalysisResult(Base):
    """
    One row per explicit execution of the XGBoost tumoral model.

    Only created when the user supplies tumoral clinical/radiomic inputs
    (cea_level_ng_ml or pyrad_* features).  Never created for image-only
    visits even if the engine returns a context tabular score.
    """

    __tablename__ = "tumor_analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Clinical inputs
    age_value: Mapped[float | None] = mapped_column(Float)
    smoking_history: Mapped[int | None] = mapped_column(Integer)
    cea_level_ng_ml: Mapped[float | None] = mapped_column(Float)
    hemoglobin_g_dl: Mapped[float | None] = mapped_column(Float)

    # Radiomic features
    pyrad_adc_mean: Mapped[float | None] = mapped_column(Float)
    pyrad_adc_std: Mapped[float | None] = mapped_column(Float)
    pyrad_entropy: Mapped[float | None] = mapped_column(Float)
    pyrad_glcm_contrast: Mapped[float | None] = mapped_column(Float)
    pyrad_glcm_homogeneity: Mapped[float | None] = mapped_column(Float)
    pyrad_shape_sphericity: Mapped[float | None] = mapped_column(Float)
    pyrad_firstorder_skewness: Mapped[float | None] = mapped_column(Float)

    # Model outputs
    prediction_score: Mapped[float] = mapped_column(Float, nullable=False)
    high_risk: Mapped[bool] = mapped_column(Boolean, nullable=False)
    top_risk_factors: Mapped[list | None] = mapped_column(JSON)

    # Back-ref
    visit: Mapped["Visit | None"] = relationship(
        "Visit",
        back_populates="tumor_result",
        foreign_keys="[Visit.tumor_result_id]",
        uselist=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ImageAnalysisResult
# ─────────────────────────────────────────────────────────────────────────────


class ImageAnalysisResult(Base):
    """
    One row per execution of the image pipeline:
      • Ensemble classifier  (Model A + Model B)
      • GradCAM explainability maps
      • U-Net polyp segmentation
    """

    __tablename__ = "image_analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Input
    colonoscopy_image_path: Mapped[str | None] = mapped_column(String(500))

    # Classifier outputs
    prediction_class: Mapped[ImageClassEnum | None] = mapped_column(
        SQLEnum(ImageClassEnum)
    )
    prediction_score: Mapped[float | None] = mapped_column(Float)
    probabilities: Mapped[dict | None] = mapped_column(JSON)

    # Ensemble metadata
    ensemble_mode: Mapped[str | None] = mapped_column(String(20))
    ensemble_used: Mapped[bool] = mapped_column(Boolean, default=False)
    alpha_used: Mapped[float | None] = mapped_column(Float)
    beta_used: Mapped[float | None] = mapped_column(Float)
    attention_ratio: Mapped[float | None] = mapped_column(Float)

    # GradCAM paths
    gradcam_a_path: Mapped[str | None] = mapped_column(String(500))
    gradcam_b_path: Mapped[str | None] = mapped_column(String(500))
    gradcam_fusion_path: Mapped[str | None] = mapped_column(String(500))

    # U-Net outputs
    polyp_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    lesion_count: Mapped[int] = mapped_column(Integer, default=0)
    segmentation_mask_path: Mapped[str | None] = mapped_column(String(500))
    segmentation_report_path: Mapped[str | None] = mapped_column(String(500))

    # Back-ref
    visit: Mapped["Visit | None"] = relationship(
        "Visit",
        back_populates="image_result",
        foreign_keys="[Visit.image_result_id]",
        uselist=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Visit
# ─────────────────────────────────────────────────────────────────────────────


class Visit(Base):
    """
    Clinical encounter header — one row per visit.
    visit_type encodes which models ran so the frontend renders correctly.
    """

    __tablename__ = "visits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )
    visit_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    visit_type: Mapped[VisitTypeEnum] = mapped_column(
        SQLEnum(VisitTypeEnum), default=VisitTypeEnum.FULL, nullable=False
    )
    notes: Mapped[str | None] = mapped_column(Text)

    # Routine analytics
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

    # Aggregate AI result
    multimodal_prediction_score: Mapped[float | None] = mapped_column(Float)
    diagnosis_result: Mapped[DiagnosisResultEnum | None] = mapped_column(
        SQLEnum(DiagnosisResultEnum)
    )
    confidence_score: Mapped[float | None] = mapped_column(Float)

    # FK links
    smoking_result_id: Mapped[int | None] = mapped_column(
        ForeignKey("smoking_triage_results.id", ondelete="SET NULL")
    )
    tumor_result_id: Mapped[int | None] = mapped_column(
        ForeignKey("tumor_analysis_results.id", ondelete="SET NULL")
    )
    image_result_id: Mapped[int | None] = mapped_column(
        ForeignKey("image_analysis_results.id", ondelete="SET NULL")
    )

    # Relationships
    patient: Mapped["Patient"] = relationship("Patient", back_populates="visits")

    smoking_result: Mapped["SmokingTriageResult | None"] = relationship(
        "SmokingTriageResult",
        back_populates="visit",
        foreign_keys=[smoking_result_id],
        uselist=False,
    )
    tumor_result: Mapped["TumorAnalysisResult | None"] = relationship(
        "TumorAnalysisResult",
        back_populates="visit",
        foreign_keys=[tumor_result_id],
        uselist=False,
    )
    image_result: Mapped["ImageAnalysisResult | None"] = relationship(
        "ImageAnalysisResult",
        back_populates="visit",
        foreign_keys=[image_result_id],
        uselist=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TrainingImage  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────


class TrainingImage(Base):
    __tablename__ = "training_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    mask_path: Mapped[str | None] = mapped_column(String(500))
    dataset_source: Mapped[str] = mapped_column(String(100), nullable=False)
    label: Mapped[int] = mapped_column(Integer, nullable=False)
    split: Mapped[str | None] = mapped_column(String(10))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
