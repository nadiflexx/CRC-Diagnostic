"""
Pydantic schemas — Endo-AID.
"""

from datetime import date, datetime

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Patient
# ─────────────────────────────────────────────────────────────────────────────


class PatientCreate(BaseModel):
    """Schema for creating a new patient. Mirrors the lean Patient model."""

    first_name: str
    last_name: str
    date_of_birth: date
    gender: str = Field(..., pattern="^(male|female|other)$")
    height_cm: float | None = None
    weight_kg: float | None = None
    smoking_status: str | None = None
    alcohol_consumption: str | None = None
    family_history_ccr: bool = False
    has_ibd: bool = False
    previous_polyps: bool = False
    previous_cancer: bool = False


class PatientOut(PatientCreate):
    id: int
    bmi: float | None = None
    created_at: datetime

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────────────────────────────────────
# Clinical data
# ─────────────────────────────────────────────────────────────────────────────


class ClinicalDataIn(BaseModel):
    """
    Clinical and radiomic features forwarded to the diagnosis engine.

    Legacy field aliases (hemoglobin / cea) are kept for backward
    compatibility with the image-only pipeline.  The newer *_g_dl / *_ng_ml
    variants are preferred; the router merges both into the visit row.
    """

    # ── Routine analytics (persisted on Visit) ────────────────────────────
    hemoglobin: float | None = None  # g/dL — legacy alias
    hematocrit: float | None = None  # %
    wbc_count: float | None = None  # ×10³/µL
    platelet_count: float | None = None  # ×10³/µL
    albumin: float | None = None  # g/dL
    iron_serum: float | None = None  # µg/dL
    ferritin: float | None = None  # ng/mL
    crp: float | None = None  # mg/L
    cea: float | None = None  # ng/mL — legacy alias
    ca19_9: float | None = None  # U/mL
    fobt_positive: bool | None = None
    fit_positive: bool | None = None
    notes: str | None = None

    # ── Tumoral model inputs ───────────────────────────────────────────────
    age_value: float | None = Field(None, description="Patient age (years)")
    smoking_history: int | None = Field(None, description="0=No · 1=Yes")
    cea_level_ng_ml: float | None = Field(None, description="CEA (ng/mL)")
    hemoglobin_g_dl: float | None = Field(None, description="Haemoglobin (g/dL)")

    # ── Radiomic features ─────────────────────────────────────────────────
    pyrad_adc_mean: float | None = Field(None, description="Mean ADC (µm²/s)")
    pyrad_adc_std: float | None = Field(None, description="ADC std deviation")
    pyrad_entropy: float | None = Field(None, description="GLCM Entropy")
    pyrad_glcm_contrast: float | None = Field(None, description="GLCM Contrast")
    pyrad_glcm_homogeneity: float | None = Field(None, description="GLCM Homogeneity")
    pyrad_shape_sphericity: float | None = Field(None, description="Shape Sphericity")
    pyrad_firstorder_skewness: float | None = Field(
        None, description="First Order Skewness"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Visit output (used in /diagnosis/history)
# ─────────────────────────────────────────────────────────────────────────────


class VisitOut(BaseModel):
    id: int
    patient_id: int
    visit_date: datetime
    visit_type: str
    notes: str | None = None

    hemoglobin: float | None = None
    hematocrit: float | None = None
    cea: float | None = None
    ca19_9: float | None = None
    crp: float | None = None
    fobt_positive: bool | None = None
    fit_positive: bool | None = None

    multimodal_prediction_score: float | None = None
    diagnosis_result: str | None = None
    confidence_score: float | None = None

    smoking_result_id: int | None = None
    tumor_result_id: int | None = None
    image_result_id: int | None = None

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────────────────────────────────────
# Diagnosis pipeline
# ─────────────────────────────────────────────────────────────────────────────


class DiagnosisRequest(BaseModel):
    patient_id: int
    clinical_data: ClinicalDataIn
    image_path: str | None = None


class DiagnosisResponse(BaseModel):
    patient_id: int
    timestamp: str
    final_diagnosis: str | None = None
    risk_level: dict | None = None
    confidence: float | None = None
    recommendations: list[str] = []
    image_analysis: dict | None = None
    tabular_analysis: dict | None = None
    multimodal_result: dict | None = None
    history_analysis: dict | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Smoking triage
# ─────────────────────────────────────────────────────────────────────────────


class SmokingTriageRequest(BaseModel):
    """
    Request body for the smoking triage endpoint.
    patient_id is required to persist the result linked to a patient visit.
    """

    patient_id: int = Field(..., description="ID of the patient being evaluated")
    sex: int = Field(..., ge=0, le=1, description="0=Female · 1=Male")
    age: float = Field(..., ge=18, le=120)
    height: float = Field(..., ge=100, le=250, description="cm")
    weight: float = Field(..., ge=30, le=200, description="kg")
    BMI: float = Field(..., ge=10, le=60)
    waistline: float = Field(..., ge=40, le=180, description="cm")
    triglyceride: float = Field(..., ge=20, le=1000, description="mg/dL")
    HDL_chole: float = Field(..., ge=5, le=200, description="mg/dL")
    LDL_chole: float = Field(..., ge=10, le=400, description="mg/dL")
    hemoglobin: float = Field(..., ge=4, le=25, description="g/dL")
    waist_height_ratio: float = Field(0.0, ge=0.0, le=2.0)
    hemoglobin_per_height: float = Field(0.0, ge=0.0, le=1.0)


class SmokingTriageResponse(BaseModel):
    smoking_probability: float
    non_smoking_probability: float
    predicted_smoker: bool
    risk_level: str
    risk_color: str
    confidence: str
    backend: str
    top_indicators: list[dict]
    recommendation: str
    visit_id: int | None = None
