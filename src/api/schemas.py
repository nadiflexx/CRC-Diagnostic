"""
Pydantic schemas for API validation.
"""

from datetime import date, datetime

from pydantic import BaseModel, Field


class PatientCreate(BaseModel):
    """Schema for creating a new patient."""

    first_name: str
    last_name: str
    date_of_birth: date
    gender: str = Field(..., pattern="^(male|female|other)$")
    ethnicity: str | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    smoking_status: str | None = None
    alcohol_consumption: str | None = None
    physical_activity: str | None = None
    diet_type: str | None = None
    family_history_ccr: bool = False
    family_history_polyps: bool = False
    family_history_lynch: bool = False
    family_history_fap: bool = False
    has_ibd: bool = False
    ibd_type: str | None = None
    has_diabetes_t2: bool = False
    previous_polyps: bool = False
    previous_polyps_count: int = 0
    previous_cancer: bool = False


class PatientOut(PatientCreate):
    """Schema for returning patient information."""

    id: int
    external_id: str | None = None
    bmi: float | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class ClinicalDataIn(BaseModel):
    """
    Schema for incoming clinical data.

    Supports both legacy fields (old tabular model) and new clinical +
    radiomic features (ClinicalDataGenerator schema). Legacy fields are
    kept for backward compatibility with the image-only pipeline and
    database persistence.
    """

    # ── Legacy fields (kept for DB persistence & image pipeline) ──
    hemoglobin: float | None = None
    hematocrit: float | None = None
    wbc_count: float | None = None
    platelet_count: float | None = None
    albumin: float | None = None
    iron_serum: float | None = None
    ferritin: float | None = None
    crp: float | None = None
    cea: float | None = None
    ca19_9: float | None = None
    fobt_positive: bool | None = None
    fit_positive: bool | None = None
    notes: str | None = None

    # ── New clinical + radiomic features (ClinicalDataGenerator) ──
    age_value: float | None = Field(None, description="Patient age (years)")
    smoking_history: int | None = Field(
        None, description="Smoking history: 0=No, 1=Yes"
    )
    cea_level_ng_ml: float | None = Field(None, description="CEA level (ng/mL)")
    hemoglobin_g_dl: float | None = Field(None, description="Haemoglobin (g/dL)")
    pyrad_adc_mean: float | None = Field(None, description="Mean ADC (µm²/s)")
    pyrad_adc_std: float | None = Field(None, description="ADC standard deviation")
    pyrad_entropy: float | None = Field(None, description="GLCM Entropy")
    pyrad_glcm_contrast: float | None = Field(None, description="GLCM Contrast")
    pyrad_glcm_homogeneity: float | None = Field(None, description="GLCM Homogeneity")
    pyrad_shape_sphericity: float | None = Field(None, description="Shape Sphericity")
    pyrad_firstorder_skewness: float | None = Field(
        None, description="First Order Skewness"
    )


class VisitOut(BaseModel):
    """Schema for returning visit information."""

    id: int
    patient_id: int
    visit_date: datetime
    notes: str | None = None
    hemoglobin: float | None = None
    hematocrit: float | None = None
    cea: float | None = None
    ca19_9: float | None = None
    image_prediction_score: float | None = None
    tabular_prediction_score: float | None = None
    multimodal_prediction_score: float | None = None
    diagnosis_result: str | None = None
    confidence_score: float | None = None

    class Config:
        from_attributes = True


class DiagnosisRequest(BaseModel):
    """Schema for incoming diagnosis requests."""

    patient_id: int
    clinical_data: ClinicalDataIn
    image_path: str | None = None


class DiagnosisResponse(BaseModel):
    """Schema for returning diagnosis results."""

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
