# tests/test_schemas.py
"""
Tests para Pydantic schemas — sin cambios, ya pasaban.
"""

from datetime import date, datetime

from pydantic import ValidationError
import pytest

from src.api.schemas import (
    ClinicalDataIn,
    DiagnosisRequest,
    DiagnosisResponse,
    PatientCreate,
    PatientOut,
    VisitOut,
)


class TestPatientCreate:
    def test_valid_minimal(self):
        p = PatientCreate(
            first_name="John",
            last_name="Doe",
            date_of_birth=date(1980, 1, 1),
            gender="male",
        )
        assert p.first_name == "John"
        assert p.family_history_ccr is False

    def test_valid_full(self):
        p = PatientCreate(
            first_name="Jane",
            last_name="Doe",
            date_of_birth=date(1990, 5, 15),
            gender="female",
            ethnicity="hispanic",
            height_cm=165.0,
            weight_kg=60.0,
            smoking_status="never",
            alcohol_consumption="low",
            physical_activity="high",
            diet_type="vegetarian",
            family_history_ccr=True,
            has_ibd=True,
            ibd_type="crohn",
            previous_polyps=True,
            previous_polyps_count=2,
        )
        assert p.gender == "female"
        assert p.has_ibd is True

    def test_invalid_gender_pattern(self):
        with pytest.raises(ValidationError):
            PatientCreate(
                first_name="X",
                last_name="Y",
                date_of_birth=date(1980, 1, 1),
                gender="unknown",
            )

    def test_gender_other_valid(self):
        p = PatientCreate(
            first_name="Alex",
            last_name="Smith",
            date_of_birth=date(1995, 3, 10),
            gender="other",
        )
        assert p.gender == "other"

    def test_gender_female_valid(self):
        p = PatientCreate(
            first_name="Maria",
            last_name="Garcia",
            date_of_birth=date(1985, 6, 15),
            gender="female",
        )
        assert p.gender == "female"

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            PatientCreate(first_name="Only")

    def test_defaults_are_false(self):
        p = PatientCreate(
            first_name="A",
            last_name="B",
            date_of_birth=date(2000, 1, 1),
            gender="male",
        )
        assert p.previous_polyps_count == 0
        assert p.previous_cancer is False
        assert p.family_history_lynch is False
        assert p.family_history_fap is False
        assert p.has_diabetes_t2 is False

    def test_optional_fields_none_by_default(self):
        p = PatientCreate(
            first_name="A",
            last_name="B",
            date_of_birth=date(2000, 1, 1),
            gender="male",
        )
        assert p.ethnicity is None
        assert p.height_cm is None
        assert p.weight_kg is None
        assert p.smoking_status is None
        assert p.ibd_type is None


class TestPatientOut:
    def test_valid_patient_out(self):
        p = PatientOut(
            id=1,
            first_name="John",
            last_name="Doe",
            date_of_birth=date(1980, 1, 1),
            gender="male",
            bmi=24.5,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
        )
        assert p.id == 1
        assert p.bmi == 24.5

    def test_optional_fields_none(self):
        p = PatientOut(
            id=2,
            first_name="Jane",
            last_name="Doe",
            date_of_birth=date(1990, 1, 1),
            gender="female",
            created_at=datetime(2024, 2, 1),
        )
        assert p.external_id is None
        assert p.bmi is None

    def test_from_attributes_enabled(self):
        assert PatientOut.model_config.get("from_attributes") is True


class TestClinicalDataIn:
    def test_all_fields_none_by_default(self):
        c = ClinicalDataIn()
        assert c.hemoglobin is None
        assert c.cea is None
        assert c.fobt_positive is None
        assert c.age_value is None
        assert c.smoking_history is None

    def test_legacy_fields_set(self):
        c = ClinicalDataIn(
            hemoglobin=13.5,
            hematocrit=40.0,
            wbc_count=6.0,
            platelet_count=250.0,
            albumin=4.0,
            iron_serum=80.0,
            ferritin=50.0,
            crp=1.2,
            cea=2.5,
            ca19_9=15.0,
            fobt_positive=False,
            fit_positive=True,
            notes="Test note",
        )
        assert c.hemoglobin == 13.5
        assert c.fobt_positive is False
        assert c.fit_positive is True
        assert c.notes == "Test note"

    def test_new_clinical_fields_set(self):
        c = ClinicalDataIn(
            age_value=45.0,
            smoking_history=1,
            cea_level_ng_ml=3.5,
            hemoglobin_g_dl=13.0,
            pyrad_adc_mean=800.0,
            pyrad_adc_std=50.0,
            pyrad_entropy=2.5,
            pyrad_glcm_contrast=0.8,
            pyrad_glcm_homogeneity=0.7,
            pyrad_shape_sphericity=0.6,
            pyrad_firstorder_skewness=0.1,
        )
        assert c.age_value == 45.0
        assert c.smoking_history == 1
        assert c.pyrad_adc_mean == 800.0

    def test_model_dump_exclude_none_removes_nulls(self):
        c = ClinicalDataIn(hemoglobin=13.5, cea=2.5)
        dumped = c.model_dump(exclude_none=True)
        assert "hemoglobin" in dumped
        assert "cea" in dumped
        assert "hematocrit" not in dumped
        assert "age_value" not in dumped

    def test_model_dump_all_none_returns_empty_dict(self):
        c = ClinicalDataIn()
        dumped = c.model_dump(exclude_none=True)
        assert dumped == {}

    def test_partial_fields_dump(self):
        c = ClinicalDataIn(age_value=50.0, smoking_history=0)
        dumped = c.model_dump(exclude_none=True)
        assert dumped == {"age_value": 50.0, "smoking_history": 0}


class TestVisitOut:
    def test_valid_visit_out(self):
        v = VisitOut(
            id=1,
            patient_id=1,
            visit_date=datetime(2024, 6, 1),
            notes="Check",
            hemoglobin=13.5,
            cea=2.5,
            image_prediction_score=0.25,
            tabular_prediction_score=0.30,
            multimodal_prediction_score=0.27,
            diagnosis_result="NEGATIVE",
            confidence_score=0.85,
        )
        assert v.id == 1
        assert v.diagnosis_result == "NEGATIVE"

    def test_visit_out_all_optional_none(self):
        v = VisitOut(
            id=2,
            patient_id=1,
            visit_date=datetime(2024, 7, 1),
        )
        assert v.notes is None
        assert v.hemoglobin is None
        assert v.confidence_score is None
        assert v.cea is None

    def test_from_attributes_enabled(self):
        assert VisitOut.model_config.get("from_attributes") is True


class TestDiagnosisRequest:
    def test_minimal_request(self):
        req = DiagnosisRequest(
            patient_id=1,
            clinical_data=ClinicalDataIn(),
        )
        assert req.patient_id == 1
        assert req.image_path is None

    def test_full_request_with_image(self):
        req = DiagnosisRequest(
            patient_id=42,
            clinical_data=ClinicalDataIn(hemoglobin=13.5),
            image_path="/data/uploads/42/colonoscopy.jpg",
        )
        assert req.patient_id == 42
        assert req.image_path == "/data/uploads/42/colonoscopy.jpg"

    def test_missing_patient_id_raises(self):
        with pytest.raises(ValidationError):
            DiagnosisRequest(clinical_data=ClinicalDataIn())

    def test_missing_clinical_data_raises(self):
        with pytest.raises(ValidationError):
            DiagnosisRequest(patient_id=1)


class TestDiagnosisResponse:
    def test_minimal_response(self):
        resp = DiagnosisResponse(
            patient_id=1,
            timestamp="2024-06-01T10:00:00",
        )
        assert resp.patient_id == 1
        assert resp.recommendations == []
        assert resp.final_diagnosis is None

    def test_full_response(self):
        resp = DiagnosisResponse(
            patient_id=1,
            timestamp="2024-06-01T10:00:00",
            final_diagnosis="Low risk",
            risk_level={"level": "LOW", "score": 0.25},
            confidence=0.85,
            recommendations=["Follow up in 12 months"],
            image_analysis={"prediction_score": 0.25},
            tabular_analysis={"prediction_score": 0.30},
            multimodal_result={"combined_score": 0.27},
            history_analysis={"previous_visits": 1},
        )
        assert resp.final_diagnosis == "Low risk"
        assert len(resp.recommendations) == 1
        assert resp.confidence == 0.85

    def test_optional_fields_default_none(self):
        resp = DiagnosisResponse(
            patient_id=5,
            timestamp="2024-06-01T10:00:00",
        )
        assert resp.image_analysis is None
        assert resp.tabular_analysis is None
        assert resp.multimodal_result is None
        assert resp.history_analysis is None
        assert resp.confidence is None

    def test_recommendations_default_empty_list(self):
        resp = DiagnosisResponse(
            patient_id=1,
            timestamp="2024-06-01T10:00:00",
        )
        assert resp.recommendations == []
        assert isinstance(resp.recommendations, list)
