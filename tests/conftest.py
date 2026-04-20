# tests/conftest.py
"""
Shared fixtures corregidos para todos los tests.
"""

from datetime import date, datetime
import logging
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def patch_db_startup():
    """Parchea create_all globalmente para toda la sesión."""
    with patch("src.api.app.Base.metadata.create_all"):
        yield


@pytest.fixture(scope="session")
def fastapi_app(patch_db_startup):
    """Instancia FastAPI real, reutilizada en toda la sesión."""
    # Importar DESPUÉS del patch para que create_all ya esté mockeado
    from src.api.app import app as _app

    return _app


@pytest.fixture
def mock_db():
    """SQLAlchemy Session mock."""
    db = MagicMock()
    db.commit = MagicMock()
    db.rollback = MagicMock()
    db.refresh = MagicMock()
    return db


@pytest.fixture
def mock_patient():
    """
    Mock de ORM Patient con campos serializables por Pydantic.
    gender es un MagicMock con .value = "male" para evitar depender
    del enum real cuyo nombre exacto desconocemos.
    """
    patient = MagicMock()
    patient.id = 1
    patient.external_id = "EXT-001"
    patient.first_name = "John"
    patient.last_name = "Doe"
    patient.date_of_birth = date(1980, 1, 1)

    # gender: mock con .value para que el route pueda hacer patient.gender.value
    gender_mock = MagicMock()
    gender_mock.value = "male"
    patient.gender = gender_mock

    patient.ethnicity = "caucasian"
    patient.height_cm = 175.0
    patient.weight_kg = 75.0
    patient.bmi = 24.5
    patient.smoking_status = "never"
    patient.alcohol_consumption = "low"
    patient.physical_activity = "moderate"
    patient.diet_type = "mixed"
    patient.family_history_ccr = False
    patient.family_history_polyps = False
    patient.family_history_lynch = False
    patient.family_history_fap = False
    patient.has_ibd = False
    patient.ibd_type = None
    patient.has_diabetes_t2 = False
    patient.previous_polyps = False
    patient.previous_polyps_count = 0
    patient.previous_cancer = False
    patient.created_at = datetime(2024, 1, 1, 12, 0, 0)
    return patient


@pytest.fixture
def serializable_patient():
    """
    Mock de paciente que Pydantic PatientOut puede serializar correctamente.
    Todos los campos tienen tipos nativos Python (no MagicMock anidados).
    """
    patient = MagicMock()
    patient.id = 1
    patient.external_id = "EXT-001"
    patient.first_name = "John"
    patient.last_name = "Doe"
    patient.date_of_birth = date(1980, 1, 1)
    patient.gender = "male"  # string plano — Pydantic lo acepta
    patient.ethnicity = "caucasian"
    patient.height_cm = 175.0
    patient.weight_kg = 75.0
    patient.bmi = 24.5
    patient.smoking_status = "never"
    patient.alcohol_consumption = "low"
    patient.physical_activity = "moderate"
    patient.diet_type = "mixed"
    patient.family_history_ccr = False
    patient.family_history_polyps = False
    patient.family_history_lynch = False
    patient.family_history_fap = False
    patient.has_ibd = False
    patient.ibd_type = None
    patient.has_diabetes_t2 = False
    patient.previous_polyps = False
    patient.previous_polyps_count = 0
    patient.previous_cancer = False
    patient.created_at = datetime(2024, 1, 1, 12, 0, 0)
    return patient


@pytest.fixture
def mock_visit():
    """Mock de visita ORM con campos serializables."""
    visit = MagicMock()
    visit.id = 1
    visit.patient_id = 1
    visit.visit_date = datetime(2024, 6, 1, 10, 0, 0)
    visit.notes = "Routine check"
    visit.hemoglobin = 13.5
    visit.hematocrit = 40.0
    visit.wbc_count = 6.0
    visit.platelet_count = 250.0
    visit.albumin = 4.0
    visit.iron_serum = 80.0
    visit.ferritin = 50.0
    visit.crp = 1.2
    visit.cea = 2.5
    visit.ca19_9 = 15.0
    visit.fobt_positive = False
    visit.fit_positive = False
    visit.image_prediction_score = 0.25
    visit.tabular_prediction_score = 0.30
    visit.multimodal_prediction_score = 0.27

    # diagnosis_result: mock con .value para la serialización en el route
    dr = MagicMock()
    dr.value = "NEGATIVE"
    visit.diagnosis_result = dr

    visit.confidence_score = 0.85
    visit.colonoscopy_performed = False
    visit.polyps_found = False
    visit.ai_snapshot = None
    return visit


@pytest.fixture
def full_diagnosis_result():
    return {
        "timestamp": "2024-06-01T10:00:00",
        "final_diagnosis": "Low risk of colorectal cancer",
        "risk_level": {"level": "LOW", "score": 0.25},
        "confidence": 0.85,
        "recommendations": ["Schedule follow-up in 12 months"],
        "image_analysis": {
            "prediction_class": "normal",
            "prediction_score": 0.25,
            "probabilities": {"normal": 0.75, "cancer": 0.25},
            "ensemble_mode": "weighted",
            "ensemble_used": True,
            "alpha_used": 0.6,
            "beta_used": 0.4,
            "attention_ratio": 0.55,
            "polyp_detected": False,
            "lesion_count": 0,
        },
        "tabular_analysis": {
            "prediction_score": 0.30,
            "high_risk": False,
            "top_risk_factors": ["age", "smoking"],
        },
        "multimodal_result": {
            "combined_score": 0.27,
        },
        "history_analysis": {"previous_visits": 0},
        "patient_id": 1,
    }


@pytest.fixture
def minimal_diagnosis_result():
    return {
        "timestamp": "2024-06-01T10:00:00",
        "final_diagnosis": "Low risk",
        "risk_level": {"level": "LOW", "score": 0.20},
        "confidence": 0.70,
        "recommendations": [],
        "patient_id": 1,
    }


@pytest.fixture
def patched_paths(tmp_path):
    """
    Return a context that makes all path properties resolve under tmp_path.
    We monkey-patch the ROOT attribute on the singleton instance so that
    all @property paths derived from ROOT automatically redirect.
    """
    # Pre-create dirs that __init__ validators may need
    (tmp_path / ".kaggle").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".kaggle" / "kaggle.json").write_text("{}")
    return tmp_path


def pytest_configure(config):
    """Silence verbose third-party loggers during test runs."""
    for name in ("onnxruntime", "mlflow", "torch", "timm", "PIL"):
        logging.getLogger(name).setLevel(logging.ERROR)
