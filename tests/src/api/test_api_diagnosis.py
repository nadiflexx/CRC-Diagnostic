# tests/test_api_diagnosis.py
"""
Tests para /diagnosis routes.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest


def _enum():
    from src.database.models import DiagnosisResultEnum

    return DiagnosisResultEnum


def _low_risk():
    return {
        "timestamp": "2024-06-01T10:00:00",
        "final_diagnosis": "Low risk",
        "risk_level": {"level": "LOW", "score": 0.20},
        "confidence": 0.80,
        "recommendations": [],
        "patient_id": 1,
    }


@pytest.fixture
def client(fastapi_app, mock_db):
    from src.database.connection import get_db_dependency

    def override_db():
        yield mock_db

    fastapi_app.dependency_overrides[get_db_dependency] = override_db
    with TestClient(fastapi_app, raise_server_exceptions=False) as c:
        yield c
    fastapi_app.dependency_overrides.clear()


@pytest.fixture
def base_payload():
    return {
        "patient_id": 1,
        "clinical_data": {
            "hemoglobin": 13.5,
            "hematocrit": 40.0,
            "cea": 2.5,
            "fobt_positive": False,
        },
    }


# ─── POST /diagnosis/run ──────────────────────────────────────────────────────


class TestRunDiagnosis:
    def test_patient_not_found_returns_404(self, client):
        with patch("src.api.routes.diagnosis.PatientRepository") as MockPR:
            pr = MagicMock()
            pr.get_by_id.return_value = None
            MockPR.return_value = pr

            resp = client.post(
                "/diagnosis/run", json={"patient_id": 999, "clinical_data": {}}
            )
            assert resp.status_code == 404
            assert "Patient not found" in resp.json()["detail"]

    def test_full_result_returns_200(
        self, client, mock_patient, full_diagnosis_result, base_payload
    ):
        with (
            patch("src.api.routes.diagnosis.PatientRepository") as MockPR,
            patch("src.api.routes.diagnosis.VisitRepository") as MockVR,
            patch("src.api.routes.diagnosis.get_diagnosis_engine") as MockEng,
        ):
            pr = MagicMock()
            pr.get_by_id.return_value = mock_patient
            MockPR.return_value = pr
            MockVR.return_value = MagicMock()
            eng = MagicMock()
            eng.diagnose.return_value = full_diagnosis_result
            MockEng.return_value = eng

            resp = client.post("/diagnosis/run", json=base_payload)
            assert resp.status_code == 200
            assert resp.json()["patient_id"] == 1

    def test_minimal_result_returns_200(
        self, client, mock_patient, minimal_diagnosis_result, base_payload
    ):
        with (
            patch("src.api.routes.diagnosis.PatientRepository") as MockPR,
            patch("src.api.routes.diagnosis.VisitRepository") as MockVR,
            patch("src.api.routes.diagnosis.get_diagnosis_engine") as MockEng,
        ):
            pr = MagicMock()
            pr.get_by_id.return_value = mock_patient
            MockPR.return_value = pr
            MockVR.return_value = MagicMock()
            eng = MagicMock()
            eng.diagnose.return_value = minimal_diagnosis_result
            MockEng.return_value = eng

            resp = client.post("/diagnosis/run", json=base_payload)
            assert resp.status_code == 200

    @pytest.mark.parametrize(
        "score,expected_enum_name",
        [
            (0.75, "POSITIVE"),
            (0.50, "POSITIVE"),
            (0.40, "SUSPICIOUS"),
            (0.30, "SUSPICIOUS"),
            (0.10, "NEGATIVE"),
            (0.00, "NEGATIVE"),
            (1.00, "POSITIVE"),
        ],
    )
    def test_score_to_diagnosis_result(
        self, client, mock_patient, base_payload, score, expected_enum_name
    ):
        """Verifica la lógica de clasificación por score."""
        DiagnosisResultEnum = _enum()
        result = {
            "timestamp": "2024-06-01T10:00:00",
            "final_diagnosis": "test",
            "risk_level": {"level": "X", "score": score},
            "confidence": 0.8,
            "recommendations": [],
            "patient_id": 1,
        }
        with (
            patch("src.api.routes.diagnosis.PatientRepository") as MockPR,
            patch("src.api.routes.diagnosis.VisitRepository") as MockVR,
            patch("src.api.routes.diagnosis.get_diagnosis_engine") as MockEng,
        ):
            pr = MagicMock()
            pr.get_by_id.return_value = mock_patient
            MockPR.return_value = pr
            vr = MagicMock()
            MockVR.return_value = vr
            eng = MagicMock()
            eng.diagnose.return_value = result
            MockEng.return_value = eng

            resp = client.post("/diagnosis/run", json=base_payload)
            assert resp.status_code == 200
            created = vr.create.call_args[1]
            expected = getattr(DiagnosisResultEnum, expected_enum_name)
            assert created["diagnosis_result"] == expected

    def test_no_risk_level_defaults_negative(self, client, mock_patient, base_payload):
        """Sin risk_level → score=0 → NEGATIVE."""
        DiagnosisResultEnum = _enum()
        with (
            patch("src.api.routes.diagnosis.PatientRepository") as MockPR,
            patch("src.api.routes.diagnosis.VisitRepository") as MockVR,
            patch("src.api.routes.diagnosis.get_diagnosis_engine") as MockEng,
        ):
            pr = MagicMock()
            pr.get_by_id.return_value = mock_patient
            MockPR.return_value = pr
            vr = MagicMock()
            MockVR.return_value = vr
            eng = MagicMock()
            eng.diagnose.return_value = {
                "timestamp": "2024-06-01T10:00:00",
                "final_diagnosis": "Unknown",
                "confidence": 0.5,
                "recommendations": [],
                "patient_id": 1,
            }
            MockEng.return_value = eng

            resp = client.post("/diagnosis/run", json=base_payload)
            assert resp.status_code == 200
            assert (
                vr.create.call_args[1]["diagnosis_result"]
                == DiagnosisResultEnum.NEGATIVE
            )

    def test_image_path_sets_colonoscopy_performed(self, client, mock_patient):
        """image_path → colonoscopy_performed=True."""
        with (
            patch("src.api.routes.diagnosis.PatientRepository") as MockPR,
            patch("src.api.routes.diagnosis.VisitRepository") as MockVR,
            patch("src.api.routes.diagnosis.get_diagnosis_engine") as MockEng,
        ):
            pr = MagicMock()
            pr.get_by_id.return_value = mock_patient
            MockPR.return_value = pr
            vr = MagicMock()
            MockVR.return_value = vr
            eng = MagicMock()
            eng.diagnose.return_value = _low_risk()
            MockEng.return_value = eng

            resp = client.post(
                "/diagnosis/run",
                json={
                    "patient_id": 1,
                    "clinical_data": {},
                    "image_path": "/data/uploads/1/scan.jpg",
                },
            )
            assert resp.status_code == 200
            kwargs = vr.create.call_args[1]
            assert kwargs["colonoscopy_performed"] is True
            assert kwargs["colonoscopy_image_path"] == "/data/uploads/1/scan.jpg"

    def test_no_image_path_colonoscopy_false(self, client, mock_patient, base_payload):
        """Sin image_path → colonoscopy_performed=False."""
        with (
            patch("src.api.routes.diagnosis.PatientRepository") as MockPR,
            patch("src.api.routes.diagnosis.VisitRepository") as MockVR,
            patch("src.api.routes.diagnosis.get_diagnosis_engine") as MockEng,
        ):
            pr = MagicMock()
            pr.get_by_id.return_value = mock_patient
            MockPR.return_value = pr
            vr = MagicMock()
            MockVR.return_value = vr
            eng = MagicMock()
            eng.diagnose.return_value = _low_risk()
            MockEng.return_value = eng

            resp = client.post("/diagnosis/run", json=base_payload)
            assert resp.status_code == 200
            assert vr.create.call_args[1]["colonoscopy_performed"] is False

    @pytest.mark.parametrize(
        "smoking_status,expected",
        [
            ("current", 1),
            ("former", 1),
            ("never", 0),
            (None, 0),
        ],
    )
    def test_smoking_status_to_history(
        self, client, mock_patient, base_payload, smoking_status, expected
    ):
        """smoking_status correcto → smoking_history inyectado."""
        mock_patient.smoking_status = smoking_status
        with (
            patch("src.api.routes.diagnosis.PatientRepository") as MockPR,
            patch("src.api.routes.diagnosis.VisitRepository") as MockVR,
            patch("src.api.routes.diagnosis.get_diagnosis_engine") as MockEng,
        ):
            pr = MagicMock()
            pr.get_by_id.return_value = mock_patient
            MockPR.return_value = pr
            MockVR.return_value = MagicMock()
            eng = MagicMock()
            eng.diagnose.return_value = _low_risk()
            MockEng.return_value = eng

            resp = client.post("/diagnosis/run", json=base_payload)
            assert resp.status_code == 200
            assert (
                eng.diagnose.call_args[1]["patient_data"]["smoking_history"] == expected
            )

    def test_gender_none_uses_unknown(self, client, mock_patient, base_payload):
        """gender=None → 'unknown'."""
        mock_patient.gender = None
        with (
            patch("src.api.routes.diagnosis.PatientRepository") as MockPR,
            patch("src.api.routes.diagnosis.VisitRepository") as MockVR,
            patch("src.api.routes.diagnosis.get_diagnosis_engine") as MockEng,
        ):
            pr = MagicMock()
            pr.get_by_id.return_value = mock_patient
            MockPR.return_value = pr
            MockVR.return_value = MagicMock()
            eng = MagicMock()
            eng.diagnose.return_value = _low_risk()
            MockEng.return_value = eng

            resp = client.post("/diagnosis/run", json=base_payload)
            assert resp.status_code == 200
            pd = eng.diagnose.call_args[1]["patient_data"]
            assert pd["gender"] == "unknown"

    def test_age_value_injected_as_float(self, client, mock_patient, base_payload):
        """age_value inyectado como float."""
        with (
            patch("src.api.routes.diagnosis.PatientRepository") as MockPR,
            patch("src.api.routes.diagnosis.VisitRepository") as MockVR,
            patch("src.api.routes.diagnosis.get_diagnosis_engine") as MockEng,
        ):
            pr = MagicMock()
            pr.get_by_id.return_value = mock_patient
            MockPR.return_value = pr
            MockVR.return_value = MagicMock()
            eng = MagicMock()
            eng.diagnose.return_value = _low_risk()
            MockEng.return_value = eng

            resp = client.post("/diagnosis/run", json=base_payload)
            assert resp.status_code == 200
            pd = eng.diagnose.call_args[1]["patient_data"]
            assert isinstance(pd["age_value"], float)

    def test_full_ai_snapshot_structure(
        self, client, mock_patient, full_diagnosis_result, base_payload
    ):
        """ai_snapshot tiene image, tabular y fusion."""
        with (
            patch("src.api.routes.diagnosis.PatientRepository") as MockPR,
            patch("src.api.routes.diagnosis.VisitRepository") as MockVR,
            patch("src.api.routes.diagnosis.get_diagnosis_engine") as MockEng,
        ):
            pr = MagicMock()
            pr.get_by_id.return_value = mock_patient
            MockPR.return_value = pr
            vr = MagicMock()
            MockVR.return_value = vr
            eng = MagicMock()
            eng.diagnose.return_value = full_diagnosis_result
            MockEng.return_value = eng

            resp = client.post("/diagnosis/run", json=base_payload)
            assert resp.status_code == 200
            snap = vr.create.call_args[1]["ai_snapshot"]
            assert snap is not None
            assert "image" in snap
            assert "tabular" in snap
            assert "fusion" in snap

    def test_attention_ratio_none_in_snapshot(self, client, mock_patient, base_payload):
        """attention_ratio=None → None en snapshot."""
        result = {
            "timestamp": "2024-06-01T10:00:00",
            "final_diagnosis": "Low",
            "risk_level": {"level": "LOW", "score": 0.10},
            "confidence": 0.8,
            "recommendations": [],
            "image_analysis": {
                "prediction_class": "normal",
                "prediction_score": 0.10,
                "probabilities": {"normal": 0.90},
                "ensemble_mode": "single",
                "ensemble_used": False,
                "alpha_used": 0.6,
                "beta_used": 0.4,
                "attention_ratio": None,
                "polyp_detected": False,
                "lesion_count": 0,
            },
            "patient_id": 1,
        }
        with (
            patch("src.api.routes.diagnosis.PatientRepository") as MockPR,
            patch("src.api.routes.diagnosis.VisitRepository") as MockVR,
            patch("src.api.routes.diagnosis.get_diagnosis_engine") as MockEng,
        ):
            pr = MagicMock()
            pr.get_by_id.return_value = mock_patient
            MockPR.return_value = pr
            vr = MagicMock()
            MockVR.return_value = vr
            eng = MagicMock()
            eng.diagnose.return_value = result
            MockEng.return_value = eng

            resp = client.post("/diagnosis/run", json=base_payload)
            assert resp.status_code == 200
            snap = vr.create.call_args[1]["ai_snapshot"]
            assert snap["image"]["attention_ratio"] is None

    def test_empty_snapshot_is_none(self, client, mock_patient, base_payload):
        """Sin image/tabular/multimodal → ai_snapshot=None."""
        with (
            patch("src.api.routes.diagnosis.PatientRepository") as MockPR,
            patch("src.api.routes.diagnosis.VisitRepository") as MockVR,
            patch("src.api.routes.diagnosis.get_diagnosis_engine") as MockEng,
        ):
            pr = MagicMock()
            pr.get_by_id.return_value = mock_patient
            MockPR.return_value = pr
            vr = MagicMock()
            MockVR.return_value = vr
            eng = MagicMock()
            eng.diagnose.return_value = _low_risk()
            MockEng.return_value = eng

            resp = client.post("/diagnosis/run", json=base_payload)
            assert resp.status_code == 200
            assert vr.create.call_args[1]["ai_snapshot"] is None

    def test_clinical_data_merged_to_patient_dict(
        self, client, mock_patient, base_payload
    ):
        """clinical_data fusionado en patient_data del engine."""
        with (
            patch("src.api.routes.diagnosis.PatientRepository") as MockPR,
            patch("src.api.routes.diagnosis.VisitRepository") as MockVR,
            patch("src.api.routes.diagnosis.get_diagnosis_engine") as MockEng,
        ):
            pr = MagicMock()
            pr.get_by_id.return_value = mock_patient
            MockPR.return_value = pr
            MockVR.return_value = MagicMock()
            eng = MagicMock()
            eng.diagnose.return_value = _low_risk()
            MockEng.return_value = eng

            resp = client.post("/diagnosis/run", json=base_payload)
            assert resp.status_code == 200
            pd = eng.diagnose.call_args[1]["patient_data"]
            assert "hemoglobin" in pd
            assert pd["hemoglobin"] == 13.5

    def test_db_commit_called(self, client, mock_patient, mock_db, base_payload):
        """db.commit() llamado tras guardar visita."""
        with (
            patch("src.api.routes.diagnosis.PatientRepository") as MockPR,
            patch("src.api.routes.diagnosis.VisitRepository") as MockVR,
            patch("src.api.routes.diagnosis.get_diagnosis_engine") as MockEng,
        ):
            pr = MagicMock()
            pr.get_by_id.return_value = mock_patient
            MockPR.return_value = pr
            MockVR.return_value = MagicMock()
            eng = MagicMock()
            eng.diagnose.return_value = _low_risk()
            MockEng.return_value = eng

            resp = client.post("/diagnosis/run", json=base_payload)
            assert resp.status_code == 200
            mock_db.commit.assert_called()

    def test_engine_receives_patient_id(self, client, mock_patient, base_payload):
        """Engine recibe patient_id correcto."""
        with (
            patch("src.api.routes.diagnosis.PatientRepository") as MockPR,
            patch("src.api.routes.diagnosis.VisitRepository") as MockVR,
            patch("src.api.routes.diagnosis.get_diagnosis_engine") as MockEng,
        ):
            pr = MagicMock()
            pr.get_by_id.return_value = mock_patient
            MockPR.return_value = pr
            MockVR.return_value = MagicMock()
            eng = MagicMock()
            eng.diagnose.return_value = _low_risk()
            MockEng.return_value = eng

            resp = client.post("/diagnosis/run", json=base_payload)
            assert resp.status_code == 200
            assert eng.diagnose.call_args[1]["patient_id"] == 1


# ─── GET /diagnosis/history/{patient_id} ─────────────────────────────────────


class TestGetHistory:
    def test_empty_history(self, client):
        with patch("src.api.routes.diagnosis.VisitRepository") as MockVR:
            vr = MagicMock()
            vr.get_patient_visits.return_value = []
            MockVR.return_value = vr

            resp = client.get("/diagnosis/history/1")
            assert resp.status_code == 200
            assert resp.json() == []

    def test_returns_visits_list(self, client, mock_visit):
        with patch("src.api.routes.diagnosis.VisitRepository") as MockVR:
            vr = MagicMock()
            vr.get_patient_visits.return_value = [mock_visit]
            MockVR.return_value = vr

            resp = client.get("/diagnosis/history/1")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["id"] == 1
            assert "date" in data[0]

    def test_visit_no_diagnosis_result_is_none(self, client, mock_visit):
        """diagnosis_result=None → campo 'diagnosis' es None."""
        mock_visit.diagnosis_result = None
        with patch("src.api.routes.diagnosis.VisitRepository") as MockVR:
            vr = MagicMock()
            vr.get_patient_visits.return_value = [mock_visit]
            MockVR.return_value = vr

            resp = client.get("/diagnosis/history/1")
            assert resp.status_code == 200
            assert resp.json()[0]["diagnosis"] is None

    def test_invalid_patient_id_type(self, client):
        resp = client.get("/diagnosis/history/abc")
        assert resp.status_code == 422

    def test_calls_repo_with_correct_patient_id(self, client):
        with patch("src.api.routes.diagnosis.VisitRepository") as MockVR:
            vr = MagicMock()
            vr.get_patient_visits.return_value = []
            MockVR.return_value = vr

            client.get("/diagnosis/history/7")
            vr.get_patient_visits.assert_called_once_with(7)

    def test_multiple_visits_returned(self, client, mock_visit):
        v2 = MagicMock()
        v2.id = 2
        v2.patient_id = 1
        v2.visit_date = datetime(2024, 7, 1, 10, 0, 0)
        v2.image_prediction_score = 0.40
        v2.tabular_prediction_score = 0.45
        v2.multimodal_prediction_score = 0.42
        dr2 = MagicMock()
        dr2.value = "SUSPICIOUS"
        v2.diagnosis_result = dr2
        v2.confidence_score = 0.75
        v2.cea = 3.5
        v2.hemoglobin = 12.0
        v2.crp = 2.0
        v2.fobt_positive = True
        v2.fit_positive = False
        v2.colonoscopy_performed = True
        v2.polyps_found = True
        v2.ai_snapshot = None

        with patch("src.api.routes.diagnosis.VisitRepository") as MockVR:
            vr = MagicMock()
            vr.get_patient_visits.return_value = [mock_visit, v2]
            MockVR.return_value = vr

            resp = client.get("/diagnosis/history/1")
            assert resp.status_code == 200
            assert len(resp.json()) == 2
