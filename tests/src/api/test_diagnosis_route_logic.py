# tests/test_diagnosis_route_logic.py
"""
Tests de lógica de negocio del route diagnosis.
Boundaries de score y verificaciones de DB.
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest


def _enum():
    from src.database.models import DiagnosisResultEnum

    return DiagnosisResultEnum


@pytest.fixture
def client(fastapi_app, mock_db):
    from src.database.connection import get_db_dependency

    def override_db():
        yield mock_db

    fastapi_app.dependency_overrides[get_db_dependency] = override_db
    with TestClient(fastapi_app, raise_server_exceptions=False) as c:
        yield c
    fastapi_app.dependency_overrides.clear()


def _result(score):
    return {
        "timestamp": "2024-06-01T10:00:00",
        "final_diagnosis": "test",
        "risk_level": {"level": "X", "score": score},
        "confidence": 0.8,
        "recommendations": [],
        "patient_id": 1,
    }


def _run(client, mock_patient, score):
    """Helper: ejecuta POST /diagnosis/run y devuelve (resp, vr_mock)."""
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
        eng.diagnose.return_value = _result(score)
        MockEng.return_value = eng

        resp = client.post(
            "/diagnosis/run", json={"patient_id": 1, "clinical_data": {}}
        )
        return resp, vr


class TestScoreBoundaries:
    def test_score_1_0_positive(self, client, mock_patient):
        resp, vr = _run(client, mock_patient, 1.0)
        assert resp.status_code == 200
        assert vr.create.call_args[1]["diagnosis_result"] == _enum().POSITIVE

    def test_score_0_5_positive(self, client, mock_patient):
        resp, vr = _run(client, mock_patient, 0.5)
        assert resp.status_code == 200
        assert vr.create.call_args[1]["diagnosis_result"] == _enum().POSITIVE

    def test_score_0_499_suspicious(self, client, mock_patient):
        resp, vr = _run(client, mock_patient, 0.499)
        assert resp.status_code == 200
        assert vr.create.call_args[1]["diagnosis_result"] == _enum().SUSPICIOUS

    def test_score_0_4_suspicious(self, client, mock_patient):
        resp, vr = _run(client, mock_patient, 0.4)
        assert resp.status_code == 200
        assert vr.create.call_args[1]["diagnosis_result"] == _enum().SUSPICIOUS

    def test_score_0_3_suspicious(self, client, mock_patient):
        resp, vr = _run(client, mock_patient, 0.3)
        assert resp.status_code == 200
        assert vr.create.call_args[1]["diagnosis_result"] == _enum().SUSPICIOUS

    def test_score_0_299_negative(self, client, mock_patient):
        resp, vr = _run(client, mock_patient, 0.299)
        assert resp.status_code == 200
        assert vr.create.call_args[1]["diagnosis_result"] == _enum().NEGATIVE

    def test_score_0_0_negative(self, client, mock_patient):
        resp, vr = _run(client, mock_patient, 0.0)
        assert resp.status_code == 200
        assert vr.create.call_args[1]["diagnosis_result"] == _enum().NEGATIVE


class TestDBInteractions:
    def test_commit_called(self, client, mock_patient, mock_db):
        resp, _ = _run(client, mock_patient, 0.1)
        assert resp.status_code == 200
        mock_db.commit.assert_called()

    def test_visit_create_called_once(self, client, mock_patient):
        _, vr = _run(client, mock_patient, 0.1)
        vr.create.assert_called_once()

    def test_engine_receives_image_path(self, client, mock_patient):
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
            eng.diagnose.return_value = _result(0.1)
            MockEng.return_value = eng

            client.post(
                "/diagnosis/run",
                json={
                    "patient_id": 1,
                    "clinical_data": {},
                    "image_path": "/path/scan.jpg",
                },
            )
            assert eng.diagnose.call_args[1]["image_path"] == "/path/scan.jpg"

    def test_multimodal_fusion_in_snapshot(self, client, mock_patient):
        result = {
            **_result(0.45),
            "multimodal_result": {"combined_score": 0.45},
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

            resp = client.post(
                "/diagnosis/run", json={"patient_id": 1, "clinical_data": {}}
            )
            assert resp.status_code == 200
            snap = vr.create.call_args[1]["ai_snapshot"]
            assert "fusion" in snap
