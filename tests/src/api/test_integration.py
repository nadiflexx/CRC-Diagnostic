# tests/test_integration.py
"""
Tests de integración — flujo completo.
Usa fastapi_app (fixture session) para evitar problemas de importación.
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def client(fastapi_app, mock_db, serializable_patient):
    from src.database.connection import get_db_dependency

    def override_db():
        yield mock_db

    fastapi_app.dependency_overrides[get_db_dependency] = override_db
    with TestClient(fastapi_app, raise_server_exceptions=False) as c:
        yield c
    fastapi_app.dependency_overrides.clear()


class TestFullWorkflow:
    def test_health_always_works(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_create_then_get_patient(self, client, mock_db, serializable_patient):
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.create.return_value = serializable_patient
            repo.get_by_id.return_value = serializable_patient
            MockRepo.return_value = repo

            create_resp = client.post(
                "/patients/",
                json={
                    "first_name": "John",
                    "last_name": "Doe",
                    "date_of_birth": "1980-01-01",
                    "gender": "male",
                },
            )
            assert create_resp.status_code == 200

            get_resp = client.get("/patients/1")
            assert get_resp.status_code == 200

    def test_diagnosis_then_history(
        self, client, mock_db, mock_patient, mock_visit, full_diagnosis_result
    ):
        # Run diagnosis
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

            diag_resp = client.post(
                "/diagnosis/run",
                json={
                    "patient_id": 1,
                    "clinical_data": {"hemoglobin": 13.5},
                },
            )
            assert diag_resp.status_code == 200

        # Get history
        with patch("src.api.routes.diagnosis.VisitRepository") as MockVR2:
            vr2 = MagicMock()
            vr2.get_patient_visits.return_value = [mock_visit]
            MockVR2.return_value = vr2

            hist_resp = client.get("/diagnosis/history/1")
            assert hist_resp.status_code == 200
            assert len(hist_resp.json()) == 1

    def test_patient_not_found_returns_404(self, client, mock_db):
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.get_by_id.return_value = None
            MockRepo.return_value = repo

            resp = client.get("/patients/9999")
            assert resp.status_code == 404

    def test_empty_patient_list(self, client, mock_db):
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.get_all.return_value = []
            MockRepo.return_value = repo

            resp = client.get("/patients/")
            assert resp.status_code == 200
            assert resp.json() == []

    def test_all_main_routes_exist(self, fastapi_app):
        routes = {r.path for r in fastapi_app.routes}
        assert "/patients/" in routes
        assert "/diagnosis/run" in routes
        assert "/health" in routes
