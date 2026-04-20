# tests/test_api_patients.py
"""
Tests para /patients routes.
USA serializable_patient para que Pydantic pueda serializar la respuesta.
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

# ─── Fixture de cliente ────────────────────────────────────────────────────────


@pytest.fixture
def client(fastapi_app, mock_db):
    from src.database.connection import get_db_dependency

    def override_db():
        yield mock_db

    fastapi_app.dependency_overrides[get_db_dependency] = override_db
    with TestClient(fastapi_app, raise_server_exceptions=False) as c:
        yield c
    fastapi_app.dependency_overrides.clear()


# ─── POST /patients/ ───────────────────────────────────────────────────────────


class TestCreatePatient:
    def test_create_patient_success(self, client, mock_db, serializable_patient):
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.create.return_value = serializable_patient
            MockRepo.return_value = repo

            resp = client.post(
                "/patients/",
                json={
                    "first_name": "John",
                    "last_name": "Doe",
                    "date_of_birth": "1980-01-01",
                    "gender": "male",
                    "height_cm": 175.0,
                    "weight_kg": 75.0,
                },
            )
            assert resp.status_code == 200

    def test_create_patient_bmi_calculated(self, client, mock_db, serializable_patient):
        """BMI calculado correctamente desde height/weight."""
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.create.return_value = serializable_patient
            MockRepo.return_value = repo

            resp = client.post(
                "/patients/",
                json={
                    "first_name": "Jane",
                    "last_name": "Doe",
                    "date_of_birth": "1990-05-15",
                    "gender": "female",
                    "height_cm": 160.0,
                    "weight_kg": 60.0,
                },
            )
            assert resp.status_code == 200
            call_kwargs = repo.create.call_args[1]
            expected_bmi = round(60.0 / (1.60**2), 1)
            assert call_kwargs["bmi"] == expected_bmi

    def test_create_patient_no_height_weight_bmi_none(
        self, client, mock_db, serializable_patient
    ):
        """Sin height/weight → bmi=None."""
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.create.return_value = serializable_patient
            MockRepo.return_value = repo

            resp = client.post(
                "/patients/",
                json={
                    "first_name": "Bob",
                    "last_name": "Smith",
                    "date_of_birth": "1975-03-20",
                    "gender": "male",
                },
            )
            assert resp.status_code == 200
            call_kwargs = repo.create.call_args[1]
            assert call_kwargs["bmi"] is None

    def test_create_patient_only_height_no_weight(
        self, client, mock_db, serializable_patient
    ):
        """Solo height sin weight → bmi=None."""
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.create.return_value = serializable_patient
            MockRepo.return_value = repo

            resp = client.post(
                "/patients/",
                json={
                    "first_name": "Half",
                    "last_name": "Data",
                    "date_of_birth": "1980-01-01",
                    "gender": "male",
                    "height_cm": 175.0,
                },
            )
            assert resp.status_code == 200
            assert repo.create.call_args[1]["bmi"] is None

    def test_create_patient_only_weight_no_height(
        self, client, mock_db, serializable_patient
    ):
        """Solo weight sin height → bmi=None."""
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.create.return_value = serializable_patient
            MockRepo.return_value = repo

            resp = client.post(
                "/patients/",
                json={
                    "first_name": "Half",
                    "last_name": "Data2",
                    "date_of_birth": "1980-01-01",
                    "gender": "male",
                    "weight_kg": 70.0,
                },
            )
            assert resp.status_code == 200
            assert repo.create.call_args[1]["bmi"] is None

    def test_create_patient_invalid_gender_pattern(self, client, mock_db):
        """gender no coincide con patrón → 422."""
        resp = client.post(
            "/patients/",
            json={
                "first_name": "Test",
                "last_name": "User",
                "date_of_birth": "1985-07-10",
                "gender": "alien",
            },
        )
        assert resp.status_code == 422

    def test_create_patient_db_error_returns_500(self, client, mock_db):
        """Excepción en repo.create → 500."""
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.create.side_effect = Exception("DB connection lost")
            MockRepo.return_value = repo

            resp = client.post(
                "/patients/",
                json={
                    "first_name": "Error",
                    "last_name": "Case",
                    "date_of_birth": "1980-01-01",
                    "gender": "male",
                },
            )
            assert resp.status_code == 500
            assert "DB error" in resp.json()["detail"]

    def test_create_patient_missing_required_fields(self, client):
        """Campos requeridos faltantes → 422."""
        resp = client.post("/patients/", json={"first_name": "Only"})
        assert resp.status_code == 422

    def test_create_patient_db_rollback_on_error(self, client, mock_db):
        """db.rollback() debe llamarse si hay excepción."""
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.create.side_effect = Exception("fail")
            MockRepo.return_value = repo

            client.post(
                "/patients/",
                json={
                    "first_name": "RollbackTest",
                    "last_name": "User",
                    "date_of_birth": "1980-01-01",
                    "gender": "male",
                },
            )
            mock_db.rollback.assert_called()

    def test_create_patient_refresh_called_on_success(
        self, client, mock_db, serializable_patient
    ):
        """db.refresh() llamado tras crear exitosamente."""
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.create.return_value = serializable_patient
            MockRepo.return_value = repo

            resp = client.post(
                "/patients/",
                json={
                    "first_name": "Refresh",
                    "last_name": "Test",
                    "date_of_birth": "1980-01-01",
                    "gender": "male",
                },
            )
            assert resp.status_code == 200
            mock_db.refresh.assert_called_once_with(serializable_patient)

    def test_create_patient_commit_called_on_success(
        self, client, mock_db, serializable_patient
    ):
        """db.commit() llamado tras crear exitosamente."""
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.create.return_value = serializable_patient
            MockRepo.return_value = repo

            resp = client.post(
                "/patients/",
                json={
                    "first_name": "Commit",
                    "last_name": "Test",
                    "date_of_birth": "1980-01-01",
                    "gender": "male",
                },
            )
            assert resp.status_code == 200
            mock_db.commit.assert_called()


# ─── GET /patients/ ────────────────────────────────────────────────────────────


class TestGetPatients:
    def test_get_all_patients_empty(self, client, mock_db):
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.get_all.return_value = []
            MockRepo.return_value = repo

            resp = client.get("/patients/")
            assert resp.status_code == 200
            assert resp.json() == []

    def test_get_all_patients_returns_list(self, client, mock_db, serializable_patient):
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.get_all.return_value = [serializable_patient]
            MockRepo.return_value = repo

            resp = client.get("/patients/")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)
            assert len(data) == 1

    def test_get_all_patients_calls_repo(self, client, mock_db):
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.get_all.return_value = []
            MockRepo.return_value = repo

            client.get("/patients/")
            repo.get_all.assert_called_once()


# ─── GET /patients/{patient_id} ───────────────────────────────────────────────


class TestGetPatientById:
    def test_get_patient_found(self, client, mock_db, serializable_patient):
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.get_by_id.return_value = serializable_patient
            MockRepo.return_value = repo

            resp = client.get("/patients/1")
            assert resp.status_code == 200

    def test_get_patient_not_found(self, client, mock_db):
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.get_by_id.return_value = None
            MockRepo.return_value = repo

            resp = client.get("/patients/999")
            assert resp.status_code == 404
            assert resp.json()["detail"] == "Patient not found"

    def test_get_patient_invalid_id_type(self, client, mock_db):
        resp = client.get("/patients/abc")
        assert resp.status_code == 422

    def test_get_patient_calls_repo_with_correct_id(
        self, client, mock_db, serializable_patient
    ):
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.get_by_id.return_value = serializable_patient
            MockRepo.return_value = repo

            client.get("/patients/42")
            repo.get_by_id.assert_called_once_with(42)

    def test_get_patient_response_has_id(self, client, mock_db, serializable_patient):
        with patch("src.api.routes.patients.PatientRepository") as MockRepo:
            repo = MagicMock()
            repo.get_by_id.return_value = serializable_patient
            MockRepo.return_value = repo

            resp = client.get("/patients/1")
            assert resp.status_code == 200
            assert resp.json()["id"] == 1
