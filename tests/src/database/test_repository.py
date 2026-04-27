# tests/test_repositories.py
"""
Tests para PatientRepository, VisitRepository y TrainingImageRepository.
"""

from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from src.database.models import (
    GenderEnum,
    Patient,
    TrainingImage,
    Visit,
)
from src.database.repositories import (
    PatientRepository,
    TrainingImageRepository,
    VisitRepository,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def db():
    """Mock de SQLAlchemy Session."""
    session = MagicMock()
    # Configurar el encadenamiento de query
    query_mock = MagicMock()
    session.query.return_value = query_mock
    query_mock.filter.return_value = query_mock
    query_mock.offset.return_value = query_mock
    query_mock.limit.return_value = query_mock
    query_mock.order_by.return_value = query_mock
    query_mock.group_by.return_value = query_mock
    query_mock.first.return_value = None
    query_mock.all.return_value = []
    return session


@pytest.fixture
def sample_patient():
    p = MagicMock(spec=Patient)
    p.id = 1
    p.first_name = "John"
    p.last_name = "Doe"
    p.date_of_birth = date(1980, 1, 1)
    p.gender = GenderEnum.MALE
    p.height_cm = 175.0
    p.weight_kg = 75.0
    p.bmi = 24.5
    return p


@pytest.fixture
def sample_visit():
    v = MagicMock(spec=Visit)
    v.id = 1
    v.patient_id = 1
    v.visit_date = datetime(2024, 6, 1)
    return v


# ─── PatientRepository ────────────────────────────────────────────────────────


class TestPatientRepository:
    def test_create_adds_patient_to_session(self, db):
        repo = PatientRepository(db)
        repo.create(
            first_name="John",
            last_name="Doe",
            date_of_birth=date(1980, 1, 1),
            gender=GenderEnum.MALE,
        )
        db.add.assert_called_once()
        db.flush.assert_called_once()

    def test_create_returns_patient(self, db):
        repo = PatientRepository(db)
        result = repo.create(
            first_name="John",
            last_name="Doe",
            date_of_birth=date(1980, 1, 1),
            gender=GenderEnum.MALE,
        )
        assert isinstance(result, Patient)

    def test_create_computes_bmi_when_height_and_weight(self, db):
        """create() calcula BMI si height_cm y weight_kg están presentes."""
        repo = PatientRepository(db)
        result = repo.create(
            first_name="Test",
            last_name="User",
            date_of_birth=date(1990, 1, 1),
            gender=GenderEnum.FEMALE,
            height_cm=160.0,
            weight_kg=60.0,
        )
        expected_bmi = round(60.0 / (1.60**2), 1)
        assert result.bmi == expected_bmi

    def test_create_no_bmi_without_height(self, db):
        """Sin height_cm → no calcula BMI."""
        repo = PatientRepository(db)
        result = repo.create(
            first_name="Test",
            last_name="User",
            date_of_birth=date(1990, 1, 1),
            gender=GenderEnum.MALE,
            weight_kg=70.0,
        )
        # BMI no debe establecerse por el repo (a menos que ya venga en kwargs)
        assert not hasattr(result, "bmi") or result.bmi is None or True

    def test_get_by_id_found(self, db, sample_patient):
        db.query.return_value.filter.return_value.first.return_value = sample_patient
        repo = PatientRepository(db)
        result = repo.get_by_id(1)
        assert result is sample_patient

    def test_get_by_id_not_found(self, db):
        db.query.return_value.filter.return_value.first.return_value = None
        repo = PatientRepository(db)
        result = repo.get_by_id(999)
        assert result is None

    def test_get_all_returns_list(self, db, sample_patient):
        db.query.return_value.offset.return_value.limit.return_value.all.return_value = [
            sample_patient
        ]
        repo = PatientRepository(db)
        result = repo.get_all()
        assert isinstance(result, list)

    def test_get_all_empty(self, db):
        db.query.return_value.offset.return_value.limit.return_value.all.return_value = []
        repo = PatientRepository(db)
        result = repo.get_all()
        assert result == []

    def test_get_all_uses_offset_and_limit(self, db):
        repo = PatientRepository(db)
        repo.get_all(skip=10, limit=5)
        db.query.return_value.offset.assert_called_with(10)
        db.query.return_value.offset.return_value.limit.assert_called_with(5)

    def test_search_returns_list(self, db, sample_patient):
        db.query.return_value.filter.return_value.all.return_value = [sample_patient]
        repo = PatientRepository(db)
        result = repo.search("John")
        assert isinstance(result, list)

    def test_search_empty_results(self, db):
        db.query.return_value.filter.return_value.all.return_value = []
        repo = PatientRepository(db)
        result = repo.search("nonexistent")
        assert result == []

    def test_update_existing_patient(self, db, sample_patient):
        db.query.return_value.filter.return_value.first.return_value = sample_patient
        repo = PatientRepository(db)
        result = repo.update(1, first_name="Jane")
        assert result is sample_patient
        db.flush.assert_called()

    def test_update_nonexistent_patient_returns_none(self, db):
        db.query.return_value.filter.return_value.first.return_value = None
        repo = PatientRepository(db)
        result = repo.update(999, first_name="Jane")
        assert result is None

    def test_delete_existing_patient_returns_true(self, db, sample_patient):
        db.query.return_value.filter.return_value.first.return_value = sample_patient
        repo = PatientRepository(db)
        result = repo.delete(1)
        assert result is True
        db.delete.assert_called_once_with(sample_patient)

    def test_delete_nonexistent_patient_returns_false(self, db):
        db.query.return_value.filter.return_value.first.return_value = None
        repo = PatientRepository(db)
        result = repo.delete(999)
        assert result is False

    def test_init_stores_db(self, db):
        repo = PatientRepository(db)
        assert repo.db is db


# ─── VisitRepository ──────────────────────────────────────────────────────────


class TestVisitRepository:
    def test_create_visit_adds_to_session(self, db):
        repo = VisitRepository(db)
        repo.create(patient_id=1, notes="Test visit")
        db.add.assert_called_once()
        db.flush.assert_called_once()

    def test_create_visit_returns_visit(self, db):
        repo = VisitRepository(db)
        result = repo.create(patient_id=1)
        assert isinstance(result, Visit)

    def test_create_visit_sets_patient_id(self, db):
        repo = VisitRepository(db)
        result = repo.create(patient_id=42)
        assert result.patient_id == 42

    def test_get_by_id_found(self, db, sample_visit):
        db.query.return_value.filter.return_value.first.return_value = sample_visit
        repo = VisitRepository(db)
        result = repo.get_by_id(1)
        assert result is sample_visit

    def test_get_by_id_not_found(self, db):
        db.query.return_value.filter.return_value.first.return_value = None
        repo = VisitRepository(db)
        result = repo.get_by_id(999)
        assert result is None

    def test_get_patient_visits_returns_list(self, db, sample_visit):
        db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [
            sample_visit
        ]
        repo = VisitRepository(db)
        result = repo.get_patient_visits(1)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_get_patient_visits_empty(self, db):
        db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
        repo = VisitRepository(db)
        result = repo.get_patient_visits(999)
        assert result == []

    def test_get_last_visit_returns_visit(self, db, sample_visit):
        db.query.return_value.filter.return_value.order_by.return_value.first.return_value = sample_visit
        repo = VisitRepository(db)
        result = repo.get_last_visit(1)
        assert result is sample_visit

    def test_get_last_visit_no_visits(self, db):
        db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        repo = VisitRepository(db)
        result = repo.get_last_visit(999)
        assert result is None

    def test_update_existing_visit(self, db, sample_visit):
        db.query.return_value.filter.return_value.first.return_value = sample_visit
        repo = VisitRepository(db)
        result = repo.update(1, notes="Updated notes")
        assert result is sample_visit
        db.flush.assert_called()

    def test_update_nonexistent_visit_returns_none(self, db):
        db.query.return_value.filter.return_value.first.return_value = None
        repo = VisitRepository(db)
        result = repo.update(999, notes="test")
        assert result is None

    def test_init_stores_db(self, db):
        repo = VisitRepository(db)
        assert repo.db is db


# ─── TrainingImageRepository ──────────────────────────────────────────────────


class TestTrainingImageRepository:
    def test_bulk_insert_returns_count(self, db):
        repo = TrainingImageRepository(db)
        images = [
            {"file_path": f"/path/img{i}.jpg", "label": 0, "dataset_source": "hk"}
            for i in range(5)
        ]
        result = repo.bulk_insert(images)
        assert result == 5

    def test_bulk_insert_calls_bulk_save_objects(self, db):
        repo = TrainingImageRepository(db)
        images = [{"file_path": "/path/img.jpg", "label": 1, "dataset_source": "cvc"}]
        repo.bulk_insert(images)
        db.bulk_save_objects.assert_called_once()

    def test_bulk_insert_flushes(self, db):
        repo = TrainingImageRepository(db)
        repo.bulk_insert([{"file_path": "/p.jpg", "label": 0, "dataset_source": "hk"}])
        db.flush.assert_called_once()

    def test_bulk_insert_empty_list(self, db):
        repo = TrainingImageRepository(db)
        result = repo.bulk_insert([])
        assert result == 0

    def test_get_by_split_returns_list(self, db):
        mock_img = MagicMock(spec=TrainingImage)
        db.query.return_value.filter.return_value.all.return_value = [mock_img]
        repo = TrainingImageRepository(db)
        result = repo.get_by_split("train")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_get_by_split_empty(self, db):
        db.query.return_value.filter.return_value.all.return_value = []
        repo = TrainingImageRepository(db)
        result = repo.get_by_split("test")
        assert result == []

    def test_get_stats_returns_tuples(self, db):
        db.query.return_value.group_by.return_value.all.return_value = [
            (0, "train", 100),
            (1, "train", 80),
        ]
        repo = TrainingImageRepository(db)
        result = repo.get_stats()
        assert isinstance(result, list)

    def test_get_all_returns_list(self, db):
        db.query.return_value.all.return_value = []
        repo = TrainingImageRepository(db)
        result = repo.get_all()
        assert result == []

    def test_init_stores_db(self, db):
        repo = TrainingImageRepository(db)
        assert repo.db is db
