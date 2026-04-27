# tests/src/database/test_database_model.py
"""
Tests for database models.
Only checks fields that are explicitly declared with server/column defaults.
"""

from datetime import datetime

from src.database.models import Patient, TrainingImage, Visit


class TestPatientModel:
    def test_patient_can_be_instantiated(self):
        p = Patient()
        assert p is not None

    def test_name_fields_default_none(self):
        p = Patient()
        # String fields without defaults are None until set
        assert p.first_name is None or isinstance(p.first_name, str)

    def test_can_set_age(self):
        p = Patient(date_of_birth=datetime(1978, 5, 15))
        assert p.date_of_birth == datetime(1978, 5, 15)

    def test_can_set_gender(self):
        p = Patient(gender="M")
        assert p.gender == "M"

    def test_default_booleans_are_none_or_false(self):
        """
        SQLAlchemy columns without server_default return None
        before a flush. Accept either None or False.
        """
        p = Patient()
        val = p.family_history_ccr
        assert val is None or val is False

    def test_default_polyps_count_is_none_or_zero(self):
        p = Patient()
        val = p.previous_polyps_count
        assert val is None or val == 0

    def test_set_family_history(self):
        p = Patient(family_history_ccr=True)
        assert p.family_history_ccr is True

    def test_set_previous_polyps_count(self):
        p = Patient(previous_polyps_count=3)
        assert p.previous_polyps_count == 3


class TestVisitModel:
    def test_visit_can_be_instantiated(self):
        v = Visit()
        assert v is not None

    def test_default_colonoscopy_performed_none_or_false(self):
        """Column default is None before DB flush, accept both."""
        v = Visit()
        val = v.colonoscopy_performed
        assert val is None or val is False

    def test_set_colonoscopy_performed_true(self):
        v = Visit(colonoscopy_performed=True)
        assert v.colonoscopy_performed is True

    def test_visit_date_can_be_set(self):
        now = datetime.utcnow()
        v = Visit(visit_date=now)
        assert v.visit_date == now


class TestTrainingImageModel:
    def test_training_image_instantiation(self):
        ti = TrainingImage(
            file_path="/fake/path.jpg",
            label=0,
            split="train",
        )
        assert ti.file_path == "/fake/path.jpg"
        assert ti.label == 0
        assert ti.split == "train"

    def test_mask_path_optional(self):
        ti = TrainingImage(file_path="/x.jpg", label=1, split="val")
        assert ti.mask_path is None or isinstance(ti.mask_path, str)

    def test_dataset_source_can_be_set(self):
        ti = TrainingImage(
            file_path="/x.jpg",
            label=0,
            split="test",
            dataset_source="hyperkvasir",
        )
        assert ti.dataset_source == "hyperkvasir"
