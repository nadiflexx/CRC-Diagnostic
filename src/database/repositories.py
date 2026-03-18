from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from src.database.models import Patient, TrainingImage, Visit


class PatientRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> Patient:
        """
        Create a new patient.
        :param kwargs: The patient's information.
        :return: The created patient object.
        """
        if kwargs.get("height_cm") and kwargs.get("weight_kg"):
            h_m = kwargs["height_cm"] / 100
            kwargs["bmi"] = round(kwargs["weight_kg"] / (h_m**2), 1)
        patient = Patient(**kwargs)
        self.db.add(patient)
        self.db.flush()
        return patient

    def get_by_id(self, patient_id: int) -> Patient | None:
        """
        Get a patient by their ID.

        :param patient_id: The ID of the patient to retrieve.
        :return: The patient object if found, otherwise None.
        """
        return self.db.query(Patient).filter(Patient.id == patient_id).first()

    def get_all(self, skip: int = 0, limit: int = 100):
        """
        Get all patients.

        :param skip: The number of records to skip.
        :param limit: The maximum number of records to retrieve.
        :return: A list of patient objects.
        """
        return self.db.query(Patient).offset(skip).limit(limit).all()

    def search(self, query: str):
        """
        Search for patients by first name, last name, or external ID.

        :param query: The search query.
        :return: A list of patient objects that match the search criteria.
        """
        pattern = f"%{query}%"
        return (
            self.db.query(Patient)
            .filter(
                (Patient.first_name.ilike(pattern))
                | (Patient.last_name.ilike(pattern))
                | (Patient.external_id.ilike(pattern))
            )
            .all()
        )

    def update(self, patient_id: int, **kwargs) -> Patient | None:
        """
        Update a patient's information.
        :param patient_id: The ID of the patient to update.
        :param kwargs: The updated patient information.
        :return: The updated patient object if found, otherwise None.
        """

        patient = self.get_by_id(patient_id)
        if patient:
            for k, v in kwargs.items():
                setattr(patient, k, v)
            self.db.flush()
        return patient

    def delete(self, patient_id: int) -> bool:
        """
        Delete a patient by their ID.
        :param patient_id: The ID of the patient to delete.
        :return: True if the patient was deleted successfully, otherwise False.
        """
        patient = self.get_by_id(patient_id)
        if patient:
            self.db.delete(patient)
            return True
        return False


class VisitRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, patient_id: int, **kwargs) -> Visit:
        """
        Create a new visit for a patient.
        :param patient_id: The ID of the patient.
        :param kwargs: The visit information.
        :return: The created visit object.
        """
        visit = Visit(patient_id=patient_id, **kwargs)
        self.db.add(visit)
        self.db.flush()
        return visit

    def get_by_id(self, visit_id: int) -> Visit | None:
        """
        Get a visit by its ID.
        :param visit_id: The ID of the visit to retrieve.
        :return: The visit object if found, otherwise None.
        """
        return self.db.query(Visit).filter(Visit.id == visit_id).first()

    def get_patient_visits(self, patient_id: int):
        """
        Get all visits for a patient.

        :param patient_id: The ID of the patient.
        :return: A list of visit objects for the patient.
        """
        return (
            self.db.query(Visit)
            .filter(Visit.patient_id == patient_id)
            .order_by(desc(Visit.visit_date))
            .all()
        )

    def get_last_visit(self, patient_id: int) -> Visit | None:
        """
        Get the last visit for a patient.
        :param patient_id: The ID of the patient.
        :return: The last visit object for the patient, or None if no visits exist.
        """
        return (
            self.db.query(Visit)
            .filter(Visit.patient_id == patient_id)
            .order_by(desc(Visit.visit_date))
            .first()
        )

    def update(self, visit_id: int, **kwargs) -> Visit | None:
        """
        Update a visit's information.

        :param visit_id: The ID of the visit to update.
        :param kwargs: The updated visit information.
        :return: The updated visit object if found, otherwise None.
        """
        visit = self.get_by_id(visit_id)
        if visit:
            for k, v in kwargs.items():
                setattr(visit, k, v)
            self.db.flush()
        return visit


class TrainingImageRepository:
    def __init__(self, db: Session):
        self.db = db

    def bulk_insert(self, images: list[dict]):
        """
        Bulk insert training images into the database.
        :param images: A list of image dictionaries.
        :return: The number of images inserted.
        """
        objs = [TrainingImage(**img) for img in images]
        self.db.bulk_save_objects(objs)
        self.db.flush()
        return len(objs)

    def get_by_split(self, split: str):
        """
        Get all training images for a specific split.
        :param split: The split to retrieve images for.
        :return: A list of training image objects.
        """
        return self.db.query(TrainingImage).filter(TrainingImage.split == split).all()

    def get_stats(self):
        """
        Get statistics for training images.
        :return: A list of statistics.
        """
        return (
            self.db.query(
                TrainingImage.label, TrainingImage.split, func.count(TrainingImage.id)
            )
            .group_by(TrainingImage.label, TrainingImage.split)
            .all()
        )

    def get_all(self):
        """
        Get all training images.
        :return: A list of training image objects.
        """
        return self.db.query(TrainingImage).all()
