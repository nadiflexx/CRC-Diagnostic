from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from src.database.models import Patient, TrainingImage, Visit


class PatientRepository:
    """
    Data access layer for ``Patient`` records.

    Provides CRUD operations and a fuzzy search method over the
    ``Patient`` table. All methods operate within the ``Session``
    supplied at construction time; the caller is responsible for
    committing or rolling back the transaction.
    """

    def __init__(self, db: Session):
        """
        Initialise the repository with a SQLAlchemy session.

        Args:
            db (Session): Active SQLAlchemy database session used for
                all queries and mutations performed by this repository.
        """
        self.db = db

    def create(self, **kwargs) -> Patient:
        """
        Create and persist a new ``Patient`` record.

        If both ``height_cm`` and ``weight_kg`` are provided the BMI is
        automatically computed and stored before the record is flushed.

        Args:
            **kwargs: Column values for the new ``Patient`` row. Accepts
                any attribute supported by the ``Patient`` model.
                Commonly used keys include ``first_name``, ``last_name``,
                ``external_id``, ``height_cm``, and ``weight_kg``.

        Returns:
            Patient: The newly created and flushed ``Patient`` instance
                with its database-assigned ``id`` populated.
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
        Retrieve a single patient by primary key.

        Args:
            patient_id (int): Primary key of the patient to retrieve.

        Returns:
            Patient | None: The matching ``Patient`` instance, or ``None``
                if no record with the given ID exists.
        """
        return self.db.query(Patient).filter(Patient.id == patient_id).first()

    def get_all(self, skip: int = 0, limit: int = 100):
        """
        Retrieve a paginated list of all patients.

        Args:
            skip (int): Number of records to skip (offset). Default is 0.
            limit (int): Maximum number of records to return. Default is
                100.

        Returns:
            list[Patient]: List of ``Patient`` instances within the
                requested page.
        """
        return self.db.query(Patient).offset(skip).limit(limit).all()

    def search(self, query: str):
        """
        Search for patients by first name, last name, or external ID.

        Performs a case-insensitive ``ILIKE`` pattern match on all three
        fields and returns every patient that matches at least one of them.

        Args:
            query (str): Search string. The pattern ``%query%`` is applied
                to ``first_name``, ``last_name``, and ``external_id``.

        Returns:
            list[Patient]: All ``Patient`` records whose first name, last
                name, or external ID contains ``query`` (case-insensitive).
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
        Update an existing patient record with new field values.

        Args:
            patient_id (int): Primary key of the patient to update.
            **kwargs: Column-value pairs to set on the ``Patient``
                instance.

        Returns:
            Patient | None: The updated ``Patient`` instance after
                flushing, or ``None`` if no patient with the given ID
                exists.
        """
        patient = self.get_by_id(patient_id)
        if patient:
            for k, v in kwargs.items():
                setattr(patient, k, v)
            self.db.flush()
        return patient

    def delete(self, patient_id: int) -> bool:
        """
        Delete a patient record by primary key.

        Args:
            patient_id (int): Primary key of the patient to delete.

        Returns:
            bool: ``True`` if the patient was found and marked for
                deletion, ``False`` if no matching record exists.
        """
        patient = self.get_by_id(patient_id)
        if patient:
            self.db.delete(patient)
            return True
        return False


class VisitRepository:
    """
    Data access layer for ``Visit`` records.

    Provides creation, retrieval, and update operations scoped to a
    single SQLAlchemy session. Visit records are always associated with
    a parent ``Patient`` via ``patient_id``.
    """

    def __init__(self, db: Session):
        """
        Initialise the repository with a SQLAlchemy session.

        Args:
            db (Session): Active SQLAlchemy database session used for
                all queries and mutations performed by this repository.
        """
        self.db = db

    def create(self, patient_id: int, **kwargs) -> Visit:
        """
        Create and persist a new ``Visit`` record linked to a patient.

        Args:
            patient_id (int): Primary key of the ``Patient`` this visit
                belongs to.
            **kwargs: Additional column values for the ``Visit`` row
                (e.g. ``visit_date``, ``notes``, ``diagnosis``).

        Returns:
            Visit: The newly created and flushed ``Visit`` instance.
        """
        visit = Visit(patient_id=patient_id, **kwargs)
        self.db.add(visit)
        self.db.flush()
        return visit

    def get_by_id(self, visit_id: int) -> Visit | None:
        """
        Retrieve a single visit by primary key.

        Args:
            visit_id (int): Primary key of the visit to retrieve.

        Returns:
            Visit | None: The matching ``Visit`` instance, or ``None``
                if no record with the given ID exists.
        """
        return self.db.query(Visit).filter(Visit.id == visit_id).first()

    def get_patient_visits(self, patient_id: int):
        """
        Retrieve all visits for a specific patient, newest first.

        Args:
            patient_id (int): Primary key of the patient whose visits
                should be retrieved.

        Returns:
            list[Visit]: All ``Visit`` records for the patient ordered
                by ``visit_date`` descending.
        """
        return (
            self.db.query(Visit)
            .filter(Visit.patient_id == patient_id)
            .order_by(desc(Visit.visit_date))
            .all()
        )

    def get_last_visit(self, patient_id: int) -> Visit | None:
        """
        Retrieve the most recent visit for a specific patient.

        Args:
            patient_id (int): Primary key of the patient.

        Returns:
            Visit | None: The ``Visit`` record with the latest
                ``visit_date`` for the patient, or ``None`` if the
                patient has no visits.
        """
        return (
            self.db.query(Visit)
            .filter(Visit.patient_id == patient_id)
            .order_by(desc(Visit.visit_date))
            .first()
        )

    def update(self, visit_id: int, **kwargs) -> Visit | None:
        """
        Update an existing visit record with new field values.

        Args:
            visit_id (int): Primary key of the visit to update.
            **kwargs: Column-value pairs to set on the ``Visit`` instance.

        Returns:
            Visit | None: The updated ``Visit`` instance after flushing,
                or ``None`` if no visit with the given ID exists.
        """
        visit = self.get_by_id(visit_id)
        if visit:
            for k, v in kwargs.items():
                setattr(visit, k, v)
            self.db.flush()
        return visit


class TrainingImageRepository:
    """
    Data access layer for ``TrainingImage`` records.

    Supports bulk insertion, split-based retrieval, aggregated
    statistics, and full-table queries used throughout the training
    and evaluation pipelines.
    """

    def __init__(self, db: Session):
        """
        Initialise the repository with a SQLAlchemy session.

        Args:
            db (Session): Active SQLAlchemy database session used for
                all queries and mutations performed by this repository.
        """
        self.db = db

    def bulk_insert(self, images: list[dict]):
        """
        Bulk-insert a list of training image records.

        Constructs ``TrainingImage`` ORM objects from the provided
        dictionaries and uses ``bulk_save_objects`` for efficiency.

        Args:
            images (list[dict]): List of dictionaries where each entry
                contains column-value pairs for a ``TrainingImage`` row.
                Common keys include ``file_path``, ``mask_path``,
                ``label``, ``split``, ``dataset_source``, ``width``,
                and ``height``.

        Returns:
            int: Number of records inserted into the database.
        """
        objs = [TrainingImage(**img) for img in images]
        self.db.bulk_save_objects(objs)
        self.db.flush()
        return len(objs)

    def get_by_split(self, split: str):
        """
        Retrieve all training images belonging to a specific data split.

        Args:
            split (str): Name of the split to query. Expected values are
                ``"train"``, ``"val"``, or ``"test"``.

        Returns:
            list[TrainingImage]: All ``TrainingImage`` records whose
                ``split`` column matches the provided value.
        """
        return self.db.query(TrainingImage).filter(TrainingImage.split == split).all()

    def get_stats(self):
        """
        Return aggregate counts grouped by label and split.

        Useful for quickly inspecting the class and split distribution
        of the full training image collection without loading all rows.

        Returns:
            list[tuple[int, str, int]]: Each tuple contains
                ``(label, split, count)`` where ``count`` is the number
                of records with that label/split combination.
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
        Retrieve every training image record from the database.

        Returns:
            list[TrainingImage]: Complete list of all ``TrainingImage``
                instances stored in the database.
        """
        return self.db.query(TrainingImage).all()
