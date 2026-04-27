from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import db

engine = create_engine(
    db.url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager that provides a transactional database session.

    Yields a ``Session`` instance bound to ``SessionLocal``. If the
    block completes without raising an exception the transaction is
    committed automatically. On any exception the transaction is rolled
    back and the exception is re-raised. The session is always closed in
    the ``finally`` block regardless of outcome.

    Yields:
        Session: An active SQLAlchemy ``Session`` ready for use.

    Raises:
        Exception: Any exception raised inside the ``with`` block is
            propagated after the rollback.

    Example::

        with get_db() as db:
            repo = PatientRepository(db)
            repo.create(first_name="Jane", last_name="Doe")
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_dependency() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session per request.

    Intended to be used with FastAPI's ``Depends`` mechanism. Unlike
    ``get_db``, this generator does **not** commit or roll back; the
    caller is responsible for transaction management. The session is
    closed in the ``finally`` block after the response has been sent.

    Yields:
        Session: An active SQLAlchemy ``Session`` for the duration of
            the HTTP request.

    Example::

        @app.get("/patients")
        def list_patients(db: Session = Depends(get_db_dependency)):
            return PatientRepository(db).get_all()
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
