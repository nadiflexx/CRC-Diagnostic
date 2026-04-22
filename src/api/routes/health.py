"""
Health & initialization endpoint.
Ensures DB tables are created on first run.
"""

from fastapi import APIRouter
from sqlalchemy import inspect

from src.database.connection import engine
from src.database.models import Base

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/init")
def initialize_database() -> dict:
    """
    Idempotently create all database tables if they do not exist.
    Safe to call multiple times (uses CREATE TABLE IF NOT EXISTS logic).

    Returns:
        dict: Status message and list of tables created or already present.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    Base.metadata.create_all(bind=engine, checkfirst=True)

    inspector = inspect(engine)
    all_tables = set(inspector.get_table_names())
    new_tables = all_tables - existing_tables

    return {
        "status": "ok",
        "tables_already_existed": sorted(existing_tables),
        "tables_created": sorted(new_tables),
        "all_tables": sorted(all_tables),
    }


@router.get("/ping")
def ping() -> dict:
    """Simple liveness probe."""
    return {"status": "alive"}
