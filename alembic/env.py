# alembic/env.py
"""
Alembic environment configuration.
Reads DATABASE_URL from environment or .env file.
Imports all models so autogenerate detects every table.
"""

from __future__ import annotations

from logging.config import fileConfig
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

from alembic import context  # type: ignore

# ── Añadir raíz del proyecto al path ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Cargar variables de entorno (.env) ────────────────────────────────────
load_dotenv(ROOT / ".env")

from src.database import models  # noqa: E402, F401
from src.database.models import Base  # noqa: E402

# ── Alembic Config ────────────────────────────────────────────────────────
config = context.config

# Leer la URL de la BD desde variable de entorno
database_url = os.getenv("DATABASE_URL")
if not database_url:
    try:
        from src.config.settings import db

        database_url = str(db.url)
    except Exception:
        db_path = ROOT / "data" / "endoaid.db"
        database_url = f"sqlite:///{db_path}"

config.set_main_option("sqlalchemy.url", database_url)

# ── Logging ───────────────────────────────────────────────────────────────
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Metadata para autogenerate ────────────────────────────────────────────
target_metadata = Base.metadata


# ── Helpers ───────────────────────────────────────────────────────────────


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode (without DB connection).
    Generates SQL script to stdout.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode (with live DB connection).
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
