"""
Configuration module.
"""

from src.config.logger import log
from src.config.paths import paths
from src.config.settings import (
    api,
    clinical,
    db,
    diagnosis,
    gdc,
    model,
    pipeline,
)

__all__ = [
    "paths",
    "db",
    "gdc",
    "model",
    "clinical",
    "diagnosis",
    "api",
    "pipeline",
    "log",
]
