"""
Configuration module.
"""

from src.config.logger import log
from src.config.paths import paths
from src.config.settings import (
    api,
    db,
    diagnosis,
    model,
)

__all__ = [
    "paths",
    "db",
    "model",
    "diagnosis",
    "api",
    "log",
]
