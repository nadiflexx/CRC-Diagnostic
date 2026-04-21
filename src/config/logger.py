"""
Logging Configuration.
Uses Loguru for structured, colorful, and persistent logging.
"""

import sys

from loguru import logger

from src.config.paths import paths

logger.remove()

logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=True,
)

logger.add(
    paths.LOGS / "crc_diagnostic_{time:YYYY-MM-DD}.log",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} — {message}"
    ),
    level="DEBUG",
    rotation="00:00",
    retention="30 days",
    compression="zip",
    encoding="utf-8",
)

log = logger
