"""
Path configuration — single source of truth for all directories.
"""

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parent.parent.parent


class PathSettings(BaseSettings):
    """All project paths, auto-created on instantiation."""

    ROOT: Path = _ROOT

    # ── Data ──
    @property
    def DATA(self) -> Path:
        return self.ROOT / "data"

    @property
    def RAW(self) -> Path:
        return self.DATA / "raw"

    @property
    def RAW_IMAGES(self) -> Path:
        return self.RAW / "images"

    @property
    def RAW_TABULAR(self) -> Path:
        return self.RAW / "tabular"

    @property
    def PROCESSED(self) -> Path:
        return self.DATA / "processed"

    @property
    def SYNTHETIC(self) -> Path:
        return self.DATA / "synthetic"

    # ── Multi-source dataset dirs ──
    @property
    def COLON_CLEAN(self) -> Path:
        return self.DATA / "colon_clean"

    @property
    def COLON_PROCESSED(self) -> Path:
        return self.DATA / "colon_processed"

    @property
    def TABULAR_PROCESSED(self) -> Path:
        return self.DATA / "tabular_processed"

    @property
    def TABULAR_CLEAN(self) -> Path:
        return self.DATA / "tabular_clean"

    @property
    def COLON_TISSUE_ONLY(self) -> Path:
        return self.DATA / "colon_processed_tissue_only"

    @property
    def HYPERKVASIR_RAW(self) -> Path:
        return self.RAW / "hyperkvasir_raw"

    # ── Models ──
    @property
    def MODELS_ROOT(self) -> Path:
        return self.ROOT / "models"

    @property
    def MODELS(self) -> Path:
        return self.ROOT / "models" / "saved"

    # ── Logs ──
    @property
    def LOGS(self) -> Path:
        return self.ROOT / "logs"

    # ── Uploads & Reports (API) ──
    @property
    def UPLOADS(self) -> Path:
        return self.DATA / "uploads"

    @property
    def REPORTS(self) -> Path:
        return self.DATA / "processed" / "reports"

    @property
    def UPLOAD_IMAGES(self) -> Path:
        return self.DATA / "processed" / "uploads"

    # ── Model artifact paths ──
    @property
    def CLASSIFIER_CHECKPOINT(self) -> Path:
        return self.MODELS / "best_classifier.pth"

    @property
    def TISSUE_CLASSIFIER_CHECKPOINT(self) -> Path:
        return self.MODELS / "best_tissue_classifier.pth"

    @property
    def SEGMENTER_CHECKPOINT(self) -> Path:
        return self.MODELS / "best_segmenter.pth"

    @property
    def TABULAR_MODEL_PATH(self) -> Path:
        return self.MODELS / "tabular_model.pkl"

    @property
    def TABULAR_PREPROCESSOR_PATH(self) -> Path:
        return self.MODELS / "tabular_preprocessor.pkl"

    @property
    def ENSEMBLE_CONFIG_PATH(self) -> Path:
        return self.MODELS / "ensemble_config.json"

    @property
    def NOTEBOOKS(self) -> Path:
        return self.ROOT / "notebooks"

    # ── Create all directories ──
    @model_validator(mode="after")
    def _create_dirs(self) -> "PathSettings":
        directories = [
            self.DATA,
            self.RAW,
            self.RAW / "limuc",
            self.HYPERKVASIR_RAW,
            self.MODELS,
            self.LOGS,
            self.NOTEBOOKS,
        ]
        for d in directories:
            d.mkdir(parents=True, exist_ok=True)
        return self

    model_config = SettingsConfigDict(extra="ignore")


paths = PathSettings()
