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

    # ── Data ──────────────────────────────────────────────────────────────────
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
    def PROCESSED_TABULAR(self) -> Path:
        """Processed tabular data: data/processed/tabular/"""
        return self.PROCESSED / "tabular"

    @property
    def SYNTHETIC(self) -> Path:
        return self.DATA / "synthetic"

    # ── Multi-source dataset dirs ──────────────────────────────────────────────
    @property
    def COLON_CLEAN(self) -> Path:
        return self.DATA / "colon_clean"

    @property
    def COLON_PROCESSED(self) -> Path:
        return self.DATA / "colon_processed"

    @property
    def COLON_TISSUE_ONLY(self) -> Path:
        return self.DATA / "colon_processed_tissue_only"

    @property
    def HYPERKVASIR_RAW(self) -> Path:
        return self.RAW / "hyperkvasir_raw"

    # ── Models ────────────────────────────────────────────────────────────────
    @property
    def MODELS_ROOT(self) -> Path:
        return self.ROOT / "models"

    @property
    def MODELS(self) -> Path:
        return self.ROOT / "models" / "saved"

    # ── Logs ──────────────────────────────────────────────────────────────────
    @property
    def LOGS(self) -> Path:
        return self.ROOT / "logs"

    # ── Uploads & Reports (API) ───────────────────────────────────────────────
    @property
    def UPLOADS(self) -> Path:
        return self.DATA / "uploads"

    @property
    def REPORTS(self) -> Path:
        return self.PROCESSED / "reports"

    @property
    def UPLOAD_IMAGES(self) -> Path:
        return self.PROCESSED / "uploads"

    # ── Model artifact paths ───────────────────────────────────────────────────
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

    # ── CSV tabular pipeline paths ─────────────────────────────────────────────
    #
    #  data/
    #  ├── raw/tabular/
    #  │   └── colorectal_cancer_dataset.csv          ← CSV_RAW_TABULAR
    #  └── processed/tabular/
    #      └── colorectal_cancer_full_dataset.csv     ← CSV_FULL_TABULAR
    #
    #  models/saved/csv_mlp/
    #  ├── mlp_model.pkl
    #  ├── scaler.pkl
    #  ├── model_config.pkl
    #  └── plots/

    @property
    def CSV_RAW_TABULAR(self) -> Path:
        """Raw Kaggle CSV → data/raw/tabular/colorectal_cancer_dataset.csv"""
        return self.RAW_TABULAR / "colorectal_cancer_dataset.csv"

    @property
    def CSV_FULL_TABULAR(self) -> Path:
        """Balanced dataset → data/processed/tabular/colorectal_cancer_full_dataset.csv"""
        return self.PROCESSED_TABULAR / "colorectal_cancer_full_dataset.csv"

    @property
    def CSV_MLP_MODEL_DIR(self) -> Path:
        """Artefacts dir: models/saved/csv_mlp/"""
        return self.MODELS / "csv_mlp"

    @property
    def CSV_MLP_MODEL_PATH(self) -> Path:
        return self.CSV_MLP_MODEL_DIR / "mlp_model.pkl"

    @property
    def CSV_MLP_SCALER_PATH(self) -> Path:
        return self.CSV_MLP_MODEL_DIR / "scaler.pkl"

    @property
    def CSV_MLP_CONFIG_PATH(self) -> Path:
        return self.CSV_MLP_MODEL_DIR / "model_config.pkl"

    @property
    def CSV_PLOTS_DIR(self) -> Path:
        return self.CSV_MLP_MODEL_DIR / "plots"

    @property
    def CSV_ANALYSIS_DIR(self) -> Path:
        """EDA outputs: heatmaps, pairplots, clean CSV."""
        return self.DATA / "analysis"

    # ── Auto-create directories on import ─────────────────────────────────────
    @model_validator(mode="after")
    def _create_dirs(self) -> "PathSettings":
        directories = [
            self.DATA,
            self.RAW,
            self.RAW / "limuc",
            self.HYPERKVASIR_RAW,
            self.PROCESSED,
            self.PROCESSED_TABULAR,   # ← new
            self.MODELS,
            self.LOGS,
            self.NOTEBOOKS,
            self.RAW_TABULAR,
            self.CSV_MLP_MODEL_DIR,
            self.CSV_PLOTS_DIR,
            self.CSV_ANALYSIS_DIR,
        ]
        for d in directories:
            d.mkdir(parents=True, exist_ok=True)
        return self

    model_config = SettingsConfigDict(extra="ignore")


paths = PathSettings()