"""
Environment-based configuration via Pydantic Settings.
Single source of truth for all runtime parameters.
"""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.paths import _ROOT


# ══════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════
class DBSettings(BaseSettings):
    """PostgreSQL connection settings."""

    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "colon_diagnosis"
    DB_USER: str = "admin"
    DB_PASSWORD: SecretStr = SecretStr("admin123")

    @property
    def url(self) -> str:
        pwd = self.DB_PASSWORD.get_secret_value()
        return f"postgresql://{self.DB_USER}:{pwd}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# ══════════════════════════════════════════════════════════════════
#  MODEL HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════
class ModelSettings(BaseSettings):
    """ML/DL model hyperparameters."""

    IMAGE_SIZE: int = 384
    IMAGE_BACKBONE: str = "efficientnet_b4"
    EPOCHS_IMAGE: int = 50
    EPOCHS_SEGMENTER: int = 50

    TABULAR_MODEL_TYPE: str = "xgboost"
    EPOCHS_TABULAR: int = 100

    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 5e-5
    WEIGHT_DECAY: float = 1e-4
    NUM_WORKERS: int = 0

    DEVICE: str = "cuda"

    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# ══════════════════════════════════════════════════════════════════
#  DIAGNOSIS THRESHOLDS
# ══════════════════════════════════════════════════════════════════
class DiagnosisSettings(BaseSettings):
    """Diagnosis decision thresholds."""

    CANCER_THRESHOLD: float = 0.3

    RISK_LOW: float = 0.3
    RISK_MODERATE: float = 0.5
    RISK_HIGH: float = 0.7

    MIN_RECALL_TARGET: float = 0.95

    FUSION_IMAGE_WEIGHT: float = 0.6
    FUSION_TABULAR_WEIGHT: float = 0.4

    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# ══════════════════════════════════════════════════════════════════
#  API SETTINGS
# ══════════════════════════════════════════════════════════════════
class APISettings(BaseSettings):
    """FastAPI configuration."""

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    API_WORKERS: int = 1

    CORS_ORIGINS: list[str] = ["*"]

    MAX_UPLOAD_SIZE_MB: int = 50

    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# ══════════════════════════════════════════════════════════════════
#  SINGLETON INSTANCES
# ══════════════════════════════════════════════════════════════════
db = DBSettings()
model = ModelSettings()
diagnosis = DiagnosisSettings()
api = APISettings()
