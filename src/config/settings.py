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
#  GDC API (Genomic Data Commons)
# ══════════════════════════════════════════════════════════════════
class GDCSettings(BaseSettings):
    """GDC API configuration."""

    GDC_BASE_URL: str = "https://api.gdc.cancer.gov"
    GDC_TIMEOUT: int = 60
    GDC_MAX_RETRIES: int = 5
    GDC_RETRY_DELAY: int = 2
    GDC_RATE_LIMIT_DELAY: float = 0.5
    GDC_MAX_CASES: int = 1000

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

    # Image model
    IMAGE_SIZE: int = 384
    IMAGE_BACKBONE: str = "efficientnet_b4"
    EPOCHS_IMAGE: int = 50
    EPOCHS_SEGMENTER: int = 50

    # Tabular model
    TABULAR_MODEL_TYPE: str = "xgboost"  # xgboost or lightgbm
    EPOCHS_TABULAR: int = 100

    # Training
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 5e-5
    WEIGHT_DECAY: float = 1e-4
    NUM_WORKERS: int = 0

    # Device
    DEVICE: str = "cuda"  # cuda or cpu

    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# ══════════════════════════════════════════════════════════════════
#  CLINICAL THRESHOLDS
# ══════════════════════════════════════════════════════════════════
class ClinicalSettings(BaseSettings):
    """Clinical thresholds for cancer markers."""

    # CEA (Carcinoembryonic Antigen)
    CEA_NORMAL_MAX: float = 3.0  # ng/mL
    CEA_ELEVATED: float = 5.0  # ng/mL
    CEA_HIGH: float = 10.0  # ng/mL

    # CA 19-9
    CA19_9_NORMAL_MAX: float = 37.0  # U/mL

    # Blood markers
    HEMOGLOBIN_LOW: float = 12.0  # g/dL (anemia threshold)
    HEMOGLOBIN_CRITICAL: float = 8.0  # g/dL
    ALBUMIN_LOW: float = 3.5  # g/dL
    FERRITIN_LOW: float = 20.0  # ng/mL
    CRP_ELEVATED: float = 1.0  # mg/dL

    # Platelet count
    PLATELET_LOW: float = 150.0  # x10³/μL
    PLATELET_HIGH: float = 400.0  # x10³/μL

    model_config = SettingsConfigDict(extra="ignore")


# ══════════════════════════════════════════════════════════════════
#  DIAGNOSIS THRESHOLDS
# ══════════════════════════════════════════════════════════════════
class DiagnosisSettings(BaseSettings):
    """Diagnosis decision thresholds."""

    # Main classification threshold (low to minimize false negatives)
    CANCER_THRESHOLD: float = 0.3

    # Risk level thresholds
    RISK_LOW: float = 0.3
    RISK_MODERATE: float = 0.5
    RISK_HIGH: float = 0.7

    # Minimum recall target
    MIN_RECALL_TARGET: float = 0.95

    # Multimodal fusion weights (default, can be learned)
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

    # CORS
    CORS_ORIGINS: list[str] = ["*"]

    # Upload limits
    MAX_UPLOAD_SIZE_MB: int = 50

    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# ══════════════════════════════════════════════════════════════════
#  PIPELINE SETTINGS
# ══════════════════════════════════════════════════════════════════
class PipelineSettings(BaseSettings):
    """Data pipeline configuration."""

    # Kaggle datasets
    KAGGLE_DATASETS: dict = {
        "tabular_risk": "ankushpanday1/colorectal-cancer-risk-and-survival-data",
        "kvasir_seg": "ipythonx/kvasirseg",
        "curated_colon": "francismon/curated-colon-dataset-for-deep-learning",
        "cvc_clinicdb": "orvile/cvc-clinicdb",
    }

    # Data balancing
    MIN_SAMPLES_PER_CLASS: int = 2000
    BALANCE_STRATEGY: str = "smote"  # smote, adasyn, smote_tomek

    # Train/Val/Test split
    TEST_SIZE: float = 0.15
    VAL_SIZE: float = 0.15

    # Synthetic data
    SYNTHETIC_HEALTHY_RATIO: float = 1.0  # 1:1 with cancer

    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# ══════════════════════════════════════════════════════════════════
#  SINGLETON INSTANCES
# ══════════════════════════════════════════════════════════════════
db = DBSettings()
gdc = GDCSettings()
model = ModelSettings()
clinical = ClinicalSettings()
diagnosis = DiagnosisSettings()
api = APISettings()
pipeline = PipelineSettings()
