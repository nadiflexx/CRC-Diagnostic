# tests/test_config_settings.py
"""
Tests for configuration settings modules.
"""


class TestDBSettings:
    def test_default_values(self):
        from src.config.settings import DBSettings

        settings = DBSettings()
        assert settings.DB_HOST == "localhost"
        assert settings.DB_PORT == 5432
        assert settings.DB_NAME == "colon_diagnosis"
        assert settings.DB_USER == "admin"

    def test_url_property(self):
        from src.config.settings import DBSettings

        settings = DBSettings()
        url = settings.url
        assert "postgresql://" in url
        assert "localhost" in url
        assert "5432" in url
        assert "colon_diagnosis" in url

    def test_url_contains_credentials(self):
        from src.config.settings import DBSettings

        settings = DBSettings()
        url = settings.url
        assert "admin" in url

    def test_password_is_secret(self):
        from src.config.settings import DBSettings

        settings = DBSettings()
        # SecretStr shouldn't expose in repr
        assert "admin123" not in repr(settings.DB_PASSWORD)


class TestModelSettings:
    def test_default_values(self):
        from src.config.settings import ModelSettings

        settings = ModelSettings()
        assert settings.IMAGE_SIZE == 384
        assert settings.IMAGE_BACKBONE == "efficientnet_b4"
        assert settings.BATCH_SIZE == 16
        assert settings.NUM_WORKERS == 0

    def test_tabular_model_type(self):
        from src.config.settings import ModelSettings

        settings = ModelSettings()
        assert settings.TABULAR_MODEL_TYPE in ("xgboost", "lightgbm")

    def test_learning_rate_type(self):
        from src.config.settings import ModelSettings

        settings = ModelSettings()
        assert isinstance(settings.LEARNING_RATE, float)

    def test_device_value(self):
        from src.config.settings import ModelSettings

        settings = ModelSettings()
        assert settings.DEVICE in ("cuda", "cpu")


class TestDiagnosisSettings:
    def test_threshold_values(self):
        from src.config.settings import DiagnosisSettings

        settings = DiagnosisSettings()
        assert 0 < settings.CANCER_THRESHOLD < 1
        assert settings.RISK_LOW < settings.RISK_MODERATE < settings.RISK_HIGH

    def test_fusion_weights_sum(self):
        from src.config.settings import DiagnosisSettings

        settings = DiagnosisSettings()
        total = settings.FUSION_IMAGE_WEIGHT + settings.FUSION_TABULAR_WEIGHT
        assert abs(total - 1.0) < 1e-6

    def test_min_recall_target(self):
        from src.config.settings import DiagnosisSettings

        settings = DiagnosisSettings()
        assert settings.MIN_RECALL_TARGET >= 0.9

    def test_risk_thresholds_range(self):
        from src.config.settings import DiagnosisSettings

        settings = DiagnosisSettings()
        assert 0 < settings.RISK_LOW < 1
        assert 0 < settings.RISK_MODERATE < 1
        assert 0 < settings.RISK_HIGH < 1


class TestAPISettings:
    def test_default_values(self):
        from src.config.settings import APISettings

        settings = APISettings()
        assert settings.API_HOST == "0.0.0.0"
        assert settings.API_PORT == 8000
        assert settings.MAX_UPLOAD_SIZE_MB == 50

    def test_cors_origins(self):
        from src.config.settings import APISettings

        settings = APISettings()
        assert isinstance(settings.CORS_ORIGINS, list)

    def test_api_workers(self):
        from src.config.settings import APISettings

        settings = APISettings()
        assert settings.API_WORKERS >= 1


class TestSingletonInstances:
    def test_singletons_importable(self):
        from src.config.settings import api, db, diagnosis, model

        assert db is not None
        assert model is not None
        assert diagnosis is not None
        assert api is not None
