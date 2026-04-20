# tests/test_config_paths.py
"""
Tests for path configuration.
"""

from pathlib import Path


class TestPathSettings:
    def test_paths_importable(self):
        from src.config.paths import paths

        assert paths is not None

    def test_root_is_path(self):
        from src.config.paths import paths

        assert isinstance(paths.ROOT, Path)

    def test_data_path(self):
        from src.config.paths import paths

        assert isinstance(paths.DATA, Path)
        assert paths.DATA == paths.ROOT / "data"

    def test_models_path(self):
        from src.config.paths import paths

        assert isinstance(paths.MODELS, Path)

    def test_logs_path(self):
        from src.config.paths import paths

        assert isinstance(paths.LOGS, Path)
        assert paths.LOGS.name == "logs"

    def test_upload_images_path(self):
        from src.config.paths import paths

        assert isinstance(paths.UPLOAD_IMAGES, Path)

    def test_reports_path(self):
        from src.config.paths import paths

        assert isinstance(paths.REPORTS, Path)

    def test_classifier_checkpoint(self):
        from src.config.paths import paths

        assert paths.CLASSIFIER_CHECKPOINT.name == "best_classifier.pth"

    def test_tabular_model_path(self):
        from src.config.paths import paths

        assert paths.TABULAR_MODEL_PATH.name == "tabular_model.pkl"

    def test_ensemble_config_path(self):
        from src.config.paths import paths

        assert paths.ENSEMBLE_CONFIG_PATH.name == "ensemble_config.json"

    def test_raw_path(self):
        from src.config.paths import paths

        assert paths.RAW == paths.DATA / "raw"

    def test_processed_path(self):
        from src.config.paths import paths

        assert paths.PROCESSED == paths.DATA / "processed"

    def test_synthetic_path(self):
        from src.config.paths import paths

        assert paths.SYNTHETIC == paths.DATA / "synthetic"

    def test_hyperkvasir_raw_path(self):
        from src.config.paths import paths

        assert paths.HYPERKVASIR_RAW.name == "hyperkvasir_raw"

    def test_colon_clean_path(self):
        from src.config.paths import paths

        assert paths.COLON_CLEAN.name == "colon_clean"

    def test_tabular_processed_path(self):
        from src.config.paths import paths

        assert paths.TABULAR_PROCESSED.name == "tabular_processed"

    def test_notebooks_path(self):
        from src.config.paths import paths

        assert paths.NOTEBOOKS.name == "notebooks"

    def test_dirs_created(self):
        from src.config.paths import paths

        assert paths.MODELS.exists()
        assert paths.LOGS.exists()

    def test_tissue_classifier_checkpoint(self):
        from src.config.paths import paths

        assert paths.TISSUE_CLASSIFIER_CHECKPOINT.name == "best_tissue_classifier.pth"

    def test_segmenter_checkpoint(self):
        from src.config.paths import paths

        assert paths.SEGMENTER_CHECKPOINT.name == "best_segmenter.pth"

    def test_tabular_preprocessor_path(self):
        from src.config.paths import paths

        assert paths.TABULAR_PREPROCESSOR_PATH.name == "tabular_preprocessor.pkl"
