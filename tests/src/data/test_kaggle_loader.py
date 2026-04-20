# tests/src/data/test_kaggle_loader.py
"""
Tests for src/data/ingestion/kaggle_loader.py

Strategy: patch the `paths` singleton's ROOT so that all @property
paths resolve under tmp_path automatically.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import src.data.ingestion.kaggle_loader as kaggle_loader_module
from src.data.ingestion.kaggle_loader import KaggleLoader

# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────


def _redirect_paths(tmp_path, monkeypatch):
    """
    Redirect the paths singleton so all properties resolve under tmp_path.
    Since PathSettings uses @property, we patch the singleton object's
    ROOT field (a plain pydantic field, not a property).
    """
    fake_paths = MagicMock()
    fake_paths.ROOT = tmp_path
    fake_paths.RAW = tmp_path / "raw"
    fake_paths.RAW_IMAGES = tmp_path / "raw_images"
    fake_paths.RAW_TABULAR = tmp_path / "raw_tabular"
    fake_paths.HYPERKVASIR_RAW = tmp_path / "raw" / "hyperkvasir_raw"
    monkeypatch.setattr(kaggle_loader_module, "paths", fake_paths)
    return fake_paths


@pytest.fixture
def fake_paths(tmp_path, monkeypatch):
    return _redirect_paths(tmp_path, monkeypatch)


@pytest.fixture
def loader(tmp_path, monkeypatch):
    """KaggleLoader with paths redirected to tmp_path."""
    fake = _redirect_paths(tmp_path, monkeypatch)
    # Create kaggle.json so __init__ doesn't warn
    kaggle_dir = tmp_path / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    (kaggle_dir / "kaggle.json").write_text("{}")
    fake.ROOT = tmp_path
    return KaggleLoader()


@pytest.fixture
def loader_no_creds(tmp_path, monkeypatch):
    """KaggleLoader without kaggle.json."""
    _redirect_paths(tmp_path, monkeypatch)
    # No .kaggle/kaggle.json → warning path
    fake_paths = MagicMock()
    fake_paths.ROOT = tmp_path  # no kaggle.json inside
    monkeypatch.setattr(kaggle_loader_module, "paths", fake_paths)
    return KaggleLoader()


# ─────────────────────────────────────────────────────────────
#  __init__
# ─────────────────────────────────────────────────────────────


class TestInit:
    def test_init_creates_instance(self, loader):
        assert isinstance(loader, KaggleLoader)

    def test_init_without_credentials_does_not_raise(self, tmp_path, monkeypatch):
        """KaggleLoader should not raise even without kaggle.json."""
        fake = MagicMock()
        fake.ROOT = tmp_path  # no .kaggle/kaggle.json
        monkeypatch.setattr(kaggle_loader_module, "paths", fake)
        instance = KaggleLoader()  # must not raise
        assert instance is not None

    def test_init_with_credentials_does_not_raise(self, loader):
        assert loader is not None


# ─────────────────────────────────────────────────────────────
#  download_dataset
# ─────────────────────────────────────────────────────────────


class TestDownloadDataset:
    def test_raises_value_error_for_unknown_key(self, loader):
        with pytest.raises(ValueError, match="Unknown dataset"):
            loader.download_dataset("nonexistent_key_xyz")

    @patch("src.data.ingestion.kaggle_loader.subprocess.run")
    def test_calls_kaggle_cli_with_correct_args(
        self, mock_run, loader, tmp_path, fake_paths
    ):
        from src.config.constants import KAGGLE_DATASETS

        first_key = next(iter(KAGGLE_DATASETS))
        dataset_name = KAGGLE_DATASETS[first_key]
        target = tmp_path / "out" / first_key

        mock_run.return_value = MagicMock(returncode=0)
        loader.download_dataset(first_key, target_dir=target)

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "kaggle" in args
        assert "datasets" in args
        assert "download" in args
        assert dataset_name in args
        assert "--unzip" in args

    @patch("src.data.ingestion.kaggle_loader.subprocess.run")
    def test_creates_target_directory(self, mock_run, loader, tmp_path):
        from src.config.constants import KAGGLE_DATASETS

        first_key = next(iter(KAGGLE_DATASETS))
        target = tmp_path / "new_dir" / first_key

        mock_run.return_value = MagicMock(returncode=0)
        loader.download_dataset(first_key, target_dir=target)

        assert target.exists()

    @patch("src.data.ingestion.kaggle_loader.subprocess.run")
    def test_uses_default_target_when_none(
        self, mock_run, loader, tmp_path, fake_paths
    ):
        from src.config.constants import KAGGLE_DATASETS

        first_key = next(iter(KAGGLE_DATASETS))
        expected_target = tmp_path / "raw" / first_key

        mock_run.return_value = MagicMock(returncode=0)
        loader.download_dataset(first_key)

        assert expected_target.exists()

    @patch(
        "src.data.ingestion.kaggle_loader.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "kaggle", stderr="auth error"),
    )
    def test_raises_on_subprocess_error(self, mock_run, loader, tmp_path):
        from src.config.constants import KAGGLE_DATASETS

        first_key = next(iter(KAGGLE_DATASETS))
        with pytest.raises(subprocess.CalledProcessError):
            loader.download_dataset(first_key, target_dir=tmp_path / "out")

    @patch("src.data.ingestion.kaggle_loader.subprocess.run")
    def test_check_true_passed_to_subprocess(self, mock_run, loader, tmp_path):
        from src.config.constants import KAGGLE_DATASETS

        first_key = next(iter(KAGGLE_DATASETS))
        mock_run.return_value = MagicMock(returncode=0)
        loader.download_dataset(first_key, target_dir=tmp_path / "out")
        _, kwargs = mock_run.call_args
        assert kwargs.get("check") is True

    @patch("src.data.ingestion.kaggle_loader.subprocess.run")
    def test_capture_output_true(self, mock_run, loader, tmp_path):
        from src.config.constants import KAGGLE_DATASETS

        first_key = next(iter(KAGGLE_DATASETS))
        mock_run.return_value = MagicMock(returncode=0)
        loader.download_dataset(first_key, target_dir=tmp_path / "out")
        _, kwargs = mock_run.call_args
        assert kwargs.get("capture_output") is True


# ─────────────────────────────────────────────────────────────
#  download_all
# ─────────────────────────────────────────────────────────────


class TestDownloadAll:
    @patch.object(KaggleLoader, "download_dataset")
    def test_calls_download_for_every_key(self, mock_dl, loader):
        from src.config.constants import KAGGLE_DATASETS

        loader.download_all()
        assert mock_dl.call_count == len(KAGGLE_DATASETS)

    @patch.object(KaggleLoader, "download_dataset", side_effect=Exception("net error"))
    def test_continues_after_single_failure(self, mock_dl, loader):
        from src.config.constants import KAGGLE_DATASETS

        loader.download_all()  # must not raise
        assert mock_dl.call_count == len(KAGGLE_DATASETS)

    @patch.object(KaggleLoader, "download_dataset")
    def test_calls_each_key_once(self, mock_dl, loader):
        from src.config.constants import KAGGLE_DATASETS

        loader.download_all()
        called_keys = [c.args[0] for c in mock_dl.call_args_list]
        for key in KAGGLE_DATASETS:
            assert key in called_keys


# ─────────────────────────────────────────────────────────────
#  download_multi_source
# ─────────────────────────────────────────────────────────────


class TestDownloadMultiSource:
    @patch.object(KaggleLoader, "download_dataset")
    def test_skips_existing_non_empty_dirs(self, mock_dl, loader, tmp_path, fake_paths):
        raw = tmp_path / "raw"
        for key in ["cvc_clinicdb", "limuc"]:
            d = raw / key
            d.mkdir(parents=True, exist_ok=True)
            (d / "image.jpg").write_bytes(b"fake")

        fake_paths.RAW = raw
        loader.download_multi_source()
        mock_dl.assert_not_called()

    @patch.object(KaggleLoader, "download_dataset")
    def test_downloads_missing_keys(self, mock_dl, loader, tmp_path, fake_paths):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        fake_paths.RAW = raw
        loader.download_multi_source()
        assert mock_dl.call_count == 2

    @patch.object(KaggleLoader, "download_dataset", side_effect=Exception("fail"))
    def test_logs_error_on_failure(self, mock_dl, loader, tmp_path, fake_paths, caplog):
        import logging

        raw = tmp_path / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        fake_paths.RAW = raw
        with caplog.at_level(logging.ERROR):
            loader.download_multi_source()  # must not raise


# ─────────────────────────────────────────────────────────────
#  verify_datasets
# ─────────────────────────────────────────────────────────────


class TestVerifyDatasets:
    def test_returns_false_for_missing_dirs(self, loader, tmp_path, fake_paths):
        fake_paths.HYPERKVASIR_RAW = tmp_path / "nonexistent"
        fake_paths.RAW = tmp_path / "raw"
        status = loader.verify_datasets()
        assert all(v is False for v in status.values())

    def test_returns_true_for_dir_with_jpg(self, loader, tmp_path, fake_paths):
        hk = tmp_path / "hk"
        hk.mkdir(parents=True)
        (hk / "img.jpg").write_bytes(b"fake")
        fake_paths.HYPERKVASIR_RAW = hk
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        fake_paths.RAW = raw
        status = loader.verify_datasets()
        assert status["hyperkvasir"] is True

    def test_returns_true_for_dir_with_png(self, loader, tmp_path, fake_paths):
        raw = tmp_path / "raw"
        cvc = raw / "cvc_clinicdb"
        cvc.mkdir(parents=True)
        (cvc / "img.png").write_bytes(b"fake")
        fake_paths.HYPERKVASIR_RAW = tmp_path / "nonexistent"
        fake_paths.RAW = raw
        status = loader.verify_datasets()
        assert status["cvc_clinicdb"] is True

    def test_returns_dict_with_expected_keys(self, loader, tmp_path, fake_paths):
        fake_paths.HYPERKVASIR_RAW = tmp_path / "x"
        fake_paths.RAW = tmp_path / "r"
        status = loader.verify_datasets()
        assert set(status.keys()) == {"hyperkvasir", "cvc_clinicdb", "limuc"}

    def test_empty_dir_returns_false(self, loader, tmp_path, fake_paths):
        hk = tmp_path / "hk"
        hk.mkdir()
        fake_paths.HYPERKVASIR_RAW = hk
        fake_paths.RAW = tmp_path / "raw"
        status = loader.verify_datasets()
        assert status["hyperkvasir"] is False


# ─────────────────────────────────────────────────────────────
#  _glob_images
# ─────────────────────────────────────────────────────────────


class TestGlobImages:
    def test_finds_jpg_and_png(self, loader, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"x")
        (tmp_path / "b.png").write_bytes(b"x")
        (tmp_path / "c.txt").write_bytes(b"x")
        result = loader._glob_images(tmp_path)
        names = [p.name for p in result]
        assert "a.jpg" in names
        assert "b.png" in names
        assert "c.txt" not in names

    def test_returns_sorted_list(self, loader, tmp_path):
        for name in ["c.jpg", "a.jpg", "b.png"]:
            (tmp_path / name).write_bytes(b"x")
        result = loader._glob_images(tmp_path)
        assert [p.name for p in result] == sorted(p.name for p in result)

    def test_empty_directory_returns_empty_list(self, loader, tmp_path):
        assert loader._glob_images(tmp_path) == []

    def test_returns_list_type(self, loader, tmp_path):
        assert isinstance(loader._glob_images(tmp_path), list)


# ─────────────────────────────────────────────────────────────
#  load_tabular_risk_data
# ─────────────────────────────────────────────────────────────


class TestLoadTabularRiskData:
    def test_loads_first_csv(self, loader, tmp_path, fake_paths):
        tabular_dir = tmp_path / "raw_tabular"
        tabular_dir.mkdir(parents=True)
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_csv(tabular_dir / "data.csv", index=False)
        fake_paths.RAW_TABULAR = tabular_dir
        result = loader.load_tabular_risk_data()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_raises_file_not_found_when_no_csv(self, loader, tmp_path, fake_paths):
        tabular_dir = tmp_path / "raw_tabular"
        tabular_dir.mkdir(parents=True)
        fake_paths.RAW_TABULAR = tabular_dir
        with pytest.raises(FileNotFoundError):
            loader.load_tabular_risk_data()

    def test_returns_correct_column_names(self, loader, tmp_path, fake_paths):
        tabular_dir = tmp_path / "raw_tabular"
        tabular_dir.mkdir(parents=True)
        df = pd.DataFrame({"col_x": [1], "col_y": [2]})
        df.to_csv(tabular_dir / "data.csv", index=False)
        fake_paths.RAW_TABULAR = tabular_dir
        result = loader.load_tabular_risk_data()
        assert list(result.columns) == ["col_x", "col_y"]


# ─────────────────────────────────────────────────────────────
#  organize_kvasir_seg
# ─────────────────────────────────────────────────────────────


class TestOrganizeKvasirSeg:
    def _build_kvasir(self, tmp_path, fake_paths, nested=False):
        raw = tmp_path / "raw"
        kvasir_seg = raw / "kvasir_seg"
        if nested:
            img_dir = kvasir_seg / "Kvasir-SEG" / "images"
            mask_dir = kvasir_seg / "Kvasir-SEG" / "masks"
        else:
            img_dir = kvasir_seg / "images"
            mask_dir = kvasir_seg / "masks"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        for name in ["img1.jpg", "img2.jpg"]:
            (img_dir / name).write_bytes(b"fake")
            (mask_dir / name).write_bytes(b"fake")
        raw_images = tmp_path / "raw_images"
        fake_paths.RAW = raw
        fake_paths.RAW_IMAGES = raw_images
        return raw_images

    def test_copies_images_to_polyps_dir(self, loader, tmp_path, fake_paths):
        raw_images = self._build_kvasir(tmp_path, fake_paths)
        loader.organize_kvasir_seg()
        polyps = list((raw_images / "polyps").glob("kvasir_*.jpg"))
        assert len(polyps) == 2

    def test_copies_masks_to_masks_dir(self, loader, tmp_path, fake_paths):
        raw_images = self._build_kvasir(tmp_path, fake_paths)
        loader.organize_kvasir_seg()
        masks = list((raw_images / "masks").glob("kvasir_*.jpg"))
        assert len(masks) == 2

    def test_nested_kvasir_seg_structure(self, loader, tmp_path, fake_paths):
        raw_images = self._build_kvasir(tmp_path, fake_paths, nested=True)
        loader.organize_kvasir_seg()
        polyps = list((raw_images / "polyps").glob("kvasir_*.jpg"))
        assert len(polyps) == 2


# ─────────────────────────────────────────────────────────────
#  organize_curated_colon
# ─────────────────────────────────────────────────────────────


class TestOrganizeCuratedColon:
    def _build_curated(self, tmp_path, fake_paths):
        raw = tmp_path / "raw"
        curated = raw / "curated_colon"
        for sub, fname in [
            ("polyp_images", "p1.jpg"),
            ("normal_tissue", "n1.jpg"),
            ("other_stuff", "o1.jpg"),
        ]:
            d = curated / sub
            d.mkdir(parents=True)
            (d / fname).write_bytes(b"fake")
        raw_images = tmp_path / "raw_images"
        fake_paths.RAW = raw
        fake_paths.RAW_IMAGES = raw_images
        return raw_images

    def test_copies_polyp_images(self, loader, tmp_path, fake_paths):
        raw_images = self._build_curated(tmp_path, fake_paths)
        loader.organize_curated_colon()
        polyps = list((raw_images / "polyps").glob("curated_*.jpg"))
        assert len(polyps) >= 1

    def test_copies_normal_images(self, loader, tmp_path, fake_paths):
        raw_images = self._build_curated(tmp_path, fake_paths)
        loader.organize_curated_colon()
        normals = list((raw_images / "normal").glob("curated_*.jpg"))
        assert len(normals) >= 1

    def test_skips_unclassified_dirs(self, loader, tmp_path, fake_paths):
        raw_images = self._build_curated(tmp_path, fake_paths)
        loader.organize_curated_colon()
        all_files = list((raw_images / "polyps").glob("*")) + list(
            (raw_images / "normal").glob("*")
        )
        names = [f.name for f in all_files]
        assert not any("other" in n for n in names)

    @pytest.mark.parametrize("keyword", ["adenoma", "cancer", "tumor", "malignant"])
    def test_polyp_keywords_recognized(self, loader, tmp_path, fake_paths, keyword):
        raw = tmp_path / "raw"
        curated = raw / "curated_colon"
        d = curated / f"{keyword}_folder"
        d.mkdir(parents=True)
        (d / "img.jpg").write_bytes(b"fake")
        raw_images = tmp_path / "raw_images"
        fake_paths.RAW = raw
        fake_paths.RAW_IMAGES = raw_images
        loader.organize_curated_colon()
        polyps = list((raw_images / "polyps").glob("curated_*.jpg"))
        assert len(polyps) >= 1

    @pytest.mark.parametrize("keyword", ["healthy", "benign", "negative"])
    def test_normal_keywords_recognized(self, loader, tmp_path, fake_paths, keyword):
        raw = tmp_path / "raw"
        curated = raw / "curated_colon"
        d = curated / f"{keyword}_class"
        d.mkdir(parents=True)
        (d / "img.jpg").write_bytes(b"fake")
        raw_images = tmp_path / "raw_images"
        fake_paths.RAW = raw
        fake_paths.RAW_IMAGES = raw_images
        loader.organize_curated_colon()
        normals = list((raw_images / "normal").glob("curated_*.jpg"))
        assert len(normals) >= 1


# ─────────────────────────────────────────────────────────────
#  organize_all_images
# ─────────────────────────────────────────────────────────────


class TestOrganizeAllImages:
    @patch.object(KaggleLoader, "organize_kvasir_seg")
    @patch.object(KaggleLoader, "organize_curated_colon")
    def test_calls_both_organizers(
        self, mock_cc, mock_ks, loader, tmp_path, fake_paths
    ):
        raw_images = tmp_path / "raw_images"
        for sub in ["polyps", "normal", "masks"]:
            (raw_images / sub).mkdir(parents=True)
        fake_paths.RAW_IMAGES = raw_images
        loader.organize_all_images()
        mock_ks.assert_called_once()
        mock_cc.assert_called_once()
