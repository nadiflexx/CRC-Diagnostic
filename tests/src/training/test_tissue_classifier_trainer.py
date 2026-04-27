# tests/src/training/test_tissue_classifier.py
"""Tests for train_tissue_classifier.py"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

# ── Fixtures / Helpers ────────────────────────────────────────────────────────

FAKE_CLASS_MAPPING = {
    "num_classes": 3,
    "classes": {"0": "Normal", "1": "Polyp", "2": "Inflammation"},
    "sources": ["hyperkvasir"],
    "project": "CRC",
}


def _make_tissue_trainer(**kwargs):
    """Build TissueClassifierTrainer with all I/O mocked."""
    with (
        patch("src.training.train_tissue_classifier.TissueOnlyClassifier") as MockModel,
        patch("src.training.train_tissue_classifier.MLflowTracker"),
        patch("src.training.train_tissue_classifier.TISSUE_DIR", Path("/fake/tissue")),
        patch("src.training.train_tissue_classifier.paths") as mock_paths,
        patch.object(Path, "exists", return_value=True),
        patch(
            "builtins.open",
            MagicMock(
                return_value=MagicMock(
                    __enter__=MagicMock(
                        return_value=MagicMock(
                            read=MagicMock(return_value=json.dumps(FAKE_CLASS_MAPPING))
                        )
                    ),
                    __exit__=MagicMock(return_value=False),
                )
            ),
        ),
    ):
        mock_paths.COLON_PROCESSED = Path("/fake/processed")
        mock_paths.COLON_CLEAN = Path("/fake/clean")
        mock_paths.TISSUE_CLASSIFIER_CHECKPOINT = Path("/fake/tissue_best.pt")

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        MockModel.return_value = mock_model

        # Patch json.load directly since open mock is complex
        with patch("json.load", return_value=FAKE_CLASS_MAPPING):
            from src.training.train_tissue_classifier import TissueClassifierTrainer

            trainer = TissueClassifierTrainer(device="cpu", **kwargs)
        trainer.model = mock_model
    return trainer


# ── get_tissue_train_transforms ───────────────────────────────────────────────


class TestGetTissueTrainTransforms:
    def test_returns_compose(self):
        import albumentations as A

        from src.training.train_tissue_classifier import get_tissue_train_transforms

        t = get_tissue_train_transforms(384)
        assert isinstance(t, A.Compose)

    def test_produces_tensor_output(self):
        import numpy as np

        from src.training.train_tissue_classifier import get_tissue_train_transforms

        t = get_tissue_train_transforms(384)
        img = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        result = t(image=img)
        assert isinstance(result["image"], torch.Tensor)

    def test_output_shape(self):
        import numpy as np

        from src.training.train_tissue_classifier import get_tissue_train_transforms

        t = get_tissue_train_transforms(384)
        img = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        result = t(image=img)
        assert result["image"].shape == (3, 384, 384)

    def test_default_image_size(self):
        from src.training.train_tissue_classifier import get_tissue_train_transforms

        t = get_tissue_train_transforms()
        assert t is not None

    def test_output_dtype_float32(self):
        import numpy as np

        from src.training.train_tissue_classifier import get_tissue_train_transforms

        t = get_tissue_train_transforms(384)
        img = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        result = t(image=img)
        assert result["image"].dtype == torch.float32


# ── get_tissue_val_transforms ─────────────────────────────────────────────────


class TestGetTissueValTransforms:
    def test_returns_compose(self):
        import albumentations as A

        from src.training.train_tissue_classifier import get_tissue_val_transforms

        t = get_tissue_val_transforms(384)
        assert isinstance(t, A.Compose)

    def test_produces_tensor_output(self):
        import numpy as np

        from src.training.train_tissue_classifier import get_tissue_val_transforms

        t = get_tissue_val_transforms(384)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = t(image=img)
        assert isinstance(result["image"], torch.Tensor)

    def test_output_dtype_float32(self):
        import numpy as np

        from src.training.train_tissue_classifier import get_tissue_val_transforms

        t = get_tissue_val_transforms()
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = t(image=img)
        assert result["image"].dtype == torch.float32

    def test_deterministic_same_input(self):
        """Val transforms should be deterministic (no randomness)."""
        import numpy as np

        from src.training.train_tissue_classifier import get_tissue_val_transforms

        t = get_tissue_val_transforms()
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        r1 = t(image=img.copy())["image"]
        r2 = t(image=img.copy())["image"]
        torch.testing.assert_close(r1, r2)


# ── TissueTrainDataset ────────────────────────────────────────────────────────


class TestTissueTrainDataset:
    def _make_dataset(self, tmp_path, n_images=3, n_crops=2, labels=None):
        from src.training.train_tissue_classifier import (
            TissueTrainDataset,
            get_tissue_val_transforms,
        )

        if labels is None:
            labels = [0] * n_images

        # Create fake image dirs matching class_name structure
        tissue_dir = tmp_path / "tissue"
        class_name = "Normal"
        crop_dir = tissue_dir / class_name
        crop_dir.mkdir(parents=True)

        db_paths = []
        for i in range(n_images):
            # Create a real small JPEG crop
            img_arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            for c in range(n_crops):
                crop_path = crop_dir / f"img{i:03d}_crop{c}.jpg"
                cv2.imwrite(str(crop_path), img_arr)

            # Fake DB path with same structure: .../Normal/img000.jpg
            fake_original = tmp_path / "db" / class_name / f"img{i:03d}.jpg"
            fake_original.parent.mkdir(parents=True, exist_ok=True)
            fake_original.touch()
            db_paths.append(str(fake_original))

        transform = get_tissue_val_transforms(32)
        ds = TissueTrainDataset(
            db_paths=db_paths,
            labels=labels,
            tissue_dir=tissue_dir,
            transform=transform,
            image_size=32,
            n_crops=n_crops,
        )
        return ds

    def test_len_matches_images_with_crops(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_images=3, n_crops=2)
        assert len(ds) == 3

    def test_getitem_returns_tuple(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_images=2, n_crops=2)
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_getitem_tensor_type(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_images=1, n_crops=2)
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)

    def test_getitem_label_is_int(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_images=1, n_crops=2, labels=[1])
        _, label = ds[0]
        assert label == 1

    def test_getitem_image_shape(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_images=1, n_crops=2)
        img, _ = ds[0]
        # (C, H, W)
        assert img.ndim == 3
        assert img.shape[0] == 3

    def test_missing_crops_excluded(self, tmp_path):
        """Images with no crops should not appear in records."""
        from src.training.train_tissue_classifier import (
            TissueTrainDataset,
            get_tissue_val_transforms,
        )

        tissue_dir = tmp_path / "tissue"
        tissue_dir.mkdir()

        # DB paths but NO crop files created
        db_paths = [str(tmp_path / "Normal" / "img000.jpg")]
        labels = [0]
        transform = get_tissue_val_transforms(32)
        ds = TissueTrainDataset(
            db_paths, labels, tissue_dir, transform, image_size=32, n_crops=2
        )
        assert len(ds) == 0

    def test_getitem_handles_corrupt_image(self, tmp_path):
        """When cv2.imread returns None, a zero array should be used."""
        ds = self._make_dataset(tmp_path, n_images=1, n_crops=1)
        with patch("cv2.imread", return_value=None):
            img, label = ds[0]
        assert isinstance(img, torch.Tensor)

    def test_getitem_resizes_wrong_size_image(self, tmp_path):
        """Images not matching image_size should be resized."""
        ds = self._make_dataset(tmp_path, n_images=1, n_crops=1)
        # Return an image of different size
        wrong_size = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        with patch("cv2.imread", return_value=wrong_size):
            img, _ = ds[0]
        assert img.shape[1] == 32  # resized to image_size
        assert img.shape[2] == 32

    def test_no_transform_returns_tensor(self, tmp_path):
        """When transform=None the fallback path should still return a Tensor."""
        ds = self._make_dataset(tmp_path, n_images=1, n_crops=1)
        ds.transform = None
        img, _ = ds[0]
        assert isinstance(img, torch.Tensor)


# ── TissueCropDataset ─────────────────────────────────────────────────────────


class TestTissueCropDataset:
    def _make_dataset(self, tmp_path, n_images=2, n_crops=3):
        from src.training.train_tissue_classifier import (
            TissueCropDataset,
            get_tissue_val_transforms,
        )

        tissue_dir = tmp_path / "tissue"
        class_name = "Normal"
        crop_dir = tissue_dir / class_name
        crop_dir.mkdir(parents=True)

        db_paths = []
        for i in range(n_images):
            img_arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            for c in range(n_crops):
                crop_path = crop_dir / f"img{i:03d}_crop{c}.jpg"
                cv2.imwrite(str(crop_path), img_arr)
            fake_path = tmp_path / "db" / class_name / f"img{i:03d}.jpg"
            fake_path.parent.mkdir(parents=True, exist_ok=True)
            fake_path.touch()
            db_paths.append(str(fake_path))

        labels = [i % 3 for i in range(n_images)]
        transform = get_tissue_val_transforms(32)
        return TissueCropDataset(
            db_paths=db_paths,
            labels=labels,
            tissue_dir=tissue_dir,
            transform=transform,
            image_size=32,
            n_crops=n_crops,
        )

    def test_total_entries_is_images_times_crops(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_images=2, n_crops=3)
        assert len(ds) == 6  # 2 images × 3 crops

    def test_getitem_returns_three_elements(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_images=1, n_crops=2)
        item = ds[0]
        assert len(item) == 3

    def test_getitem_third_element_is_parent_idx(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_images=2, n_crops=2)
        _, _, parent_idx = ds[0]
        assert isinstance(parent_idx, int)

    def test_parent_idx_increases_with_images(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_images=2, n_crops=2)
        parents = [ds[i][2] for i in range(len(ds))]
        # First 2 entries belong to parent 0, next 2 to parent 1
        assert 0 in parents
        assert 1 in parents

    def test_getitem_image_is_tensor(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_images=1, n_crops=2)
        img, _, _ = ds[0]
        assert isinstance(img, torch.Tensor)

    def test_getitem_handles_corrupt_image(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_images=1, n_crops=1)
        with patch("cv2.imread", return_value=None):
            img, _, _ = ds[0]
        assert isinstance(img, torch.Tensor)

    def test_n_parents_reflects_images_with_crops(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_images=3, n_crops=2)
        assert ds.n_parents == 3

    def test_len_zero_when_no_crops(self, tmp_path):
        from src.training.train_tissue_classifier import (
            TissueCropDataset,
            get_tissue_val_transforms,
        )

        tissue_dir = tmp_path / "tissue"
        tissue_dir.mkdir()
        db_paths = [str(tmp_path / "Normal" / "img.jpg")]
        transform = get_tissue_val_transforms(32)
        ds = TissueCropDataset(db_paths, [0], tissue_dir, transform, 32, 2)
        assert len(ds) == 0


# ── validate_multicrop ────────────────────────────────────────────────────────


class TestValidateMulticrop:
    def _make_crop_dataset(self, n_parents=3, n_crops=2, n_classes=3):
        """Build a fake TissueCropDataset-like iterable."""
        entries = []
        for p in range(n_parents):
            for c in range(n_crops):
                entries.append(
                    {
                        "img": torch.rand(3, 32, 32),
                        "label": p % n_classes,
                        "parent": p,
                    }
                )
        return entries

    def _make_loader(self, n_parents=3, n_crops=2, n_classes=3, batch_size=4):
        """Returns a list of (images, labels, parent_indices) batches."""
        batches = []
        imgs, lbls, pars = [], [], []
        for p in range(n_parents):
            for _ in range(n_crops):
                imgs.append(torch.rand(3, 32, 32))
                lbls.append(p % n_classes)
                pars.append(p)

        # Split into batches
        for i in range(0, len(imgs), batch_size):
            batch_imgs = torch.stack(imgs[i : i + batch_size])
            batch_lbls = torch.tensor(lbls[i : i + batch_size])
            batch_pars = torch.tensor(pars[i : i + batch_size])
            batches.append((batch_imgs, batch_lbls, batch_pars))
        return batches

    def test_returns_required_keys(self, tmp_path):
        from src.training.train_tissue_classifier import (
            TissueCropDataset,
            get_tissue_val_transforms,
            validate_multicrop,
        )

        # Build a real (tiny) TissueCropDataset
        tissue_dir = tmp_path / "tissue"
        tissue_dir.mkdir()
        class_dir = tissue_dir / "Normal"
        class_dir.mkdir()

        n_images, n_crops = 3, 2
        db_paths = []
        for i in range(n_images):
            img_arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            for c in range(n_crops):
                cv2.imwrite(str(class_dir / f"img{i:03d}_crop{c}.jpg"), img_arr)
            fake = tmp_path / "db" / "Normal" / f"img{i:03d}.jpg"
            fake.parent.mkdir(parents=True, exist_ok=True)
            fake.touch()
            db_paths.append(str(fake))

        labels = [i % 3 for i in range(n_images)]
        transform = get_tissue_val_transforms(32)
        ds = TissueCropDataset(db_paths, labels, tissue_dir, transform, 32, n_crops)

        model = MagicMock()
        model.eval = MagicMock()
        # Asignación directa del comportamiento (sin 'with patch.object')
        model.side_effect = lambda imgs: torch.rand(imgs.size(0), 3)
        criterion = MagicMock(return_value=torch.tensor(0.3))

        # Llamada directa
        result = validate_multicrop(model, ds, criterion, "cpu", batch_size=8)

        expected_keys = {
            "loss",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "auc",
            "predictions",
            "labels",
            "probabilities",
            "n_images",
            "n_crops",
        }
        assert expected_keys == set(result.keys())

    def test_n_images_correct(self, tmp_path):
        from src.training.train_tissue_classifier import (
            TissueCropDataset,
            get_tissue_val_transforms,
            validate_multicrop,
        )

        tissue_dir = tmp_path / "tissue2"
        class_dir = tissue_dir / "Normal"
        class_dir.mkdir(parents=True)

        n_images, n_crops = 4, 2
        db_paths = []
        for i in range(n_images):
            img_arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            for c in range(n_crops):
                cv2.imwrite(str(class_dir / f"img{i:03d}_crop{c}.jpg"), img_arr)
            fake = tmp_path / "db2" / "Normal" / f"img{i:03d}.jpg"
            fake.parent.mkdir(parents=True, exist_ok=True)
            fake.touch()
            db_paths.append(str(fake))

        labels = [i % 3 for i in range(n_images)]
        transform = get_tissue_val_transforms(32)
        ds = TissueCropDataset(db_paths, labels, tissue_dir, transform, 32, n_crops)

        model = MagicMock()
        model.eval = MagicMock()
        # Asignación directa del comportamiento (sin 'with patch.object')
        model.side_effect = lambda imgs: torch.rand(imgs.size(0), 3)
        criterion = MagicMock(return_value=torch.tensor(0.3))

        # Llamada directa
        result = validate_multicrop(model, ds, criterion, "cpu", batch_size=8)

        assert result["n_images"] == n_images
        assert result["n_crops"] == n_images * n_crops

    def test_accuracy_in_zero_one(self, tmp_path):
        from src.training.train_tissue_classifier import (
            TissueCropDataset,
            get_tissue_val_transforms,
            validate_multicrop,
        )

        tissue_dir = tmp_path / "tissue3"
        class_dir = tissue_dir / "Normal"
        class_dir.mkdir(parents=True)

        n_images, n_crops = 3, 2
        db_paths = []
        for i in range(n_images):
            img_arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            for c in range(n_crops):
                cv2.imwrite(str(class_dir / f"img{i:03d}_crop{c}.jpg"), img_arr)
            fake = tmp_path / "db3" / "Normal" / f"img{i:03d}.jpg"
            fake.parent.mkdir(parents=True, exist_ok=True)
            fake.touch()
            db_paths.append(str(fake))

        labels = [0] * n_images  # single class → AUC=0
        transform = get_tissue_val_transforms(32)
        ds = TissueCropDataset(db_paths, labels, tissue_dir, transform, 32, n_crops)

        model = MagicMock()
        model.eval = MagicMock()
        # Asignación directa del comportamiento (sin 'with patch.object')
        model.side_effect = lambda imgs: torch.rand(imgs.size(0), 3)
        criterion = MagicMock(return_value=torch.tensor(0.3))

        # Llamada directa
        result = validate_multicrop(model, ds, criterion, "cpu", batch_size=8)

        assert 0.0 <= result["accuracy"] <= 1.0

    def test_auc_zero_when_single_class(self, tmp_path):
        from src.training.train_tissue_classifier import (
            TissueCropDataset,
            get_tissue_val_transforms,
            validate_multicrop,
        )

        tissue_dir = tmp_path / "tissue4"
        class_dir = tissue_dir / "Normal"
        class_dir.mkdir(parents=True)

        n_images, n_crops = 3, 1
        db_paths = []
        for i in range(n_images):
            img_arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            cv2.imwrite(str(class_dir / f"img{i:03d}_crop0.jpg"), img_arr)
            fake = tmp_path / "db4" / "Normal" / f"img{i:03d}.jpg"
            fake.parent.mkdir(parents=True, exist_ok=True)
            fake.touch()
            db_paths.append(str(fake))

        labels = [0] * n_images  # only class 0
        transform = get_tissue_val_transforms(32)
        ds = TissueCropDataset(db_paths, labels, tissue_dir, transform, 32, 1)

        model = MagicMock()
        model.eval = MagicMock()
        # Asignación directa del comportamiento (sin 'with patch.object')
        model.side_effect = lambda imgs: torch.rand(imgs.size(0), 3)
        criterion = MagicMock(return_value=torch.tensor(0.3))

        # Llamada directa
        result = validate_multicrop(model, ds, criterion, "cpu", batch_size=8)

        assert result["auc"] == 0.0


# ── TemperatureScalerB ────────────────────────────────────────────────────────


class TestTemperatureScalerB:
    def _cls(self):
        from src.training.train_tissue_classifier import TemperatureScalerB

        return TemperatureScalerB

    def test_default_temperature(self):
        ts = self._cls()()
        assert ts.temperature == 1.0

    def _make_real_crop_dataset(self, tmp_path, n_images=4, n_crops=2, suffix="tsB"):
        import cv2
        import numpy as np

        from src.training.train_tissue_classifier import (
            TissueCropDataset,
            get_tissue_val_transforms,
        )

        tissue_dir = tmp_path / f"tissue_{suffix}"
        class_dir = tissue_dir / "Normal"
        class_dir.mkdir(parents=True)
        db_paths = []
        for i in range(n_images):
            img_arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            for c in range(n_crops):
                cv2.imwrite(str(class_dir / f"img{i:03d}_crop{c}.jpg"), img_arr)
            fake = tmp_path / f"db_{suffix}" / "Normal" / f"img{i:03d}.jpg"
            fake.parent.mkdir(parents=True, exist_ok=True)
            fake.touch()
            db_paths.append(str(fake))
        labels = [i % 3 for i in range(n_images)]
        transform = get_tissue_val_transforms(32)
        return TissueCropDataset(db_paths, labels, tissue_dir, transform, 32, n_crops)

    def _make_model(self, n_classes=3):
        model = MagicMock()
        model.eval = MagicMock()
        # KEY FIX: side_effect returns real Tensor
        model.side_effect = lambda imgs: torch.rand(imgs.shape[0], n_classes)
        return model

    def test_fit_returns_float(self, tmp_path):
        ts = self._cls()()
        ds = self._make_real_crop_dataset(tmp_path, suffix="r1")
        model = self._make_model()
        result = ts.fit(model, ds, "cpu", batch_size=8)
        assert isinstance(result, float)
        assert 0.5 <= result < 5.0

    def test_fit_stores_temperature(self, tmp_path):
        ts = self._cls()()
        ds = self._make_real_crop_dataset(tmp_path, suffix="r2")
        model = self._make_model()
        temp = ts.fit(model, ds, "cpu", batch_size=8)
        assert ts.temperature == temp

    def test_fit_calls_model_eval(self, tmp_path):
        ts = self._cls()()
        ds = self._make_real_crop_dataset(tmp_path, n_images=1, n_crops=1, suffix="r3")
        model = self._make_model()
        ts.fit(model, ds, "cpu", batch_size=8)
        model.eval.assert_called_once()


# ── TissueClassifierTrainer ───────────────────────────────────────────────────


class TestTissueClassifierTrainerInit:
    def test_device_stored(self):
        trainer = _make_tissue_trainer()
        assert trainer.device == "cpu"

    def test_num_classes(self):
        trainer = _make_tissue_trainer()
        assert trainer.num_classes == 3

    def test_class_names_parsed(self):
        trainer = _make_tissue_trainer()
        assert trainer.class_names == {0: "Normal", 1: "Polyp", 2: "Inflammation"}

    def test_history_keys(self):
        trainer = _make_tissue_trainer()
        assert set(trainer.history.keys()) == {
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_f1",
            "val_auc",
            "val_recall",
            "val_precision",
        }

    def test_best_f1_starts_zero(self):
        trainer = _make_tissue_trainer()
        assert trainer.best_f1 == 0.0

    def test_max_patience(self):
        trainer = _make_tissue_trainer()
        assert trainer.max_patience == 12

    def test_temp_scaler_default_temperature(self):
        trainer = _make_tissue_trainer()
        assert trainer.temp_scaler.temperature == 1.0


class TestLoadClassMappingTissue:
    def test_raises_when_no_file_found(self):
        with (
            patch("src.training.train_tissue_classifier.TissueOnlyClassifier"),
            patch("src.training.train_tissue_classifier.MLflowTracker"),
            patch(
                "src.training.train_tissue_classifier.TISSUE_DIR", Path("/fake/tissue")
            ),
            patch("src.training.train_tissue_classifier.paths") as mock_paths,
            patch.object(Path, "exists", return_value=False),
        ):
            mock_paths.COLON_PROCESSED = Path("/fake/processed")
            mock_paths.COLON_CLEAN = Path("/fake/clean")
            from src.training.train_tissue_classifier import TissueClassifierTrainer

            with pytest.raises(FileNotFoundError):
                TissueClassifierTrainer(device="cpu")


class TestEnsureTissuePreprocessing:
    def test_skips_if_tissue_dir_non_empty(self):
        trainer = _make_tissue_trainer()
        with (
            patch.object(Path, "exists", return_value=True),  # <-- Patch the Class
            patch.object(
                Path, "iterdir", return_value=iter([Path("/fake/tissue/Normal")])
            ),
            patch.object(
                Path, "rglob", return_value=iter([Path("/fake/tissue/Normal/img.jpg")])
            ),
            patch(
                "src.training.train_tissue_classifier.process_dataset_tissue_only"
            ) as mock_proc,
        ):
            trainer._ensure_tissue_preprocessing()
        mock_proc.assert_not_called()

    def test_runs_preprocessing_if_empty(self):
        trainer = _make_tissue_trainer()
        with (
            patch.object(Path, "exists", return_value=False),  # <-- Patch the Class
            patch(
                "src.training.train_tissue_classifier.process_dataset_tissue_only"
            ) as mock_proc,
            patch("src.training.train_tissue_classifier.paths") as mock_paths,
        ):
            mock_paths.COLON_CLEAN = Path("/fake/clean")
            trainer._ensure_tissue_preprocessing()
        mock_proc.assert_called_once()


class TestTissueCheckpointOperations:
    def test_save_checkpoint_creates_file(self, tmp_path):
        trainer = _make_tissue_trainer()
        trainer.model.state_dict.return_value = {}
        ckpt_path = tmp_path / "tissue_best.pt"

        with patch("src.training.train_tissue_classifier.paths") as mock_paths:
            mock_paths.TISSUE_CLASSIFIER_CHECKPOINT = ckpt_path
            trainer._save_checkpoint(
                epoch=3, metrics={"loss": 0.2, "dice": 0.8, "accuracy": 0.9}
            )

        assert ckpt_path.exists()

    def test_save_checkpoint_excludes_array_fields(self, tmp_path):
        trainer = _make_tissue_trainer()
        trainer.model.state_dict.return_value = {}
        ckpt_path = tmp_path / "tissue_best.pt"
        metrics = {
            "loss": 0.1,
            "predictions": np.array([0, 1]),
            "labels": np.array([0, 1]),
            "probabilities": np.array([[0.9, 0.05, 0.05]]),
            "accuracy": 0.9,
        }
        with patch("src.training.train_tissue_classifier.paths") as mock_paths:
            mock_paths.TISSUE_CLASSIFIER_CHECKPOINT = ckpt_path
            trainer._save_checkpoint(epoch=0, metrics=metrics)

        ckpt = torch.load(ckpt_path, weights_only=False)
        for key in ("predictions", "labels", "probabilities"):
            assert key not in ckpt["metrics"]

    def test_save_checkpoint_stores_crop_strategy(self, tmp_path):
        trainer = _make_tissue_trainer()
        trainer.model.state_dict.return_value = {}
        ckpt_path = tmp_path / "tissue_best.pt"
        with patch("src.training.train_tissue_classifier.paths") as mock_paths:
            mock_paths.TISSUE_CLASSIFIER_CHECKPOINT = ckpt_path
            trainer._save_checkpoint(epoch=0, metrics={})
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert "crop_strategy" in ckpt

    def test_update_checkpoint_temperature(self, tmp_path):
        trainer = _make_tissue_trainer()
        ckpt_path = tmp_path / "tissue_best.pt"
        torch.save({"temperature": 1.0, "model_state_dict": {}}, ckpt_path)
        with patch("src.training.train_tissue_classifier.paths") as mock_paths:
            mock_paths.TISSUE_CLASSIFIER_CHECKPOINT = ckpt_path
            trainer._update_checkpoint_temperature(2.3)
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert ckpt["temperature"] == pytest.approx(2.3)

    def test_update_checkpoint_noop_if_no_file(self, tmp_path):
        trainer = _make_tissue_trainer()
        with patch("src.training.train_tissue_classifier.paths") as mock_paths:
            mock_paths.TISSUE_CLASSIFIER_CHECKPOINT = tmp_path / "missing.pt"
            trainer._update_checkpoint_temperature(1.5)  # Should not raise

    def test_load_best_warns_when_no_file(self):
        trainer = _make_tissue_trainer()
        with (
            patch("src.training.train_tissue_classifier.paths") as mock_paths,
            patch("src.training.train_tissue_classifier.logger") as mock_log,
        ):
            mock_paths.TISSUE_CLASSIFIER_CHECKPOINT = Path("/nonexistent.pt")
            trainer._load_best()
        mock_log.warning.assert_called_once()

    def test_load_best_restores_weights(self, tmp_path):
        trainer = _make_tissue_trainer()
        ckpt_path = tmp_path / "tissue_best.pt"
        torch.save(
            {
                "model_state_dict": {},
                "best_f1": 0.88,
                "temperature": 1.8,
            },
            ckpt_path,
        )
        with patch("src.training.train_tissue_classifier.paths") as mock_paths:
            mock_paths.TISSUE_CLASSIFIER_CHECKPOINT = ckpt_path
            trainer._load_best()
        trainer.model.load_state_dict.assert_called_once()
        assert trainer.temp_scaler.temperature == pytest.approx(1.8)

    def test_load_best_no_temperature_key(self, tmp_path):
        trainer = _make_tissue_trainer()
        ckpt_path = tmp_path / "tissue_best.pt"
        torch.save({"model_state_dict": {}, "best_f1": 0.7}, ckpt_path)
        with patch("src.training.train_tissue_classifier.paths") as mock_paths:
            mock_paths.TISSUE_CLASSIFIER_CHECKPOINT = ckpt_path
            trainer._load_best()
        assert trainer.temp_scaler.temperature == 1.0  # unchanged


class TestLoadDataFromDb:
    def _make_db_record(self, path, label):
        rec = MagicMock()
        rec.file_path = path
        rec.label = label
        return rec

    def test_raises_when_no_train_data(self):
        trainer = _make_tissue_trainer()
        mock_repo = MagicMock()
        mock_repo.get_by_split.return_value = []
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=MagicMock())
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch("src.training.train_tissue_classifier.get_db", return_value=mock_ctx),
            patch(
                "src.training.train_tissue_classifier.TrainingImageRepository",
                return_value=mock_repo,
            ),
            pytest.raises(ValueError, match="No training data"),
        ):
            trainer._load_data_from_db()

    def test_raises_when_crop_missing(self, tmp_path):
        trainer = _make_tissue_trainer()
        trainer.tissue_dir = tmp_path / "tissue_no_crops"
        trainer.tissue_dir.mkdir()

        img = tmp_path / "Normal" / "img000.jpg"
        img.parent.mkdir(parents=True)
        img.touch()
        rec = self._make_db_record(str(img), 0)

        mock_repo = MagicMock()
        mock_repo.get_by_split.return_value = [rec]
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=MagicMock())
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch("src.training.train_tissue_classifier.get_db", return_value=mock_ctx),
            patch(
                "src.training.train_tissue_classifier.TrainingImageRepository",
                return_value=mock_repo,
            ),
            pytest.raises(FileNotFoundError),
        ):
            trainer._load_data_from_db()

    def test_returns_three_splits(self, tmp_path):
        trainer = _make_tissue_trainer()
        trainer.tissue_dir = tmp_path / "tissue_db"
        tissue_class_dir = trainer.tissue_dir / "Normal"
        tissue_class_dir.mkdir(parents=True)

        img = tmp_path / "Normal" / "img000.jpg"
        img.parent.mkdir(parents=True)
        img.touch()

        # Create the expected crop file
        crop_file = tissue_class_dir / "img000_crop0.jpg"
        img_arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(crop_file), img_arr)

        rec = self._make_db_record(str(img), 0)
        mock_repo = MagicMock()
        mock_repo.get_by_split.return_value = [rec]
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=MagicMock())
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch("src.training.train_tissue_classifier.get_db", return_value=mock_ctx),
            patch(
                "src.training.train_tissue_classifier.TrainingImageRepository",
                return_value=mock_repo,
            ),
        ):
            result = trainer._load_data_from_db()

        assert set(result.keys()) == {"train", "val", "test"}


class TestEvaluateTest:
    def test_runs_without_error(self, tmp_path):
        trainer = _make_tissue_trainer()
        data = {
            "test": (
                [str(tmp_path / "Normal" / f"img{i:03d}.jpg") for i in range(4)],
                [0, 1, 2, 0],
            )
        }
        fake_m = {
            "accuracy": 0.75,
            "f1": 0.70,
            "precision": 0.72,
            "recall": 0.68,
            "auc": 0.80,
            "predictions": np.array([0, 1, 2, 0]),
            "labels": np.array([0, 1, 2, 1]),
            "probabilities": np.random.dirichlet([1, 1, 1], size=4),
            "n_images": 4,
            "n_crops": 8,
        }
        with (
            patch(
                "src.training.train_tissue_classifier.TissueCropDataset",
                return_value=MagicMock(),
            ),
            patch(
                "src.training.train_tissue_classifier.validate_multicrop",
                return_value=fake_m,
            ),
            patch("src.training.train_tissue_classifier.FocalLoss"),
            patch("src.training.train_tissue_classifier.get_tissue_val_transforms"),
            patch(
                "src.training.train_tissue_classifier.detect_source_from_stem",
                return_value="hyperkvasir",
            ),
        ):
            trainer._evaluate_test(data, image_size=32, n_crops=2)
