# tests/src/training/test_image_segmenter_trainer.py
"""Tests for ImageSegmenterTrainer — FIXED import paths."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def trainer():
    with (
        patch("src.training.train_segmenter.ColonPolypSegmenter") as MockSeg,
        patch("src.training.train_segmenter.MLflowTracker"),
    ):
        MockSeg.return_value = MagicMock()
        from src.training.train_segmenter import ImageSegmenterTrainer

        t = ImageSegmenterTrainer(device="cpu")
    return t


# ── __init__ ──────────────────────────────────────────────────────────────────


class TestSegmenterInit:
    def test_device_stored(self, trainer):
        assert trainer.device == "cpu"

    def test_best_dice_starts_at_zero(self, trainer):
        assert trainer.best_dice == 0.0

    def test_best_iou_starts_at_zero(self, trainer):
        assert trainer.best_iou == 0.0

    def test_history_keys(self, trainer):
        assert set(trainer.history.keys()) == {
            "train_loss",
            "val_loss",
            "val_dice",
            "val_iou",
        }

    def test_max_patience(self, trainer):
        assert trainer.max_patience == 15

    def test_patience_counter_starts_zero(self, trainer):
        assert trainer.patience_counter == 0

    def test_model_moved_to_device(self):
        with (
            patch("src.training.train_segmenter.ColonPolypSegmenter") as MockSeg,
            patch("src.training.train_segmenter.MLflowTracker"),
        ):
            mock_model = MagicMock()
            MockSeg.return_value = mock_model
            from src.training.train_segmenter import ImageSegmenterTrainer

            ImageSegmenterTrainer(device="cpu")
        mock_model.to.assert_called_once_with("cpu")


# ── _validate ─────────────────────────────────────────────────────────────────


class TestSegmenterValidate:
    def _make_loader(self, n_batches=2, batch_size=2, h=16, w=16):
        images = torch.rand(batch_size, 3, h, w)
        masks = torch.randint(0, 2, (batch_size, 1, h, w)).float()
        return [(images, masks)] * n_batches

    def _setup_trainer(self, tmp_path=None):
        """Trainer whose model.side_effect returns real Tensors."""
        with (
            patch("src.training.train_segmenter.ColonPolypSegmenter") as MockSeg,
            patch("src.training.train_segmenter.MLflowTracker"),
        ):
            MockSeg.return_value = MagicMock()
            from src.training.train_segmenter import ImageSegmenterTrainer

            t = ImageSegmenterTrainer(device="cpu")
        t.model.eval = MagicMock()
        # side_effect makes mock(x) return a real Tensor
        t.model.side_effect = lambda imgs: torch.rand(
            imgs.shape[0], 1, imgs.shape[2], imgs.shape[3]
        )
        return t

    def test_returns_required_keys(self):
        trainer = self._setup_trainer()
        loader = self._make_loader()
        criterion = MagicMock(return_value=torch.tensor(0.4))
        result = trainer._validate(loader, criterion)
        assert set(result.keys()) == {"loss", "dice", "iou", "pixel_acc"}

    def test_dice_in_zero_one(self):
        trainer = self._setup_trainer()
        loader = self._make_loader(n_batches=1)
        criterion = MagicMock(return_value=torch.tensor(0.3))
        result = trainer._validate(loader, criterion)
        assert 0.0 <= result["dice"] <= 1.0

    def test_iou_in_zero_one(self):
        trainer = self._setup_trainer()
        loader = self._make_loader(n_batches=1)
        criterion = MagicMock(return_value=torch.tensor(0.3))
        result = trainer._validate(loader, criterion)
        assert 0.0 <= result["iou"] <= 1.0

    def test_pixel_acc_in_zero_one(self):
        trainer = self._setup_trainer()
        loader = self._make_loader(n_batches=1)
        criterion = MagicMock(return_value=torch.tensor(0.3))
        result = trainer._validate(loader, criterion)
        assert 0.0 <= result["pixel_acc"] <= 1.0

    def test_perfect_predictions_high_dice(self):
        with (
            patch("src.training.train_segmenter.ColonPolypSegmenter") as MockSeg,
            patch("src.training.train_segmenter.MLflowTracker"),
        ):
            MockSeg.return_value = MagicMock()
            from src.training.train_segmenter import ImageSegmenterTrainer

            trainer = ImageSegmenterTrainer(device="cpu")
        trainer.model.eval = MagicMock()

        batch_size, h, w = 2, 8, 8
        masks = torch.ones(batch_size, 1, h, w)
        loader = [(torch.rand(batch_size, 3, h, w), masks)]
        criterion = MagicMock(return_value=torch.tensor(0.0))
        # Very large logits → sigmoid ≈ 1 → pred matches all-ones mask
        trainer.model.side_effect = lambda imgs: torch.full(
            (imgs.shape[0], 1, h, w), 10.0
        )
        result = trainer._validate(loader, criterion)
        assert result["dice"] > 0.95

    def test_loss_averaged_over_batches(self):
        trainer = self._setup_trainer()
        losses = [torch.tensor(0.2), torch.tensor(0.4)]
        idx = [0]

        def fake_criterion(*args, **kwargs):
            val = losses[idx[0]]
            idx[0] = min(idx[0] + 1, 1)
            return val

        loader = self._make_loader(n_batches=2)
        result = trainer._validate(loader, fake_criterion)
        assert result["loss"] == pytest.approx(0.3, abs=1e-5)

    def test_all_zero_masks_low_dice(self):
        with (
            patch("src.training.train_segmenter.ColonPolypSegmenter") as MockSeg,
            patch("src.training.train_segmenter.MLflowTracker"),
        ):
            MockSeg.return_value = MagicMock()
            from src.training.train_segmenter import ImageSegmenterTrainer

            trainer = ImageSegmenterTrainer(device="cpu")
        trainer.model.eval = MagicMock()

        batch_size, h, w = 2, 8, 8
        masks = torch.zeros(batch_size, 1, h, w)
        loader = [(torch.rand(batch_size, 3, h, w), masks)]
        criterion = MagicMock(return_value=torch.tensor(0.5))
        trainer.model.side_effect = lambda imgs: torch.full(
            (imgs.shape[0], 1, h, w), 10.0
        )
        result = trainer._validate(loader, criterion)
        assert result["dice"] < 0.5

    def test_empty_loader_returns_zero_loss(self):
        trainer = self._setup_trainer()
        criterion = MagicMock(return_value=torch.tensor(1.0))
        result = trainer._validate([], criterion)
        assert result["loss"] == pytest.approx(0.0, abs=1e-6)


# ── _save_checkpoint / load_best ──────────────────────────────────────────────


class TestSegmenterCheckpoint:
    def test_save_creates_file(self, tmp_path, trainer):
        trainer.model.state_dict.return_value = {}
        ckpt_path = tmp_path / "seg_best.pt"
        with patch("src.training.train_segmenter.paths") as mock_paths:
            mock_paths.SEGMENTER_CHECKPOINT = ckpt_path
            trainer._save_checkpoint(
                epoch=3,
                metrics={"dice": 0.8, "iou": 0.7, "loss": 0.2, "pixel_acc": 0.9},
            )
        assert ckpt_path.exists()

    def test_save_checkpoint_content(self, tmp_path, trainer):
        trainer.model.state_dict.return_value = {"w": torch.tensor([1.0])}
        trainer.best_dice = 0.88
        trainer.best_iou = 0.79
        ckpt_path = tmp_path / "seg_best.pt"
        with patch("src.training.train_segmenter.paths") as mock_paths:
            mock_paths.SEGMENTER_CHECKPOINT = ckpt_path
            trainer._save_checkpoint(epoch=2, metrics={"dice": 0.88})
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert ckpt["epoch"] == 2
        assert ckpt["best_dice"] == pytest.approx(0.88)

    def test_save_checkpoint_includes_history(self, tmp_path, trainer):
        trainer.model.state_dict.return_value = {}
        trainer.history["val_dice"] = [0.7, 0.8]
        ckpt_path = tmp_path / "seg_best.pt"
        with patch("src.training.train_segmenter.paths") as mock_paths:
            mock_paths.SEGMENTER_CHECKPOINT = ckpt_path
            trainer._save_checkpoint(epoch=1, metrics={"dice": 0.8})
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert "history" in ckpt

    def test_load_best_warns_when_no_file(self, trainer):
        with (
            patch("src.training.train_segmenter.paths") as mock_paths,
            patch("src.training.train_segmenter.logger") as mock_log,
        ):
            mock_paths.SEGMENTER_CHECKPOINT = Path("/no/such/file.pt")
            trainer.load_best()
        mock_log.warning.assert_called_once()

    def test_load_best_restores_state_dict(self, tmp_path, trainer):
        ckpt_path = tmp_path / "seg_best.pt"
        torch.save(
            {
                "model_state_dict": {"w": torch.tensor([5.0])},
                "best_dice": 0.91,
                "best_iou": 0.83,
            },
            ckpt_path,
        )
        with patch("src.training.train_segmenter.paths") as mock_paths:
            mock_paths.SEGMENTER_CHECKPOINT = ckpt_path
            trainer.load_best()
        trainer.model.load_state_dict.assert_called_once()

    def test_load_best_no_error_with_valid_file(self, tmp_path, trainer):
        ckpt_path = tmp_path / "seg_best.pt"
        torch.save(
            {
                "model_state_dict": {},
                "best_dice": 0.85,
                "best_iou": 0.78,
            },
            ckpt_path,
        )
        with patch("src.training.train_segmenter.paths") as mock_paths:
            mock_paths.SEGMENTER_CHECKPOINT = ckpt_path
            trainer.load_best()  # Should not raise


# ── _load_segmentation_data ───────────────────────────────────────────────────


class TestLoadSegmentationData:
    def _make_db_record(self, file_path, mask_path, label=1):
        rec = MagicMock()
        rec.file_path = file_path
        rec.mask_path = mask_path
        rec.label = label
        return rec

    def _patch_db(self, records):
        mock_repo = MagicMock()
        mock_repo.get_by_split.return_value = records
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=MagicMock())
        mock_ctx.__exit__ = MagicMock(return_value=False)
        return mock_repo, mock_ctx

    def test_raises_when_no_masks(self, trainer, tmp_path):
        rec = self._make_db_record(str(tmp_path / "img.jpg"), None)
        mock_repo, mock_ctx = self._patch_db([rec])
        with (
            patch("src.training.train_segmenter.get_db", return_value=mock_ctx),
            patch(
                "src.training.train_segmenter.TrainingImageRepository",
                return_value=mock_repo,
            ),
            pytest.raises(RuntimeError, match="No images with masks"),
        ):
            trainer._load_segmentation_data()

    def test_returns_train_val_keys(self, trainer, tmp_path):
        img = tmp_path / "hk_img001.jpg"
        mask = tmp_path / "hk_img001_mask.png"
        img.touch()
        mask.touch()
        rec = self._make_db_record(str(img), str(mask))
        mock_repo, mock_ctx = self._patch_db([rec])
        with (
            patch("src.training.train_segmenter.get_db", return_value=mock_ctx),
            patch(
                "src.training.train_segmenter.TrainingImageRepository",
                return_value=mock_repo,
            ),
        ):
            result = trainer._load_segmentation_data()
        assert "train" in result
        assert "val" in result

    def test_filters_records_without_mask_path(self, trainer, tmp_path):
        """Records with mask_path=None should be excluded."""
        img = tmp_path / "img.jpg"
        img.touch()
        rec_no_mask = self._make_db_record(str(img), None)
        mock_repo, mock_ctx = self._patch_db([rec_no_mask])
        with (
            patch("src.training.train_segmenter.get_db", return_value=mock_ctx),
            patch(
                "src.training.train_segmenter.TrainingImageRepository",
                return_value=mock_repo,
            ),
            pytest.raises(RuntimeError),
        ):
            trainer._load_segmentation_data()

    def test_filters_missing_mask_files(self, trainer, tmp_path):
        """mask_path set but file missing on disk → excluded → RuntimeError."""
        img = tmp_path / "img.jpg"
        img.touch()
        rec = self._make_db_record(str(img), "/nonexistent/mask.png")
        mock_repo, mock_ctx = self._patch_db([rec])
        with (
            patch("src.training.train_segmenter.get_db", return_value=mock_ctx),
            patch(
                "src.training.train_segmenter.TrainingImageRepository",
                return_value=mock_repo,
            ),
            pytest.raises(RuntimeError),
        ):
            trainer._load_segmentation_data()

    def test_cvc_prefix_counted_correctly(self, trainer, tmp_path):
        img = tmp_path / "cvc_img001.jpg"
        mask = tmp_path / "cvc_img001_mask.png"
        img.touch()
        mask.touch()
        rec = self._make_db_record(str(img), str(mask))
        mock_repo, mock_ctx = self._patch_db([rec])
        with (
            patch("src.training.train_segmenter.get_db", return_value=mock_ctx),
            patch(
                "src.training.train_segmenter.TrainingImageRepository",
                return_value=mock_repo,
            ),
        ):
            result = trainer._load_segmentation_data()
        assert "train" in result

    def test_extract_returns_three_lists(self, trainer, tmp_path):
        img = tmp_path / "hk_img002.jpg"
        mask = tmp_path / "hk_img002_mask.png"
        img.touch()
        mask.touch()
        rec = self._make_db_record(str(img), str(mask), label=0)
        mock_repo, mock_ctx = self._patch_db([rec])
        with (
            patch("src.training.train_segmenter.get_db", return_value=mock_ctx),
            patch(
                "src.training.train_segmenter.TrainingImageRepository",
                return_value=mock_repo,
            ),
        ):
            result = trainer._load_segmentation_data()
        # Each split value is a tuple of 3 lists
        for split in ("train", "val"):
            assert len(result[split]) == 3
