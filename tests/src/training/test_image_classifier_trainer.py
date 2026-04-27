# tests/src/training/test_image_classifier_trainer.py
"""Tests for ImageClassifierTrainer — FIXED model mock returns Tensor."""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
import torch

FAKE_CLASS_MAPPING = {
    "num_classes": 3,
    "classes": {"0": "Normal", "1": "Polyp", "2": "Inflammation"},
    "sources": ["hyperkvasir", "cvc"],
    "project": "CRC",
    "clinical_info": {
        "1": {"risk_level": "high", "action": "biopsy"},
    },
}


def _make_trainer(**kwargs):
    """Build ImageClassifierTrainer with all heavy I/O mocked."""
    with (
        patch("src.training.train_image_classifier.ColonCancerClassifier") as MockModel,
        patch("src.training.train_image_classifier.MLflowTracker"),
        patch("src.training.train_image_classifier.paths") as mock_paths,
        patch("builtins.open", mock_open(read_data=json.dumps(FAKE_CLASS_MAPPING))),
        patch.object(Path, "exists", return_value=True),
    ):
        mock_paths.COLON_PROCESSED = Path("/fake")
        mock_paths.COLON_CLEAN = Path("/fake/clean")
        mock_paths.DATA = Path("/fake/data")
        mock_paths.CLASSIFIER_CHECKPOINT = Path("/fake/best.pt")

        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        MockModel.return_value = mock_model_instance

        from src.training.train_image_classifier import ImageClassifierTrainer

        trainer = ImageClassifierTrainer(device="cpu", **kwargs)
        trainer.model = mock_model_instance

    return trainer


# ── __init__ ──────────────────────────────────────────────────────────────────


class TestImageClassifierTrainerInit:
    def test_device_cpu(self):
        trainer = _make_trainer()
        assert trainer.device == "cpu"

    def test_num_classes_from_mapping(self):
        trainer = _make_trainer()
        assert trainer.num_classes == 3

    def test_class_names_parsed(self):
        trainer = _make_trainer()
        assert trainer.class_names == {0: "Normal", 1: "Polyp", 2: "Inflammation"}

    def test_history_initialized_empty(self):
        trainer = _make_trainer()
        for key in (
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_f1",
            "val_auc",
            "val_recall",
            "val_precision",
        ):
            assert trainer.history[key] == []

    def test_best_metrics_start_at_zero(self):
        trainer = _make_trainer()
        assert trainer.best_f1 == 0.0
        assert trainer.best_acc == 0.0
        assert trainer.best_recall == 0.0
        assert trainer.patience_counter == 0

    def test_max_patience_default(self):
        trainer = _make_trainer()
        assert trainer.max_patience == 10

    def test_temp_scaler_initial_temperature(self):
        trainer = _make_trainer()
        assert trainer.temp_scaler.temperature == 1.0


# ── _load_class_mapping ───────────────────────────────────────────────────────


class TestLoadClassMapping:
    def test_raises_if_no_file_found(self):
        with (
            patch("src.training.train_image_classifier.MLflowTracker"),
            patch("src.training.train_image_classifier.ColonCancerClassifier"),
            patch("src.training.train_image_classifier.paths") as mock_paths,
            patch.object(Path, "exists", return_value=False),
        ):
            mock_paths.COLON_PROCESSED = Path("/fake")
            mock_paths.COLON_CLEAN = Path("/fake/clean")
            mock_paths.DATA = Path("/fake/data")
            from src.training.train_image_classifier import ImageClassifierTrainer

            with pytest.raises(FileNotFoundError, match="class_mapping.json"):
                ImageClassifierTrainer(device="cpu")

    def test_returns_dict_with_num_classes(self):
        trainer = _make_trainer()
        assert "num_classes" in trainer.class_mapping

    def test_returns_dict_with_classes_key(self):
        trainer = _make_trainer()
        assert "classes" in trainer.class_mapping


# ── _verify_no_leakage ────────────────────────────────────────────────────────


class TestImageClassifierVerifyNoLeakage:
    def test_clean_splits_pass(self):
        trainer = _make_trainer()
        data = {
            "train": (["/a/img1.jpg", "/a/img2.jpg"], [0, 1], [None, None]),
            "val": (["/a/img3.jpg"], [1], [None]),
            "test": (["/a/img4.jpg"], [0], [None]),
        }
        trainer._verify_no_leakage(data)  # Should not raise

    def test_train_val_overlap_raises(self):
        trainer = _make_trainer()
        data = {
            "train": (["/a/shared.jpg"], [0], [None]),
            "val": (["/a/shared.jpg"], [0], [None]),
            "test": (["/a/other.jpg"], [1], [None]),
        }
        with pytest.raises(ValueError, match="Data leakage"):
            trainer._verify_no_leakage(data)

    def test_train_test_overlap_raises(self):
        trainer = _make_trainer()
        data = {
            "train": (["/a/shared.jpg"], [0], [None]),
            "val": (["/a/other.jpg"], [1], [None]),
            "test": (["/a/shared.jpg"], [0], [None]),
        }
        with pytest.raises(ValueError, match="Data leakage"):
            trainer._verify_no_leakage(data)

    def test_leakage_count_in_message(self):
        trainer = _make_trainer()
        data = {
            "train": (["/a/x.jpg", "/a/y.jpg"], [0, 1], [None, None]),
            "val": (["/a/x.jpg", "/a/y.jpg"], [0, 1], [None, None]),
            "test": (["/a/z.jpg"], [0], [None]),
        }
        with pytest.raises(ValueError, match="2"):
            trainer._verify_no_leakage(data)


# ── _log_distribution ─────────────────────────────────────────────────────────


class TestLogDistribution:
    def test_runs_without_error(self):
        trainer = _make_trainer()
        data = {
            "train": (["/a/hk_img1.jpg"], [0], [None]),
            "val": (["/a/hk_img2.jpg"], [1], [None]),
            "test": (["/a/hk_img3.jpg"], [0], [None]),
        }
        with patch(
            "src.training.train_image_classifier.detect_source_from_stem",
            return_value="hyperkvasir",
        ):
            trainer._log_distribution(data)

    def test_unknown_class_label_shows_question_mark(self):
        trainer = _make_trainer()
        data = {
            "train": (["/a/img.jpg"], [99], [None]),  # 99 not in class_names
            "val": (["/a/img2.jpg"], [0], [None]),
            "test": (["/a/img3.jpg"], [1], [None]),
        }
        with patch(
            "src.training.train_image_classifier.detect_source_from_stem",
            return_value="unknown",
        ):
            trainer._log_distribution(data)  # Should not raise


# ── _validate ─────────────────────────────────────────────────────────────────


class TestImageClassifierValidate:
    def _make_loader(self, batch_size=4, num_classes=3, n_batches=2):
        images = torch.rand(batch_size, 3, 32, 32)
        labels = torch.tensor([i % num_classes for i in range(batch_size)])
        return [(images, labels)] * n_batches

    def _make_trainer_with_side_effect(self, num_classes=3):
        """
        Build trainer whose model.side_effect returns real Tensors.
        MagicMock's __call__ IS side_effect when set directly.
        """
        trainer = _make_trainer()
        trainer.model.eval = MagicMock()
        # Setting side_effect on the MagicMock instance makes
        # mock(x) call side_effect(x) and return the result.
        trainer.model.side_effect = lambda imgs: torch.rand(imgs.shape[0], num_classes)
        return trainer

    def test_returns_required_keys(self):
        trainer = self._make_trainer_with_side_effect()
        loader = self._make_loader()
        criterion = MagicMock(return_value=torch.tensor(0.5))
        metrics = trainer._validate(loader, criterion)
        expected = {
            "loss",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "auc",
            "predictions",
            "labels",
            "probabilities",
        }
        assert expected == set(metrics.keys())

    def test_predictions_shape(self):
        trainer = self._make_trainer_with_side_effect()
        loader = self._make_loader(batch_size=4, n_batches=1)
        criterion = MagicMock(return_value=torch.tensor(0.3))
        metrics = trainer._validate(loader, criterion)
        assert metrics["predictions"].shape == (4,)
        assert metrics["labels"].shape == (4,)

    def test_probabilities_sum_to_one(self):
        trainer = self._make_trainer_with_side_effect()
        loader = self._make_loader(batch_size=4, n_batches=1)
        criterion = MagicMock(return_value=torch.tensor(0.3))
        metrics = trainer._validate(loader, criterion)
        row_sums = metrics["probabilities"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_auc_zero_when_single_class(self):
        trainer = _make_trainer()
        trainer.model.eval = MagicMock()
        trainer.model.side_effect = lambda imgs: torch.rand(imgs.shape[0], 3)
        labels = torch.tensor([0, 0, 0, 0])
        loader = [(torch.rand(4, 3, 32, 32), labels)]
        criterion = MagicMock(return_value=torch.tensor(0.5))
        metrics = trainer._validate(loader, criterion)
        assert metrics["auc"] == 0.0

    def test_loss_is_float(self):
        trainer = self._make_trainer_with_side_effect()
        loader = self._make_loader(batch_size=4, n_batches=2)
        criterion = MagicMock(return_value=torch.tensor(0.4))
        metrics = trainer._validate(loader, criterion)
        assert isinstance(metrics["loss"], float)

    def test_accuracy_in_zero_one(self):
        trainer = self._make_trainer_with_side_effect()
        loader = self._make_loader()
        criterion = MagicMock(return_value=torch.tensor(0.5))
        metrics = trainer._validate(loader, criterion)
        assert 0.0 <= metrics["accuracy"] <= 1.0


# ── _save_checkpoint / load_best ──────────────────────────────────────────────


class TestCheckpointOperations:
    def test_save_creates_file(self, tmp_path):
        trainer = _make_trainer()
        trainer.model.state_dict.return_value = {}
        ckpt_path = tmp_path / "best.pt"

        with patch("src.training.train_image_classifier.paths") as mock_paths:
            mock_paths.CLASSIFIER_CHECKPOINT = ckpt_path
            trainer._save_checkpoint(epoch=0, metrics={"loss": 0.1, "f1": 0.8})

        assert ckpt_path.exists()

    def test_save_excludes_array_fields(self, tmp_path):
        trainer = _make_trainer()
        trainer.model.state_dict.return_value = {}
        ckpt_path = tmp_path / "best.pt"
        metrics = {
            "loss": 0.1,
            "predictions": np.array([0, 1]),
            "labels": np.array([0, 1]),
            "probabilities": np.array([[0.9, 0.05, 0.05]]),
        }
        with patch("src.training.train_image_classifier.paths") as mock_paths:
            mock_paths.CLASSIFIER_CHECKPOINT = ckpt_path
            trainer._save_checkpoint(epoch=0, metrics=metrics)

        ckpt = torch.load(ckpt_path, weights_only=False)
        for key in ("predictions", "labels", "probabilities"):
            assert key not in ckpt["metrics"]

    def test_save_stores_epoch(self, tmp_path):
        trainer = _make_trainer()
        trainer.model.state_dict.return_value = {}
        ckpt_path = tmp_path / "best.pt"
        with patch("src.training.train_image_classifier.paths") as mock_paths:
            mock_paths.CLASSIFIER_CHECKPOINT = ckpt_path
            trainer._save_checkpoint(epoch=7, metrics={"loss": 0.2})
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert ckpt["epoch"] == 7

    def test_save_stores_num_classes(self, tmp_path):
        trainer = _make_trainer()
        trainer.model.state_dict.return_value = {}
        ckpt_path = tmp_path / "best.pt"
        with patch("src.training.train_image_classifier.paths") as mock_paths:
            mock_paths.CLASSIFIER_CHECKPOINT = ckpt_path
            trainer._save_checkpoint(epoch=0, metrics={})
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert ckpt["num_classes"] == 3

    def test_load_best_warns_when_no_file(self):
        trainer = _make_trainer()
        with (
            patch("src.training.train_image_classifier.paths") as mock_paths,
            patch("src.training.train_image_classifier.logger") as mock_log,
        ):
            mock_paths.CLASSIFIER_CHECKPOINT = Path("/nonexistent/best.pt")
            trainer.load_best()
        mock_log.warning.assert_called_once()

    def test_load_best_restores_weights(self, tmp_path):
        trainer = _make_trainer()
        trainer.model.state_dict.return_value = {}
        ckpt_path = tmp_path / "best.pt"
        torch.save(
            {
                "epoch": 5,
                "model_state_dict": {},
                "best_f1": 0.92,
                "best_acc": 0.95,
                "temperature": 1.5,
            },
            ckpt_path,
        )
        with patch("src.training.train_image_classifier.paths") as mock_paths:
            mock_paths.CLASSIFIER_CHECKPOINT = ckpt_path
            trainer.load_best()
        trainer.model.load_state_dict.assert_called_once()
        assert trainer.temp_scaler.temperature == 1.5

    def test_load_best_without_temperature_key(self, tmp_path):
        trainer = _make_trainer()
        ckpt_path = tmp_path / "best.pt"
        torch.save(
            {
                "model_state_dict": {},
                "best_f1": 0.80,
            },
            ckpt_path,
        )
        with patch("src.training.train_image_classifier.paths") as mock_paths:
            mock_paths.CLASSIFIER_CHECKPOINT = ckpt_path
            trainer.load_best()
        # Temperature should stay at default 1.0
        assert trainer.temp_scaler.temperature == 1.0

    def test_save_checkpoint_with_temperature_updates(self, tmp_path):
        trainer = _make_trainer()
        ckpt_path = tmp_path / "best.pt"
        torch.save({"temperature": 1.0, "model_state_dict": {}}, ckpt_path)
        with patch("src.training.train_image_classifier.paths") as mock_paths:
            mock_paths.CLASSIFIER_CHECKPOINT = ckpt_path
            trainer._save_checkpoint_with_temperature(2.5)
        loaded = torch.load(ckpt_path, weights_only=False)
        assert loaded["temperature"] == pytest.approx(2.5)

    def test_save_checkpoint_with_temperature_noop_if_missing(self, tmp_path):
        trainer = _make_trainer()
        with patch("src.training.train_image_classifier.paths") as mock_paths:
            mock_paths.CLASSIFIER_CHECKPOINT = tmp_path / "missing.pt"
            trainer._save_checkpoint_with_temperature(2.0)  # Should not raise


# ── _evaluate_on_test ─────────────────────────────────────────────────────────


class TestEvaluateOnTest:
    def _make_data(self, n=8):
        paths_list = [f"/data/hk_img{i:03d}.jpg" for i in range(n)]
        labels = [i % 3 for i in range(n)]
        masks = [None] * n
        return {
            "train": ([], [], []),
            "val": ([], [], []),
            "test": (paths_list, labels, masks),
        }

    def test_returns_none_for_empty_test(self):
        trainer = _make_trainer()
        data = {"train": ([], [], []), "val": ([], [], []), "test": ([], [], [])}
        trainer.load_best = MagicMock()
        result = trainer._evaluate_on_test(data, img_size=64)
        assert result is None

    def test_returns_dict_for_non_empty_test(self):
        trainer = _make_trainer()
        data = self._make_data(n=6)
        trainer.load_best = MagicMock()

        fake_metrics = {
            "accuracy": 0.75,
            "f1": 0.72,
            "precision": 0.74,
            "recall": 0.70,
            "auc": 0.85,
            "predictions": np.array([0, 1, 2, 0, 1, 2]),
            "labels": np.array([0, 1, 2, 1, 1, 2]),
            "probabilities": np.random.dirichlet([1, 1, 1], size=6),
        }

        with (
            patch.object(trainer, "_validate", return_value=fake_metrics),
            patch(
                "src.training.train_image_classifier.ColonoscopyDataset",
                return_value=MagicMock(),
            ),
            patch("src.training.train_image_classifier.DataLoader", return_value=[]),
            patch(
                "src.training.train_image_classifier.get_val_transforms",
                return_value=MagicMock(),
            ),
            patch(
                "src.training.train_image_classifier.detect_source_from_stem",
                return_value="hyperkvasir",
            ),
        ):
            result = trainer._evaluate_on_test(data, img_size=64)

        assert result is not None
        assert "accuracy" in result

    def test_evaluate_on_test_public_method(self):
        trainer = _make_trainer()
        fake_data = {
            "train": ([], [], []),
            "val": ([], [], []),
            "test": ([], [], []),
        }
        with (
            patch.object(trainer, "_load_data_from_db", return_value=fake_data),
            patch.object(trainer, "_evaluate_on_test", return_value=None) as mock_eval,
        ):
            trainer.evaluate_on_test()
        mock_eval.assert_called_once()
