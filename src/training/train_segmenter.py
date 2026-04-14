"""
src/training/train_image_segmenter.py

Training pipeline for the polyp segmenter (U-Net).
Locates the exact position of a polyp within a colonoscopy image.

Only trains on images that have an associated ground-truth mask
(Kvasir-SEG subset and CVC-ClinicDB).

Metrics:
    - Dice coefficient (global overlap).
    - IoU / Jaccard (intersection over union).
    - Pixel accuracy.

Includes MLflow experiment tracking.
"""

from pathlib import Path
import platform

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.logger import log as logger
from src.config.paths import paths
from src.config.settings import model as model_cfg
from src.data.processing.image_preprocessor import (
    ColonoscopyDataset,
    get_segmentation_train_transforms,
    get_segmentation_val_transforms,
)
from src.database.connection import get_db
from src.database.repositories import TrainingImageRepository
from src.models.polyp_segmenter import ColonPolypSegmenter, DiceBCELoss
from src.training.tracking import MLflowTracker


class ImageSegmenterTrainer:
    """
    Training orchestrator for the U-Net polyp segmentation model.

    Manages data loading, dataset construction, training loop with early
    stopping, checkpoint saving, and MLflow tracking.
    """

    def __init__(self, device: str | None = None):
        """
        Initialise the segmentation trainer.

        Args:
            device (str | None): Target device. If ``None``, CUDA is used
                when available, otherwise CPU.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ColonPolypSegmenter().to(self.device)
        self.tracker = MLflowTracker(experiment_name="Colon_Cancer_Polyp_Segmentation")
        self.best_dice = 0.0
        self.best_iou = 0.0
        self.patience_counter = 0
        self.max_patience = 15
        self.history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
        }

    def _load_segmentation_data(self):
        """
        Load images that have an associated ground-truth mask from the database.

        Queries all splits (train, val, test) and retains only the records
        for which a mask file exists on disk. Supports both HyperKvasir and
        CVC-ClinicDB sources. The resulting collection is then split 80/20
        for training and validation.

        Returns:
            dict: Dictionary with keys ``"train"`` and ``"val"``. Each value
                is a tuple ``(paths, labels, mask_paths)`` where every element
                is a ``list[str]``.

        Raises:
            RuntimeError: If no images with a valid mask are found in the
                database after filtering.
        """
        with get_db() as db:
            repo = TrainingImageRepository(db)

            all_data = (
                repo.get_by_split("train")
                + repo.get_by_split("val")
                + repo.get_by_split("test")
            )

            seg_data = []
            for d in all_data:
                file_path = str(d.file_path)
                mask_path = str(d.mask_path) if d.mask_path else None
                label = int(d.label)

                if mask_path and Path(mask_path).exists():
                    seg_data.append(
                        {
                            "file_path": file_path,
                            "label": label,
                            "mask_path": mask_path,
                        }
                    )

        if not seg_data:
            raise RuntimeError(
                "No images with masks found.\n"
                "Run:\n"
                "  1. python -m src.data_processing.preprocess_masks\n"
                "  2. python -m src.data_ingestion.organize_multi_dataset"
            )

        from collections import Counter

        sources = Counter()
        for d in seg_data:
            stem = Path(d["file_path"]).stem
            if stem.startswith("hk_"):
                sources["hyperkvasir"] += 1
            elif stem.startswith("cvc_"):
                sources["cvc_clinicdb"] += 1
            else:
                sources["other"] += 1

        logger.info(f"  Images with masks: {len(seg_data)} ({dict(sources)})")

        train_data, val_data = train_test_split(
            seg_data, test_size=0.2, random_state=42
        )

        def extract(data):
            return (
                [d["file_path"] for d in data],
                [d["label"] for d in data],
                [d["mask_path"] for d in data],
            )

        logger.info(f"  Train: {len(train_data)}, Val: {len(val_data)}")

        return {"train": extract(train_data), "val": extract(val_data)}

    def train(
        self,
        epochs: int = 50,
        batch_size: int = 8,
        lr: float = 1e-4,
    ):
        """
        Run the full U-Net training pipeline with MLflow tracking.

        Steps:
            1. Start an MLflow run and log hyperparameters.
            2. Load segmentation data from the database.
            3. Build ``ColonoscopyDataset`` instances for train and val.
            4. Instantiate DiceBCE loss, AdamW optimiser, and cosine scheduler.
            5. Execute the training loop with early stopping based on Dice.
            6. Save the best checkpoint and log it to MLflow.

        Args:
            epochs (int): Maximum number of training epochs. Default is 50.
            batch_size (int): Number of samples per batch. Default is 8.
            lr (float): Initial learning rate for AdamW. Default is 1e-4.
        """
        image_size = model_cfg.IMAGE_SIZE

        with self.tracker.start_run("unet_efficientb4_segmentation"):
            self.tracker.log_params(
                {
                    "model": "U-Net + EfficientNet-B4",
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "image_size": image_size,
                    "device": self.device,
                    "loss": "DiceBCE (0.6/0.4)",
                    "optimizer": "AdamW",
                    "task": "polyp_segmentation",
                }
            )

            logger.info("\n═══ LOADING SEGMENTATION DATA ═══")
            data = self._load_segmentation_data()
            train_paths, train_labels, train_masks = data["train"]
            val_paths, val_labels, val_masks = data["val"]

            self.tracker.log_params(
                {
                    "train_samples": len(train_paths),
                    "val_samples": len(val_paths),
                }
            )

            train_dataset = ColonoscopyDataset(
                train_paths,
                train_labels,
                train_masks,
                transform=get_segmentation_train_transforms(image_size),
                image_size=image_size,
                mode="segmentation",
            )
            val_dataset = ColonoscopyDataset(
                val_paths,
                val_labels,
                val_masks,
                transform=get_segmentation_val_transforms(image_size),
                image_size=image_size,
                mode="segmentation",
            )

            is_win = platform.system() == "Windows"
            loader_kw = {
                "batch_size": batch_size,
                "num_workers": 0 if is_win else 4,
                "pin_memory": not is_win,
            }

            train_loader = DataLoader(
                train_dataset, shuffle=True, drop_last=True, **loader_kw
            )
            val_loader = DataLoader(val_dataset, shuffle=False, **loader_kw)

            logger.info(
                f"  Train: {len(train_dataset)} imgs, {len(train_loader)} batches"
            )
            logger.info(f"  Val:   {len(val_dataset)} imgs, {len(val_loader)} batches")

            criterion = DiceBCELoss(dice_weight=0.6, bce_weight=0.4)
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2
            )

            logger.info("\n═══ TRAINING: POLYP SEGMENTER ═══")
            logger.info(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
            logger.info(f"  Device: {self.device}")
            logger.info("  Loss: DiceBCE (dice=0.6, bce=0.4)")

            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0
                batch_count = 0

                train_pbar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{epochs} [Seg Train]",
                    leave=False,
                    dynamic_ncols=True,
                )

                for images, masks in train_pbar:
                    try:
                        images = images.to(self.device)
                        masks = masks.to(self.device)

                        optimizer.zero_grad()
                        outputs = self.model(images)
                        loss = criterion(outputs, masks)
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()

                        train_loss += loss.item()
                        batch_count += 1
                        train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                    except Exception as e:
                        logger.error(f"Batch error: {e}")
                        continue

                scheduler.step()
                avg_train_loss = train_loss / max(batch_count, 1)

                val_metrics = self._validate(val_loader, criterion)

                self.history["train_loss"].append(avg_train_loss)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_dice"].append(val_metrics["dice"])
                self.history["val_iou"].append(val_metrics["iou"])

                current_lr = optimizer.param_groups[0]["lr"]
                self.tracker.log_metrics(
                    {
                        "seg_train_loss": avg_train_loss,
                        "seg_val_loss": val_metrics["loss"],
                        "seg_val_dice": val_metrics["dice"],
                        "seg_val_iou": val_metrics["iou"],
                        "seg_val_pixel_acc": val_metrics["pixel_acc"],
                        "seg_learning_rate": current_lr,
                    },
                    step=epoch,
                )

                if val_metrics["dice"] > self.best_dice:
                    self.best_dice = val_metrics["dice"]
                    self.best_iou = val_metrics["iou"]
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, val_metrics)
                    logger.info(
                        f"  ✅ Best: Dice={self.best_dice:.4f} IoU={self.best_iou:.4f}"
                    )
                else:
                    self.patience_counter += 1

                logger.info(
                    f"E[{epoch + 1:3d}/{epochs}] "
                    f"TrL:{avg_train_loss:.4f} "
                    f"VaL:{val_metrics['loss']:.4f} "
                    f"Dice:{val_metrics['dice']:.4f} "
                    f"IoU:{val_metrics['iou']:.4f} "
                    f"PixAcc:{val_metrics['pixel_acc']:.4f} "
                    f"P:{self.patience_counter}/{self.max_patience}"
                )

                if self.patience_counter >= self.max_patience:
                    logger.info(f"⚠️  Early stopping at epoch {epoch + 1}")
                    self.tracker.log_param("seg_early_stopped_epoch", epoch + 1)
                    break

            self.tracker.log_metrics(
                {
                    "seg_best_dice": self.best_dice,
                    "seg_best_iou": self.best_iou,
                    "seg_total_epochs": epoch + 1,
                }
            )

            logger.info(
                f"\n✅ Segmenter trained. "
                f"Best Dice: {self.best_dice:.4f} | "
                f"IoU: {self.best_iou:.4f}"
            )

            self.tracker.log_model(self.model, "polyp_segmenter")
            logger.info("  📦 Model logged to MLflow")

    def _validate(self, loader, criterion):
        """
        Compute segmentation metrics on the validation set.

        For each sample the Dice coefficient, IoU (Jaccard index), and pixel
        accuracy are computed individually and then averaged across the full
        loader. Binary predictions are obtained by thresholding sigmoid
        outputs at 0.5.

        Formulas applied per sample (with smoothing factor ``smooth=1.0``):
            - Dice  = (2 * |A ∩ B| + smooth) / (|A| + |B| + smooth)
            - IoU   = (|A ∩ B| + smooth) / (|A ∪ B| + smooth)
            - Pixel accuracy = correct_pixels / total_pixels

        Args:
            loader (DataLoader): Validation data loader yielding
                ``(images, masks)`` batches.
            criterion (torch.nn.Module): Loss function used to accumulate
                the reported validation loss.

        Returns:
            dict: Dictionary with the following keys:
                - ``"loss"`` (float): Average batch loss.
                - ``"dice"`` (float): Mean Dice coefficient over all samples.
                - ``"iou"`` (float): Mean IoU over all samples.
                - ``"pixel_acc"`` (float): Mean pixel accuracy over all
                  samples.
        """
        self.model.eval()
        val_loss = 0.0
        dice_scores = []
        iou_scores = []
        pixel_accs = []
        smooth = 1.0

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                for i in range(preds.shape[0]):
                    pred_flat = preds[i].view(-1)
                    mask_flat = masks[i].view(-1)

                    intersection = (pred_flat * mask_flat).sum().item()
                    pred_sum = pred_flat.sum().item()
                    mask_sum = mask_flat.sum().item()
                    union = pred_sum + mask_sum - intersection

                    dice = (2.0 * intersection + smooth) / (
                        pred_sum + mask_sum + smooth
                    )
                    dice_scores.append(dice)

                    iou = (intersection + smooth) / (union + smooth)
                    iou_scores.append(iou)

                    correct = (pred_flat == mask_flat).sum().item()
                    total = mask_flat.numel()
                    pixel_accs.append(correct / total)

        return {
            "loss": val_loss / max(len(loader), 1),
            "dice": float(np.mean(dice_scores)),
            "iou": float(np.mean(iou_scores)),
            "pixel_acc": float(np.mean(pixel_accs)),
        }

    def _save_checkpoint(self, epoch, metrics):
        """
        Save the current model state as the best segmentation checkpoint.

        Args:
            epoch (int): Zero-based epoch index of the current checkpoint.
            metrics (dict): Validation metrics dictionary as returned by
                ``_validate``.
        """
        path = paths.SEGMENTER_CHECKPOINT
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "best_dice": self.best_dice,
                "best_iou": self.best_iou,
                "metrics": metrics,
                "history": self.history,
            },
            path,
        )

    def load_best(self):
        """
        Restore model weights from the best saved segmentation checkpoint.

        Logs a warning if no checkpoint file is found at
        ``paths.SEGMENTER_CHECKPOINT``.
        """
        path = paths.SEGMENTER_CHECKPOINT
        if not path.exists():
            logger.warning("No segmenter checkpoint found.")
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            f"✅ Segmenter loaded: "
            f"Dice={ckpt.get('best_dice', 0):.4f} "
            f"IoU={ckpt.get('best_iou', 0):.4f}"
        )


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("POLYP SEGMENTER (U-Net)")
    logger.info("  Exact polyp localisation in colonoscopy images")
    logger.info("=" * 70)

    trainer = ImageSegmenterTrainer(
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    trainer.train(epochs=100, batch_size=8, lr=1e-4)
