"""
src/training/train_image_classifier.py

Training pipeline for the Colon Cancer Image Classifier with MLflow.
3 classes (colonoscopy only): Normal | Polyp | Inflammation.

Features:
    - Mixup augmentation (breaks spatial shortcuts).
    - Post-training temperature scaling (calibrates prediction confidence).
    - Multi-source support (HyperKvasir + CVC-ClinicDB + LIMUC).
"""

from collections import Counter
import json
from pathlib import Path
import platform

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.constants import detect_source_from_stem
from src.config.logger import log as logger
from src.config.paths import paths
from src.data.processing.image_preprocessor import (
    ColonoscopyDataset,
    get_train_transforms,
    get_val_transforms,
)
from src.database.connection import get_db
from src.database.repositories import TrainingImageRepository
from src.models.image_classifier import ColonCancerClassifier, FocalLoss
from src.training.tracking import MLflowTracker

# ─── Mixup ───────────────────────────────────────────────


def mixup_data(x, y, alpha=0.2):
    """
    Apply Mixup augmentation to a batch of images and labels.

    Samples a mixing coefficient ``lam`` from a Beta distribution and
    produces a convex combination of each image with a randomly permuted
    partner. Returns both the mixed images and the two sets of labels so
    that the loss can be computed as a weighted sum.

    Reference: Zhang et al. (2018) "mixup: Beyond Empirical Risk
    Minimization".

    Args:
        x (torch.Tensor): Input image batch of shape (N, C, H, W).
        y (torch.Tensor): Integer label batch of shape (N,).
        alpha (float): Beta distribution parameter controlling the
            strength of the mixing. When ``alpha > 0`` a random ``lam``
            is drawn; otherwise ``lam = 1`` (no mixing). Default is 0.2.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
            - ``mixed_x``: Linearly interpolated image batch.
            - ``y_a``: Original labels.
            - ``y_b``: Labels of the randomly paired partners.
            - ``lam``: Mixing coefficient in [0, 1].
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute the Mixup-weighted loss from two label targets.

    Args:
        criterion (callable): Loss function that accepts ``(pred, target)``.
        pred (torch.Tensor): Model output logits of shape (N, C).
        y_a (torch.Tensor): Primary labels (original batch order).
        y_b (torch.Tensor): Secondary labels (randomly permuted partners).
        lam (float): Mixing coefficient used to weight each loss term.

    Returns:
        torch.Tensor: Scalar loss value computed as
            ``lam * L(pred, y_a) + (1 - lam) * L(pred, y_b)``.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─── Temperature Scaling ─────────────────────────────────


class TemperatureScaler:
    """
    Post-hoc confidence calibrator using temperature scaling.

    Divides logits by a learned scalar temperature ``T`` before the
    softmax. Values of ``T > 1`` soften (widen) the distribution;
    values of ``T < 1`` sharpen it.

    Reference: Guo et al. (2017) "On Calibration of Modern Neural
    Networks".
    """

    def __init__(self):
        """Initialise with temperature set to 1.0 (identity)."""
        self.temperature = 1.0

    def fit(self, model, val_loader, device):
        """
        Find the optimal temperature by minimising NLL on the validation set.

        Collects all logits from the validation loader in a single forward
        pass, then performs a grid search over temperatures in [0.5, 5.0)
        with step 0.1, selecting the value that yields the lowest cross-
        entropy loss.

        Args:
            model (torch.nn.Module): Trained model used to collect logits.
                Set to eval mode internally.
            val_loader (DataLoader): Validation data loader yielding
                ``(images, labels)`` batches.
            device (str): Target device identifier.

        Returns:
            float: Optimal temperature value. Also stored in
                ``self.temperature``.
        """
        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        best_nll = float("inf")
        best_t = 1.0

        for t in np.arange(0.5, 5.0, 0.1):
            scaled_logits = logits / t
            nll = F.cross_entropy(scaled_logits, labels).item()
            if nll < best_nll:
                best_nll = nll
                best_t = t

        self.temperature = best_t
        logger.info(f"  🌡️  Optimal temperature: {self.temperature:.2f}")

        return self.temperature

    def scale(self, logits):
        """
        Apply temperature scaling to a batch of logits.

        Args:
            logits (torch.Tensor): Raw model output of shape (N, C).

        Returns:
            torch.Tensor: Temperature-scaled logits of the same shape.
        """
        return logits / self.temperature


# ─── Trainer ─────────────────────────────────────────────


class ImageClassifierTrainer:
    """
    Training orchestrator for the multi-source colon cancer image classifier.

    Handles the complete pipeline: class mapping loading, data retrieval from
    the database, leakage verification, dataset construction, training with
    optional Mixup, temperature calibration, test evaluation, and checkpoint
    management.
    """

    def __init__(
        self,
        model_name: str = "tf_efficientnetv2_s.in21k",
        device: str | None = None,
    ):
        """
        Initialise the trainer and instantiate the classifier model.

        Args:
            model_name (str): Timm model identifier for the backbone.
                Default is ``"tf_efficientnetv2_s.in21k"``.
            device (str | None): Target device. If ``None``, CUDA is used
                when available, otherwise CPU.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tracker = MLflowTracker(experiment_name="Colon_Cancer_Classification")

        self.class_mapping = self._load_class_mapping()
        self.num_classes = self.class_mapping["num_classes"]
        self.class_names = {int(k): v for k, v in self.class_mapping["classes"].items()}

        logger.info(f"  Project: {self.class_mapping.get('project', 'N/A')}")
        logger.info(
            f"  Classes ({self.num_classes}): {list(self.class_names.values())}"
        )
        logger.info(f"  Sources: {self.class_mapping.get('sources', ['hyperkvasir'])}")

        self.model = ColonCancerClassifier(
            model_name=model_name,
            pretrained=True,
            dropout=0.4,
            num_classes=self.num_classes,
        ).to(self.device)

        self.temp_scaler = TemperatureScaler()

        self.best_f1 = 0.0
        self.best_acc = 0.0
        self.best_recall = 0.0
        self.patience_counter = 0
        self.max_patience = 10
        self.history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_auc": [],
            "val_recall": [],
            "val_precision": [],
        }

    def _load_class_mapping(self):
        """
        Load the class mapping JSON from the first available candidate path.

        Search order:
            1. ``paths.COLON_PROCESSED / class_mapping.json``
            2. ``paths.COLON_CLEAN / class_mapping.json``
            3. ``paths.DATA / hyperkvasir_processed / class_mapping.json``
            4. ``paths.DATA / hyperkvasir_clean / class_mapping.json``

        Returns:
            dict: Parsed class mapping containing at least
                ``"num_classes"``, ``"classes"``, and optionally
                ``"sources"`` and ``"clinical_info"`` keys.

        Raises:
            FileNotFoundError: If no ``class_mapping.json`` is found in
                any of the candidate paths.
        """
        for p in [
            paths.COLON_PROCESSED / "class_mapping.json",
            paths.COLON_CLEAN / "class_mapping.json",
            paths.DATA / "hyperkvasir_processed" / "class_mapping.json",
            paths.DATA / "hyperkvasir_clean" / "class_mapping.json",
        ]:
            if p.exists():
                with open(p) as f:
                    mapping = json.load(f)
                logger.info(f"  📄 class_mapping loaded from: {p}")
                return mapping

        raise FileNotFoundError(
            "class_mapping.json not found. "
            "Run: organize_multi_dataset.py or organize_hyperkvasir.py"
        )

    def _load_data_from_db(self):
        """
        Load train, validation, and test splits from the database.

        Retrieves file paths, integer labels, and optional mask paths for
        each split. Runs leakage and distribution checks before returning.

        Returns:
            dict: Dictionary with keys ``"train"``, ``"val"``, and
                ``"test"``. Each value is a tuple
                ``(paths: list[str], labels: list[int],
                masks: list[str | None])``.

        Raises:
            ValueError: If no training images are found in the database.
        """

        def extract(data):
            return (
                [str(d.file_path) for d in data],
                [int(d.label) for d in data],
                [str(d.mask_path) if d.mask_path else None for d in data],
            )

        with get_db() as db:
            repo = TrainingImageRepository(db)
            train = repo.get_by_split("train")
            val = repo.get_by_split("val")
            test = repo.get_by_split("test")

            if not train:
                raise ValueError("No training data found.")

            result = {
                "train": extract(train),
                "val": extract(val),
                "test": extract(test),
            }

        self._verify_no_leakage(result)
        self._log_distribution(result)
        return result

    def _verify_no_leakage(self, data):
        """
        Assert that no image filename is shared across train, val, and test.

        Args:
            data (dict): Data dictionary as returned by ``_load_data_from_db``.

        Raises:
            ValueError: If any filename appears in more than one split,
                indicating potential data leakage.
        """
        names = {}
        for split in ["train", "val", "test"]:
            names[split] = {Path(p).name for p in data[split][0]}

        overlaps = (
            (names["train"] & names["val"])
            | (names["train"] & names["test"])
            | (names["val"] & names["test"])
        )
        if overlaps:
            raise ValueError(
                f"Data leakage: {len(overlaps)} images shared between splits"
            )
        logger.info("  ✅ No data leakage between splits")

    def _log_distribution(self, data):
        """
        Log per-class and per-source sample counts for each split.

        Args:
            data (dict): Data dictionary as returned by ``_load_data_from_db``.
        """
        for split in ["train", "val", "test"]:
            paths_list, labels, _ = data[split]
            label_counts = Counter(labels)
            detail = ", ".join(
                f"{self.class_names.get(k, '?')}={v}"
                for k, v in sorted(label_counts.items())
            )

            source_counts = Counter()
            for p in paths_list:
                source_counts[detect_source_from_stem(Path(p).stem)] += 1

            source_detail = ", ".join(
                f"{k}={v}" for k, v in sorted(source_counts.items())
            )

            logger.info(f"  {split}: {len(labels)} ({detail})")
            logger.info(f"         sources: {source_detail}")

    def train(
        self,
        epochs: int = 25,
        batch_size: int = 16,
        lr: float = 1e-4,
        image_size: int = 384,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
    ):
        """
        Run the full image classifier training pipeline.

        Steps:
            1. Start an MLflow run and log all hyperparameters.
            2. Load data splits and compute balanced class weights.
            3. Build ``ColonoscopyDataset`` instances and data loaders.
            4. Instantiate FocalLoss, AdamW optimiser, and cosine scheduler.
            5. Execute the training loop; optionally apply Mixup with
               probability 0.5 per batch when ``use_mixup=True``.
            6. Apply temperature scaling on the validation set.
            7. Evaluate on the test set and log all metrics to MLflow.
            8. Save the model artifact to MLflow.

        Args:
            epochs (int): Maximum number of training epochs. Default is 25.
            batch_size (int): Number of samples per training batch.
                Default is 16.
            lr (float): Initial learning rate for AdamW. Default is 1e-4.
            image_size (int): Input spatial resolution in pixels.
                Default is 384.
            use_mixup (bool): Whether to apply Mixup augmentation during
                training. Default is ``False``.
            mixup_alpha (float): Beta distribution parameter for Mixup.
                Ignored when ``use_mixup=False``. Default is 0.2.
        """
        data_sources = self.class_mapping.get("sources", ["hyperkvasir"])
        sources_str = (
            "+".join(data_sources)
            if isinstance(data_sources, list)
            else str(data_sources)
        )

        with self.tracker.start_run(f"{self.model_name}_multisource"):
            self.tracker.log_params(
                {
                    "model_name": self.model_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "image_size": image_size,
                    "device": self.device,
                    "num_classes": self.num_classes,
                    "classes": str(list(self.class_names.values())),
                    "preprocessing": "green_suppression+component_filtering+augmentation",
                    "data_sources": sources_str,
                    "mixup": use_mixup,
                    "mixup_alpha": mixup_alpha if use_mixup else 0,
                    "loss": "focal_loss_smoothed",
                    "optimizer": "AdamW",
                    "scheduler": "CosineAnnealingWarmRestarts",
                    "temperature_scaling": True,
                    "note": "3-class multi-source, frame artifacts neutralized",
                }
            )

            logger.info("\n═══ LOADING DATA ═══")
            data = self._load_data_from_db()
            train_paths, train_labels, train_masks = data["train"]
            val_paths, val_labels, val_masks = data["val"]

            classes = np.unique(train_labels)
            weights = compute_class_weight(
                "balanced", classes=classes, y=np.array(train_labels)
            )
            weight_dict = dict(
                zip(classes.tolist(), weights.round(3).tolist(), strict=True)
            )
            logger.info(f"  Class weights: {weight_dict}")

            train_dataset = ColonoscopyDataset(
                train_paths,
                train_labels,
                train_masks,
                transform=get_train_transforms(image_size),
                image_size=image_size,
                mode="classification",
            )
            val_dataset = ColonoscopyDataset(
                val_paths,
                val_labels,
                val_masks,
                transform=get_val_transforms(image_size),
                image_size=image_size,
                mode="classification",
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

            criterion = FocalLoss(
                alpha=weights.tolist(),
                gamma=2.0,
                num_classes=self.num_classes,
                label_smoothing=0.1,
            )
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=1e-3
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2
            )

            logger.info(f"\n═══ TRAINING ({self.num_classes} CLASSES) ═══")
            logger.info(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
            logger.info(f"  Mixup: {'✅' if use_mixup else '❌'} (alpha={mixup_alpha})")
            logger.info(f"  Classes: {list(self.class_names.values())}")
            logger.info(f"  Sources: {sources_str}")

            for epoch in range(epochs):
                self.model.train()
                train_loss = 0
                batch_count = 0

                train_pbar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{epochs} [Train]",
                    leave=False,
                    dynamic_ncols=True,
                )

                for images, labels in train_pbar:
                    try:
                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        optimizer.zero_grad()

                        if use_mixup and np.random.random() > 0.5:
                            mixed_images, y_a, y_b, lam = mixup_data(
                                images, labels, alpha=mixup_alpha
                            )
                            outputs = self.model(mixed_images)
                            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                        else:
                            outputs = self.model(images)
                            loss = criterion(outputs, labels)

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
                self.history["val_accuracy"].append(val_metrics["accuracy"])
                self.history["val_f1"].append(val_metrics["f1"])
                self.history["val_auc"].append(val_metrics["auc"])
                self.history["val_recall"].append(val_metrics["recall"])
                self.history["val_precision"].append(val_metrics["precision"])

                current_lr = optimizer.param_groups[0]["lr"]
                self.tracker.log_metrics(
                    {
                        "train_loss": avg_train_loss,
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_f1_macro": val_metrics["f1"],
                        "val_precision_macro": val_metrics["precision"],
                        "val_recall_macro": val_metrics["recall"],
                        "val_auc_macro": val_metrics["auc"],
                        "learning_rate": current_lr,
                    },
                    step=epoch,
                )

                if epoch == 0:
                    baseline = 1.0 / self.num_classes
                    logger.info("\n🔍 EPOCH 1 CHECK:")
                    logger.info(
                        f"  Acc: {val_metrics['accuracy']:.4f} "
                        f"(baseline: {baseline:.4f})"
                    )
                    logger.info(f"  F1:  {val_metrics['f1']:.4f}")
                    logger.info(f"  AUC: {val_metrics['auc']:.4f}")

                    if val_metrics["accuracy"] > 0.98:
                        logger.error(
                            "🚨 Accuracy >98% at epoch 1 → Possible Data Leakage"
                        )
                    elif val_metrics["accuracy"] > 0.90:
                        logger.info("  ✅ Normal with ImageNet-21k pretrain")
                    else:
                        logger.info("  ✅ Normal growth")

                improved = val_metrics["f1"] > self.best_f1

                if improved:
                    self.best_f1 = val_metrics["f1"]
                    self.best_acc = val_metrics["accuracy"]
                    self.best_recall = val_metrics["recall"]
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, val_metrics)
                    logger.info(
                        f"  ✅ Best: F1={self.best_f1:.4f} "
                        f"Acc={self.best_acc:.4f} "
                        f"Recall={self.best_recall:.4f}"
                    )
                else:
                    self.patience_counter += 1

                logger.info(
                    f"E[{epoch + 1:3d}/{epochs}] "
                    f"TrL:{avg_train_loss:.4f} VaL:{val_metrics['loss']:.4f} "
                    f"Acc:{val_metrics['accuracy']:.4f} "
                    f"F1:{val_metrics['f1']:.4f} "
                    f"Recall:{val_metrics['recall']:.4f} "
                    f"AUC:{val_metrics['auc']:.4f} "
                    f"P:{self.patience_counter}/{self.max_patience}"
                )

                if self.patience_counter >= self.max_patience:
                    logger.info(f"⚠️  Early stopping at epoch {epoch + 1}")
                    break

            logger.info(f"\n✅ Training complete. Best F1={self.best_f1:.4f}")

            logger.info("\n═══ CONFIDENCE CALIBRATION ═══")
            self.load_best()
            temp = self.temp_scaler.fit(self.model, val_loader, self.device)
            self.tracker.log_metric("temperature", temp)

            if data["test"][0]:
                test_metrics = self._evaluate_on_test(data, image_size)

                if test_metrics:
                    self.tracker.log_metrics(
                        {
                            "test_accuracy": test_metrics["accuracy"],
                            "test_f1_macro": test_metrics["f1"],
                            "test_precision_macro": test_metrics["precision"],
                            "test_recall_macro": test_metrics["recall"],
                            "test_auc_macro": test_metrics["auc"],
                        }
                    )

            self._save_checkpoint_with_temperature(temp)

            self.tracker.log_model(self.model, "colon_cancer_classifier")
            logger.info("  📦 Model logged to MLflow")

    def _validate(self, loader, criterion):
        """
        Evaluate the model on a data loader and return classification metrics.

        Args:
            loader (DataLoader): Data loader yielding ``(images, labels)``
                batches.
            criterion (torch.nn.Module): Loss function used to accumulate
                the reported validation loss.

        Returns:
            dict: Dictionary with the following keys:
                - ``"loss"`` (float): Average batch loss.
                - ``"accuracy"`` (float): Overall accuracy.
                - ``"f1"`` (float): Macro-averaged F1 score.
                - ``"precision"`` (float): Macro-averaged precision.
                - ``"recall"`` (float): Macro-averaged recall.
                - ``"auc"`` (float): Macro one-vs-rest ROC AUC.
                - ``"predictions"`` (np.ndarray): Predicted class per sample.
                - ``"labels"`` (np.ndarray): Ground-truth label per sample.
                - ``"probabilities"`` (np.ndarray): Softmax probability
                  matrix, shape (N, num_classes).
        """
        self.model.eval()
        val_loss = 0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                val_loss += criterion(outputs, labels).item()

                scaled = self.temp_scaler.scale(outputs)
                all_probs.append(torch.softmax(scaled, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        probs = np.vstack(all_probs)
        labels = np.array(all_labels)
        preds = np.argmax(probs, axis=1)

        try:
            auc = (
                roc_auc_score(labels, probs, multi_class="ovr", average="macro")
                if len(np.unique(labels)) > 1
                else 0.0
            )
        except Exception:
            auc = 0.0

        return {
            "loss": val_loss / max(len(loader), 1),
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro", zero_division=0),
            "precision": precision_score(
                labels, preds, average="macro", zero_division=0
            ),
            "recall": recall_score(labels, preds, average="macro", zero_division=0),
            "auc": auc,
            "predictions": preds,
            "labels": labels,
            "probabilities": probs,
        }

    def _evaluate_on_test(self, data, img_size):
        """
        Run a detailed evaluation on the held-out test set.

        Loads the best checkpoint, builds a test ``ColonoscopyDataset``,
        computes all classification metrics, logs a per-class report and
        confusion matrix, breaks down accuracy by image source, and reports
        per-class recall with clinical significance when available.

        Args:
            data (dict): Data dictionary as returned by ``_load_data_from_db``,
                containing the ``"test"`` key.
            img_size (int): Spatial resolution used when constructing the
                test dataset.

        Returns:
            dict | None: Metrics dictionary as returned by ``_validate``,
                or ``None`` if the test split is empty.
        """
        logger.info("\n" + "=" * 70)
        logger.info("TEST SET - FINAL EVALUATION (MULTI-SOURCE)")
        logger.info("=" * 70)

        self.load_best()

        test_paths, test_labels, test_masks = data["test"]

        if not test_paths:
            logger.warning("No test data available.")
            return None

        test_ds = ColonoscopyDataset(
            test_paths,
            test_labels,
            test_masks,
            transform=get_val_transforms(img_size),
            image_size=img_size,
            mode="classification",
        )
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

        criterion = FocalLoss(num_classes=self.num_classes, label_smoothing=0.0)
        m = self._validate(test_loader, criterion)

        logger.info(f"\n  Accuracy:    {m['accuracy']:.4f}")
        logger.info(f"  F1 macro:    {m['f1']:.4f}")
        logger.info(f"  Precision:   {m['precision']:.4f}")
        logger.info(f"  Recall:      {m['recall']:.4f}")
        logger.info(f"  AUC macro:   {m['auc']:.4f}")
        logger.info(f"  Temperature: {self.temp_scaler.temperature:.2f}")

        names = [self.class_names.get(i, str(i)) for i in range(self.num_classes)]
        report = classification_report(
            m["labels"],
            m["predictions"],
            target_names=names,
            zero_division=0,
        )
        logger.info(f"\n{report}")

        cm = confusion_matrix(m["labels"], m["predictions"])
        logger.info("  Confusion Matrix:")
        header = "            " + " ".join(f"{n[:12]:>12s}" for n in names)
        logger.info(header)
        for i, row in enumerate(cm):
            logger.info(f"  {names[i][:12]:>12s} " + " ".join(f"{v:12d}" for v in row))

        logger.info("\n  📊 ANALYSIS BY SOURCE:")
        source_results = {}
        for idx, p in enumerate(test_paths):
            source = detect_source_from_stem(Path(p).stem)

            if source not in source_results:
                source_results[source] = {"correct": 0, "total": 0}

            source_results[source]["total"] += 1
            if m["predictions"][idx] == m["labels"][idx]:
                source_results[source]["correct"] += 1

        for source, stats in sorted(source_results.items()):
            acc = stats["correct"] / max(stats["total"], 1)
            logger.info(
                f"    {source:15s}: {stats['correct']}/{stats['total']} ({acc:.1%})"
            )

        clinical = self.class_mapping.get("clinical_info", {})
        if clinical:
            logger.info("\n  📋 CLINICAL SIGNIFICANCE:")
            for label_str, info in clinical.items():
                label = int(label_str)
                name = self.class_names.get(label, "?")
                r = recall_score(
                    m["labels"],
                    m["predictions"],
                    labels=[label],
                    average=None,
                    zero_division=0,
                )
                r_val = r[0] if len(r) > 0 else 0
                logger.info(
                    f"    {name}: recall={r_val:.2%} | "
                    f"Risk: {info.get('risk_level', 'N/A')} | "
                    f"Action: {info.get('action', 'N/A')}"
                )

        logger.info("=" * 70)
        return m

    def evaluate_on_test(self):
        """
        Standalone entry point to evaluate the best checkpoint on the test set.

        Loads data from the database and delegates to ``_evaluate_on_test``.

        Returns:
            dict | None: Metrics dictionary or ``None`` if the test split
                is empty.
        """
        data = self._load_data_from_db()
        return self._evaluate_on_test(data, 384)

    def _save_checkpoint(self, epoch, metrics):
        """
        Persist the current model state to disk as the best checkpoint.

        Args:
            epoch (int): Zero-based epoch index at which this checkpoint
                was produced.
            metrics (dict): Validation metrics dictionary as returned by
                ``_validate``. Array fields are excluded from the saved file.
        """
        path = paths.CLASSIFIER_CHECKPOINT
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "model_name": self.model_name,
                "num_classes": self.num_classes,
                "class_mapping": self.class_mapping,
                "best_f1": self.best_f1,
                "best_acc": self.best_acc,
                "best_recall": self.best_recall,
                "temperature": self.temp_scaler.temperature,
                "metrics": {
                    k: v
                    for k, v in metrics.items()
                    if k not in ("predictions", "labels", "probabilities")
                },
            },
            path,
        )

    def _save_checkpoint_with_temperature(self, temperature):
        """
        Update the temperature field in an existing checkpoint file.

        Loads the checkpoint at ``paths.CLASSIFIER_CHECKPOINT``, overwrites
        the ``"temperature"`` key, and saves it back to the same path.

        Args:
            temperature (float): Calibrated temperature value to store.
        """
        path = paths.CLASSIFIER_CHECKPOINT
        if path.exists():
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            ckpt["temperature"] = temperature
            torch.save(ckpt, path)
            logger.info(f"  🌡️ Checkpoint updated with temperature={temperature:.2f}")

    def load_best(self):
        """
        Restore model weights and temperature from the best saved checkpoint.

        Logs a warning if no checkpoint file is found at
        ``paths.CLASSIFIER_CHECKPOINT``.
        """
        path = paths.CLASSIFIER_CHECKPOINT
        if not path.exists():
            logger.warning("No checkpoint found.")
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])

        if "temperature" in ckpt:
            self.temp_scaler.temperature = ckpt["temperature"]

        logger.info(
            f"✅ Loaded: F1={ckpt.get('best_f1', 0):.4f} "
            f"Acc={ckpt.get('best_acc', 0):.4f} "
            f"Temp={ckpt.get('temperature', 1.0):.2f}"
        )


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("COLON CANCER IMAGE CLASSIFIER (MULTI-SOURCE)")
    logger.info("  Classes: Normal | Polyp | Inflammation")
    logger.info("  Sources: HyperKvasir + CVC-ClinicDB + LIMUC")
    logger.info("  Mixup + Temperature Scaling")
    logger.info("=" * 70)

    trainer = ImageClassifierTrainer(
        model_name="tf_efficientnetv2_s.in21k",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    trainer.train(
        epochs=25,
        batch_size=16,
        lr=1e-4,
        image_size=384,
        use_mixup=False,
        mixup_alpha=0.2,
    )
