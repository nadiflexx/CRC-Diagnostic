"""
src/training/train_tissue_classifier.py

Training pipeline for Model B (Tissue-Only) using EfficientNetV2-S.
SAME backbone as Model A. Diversity is achieved through different preprocessing.

Strategy:
    - Training: random selection of 1 out of N crops per image per epoch.
    - Validation: average of ALL crops per image.
    - Same learning rate and optimizer as Model A (same architecture).
"""

from collections import Counter
import json
from pathlib import Path
import platform
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config.constants import IMAGENET_MEAN, IMAGENET_STD, detect_source_from_stem
from src.config.logger import log as logger
from src.config.paths import paths
from src.data.processing.tissue_only_preprocessor import (
    process_dataset_tissue_only,
)
from src.database.connection import get_db
from src.database.repositories import TrainingImageRepository
from src.models.image_classifier import FocalLoss
from src.models.tissue_classifier import TissueOnlyClassifier
from src.training.tracking import MLflowTracker

TISSUE_DIR = paths.COLON_TISSUE_ONLY

# ═══════════════════════════════════════════════════════════
# TRANSFORMS
# ═══════════════════════════════════════════════════════════


def get_tissue_train_transforms(image_size: int = 384) -> A.Compose:
    """
    Build the training augmentation pipeline for tissue-only crops.

    Augmentation is intentionally reduced because multi-crop sampling
    already provides sufficient variability during training.

    Args:
        image_size (int): Target spatial resolution. Default is 384.
            Currently unused in the transform body (crops are already
            resized during preprocessing), but kept for API consistency.

    Returns:
        A.Compose: Albumentations composition with the following steps:
            - Random rotation up to 15 degrees (p=0.4).
            - Horizontal flip (p=0.5).
            - Vertical flip (p=0.5).
            - One-of: brightness/contrast or hue/saturation jitter (p=0.5).
            - Gaussian blur with kernel (3, 5) (p=0.2).
            - ImageNet normalization.
            - Conversion to PyTorch tensor.
    """
    return A.Compose(
        [
            A.Rotate(
                limit=15,
                p=0.4,
                border_mode=cv2.BORDER_REFLECT_101,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.12,
                        contrast_limit=0.12,
                        p=1.0,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=8,
                        sat_shift_limit=12,
                        val_shift_limit=8,
                        p=1.0,
                    ),
                ],
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_tissue_val_transforms(image_size: int = 384) -> A.Compose:
    """
    Build the validation/test transform pipeline for tissue-only crops.

    No geometric or color augmentation is applied. Only normalization
    and tensor conversion are performed to ensure deterministic evaluation.

    Args:
        image_size (int): Target spatial resolution. Default is 384.
            Kept for API consistency with the training counterpart.

    Returns:
        A.Compose: Albumentations composition containing:
            - ImageNet normalization.
            - Conversion to PyTorch tensor.
    """
    return A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


# ═══════════════════════════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════════════════════════


class TissueTrainDataset(Dataset):
    """
    Training dataset that selects one random crop per image per epoch.

    For each original image, up to ``n_crops`` pre-generated tissue-only
    crop files are discovered. At every ``__getitem__`` call a single crop
    is chosen at random, introducing stochasticity without storing all
    crops in memory simultaneously.
    """

    def __init__(
        self,
        db_paths: list[str],
        labels: list[int],
        tissue_dir: Path,
        transform: A.Compose,
        image_size: int = 384,
        n_crops: int = 5,
    ):
        """
        Initialize the training dataset by discovering crop files on disk.

        Args:
            db_paths (list[str]): Absolute paths to the original images
                as stored in the database. Used to derive crop filenames.
            labels (list[int]): Integer class label for each entry in
                ``db_paths`` (must be the same length and order).
            tissue_dir (Path): Root directory that contains tissue-only
                crop subdirectories organized by class name.
            transform (A.Compose): Albumentations augmentation pipeline
                applied to each loaded crop.
            image_size (int): Expected spatial size (height and width) of
                each crop in pixels. Crops that deviate are resized.
                Default is 384.
            n_crops (int): Maximum number of crop variants expected per
                image (``_crop0.jpg`` … ``_crop{n_crops-1}.jpg``).
                Default is 5.
        """
        self.transform = transform
        self.image_size = image_size
        self.records: list[dict] = []

        missing = 0
        for db_path, label in zip(db_paths, labels, strict=True):
            stem = Path(db_path).stem
            class_name = Path(db_path).parent.name

            crop_paths = []
            for i in range(n_crops):
                crop_path = tissue_dir / class_name / f"{stem}_crop{i}.jpg"
                if crop_path.exists():
                    crop_paths.append(str(crop_path))

            if crop_paths:
                self.records.append({"crop_paths": crop_paths, "label": label})
            else:
                missing += 1

        if missing > 0:
            logger.warning(
                f"  ⚠️  {missing}/{missing + len(self.records)} "
                f"images without tissue-only crops"
            )

    def __len__(self) -> int:
        """
        Return the number of images (parent records) in the dataset.

        Returns:
            int: Total number of records with at least one valid crop.
        """
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Load and return one randomly chosen crop for the image at ``idx``.

        Args:
            idx (int): Index into ``self.records``.

        Returns:
            tuple[torch.Tensor, int]:
                - Transformed image tensor of shape (C, H, W).
                - Integer class label.
        """
        record = self.records[idx]
        crop_path = random.choice(record["crop_paths"])

        img = cv2.imread(crop_path)
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if h != self.image_size or w != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img, record["label"]


class TissueCropDataset(Dataset):
    """
    Flat validation dataset that exposes every crop as an individual sample.

    All crops for every parent image are enumerated and stored as separate
    entries. Each entry carries a ``parent_idx`` field that links it back
    to the originating image, enabling multi-crop averaging during
    validation.
    """

    def __init__(
        self,
        db_paths: list[str],
        labels: list[int],
        tissue_dir: Path,
        transform: A.Compose,
        image_size: int = 384,
        n_crops: int = 5,
    ):
        """
        Initialize the flat crop dataset by scanning all crop files on disk.

        Args:
            db_paths (list[str]): Absolute paths to the original images
                as stored in the database.
            labels (list[int]): Integer class label aligned with
                ``db_paths``.
            tissue_dir (Path): Root directory containing tissue-only crop
                subdirectories organized by class name.
            transform (A.Compose): Albumentations pipeline applied to
                each crop before returning it.
            image_size (int): Expected spatial size of each crop in pixels.
                Crops that deviate are resized. Default is 384.
            n_crops (int): Maximum number of crop variants expected per
                image. Default is 5.
        """
        self.transform = transform
        self.image_size = image_size
        self.entries: list[dict] = []
        self.n_parents = 0

        for parent_idx, (db_path, label) in enumerate(
            zip(db_paths, labels, strict=True)
        ):
            stem = Path(db_path).stem
            class_name = Path(db_path).parent.name
            found = False

            for i in range(n_crops):
                crop_path = tissue_dir / class_name / f"{stem}_crop{i}.jpg"
                if crop_path.exists():
                    self.entries.append(
                        {
                            "crop_path": str(crop_path),
                            "label": label,
                            "parent_idx": parent_idx,
                        }
                    )
                    found = True

            if found:
                self.n_parents = parent_idx + 1

    def __len__(self) -> int:
        """
        Return the total number of individual crop entries.

        Returns:
            int: Number of crop files discovered across all parent images.
        """
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        """
        Load and return the crop at position ``idx``.

        Args:
            idx (int): Index into ``self.entries``.

        Returns:
            tuple[torch.Tensor, int, int]:
                - Transformed image tensor of shape (C, H, W).
                - Integer class label of the parent image.
                - Parent image index used for multi-crop aggregation.
        """
        entry = self.entries[idx]

        img = cv2.imread(entry["crop_path"])
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if h != self.image_size or w != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        return img, entry["label"], entry["parent_idx"]


# ═══════════════════════════════════════════════════════════
# MULTI-CROP VALIDATION
# ═══════════════════════════════════════════════════════════


def validate_multicrop(
    model: TissueOnlyClassifier,
    crop_dataset: TissueCropDataset,
    criterion: torch.nn.Module,
    device: str,
    temperature: float = 1.0,
    batch_size: int = 32,
) -> dict:
    """
    Evaluate the model using multi-crop averaging per parent image.

    All crops belonging to the same parent image are forwarded through
    the model individually. Their softmax probability vectors are then
    averaged to produce a single prediction per image, reducing variance
    caused by crop position.

    Args:
        model (TissueOnlyClassifier): The model to evaluate. Set to eval
            mode internally.
        crop_dataset (TissueCropDataset): Flat dataset containing all
            crops with their associated ``parent_idx`` fields.
        criterion (torch.nn.Module): Loss function used to compute the
            reported validation loss (applied per batch before averaging).
        device (str): Target device identifier, e.g. ``"cuda"`` or
            ``"cpu"``.
        temperature (float): Temperature scaling factor applied to logits
            before the softmax. Values greater than 1 soften predictions.
            Default is 1.0 (no scaling).
        batch_size (int): Number of crops per forward pass. Default is 32.

    Returns:
        dict: Dictionary with the following keys:
            - ``"loss"`` (float): Average batch loss over all crops.
            - ``"accuracy"`` (float): Image-level accuracy after averaging.
            - ``"f1"`` (float): Macro-averaged F1 score.
            - ``"precision"`` (float): Macro-averaged precision.
            - ``"recall"`` (float): Macro-averaged recall.
            - ``"auc"`` (float): Macro one-vs-rest ROC AUC score.
            - ``"predictions"`` (np.ndarray): Predicted class per image.
            - ``"labels"`` (np.ndarray): Ground-truth label per image.
            - ``"probabilities"`` (np.ndarray): Averaged probability
              vectors, shape (n_images, n_classes).
            - ``"n_images"`` (int): Number of unique parent images.
            - ``"n_crops"`` (int): Total number of crops processed.
    """
    model.eval()

    is_win = platform.system() == "Windows"
    loader = DataLoader(
        crop_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if is_win else 4,
        pin_memory=not is_win,
    )

    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []
    all_parents: list[int] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, labels, parent_indices in loader:
            images = images.to(device)
            labels_dev = labels.to(device)

            logits = model(images)
            total_loss += criterion(logits, labels_dev).item()
            n_batches += 1

            scaled = logits / temperature
            probs = F.softmax(scaled, dim=1).cpu().numpy()

            all_probs.append(probs)
            all_labels.extend(labels.numpy().tolist())
            all_parents.extend(parent_indices.numpy().tolist())

    probs_flat = np.vstack(all_probs)
    parents_arr = np.array(all_parents)
    labels_arr = np.array(all_labels)

    unique_parents = sorted(set(all_parents))
    avg_probs = []
    avg_labels = []

    for p in unique_parents:
        mask = parents_arr == p
        avg_probs.append(probs_flat[mask].mean(axis=0))
        avg_labels.append(labels_arr[mask][0])

    avg_probs = np.array(avg_probs)
    avg_labels = np.array(avg_labels)
    preds = np.argmax(avg_probs, axis=1)

    try:
        auc = roc_auc_score(avg_labels, avg_probs, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": accuracy_score(avg_labels, preds),
        "f1": f1_score(avg_labels, preds, average="macro", zero_division=0),
        "precision": precision_score(
            avg_labels, preds, average="macro", zero_division=0
        ),
        "recall": recall_score(avg_labels, preds, average="macro", zero_division=0),
        "auc": auc,
        "predictions": preds,
        "labels": avg_labels,
        "probabilities": avg_probs,
        "n_images": len(unique_parents),
        "n_crops": len(probs_flat),
    }


# ═══════════════════════════════════════════════════════════
# TEMPERATURE SCALING
# ═══════════════════════════════════════════════════════════


class TemperatureScalerB:
    """
    Post-hoc temperature scaling calibrator for Model B.

    Performs a grid search over temperature values to find the one that
    minimises negative log-likelihood on multi-crop averaged logits from
    the validation set.
    """

    def __init__(self):
        """Initialise with temperature set to 1.0 (identity scaling)."""
        self.temperature = 1.0

    def fit(
        self,
        model: TissueOnlyClassifier,
        crop_dataset: TissueCropDataset,
        device: str,
        batch_size: int = 32,
    ) -> float:
        """
        Find the optimal temperature by minimising NLL on the validation set.

        Logits from all crops are first collected, then averaged per parent
        image, and finally a grid search over [0.5, 5.0) with step 0.1 is
        performed to select the temperature that yields the lowest cross-
        entropy loss.

        Args:
            model (TissueOnlyClassifier): Trained model used to collect
                logits. Set to eval mode internally.
            crop_dataset (TissueCropDataset): Flat validation crop dataset
                that provides ``parent_idx`` for aggregation.
            device (str): Target device identifier, e.g. ``"cuda"`` or
                ``"cpu"``.
            batch_size (int): Number of crops per forward pass. Default is 32.

        Returns:
            float: The optimal temperature value found by grid search.
                Also stored in ``self.temperature``.
        """
        model.eval()

        is_win = platform.system() == "Windows"
        loader = DataLoader(
            crop_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0 if is_win else 4,
        )

        all_logits, all_labels, all_parents = [], [], []

        with torch.no_grad():
            for images, labels, parents in loader:
                images = images.to(device)
                logits = model(images).cpu()
                all_logits.append(logits)
                all_labels.extend(labels.numpy().tolist())
                all_parents.extend(parents.numpy().tolist())

        logits_flat = torch.cat(all_logits)
        parents_arr = np.array(all_parents)
        labels_arr = np.array(all_labels)

        unique_parents = sorted(set(all_parents))
        avg_logits, avg_labels_list = [], []

        for p in unique_parents:
            mask_np = parents_arr == p
            mask_idx = np.where(mask_np)[0]
            avg_logits.append(logits_flat[mask_idx].mean(dim=0))
            avg_labels_list.append(labels_arr[mask_idx[0]])

        avg_logits_t = torch.stack(avg_logits)
        avg_labels_t = torch.tensor(avg_labels_list)

        best_nll = float("inf")
        best_t = 1.0

        for t in np.arange(0.5, 5.0, 0.1):
            scaled = avg_logits_t / t
            nll = F.cross_entropy(scaled, avg_labels_t).item()
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)

        self.temperature = best_t
        logger.info(f"  🌡️  Optimal temperature (Model B): {self.temperature:.2f}")
        return self.temperature


# ═══════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════


class TissueClassifierTrainer:
    """
    Training orchestrator for Model B (tissue-only, same backbone as Model A).

    Handles the full pipeline: preprocessing verification, data loading from
    the database, dataset construction, training loop with early stopping,
    temperature calibration, test evaluation, and checkpoint management.
    """

    def __init__(
        self,
        model_name: str = "tf_efficientnetv2_s.in21k",
        device: str | None = None,
    ):
        """
        Initialise the trainer and instantiate the model.

        Args:
            model_name (str): Timm model identifier for the backbone.
                Default is ``"tf_efficientnetv2_s.in21k"``.
            device (str | None): Target device. If ``None``, CUDA is used
                when available, otherwise CPU.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tissue_dir = TISSUE_DIR
        self.tracker = MLflowTracker(experiment_name="Colon_Cancer_Classification")

        self.class_mapping = self._load_class_mapping()
        self.num_classes = self.class_mapping["num_classes"]
        self.class_names = {int(k): v for k, v in self.class_mapping["classes"].items()}

        logger.info(f"  Model B: {model_name} (SAME as Model A)")
        logger.info(
            f"  Classes ({self.num_classes}): {list(self.class_names.values())}"
        )

        self.model = TissueOnlyClassifier(
            model_name=model_name,
            pretrained=True,
            dropout=0.4,
            num_classes=self.num_classes,
        ).to(self.device)

        self.temp_scaler = TemperatureScalerB()

        self.best_f1 = 0.0
        self.best_acc = 0.0
        self.best_recall = 0.0
        self.patience_counter = 0
        self.max_patience = 12
        self.history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_auc": [],
            "val_recall": [],
            "val_precision": [],
        }

    def _load_class_mapping(self) -> dict:
        """
        Load the class mapping JSON from the first path that exists.

        Searches in the following order:
            1. ``TISSUE_DIR / class_mapping.json``
            2. ``paths.COLON_PROCESSED / class_mapping.json``
            3. ``paths.COLON_CLEAN / class_mapping.json``

        Returns:
            dict: Parsed class mapping containing at least
                ``"num_classes"`` and ``"classes"`` keys.

        Raises:
            FileNotFoundError: If no ``class_mapping.json`` is found in
                any of the candidate paths.
        """
        for p in [
            TISSUE_DIR / "class_mapping.json",
            paths.COLON_PROCESSED / "class_mapping.json",
            paths.COLON_CLEAN / "class_mapping.json",
        ]:
            if p.exists():
                with open(p) as f:
                    mapping = json.load(f)
                logger.info(f"  📄 class_mapping: {p}")
                return mapping
        raise FileNotFoundError("class_mapping.json not found.")

    def _ensure_tissue_preprocessing(self):
        """
        Verify that tissue-only crops exist and run preprocessing if they do not.

        If ``TISSUE_DIR`` is non-empty the method logs the number of crops
        found and returns immediately. Otherwise it triggers
        ``process_dataset_tissue_only`` to generate the crops from the clean
        dataset.
        """
        if self.tissue_dir.exists() and any(self.tissue_dir.iterdir()):
            n = sum(
                1 for _ in self.tissue_dir.rglob("*.jpg") if _.parent.name != "masks"
            )
            logger.info(f"  ✅ Tissue-only already exists: {n} crops")
            return

        logger.info("  🔄 Running tissue-only preprocessing...")
        process_dataset_tissue_only(
            clean_dir=paths.COLON_CLEAN,
            processed_dir=self.tissue_dir,
            target_size=384,
            n_crops=5,
        )

    def _load_data_from_db(self) -> dict:
        """
        Load train, validation, and test splits from the database.

        Also validates that at least one tissue crop exists for a sample
        image before returning, to catch missing preprocessing early.

        Returns:
            dict: Dictionary with keys ``"train"``, ``"val"``, and
                ``"test"``. Each value is a tuple
                ``(paths: list[str], labels: list[int])``.

        Raises:
            ValueError: If no training images are found in the database.
            FileNotFoundError: If the expected tissue crop file for the
                first training image does not exist on disk.
        """
        with get_db() as db:
            repo = TrainingImageRepository(db)

            train_data = repo.get_by_split("train")
            train_paths = [str(d.file_path) for d in train_data]
            train_labels = [int(d.label) for d in train_data]

            val_data = repo.get_by_split("val")
            val_paths = [str(d.file_path) for d in val_data]
            val_labels = [int(d.label) for d in val_data]

            test_data = repo.get_by_split("test")
            test_paths = [str(d.file_path) for d in test_data]
            test_labels = [int(d.label) for d in test_data]

        if not train_paths:
            raise ValueError("No training data found in the database.")

        result = {
            "train": (train_paths, train_labels),
            "val": (val_paths, val_labels),
            "test": (test_paths, test_labels),
        }

        sample = Path(train_paths[0])
        stem = sample.stem
        class_name = sample.parent.name
        test_crop = self.tissue_dir / class_name / f"{stem}_crop0.jpg"

        if not test_crop.exists():
            raise FileNotFoundError(
                f"Tissue crop not found: {test_crop}\n"
                f"Run: python -m src.data_processing."
                f"tissue_only_preprocessor"
            )

        for split in ["train", "val", "test"]:
            p, labels = result[split]
            counts = Counter(labels)
            detail = ", ".join(
                f"{self.class_names.get(k, '?')}={v}" for k, v in sorted(counts.items())
            )
            logger.info(f"  {split}: {len(labels)} images ({detail})")

        return result

    def train(
        self,
        epochs: int = 25,
        batch_size: int = 16,
        lr: float = 1e-4,
        image_size: int = 384,
        n_crops: int = 5,
    ):
        """
        Run the full training pipeline for the tissue-only classifier.

        Steps performed:
            1. Ensure tissue-only crops exist on disk.
            2. Start an MLflow run and log hyperparameters.
            3. Load data splits from the database.
            4. Compute balanced class weights.
            5. Build datasets and data loaders.
            6. Instantiate FocalLoss, AdamW optimiser, and cosine scheduler.
            7. Execute the training loop with early stopping.
            8. Calibrate the model via temperature scaling.
            9. Evaluate on the test set if available.
            10. Log the model artifact to MLflow.

        Args:
            epochs (int): Maximum number of training epochs. Default is 25.
            batch_size (int): Number of samples per training batch.
                Default is 16.
            lr (float): Initial learning rate for AdamW. Default is 1e-4.
            image_size (int): Spatial resolution of the input crops in
                pixels. Default is 384.
            n_crops (int): Number of crop variants per image used to
                build the datasets. Default is 5.
        """
        self._ensure_tissue_preprocessing()

        with self.tracker.start_run(f"tissue_only_{self.model_name}"):
            self.tracker.log_params(
                {
                    "model_type": "tissue_only_model_B",
                    "model_name": self.model_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "image_size": image_size,
                    "n_crops": n_crops,
                    "preprocessing": "slide_shrink_resize_direct",
                    "padding": "NONE",
                    "loss": "focal_loss_smoothed",
                    "note": "SAME backbone as Model A, different preprocessing",
                }
            )

            logger.info("\n═══ LOADING DATA (TISSUE-ONLY) ═══")
            data = self._load_data_from_db()
            train_paths, train_labels = data["train"]
            val_paths, val_labels = data["val"]

            classes = np.unique(train_labels)
            weights = compute_class_weight(
                "balanced", classes=classes, y=np.array(train_labels)
            )
            weight_dict = dict(
                zip(classes.tolist(), weights.round(3).tolist(), strict=True)
            )
            logger.info(f"  Class weights: {weight_dict}")

            train_ds = TissueTrainDataset(
                train_paths,
                train_labels,
                self.tissue_dir,
                transform=get_tissue_train_transforms(image_size),
                image_size=image_size,
                n_crops=n_crops,
            )
            val_crop_ds = TissueCropDataset(
                val_paths,
                val_labels,
                self.tissue_dir,
                transform=get_tissue_val_transforms(image_size),
                image_size=image_size,
                n_crops=n_crops,
            )

            logger.info(f"  Train: {len(train_ds)} images (random crop from {n_crops})")
            logger.info(
                f"  Val: {val_crop_ds.n_parents} images "
                f"× {n_crops} crops = {len(val_crop_ds)} crops"
            )

            is_win = platform.system() == "Windows"
            loader_kw = {
                "batch_size": batch_size,
                "num_workers": 0 if is_win else 4,
                "pin_memory": not is_win,
            }

            train_loader = DataLoader(
                train_ds, shuffle=True, drop_last=True, **loader_kw
            )

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

            logger.info(f"\n═══ TISSUE-ONLY TRAINING ({self.num_classes} CLASSES) ═══")
            logger.info(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
            logger.info(f"  Backbone: {self.model_name} (SAME as Model A)")

            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0
                batch_count = 0

                pbar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{epochs} [Tissue]",
                    leave=False,
                    dynamic_ncols=True,
                )

                for images, labels in pbar:
                    try:
                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        optimizer.zero_grad()
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()

                        train_loss += loss.item()
                        batch_count += 1
                        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                    except Exception as e:
                        logger.error(f"  Batch error: {e}")
                        continue

                scheduler.step()
                avg_train_loss = train_loss / max(batch_count, 1)

                val_metrics = validate_multicrop(
                    model=self.model,
                    crop_dataset=val_crop_ds,
                    criterion=criterion,
                    device=self.device,
                    temperature=self.temp_scaler.temperature,
                    batch_size=batch_size * 2,
                )

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
                        "tissue_train_loss": avg_train_loss,
                        "tissue_val_loss": val_metrics["loss"],
                        "tissue_val_accuracy": val_metrics["accuracy"],
                        "tissue_val_f1_macro": val_metrics["f1"],
                        "tissue_val_recall_macro": val_metrics["recall"],
                        "tissue_val_auc_macro": val_metrics["auc"],
                        "tissue_lr": current_lr,
                    },
                    step=epoch,
                )

                improved = val_metrics["f1"] > self.best_f1

                if improved:
                    self.best_f1 = val_metrics["f1"]
                    self.best_acc = val_metrics["accuracy"]
                    self.best_recall = val_metrics["recall"]
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, val_metrics)
                else:
                    self.patience_counter += 1

                logger.info(
                    f"E[{epoch + 1:3d}/{epochs}] "
                    f"TrL:{avg_train_loss:.4f} "
                    f"VaL:{val_metrics['loss']:.4f} "
                    f"Acc:{val_metrics['accuracy']:.4f} "
                    f"F1:{val_metrics['f1']:.4f} "
                    f"Recall:{val_metrics['recall']:.4f} "
                    f"AUC:{val_metrics['auc']:.4f} "
                    f"({val_metrics['n_images']} imgs, "
                    f"{val_metrics['n_crops']} crops) "
                    f"P:{self.patience_counter}/{self.max_patience}"
                    + (" ★" if improved else "")
                )

                if self.patience_counter >= self.max_patience:
                    logger.info(f"  ⚠️  Early stopping at epoch {epoch + 1}")
                    break

            logger.info("\n═══ CALIBRATION (MULTI-CROP) ═══")
            self._load_best()
            temp = self.temp_scaler.fit(
                self.model, val_crop_ds, self.device, batch_size * 2
            )
            self.tracker.log_metric("tissue_temperature", temp)
            self._update_checkpoint_temperature(temp)

            if data["test"][0]:
                self._evaluate_test(data, image_size, n_crops)

            self.tracker.log_model(self.model, "tissue_only_classifier")
            logger.info(f"\n✅ TISSUE-ONLY COMPLETE. Best F1={self.best_f1:.4f}")

    def _evaluate_test(self, data: dict, image_size: int, n_crops: int):
        """
        Evaluate the best checkpoint on the test set using multi-crop averaging.

        Loads the best saved checkpoint, builds a ``TissueCropDataset`` for
        the test split, runs ``validate_multicrop``, logs a per-class
        classification report and confusion matrix, and prints a breakdown
        of accuracy per image source.

        Args:
            data (dict): Data dictionary as returned by ``_load_data_from_db``,
                containing ``"test"`` key with ``(paths, labels)`` tuple.
            image_size (int): Spatial resolution expected by the model.
            n_crops (int): Number of crop variants per test image.
        """
        logger.info("\n" + "=" * 70)
        logger.info("TEST SET - MODEL B (TISSUE-ONLY)")
        logger.info("=" * 70)

        self._load_best()
        test_paths, test_labels = data["test"]

        test_crop_ds = TissueCropDataset(
            test_paths,
            test_labels,
            self.tissue_dir,
            transform=get_tissue_val_transforms(image_size),
            image_size=image_size,
            n_crops=n_crops,
        )

        criterion = FocalLoss(num_classes=self.num_classes, label_smoothing=0.0)
        m = validate_multicrop(
            model=self.model,
            crop_dataset=test_crop_ds,
            criterion=criterion,
            device=self.device,
            temperature=self.temp_scaler.temperature,
        )

        logger.info(f"\n  Accuracy:    {m['accuracy']:.4f}")
        logger.info(f"  F1 macro:    {m['f1']:.4f}")
        logger.info(f"  Precision:   {m['precision']:.4f}")
        logger.info(f"  Recall:      {m['recall']:.4f}")
        logger.info(f"  AUC macro:   {m['auc']:.4f}")
        logger.info(f"  Images/Crops: {m['n_images']}/{m['n_crops']}")

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
        source_results: dict[str, dict] = {}
        for idx, p in enumerate(test_paths):
            source = detect_source_from_stem(Path(p).stem)

            if source not in source_results:
                source_results[source] = {"correct": 0, "total": 0}

            if idx < len(m["predictions"]):
                source_results[source]["total"] += 1
                if m["predictions"][idx] == m["labels"][idx]:
                    source_results[source]["correct"] += 1

        for source, stats in sorted(source_results.items()):
            acc = stats["correct"] / max(stats["total"], 1)
            logger.info(
                f"    {source:15s}: {stats['correct']}/{stats['total']} ({acc:.1%})"
            )
        logger.info("=" * 70)

    def _save_checkpoint(self, epoch: int, metrics: dict):
        """
        Persist the current model state to disk as the best checkpoint.

        Saves model weights, metadata, and scalar metrics (excluding arrays)
        to ``paths.TISSUE_CLASSIFIER_CHECKPOINT``.

        Args:
            epoch (int): Zero-based epoch index at which this checkpoint was
                produced.
            metrics (dict): Validation metrics dictionary as returned by
                ``validate_multicrop``. Array fields (``"predictions"``,
                ``"labels"``, ``"probabilities"``) are excluded from the
                saved file.
        """
        path = paths.TISSUE_CLASSIFIER_CHECKPOINT
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
                "crop_strategy": "slide_shrink_resize_direct_no_padding",
                "n_crops": 5,
                "model_type": "tissue_only_same_backbone",
                "metrics": {
                    k: v
                    for k, v in metrics.items()
                    if k not in ("predictions", "labels", "probabilities")
                },
            },
            path,
        )
        logger.info(f"  💾 Checkpoint saved: {path}")

    def _update_checkpoint_temperature(self, temperature: float):
        """
        Update the temperature field in an existing checkpoint file.

        Loads the checkpoint, overwrites the ``"temperature"`` key, and
        saves it back to the same path.

        Args:
            temperature (float): Calibrated temperature value to store.
        """
        path = paths.TISSUE_CLASSIFIER_CHECKPOINT
        if path.exists():
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            ckpt["temperature"] = temperature
            torch.save(ckpt, path)

    def _load_best(self):
        """
        Restore model weights from the best saved checkpoint.

        Also restores the temperature value to ``self.temp_scaler.temperature``
        if it is present in the checkpoint. Logs a warning if no checkpoint
        file is found.
        """
        path = paths.TISSUE_CLASSIFIER_CHECKPOINT
        if not path.exists():
            logger.warning("No tissue-only checkpoint found.")
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "temperature" in ckpt:
            self.temp_scaler.temperature = ckpt["temperature"]
        logger.info(
            f"  ✅ Tissue-only checkpoint loaded: "
            f"F1={ckpt.get('best_f1', 0):.4f} "
            f"T={ckpt.get('temperature', 1.0):.2f}"
        )


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("MODEL B: TISSUE-ONLY (EfficientNetV2-S)")
    logger.info("  SAME backbone as Model A")
    logger.info("  Diversity = different preprocessing")
    logger.info("  ZERO padding, ZERO border artifacts")
    logger.info("=" * 70)

    trainer = TissueClassifierTrainer(
        model_name="tf_efficientnetv2_s.in21k",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    trainer.train(
        epochs=25,
        batch_size=16,
        lr=1e-4,
        image_size=384,
        n_crops=5,
    )
