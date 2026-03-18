"""
Registers images in the database with stratified splits.
"""

from data.database.connection import get_db
from data.database.repositories import TrainingImageRepository
from PIL import Image
from sklearn.model_selection import train_test_split

from config.constants import IMAGE_EXTENSIONS
from config.logger import log as logger
from config.paths import paths


def register_images_in_db():
    """
    Register images in the database with stratified splits.
    """
    polyp_dir = paths.RAW_IMAGES / "polyps"
    normal_dir = paths.RAW_IMAGES / "normal"
    mask_dir = paths.RAW_IMAGES / "masks"

    records = []

    for img_path in polyp_dir.glob("*.*"):
        if img_path.suffix.lower() in IMAGE_EXTENSIONS:
            mask_path = mask_dir / img_path.name
            try:
                img = Image.open(img_path)
                w, h = img.size
            except Exception:
                w, h = None, None
            records.append(
                {
                    "file_path": str(img_path),
                    "mask_path": str(mask_path) if mask_path.exists() else None,
                    "dataset_source": img_path.name.split("_")[0],
                    "label": 1,
                    "width": w,
                    "height": h,
                }
            )

    for img_path in normal_dir.glob("*.*"):
        if img_path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                img = Image.open(img_path)
                w, h = img.size
            except Exception:
                w, h = None, None
            records.append(
                {
                    "file_path": str(img_path),
                    "mask_path": None,
                    "dataset_source": img_path.name.split("_")[0],
                    "label": 0,
                    "width": w,
                    "height": h,
                }
            )

    if not records:
        logger.error("No images found to register")
        return

    labels = [r["label"] for r in records]
    indices = list(range(len(records)))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=labels, random_state=42
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
    )

    for i in train_idx:
        records[i]["split"] = "train"
    for i in val_idx:
        records[i]["split"] = "val"
    for i in test_idx:
        records[i]["split"] = "test"

    with get_db() as db:
        repo = TrainingImageRepository(db)
        count = repo.bulk_insert(records)
        logger.info(f"✅ {count} images registered")
        stats = repo.get_stats()
        for label, split, cnt in stats:
            label_name = "polyp" if label == 1 else "normal"
            logger.info(f"  {split}: {label_name} = {cnt}")


if __name__ == "__main__":
    register_images_in_db()
