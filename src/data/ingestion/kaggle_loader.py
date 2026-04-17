"""
Downloads and organizes Kaggle datasets.
"""

from pathlib import Path
import shutil
import subprocess

import pandas as pd

from src.config.constants import IMAGE_EXTENSIONS, KAGGLE_DATASETS
from src.config.logger import log as logger
from src.config.paths import paths


class KaggleLoader:
    """
    Utility class for downloading Kaggle datasets via the Kaggle CLI and
    organising the resulting files into the project directory structure.

    Requires a valid ``kaggle.json`` credentials file located at
    ``paths.ROOT / ".kaggle" / "kaggle.json"``. A warning is logged if
    the file is absent, but initialisation does not fail so that other
    methods (e.g. ``verify_datasets``) remain usable.
    """

    def __init__(self):
        """
        Initialise the loader and verify that Kaggle credentials are present.

        Logs a warning if ``kaggle.json`` is not found. No exception is
        raised so that non-download operations remain available.
        """
        kaggle_dir = paths.ROOT / ".kaggle"
        if not (kaggle_dir / "kaggle.json").exists():
            logger.warning("kaggle.json not found. Configure Kaggle API.")

    def download_dataset(self, dataset_key, target_dir=None):
        """
        Download and unzip a single Kaggle dataset by its project key.

        Invokes the Kaggle CLI (``kaggle datasets download``) as a
        subprocess with the ``--unzip`` flag.

        Args:
            dataset_key (str): Key identifying the dataset in
                ``KAGGLE_DATASETS`` (e.g. ``"cvc_clinicdb"``).
            target_dir (Path | None): Destination directory. Defaults to
                ``paths.RAW / dataset_key`` if ``None``.

        Raises:
            ValueError: If ``dataset_key`` is not present in
                ``KAGGLE_DATASETS``.
            subprocess.CalledProcessError: If the Kaggle CLI returns a
                non-zero exit code.
        """
        if dataset_key not in KAGGLE_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        dataset_name = KAGGLE_DATASETS[dataset_key]
        target = target_dir or paths.RAW / dataset_key
        target.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {dataset_name} → {target}")
        try:
            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    dataset_name,
                    "-p",
                    str(target),
                    "--unzip",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"✅ Downloaded: {dataset_key}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error: {e.stderr}")
            raise

    def download_all(self):
        """
        Download all datasets defined in ``KAGGLE_DATASETS``.

        Iterates over every key in ``KAGGLE_DATASETS`` and calls
        ``download_dataset`` for each. Failures are caught and logged
        without stopping the remaining downloads.
        """
        logger.info("\n═══ Downloading datasets ═══")
        for key in KAGGLE_DATASETS:
            try:
                self.download_dataset(key)
            except Exception as e:
                logger.error(f"Could not download {key}: {e}")

    def download_multi_source(self):
        """
        Download only the supplementary multi-source datasets.

        Downloads ``"cvc_clinicdb"`` and ``"limuc"`` if their target
        directories are absent or empty. Datasets that already exist on
        disk are skipped.
        """
        for key in ["cvc_clinicdb", "limuc"]:
            target = paths.RAW / key
            if target.exists() and any(target.iterdir()):
                logger.info(f"  ✅ {key} already exists")
                continue
            try:
                self.download_dataset(key, target)
            except Exception as e:
                logger.error(f"  ❌ Error downloading {key}: {e}")

    def verify_datasets(self):
        """
        Check whether the expected raw dataset directories contain images.

        Inspects HyperKvasir, CVC-ClinicDB, and LIMUC directories for
        at least one JPEG or PNG file using a short-circuit glob.

        Returns:
            dict[str, bool]: Mapping from dataset name to ``True`` if at
                least one image file was found, ``False`` otherwise.
        """
        logger.info("\n═══ Verifying datasets ═══")
        checks = {
            "hyperkvasir": paths.HYPERKVASIR_RAW,
            "cvc_clinicdb": paths.RAW / "cvc_clinicdb",
            "limuc": paths.RAW / "limuc",
        }
        status = {}
        for name, path in checks.items():
            exists = False
            if path.exists():
                exists = bool(list(path.rglob("*.jpg"))[:1]) or bool(
                    list(path.rglob("*.png"))[:1]
                )
            status[name] = exists
            logger.info(f"  {'✅' if exists else '❌'} {name}: {path}")
        return status

    def _glob_images(self, directory: Path) -> list[Path]:
        """
        Return all image files in a directory matching known extensions.

        Args:
            directory (Path): Directory to scan.

        Returns:
            list[Path]: Sorted list of image file paths whose suffix
                appears in ``IMAGE_EXTENSIONS``.
        """
        images: list[Path] = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(directory.glob(f"*{ext}"))
        return sorted(images)

    def load_tabular_risk_data(self):
        """
        Load the first CSV file found in the tabular risk data directory.

        Args:
            None

        Returns:
            pd.DataFrame: DataFrame loaded from the first CSV file found
                in ``paths.RAW / "tabular_risk"``.

        Raises:
            FileNotFoundError: If no CSV files are found in the expected
                directory.
        """
        data_dir = paths.RAW / "tabular_risk"
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSVs in {data_dir}")
        df = pd.read_csv(csv_files[0])
        logger.info(f"Tabular loaded: {len(df)} rows, {len(df.columns)} cols")
        return df

    def organize_kvasir_seg(self):
        """
        Copy Kvasir-SEG images and their paired masks into the raw image store.

        Looks for the standard ``images/`` and ``masks/`` subdirectories
        within ``paths.RAW / "kvasir_seg"``. Each image is prefixed with
        ``"kvasir_"`` and copied to ``paths.RAW_IMAGES / "polyps/"``.
        The matching mask (if present) is copied to
        ``paths.RAW_IMAGES / "masks/"`` with the same prefixed name.

        Logs an error if the Kvasir-SEG directory structure is not
        recognised.
        """
        src_dir = paths.RAW / "kvasir_seg"
        img_src = mask_src = None
        for candidate in [src_dir / "Kvasir-SEG", src_dir]:
            if (candidate / "images").exists():
                img_src = candidate / "images"
                mask_src = candidate / "masks"
                break
        if not img_src:
            logger.error("Kvasir-SEG structure not recognized")
            return
        polyp_dir = paths.RAW_IMAGES / "polyps"
        mask_dir = paths.RAW_IMAGES / "masks"
        polyp_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for img_file in self._glob_images(img_src):
            shutil.copy2(img_file, polyp_dir / f"kvasir_{img_file.name}")
            mask_file = mask_src / img_file.name
            if mask_file.exists():
                shutil.copy2(mask_file, mask_dir / f"kvasir_{img_file.name}")
            count += 1
        logger.info(f"✅ Kvasir-SEG organized: {count} polyp images with masks")

    def organize_curated_colon(self):
        """
        Copy Curated Colon images into the raw image store by folder classification.

        Recursively scans ``paths.RAW / "curated_colon"`` and classifies
        each subdirectory as polyp or normal based on its lowercased name.
        Polyp keywords: ``"polyp"``, ``"adenoma"``, ``"cancer"``,
        ``"tumor"``, ``"malignant"``. Normal keywords: ``"normal"``,
        ``"healthy"``, ``"benign"``, ``"negative"``. Directories matching
        neither keyword set are skipped.

        Each image is prefixed with ``"curated_"`` and copied to the
        appropriate target directory.
        """
        src_dir = paths.RAW / "curated_colon"
        polyp_dir = paths.RAW_IMAGES / "polyps"
        normal_dir = paths.RAW_IMAGES / "normal"
        polyp_dir.mkdir(parents=True, exist_ok=True)
        normal_dir.mkdir(parents=True, exist_ok=True)
        polyp_count = normal_count = 0
        for subdir in src_dir.rglob("*"):
            if not subdir.is_dir():
                continue
            dir_name = subdir.name.lower()
            is_polyp = any(
                kw in dir_name
                for kw in ["polyp", "adenoma", "cancer", "tumor", "malignant"]
            )
            is_normal = any(
                kw in dir_name for kw in ["normal", "healthy", "benign", "negative"]
            )
            if not is_polyp and not is_normal:
                continue
            target = polyp_dir if is_polyp else normal_dir
            for img_file in self._glob_images(subdir):
                shutil.copy2(img_file, target / f"curated_{img_file.name}")
                if is_polyp:
                    polyp_count += 1
                else:
                    normal_count += 1
        logger.info(f"✅ Curated colon: {polyp_count} polyps, {normal_count} normals")

    def organize_all_images(self):
        """
        Run all image organisation steps in sequence and log a summary.

        Calls ``organize_kvasir_seg`` and ``organize_curated_colon``,
        then counts and logs the total number of polyp, normal, and mask
        files in ``paths.RAW_IMAGES``.
        """
        logger.info("\n═══ Organizing images ═══")
        self.organize_kvasir_seg()
        self.organize_curated_colon()
        polyp_count = len(list(paths.RAW_IMAGES.glob("polyps/*")))
        normal_count = len(list(paths.RAW_IMAGES.glob("normal/*")))
        mask_count = len(list(paths.RAW_IMAGES.glob("masks/*")))
        logger.info(
            f"\n📊 SUMMARY:\n   Polyps: {polyp_count}\n"
            f"   Normals: {normal_count}\n"
            f"   Masks: {mask_count}"
        )


if __name__ == "__main__":
    loader = KaggleLoader()
    loader.download_all()
    loader.organize_all_images()
    df = loader.load_tabular_risk_data()
    print(df.describe())
