"""
Multi-source organizer: Mixes HyperKvasir + CVC-ClinicDB + LIMUC + Curated Colon.
Get data from all available raw sources:
    - HyperKvasir
    - CVC-ClinicDB
    - LIMUC
    - Curated Colon
    - Tabular Data

Processing steps:
    1. Collect images from all available raw sources.
    2. Log and optionally subsample to ``target_per_class`` images
       per class using source-stratified sampling.
    3. Copy selected images to a clean directory with unified naming.
    4. Run multi-source standardisation preprocessing.
    5. Build per-image records with mask and metadata.
    6. Create source-stratified train / val / test splits.
    7. Register all records in the database.
    8. Verify source diversity and the absence of data leakage.
    9. Run the tissue-only preprocessor.
    10. Run tabular data processing.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sqlalchemy import inspect as sa_inspect
from tqdm import tqdm

from src.config.constants import (
    CLASS_CONFIG,
    IMAGE_EXTENSIONS,
    MAYO_FOLDER_PATTERNS,
    NORMAL_KEYWORDS,
    NUM_CLASSES,
    POLYP_KEYWORDS,
    REVERSE_DROP_FEATURES,
    SOURCE_MAP,
    TABULAR_COLON_ACTIVITY_MAP,
    TABULAR_COLON_BINARY_MAP,
    TABULAR_COLON_COLUMNS_TO_DROP,
    TABULAR_COLON_DIET_RISK_MAP,
    TABULAR_COLON_FEATURES,
    TABULAR_COLON_TARGET,
    detect_source_from_stem,
)
from src.config.logger import log as logger
from src.config.paths import paths
from src.data.processing.reverse_logic_processor import ReverseLogicDataAnalyzer
from src.data.processing.standardizer import process_dataset as preprocess
from src.data.processing.tissue_only_preprocessor import process_dataset_tissue_only
from src.database.connection import engine, get_db
from src.database.models import Base, TrainingImage
from src.database.repositories import TrainingImageRepository


class MultiDatasetOrganizer:
    """
    Orchestrates the collection, balancing, preprocessing, and database
    registration of images from four colonoscopy datasets:
    HyperKvasir, CVC-ClinicDB, LIMUC, and Curated Colon.

    By mixing images from multiple sources within each class the model
    cannot exploit frame or border artifacts that are unique to a single
    acquisition system, reducing the risk of learning non-tissue shortcuts.

    Pipeline:
        1. Collect images from all available raw sources.
        2. Log and optionally subsample to ``target_per_class`` images
           per class using source-stratified sampling.
        3. Copy selected images to a clean directory with unified naming.
        4. Run multi-source standardisation preprocessing.
        5. Build per-image records with mask and metadata.
        6. Create source-stratified train / val / test splits.
        7. Register all records in the database.
        8. Verify source diversity and the absence of data leakage.
        9. Run the tissue-only preprocessor.
    """

    def __init__(self, target_per_class: int = 2000, seed: int = 42):
        """
        Initialise the organizer with dataset paths and sampling parameters.

        Args:
            target_per_class (int): Maximum number of images to retain per
                class after source-stratified subsampling. If fewer images
                are available for a class all of them are kept and a warning
                is logged. Default is 2000.
            seed (int): Random seed used for reproducible shuffling and
                stratified splitting. Default is 42.
        """
        self.target_per_class = target_per_class
        self.seed = seed
        random.seed(seed)

        self.hk_raw = paths.HYPERKVASIR_RAW
        self.cvc_raw = paths.RAW / "cvc_clinicdb"
        self.limuc_raw = paths.RAW / "limuc"
        self.curated_raw = paths.RAW / "curated_colon"

        self.clean_dir = paths.COLON_CLEAN
        self.processed_dir = paths.COLON_PROCESSED

        self.collected: dict[str, list[dict]] = defaultdict(list)
        self.image_records: list[dict] = []

    def run(self):
        """
        Execute the complete multi-source dataset organization pipeline.

        Removes and recreates both the clean and processed output
        directories, then runs all nine pipeline phases in sequence.
        Logs progress at each phase boundary.
        """
        logger.info("\n" + "=" * 70)
        logger.info("MULTI-SOURCE DATASET ORGANIZER (4 DATASETS)")
        logger.info("  Breaking shortcuts from frame/border artifacts")
        logger.info("  Sources: HyperKvasir + CVC-ClinicDB + LIMUC + Curated Colon")
        logger.info(f"  Target: ~{self.target_per_class} images/class")
        logger.info("=" * 70)

        for d in [self.clean_dir, self.processed_dir]:
            if d.exists():
                shutil.rmtree(d)
                logger.info(f"  🗑️  {d.name} removed")

        self._collect_all_sources()
        self._log_source_distribution()
        self._balance_and_copy()

        logger.info("\n═══ Phase 4: Preprocessing ═══")
        self._save_mapping()
        preprocess(self.clean_dir, self.processed_dir)

        self._build_processed_records()
        self._create_source_stratified_splits()
        self._register_in_database()
        self._verify_source_mixing()
        self._start_tissue_preprocessor()
        self._process_tabular_dataset()

        logger.info("\n✅ MULTI-SOURCE ORGANIZATION COMPLETED")

    # ═══════════════════════════════════════════════════════════
    #  TABULAR DATA PROCESSING METHODS
    # ═══════════════════════════════════════════════════════════

    def _process_tabular_dataset(self):
        """
        Pipeline to process the tabular dataset.
        Process data from multiple sources.
        Output 2 csv files for Tabular model training.
        """
        logger.info("\n═══ Phase 10: Tabular Data Processing ═══")
        input_path = paths.RAW_TABULAR / "colorectal_cancer_dataset.csv"
        output_path = paths.TABULAR_PROCESSED / "colorectal_cancer_cleaned.csv"
        input_path_alk_smk = (
            paths.RAW_TABULAR_SMOKE / "smoking_driking_dataset_Ver01.csv"
        )
        output_path_alk_smk = paths.TABULAR_PROCESSED / "smoking_drinking_cleaned.csv"

        try:
            df = self._load_colorectal_cancer_csv(input_path)
            df_clean = self._clean_colorectal_dataset(df)
            df_alk_smk = self._load_dataset_smoking_drinking(input_path_alk_smk)
            df_clean_alk_smk = self._clean_smoking_drinking_dataset(df_alk_smk)

            if df_clean is not None:
                is_valid = self._validate_colorectal_data(df_clean)
                if is_valid:
                    self._save_cleaned_dataset(df_clean, output_path)
                else:
                    logger.error(
                        "  ❌ Tabular dataset validation failed. File will not be saved."
                    )
            else:
                logger.error("  ❌ Cleaning process returned None.")

            if df_clean_alk_smk is not None:
                is_valid = self._validate_smoking_drinking_data(df_clean_alk_smk)
                if is_valid:
                    self._save_cleaned_smoking_drinking_dataset(
                        df_clean_alk_smk, output_path_alk_smk
                    )
                else:
                    logger.error(
                        "  ❌ Tabular smoking/drinking dataset validation failed. File will not be saved."
                    )
            else:
                logger.error("  ❌ Cleaning process returned None.")
        except Exception as e:
            logger.error(f"  ❌ Error during tabular processing: {e}")
            import traceback

            logger.debug(traceback.format_exc())

    def _load_colorectal_cancer_csv(self, filepath: Path) -> pd.DataFrame:
        """Load the colorectal cancer dataset from a CSV file.
        Args:
            filepath (Path): Path to the CSV file.
        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info(f"  [LOAD] Loading dataset from: {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"  [LOAD] Records: {len(df):,}")
        logger.info(f"  [LOAD] Columns: {df.shape[1]}")
        return df

    def _clean_colorectal_dataset(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Clean the colorectal cancer dataset.
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame | None: Cleaned DataFrame or None if validation fails.
        """
        logger.info("  [CLEAN] Starting data cleaning...")
        df_clean = df.copy()

        df_clean.columns = df_clean.columns.str.strip()

        target_aliases = [
            "Survival_Prediction",
            "Survival_5_years",
            "Diagnosis",
            "has_cancer",
            "diagnosis",
            "Target",
            "Class",
            "label",
        ]

        target_found = False
        if TABULAR_COLON_TARGET not in df_clean.columns:
            for alias in target_aliases:
                match = [c for c in df_clean.columns if c.lower() == alias.lower()]
                if match:
                    df_clean = df_clean.rename(columns={match[0]: TABULAR_COLON_TARGET})
                    logger.info(
                        f"  [CLEAN] Identified target: '{match[0]}' -> '{TABULAR_COLON_TARGET}'"
                    )
                    target_found = True
                    break
        else:
            target_found = True

        if not target_found:
            logger.error(
                f"  [CLEAN] Target not found. Available: {df_clean.columns.tolist()[:5]}..."
            )
            return None

        cols_to_drop = [
            col
            for col in TABULAR_COLON_COLUMNS_TO_DROP
            if col in df_clean.columns
            and col != TABULAR_COLON_TARGET
            and col != "Country"
        ]
        df_clean = df_clean.drop(columns=cols_to_drop)
        logger.info(f"  [CLEAN] Dropped {len(cols_to_drop)} leakage columns.")

        df_clean = df_clean.drop_duplicates()
        critical_features = [
            col for col in TABULAR_COLON_FEATURES if col in df_clean.columns
        ]
        df_clean = df_clean.dropna(subset=critical_features, how="any")

        logger.info("  [CLEAN] Encoding variables...")

        binary_cols = [
            "Family_History",
            "Smoking_History",
            "Alcohol_Consumption",
            "Diabetes",
            "Inflammatory_Bowel_Disease",
            "Genetic_Mutation",
            TABULAR_COLON_TARGET,
        ]

        for col in binary_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map(TABULAR_COLON_BINARY_MAP)

        if "Diet_Risk" in df_clean.columns:
            df_clean["Diet_Risk"] = df_clean["Diet_Risk"].map(
                TABULAR_COLON_DIET_RISK_MAP
            )
        if "Physical_Activity" in df_clean.columns:
            df_clean["Physical_Activity"] = df_clean["Physical_Activity"].map(
                TABULAR_COLON_ACTIVITY_MAP
            )
        if "Gender" in df_clean.columns:
            df_clean["Gender"] = df_clean["Gender"].map(
                {"M": 0, "F": 1, "Male": 0, "Female": 1}
            )
        if "Urban_or_Rural" in df_clean.columns:
            df_clean["Urban_or_Rural"] = df_clean["Urban_or_Rural"].map(
                {"Urban": 1, "Rural": 0}
            )

        if "Obesity_BMI" in df_clean.columns:
            bmi_dummies = pd.get_dummies(df_clean["Obesity_BMI"], prefix="BMI")
            df_clean = pd.concat([df_clean, bmi_dummies], axis=1).drop(
                columns=["Obesity_BMI"]
            )

        if "Country" in df_clean.columns:
            country_dummies = pd.get_dummies(
                df_clean["Country"], prefix="Country", drop_first=True
            )
            df_clean = pd.concat([df_clean, country_dummies], axis=1).drop(
                columns=["Country"]
            )

        for col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        df_clean = df_clean.fillna(0)

        return df_clean

    def _validate_colorectal_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the colorectal cancer dataset.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            bool: True if validation passes, False otherwise.
        """

        logger.info("  [VALIDATE] Running checks...")

        if TABULAR_COLON_TARGET not in df.columns:
            logger.error(f"  [VALIDATE] ✗ Target '{TABULAR_COLON_TARGET}' missing.")
            return False

        counts = df[TABULAR_COLON_TARGET].value_counts()
        logger.info(f"  [VALIDATE] Class distribution: {counts.to_dict()}")

        expected_missing = ["Obesity_BMI", "Country"]
        missing = [
            c
            for c in TABULAR_COLON_FEATURES
            if c not in df.columns and c not in expected_missing
        ]

        if missing:
            logger.warning(f"  [VALIDATE] Missing original features: {missing}")

        logger.info(
            f"  [VALIDATE] ✓ Dataset ready ({df.shape[0]} rows, {df.shape[1]} cols)"
        )
        return True

    def _save_cleaned_dataset(self, df: pd.DataFrame, filepath: Path):
        """Save the cleaned dataset to a CSV file.
        Args:
            df (pd.DataFrame): Input DataFrame.
            filepath (Path): Output file path.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"  [SAVE] Cleaned dataset saved to: {filepath}")

    def _load_dataset_smoking_drinking(self, filepath: Path) -> pd.DataFrame:
        """Load the alcohol & smoking dataset."""
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info(f"  [LOAD] Loading dataset from: {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"  [LOAD] Records: {len(df):,}")
        logger.info(f"  [LOAD] Columns: {df.shape[1]}")
        return df

    def _clean_smoking_drinking_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the alcohol & smoking dataset ensuring correct types and no missing data.
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        logger.info("  [CLEAN] Starting cleaning of alcohol & smoking dataset...")
        df_clean = df.copy()

        df_clean.columns = df_clean.columns.str.strip()
        df_clean = df_clean.drop_duplicates()
        df_clean = df_clean.dropna()

        # Drop excluded features
        cols_to_drop = [col for col in REVERSE_DROP_FEATURES if col in df_clean.columns]
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
            logger.info(
                f"  [CLEAN] Dropped {len(cols_to_drop)} excluded columns: {cols_to_drop}"
            )

        numeric_cols = [
            "height",
            "weight",
            "waistline",
            "SBP",
            "DBP",
            "BLDS",
            "tot_chole",
            "HDL_chole",
            "LDL_chole",
            "triglyceride",
            "hemoglobin",
            "urine_protein",
            "serum_creatinine",
            "SGOT_AST",
            "SGOT_ALT",
            "gamma_GTP",
            "SMK_stat_type_cd",
        ]

        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        if {"weight", "height"}.issubset(df_clean.columns):
            valid_height = df_clean["height"].replace(0, np.nan)
            df_clean["BMI"] = df_clean["weight"] / ((valid_height / 100.0) ** 2)

        if {"SGOT_AST", "SGOT_ALT"}.issubset(df_clean.columns):
            valid_alt = df_clean["SGOT_ALT"].replace(0, np.nan)
            df_clean["AST_ALT_ratio"] = df_clean["SGOT_AST"] / valid_alt

        if "sex" in df_clean.columns:
            df_clean["sex"] = df_clean["sex"].str.strip().str.capitalize()

        if "DRK_YN" in df_clean.columns:
            df_clean["DRK_YN"] = df_clean["DRK_YN"].str.strip().str.upper()

        if "SMK_stat_type_cd" in df_clean.columns:
            df_clean["SMK_stat_type_cd"] = df_clean["SMK_stat_type_cd"].apply(
                lambda x: "No" if x == 1.0 else "Yes"
            )

        if "DRK_YN" in df_clean.columns:
            df_clean["DRK_YN"] = df_clean["DRK_YN"].map({"Y": "Yes", "N": "No"})

        # ═══════════════════════════════════════════════════════════════════════════
        # ENGINEERED FEATURES
        # ═══════════════════════════════════════════════════════════════════════════

        if {"waistline", "height"}.issubset(df_clean.columns):
            valid_height = df_clean["height"].replace(0, np.nan)
            df_clean["waist_height_ratio"] = df_clean["waistline"] / valid_height

        if {"hemoglobin", "height"}.issubset(df_clean.columns):
            valid_height = df_clean["height"].replace(0, np.nan)
            df_clean["hemoglobin_per_height"] = df_clean["hemoglobin"] / valid_height

        if "gamma_GTP" in df_clean.columns:
            df_clean["gamma_GTP_log"] = np.log1p(df_clean["gamma_GTP"])

        if {"gamma_GTP", "SGOT_AST", "SGOT_ALT"}.issubset(df_clean.columns):
            df_clean["liver_index"] = df_clean["gamma_GTP"] * (
                df_clean["SGOT_AST"] + df_clean["SGOT_ALT"]
            )

        if {"age", "sex"}.issubset(df_clean.columns):
            df_clean["age_sex_interaction"] = df_clean["age"] * (
                df_clean["sex"] == "Male"
            ).astype(int)

        if "BMI" in df_clean.columns:
            df_clean["bmi_category"] = pd.cut(
                df_clean["BMI"],
                bins=[0, 18.5, 25, 30, 100],
                labels=[0, 1, 2, 3],
                include_lowest=True,
            ).astype("int8")

        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna()

        return df_clean

    def _validate_smoking_drinking_data(self, df: pd.DataFrame) -> bool:
        """Validate if the dataset contains all required features and targets.
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            bool: True if all required features and targets are present, False otherwise.
        """
        logger.info("  [VALIDATE] Running checks on alcohol & smoking dataset...")

        from src.config.constants import (
            REVERSE_ANALYSIS_FEATURES,
            REVERSE_ANALYSIS_TARGET_ALCOHOL,
            REVERSE_ANALYSIS_TARGET_SMOKING,
        )

        expected_targets = [
            REVERSE_ANALYSIS_TARGET_SMOKING,
            REVERSE_ANALYSIS_TARGET_ALCOHOL,
        ]

        missing_features = [c for c in REVERSE_ANALYSIS_FEATURES if c not in df.columns]
        missing_targets = [c for c in expected_targets if c not in df.columns]

        if missing_features or missing_targets:
            if missing_features:
                logger.error(f"  [VALIDATE] ❌ Missing features: {missing_features}")
            if missing_targets:
                logger.error(f"  [VALIDATE] ❌ Missing targets: {missing_targets}")
            return False

        logger.info(
            f"  [VALIDATE] ✓ Alcohol & smoking dataset ready ({df.shape[0]} rows, {df.shape[1]} cols)"
        )
        return True

    def _save_cleaned_smoking_drinking_dataset(self, df: pd.DataFrame, filepath: Path):
        """Save the cleaned dataset to disk.
        Args:
            df (pd.DataFrame): Input DataFrame.
            filepath (Path): Output file path.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"  [SAVE] Cleaned dataset saved to: {filepath}")
        analyzer = ReverseLogicDataAnalyzer()
        analyzer.generate_dataset_report(df)

    # ═══════════════════════════════════════════════════════════
    #  IMAGE PROCESSING METHODS
    # ═══════════════════════════════════════════════════════════

    def _start_tissue_preprocessor(self):
        """
        Trigger the tissue-only preprocessing step after the main pipeline.

        Calls ``process_dataset_tissue_only`` with fixed parameters
        (``target_size=384``, ``n_crops=5``, ``max_black_pct=0.5``) and
        logs the outcome.
        """
        logger.info("\nSTARTING TISSUE PREPROCESSOR")
        process_dataset_tissue_only(target_size=384, n_crops=5, max_black_pct=0.5)
        logger.info("✅ TISSUE PREPROCESSOR COMPLETED")

    def _collect_all_sources(self):
        """
        Dispatch collection from all four raw dataset sources.

        Calls ``_collect_hyperkvasir``, ``_collect_cvc_clinicdb``,
        ``_collect_limuc``, and ``_collect_curated_colon`` in order.
        Results are accumulated in ``self.collected``.
        """
        logger.info("\n═══ Phase 1: Collecting from all sources ═══")
        self._collect_hyperkvasir()
        self._collect_cvc_clinicdb()
        self._collect_limuc()
        self._collect_curated_colon()

    def _collect_hyperkvasir(self):
        """
        Scan the HyperKvasir raw directory and collect images per class.

        Looks for a labeled-images subdirectory using several candidate
        path patterns. For polyp images a matching mask is looked up in
        the segmented-images directory when available. Each discovered
        image is appended to ``self.collected[class_name]`` with source
        metadata.

        Logs a warning and returns early if the HyperKvasir root
        directory is not found.
        """
        logger.info("\n  📦 HyperKvasir:")
        labeled_dir = self._find_dir(
            self.hk_raw,
            ["labeled-images", "hyperkvasir_labeled/labeled-images", "labeled_images"],
        )
        if not labeled_dir:
            logger.warning("  ⚠️  HyperKvasir not found, skipping...")
            return

        segmented_dir = self._find_dir(
            self.hk_raw,
            ["segmented-images", "hyperkvasir_segmented/segmented-images"],
        )

        for class_name, sources in SOURCE_MAP["hyperkvasir"].items():
            count = 0
            for source_path in sources:
                src_dir = labeled_dir / source_path
                if not src_dir.exists():
                    logger.warning(f"    ⚠️  Not found: {source_path}")
                    continue

                images = self._glob_images(src_dir)
                source_short = source_path.split("/")[-1]

                for img_path in images:
                    mask_path = None
                    if class_name == "polyp" and segmented_dir:
                        mask_candidate = segmented_dir / "masks" / img_path.name
                        if mask_candidate.exists():
                            mask_path = str(mask_candidate)

                    self.collected[class_name].append(
                        {
                            "original_path": str(img_path),
                            "mask_path": mask_path,
                            "source": "hyperkvasir",
                            "source_detail": f"hk_{source_short}",
                            "class_name": class_name,
                            "patient_id": "",
                        }
                    )
                    count += 1

            logger.info(f"    {class_name}: {count} images")

    def _collect_cvc_clinicdb(self):
        """
        Scan the CVC-ClinicDB raw directory and collect polyp images.

        Searches for the image directory using several candidate path
        patterns. Attempts to locate a paired ground-truth mask for each
        image by trying common extensions. All images are assigned to the
        ``"polyp"`` class.

        Logs a warning and returns early if the CVC-ClinicDB root
        directory is not found.
        """
        logger.info("\n  📦 CVC-ClinicDB:")
        img_dir = self._find_dir(
            self.cvc_raw,
            ["Original", "CVC-ClinicDB/Original", "PNG/Original", "."],
        )
        if not img_dir:
            logger.warning("  ⚠️  CVC-ClinicDB not found, skipping...")
            return

        mask_dir = self._find_dir(
            self.cvc_raw,
            ["Ground Truth", "CVC-ClinicDB/Ground Truth", "PNG/Ground Truth", "masks"],
        )

        images = self._glob_images(img_dir)
        count = 0

        for img_path in images:
            mask_path = None
            if mask_dir:
                for ext in [".png", ".jpg", ".tif", ".bmp"]:
                    candidate = mask_dir / (img_path.stem + ext)
                    if candidate.exists():
                        mask_path = str(candidate)
                        break

            self.collected["polyp"].append(
                {
                    "original_path": str(img_path),
                    "mask_path": mask_path,
                    "source": "cvc_clinicdb",
                    "source_detail": "cvc_polyp",
                    "class_name": "polyp",
                    "patient_id": "",
                }
            )
            count += 1

        logger.info(
            f"    polyp: {count} images" + (" (with masks)" if mask_dir else "")
        )

    def _collect_limuc(self):
        """
        Scan the LIMUC dataset and collect images grouped by Mayo score.

        Discovers patient directories using ``_find_limuc_patient_dirs``
        and maps each Mayo score to a class name as defined in
        ``SOURCE_MAP["limuc"]``. Images are appended with a
        ``patient_id`` field to support patient-level splitting if
        needed.

        Logs a warning and returns early if the LIMUC root directory or
        patient folders are not found.
        """
        logger.info("\n  📦 LIMUC:")
        if not self.limuc_raw.exists():
            logger.warning("  ⚠️  LIMUC not found, skipping...")
            return

        patient_dirs = self._find_limuc_patient_dirs()
        if not patient_dirs:
            logger.warning("  ⚠️  No LIMUC patient folders found")
            return

        logger.info(f"    Patients found: {len(patient_dirs)}")
        mayo_totals = {"0": 0, "1": 0, "2": 0, "3": 0}

        for class_name, mayo_scores in SOURCE_MAP["limuc"].items():
            count = 0
            for score in mayo_scores:
                score_count = 0
                for patient_dir in patient_dirs:
                    score_dir = self._find_mayo_folder(patient_dir, score)
                    if not score_dir:
                        continue
                    images = self._glob_images(score_dir)
                    for img_path in images:
                        patient_id = patient_dir.name
                        self.collected[class_name].append(
                            {
                                "original_path": str(img_path),
                                "mask_path": None,
                                "source": "limuc",
                                "source_detail": f"limuc_mayo{score}",
                                "class_name": class_name,
                                "patient_id": f"limuc_p{patient_id}",
                            }
                        )
                        score_count += 1
                    mayo_totals[score] = mayo_totals.get(score, 0) + len(images)
                count += score_count
                if score_count > 0:
                    logger.info(
                        f"    Mayo {score} → {class_name}: {score_count} images"
                    )
            logger.info(f"    ✅ {class_name}: {count} total from LIMUC")

    def _collect_curated_colon(self):
        """
        Scan the Curated Colon raw directory and classify images by folder name.

        Recursively searches all subdirectories. Each directory whose
        name contains a keyword from ``POLYP_KEYWORDS`` is assigned to
        the ``"polyp"`` class; directories matching ``NORMAL_KEYWORDS``
        are assigned to ``"normal"``. Directories matching neither
        keyword set are skipped.

        Logs a warning and returns early if the Curated Colon root
        directory does not exist or no classifiable folders are found.
        """
        logger.info("\n  📦 Curated Colon:")
        if not self.curated_raw.exists():
            logger.warning("  ⚠️  Curated colon not found, skipping...")
            return

        folder_mapping = {}
        for subdir in sorted(self.curated_raw.rglob("*")):
            if not subdir.is_dir():
                continue
            images = self._glob_images(subdir)
            if not images:
                continue
            dir_name = subdir.name.lower()
            is_polyp = any(kw in dir_name for kw in POLYP_KEYWORDS)
            is_normal = any(kw in dir_name for kw in NORMAL_KEYWORDS)
            if is_polyp:
                folder_mapping[subdir] = "polyp"
            elif is_normal:
                folder_mapping[subdir] = "normal"

        if not folder_mapping:
            logger.warning("  ⚠️  No classifiable folders found")
            return

        counts = defaultdict(int)
        for folder, class_name in folder_mapping.items():
            images = self._glob_images(folder)
            folder_short = folder.name
            for img_path in images:
                self.collected[class_name].append(
                    {
                        "original_path": str(img_path),
                        "mask_path": None,
                        "source": "curated_colon",
                        "source_detail": f"curated_{folder_short}",
                        "class_name": class_name,
                        "patient_id": "",
                    }
                )
                counts[class_name] += 1

        for class_name, count in sorted(counts.items()):
            logger.info(f"    {class_name}: {count} images")

    def _log_source_distribution(self):
        """
        Log the total image count and per-source breakdown for each class.

        Also emits a warning if any class has images from only a single
        source, as this indicates insufficient source diversity to break
        shortcut features.
        """
        logger.info("\n═══ Phase 2: Source Distribution ═══")
        for class_name in CLASS_CONFIG:
            records = self.collected[class_name]
            total = len(records)
            by_source = Counter(r["source"] for r in records)
            detail = ", ".join(f"{s}={c}" for s, c in sorted(by_source.items()))
            logger.info(f"  {class_name:15s}: {total:5d} total ({detail})")
            if len(by_source) < 2:
                logger.warning(f"    ⚠️  Only 1 source for {class_name}!")
        grand_total = sum(len(self.collected[c]) for c in CLASS_CONFIG)
        logger.info(f"\n  📊 GRAND TOTAL: {grand_total} images available")

    def _balance_and_copy(self):
        """
        Subsample each class to ``target_per_class`` and copy to the clean directory.

        For classes with more images than the target, a source-stratified
        subsample is drawn via ``_stratified_subsample``. For classes
        with fewer images a warning is logged but all available images
        are used.

        Each image is renamed to
        ``<source_detail>[_<patient_id>]_<original_filename>`` to ensure
        globally unique filenames. When a mask is present it is copied to
        ``clean_dir/masks/`` with the same stem.
        """
        logger.info(f"\n═══ Phase 3: Balancing to {self.target_per_class}/class ═══")
        for class_name in CLASS_CONFIG:
            records = self.collected[class_name]
            total = len(records)
            if total == 0:
                logger.error(f"  ❌ No images for {class_name}!")
                continue
            if total > self.target_per_class:
                records = self._stratified_subsample(
                    records, self.target_per_class, key="source"
                )
                logger.info(f"  {class_name}: {total} → {len(records)} (subsampled)")
            elif total < self.target_per_class:
                logger.warning(
                    f"  {class_name}: only {total} available "
                    f"(target: {self.target_per_class})"
                )

            class_dir = self.clean_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for r in tqdm(records, desc=f"  Copying {class_name}"):
                src = Path(r["original_path"])
                if not src.exists():
                    continue
                patient_id = r.get("patient_id", "")
                if patient_id:
                    new_name = f"{r['source_detail']}_{patient_id}_{src.name}"
                else:
                    new_name = f"{r['source_detail']}_{src.name}"
                dest = class_dir / new_name
                shutil.copy2(src, dest)

                if r.get("mask_path"):
                    mask_src = Path(r["mask_path"])
                    if mask_src.exists():
                        mask_dir = self.clean_dir / "masks"
                        mask_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(mask_src, mask_dir / new_name)

            by_source = Counter(r["source"] for r in records)
            detail = ", ".join(f"{s}={c}" for s, c in sorted(by_source.items()))
            logger.info(f"  ✅ {class_name}: {len(records)} ({detail})")

    def _stratified_subsample(self, records, target, key):
        """
        Draw a source-stratified subsample of ``target`` records.

        Each source group receives a number of samples proportional to
        its share of the total. The last group absorbs any rounding
        remainder to guarantee exactly ``target`` records are returned
        when enough data is available.

        Args:
            records (list[dict]): Full list of image record dictionaries,
                each containing at least the field named by ``key``.
            target (int): Total number of records to select.
            key (str): Record field used to define strata (e.g.
                ``"source"``).

        Returns:
            list[dict]: Sampled records totalling at most ``target``
                entries, preserving source proportions.
        """
        by_group = defaultdict(list)
        for r in records:
            by_group[r[key]].append(r)
        total = len(records)
        selected = []
        remaining = target
        groups = sorted(by_group.keys())
        for i, group in enumerate(groups):
            group_records = by_group[group]
            proportion = len(group_records) / total
            if i == len(groups) - 1:
                n = remaining
            else:
                n = max(1, round(proportion * target))
                n = min(n, remaining, len(group_records))
            sampled = random.sample(group_records, min(n, len(group_records)))
            selected.extend(sampled)
            remaining -= len(sampled)
        return selected

    def _build_processed_records(self):
        """
        Scan the processed directory and build per-image metadata records.

        For each image found in the processed class subdirectories, the
        method resolves the source identifier from the filename stem,
        looks up any aligned mask in both the processed and clean mask
        directories, reads the image dimensions via PIL, and appends a
        complete record dictionary to ``self.image_records``.

        Logs the number of images and masks found per class, as well as
        totals across the full dataset.
        """
        logger.info("\n═══ Phase 5: Building Records ═══")
        self.image_records = []

        mask_dir = self.clean_dir / "masks"
        available_masks: dict[str, Path] = {}
        if mask_dir.exists():
            for mask_file in mask_dir.iterdir():
                if mask_file.is_file():
                    available_masks[mask_file.stem] = mask_file

        proc_mask_dir = self.processed_dir / "masks"
        processed_masks: dict[str, Path] = {}
        if proc_mask_dir.exists():
            for mask_file in proc_mask_dir.iterdir():
                if mask_file.is_file():
                    processed_masks[mask_file.stem] = mask_file

        for class_name, config in CLASS_CONFIG.items():
            proc_dir = self.processed_dir / class_name
            if not proc_dir.exists():
                logger.warning(f"  ⚠️  Not found: {proc_dir}")
                continue

            images = sorted(self._glob_images(proc_dir))
            masks_found = 0

            for img_path in images:
                stem = img_path.stem
                source = detect_source_from_stem(stem)

                mask_path_str = None
                if stem in processed_masks:
                    mask_path_str = str(processed_masks[stem])
                elif stem in available_masks:
                    mask_path_str = str(available_masks[stem])

                if mask_path_str:
                    masks_found += 1

                try:
                    img = Image.open(img_path)
                    w, h = img.size
                except Exception:
                    w, h = 384, 384

                self.image_records.append(
                    {
                        "file_path": str(img_path),
                        "mask_path": mask_path_str,
                        "label": config["label"],
                        "class_name": class_name,
                        "source": source,
                        "dataset_source": source,
                        "width": w,
                        "height": h,
                    }
                )

            logger.info(
                f"  ✅ {class_name}: {len(images)} processed"
                + (f" ({masks_found} with mask)" if masks_found > 0 else "")
            )

        total_masks = sum(1 for r in self.image_records if r.get("mask_path"))
        logger.info(f"  📊 Total records: {len(self.image_records)}")
        logger.info(f"  🎭 Total with mask: {total_masks}")

    def _create_source_stratified_splits(self):
        """
        Assign each record to train, val, or test using composite stratification.

        Composite labels are formed as ``<class_name>_<source>`` so that
        the class and source distribution is preserved across all three
        splits. If any composite group has fewer than four samples the
        method falls back to class-only stratification.

        Split proportions: 70% train / 15% val / 15% test.
        The ``"split"`` key is written directly into each record in
        ``self.image_records``.
        """
        logger.info("\n═══ Phase 6: Stratified Split (class + source) ═══")

        composite_labels = [
            f"{r['class_name']}_{r['source']}" for r in self.image_records
        ]
        counts = Counter(composite_labels)
        min_group_size = min(counts.values())
        logger.info(f"  Composite groups: {len(counts)}")
        logger.info(f"  Smallest group: {min_group_size}")

        for label, cnt in sorted(counts.items()):
            logger.info(f"    {label}: {cnt}")

        if min_group_size < 4:
            logger.warning(
                "  ⚠️  Group(s) too small, fallback to class-only stratification"
            )
            composite_labels = [r["class_name"] for r in self.image_records]

        indices = list(range(len(self.image_records)))
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=0.30,
            stratify=composite_labels,
            random_state=self.seed,
        )
        temp_labels = [composite_labels[i] for i in temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            stratify=temp_labels,
            random_state=self.seed,
        )

        for i in train_idx:
            self.image_records[i]["split"] = "train"
        for i in val_idx:
            self.image_records[i]["split"] = "val"
        for i in test_idx:
            self.image_records[i]["split"] = "test"

        for split in ["train", "val", "test"]:
            records = [r for r in self.image_records if r["split"] == split]
            by_class = Counter(r["class_name"] for r in records)
            by_source = Counter(r["source"] for r in records)
            logger.info(f"  {split.upper():5s}: {len(records):4d}")
            logger.info(
                f"         classes:  "
                f"{', '.join(f'{k}={v}' for k, v in sorted(by_class.items()))}"
            )
            logger.info(
                f"         sources:  "
                f"{', '.join(f'{k}={v}' for k, v in sorted(by_source.items()))}"
            )

    def setup_database(self):
        """
        Create all database tables defined in ``Base.metadata``.

        Should be called once before attempting to insert records if the
        tables do not yet exist. Safe to call multiple times; existing
        tables are not modified.
        """
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully.")

    def _register_in_database(self):
        """
        Persist all processed image records to the database.

        Checks whether the ``TrainingImage`` table exists and creates it
        if not. Deletes any existing rows before inserting the new
        records via ``TrainingImageRepository.bulk_insert`` to ensure
        idempotent runs.
        """
        logger.info("\n═══ Phase 7: Registering in DB ═══")

        inspector = sa_inspect(engine)
        if not inspector.has_table(TrainingImage.__tablename__):
            self.setup_database()

        with get_db() as db:
            db.query(TrainingImage).delete()
            db.commit()
            db_records = [
                {
                    "file_path": r["file_path"],
                    "mask_path": r.get("mask_path"),
                    "label": r["label"],
                    "dataset_source": r["dataset_source"],
                    "split": r["split"],
                    "width": r.get("width"),
                    "height": r.get("height"),
                }
                for r in self.image_records
            ]
            repo = TrainingImageRepository(db)
            count = repo.bulk_insert(db_records)
            logger.info(f"  ✅ {count} images registered (multi-source 4 datasets)")

    def _verify_source_mixing(self):
        """
        Verify source diversity and the absence of data leakage across splits.

        For each (split, class) combination, logs the number of images
        and distinct sources. A warning is emitted if any combination
        has images from only one source.

        Subsequently checks that no image filename appears in more than
        one split. Logs an error if overlaps are detected.
        """
        logger.info("\n═══ Phase 8: Anti-Leakage Verification ═══")
        issues = []
        for split in ["train", "val", "test"]:
            for class_name in CLASS_CONFIG:
                records = [
                    r
                    for r in self.image_records
                    if r["split"] == split and r["class_name"] == class_name
                ]
                sources = {r["source"] for r in records}
                n_sources = len(sources)
                status = "✅" if n_sources >= 2 else "⚠️"
                logger.info(
                    f"  {status} {split}/{class_name}: {len(records)} imgs, "
                    f"{n_sources} source(s): {sources}"
                )
                if n_sources < 2:
                    issues.append(f"{split}/{class_name}")

        if issues:
            logger.warning(
                f"\n  ⚠️  {len(issues)} group(s) with single source: {issues}"
            )
        else:
            logger.info("\n  ✅ All classes in all splits have ≥2 sources")

        names = {}
        for split in ["train", "val", "test"]:
            names[split] = {
                Path(r["file_path"]).name
                for r in self.image_records
                if r["split"] == split
            }
        overlaps = (
            (names["train"] & names["val"])
            | (names["train"] & names["test"])
            | (names["val"] & names["test"])
        )
        if overlaps:
            logger.error(f"  ❌ DATA LEAKAGE: {len(overlaps)} shared images")
        else:
            logger.info("  ✅ No image overlap between splits")

    def _find_limuc_patient_dirs(self):
        """
        Locate LIMUC patient directories by scanning the raw root.

        Searches the LIMUC root and known subdirectory patterns for
        directories whose names are purely numeric and that contain at
        least one Mayo score folder with images. Falls back to a full
        recursive scan if the primary search yields no results.

        Returns:
            list[Path]: Sorted list of patient directory paths, ordered
                by numeric patient ID.
        """
        patient_dirs = []
        search_roots = [
            self.limuc_raw,
            self.limuc_raw / "LIMUC",
            self.limuc_raw / "limuc",
        ]
        for root in search_roots:
            if not root.exists():
                continue
            for item in sorted(root.iterdir()):
                if (
                    item.is_dir()
                    and item.name.isdigit()
                    and self._has_mayo_folders(item)
                ):
                    patient_dirs.append(item)
            if patient_dirs:
                break
        if not patient_dirs:
            for dirpath in sorted(self.limuc_raw.rglob("*")):
                if (
                    dirpath.is_dir()
                    and dirpath.name.isdigit()
                    and self._has_mayo_folders(dirpath)
                ):
                    patient_dirs.append(dirpath)
        return sorted(patient_dirs, key=lambda p: int(p.name))

    def _has_mayo_folders(self, base):
        """
        Check whether a directory contains at least one non-empty Mayo score folder.

        Tests all combinations of Mayo scores (0–3) and folder name
        patterns from ``MAYO_FOLDER_PATTERNS``.

        Args:
            base (Path): Directory to inspect.

        Returns:
            bool: ``True`` if at least one Mayo score folder is found
                and contains at least one image file, ``False`` otherwise.
        """
        for score in ["0", "1", "2", "3"]:
            for pattern in MAYO_FOLDER_PATTERNS:
                candidate = base / pattern.format(score=score)
                if candidate.is_dir() and self._glob_images(candidate):
                    return True
        return False

    def _find_mayo_folder(self, base, score):
        """
        Locate the Mayo score subfolder for a given patient directory.

        Tries each pattern in ``MAYO_FOLDER_PATTERNS`` with the
        provided score substituted in.

        Args:
            base (Path): Patient-level directory to search within.
            score (str): Mayo score string (``"0"``, ``"1"``, ``"2"``,
                or ``"3"``).

        Returns:
            Path | None: Path to the first matching directory, or
                ``None`` if no pattern matches.
        """
        for pattern in MAYO_FOLDER_PATTERNS:
            d = base / pattern.format(score=score)
            if d.is_dir():
                return d
        return None

    def _find_dir(self, base, candidates):
        """
        Return the first existing subdirectory from a list of candidates.

        Args:
            base (Path): Root directory to search within.
            candidates (list[str]): Relative path strings to try in
                order. The special value ``"."`` refers to ``base``
                itself.

        Returns:
            Path | None: The first candidate that exists as a directory,
                or ``None`` if none are found or ``base`` does not exist.
        """
        if not base.exists():
            return None
        for c in candidates:
            d = base / c if c != "." else base
            if d.exists() and d.is_dir():
                return d
        return None

    def _glob_images(self, directory):
        """
        Return all image files in a directory matching known extensions.

        Args:
            directory (Path): Directory to scan.

        Returns:
            list[Path]: Sorted list of image file paths whose suffix
                appears in ``IMAGE_EXTENSIONS``.
        """
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(directory.glob(f"*{ext}"))
        return sorted(images)

    def _save_mapping(self):
        """
        Write the class mapping JSON to both the clean and processed directories.

        The mapping includes project metadata, the number of classes, a
        label-to-class-name dictionary, per-class clinical information,
        and the list of dataset sources. The file is saved as
        ``class_mapping.json`` in both directories; parent directories
        are created if they do not exist.
        """
        logger.info("\n═══ Phase 9: Saving Class Mapping ═══")
        mapping = {
            "project": "Colon Cancer Detection (Multi-Source 4 datasets)",
            "num_classes": NUM_CLASSES,
            "classes": {},
            "clinical_info": {},
            "sources": list(SOURCE_MAP.keys()),
        }
        for class_name, config in CLASS_CONFIG.items():
            label = str(config["label"])
            mapping["classes"][label] = class_name
            mapping["clinical_info"][label] = {
                "clinical_name": config["clinical_name"],
                "cancer_role": config["cancer_role"],
                "risk_level": config["risk_level"],
                "action": config["action"],
            }
        for base_dir in [self.clean_dir, self.processed_dir]:
            base_dir.mkdir(parents=True, exist_ok=True)
            out = base_dir / "class_mapping.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(mapping, f, indent=2, ensure_ascii=False)
            logger.info(f"  📄 {out}")


if __name__ == "__main__":
    organizer = MultiDatasetOrganizer(target_per_class=2000, seed=42)
    organizer.run()
