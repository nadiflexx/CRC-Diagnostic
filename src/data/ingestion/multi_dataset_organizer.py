"""
Multi-source organizer: Mixes HyperKvasir + CVC-ClinicDB + LIMUC + Curated Colon.
Breaks data leakage from frame/border artifacts by diversifying sources per class.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import shutil

import pandas as pd
import numpy as np
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
    SOURCE_MAP,
    TABULAR_COLON_FEATURES,
    TABULAR_COLON_TARGET,
    TABULAR_COLON_COLUMNS_TO_DROP,
    TABULAR_COLON_BINARY_MAP,
    TABULAR_COLON_DIET_RISK_MAP,
    TABULAR_COLON_ACTIVITY_MAP,
    TABULAR_COLON_BMI_MAP,
    detect_source_from_stem,
)
from src.config.logger import log as logger
from src.config.paths import paths
from src.data.processing.standardizer import process_dataset as preprocess
from src.data.processing.tissue_only_preprocessor import process_dataset_tissue_only
from src.data.processing.reverse_logic_processor import ReverseLogicDataAnalyzer
from src.database.connection import engine, get_db
from src.database.models import Base, TrainingImage
from src.database.repositories import TrainingImageRepository


class MultiDatasetOrganizer:
    """
    Organizes and mixes multiple datasets for colon cancer classification.
    By mixing sources per class, frame/border artifacts are distributed
    uniformly and stop being predictive.
    """

    def __init__(self, target_per_class: int = 2000, seed: int = 42):
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
        Run the multi-source dataset organization process.
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
        Pipeline completo para cargar, limpiar, validar y guardar datos tabulares.
        """
        logger.info("\n═══ Phase 10: Tabular Data Processing ═══")
        input_path = paths.RAW_TABULAR / "colorectal_cancer_dataset.csv"
        output_path = paths.PROCESSED_TABULAR / "colorectal_cancer_cleaned.csv"
        input_path_alk_smk = paths.RAW_TABULAR / "smoking_drinking_dataset_Ver01.csv"
        output_path_alk_smk = paths.PROCESSED_TABULAR / "smoking_drinking_cleaned.csv"
        
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
                    logger.error("  ❌ Tabular dataset validation failed. File will not be saved.")
            else:
                logger.error("  ❌ Cleaning process returned None.")
            
            if df_clean_alk_smk is not None:
                is_valid = self._validate_smoking_drinking_data(df_clean_alk_smk)
                if is_valid:
                    self._save_cleaned_smoking_drinking_dataset(df_clean_alk_smk, output_path_alk_smk)
                else:
                    logger.error("  ❌ Tabular smoking/drinking dataset validation failed. File will not be saved.")
            else:
                logger.error("  ❌ Cleaning process returned None.")
        except Exception as e:
            logger.error(f"  ❌ Error during tabular processing: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _load_colorectal_cancer_csv(self, filepath: Path) -> pd.DataFrame:
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"  [LOAD] Loading dataset from: {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"  [LOAD] Records: {len(df):,}")
        logger.info(f"  [LOAD] Columns: {df.shape[1]}")
        return df

    def _clean_colorectal_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("  [CLEAN] Starting data cleaning...")
        df_clean = df.copy()
        
        # 0: Estandarizar nombres de columnas (eliminar espacios y normalizar)
        df_clean.columns = df_clean.columns.str.strip()
        
        # 1: IDENTIFICAR Y RENOMBRAR TARGET (Antes de descartar columnas)
        # Añadimos Survival_Prediction y Survival_5_years como alias comunes
        target_aliases = [
            'Survival_Prediction', 'Survival_5_years', 'Diagnosis', 
            'has_cancer', 'diagnosis', 'Target', 'Class', 'label'
        ]
        
        target_found = False
        if TABULAR_COLON_TARGET not in df_clean.columns:
            for alias in target_aliases:
                match = [c for c in df_clean.columns if c.lower() == alias.lower()]
                if match:
                    df_clean = df_clean.rename(columns={match[0]: TABULAR_COLON_TARGET})
                    logger.info(f"  [CLEAN] Identified target: '{match[0]}' -> '{TABULAR_COLON_TARGET}'")
                    target_found = True
                    break
        else:
            target_found = True

        if not target_found:
            logger.error(f"  [CLEAN] Target not found. Available: {df_clean.columns.tolist()[:5]}...")
            return None
        
        # 2: Drop leakage columns (Asegurándonos de no borrar el target ya renombrado)
        # Filtramos la lista de drop para que no elimine el nuevo nombre del target ni el país si es feature
        cols_to_drop = [
            col for col in TABULAR_COLON_COLUMNS_TO_DROP 
            if col in df_clean.columns and col != TABULAR_COLON_TARGET and col != 'Country'
        ]
        df_clean = df_clean.drop(columns=cols_to_drop)
        logger.info(f"  [CLEAN] Dropped {len(cols_to_drop)} leakage columns.")
        
        # 3: Manejo de duplicados y nulos
        df_clean = df_clean.drop_duplicates()
        critical_features = [col for col in TABULAR_COLON_FEATURES if col in df_clean.columns]
        df_clean = df_clean.dropna(subset=critical_features, how='any')
        
        # 4: Codificación de variables (Mapeos desde constants.py)
        logger.info("  [CLEAN] Encoding variables...")
        
        # Mapeos Binarios (Yes/No -> 1/0)
        binary_cols = [
            'Family_History', 'Smoking_History', 'Alcohol_Consumption', 
            'Diabetes', 'Inflammatory_Bowel_Disease', 'Genetic_Mutation',
            TABULAR_COLON_TARGET  # Survival_Prediction también es Yes/No
        ]
        
        for col in binary_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map(TABULAR_COLON_BINARY_MAP)
        
        # Mapeos categóricos ordinales
        if 'Diet_Risk' in df_clean.columns:
            df_clean['Diet_Risk'] = df_clean['Diet_Risk'].map(TABULAR_COLON_DIET_RISK_MAP)
        if 'Physical_Activity' in df_clean.columns:
            df_clean['Physical_Activity'] = df_clean['Physical_Activity'].map(TABULAR_COLON_ACTIVITY_MAP)
        if 'Gender' in df_clean.columns:
            df_clean['Gender'] = df_clean['Gender'].map({'M': 0, 'F': 1, 'Male': 0, 'Female': 1})
        if 'Urban_or_Rural' in df_clean.columns:
            df_clean['Urban_or_Rural'] = df_clean['Urban_or_Rural'].map({'Urban': 1, 'Rural': 0})
            
        # BMI (One-hot encoding)
        if 'Obesity_BMI' in df_clean.columns:
            bmi_dummies = pd.get_dummies(df_clean['Obesity_BMI'], prefix='BMI')
            df_clean = pd.concat([df_clean, bmi_dummies], axis=1).drop(columns=['Obesity_BMI'])
            
        # Country (One-hot encoding - Solo si se mantuvo)
        if 'Country' in df_clean.columns:
            country_dummies = pd.get_dummies(df_clean['Country'], prefix='Country', drop_first=True)
            df_clean = pd.concat([df_clean, country_dummies], axis=1).drop(columns=['Country'])
        
        # 5: Asegurar tipos numéricos
        for col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Rellenar nulos finales con 0 o promedios si quedaron tras el coerce
        df_clean = df_clean.fillna(0)
        
        return df_clean

    def _validate_colorectal_data(self, df: pd.DataFrame) -> bool:
        logger.info("  [VALIDATE] Running checks...")
        
        # 1. Verificar Target
        if TABULAR_COLON_TARGET not in df.columns:
            logger.error(f"  [VALIDATE] ✗ Target '{TABULAR_COLON_TARGET}' missing.")
            return False
            
        # 2. Verificar balance de clases
        counts = df[TABULAR_COLON_TARGET].value_counts()
        logger.info(f"  [VALIDATE] Class distribution: {counts.to_dict()}")
        
        # 3. Verificar Features (considerando las que cambiaron de nombre)
        # Country y Obesity_BMI cambian de nombre por el get_dummies
        expected_missing = ['Obesity_BMI', 'Country']
        missing = [c for c in TABULAR_COLON_FEATURES if c not in df.columns and c not in expected_missing]
        
        if missing:
            logger.warning(f"  [VALIDATE] Missing original features: {missing}")
        
        logger.info(f"  [VALIDATE] ✓ Dataset ready ({df.shape[0]} rows, {df.shape[1]} cols)")
        return True

    def _save_cleaned_dataset(self, df: pd.DataFrame, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"  [SAVE] Cleaned dataset saved to: {filepath}")
    
    # Alcohol & Smoking dataset cleaning data and analysis data values
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
        """
        logger.info("  [CLEAN] Starting cleaning of alcohol & smoking dataset...")
        df_clean = df.copy()
        
        df_clean.columns = df_clean.columns.str.strip()
        df_clean = df_clean.drop_duplicates()
        df_clean = df_clean.dropna()
        numeric_cols = [
            "height", "weight", "waistline", "SBP", "DBP", "BLDS", "tot_chole", "HDL_chole",
            "LDL_chole", "triglyceride", "hemoglobin", "urine_protein", "serum_creatinine",
            "SGOT_AST", "SGOT_ALT", "gamma_GTP", "SMK_stat_type_cd"
        ]
        
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # Derived features to enrich reverse-logic models
        if {"weight", "height"}.issubset(df_clean.columns):
            valid_height = df_clean["height"].replace(0, np.nan)
            df_clean["BMI"] = df_clean["weight"] / ((valid_height / 100.0) ** 2)

        if {"SGOT_AST", "SGOT_ALT"}.issubset(df_clean.columns):
            valid_alt = df_clean["SGOT_ALT"].replace(0, np.nan)
            df_clean["AST_ALT_ratio"] = df_clean["SGOT_AST"] / valid_alt
            
        if 'sex' in df_clean.columns:
            df_clean['sex'] = df_clean['sex'].str.strip().str.capitalize()
            
        if 'DRK_YN' in df_clean.columns:
            df_clean['DRK_YN'] = df_clean['DRK_YN'].str.strip().str.upper()

        if 'SMK_stat_type_cd' in df_clean.columns:
            df_clean['SMK_stat_type_cd'] = df_clean['SMK_stat_type_cd'].apply(
                lambda x: "No" if x == 1.0 else "Yes"
            )

        if 'DRK_YN' in df_clean.columns:
            df_clean['DRK_YN'] = df_clean['DRK_YN'].map({"Y": "Yes", "N": "No"})

        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna()
        
        return df_clean

    def _validate_smoking_drinking_data(self, df: pd.DataFrame) -> bool:
        """Validate if the dataset contains all required features and targets."""
        logger.info("  [VALIDATE] Running checks on alcohol & smoking dataset...")
        
        from src.config.constants import (
            REVERSE_ANALYSIS_FEATURES,
            REVERSE_ANALYSIS_TARGET_SMOKING,
            REVERSE_ANALYSIS_TARGET_ALCOHOL
        )
        
        expected_targets = [REVERSE_ANALYSIS_TARGET_SMOKING, REVERSE_ANALYSIS_TARGET_ALCOHOL]
        
        missing_features = [c for c in REVERSE_ANALYSIS_FEATURES if c not in df.columns]
        missing_targets = [c for c in expected_targets if c not in df.columns]
        
        if missing_features or missing_targets:
            if missing_features:
                logger.error(f"  [VALIDATE] ❌ Missing features: {missing_features}")
            if missing_targets:
                logger.error(f"  [VALIDATE] ❌ Missing targets: {missing_targets}")
            return False
            
        logger.info(f"  [VALIDATE] ✓ Alcohol & smoking dataset ready ({df.shape[0]} rows, {df.shape[1]} cols)")
        return True

    def _save_cleaned_smoking_drinking_dataset(self, df: pd.DataFrame, filepath: Path):
        """Save the cleaned dataset to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"  [SAVE] Cleaned dataset saved to: {filepath}")
        analyzer = ReverseLogicDataAnalyzer()
        analyzer.generate_dataset_report(df)

    # ═══════════════════════════════════════════════════════════
    #  IMAGE PROCESSING METHODS (Original implementations)
    # ═══════════════════════════════════════════════════════════

    def _start_tissue_preprocessor(self):
        """
        Start the tissue-only preprocessor.
        """
        logger.info("\nSTARTING TISSUE PREPROCESSOR")
        process_dataset_tissue_only(target_size=384, n_crops=5, max_black_pct=0.5)
        logger.info("✅ TISSUE PREPROCESSOR COMPLETED")

    def _collect_all_sources(self):
        """
        Collect images from all available sources.
        """
        logger.info("\n═══ Phase 1: Collecting from all sources ═══")
        self._collect_hyperkvasir()
        self._collect_cvc_clinicdb()
        self._collect_limuc()
        self._collect_curated_colon()

    def _collect_hyperkvasir(self):
        """
        Collect images from HyperKvasir.
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
        Collect images from CVC-ClinicDB.
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
        Collect images from LIMUC.
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
        Collect images from Curated Colon.
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
        Log the distribution of images across sources.
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
        Balance and copy images to the clean directory.
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
                    f"  {class_name}: only {total} available (target: {self.target_per_class})"
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
        Perform stratified subsampling of records.

        :param records: The list of records to sample from.
        :param target: The target number of records to select.
        :param key: The key to use for stratification.
        :return: A list of sampled records.
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
        Build records for the processed dataset.
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
        Create stratified splits based on source.
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
                f"         classes:  {', '.join(f'{k}={v}' for k, v in sorted(by_class.items()))}"
            )
            logger.info(
                f"         sources:  {', '.join(f'{k}={v}' for k, v in sorted(by_source.items()))}"
            )

    def setup_database(self):
        """
        Setup the database tables.
        """
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully.")

    def _register_in_database(self):
        """
        Register the processed images in the database.
        Includes error handling to prevent the pipeline from crashing.
        """
        logger.info("\n═══ Phase 7: Registering in DB ═══")

        try:
            inspector = sa_inspect(engine)
            if not inspector.has_table(TrainingImage.__tablename__):
                self.setup_database()

            with get_db() as db:
                # Limpiar registros anteriores
                db.query(TrainingImage).delete()
                db.commit()
                
                # Preparar los nuevos registros
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
                
                # Inserción masiva
                repo = TrainingImageRepository(db)
                count = repo.bulk_insert(db_records)
                logger.info(f"  ✅ {count} images registered (multi-source 4 datasets)")
                
        except Exception as e:
            logger.error(f"  ❌ Error registering images in the database: {e}")
            logger.warning("  ⚠️ Continuing pipeline execution without database registration.")

    def _verify_source_mixing(self):
        """
        Verify that each class has images from multiple sources in each split.
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
                    f"  {status} {split}/{class_name}: {len(records)} imgs, {n_sources} source(s): {sources}"
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
        Find LIMUC patient directories.
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
        Check if a base directory has Mayo folders.
        """
        for score in ["0", "1", "2", "3"]:
            for pattern in MAYO_FOLDER_PATTERNS:
                candidate = base / pattern.format(score=score)
                if candidate.is_dir() and self._glob_images(candidate):
                    return True
        return False

    def _find_mayo_folder(self, base, score):
        """
        Find a Mayo folder in a base directory.
        """
        for pattern in MAYO_FOLDER_PATTERNS:
            d = base / pattern.format(score=score)
            if d.is_dir():
                return d
        return None

    def _find_dir(self, base, candidates):
        """
        Find a directory in a base directory.
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
        Glob images in a directory.
        """
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(directory.glob(f"*{ext}"))
        return sorted(images)

    def _save_mapping(self):
        """
        Save the class mapping.
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
