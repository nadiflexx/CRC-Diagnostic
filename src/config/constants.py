"""
Centralized constants, mappings, and domain dictionaries.
Single source of truth for all magic values scattered across the codebase.
"""

from collections import OrderedDict

import numpy as np

from src.config.paths import paths

# ═══════════════════════════════════════════════════════════
#  IMAGE CLASSIFICATION
# ═══════════════════════════════════════════════════════════

CLASS_CONFIG = OrderedDict(
    {
        "normal": {
            "label": 0,
            "clinical_name": "Colon Normal",
            "cancer_role": "Mucosa colónica sana. Sin hallazgos patológicos.",
            "risk_level": "ninguno",
            "action": "Control rutinario según edad y factores de riesgo",
        },
        "polyp": {
            "label": 1,
            "clinical_name": "Pólipo",
            "cancer_role": (
                "Precursor directo: 95% de cánceres colorrectales "
                "empiezan como pólipo adenomatoso"
            ),
            "risk_level": "alto",
            "action": "Extirpar (polipectomía) + biopsia",
        },
        "inflammation": {
            "label": 2,
            "clinical_name": "Inflamación (Colitis Ulcerosa)",
            "cancer_role": (
                "Inflamación crónica del colon aumenta riesgo "
                "de cáncer colorrectal 2-5x"
            ),
            "risk_level": "medio-alto",
            "action": "Tratamiento antiinflamatorio + vigilancia endoscópica",
        },
    }
)

NUM_CLASSES = len(CLASS_CONFIG)

DEFAULT_CLASS_NAMES: dict[int, str] = {0: "normal", 1: "polyp", 2: "inflammation"}

# ═══════════════════════════════════════════════════════════
#  MULTI-SOURCE DATASET MAPPING
# ═══════════════════════════════════════════════════════════

SOURCE_MAP = {
    "hyperkvasir": {
        "normal": [
            "lower-gi-tract/anatomical-landmarks/cecum",
            "lower-gi-tract/anatomical-landmarks/retroflex-rectum",
        ],
        "polyp": [
            "lower-gi-tract/pathological-findings/polyps",
        ],
        "inflammation": [
            "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-0-1",
            "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-1",
            "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-1-2",
            "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-2",
            "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-2-3",
            "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-3",
        ],
    },
    "cvc_clinicdb": {
        "polyp": ["Original"],
    },
    "limuc": {
        "normal": ["0"],
        "inflammation": ["1", "2", "3"],
    },
    "curated_colon": {
        "polyp": ["polyp"],
    },
}

# ═══════════════════════════════════════════════════════════
#  KAGGLE DATASETS
# ═══════════════════════════════════════════════════════════

KAGGLE_DATASETS = {
    "tabular_risk": "ankushpanday2/colorectal-cancer-global-dataset-and-predictions",
    "curated_colon": "francismon/curated-colon-dataset-for-deep-learning",
    "cvc_clinicdb": "balraj98/cvcclinicdb",
    "tabular_smoke": "sooyoungher/smoking-drinking-dataset",
}

# ═══════════════════════════════════════════════════════════
#  IMAGE PREPROCESSING
# ═══════════════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGE_GLOB_PATTERNS = [f"*{ext}" for ext in IMAGE_EXTENSIONS]

UPLOAD_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ═══════════════════════════════════════════════════════════
#  TABULAR FEATURES
# ═══════════════════════════════════════════════════════════

NUMERIC_FEATURES = [
    "age",
    "bmi",
    "hemoglobin",
    "hematocrit",
    "wbc_count",
    "platelet_count",
    "albumin",
    "iron_serum",
    "ferritin",
    "crp",
    "cea",
    "ca19_9",
    "pack_years_smoked",
    "previous_polyps_count",
]

CATEGORICAL_FEATURES = [
    "gender",
    "ethnicity",
    "smoking_status",
    "alcohol_consumption",
    "physical_activity",
    "diet_type",
]

BINARY_FEATURES = [
    "family_history_ccr",
    "family_history_polyps",
    "family_history_lynch",
    "family_history_fap",
    "has_ibd",
    "has_diabetes_t2",
    "previous_polyps",
    "previous_cancer",
    "fobt_positive",
    "fit_positive",
]

CLINICAL_NUMERIC_FEATURES = [
    "Age",
    "Smoking_History",
    "CEA_Level_ng_mL",
    "Hemoglobin_g_dL",
    "PyRad_ADC_Mean",
    "PyRad_ADC_Std",
    "PyRad_Entropy",
    "PyRad_GLCM_Contrast",
    "PyRad_GLCM_Homogeneity",
    "PyRad_Shape_Sphericity",
    "PyRad_FirstOrder_Skewness",
]

TABULAR_TARGET = "Diagnosis"

# Default values for missing clinical data (used in diagnosis engine)
CLINICAL_DEFAULTS = {
    "pack_years_smoked": 0.0,
    "albumin": 4.5,
    "hematocrit": 45.0,
    "iron_serum": 100.0,
    "wbc_count": 7.0,
    "platelet_count": 250.0,
    "ferritin": 100.0,
    "crp": 1.0,
    "cea": 1.5,
    "ca19_9": 10.0,
    "previous_polyps_count": 0,
}

# ═══════════════════════════════════════════════════════════
#  PHYSIOLOGICAL CLIP RANGES
# ═══════════════════════════════════════════════════════════

PHYSIOLOGICAL_RANGES = {
    "hemoglobin": (5, 19),
    "hematocrit": (15, 58),
    "iron_serum": (5, 200),
    "ferritin": (1, 350),
    "wbc_count": (2, 25),
    "platelet_count": (100, 700),
    "albumin": (1.5, 5.8),
    "bmi": (14, 55),
    "crp": (0.01, 25),
    "cea": (0.05, 250),
    "ca19_9": (0.1, 600),
    "age": (18, 100),
    "pack_years_smoked": (0, 80),
    "previous_polyps_count": (0, 20),
}

# ═══════════════════════════════════════════════════════════
#  GDC API MAPPINGS
# ═══════════════════════════════════════════════════════════

GDC_TOBACCO_MAP = {
    "current smoker": "current",
    "current reformed smoker for > 15 years": "former",
    "current reformed smoker for < or = 15 years": "former",
    "lifelong non-smoker": "never",
    "not reported": "unknown",
}

GDC_ALCOHOL_MAP = {
    "yes": "moderate",
    "no": "none",
    "not reported": "unknown",
}

# ═══════════════════════════════════════════════════════════
#  KAGGLE RISK DATA MAPPINGS
# ═══════════════════════════════════════════════════════════

KAGGLE_RACE_MAP = {
    "White": "white",
    "Black": "black",
    "Asian": "asian",
    "Hispanic": "hispanic",
    "Other": "other",
}

KAGGLE_SMOKING_MAP = {"Current": "current", "Former": "former", "Never": "never"}

KAGGLE_ALCOHOL_MAP = {
    "High": "heavy",
    "Moderate": "moderate",
    "Low": "moderate",
    "None": "none",
}

KAGGLE_ACTIVITY_MAP = {"High": "active", "Moderate": "moderate", "Low": "sedentary"}

KAGGLE_DIET_MAP = {
    "Western": "high_fat_low_fiber",
    "Mediterranean": "high_fiber",
    "Balanced": "balanced",
    "Vegetarian": "high_fiber",
    "High-Fiber": "high_fiber",
    "Low-Fiber": "high_fat_low_fiber",
}

KAGGLE_STAGE_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4}

# ═══════════════════════════════════════════════════════════
#  CURATED COLON FOLDER KEYWORDS
# ═══════════════════════════════════════════════════════════

POLYP_KEYWORDS = ["polyp", "adenoma", "cancer", "tumor", "malignant", "lesion"]
NORMAL_KEYWORDS = ["normal", "healthy", "benign", "negative", "clean"]

# ═══════════════════════════════════════════════════════════
#  MAYO SCORE PATTERNS (LIMUC)
# ═══════════════════════════════════════════════════════════

MAYO_FOLDER_PATTERNS = [
    "{score}",
    "Mayo {score}",
    "Mayo_{score}",
    "mayo {score}",
    "mayo_{score}",
    "Mayo{score}",
    "mayo{score}",
]

# ═══════════════════════════════════════════════════════════
#  DIAGNOSIS RISK LEVELS
# ═══════════════════════════════════════════════════════════

RISK_LEVELS = {
    "high": {"level": "ALTO", "color": "#D32F2F", "threshold": 0.7},
    "moderate": {"level": "MODERADO", "color": "#F57C00", "threshold": 0.4},
    "low": {"level": "BAJO", "color": "#388E3C", "threshold": 0.0},
}

# ═══════════════════════════════════════════════════════════
#  SOURCE DETECTION PREFIXES
# ═══════════════════════════════════════════════════════════

SOURCE_PREFIXES = {
    "hk_": "hyperkvasir",
    "cvc_": "cvc_clinicdb",
    "limuc_": "limuc",
    "curated_": "curated_colon",
}


DATASET_CONFIG = {
    "hyperkvasir": {
        "base_path": paths.DATA / "hyperkvasir_raw",
        "search_dirs": [
            "labeled-images",
            "hyperkvasir_labeled/labeled-images",
        ],
        "description": "HyperKvasir: Dataset noruego de GI (Simula)",
        "color": "#3498db",
        "subcarpetas_interes": {
            "normal": [
                "lower-gi-tract/anatomical-landmarks/cecum",
                "lower-gi-tract/anatomical-landmarks/retroflex-rectum",
            ],
            "polyp": [
                "lower-gi-tract/pathological-findings/polyps",
            ],
            "inflammation": [
                "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-0-1",
                "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-1",
                "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-1-2",
                "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-2",
                "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-2-3",
                "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-3",
            ],
        },
    },
    "cvc_clinicdb": {
        "base_path": paths.DATA / "raw" / "cvc_clinicdb",
        "search_dirs": [
            "Original",
            "CVC-ClinicDB/Original",
            "PNG/Original",
            ".",
        ],
        "description": "CVC-ClinicDB: Pólipos (Barcelona)",
        "color": "#e74c3c",
        "subcarpetas_interes": {
            "polyp": ["Original", "."],
        },
    },
    "limuc": {
        "base_path": paths.DATA / "raw" / "limuc",
        "search_dirs": [],  # Estructura especial: paciente/mayo_score/
        "description": "LIMUC: Colitis Ulcerosa por Mayo score",
        "color": "#2ecc71",
        "subcarpetas_interes": {
            "normal": ["Mayo 0", "0"],
            "inflammation": ["Mayo 1", "1", "Mayo 2", "2", "Mayo 3", "3"],
        },
    },
    "curated_colon": {
        "base_path": paths.DATA / "raw" / "curated_colon",
        "search_dirs": [],
        "description": "Curated Colon: Dataset curado para DL",
        "color": "#f39c12",
        "subcarpetas_interes": {},
    },
}


def detect_source_from_stem(stem: str) -> str:
    """Detects dataset source from filename prefix."""
    for prefix, source in SOURCE_PREFIXES.items():
        if stem.startswith(prefix):
            return source
    return "unknown"


# ═══════════════════════════════════════════════════════════
#  TABULAR COLON ANALYSIS — Análisis de riesgo por demografía
# ═══════════════════════════════════════════════════════════

# Características demográficas para análisis
TABULAR_COLON_FEATURES = [
    "Age",
    "Gender",
    "Country",
    "Family_History",
    "Smoking_History",
    "Alcohol_Consumption",
    "Obesity_BMI",
    "Diet_Risk",
    "Physical_Activity",
    "Diabetes",
    "Inflammatory_Bowel_Disease",
    "Urban_or_Rural",
    "Genetic_Mutation",
]

# Target
TABULAR_COLON_TARGET = "Diagnosis"

# Grupos de edad para análisis
TABULAR_COLON_AGE_GROUPS = {
    "30-40": (30, 40),
    "40-50": (40, 50),
    "50-60": (50, 60),
    "60-70": (60, 70),
    "70-80": (70, 80),
    "80+": (80, 100),
}

# Mapeo de países a regiones
TABULAR_COLON_COUNTRY_TO_REGION = {
    "UK": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Spain": "Europe",
    "Italy": "Europe",
    "Canada": "North America",
    "USA": "North America",
    "Mexico": "North America",
    "Japan": "Asia",
    "China": "Asia",
    "South Korea": "Asia",
    "India": "Asia",
    "Brazil": "South America",
    "Argentina": "South America",
    "Chile": "South America",
    "Australia": "Oceania",
    "New Zealand": "Oceania",
}

TABULAR_COLON_COUNTRIES = list(TABULAR_COLON_COUNTRY_TO_REGION.keys()) + [
    "Nigeria",
    "Pakistan",
    "South Africa",
]

# Mapeo binario (Yes/No)
TABULAR_COLON_BINARY_MAP = {
    "Yes": 1,
    "No": 0,
}

# Mapeo de BMI
TABULAR_COLON_BMI_MAP = {
    "Normal": "Normal",
    "Overweight": "Overweight",
    "Obese": "Obese",
}

# Mapeo de riesgo dietético
TABULAR_COLON_DIET_RISK_MAP = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
}

# Mapeo de actividad física
TABULAR_COLON_ACTIVITY_MAP = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
}

# Configuración de entrenamiento
TABULAR_COLON_RANDOM_SEED = 42
TABULAR_COLON_TEST_SIZE = 0.30
TABULAR_COLON_VAL_SIZE = 0.10

# Modelos a entrenar
TABULAR_COLON_MODELS = ["logistic_regression", "cox_model"]

# Umbrales de riesgo
TABULAR_COLON_RISK_THRESHOLDS = {
    "low": 0.3,
    "moderate": 0.5,
    "high": 0.7,
    "very_high": 0.85,
}

# Columnas a descartar
TABULAR_COLON_COLUMNS_TO_DROP = [
    "Patient_ID",
    "Country",
    "Cancer_Stage",
    "Tumor_Size_mm",
    "Treatment_Type",
    "Survival_5_years",
    "Mortality",
    "Healthcare_Costs",
    "Survival_Prediction",
    "Economic_Classification",
    "Healthcare_Access",
    "Insurance_Status",
    "Incidence_Rate_per_100K",
    "Mortality_Rate_per_100K",
    "Prevention_Index",
    "Early_Detection",
    "LC_Healthy",
    "Screening_History",
    "Risk_Score",
    "Access_Score",
    "Age_Risk_Group",
    "LC_Dietary",
    "LC_High_Risk",
    "LC_Sedentary",
]

# Mapeo de columnas originales a nombres limpios
TABULAR_COLON_COLUMN_RENAME = {
    "Gender": "Gender",
    "Family_History": "Family_History",
    "Smoking_History": "Smoking_History",
    "Alcohol_Consumption": "Alcohol_Consumption",
    "Obesity_BMI": "Obesity_BMI",
    "Diet_Risk": "Diet_Risk",
    "Physical_Activity": "Physical_Activity",
    "Diabetes": "Diabetes",
    "Inflammatory_Bowel_Disease": "Inflammatory_Bowel_Disease",
    "Urban_or_Rural": "Urban_or_Rural",
    "Genetic_Mutation": "Genetic_Mutation",
    "Diagnosis": "Diagnosis",
}

# Configuración para Cox
TABULAR_COLON_COX_CONFIG = {
    "penalizer": 0.1,
    "n_baseline_knots": 10,
}

# Configuración para regresión logística
TABULAR_COLON_LOGISTIC_CONFIG = {
    "max_iter": 1000,
    "solver": "lbfgs",
    "class_weight": "balanced",
}

# Métricas a calcular
TABULAR_COLON_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "specificity",
    "npv",
    "sensitivity",
]

# Colores para visualización
TABULAR_COLON_COLORS = {
    "logistic_regression": "#3498db",
    "cox_model": "#e74c3c",
}

# Nombres legibles para modelos
TABULAR_COLON_MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "cox_model": "Cox Proportional Hazards",
}

TABULAR_COLON_ANALYSIS_IMAGES = {
    "roc_comparison": {
        "filename": "roc_comparison.png",
        "title": "ROC Curves",
        "description": "Compares ranking power across all thresholds.",
    },
    "precision_recall_comparison": {
        "filename": "precision_recall_comparison.png",
        "title": "Precision-Recall Curves",
        "description": "Shows class-detection quality under imbalance.",
    },
    "risk_distribution_comparison": {
        "filename": "risk_distribution_comparison.png",
        "title": "Risk Score Distribution",
        "description": "Explains separation between healthy and CRC cases.",
    },
    "threshold_analysis_logistic": {
        "filename": "threshold_analysis_logistic.png",
        "title": "Threshold Analysis",
        "description": "Displays sensitivity, specificity and F1 trade-offs for Logistic Regression.",
    },
    "calibration_logistic": {
        "filename": "calibration_logistic.png",
        "title": "Calibration Curve",
        "description": "Checks whether predicted risk matches observed frequency.",
    },
    "feature_importance_comparison": {
        "filename": "feature_importance_comparison.png",
        "title": "Feature Importance",
        "description": "Highlights the variables driving each model.",
    },
    "confusion_matrix_logistic": {
        "filename": "confusion_matrix_Logistic_Regression.png",
        "title": "Logistic Confusion Matrix",
        "description": "Summarizes the classification errors at the chosen threshold.",
    },
    "confusion_matrix_cox": {
        "filename": "confusion_matrix_Cox_Model.png",
        "title": "Cox Confusion Matrix",
        "description": "Summarizes Cox classification errors using the risk cutoff.",
    },
}


# ═══════════════════════════════════════════════════════════
#  XGBOOST TABULAR ANALYSIS — Country, smoking and alcohol
# ═══════════════════════════════════════════════════════════

TABULAR_ANALYSIS_RAW_FEATURES = [
    "Country",
    "Age",
    "Gender",
    "Smoking_History",
    "Alcohol_Consumption",
    "Physical_Activity",
]

TABULAR_ANALYSIS_NUMERIC_FEATURES = ["Age"]

TABULAR_ANALYSIS_CATEGORICAL_FEATURES = [
    "Country",
    "Gender",
    "Smoking_History",
    "Alcohol_Consumption",
    "Physical_Activity",
]

TABULAR_ANALYSIS_STAGE_COLUMN = "Cancer_Stage"
TABULAR_ANALYSIS_TARGET = "Advanced_Cancer"
TABULAR_ANALYSIS_POSITIVE_STAGES = ["Regional", "Metastatic"]

TABULAR_ANALYSIS_RANDOM_SEED = 42
TABULAR_ANALYSIS_TEST_SIZE = 0.25

TABULAR_ANALYSIS_DEFAULT_PROFILE = {
    "Country": "USA",
    "Age": 55,
    "Gender": "M",
    "Smoking_History": "No",
    "Alcohol_Consumption": "No",
    "Physical_Activity": "Moderate",
}

TABULAR_ANALYSIS_XGB_CONFIG = {
    "n_estimators": 220,
    "max_depth": 4,
    "learning_rate": 0.06,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "reg_lambda": 1.0,
}

TABULAR_ANALYSIS_IMAGE_CONFIG = {
    "roc_curve": {
        "filename": "roc_curve.png",
        "title": "ROC Curve",
        "description": "Capacidad del modelo para separar casos localizados de avanzados.",
    },
    "feature_importance": {
        "filename": "feature_importance.png",
        "title": "Feature Importance",
        "description": "Peso relativo de país, edad, género, tabaco, alcohol y actividad física.",
    },
    "country_advanced_rate": {
        "filename": "country_advanced_rate.png",
        "title": "Advanced Cancer by Country",
        "description": "Tasa observada de cáncer avanzado por país.",
    },
    "country_smoking_heatmap": {
        "filename": "country_smoking_heatmap.png",
        "title": "Smoking Effect by Country",
        "description": "Comparación del impacto del tabaquismo por país.",
    },
    "country_alcohol_heatmap": {
        "filename": "country_alcohol_heatmap.png",
        "title": "Alcohol Effect by Country",
        "description": "Comparación del impacto del alcohol por país.",
    },
    "age_smoking_distribution": {
        "filename": "age_smoking_distribution.png",
        "title": "Age Distribution by Smoking",
        "description": "Cómo se reparte la edad según tabaquismo y estadio avanzado.",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  REVERSE LOGIC TABULAR ANALYSIS — Predicting Smoking & Alcohol from Clinical Data
# ═══════════════════════════════════════════════════════════════════════════════

# FEATURES (Independent Variables - Clinical & Demographic Indicators)
REVERSE_ANALYSIS_FEATURES = [
    "sex",
    "age",
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
    "BMI",
    "AST_ALT_ratio",
]

REVERSE_ANALYSIS_SMOKE_FEATURES = [
    "sex",
    "age",
    "height",
    "BMI",
    "weight",
    "waistline",
    "triglyceride",
    "HDL_chole",
    "LDL_chole",
    "hemoglobin",
    "waist_height_ratio",
    "hemoglobin_per_height",
]

REVERSE_ANALYSIS_DRINK_FEATURES = [
    "age",
    "BMI",
    "waistline",
    "triglyceride",
    "HDL_chole",
    "LDL_chole",
    "gamma_GTP",
    "SGOT_AST",
    "SGOT_ALT",
    "AST_ALT_ratio",
    "height",
    "waist_height_ratio",
    "gamma_GTP_log",
    "liver_index",
    "age_sex_interaction",
    "bmi_category",
]

# Engineered Features (calculated from raw features)
REVERSE_ANALYSIS_ENGINEERED_FEATURES = [
    "waist_height_ratio",
    "hemoglobin_per_height",
    "gamma_GTP_log",
    "liver_index",
    "age_sex_interaction",
    "bmi_category",
]

REVERSE_DROP_FEATURES = ["sight_left", "sight_right", "hear_left", "hear_right"]

# Biomarcadores específicos a analizar
REVERSE_ANALYSIS_BIOMARKERS = ["SGOT_AST", "SGOT_ALT", "gamma_GTP", "AST_ALT_ratio"]

# TARGETS (Binary Outcomes to Predict)
REVERSE_ANALYSIS_TARGET_SMOKING = "SMK_stat_type_cd"
REVERSE_ANALYSIS_TARGET_ALCOHOL = "DRK_YN"

# Feature Type Specifications
REVERSE_ANALYSIS_NUMERIC_FEATURES = [
    "age",
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
    "BMI",
    "AST_ALT_ratio",
    "waist_height_ratio",
    "hemoglobin_per_height",
    "gamma_GTP_log",
    "liver_index",
    "age_sex_interaction",
    "bmi_category",  # Ordinal feature (0-3), not categorical
]

REVERSE_ANALYSIS_CATEGORICAL_FEATURES = ["sex"]  # Only sex is truly categorical

# Training Configuration
REVERSE_ANALYSIS_RANDOM_SEED = 42
REVERSE_ANALYSIS_TEST_SIZE = 0.25
REVERSE_ANALYSIS_VAL_SIZE = 0.15
REVERSE_ANALYSIS_CV_FOLDS = 5

# Model Configurations XGBoost y Random Forest
# ─ Smoking History
# SMOKING (SMK_stat_type_cd) → imbalanced (60/40)
# Basado en: Davagdorj 2020 + Oh 2026 (cáncer gástrico NHIS) + Hwang 2024
REVERSE_ANALYSIS_SMOKING_XGBOOST_CONFIG = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "scale_pos_weight": 602431 / 388889,
}

REVERSE_ANALYSIS_SMOKING_RANDOM_FOREST_CONFIG = {
    "n_estimators": 400,
    "max_depth": 10,
    "class_weight": {0: 1.0, 1: 1.55},
    "random_state": REVERSE_ANALYSIS_RANDOM_SEED,
    "n_jobs": -1,
    "min_samples_leaf": 5,  # evita overfitting
}

# ─ Alcohol Consumption
# ALCOHOL (DRK_YN) → casi balanceado + gamma_GTP dominante
# Basado en: Dalal 2022 (enfermedad hepática) + Lee 2025 (diabetes NHIS)
REVERSE_ANALYSIS_ALCOHOL_XGBOOST_CONFIG = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.5,
    "reg_lambda": 1.5,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
}

REVERSE_ANALYSIS_ALCOHOL_RANDOM_FOREST_CONFIG = {
    "n_estimators": 400,
    "max_depth": 10,
    "class_weight": "balanced",
    "random_state": REVERSE_ANALYSIS_RANDOM_SEED,
    "n_jobs": -1,
    "min_samples_leaf": 5,
}

# ─ LightGBM (para Stacking)
REVERSE_ANALYSIS_SMOKING_LIGHTGBM_CONFIG = {
    "n_estimators": 300,
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "num_leaves": 31,
    "objective": "binary",
    "metric": "auc",
    "scale_pos_weight": 602431 / 388889,
    "verbose": -1,
}

REVERSE_ANALYSIS_ALCOHOL_LIGHTGBM_CONFIG = {
    "n_estimators": 300,
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "num_leaves": 31,
    "objective": "binary",
    "metric": "auc",
    "verbose": -1,
}

# Feature Weights - Reduce impact of 'sex' on model predictions
# Emphasize clinical biomarkers over demographic variables
# Weights are relative importance scores (normalized 0-1)
REVERSE_ANALYSIS_FEATURE_WEIGHTS = {
    # SMOKING FEATURES - Emphasis on hemoglobin & metabolic indicators
    "sex": 0.05,  # Very low weight - reduce demographic bias
    "age": 0.10,  # Low weight - age has minimal effect on smoking detection
    "height": 0.08,  # Low weight
    "BMI": 0.15,  # Moderate weight
    "weight": 0.12,  # Moderate weight
    "waistline": 0.18,  # High weight - strong smoking indicator
    "triglyceride": 0.15,  # High weight - metabolic marker
    "HDL_chole": 0.12,  # Moderate weight
    "LDL_chole": 0.10,  # Moderate weight
    "hemoglobin": 0.20,  # VERY HIGH weight - dominant smoking indicator
    "waist_height_ratio": 0.22,  # VERY HIGH weight - engineered metabolic indicator
    "hemoglobin_per_height": 0.25,  # CRITICAL weight - strongest smoking predictor
    # ALCOHOL FEATURES
    "gamma_GTP": 0.28,  # CRITICAL weight - dominant alcohol indicator (liver enzyme)
    "SGOT_AST": 0.18,  # High weight - hepatic damage marker
    "SGOT_ALT": 0.16,  # High weight - hepatic damage marker
    "AST_ALT_ratio": 0.15,  # Moderate weight - derived hepatic ratio
    "gamma_GTP_log": 0.26,  # VERY HIGH weight - log-transformed liver enzyme
    "liver_index": 0.20,  # High weight - composite hepatic indicator
    "age_sex_interaction": 0.08,  # Low weight - age-sex effect is weak
    "bmi_category": 0.10,  # Low weight - categorical BMI discretization
}

# ─ Optuna Hyperparameter Optimization
REVERSE_ANALYSIS_USE_OPTUNA = False  # Set to True to enable Optuna tuning
REVERSE_ANALYSIS_OPTUNA_TRIALS = 100  # Number of Optuna trials
REVERSE_ANALYSIS_OPTUNA_TIMEOUT = 3600  # Timeout in seconds (1 hour)

# Default Profile for Frontend Screening
REVERSE_ANALYSIS_DEFAULT_PROFILE = {
    "height": 175,
    "weight": 75,
    "waistline": 85.0,
    "SBP": 120.0,
    "DBP": 80.0,
    "BLDS": 100.0,
    "tot_chole": 190.0,
    "HDL_chole": 50.0,
    "LDL_chole": 110.0,
    "triglyceride": 120.0,
    "hemoglobin": 15.0,
    "urine_protein": 1.0,
    "serum_creatinine": 1.0,
    "SGOT_AST": 25.0,
    "SGOT_ALT": 25.0,
    "gamma_GTP": 30.0,
    "BMI": 24.49,
    "AST_ALT_ratio": 1.0,
}

REVERSE_ANALYSIS_MODELS = ["xgboost", "random_forest", "lightgbm"]
REVERSE_ANALYSIS_MODEL_NAMES = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
}
REVERSE_ANALYSIS_COLORS = {
    "xgboost": "#1f77b4",
    "random_forest": "#2ca02c",
    "lightgbm": "#ff7f0e",
}

# ─ Stacking Configuration
REVERSE_ANALYSIS_USE_STACKING = True  # Enable XGBoost + LightGBM stacking
REVERSE_ANALYSIS_STACKING_MODELS = ["xgboost", "lightgbm"]  # Models to stack

CANCER_MEANS = np.array([np.log(7.0), 11.40, 1240.0, 6.10, 33.0])
CANCER_STDS = np.array([1.55, 2.80, 360.0, 2.10, 20.0])
CANCER_CORR = np.array(
    [
        [1.00, -0.30, -0.22, 0.32, 0.25],
        [-0.30, 1.00, 0.28, -0.25, -0.18],
        [-0.22, 0.28, 1.00, -0.38, -0.30],
        [0.32, -0.25, -0.38, 1.00, 0.48],
        [0.25, -0.18, -0.30, 0.48, 1.00],
    ]
)

HEALTHY_MEANS = np.array([np.log(2.10), 13.50, 1540.0, 4.80, 21.0])
HEALTHY_STDS = np.array([0.85, 2.10, 260.0, 1.35, 9.5])

HEALTHY_CORR = np.array(
    [
        [1.00, -0.08, -0.06, 0.12, 0.09],
        [-0.08, 1.00, 0.16, -0.07, -0.05],
        [-0.06, 0.16, 1.00, -0.22, -0.16],
        [0.12, -0.07, -0.22, 1.00, 0.32],
        [0.09, -0.05, -0.16, 0.32, 1.00],
    ]
)


# Calibrated against: Gollub 2018, Lambregts 2013, Horvat 2019, NCCN 2023.
STAGE_PARAMS = {
    # T1 — confined to mucosa/submucosa. Clinically almost indistinguishable
    # from benign tissue by serum biomarkers: CEA normal or marginally
    # elevated, ADC mildly restricted only in lesions >1 cm. Intentional
    # overlap with healthy forces probabilities 30-60%.
    1: (
        [np.log(1.9), 13.4, 1430.0, 4.92, 22.0],
        [0.72, 2.00, 280.0, 1.35, 10.0],
        0.67,
        0.72,
        0.14,
    ),
    # T2 — invades muscularis propria: moderate signal; still considerable
    # overlap with inflamed healthy tissue and high individual variability.
    2: (
        [np.log(5.0), 12.2, 1120.0, 5.70, 28.0],
        [0.88, 2.00, 280.0, 1.45, 12.0],
        0.50,
        0.67,
        0.48,
    ),
    # T3 — penetrates subserosa: elevated CEA, clearly restricted ADC,
    # anaemia.
    3: (
        [np.log(18.0), 10.3, 840.0, 6.60, 41.0],
        [1.00, 2.20, 280.0, 1.40, 14.0],
        0.32,
        0.56,
        0.90,
    ),
    # T4/M1 — perforation or metastasis: very high CEA, very low ADC,
    # necrotic tumour.
    4: (
        [np.log(90.0), 8.0, 650.0, 7.60, 56.0],
        [1.20, 1.80, 230.0, 1.45, 17.0],
        0.18,
        0.43,
        1.35,
    ),
}

CATEGORICAL_ENCODINGS: dict[str, dict[str, int]] = {
    "gender": {
        "male": 0,
        "m": 0,
        "masculino": 0,
        "female": 1,
        "f": 1,
        "femenino": 1,
    },
    "smoking_status": {
        "never": 0,
        "nunca": 0,
        "former": 1,
        "exfumador": 1,
        "current": 2,
        "fumador": 2,
    },
    "alcohol_consumption": {
        "none": 0,
        "ninguno": 0,
        "moderate": 1,
        "moderado": 1,
        "heavy": 2,
        "alto": 2,
    },
    "physical_activity": {
        "sedentary": 0,
        "sedentario": 0,
        "low": 1,
        "bajo": 1,
        "moderate": 2,
        "moderado": 2,
        "high": 3,
        "alto": 3,
    },
    "diet_type": {
        "western": 0,
        "occidental": 0,
        "mediterranean": 1,
        "mediterranea": 1,
        "vegetarian": 2,
        "vegetariana": 2,
    },
    "ethnicity": {
        "caucasian": 0,
        "caucasico": 0,
        "hispanic": 1,
        "hispanico": 1,
        "african": 2,
        "africano": 2,
        "asian": 3,
        "asiatico": 3,
        "other": 4,
        "otro": 4,
    },
}

BOOLEAN_FEATURES: frozenset[str] = frozenset(
    {
        "family_history_ccr",
        "family_history_polyps",
        "family_history_lynch",
        "family_history_fap",
        "has_ibd",
        "has_diabetes_t2",
        "previous_polyps",
        "previous_cancer",
        "fobt_positive",
        "fit_positive",
    }
)

BOOL_TRUE_STRINGS = frozenset({"yes", "true", "1", "sí", "si"})
