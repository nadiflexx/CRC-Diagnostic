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
#  TABULAR FEATURES  (Clinical + Radiomic — CRC synthetic dataset)
# ═══════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════
#  PHYSIOLOGICAL CLIP RANGES
# ═══════════════════════════════════════════════════════════

PHYSIOLOGICAL_RANGES = {
    "CEA_Level_ng_mL": (0.10, 5000.0),
    "Hemoglobin_g_dL": (5.00, 20.0),
    "PyRad_ADC_Mean": (200.0, 2500.0),
    "PyRad_ADC_Std": (5.0, 400.0),
    "PyRad_Entropy": (0.1, 10.0),
    "PyRad_GLCM_Contrast": (0.1, 200.0),
    "PyRad_GLCM_Homogeneity": (0.01, 1.0),
    "PyRad_Shape_Sphericity": (0.25, 1.0),
    "Age": (18.0, 100.0),
    "Smoking_History": (0.0, 1.0),
}

# Default values for missing clinical data (used in diagnosis engine)
CLINICAL_DEFAULTS = {
    "Age": 50.0,
    "Smoking_History": 0.0,
    "CEA_Level_ng_mL": 2.1,
    "Hemoglobin_g_dL": 13.5,
    "PyRad_ADC_Mean": 1540.0,
    "PyRad_ADC_Std": 80.0,
    "PyRad_Entropy": 4.8,
    "PyRad_GLCM_Contrast": 21.0,
    "PyRad_GLCM_Homogeneity": 0.72,
    "PyRad_Shape_Sphericity": 0.73,
    "PyRad_FirstOrder_Skewness": 0.05,
}

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
        "search_dirs": [],
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

# Biomarcadores específicos a analizar
REVERSE_ANALYSIS_BIOMARKERS = ["SGOT_AST", "SGOT_ALT", "gamma_GTP", "AST_ALT_ratio"]

# TARGETS (Binary Outcomes to Predict)
REVERSE_ANALYSIS_TARGET_SMOKING = "SMK_stat_type_cd"
REVERSE_ANALYSIS_TARGET_ALCOHOL = "DRK_YN"

# Feature Type Specifications
REVERSE_ANALYSIS_NUMERIC_FEATURES = [
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

REVERSE_ANALYSIS_CATEGORICAL_FEATURES = ["sex"]

# Training Configuration
REVERSE_ANALYSIS_RANDOM_SEED = 42
REVERSE_ANALYSIS_TEST_SIZE = 0.25
REVERSE_ANALYSIS_VAL_SIZE = 0.15
REVERSE_ANALYSIS_CV_FOLDS = 5

# Model Configurations XGBoost y Random Forest
REVERSE_ANALYSIS_XGBOOST_CONFIG = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.08,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}

REVERSE_ANALYSIS_XGBOOST_FEATURE_WEIGHTS = {
    "sex": 0.1,
}

REVERSE_ANALYSIS_RANDOM_FOREST_CONFIG = {
    "n_estimators": 200,
    "max_depth": 10,
    "class_weight": "balanced",
    "random_state": REVERSE_ANALYSIS_RANDOM_SEED,
    "n_jobs": 1,
}

# Default Profile for Frontend Screening
REVERSE_ANALYSIS_DEFAULT_PROFILE = {
    "sex": "Male",
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

REVERSE_ANALYSIS_MODELS = ["xgboost", "random_forest"]
REVERSE_ANALYSIS_MODEL_NAMES = {"xgboost": "XGBoost", "random_forest": "Random Forest"}
REVERSE_ANALYSIS_COLORS = {"xgboost": "#1f77b4", "random_forest": "#2ca02c"}


def detect_source_from_stem(stem: str) -> str:
    """Detects dataset source from filename prefix."""
    for prefix, source in SOURCE_PREFIXES.items():
        if stem.startswith(prefix):
            return source
    return "unknown"


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
