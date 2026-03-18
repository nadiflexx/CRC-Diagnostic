"""
Centralized constants, mappings, and domain dictionaries.
Single source of truth for all magic values scattered across the codebase.
"""

from collections import OrderedDict

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

TABULAR_TARGET = "has_cancer"

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


def detect_source_from_stem(stem: str) -> str:
    """Detects dataset source from filename prefix."""
    for prefix, source in SOURCE_PREFIXES.items():
        if stem.startswith(prefix):
            return source
    return "unknown"
