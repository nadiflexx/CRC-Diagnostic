"""
Configuración de rutas y features para el modelo tabular de predicción de cáncer colorrectal.
Todas las rutas se obtienen del sistema centralizado de paths.
"""

from pathlib import Path
from src.config.paths import paths

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS DE DATOS (del sistema centralizado)
# ─────────────────────────────────────────────────────────────────────────────

# Entrada: datos originales (raw)
INPUT_CSV_TABULAR: str = str(paths.RAW_TABULAR / "colorectal_cancer_dataset.csv")

# Salida: datos procesados y sintéticos combinados
OUTPUT_CSV_TABULAR: str = str(paths.PROCESSED_TABULAR / "colorectal_cancer_full_dataset_v2.csv")
INPUT_CSV_TABULAR_PROCESSED: str = str(paths.PROCESSED_TABULAR / "colorectal_cancer_full_dataset_v2.csv")

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS DEL MODELO (del sistema centralizado)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR: str = str(paths.GENERIC_TABULAR_MODEL_DIR)
MODEL_PATH: str = str(paths.GENERIC_TABULAR_MODEL_PATH)
SCALER_PATH: str = str(paths.GENERIC_TABULAR_SCALER_PATH)
CONFIG_PATH: str = str(paths.GENERIC_TABULAR_CONFIG_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS DE ANÁLISIS (del sistema centralizado)
# ─────────────────────────────────────────────────────────────────────────────

ANALYSIS_DIR: str = str(paths.ANALYSIS)


# Columnas del dataset original que se eliminan debido a que no aportan al modelo de predicción
# INCLUIDO DATA LEAKAGE: Prevention_Index, Early_Detection, LC_Healthy, Screening_History
COLUMNS_TO_DROP_TABULAR = [
    'Patient_ID',
    'Country',
    'Cancer_Stage',
    'Tumor_Size_mm',
    'Treatment_Type',
    'Survival_5_years',
    'Mortality',
    'Healthcare_Costs',
    'Survival_Prediction',
    'Economic_Classification',
    'Healthcare_Access',
    'Insurance_Status',
    'Incidence_Rate_per_100K',
    'Mortality_Rate_per_100K',
    'Prevention_Index',        # ⚠️ Data leakage
    'Early_Detection',         # ⚠️ Data leakage
    'LC_Healthy',              # ⚠️ Data leakage
    'Screening_History',       # ⚠️ Data leakage
    'Obesity_BMI',             # ⚠️ May cause multicollinearity
]


# Columnas del dataset final (sin leakage, sin Obesity_BMI)
FINAL_COLUMNS_TABULAR = [
    'Age', 'Gender', 'Family_History', 'Smoking_History', 'Alcohol_Consumption',
    'Diet_Risk', 'Physical_Activity', 'Diabetes',
    'Inflammatory_Bowel_Disease', 'Genetic_Mutation',
    'Urban_or_Rural', 'Risk_Score', 'Access_Score',
    'Age_Risk_Group', 'Diagnosis',
    'LC_Dietary', 'LC_High_Risk', 'LC_Sedentary',
]

