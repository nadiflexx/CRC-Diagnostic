# Ruta de los imputs & outputs de datos
INPUT_CSV_TABULAR  = 'data/colorectal_cancer_dataset.csv'
OUTPUT_CSV_TABULAR = 'data/colorectal_cancer_full_dataset.csv'
INPUT_CSV_TABULAR_PROCESSED = 'data/colorectal_cancer_full_dataset.csv'
OUTPUT_MODEL_TABULAR = 'data/outputs/nn_model'

# Columnas del dataset original que se eliminan devido a que no aportan al modelo de predicción
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
]

# Columnas del dataset final
FINAL_COLUMNS_TABULAR = [
    'Age', 'Gender', 'Family_History', 'Smoking_History', 'Alcohol_Consumption',
    'Obesity_BMI', 'Diet_Risk', 'Physical_Activity', 'Diabetes',
    'Inflammatory_Bowel_Disease', 'Genetic_Mutation', 'Screening_History',
    'Early_Detection', 'Incidence_Rate_per_100K', 'Mortality_Rate_per_100K',
    'Urban_or_Rural', 'Risk_Score', 'Prevention_Index', 'Access_Score',
    'Age_Risk_Group', 'Diagnosis',
    'LC_Dietary', 'LC_Healthy', 'LC_High_Risk', 'LC_Sedentary',
]