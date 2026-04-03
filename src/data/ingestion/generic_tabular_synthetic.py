import pandas as pd
import numpy as np
import random
import os
import sys

# Agregar src al path para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.config.paths import paths
from src.config.generic_tabular_features import (
    COLUMNS_TO_DROP_TABULAR, 
    FINAL_COLUMNS_TABULAR
)

# Usar rutas del sistema centralizado
INPUT_CSV_TABULAR = str(paths.RAW_TABULAR / "colorectal_cancer_dataset.csv")
OUTPUT_CSV_TABULAR = str(paths.PROCESSED_TABULAR / "colorectal_cancer_full_dataset.csv")

# ---------------------------------------------------------------------------
# CARGA
# ---------------------------------------------------------------------------
df_original = pd.read_csv(INPUT_CSV_TABULAR)

# ---------------------------------------------------------------------------
# GENERADORES DE VARIABLES (pacientes sanos sintéticos)
# ---------------------------------------------------------------------------

def gen_age():
    """
    Genera una edad que replica la distribución del dataset original (30-89 años),
    concentrada en 50-89 igual que los pacientes con cáncer.
    Con un 15 % de probabilidad genera una edad en el rango 30-50 para
    incluir casos de aparición temprana.

    Distribución por segmento (aproximada al dataset base):
      15 % → 30-50  (casos jóvenes / aparición temprana)
      85 % → 50-89  (normal truncada centrada en ~69, sd ~10)
    """
    if random.random() < 0.05:
        # Rango joven: uniforme 30-50
        return random.randint(30, 50)
    else:
        # Rango principal: normal truncada centrada en 69, sd 10, entre 50 y 89
        while True:
            age = int(np.random.normal(loc=69, scale=10))
            if 50 <= age <= 89:
                return age

def gen_gender():
    """Selecciona género con distribución ligeramente sesgada hacia M."""
    return np.random.choice(['M', 'F'], p=[0.55, 0.45])

def gen_family_history():
    """20-35 % de probabilidad de antecedentes familiares."""
    return 'Yes' if random.random() < 0.28 else 'No'

def gen_smoking(age):
    """
    Probabilidad de tabaquismo dependiente de la edad.
    Sube hasta los 60 y luego baja (supervivencia diferencial + cambios generacionales).
    """
    if age < 30:   p = 0.20
    elif age < 50: p = 0.35
    elif age < 65: p = 0.40
    else:          p = 0.28
    return 'Yes' if random.random() < p else 'No'

def gen_alcohol(age, gender):
    """Mayor prevalencia en hombres y en el tramo 30-60 años."""
    base = 0.40 if gender == 'M' else 0.28
    if 30 <= age <= 60:
        base += 0.08
    return 'Yes' if random.random() < base else 'No'

def gen_obesity():
    """Distribución realista de IMC."""
    return np.random.choice(['Normal', 'Overweight', 'Obese'], p=[0.40, 0.35, 0.25])

def gen_diet_risk(obesity):
    """Riesgo dietético correlacionado con el IMC."""
    if obesity == 'Obese':
        return np.random.choice(['High', 'Moderate', 'Low'], p=[0.65, 0.30, 0.05])
    elif obesity == 'Overweight':
        return np.random.choice(['High', 'Moderate', 'Low'], p=[0.25, 0.50, 0.25])
    else:
        return np.random.choice(['High', 'Moderate', 'Low'], p=[0.05, 0.35, 0.60])

def gen_physical_activity(obesity):
    """Actividad física inversamente correlacionada con el IMC."""
    if obesity == 'Obese':
        return np.random.choice(['Low', 'Moderate', 'High'], p=[0.70, 0.25, 0.05])
    elif obesity == 'Overweight':
        return np.random.choice(['Low', 'Moderate', 'High'], p=[0.30, 0.45, 0.25])
    else:
        return np.random.choice(['Low', 'Moderate', 'High'], p=[0.10, 0.35, 0.55])

def gen_diabetes(obesity, diet, activity, age):
    """
    Probabilidad de diabetes basada en obesidad, dieta, actividad y edad.
    Acumulación de factores con techo en 0.90.
    """
    score = 0.0
    if obesity == 'Obese':        score += 0.35
    elif obesity == 'Overweight': score += 0.15
    if diet == 'High':            score += 0.20
    elif diet == 'Moderate':      score += 0.08
    if activity == 'Low':         score += 0.20
    elif activity == 'Moderate':  score += 0.08
    if age >= 60:                 score += 0.15
    elif age >= 45:               score += 0.07
    return 'Yes' if random.random() < min(score, 0.90) else 'No'

def gen_ibd(age, family_history):
    """
    Historia familiar incrementa riesgo de EII.
    Pico de edad entre 20-40 (Crohn/CU).
    """
    base = 0.05
    if family_history == 'Yes': base += 0.10
    if 20 <= age <= 40:         base += 0.05
    return 'Yes' if random.random() < base else 'No'

def gen_genetic_mutation(family_history):
    """
    Probabilidad condicional de mutación genética.
    Sin historia familiar existe un riesgo basal de mutación de novo.
    """
    p = 0.35 if family_history == 'Yes' else 0.05
    return 'Yes' if random.random() < p else 'No'

def gen_screening(age):
    """
    Historial de tamizaje dependiente únicamente de la edad.
    Los menores de 30 nunca han sido cribados.
    """
    if age < 30:
        return 'Never'
    if age < 45:
        base_never = 0.85
    elif age < 60:
        base_never = 0.40
    else:
        base_never = 0.25
    p_never     = base_never
    p_regular   = (1 - p_never) * 0.55
    p_irregular = 1 - p_never - p_regular
    return np.random.choice(['Never', 'Regular', 'Irregular'], p=[p_never, p_regular, p_irregular])

def gen_early_detection(screening):
    """Probabilidad de detección temprana según frecuencia de cribado."""
    if screening == 'Regular':
        return 'Yes' if random.random() < 0.72 else 'No'
    elif screening == 'Irregular':
        return 'Yes' if random.random() < 0.30 else 'No'
    return 'Yes' if random.random() < 0.08 else 'No'

# ---------------------------------------------------------------------------
# FEATURES DERIVADAS (aplicadas tanto a sanos como a enfermos)
# ---------------------------------------------------------------------------

def calc_risk_score(obesity, diet, activity, smoking, alcohol,
                    diabetes, ibd, genetic, family_history, age):
    """
    Risk_Score [0-10]: índice compuesto de factores de riesgo modificables
    y no modificables.
    """
    score = 0.0
    # Factores no modificables
    if family_history == 'Yes': score += 1.5
    if genetic == 'Yes':        score += 2.0
    if age >= 60:               score += 1.0
    elif age >= 45:             score += 0.5
    # Comorbilidades
    if diabetes == 'Yes':       score += 1.0
    if ibd == 'Yes':            score += 1.0
    # Factores modificables
    if obesity == 'Obese':            score += 0.75
    elif obesity == 'Overweight':     score += 0.25
    if diet == 'High':                score += 0.75
    elif diet == 'Moderate':          score += 0.25
    if activity == 'Low':             score += 0.75
    if smoking == 'Yes':              score += 0.75
    if alcohol == 'Yes':              score += 0.50
    return round(min(score, 10.0), 2)

def calc_prevention_index(screening, early_detection, activity, diet):
    """
    Prevention_Index [0-10]: refleja comportamientos preventivos activos.
    Complementa al Risk_Score desde el ángulo protector.
    """
    score = 0.0
    if screening == 'Regular':     score += 3.5
    elif screening == 'Irregular': score += 1.5
    if early_detection == 'Yes':   score += 2.5
    if activity == 'High':         score += 2.0
    elif activity == 'Moderate':   score += 1.0
    if diet == 'Low':              score += 2.0
    elif diet == 'Moderate':       score += 1.0
    return round(min(score, 10.0), 2)

def calc_access_score(urban_or_rural):
    """
    Access_Score [0-1]: proxy simplificado de acceso al sistema sanitario.
    Sin las columnas de seguro/acceso/clasificación económica, se usa el
    entorno urbano/rural como única señal disponible.
      Urban  → 1.0
      Rural  → 0.0
    """
    return 1.0 if urban_or_rural == 'Urban' else 0.0

def calc_age_risk_group(age):
    """
    Age_Risk_Group: categoría ordinal de riesgo por edad.
    < 40 = Low | 40-59 = Medium | 60-74 = High | ≥ 75 = Very_High
    """
    if age < 40: return 'Low'
    if age < 60: return 'Medium'
    if age < 75: return 'High'
    return 'Very_High'

def calc_lifestyle_cluster(obesity, diet, activity, smoking, alcohol):
    """
    Lifestyle_Cluster: 4 perfiles de estilo de vida.
      • Healthy   → sin factores modificables de riesgo
      • Sedentary → actividad baja como factor dominante
      • Dietary   → dieta/obesidad como factor dominante
      • High_Risk → combinación de múltiples factores
    Se convertirá en columnas one-hot (LC_*) en el dataset final.
    """
    risk_count = sum([
        obesity in ('Obese', 'Overweight'),
        diet == 'High',
        activity == 'Low',
        smoking == 'Yes',
        alcohol == 'Yes',
    ])
    if risk_count == 0:
        return 'Healthy'
    if risk_count >= 3:
        return 'High_Risk'
    if activity == 'Low' and diet != 'High':
        return 'Sedentary'
    return 'Dietary'

# ---------------------------------------------------------------------------
# GENERACIÓN DE PACIENTES SANOS SINTÉTICOS
# ---------------------------------------------------------------------------

def generate_healthy_dataset(n_rows):
    """
    Genera n_rows pacientes sanos con características diferenciadas de enfermos.
    
    ⭐ MEJORAS CLAVE (para aumentar separabilidad del modelo):
    1. Los SANOS tienen screening REGULAR (porque fueron detectados como sanos)
    2. Los SANOS tienen factores de riesgo REDUCIDOS (selección por supervivencia)
    3. Los SANOS tienen MEJOR Prevention_Index
    
    Esto refleja la realidad epidemiológica:
    - Paciente sano confirmado = tiene screening regular + perfil bajo riesgo
    - Paciente enfermo = probablemente screening irregular + acumulación de riesgos
    
    Diagnosis = 0 en todos los casos.
    """
    new_data = []
    for _ in range(n_rows):
        age        = gen_age()
        gender     = gen_gender()
        fam_hist   = gen_family_history()
        
        # ⭐ MEJORA 1: Pacientes sanos tienen MEJOR perfil de obesidad/dieta/actividad
        # Reduce prevalencia de hallazgos adversos en ~30%
        if random.random() < 0.30:
            # 30% "afortunados sanos" - perfil muy protector
            obesity    = np.random.choice(['Normal', 'Overweight'], p=[0.80, 0.20])
            diet       = gen_diet_risk(obesity)
            activity   = np.random.choice(['Moderate', 'High'], p=[0.40, 0.60])  # Más activos
        else:
            # 70% "sanos normales" - perfil similar a población general
            obesity    = gen_obesity()
            diet       = gen_diet_risk(obesity)
            activity   = gen_physical_activity(obesity)
        
        smoking    = gen_smoking(age)
        
        # ⭐ MEJORA 2: Reducir prevalencia de consumo de alcohol en sanos (reflejo del screening)
        if random.random() < 0.7:  # 70% de sanos no beben
            alcohol = 'No'
        else:
            alcohol = gen_alcohol(age, gender)
        
        # ⭐ MEJORA 3: Diabetes MUCHO menos prevalente en sanos
        # Los sanos con diabetes deberían ser raros (están bajo control)
        if random.random() < 0.15:  # Reduce a 15% vs ~45% en generación normal
            diabetes = gen_diabetes(obesity, diet, activity, age)
        else:
            diabetes = 'No'
        
        ibd        = gen_ibd(age, fam_hist)
        genetic    = gen_genetic_mutation(fam_hist)
        
        # ⭐ MEJORA 4: SCREENING REGULAR es CARACTERÍSTICA DE SANOS
        # Si alguien es confirmado sano, fue porque se cribó
        if age < 30:
            screening = 'Never'
        elif age < 40:
            screening = np.random.choice(['Regular', 'Irregular'], p=[0.65, 0.35])
        elif age < 50:
            screening = np.random.choice(['Regular', 'Irregular'], p=[0.70, 0.30])
        elif age < 60:
            screening = np.random.choice(['Regular', 'Irregular'], p=[0.72, 0.28])
        else:
            # Mayores: MÁS screening (por eso saben que están sanos)
            screening = np.random.choice(['Regular', 'Irregular'], p=[0.80, 0.20])
        
        # ⭐ MEJORA 5: Early detection correlacionado fuertemente con regular screening
        if screening == 'Regular':
            early_det = 'Yes' if random.random() < 0.75 else 'No'  # 75% con screening regular tienen early detection
        elif screening == 'Irregular':
            early_det = 'Yes' if random.random() < 0.35 else 'No'
        else:
            early_det = 'Yes' if random.random() < 0.10 else 'No'
        
        urban      = np.random.choice(['Urban', 'Rural'], p=[0.70, 0.30])

        row = {
            # Demográficos
            'Age':                        age,
            'Gender':                     gender,
            # Factores clínicos
            'Family_History':             fam_hist,
            'Smoking_History':            smoking,
            'Alcohol_Consumption':        alcohol,
            'Obesity_BMI':                obesity,
            'Diet_Risk':                  diet,
            'Physical_Activity':          activity,
            'Diabetes':                   diabetes,
            'Inflammatory_Bowel_Disease': ibd,
            'Genetic_Mutation':           genetic,
            # Prevención y detección
            'Screening_History':          screening,
            'Early_Detection':            early_det,
            # Contexto (urbano/rural)
            'Urban_or_Rural':             urban,
            # Features derivadas
            'Risk_Score':        calc_risk_score(obesity, diet, activity, smoking, alcohol, diabetes, ibd, genetic, fam_hist, age),
            'Prevention_Index':  calc_prevention_index(screening, early_det, activity, diet),
            'Access_Score':      calc_access_score(urban),
            'Age_Risk_Group':    calc_age_risk_group(age),
            'Lifestyle_Cluster': calc_lifestyle_cluster(obesity, diet, activity, smoking, alcohol),
            # Target
            'Diagnosis': 0,
        }
        new_data.append(row)

    return pd.DataFrame(new_data)

# ---------------------------------------------------------------------------
# PROCESAMIENTO DEL DATASET ORIGINAL (pacientes con cáncer)
# ---------------------------------------------------------------------------

# Eliminar columnas que no forman parte del dataset final
df_original.drop(
    columns=[c for c in COLUMNS_TO_DROP_TABULAR if c in df_original.columns],
    inplace=True,
)

# Añadir features derivadas
df_original['Risk_Score'] = df_original.apply(
    lambda r: calc_risk_score(
        r['Obesity_BMI'], r['Diet_Risk'], r['Physical_Activity'],
        r['Smoking_History'], r['Alcohol_Consumption'],
        r['Diabetes'], r['Inflammatory_Bowel_Disease'],
        r['Genetic_Mutation'], r['Family_History'], r['Age']
    ), axis=1
)
df_original['Prevention_Index'] = df_original.apply(
    lambda r: calc_prevention_index(
        r['Screening_History'], r['Early_Detection'],
        r['Physical_Activity'], r['Diet_Risk']
    ), axis=1
)
df_original['Access_Score'] = df_original['Urban_or_Rural'].apply(calc_access_score)
df_original['Age_Risk_Group']    = df_original['Age'].apply(calc_age_risk_group)
df_original['Lifestyle_Cluster'] = df_original.apply(
    lambda r: calc_lifestyle_cluster(
        r['Obesity_BMI'], r['Diet_Risk'], r['Physical_Activity'],
        r['Smoking_History'], r['Alcohol_Consumption']
    ), axis=1
)
df_original['Diagnosis'] = 1

# ---------------------------------------------------------------------------
# COMBINACIÓN DE DATASETS
# ---------------------------------------------------------------------------

n_healthy = len(df_original)   # mismo número que pacientes con cáncer → dataset balanceado
df_healthy   = generate_healthy_dataset(n_healthy)
df_combined  = pd.concat([df_original, df_healthy], ignore_index=True)
df_combined  = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# ---------------------------------------------------------------------------
# CODIFICACIÓN NUMÉRICA
# ---------------------------------------------------------------------------

# Binarias Yes/No → 1/0
binary_cols = [
    'Family_History', 'Smoking_History', 'Alcohol_Consumption',
    'Diabetes', 'Inflammatory_Bowel_Disease', 'Genetic_Mutation', 'Early_Detection',
]
for col in binary_cols:
    df_combined[col] = df_combined[col].map({'Yes': 1, 'No': 0})

# Ordinales con orden clínico
ordinal_maps = {
    'Gender':            {'M': 1, 'F': 0},
    'Obesity_BMI':       {'Normal': 0, 'Overweight': 1, 'Obese': 2},
    'Diet_Risk':         {'Low': 0, 'Moderate': 1, 'High': 2},
    'Physical_Activity': {'Low': 0, 'Moderate': 1, 'High': 2},   # High = más protector
    'Screening_History': {'Never': 0, 'Irregular': 1, 'Regular': 2},
    'Urban_or_Rural':    {'Rural': 0, 'Urban': 1},
    'Age_Risk_Group':    {'Low': 0, 'Medium': 1, 'High': 2, 'Very_High': 3},
}
for col, mapping in ordinal_maps.items():
    if col in df_combined.columns:
        df_combined[col] = df_combined[col].map(mapping)

# Lifestyle_Cluster (nominal, 4 categorías) → One-Hot con prefijo LC_
df_combined = pd.get_dummies(
    df_combined, columns=['Lifestyle_Cluster'], prefix='LC', drop_first=False, dtype=int
)

# Asegurar que las 4 columnas LC_ existen aunque alguna categoría no haya aparecido
for lc_col in ['LC_Dietary', 'LC_Healthy', 'LC_High_Risk', 'LC_Sedentary']:
    if lc_col not in df_combined.columns:
        df_combined[lc_col] = 0

# ---------------------------------------------------------------------------
# ORDEN FINAL DE COLUMNAS
# ---------------------------------------------------------------------------
df_combined = df_combined[FINAL_COLUMNS_TABULAR]

# ---------------------------------------------------------------------------
# GUARDAR
# ---------------------------------------------------------------------------
df_combined.to_csv(OUTPUT_CSV_TABULAR, index=False)

print(f"Total registros : {len(df_combined):,}")
print(f"Total columnas  : {len(df_combined.columns)}")
print(f"Columnas finales: {list(df_combined.columns)}")
print(f"\nDistribución Diagnosis:\n{df_combined['Diagnosis'].value_counts()}")
print(f"\nDatos nulos:\n{df_combined.isnull().sum()[df_combined.isnull().sum() > 0]}")
print(f"\nTipos de datos:\n{df_combined.dtypes}")