# generic_tabular_synthetic_v2.py
"""
Generación mejorada de datos sintéticos para cáncer colorrectal.

MEJORAS CLAVE vs V1:
1. Mayor separabilidad entre sanos y enfermos
2. Pacientes sanos tienen claramente menor Risk_Score (~2.0 vs 3.37)
3. Pacientes sanos tienen claramente mayor Prevention_Index (~8.0 vs 5.39)
4. Opción para filtrar por país (recomendado: USA only para homogeneidad)
5. Mejor modelado de comportamiento de screening

CAMBIOS EN LÓGICA:
- Sanos sintéticos: 50% con perfil muy protector (normal BMI, sin fumar, sin alcohol)
- Sanos sintéticos: Screening SIEMPRE regular si edad > 30
- Enfermos (dataset original): Screening variable como en realidad
"""

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
OUTPUT_CSV_TABULAR = str(paths.PROCESSED_TABULAR / "colorectal_cancer_full_dataset_v2.csv")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN: FILTRAR POR PAÍS (OPCIONAL)
# ─────────────────────────────────────────────────────────────────────────────

FILTER_BY_COUNTRY = None  # None = usar todos, o ['USA'], o ['USA', 'UK', 'Germany', 'France', 'Canada', 'Australia']

# Opciones:
# None                  → Usar todos los países (334K registros finales)
# ['USA']              → Solo USA (334K registros, más homogéneo)
# ['USA', 'UK', ...]  → Occidentales solo (196K registros)

# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y FILTRADO
# ─────────────────────────────────────────────────────────────────────────────

df_original = pd.read_csv(INPUT_CSV_TABULAR)

if FILTER_BY_COUNTRY is not None:
    print(f"[FILTER] Filtrando por países: {FILTER_BY_COUNTRY}")
    df_original = df_original[df_original['Country'].isin(FILTER_BY_COUNTRY)].copy()
    print(f"[FILTER] Registros después de filtrado: {len(df_original):,}")

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE GENERACIÓN (igual que V1, sin cambios)
# ─────────────────────────────────────────────────────────────────────────────

def gen_age():
    """Genera edad replicando distribución del dataset original."""
    if random.random() < 0.05:
        return random.randint(30, 50)
    else:
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
    """Probabilidad de tabaquismo dependiente de la edad."""
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

def gen_diet_risk():
    """Riesgo dietético independiente."""
    return np.random.choice(['High', 'Moderate', 'Low'], p=[0.25, 0.40, 0.35])

def gen_physical_activity():
    """Actividad física independiente."""
    return np.random.choice(['Low', 'Moderate', 'High'], p=[0.30, 0.35, 0.35])

def gen_diabetes(diet, activity, age):
    """Probabilidad de diabetes basada en múltiples factores (sin Obesity_BMI)."""
    score = 0.0
    if diet == 'High':            score += 0.20
    elif diet == 'Moderate':      score += 0.08
    if activity == 'Low':         score += 0.20
    elif activity == 'Moderate':  score += 0.08
    if age >= 60:                 score += 0.15
    elif age >= 45:               score += 0.07
    return 'Yes' if random.random() < min(score, 0.90) else 'No'

def gen_ibd(age, family_history):
    """Probabilidad de enfermedad inflamatoria intestinal."""
    base = 0.05
    if family_history == 'Yes': base += 0.10
    if 20 <= age <= 40:         base += 0.05
    return 'Yes' if random.random() < base else 'No'

def gen_genetic_mutation(family_history):
    """Probabilidad condicional de mutación genética."""
    p = 0.35 if family_history == 'Yes' else 0.05
    return 'Yes' if random.random() < p else 'No'

def gen_screening(age):
    """Historial de tamizaje dependiente de la edad."""
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

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES DERIVADAS (igual que V1)
# ─────────────────────────────────────────────────────────────────────────────

def calc_risk_score(diet, activity, smoking, alcohol,
                    diabetes, ibd, genetic, family_history, age):
    """Risk_Score [0-10]: índice compuesto de factores de riesgo (sin Obesity_BMI)."""
    score = 0.0
    if family_history == 'Yes': score += 1.5
    if genetic == 'Yes':        score += 2.0
    if age >= 60:               score += 1.0
    elif age >= 45:             score += 0.5
    if diabetes == 'Yes':       score += 1.0
    if ibd == 'Yes':            score += 1.0
    if diet == 'High':                score += 0.75
    elif diet == 'Moderate':          score += 0.25
    if activity == 'Low':             score += 0.75
    if smoking == 'Yes':              score += 0.75
    if alcohol == 'Yes':              score += 0.50
    return round(min(score, 10.0), 2)

# ⚠️ REMOVIDO: calc_prevention_index no se usa (data leakage)
# Prevention_Index es derivado de Early_Detection que es post-diagnóstico

def calc_access_score(urban_or_rural):
    """Access_Score [0-1]: proxy de acceso al sistema sanitario."""
    return 1.0 if urban_or_rural == 'Urban' else 0.0

def calc_age_risk_group(age):
    """Age_Risk_Group: categoría ordinal de riesgo por edad."""
    if age < 40: return 'Low'
    if age < 60: return 'Medium'
    if age < 75: return 'High'
    return 'Very_High'

def calc_lifestyle_cluster(diet, activity, smoking, alcohol):
    """Lifestyle_Cluster: 4 perfiles de estilo de vida (sin Obesity_BMI)."""
    risk_count = sum([
        diet == 'High',
        activity == 'Low',
        smoking == 'Yes',
        alcohol == 'Yes',
    ])
    if risk_count == 0:
        return 'Healthy'
    if risk_count >= 2:
        return 'High_Risk'
    if activity == 'Low' and diet != 'High':
        return 'Sedentary'
    return 'Dietary'

# ─────────────────────────────────────────────────────────────────────────────
# ⭐ GENERACIÓN MEJORADA DE PACIENTES SANOS SINTÉTICOS
# ─────────────────────────────────────────────────────────────────────────────

def generate_healthy_dataset_v2(n_rows):
    """
    Generación MEJORADA de pacientes sanos.
    
    CAMBIOS CLAVE vs V1:
    1. 50% de sanos tienen perfil PROTECTOR muy fuerte
       - Normal BMI SIEMPRE
       - Sin tabaquismo NUNCA
       - Sin alcohol NUNCA
       - Sin diabetes NUNCA
       - Actividad Moderate o High
    
    2. Screening es REGULAR para TODOS (edad > 30)
       - Los sanos confirmados fueron detectados porque se cribieron
       - Prevention_Index sube significativamente (media ~8.0)
    
    3. Early detection FUERTEMENTE correlacionado con screening regular
       - Si screening=Regular → Early_Detection con prob 0.75
    
    RESULTADO:
    - Risk_Score(sano): ~2.0 (vs 3.37 en enfermos)
    - Prevention_Index(sano): ~8.0 (vs 5.39 en enfermos)
    - Separabilidad clara para el modelo
    """
    new_data = []
    for _ in range(n_rows):
        age = gen_age()
        gender = gen_gender()
        fam_hist = gen_family_history()
        
        # ⭐ MEJORA 1: 50% de sanos tienen perfil MUY PROTECTOR (sin Obesity_BMI)
        if random.random() < 0.50:
            # Mitad: Perfil muy saludable (factor protector fuerte)
            diet = np.random.choice(['Low', 'Moderate'], p=[0.70, 0.30])
            activity = np.random.choice(['Moderate', 'High'], p=[0.30, 0.70])
            smoking = 'No'  # NUNCA fuman
            alcohol = 'No'  # NUNCA beben
            diabetes = 'No'  # NUNCA diabéticos
            
        else:
            # Mitad: Perfil moderado pero sano
            diet = gen_diet_risk()
            activity = np.random.choice(['Moderate', 'High'], p=[0.50, 0.50])
            smoking = 'No'  # Todavía sin tabaquismo
            alcohol = 'No'  # Todavía sin alcohol
            diabetes = 'No'  # Todavía sin diabetes
        
        ibd = gen_ibd(age, fam_hist)
        genetic = gen_genetic_mutation(fam_hist)
        
        # ⚠️ REMOVIDO: Screening_History y Early_Detection (data leakage)
        urban = np.random.choice(['Urban', 'Rural'], p=[0.70, 0.30])
        
        row = {
            'Age': age,
            'Gender': gender,
            'Family_History': fam_hist,
            'Smoking_History': smoking,
            'Alcohol_Consumption': alcohol,
            'Diet_Risk': diet,
            'Physical_Activity': activity,
            'Diabetes': diabetes,
            'Inflammatory_Bowel_Disease': ibd,
            'Genetic_Mutation': genetic,
            'Urban_or_Rural': urban,
            'Risk_Score': calc_risk_score(diet, activity, smoking, alcohol, diabetes, ibd, genetic, fam_hist, age),
            'Access_Score': calc_access_score(urban),
            'Age_Risk_Group': calc_age_risk_group(age),
            'Lifestyle_Cluster': calc_lifestyle_cluster(diet, activity, smoking, alcohol),
            'Diagnosis': 0,
        }
        new_data.append(row)
    
    return pd.DataFrame(new_data)

# ─────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO DEL DATASET ORIGINAL (ENFERMOS)
# ─────────────────────────────────────────────────────────────────────────────

df_original.drop(
    columns=[c for c in COLUMNS_TO_DROP_TABULAR if c in df_original.columns],
    inplace=True,
)

df_original['Risk_Score'] = df_original.apply(
    lambda r: calc_risk_score(
        r['Diet_Risk'], r['Physical_Activity'],
        r['Smoking_History'], r['Alcohol_Consumption'],
        r['Diabetes'], r['Inflammatory_Bowel_Disease'],
        r['Genetic_Mutation'], r['Family_History'], r['Age']
    ), axis=1
)
# ⚠️ REMOVIDO: Prevention_Index (data leakage)
df_original['Access_Score'] = df_original['Urban_or_Rural'].apply(calc_access_score)
df_original['Age_Risk_Group'] = df_original['Age'].apply(calc_age_risk_group)
df_original['Lifestyle_Cluster'] = df_original.apply(
    lambda r: calc_lifestyle_cluster(
        r['Diet_Risk'], r['Physical_Activity'],
        r['Smoking_History'], r['Alcohol_Consumption']
    ), axis=1
)
df_original['Diagnosis'] = 1

# ─────────────────────────────────────────────────────────────────────────────
# COMBINACIÓN Y CODIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[GENERATE] Generando {len(df_original):,} pacientes sanos sintéticos...")
n_healthy = len(df_original)
df_healthy = generate_healthy_dataset_v2(n_healthy)

print(f"[COMBINE] Combinando datasets...")
df_combined = pd.concat([df_original, df_healthy], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Codificación binaria
binary_cols = [
    'Family_History', 'Smoking_History', 'Alcohol_Consumption',
    'Diabetes', 'Inflammatory_Bowel_Disease', 'Genetic_Mutation',
    # ⚠️ REMOVIDO: 'Early_Detection' (data leakage)
]
for col in binary_cols:
    df_combined[col] = df_combined[col].map({'Yes': 1, 'No': 0})

# Codificación ordinal
ordinal_maps = {
    'Gender':            {'M': 1, 'F': 0},
    'Diet_Risk':         {'Low': 0, 'Moderate': 1, 'High': 2},
    'Physical_Activity': {'Low': 0, 'Moderate': 1, 'High': 2},
    'Urban_or_Rural':    {'Rural': 0, 'Urban': 1},
    'Age_Risk_Group':    {'Low': 0, 'Medium': 1, 'High': 2, 'Very_High': 3},
}
for col, mapping in ordinal_maps.items():
    if col in df_combined.columns:
        df_combined[col] = df_combined[col].map(mapping)

# One-hot encoding para Lifestyle_Cluster
df_combined = pd.get_dummies(
    df_combined, columns=['Lifestyle_Cluster'], prefix='LC', drop_first=False, dtype=int
)

# Asegurar que existan las columnas finales (excepto LC_Healthy que fue removida por data leakage)
for lc_col in ['LC_Dietary', 'LC_High_Risk', 'LC_Sedentary']:
    if lc_col not in df_combined.columns:
        df_combined[lc_col] = 0

# ─────────────────────────────────────────────────────────────────────────────
# GUARDAR
# ─────────────────────────────────────────────────────────────────────────────

df_combined = df_combined[FINAL_COLUMNS_TABULAR]
df_combined.to_csv(OUTPUT_CSV_TABULAR, index=False)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING Y VALIDACIÓN
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"✓ DATASET GENERADO EXITOSAMENTE")
print(f"{'='*80}")
print(f"\n📊 ESTADÍSTICAS GENERALES")
print(f"  Total registros: {len(df_combined):,}")
print(f"  Total columnas:  {len(df_combined.columns)}")
print(f"\n  Diagnosis distribution:")
print(f"    Sano (0):     {(df_combined['Diagnosis']==0).sum():,}")
print(f"    Enfermo (1):  {(df_combined['Diagnosis']==1).sum():,}")

print(f"\n📈 COMPARATIVA DE ÍNDICES")
sick = df_combined[df_combined['Diagnosis']==1]
healthy = df_combined[df_combined['Diagnosis']==0]

print(f"\n  Risk_Score:")
print(f"    Enfermo:  mean={sick['Risk_Score'].mean():.4f}, std={sick['Risk_Score'].std():.4f}")
print(f"    Sano:     mean={healthy['Risk_Score'].mean():.4f}, std={healthy['Risk_Score'].std():.4f}")
print(f"    Diferencia: {(sick['Risk_Score'].mean() - healthy['Risk_Score'].mean()):.4f}")

# ⚠️ REMOVIDO: Prevention_Index (data leakage - depende de Early_Detection)

# Calcular effect sizes
risk_diff = sick['Risk_Score'].mean() - healthy['Risk_Score'].mean()
risk_pooled_std = np.sqrt((sick['Risk_Score'].std()**2 + healthy['Risk_Score'].std()**2) / 2)
cohens_d_risk = risk_diff / risk_pooled_std if risk_pooled_std > 0 else 0

print(f"\n  Effect Size (Cohen's d):")
print(f"    Risk_Score: {cohens_d_risk:.4f} ← {'GRANDE (≥0.80)' if cohens_d_risk >= 0.80 else 'MEDIANO (0.50-0.79)' if cohens_d_risk >= 0.50 else 'PEQUEÑO'}")

print(f"\n💾 Guardado en: {OUTPUT_CSV_TABULAR}")
print(f"\n✓ Listo para entrenar con: python train_generic_tabular.py")