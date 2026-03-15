"""
patient_features.py
===================
Módulo de feature engineering para un registro individual de paciente.

Contiene la misma lógica de cálculo que feature_engineering_synthetic_data.py
pero orientada a procesar una única fila proveniente del formulario de la app,
en lugar de un dataset completo.

Importado por: app_prediccion.py
"""

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# MAPEOS NUMÉRICOS
# Idénticos a los del script de entrenamiento (feature_engineering_synthetic_data.py).
# Centralizados aquí para no repetirlos ni introducir inconsistencias.
# ─────────────────────────────────────────────────────────────────────────────

BINARY_MAP  = {'Yes': 1, 'No': 0}
GENDER_MAP  = {'M': 1, 'F': 0}
OBESITY_MAP = {'Normal': 0, 'Overweight': 1, 'Obese': 2}
DIET_MAP    = {'Low': 0, 'Moderate': 1, 'High': 2}
ACT_MAP     = {'Low': 0, 'Moderate': 1, 'High': 2}
SCR_MAP     = {'Never': 0, 'Irregular': 1, 'Regular': 2}
URB_MAP     = {'Rural': 0, 'Urban': 1}
AGE_GRP_MAP = {'Low': 0, 'Medium': 1, 'High': 2, 'Very_High': 3}
LC_OPTIONS  = ['Dietary', 'Healthy', 'High_Risk', 'Sedentary']


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE CÁLCULO DE FEATURES DERIVADAS
# ─────────────────────────────────────────────────────────────────────────────

def calc_risk_score(
    obesity: str, diet: str, activity: str, smoking: str, alcohol: str,
    diabetes: str, ibd: str, genetic: str, family_history: str, age: int
) -> float:
    """
    Risk_Score [0-10]: índice compuesto de factores de riesgo modificables
    y no modificables.

    Pesos por factor:
      · Mutación genética    → 2.00  (mayor impacto en CRC hereditario)
      · Historia familiar    → 1.50
      · Edad ≥ 60            → 1.00
      · Diabetes             → 1.00
      · EII                  → 1.00
      · Obesidad (obeso)     → 0.75
      · Dieta alto riesgo    → 0.75
      · Actividad baja       → 0.75
      · Tabaquismo           → 0.75
      · Alcohol              → 0.50
      · Edad 45-59           → 0.50
      · Obesidad (sobrepeso) → 0.25
      · Dieta moderada       → 0.25
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
    if obesity == 'Obese':          score += 0.75
    elif obesity == 'Overweight':   score += 0.25
    if diet == 'High':              score += 0.75
    elif diet == 'Moderate':        score += 0.25
    if activity == 'Low':           score += 0.75
    if smoking == 'Yes':            score += 0.75
    if alcohol == 'Yes':            score += 0.50
    return round(min(score, 10.0), 2)


def calc_prevention_index(
    screening: str, early_detection: str, activity: str, diet: str
) -> float:
    """
    Prevention_Index [0-10]: refleja comportamientos preventivos activos.
    Complementa al Risk_Score desde el ángulo protector.

    Pesos por factor:
      · Cribado regular      → 3.50
      · Detección temprana   → 2.50
      · Actividad alta       → 2.00
      · Dieta baja en riesgo → 2.00
      · Cribado irregular    → 1.50
      · Actividad moderada   → 1.00
      · Dieta moderada       → 1.00
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


def calc_access_score(urban_or_rural: str) -> float:
    """
    Access_Score [0 | 1]: proxy de acceso al sistema sanitario basado en
    el entorno geográfico (única señal disponible tras eliminar las columnas
    de seguro médico y clasificación económica).
      Urban → 1.0
      Rural → 0.0
    """
    return 1.0 if urban_or_rural == 'Urban' else 0.0


def calc_age_risk_group(age: int) -> str:
    """
    Age_Risk_Group: categoría ordinal de riesgo por edad.
      < 40     → Low
      40-59    → Medium
      60-74    → High
      ≥ 75     → Very_High
    """
    if age < 40: return 'Low'
    if age < 60: return 'Medium'
    if age < 75: return 'High'
    return 'Very_High'


def calc_lifestyle_cluster(
    obesity: str, diet: str, activity: str, smoking: str, alcohol: str
) -> str:
    """
    Lifestyle_Cluster: perfil de estilo de vida basado en el número de
    factores de riesgo modificables presentes.

      0 factores          → Healthy
      ≥ 3 factores        → High_Risk
      Actividad baja      → Sedentary  (sin dieta de alto riesgo)
      Resto               → Dietary
    """
    risk_count = sum([
        obesity in ('Obese', 'Overweight'),
        diet == 'High',
        activity == 'Low',
        smoking == 'Yes',
        alcohol == 'Yes',
    ])
    if risk_count == 0:                          return 'Healthy'
    if risk_count >= 3:                          return 'High_Risk'
    if activity == 'Low' and diet != 'High':     return 'Sedentary'
    return 'Dietary'


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_row(inputs: dict) -> tuple[pd.DataFrame, float, float, float, str, str]:
    """
    Procesa el diccionario de valores raw del formulario y devuelve una fila
    de features numéricas lista para ser escalada y enviada al modelo.

    Parámetros
    ----------
    inputs : dict
        Claves esperadas:
          age, gender, family_history, smoking, alcohol, obesity,
          diet_risk, physical_activity, diabetes, ibd, genetic,
          screening, early_detection, incidence_rate, mortality_rate,
          urban_rural

    Retorna
    -------
    df_row           : pd.DataFrame  — 1 fila × 24 features numéricas
    risk_score       : float         — Risk_Score calculado [0-10]
    prevention_index : float         — Prevention_Index calculado [0-10]
    access_score     : float         — Access_Score calculado [0|1]
    age_risk_group   : str           — Categoría de riesgo por edad
    lifestyle_cluster: str           — Perfil de estilo de vida
    """
    # ── Extraer valores del diccionario ──────────────────────────────────────
    age       = inputs['age']
    gender    = inputs['gender']
    fam_hist  = inputs['family_history']
    smoking   = inputs['smoking']
    alcohol   = inputs['alcohol']
    obesity   = inputs['obesity']
    diet      = inputs['diet_risk']
    activity  = inputs['physical_activity']
    diabetes  = inputs['diabetes']
    ibd       = inputs['ibd']
    genetic   = inputs['genetic']
    screening = inputs['screening']
    early_det = inputs['early_detection']
    incidence = inputs['incidence_rate']
    mortality = inputs['mortality_rate']
    urban     = inputs['urban_rural']

    # ── Calcular features derivadas ──────────────────────────────────────────
    risk_score        = calc_risk_score(obesity, diet, activity, smoking, alcohol,
                                        diabetes, ibd, genetic, fam_hist, age)
    prevention_index  = calc_prevention_index(screening, early_det, activity, diet)
    access_score      = calc_access_score(urban)
    age_risk_group    = calc_age_risk_group(age)
    lifestyle_cluster = calc_lifestyle_cluster(obesity, diet, activity, smoking, alcohol)

    # ── Construir la fila numérica ────────────────────────────────────────────
    row = {
        'Age':                        age,
        'Gender':                     GENDER_MAP[gender],
        'Family_History':             BINARY_MAP[fam_hist],
        'Smoking_History':            BINARY_MAP[smoking],
        'Alcohol_Consumption':        BINARY_MAP[alcohol],
        'Obesity_BMI':                OBESITY_MAP[obesity],
        'Diet_Risk':                  DIET_MAP[diet],
        'Physical_Activity':          ACT_MAP[activity],
        'Diabetes':                   BINARY_MAP[diabetes],
        'Inflammatory_Bowel_Disease': BINARY_MAP[ibd],
        'Genetic_Mutation':           BINARY_MAP[genetic],
        'Screening_History':          SCR_MAP[screening],
        'Early_Detection':            BINARY_MAP[early_det],
        'Incidence_Rate_per_100K':    incidence,
        'Mortality_Rate_per_100K':    mortality,
        'Urban_or_Rural':             URB_MAP[urban],
        'Risk_Score':                 risk_score,
        'Prevention_Index':           prevention_index,
        'Access_Score':               access_score,
        'Age_Risk_Group':             AGE_GRP_MAP[age_risk_group],
        # One-hot Lifestyle_Cluster: 1 en la categoría correspondiente, 0 en el resto
        **{f'LC_{opt}': int(lifestyle_cluster == opt) for opt in LC_OPTIONS},
    }

    df_row = pd.DataFrame([row])
    return df_row, risk_score, prevention_index, access_score, age_risk_group, lifestyle_cluster