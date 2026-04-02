"""
src/data/processing/patient_feature_row.py
===========================================
Feature engineering para un registro individual de paciente.

Contiene la misma lógica de cálculo que tabular_preprocessor.py
pero orientada a procesar una única fila proveniente del formulario
de la app, en lugar de un dataset completo.

Importado por: diagnosis/engine.py, frontend/pages/02_cribado.py
"""

from __future__ import annotations

import pandas as pd

from src.config.constants import (
    ACTIVITY_MAP,
    AGE_RISK_GROUP_MAP,
    BINARY_MAP,
    DIET_MAP,
    GENDER_MAP,
    LIFESTYLE_CLUSTER_OPTIONS,
    OBESITY_MAP,
    SCREENING_MAP,
    URBAN_MAP,
)
from src.core.types import PatientFeatureResult, PatientFormInput
from src.data.processing.tabular_preprocessor import (
    calc_access_score,
    calc_age_risk_group,
    calc_lifestyle_cluster,
    calc_prevention_index,
    calc_risk_score,
)


def build_feature_row(inputs: PatientFormInput) -> PatientFeatureResult:
    """
    Procesa el diccionario de valores raw del formulario clínico y devuelve
    una fila de features numéricas lista para ser enviada al modelo tabular.

    Parámetros
    ----------
    inputs : PatientFormInput
        Valores crudos del formulario (strings categóricos, int/float para
        variables continuas).

    Retorna
    -------
    PatientFeatureResult con:
      · df_row           — 1 fila × 24 features numéricas (pd.DataFrame)
      · risk_score       — Risk_Score calculado [0-10]
      · prevention_index — Prevention_Index calculado [0-10]
      · access_score     — Access_Score calculado [0|1]
      · age_risk_group   — Categoría de riesgo por edad (str)
      · lifestyle_cluster— Perfil de estilo de vida (str)
    """
    age       = inputs["age"]
    gender    = inputs["gender"]
    fam_hist  = inputs["family_history"]
    smoking   = inputs["smoking"]
    alcohol   = inputs["alcohol"]
    obesity   = inputs["obesity"]
    diet      = inputs["diet_risk"]
    activity  = inputs["physical_activity"]
    diabetes  = inputs["diabetes"]
    ibd       = inputs["ibd"]
    genetic   = inputs["genetic"]
    screening = inputs["screening"]
    early_det = inputs["early_detection"]
    incidence = inputs["incidence_rate"]
    mortality = inputs["mortality_rate"]
    urban     = inputs["urban_rural"]

    # ── Features derivadas ──────────────────────────────────────────────────
    risk_score        = calc_risk_score(
        obesity, diet, activity, smoking, alcohol,
        diabetes, ibd, genetic, fam_hist, age,
    )
    prevention_index  = calc_prevention_index(screening, early_det, activity, diet)
    access_score      = calc_access_score(urban)
    age_risk_group    = calc_age_risk_group(age)
    lifestyle_cluster = calc_lifestyle_cluster(obesity, diet, activity, smoking, alcohol)

    # ── Fila numérica ───────────────────────────────────────────────────────
    row: dict = {
        "Age":                        age,
        "Gender":                     GENDER_MAP[gender],
        "Family_History":             BINARY_MAP[fam_hist],
        "Smoking_History":            BINARY_MAP[smoking],
        "Alcohol_Consumption":        BINARY_MAP[alcohol],
        "Obesity_BMI":                OBESITY_MAP[obesity],
        "Diet_Risk":                  DIET_MAP[diet],
        "Physical_Activity":          ACTIVITY_MAP[activity],
        "Diabetes":                   BINARY_MAP[diabetes],
        "Inflammatory_Bowel_Disease": BINARY_MAP[ibd],
        "Genetic_Mutation":           BINARY_MAP[genetic],
        "Screening_History":          SCREENING_MAP[screening],
        "Early_Detection":            BINARY_MAP[early_det],
        "Incidence_Rate_per_100K":    incidence,
        "Mortality_Rate_per_100K":    mortality,
        "Urban_or_Rural":             URBAN_MAP[urban],
        "Risk_Score":                 risk_score,
        "Prevention_Index":           prevention_index,
        "Access_Score":               access_score,
        "Age_Risk_Group":             AGE_RISK_GROUP_MAP[age_risk_group],
        # One-hot Lifestyle_Cluster
        **{f"LC_{opt}": int(lifestyle_cluster == opt) for opt in LIFESTYLE_CLUSTER_OPTIONS},
    }

    df_row = pd.DataFrame([row])

    return PatientFeatureResult(
        df_row=df_row,
        risk_score=risk_score,
        prevention_index=prevention_index,
        access_score=access_score,
        age_risk_group=age_risk_group,
        lifestyle_cluster=lifestyle_cluster,
    )
