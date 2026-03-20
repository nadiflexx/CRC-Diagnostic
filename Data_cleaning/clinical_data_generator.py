"""
clinical_data_generator.py

Genera un dataset clínico sintético de cáncer colorrectal (CRC) con
biomarcadores hematológicos y features radiómicas tipo PyRadiomics.

Partimos del CSV base (v1) y para cada paciente pasamos los valores
clínicos con distribuciones multivariante calibradas a partir de los
informes (NCCN 2023, ESGAR 2022, Duffy et al. 2021).

"""

import os

import numpy as np
import pandas as pd

# Semilla global para que el dataset sea reproducible en cualquier máquina
RNG = np.random.default_rng(42)


# Parámetros de distribución por grupo diagnóstico.
# cambiamos 5 variables base con multivariate_normal:
#   [log(CEA), Hemoglobina, ADC_medio, Entropía, Contraste]
# Usamos log(CEA) porque el CEA real sigue una distribución log-normal

# Grupo CANCER
CANCER_MEANS = np.array([np.log(7.0), 11.40, 1240.0, 6.10, 33.0])
CANCER_STDS  = np.array([1.10,         2.00,  270.0,  1.40, 13.0])

# Correlaciones entre biomarcadores para pacientes con cáncer.
# Los signos negativos tienen sentido clínico:
#   - Tumores con CEA alto suelen causar anemia (Hgb baja)
#   - ADC bajo (difusión restringida) va ligado a más CEA y más entropía
CANCER_CORR = np.array([
    [ 1.00, -0.30, -0.22,  0.32,  0.25],   # log(CEA)
    [-0.30,  1.00,  0.28, -0.25, -0.18],   # Hemoglobina
    [-0.22,  0.28,  1.00, -0.38, -0.30],   # ADC medio
    [ 0.32, -0.25, -0.38,  1.00,  0.48],   # Entropia GLCM
    [ 0.25, -0.18, -0.30,  0.48,  1.00],   # Contraste GLCM
])

# Grupo SANO
# Las correlaciones son mucho más débiles: sin tumor no hay la sinergia
# fisiopatológica que acopla estos biomarcadores en pacientes con cáncer
HEALTHY_MEANS = np.array([np.log(2.10), 13.50, 1540.0, 4.80, 21.0])
HEALTHY_STDS  = np.array([0.70,          1.80,  210.0,  1.10,  7.5])

HEALTHY_CORR = np.array([
    [ 1.00, -0.08, -0.06,  0.12,  0.09],   # log(CEA)
    [-0.08,  1.00,  0.16, -0.07, -0.05],   # Hemoglobina
    [-0.06,  0.16,  1.00, -0.22, -0.16],   # ADC medio
    [ 0.12, -0.07, -0.22,  1.00,  0.32],   # Entropia GLCM
    [ 0.09, -0.05, -0.16,  0.32,  1.00],   # Contraste GLCM
])


def corr_a_cov(corr: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """
    Convierte una matriz de correlación y un vector de std en una covarianza.
    Sigma = D * R * D, siendo D = diag(stds).

    Necesitamos esto para pasarle la covarianza a multivariate_normal.
    """
    D = np.diag(stds)
    return D @ corr @ D


def generar_features_clinicas(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Genera los valores clínicos y radiómicos para cada paciente del dataset base.

    Necesita como mínimo las columnas Age y Diagnosis.
    Si existen Gender, Smoking_History, etc., las usa para ajustar los valores.

    Devuelve un DataFrame con las 9 features clínicas y la variable Diagnosis.
    """
    df = df_base.copy()
    n  = len(df)

    # Añadimos Patient_ID si el CSV base no lo incluye
    if "Patient_ID" not in df.columns:
        df.insert(0, "Patient_ID", [f"PT-{i:05d}" for i in range(n)])

    # Extraemos las covariables auxiliares; si no existen en el CSV usamos valores neutros
    ages    = df["Age"].to_numpy(float)
    gender  = df.get("Gender",                     pd.Series(np.ones(n,  int))).to_numpy(int)
    smoking = df.get("Smoking_History",            pd.Series(np.zeros(n, int))).to_numpy(int)
    ibd     = df.get("Inflammatory_Bowel_Disease", pd.Series(np.zeros(n, int))).to_numpy(int)
    genetic = df.get("Genetic_Mutation",           pd.Series(np.zeros(n, int))).to_numpy(int)
    diag    = df["Diagnosis"].to_numpy(int)

    # Calculamos las matrices de covarianza una sola vez antes de los bucles
    cov_cancer  = corr_a_cov(CANCER_CORR,  CANCER_STDS)
    cov_healthy = corr_a_cov(HEALTHY_CORR, HEALTHY_STDS)

    # Arrays de salida que rellenaremos por grupo
    cea      = np.zeros(n)
    hgb      = np.zeros(n)
    adc_mean = np.zeros(n)
    entropy  = np.zeros(n)
    contrast = np.zeros(n)

    # Grupo cáncer
    cancer_idx = np.where(diag == 1)[0]
    n_c = len(cancer_idx)
    if n_c > 0:
        samples_c = RNG.multivariate_normal(CANCER_MEANS, cov_cancer, size=n_c)

        # Pacientes >70 años: CEA un poco más alto y Hgb algo más baja (NCCN 2023)
        ages_c = ages[cancer_idx]
        exceso = np.maximum(ages_c - 70, 0.0)
        delta_cea_edad = np.where(ages_c > 70, np.minimum(0.30 + 0.012 * exceso, 0.70), 0.0)
        delta_hgb_edad = np.where(ages_c > 70, np.maximum(-0.80 - 0.04 * exceso, -2.00), 0.0)

        # Las mujeres tienen de media 1.5 g/dL menos de hemoglobina (OMS 2011)
        offset_genero = np.where(gender[cancer_idx] == 1, 0.0, -1.5)

        # exp deshace el log: obtenemos CEA real con distribución log-normal
        cea[cancer_idx]      = np.exp(samples_c[:, 0] + delta_cea_edad)
        hgb[cancer_idx]      = samples_c[:, 1] + delta_hgb_edad + offset_genero
        adc_mean[cancer_idx] = samples_c[:, 2]
        entropy[cancer_idx]  = samples_c[:, 3]
        contrast[cancer_idx] = samples_c[:, 4]

    # Grupo sano
    healthy_idx = np.where(diag == 0)[0]
    n_h = len(healthy_idx)
    if n_h > 0:
        # Ajustamos el CEA basal por hábitos: tabaco, EII y mutaciones genéticas
        # elevan el CEA en personas sanas (Duffy et al. 2021)
        cea_log_base = (
            HEALTHY_MEANS[0]
            + 0.85 * smoking[healthy_idx]
            + 0.30 * ibd[healthy_idx]
            + 0.20 * genetic[healthy_idx]
        )
        # pasamos con media 0 para log(CEA) y sumamos el offset individual
        means_cero = HEALTHY_MEANS.copy()
        means_cero[0] = 0.0
        samples_h = RNG.multivariate_normal(means_cero, cov_healthy, size=n_h)

        offset_genero_h = np.where(gender[healthy_idx] == 1, 0.0, -1.5)

        cea[healthy_idx]      = np.exp(samples_h[:, 0] + cea_log_base)
        hgb[healthy_idx]      = samples_h[:, 1] + offset_genero_h
        adc_mean[healthy_idx] = samples_h[:, 2]
        entropy[healthy_idx]  = samples_h[:, 3]
        contrast[healthy_idx] = samples_h[:, 4]

    # Features radiómicas derivadas (calculadas para todos los pacientes)
    # ADC_Std: el tejido maligno es más heterogéneo, de ahí la mayor variabilidad
    adc_std = np.where(
        diag == 1,
        np.abs(adc_mean * 0.20) + RNG.normal(0, 38.0, n),
        np.abs(adc_mean * 0.11) + RNG.normal(0, 24.0, n),
    )

    # Homogeneidad GLCM: disminuye cuando contraste y entropía son altos
    base_hom = np.where(diag == 1, 0.32, 0.72)
    coef_c   = np.where(diag == 1, 0.003, 0.002)
    coef_e   = np.where(diag == 1, 0.016, 0.007)
    umbral_c = np.where(diag == 1, 30.0, 12.0)
    umbral_e = np.where(diag == 1, 5.0,   4.0)
    hom_raw = (
        RNG.normal(base_hom, 0.10, n)
        - coef_c * np.maximum(contrast - umbral_c, 0)
        - coef_e * np.maximum(entropy  - umbral_e, 0)
    )

    # Los tumores tienden a ser más irregulares (esfericidad menor)
    sph_raw  = np.where(diag == 1,
                        RNG.normal(0.62, 0.13, n),
                        RNG.normal(0.73, 0.13, n))

    # Distribución sesgada a la derecha en malignos (más asimetría de intensidades)
    skew_raw = np.where(diag == 1,
                        RNG.normal(0.85, 0.40, n),
                        RNG.normal(0.05, 0.48, n))

    # Montamos el DataFrame con np.clip para mantener rangos clínicamente válidos
    df_out = pd.DataFrame({
        "Patient_ID":                df["Patient_ID"].values,
        "Age":                       ages,
        "Smoking_History":           smoking,
        "CEA_Level_ng_mL":           np.clip(cea,      0.10, 5_000.0).round(2),
        "Hemoglobin_g_dL":           np.clip(hgb,      5.00,    20.0).round(2),
        "PyRad_ADC_Mean":            np.clip(adc_mean, 200.0, 2_500.0).round(2),
        "PyRad_ADC_Std":             np.clip(adc_std,    5.0,   400.0).round(2),
        "PyRad_Entropy":             np.clip(entropy,    0.1,    10.0).round(4),
        "PyRad_GLCM_Contrast":       np.clip(contrast,   0.1,   200.0).round(4),
        "PyRad_GLCM_Homogeneity":    np.clip(hom_raw,   0.01,    1.0).round(4),
        "PyRad_Shape_Sphericity":    np.clip(sph_raw,   0.25,    1.0).round(4),
        "PyRad_FirstOrder_Skewness": skew_raw.round(4),
        "Diagnosis":                 diag,
    })

    # Zona gris biológica:Ruido realista para simular casos clínicos ambiguos y evitar un dataset "perfecto"
    # A. Cánceres de estadio temprano (~12%) con perfil clínico casi normal
    # En clínica real, un 15-20% de CRC estadio I tienen CEA <3 ng/mL y Hgb preservada
    c_idx     = df_out.index[df_out["Diagnosis"] == 1].to_numpy()
    n_early   = max(1, int(len(c_idx) * 0.12))
    early_idx = RNG.choice(c_idx, size=n_early, replace=False)

    df_out.loc[early_idx, "CEA_Level_ng_mL"]          = RNG.lognormal(np.log(1.7), 0.45, n_early).clip(0.5,  3.0).round(2)
    df_out.loc[early_idx, "Hemoglobin_g_dL"]           = RNG.normal(14.2,    1.00, n_early).clip(13.5, 17.5).round(2)
    df_out.loc[early_idx, "PyRad_ADC_Mean"]            = RNG.normal(1430.0, 190.0, n_early).clip(1000.0, 1900.0).round(2)
    df_out.loc[early_idx, "PyRad_ADC_Std"]             = RNG.normal(105.0,   35.0, n_early).clip(20.0, 260.0).round(2)
    df_out.loc[early_idx, "PyRad_Entropy"]             = RNG.normal(4.8,      1.0, n_early).clip(2.5,  6.5).round(4)
    df_out.loc[early_idx, "PyRad_GLCM_Contrast"]       = RNG.normal(22.0,     8.0, n_early).clip(8.0, 50.0).round(4)
    df_out.loc[early_idx, "PyRad_GLCM_Homogeneity"]    = RNG.normal(0.60,    0.11, n_early).clip(0.35, 0.90).round(4)
    df_out.loc[early_idx, "PyRad_Shape_Sphericity"]    = RNG.normal(0.75,    0.09, n_early).clip(0.50, 1.00).round(4)
    df_out.loc[early_idx, "PyRad_FirstOrder_Skewness"] = RNG.normal(0.04,    0.38, n_early).round(4)

    # B. Casos inflamatorios (~12% de sanos con perfil radiómico parecido al maligno)
    # EII activa o diverticulitis puede elevar el CEA y restringir el ADC (ESGAR 2022)
    h_idx      = df_out.index[df_out["Diagnosis"] == 0].to_numpy()
    n_inflam   = max(1, int(len(h_idx) * 0.12))
    inflam_idx = RNG.choice(h_idx, size=n_inflam, replace=False)

    df_out.loc[inflam_idx, "CEA_Level_ng_mL"]          = RNG.lognormal(np.log(12.0), 0.55, n_inflam).clip(8.0, 60.0).round(2)
    df_out.loc[inflam_idx, "PyRad_ADC_Mean"]           = RNG.normal(1180.0, 220.0, n_inflam).clip(700.0, 1700.0).round(2)
    df_out.loc[inflam_idx, "PyRad_ADC_Std"]            = RNG.normal(170.0,   55.0, n_inflam).clip(40.0, 380.0).round(2)
    df_out.loc[inflam_idx, "PyRad_Entropy"]            = RNG.normal(6.3,      1.0, n_inflam).clip(4.5, 9.5).round(4)
    df_out.loc[inflam_idx, "PyRad_GLCM_Contrast"]      = RNG.normal(48.0,    14.0, n_inflam).clip(22.0, 120.0).round(4)
    df_out.loc[inflam_idx, "PyRad_GLCM_Homogeneity"]   = RNG.normal(0.32,    0.10, n_inflam).clip(0.08, 0.52).round(4)
    df_out.loc[inflam_idx, "PyRad_Shape_Sphericity"]   = RNG.normal(0.60,    0.12, n_inflam).clip(0.25, 0.88).round(4)
    df_out.loc[inflam_idx, "PyRad_FirstOrder_Skewness"] = RNG.normal(0.38,   0.45, n_inflam).round(4)

    # Ruido biológico: 18% de los sanos reciben perturbaciones que simulan
    # variabilidad real de laboratorio e imagen (artefactos, diferencias de equipo...)
    sanos_idx = df_out.index[df_out["Diagnosis"] == 0].to_numpy()
    n_noise   = max(1, int(len(sanos_idx) * 0.18))
    noisy_idx = RNG.choice(sanos_idx, size=n_noise, replace=False)
    q = n_noise // 4
    idx_a = noisy_idx[:q]
    idx_b = noisy_idx[q:2*q]
    idx_c = noisy_idx[2*q:3*q]
    idx_d = noisy_idx[3*q:]

    # Pico de CEA benigno (tabaco activo o proceso inflamatorio agudo)
    df_out.loc[idx_a, "CEA_Level_ng_mL"] = (
        RNG.lognormal(np.log(8.5), 0.60, len(idx_a)).clip(5.0, 40.0).round(2)
    )
    # Anemia leve no oncológica (ferropénica o por déficit de B12)
    df_out.loc[idx_b, "Hemoglobin_g_dL"] = (
        RNG.normal(11.5, 1.10, len(idx_b)).clip(8.5, 13.5).round(2)
    )
    # Variabilidad del ADC por diferencias de equipo o protocolo de adquisición
    adc_c = df_out.loc[idx_c, "PyRad_ADC_Mean"].to_numpy()
    df_out.loc[idx_c, "PyRad_ADC_Mean"] = (
        (adc_c * RNG.lognormal(0.0, 0.15, len(idx_c))).clip(800.0, 2500.0).round(2)
    )
    # Artefactos de textura en la imagen (movimiento, distorsión B0)
    df_out.loc[idx_d, "PyRad_Entropy"] = (
        (df_out.loc[idx_d, "PyRad_Entropy"] + RNG.normal(1.20, 0.45, len(idx_d))).clip(0.1, 10.0).round(4)
    )
    df_out.loc[idx_d, "PyRad_GLCM_Contrast"] = (
        (df_out.loc[idx_d, "PyRad_GLCM_Contrast"] + RNG.normal(10.0, 4.0, len(idx_d))).clip(0.1, 120.0).round(4)
    )

    # Target noise: invertimos la etiqueta de un 12% de pacientes al azar.
    # Esto simula casos biológicamente ambiguos y fija un techo teórico de AUC (~0.88).
    np.random.seed(42)
    swap_mask = np.random.rand(len(df_out)) < 0.12
    df_out.loc[swap_mask, "Diagnosis"] = 1 - df_out.loc[swap_mask, "Diagnosis"]

    print(f"Dataset generado: {len(df_out):,} pacientes | "
          f"{int(df_out['Diagnosis'].sum()):,} cancer / "
          f"{int((df_out['Diagnosis'] == 0).sum()):,} sanos")
    return df_out


def validar_distribuciones(df: pd.DataFrame) -> None:
    """
    Imprime media ± std por grupo diagnóstico para comprobar que los valores
    generados son clínicamente plausibles.
    """
    features = [
        "CEA_Level_ng_mL", "Hemoglobin_g_dL", "PyRad_ADC_Mean", "PyRad_ADC_Std",
        "PyRad_Entropy", "PyRad_GLCM_Contrast", "PyRad_GLCM_Homogeneity",
        "PyRad_Shape_Sphericity", "PyRad_FirstOrder_Skewness",
    ]
    print("\n" + "=" * 80)
    print("  VALIDACION -- Media +/- Std por grupo diagnostico")
    print("=" * 80)
    for col in features:
        c = df[df["Diagnosis"] == 1][col]
        h = df[df["Diagnosis"] == 0][col]
        print(f"  {col:<35}  Cancer: {c.mean():>8.2f} +/- {c.std():.2f}  |  "
              f"Sano: {h.mean():>8.2f} +/- {h.std():.2f}")
    print("=" * 80)


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":

    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    CSV_BASE   = os.path.join(BASE_DIR, "..", "Data", "raw", "colorectal_cancer_full_dataset_v1.csv")
    CSV_OUTPUT = os.path.join(BASE_DIR, "..", "Data", "processed", "dataset_clinico_tumoral.csv")

    print(f"Cargando dataset base: {CSV_BASE}")
    df_companion = pd.read_csv(CSV_BASE)
    print(f"Registros cargados: {len(df_companion):,}")

    df_clinical = generar_features_clinicas(df_companion)
    validar_distribuciones(df_clinical)

    df_clinical.to_csv(CSV_OUTPUT, index=False)
    print(f"\nDataset clinico guardado en: {CSV_OUTPUT}")
    print(f"Shape final: {df_clinical.shape}")
    print(df_clinical.head(3).to_string())
