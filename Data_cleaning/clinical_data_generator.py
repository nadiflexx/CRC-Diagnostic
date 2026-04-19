"""
clinical_data_generator.py

Genera un dataset clínico sintético de cáncer colorrectal (CRC) con
biomarcadores hematológicos y features radiómicas tipo PyRadiomics.

Toma el CSV base (v1) y para cada paciente sintetiza valores clínicos
mediante distribuciones multivariantes calibradas con los rangos de referencia
de NCCN 2023, ESGAR 2022 y Duffy et al. 2021.

Uso:
    python Data_cleaning/clinical_data_generator.py
"""

import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ── Parámetros de la distribución multivariante por grupo diagnóstico ────────
# Variables base modeladas: [log(CEA), Hemoglobina, ADC_medio, Entropía, Contraste]
# Se usa log(CEA) porque la distribución real del CEA es log-normal.

CANCER_MEANS = np.array([np.log(7.0), 11.40, 1240.0, 6.10, 33.0])
# Stds amplios: el cáncer es biológicamente heterogéneo (distinto estadio, vascularización, necrosis).
# Valores estrechos aquí producen separabilidad artificial y probabilidades extremas.
CANCER_STDS  = np.array([1.55,         2.80,  360.0,  2.10, 20.0])

# Correlaciones clínicas (cáncer): CEA↑ → Hgb↓ anemia; ADC↓ difusión restringida → CEA↑ y entropía↑.
CANCER_CORR = np.array([
    [ 1.00, -0.30, -0.22,  0.32,  0.25],   # log(CEA)
    [-0.30,  1.00,  0.28, -0.25, -0.18],   # Hemoglobina
    [-0.22,  0.28,  1.00, -0.38, -0.30],   # ADC medio
    [ 0.32, -0.25, -0.38,  1.00,  0.48],   # Entropia GLCM
    [ 0.25, -0.18, -0.30,  0.48,  1.00],   # Contraste GLCM
])

# Correlaciones (sano): mucho más débiles al no existir sinergia fisiopatológica tumoral.
HEALTHY_MEANS = np.array([np.log(2.10), 13.50, 1540.0, 4.80, 21.0])
# Stds sanos también ampliados: variabilidad inter-individuo e inter-equipo real.
HEALTHY_STDS  = np.array([0.85,          2.10,  260.0,  1.35,  9.5])

HEALTHY_CORR = np.array([
    [ 1.00, -0.08, -0.06,  0.12,  0.09],   # log(CEA)
    [-0.08,  1.00,  0.16, -0.07, -0.05],   # Hemoglobina
    [-0.06,  0.16,  1.00, -0.22, -0.16],   # ADC medio
    [ 0.12, -0.07, -0.22,  1.00,  0.32],   # Entropia GLCM
    [ 0.09, -0.05, -0.16,  0.32,  1.00],   # Contraste GLCM
])

# ── Parámetros por estadio tumoral (T1–T4/M1) ────────────────────────────────
# Columnas: [log(CEA)_mu, Hgb_mu, ADC_mu, Entropía_mu, Contraste_mu]
#           [log(CEA)_sg, Hgb_sg, ADC_sg, Entropía_sg, Contraste_sg]
#            hom_base, sph_mu, skew_mu
# Calibrado con: Gollub 2018, Lambregts 2013, Horvat 2019, NCCN 2023.
STAGE_PARAMS = {
    # T1 — confinado a mucosa/submucosa. En clínica real el T1 temprano es casi
    # indistinguible de tejido benigno por biomarcadores séricos: CEA normal o
    # marginalmente elevado, ADC levemente restringido solo en lesiones >1 cm.
    # Solapamiento intencionado con sanos para forzar probabilidades 30-60%.
    1: ([np.log(1.9),  13.4, 1430., 4.92, 22.], [0.72, 2.00, 280., 1.35, 10.0], 0.67, 0.72, 0.14),
    # T2 — invade muscular propia: señal moderada; sigue habiendo solapamiento
    # considerable con sanos inflamados y variabilidad individual alta.
    2: ([np.log(5.0),  12.2, 1120., 5.70, 28.], [0.88, 2.00, 280., 1.45, 12.0], 0.50, 0.67, 0.48),
    # T3 — penetra subserosa: CEA elevado, ADC claramente restringido, anemia.
    3: ([np.log(18.0), 10.3,  840., 6.60, 41.], [1.00, 2.20, 280., 1.40, 14.0], 0.32, 0.56, 0.90),
    # T4/M1 — perforación o metástasis: CEA muy alto, ADC muy bajo, tumor necrótico.
    4: ([np.log(90.0),  8.0,  650., 7.60, 56.], [1.20, 1.80, 230., 1.45, 17.0], 0.18, 0.43, 1.35),
}


def corr_a_cov(corr: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Convierte una matriz de correlación y desviaciones estándar en una matriz de covarianza.

    Aplica la fórmula Σ = D · R · D, donde D = diag(stds).

    Args:
        corr: Matriz de correlación de forma (n, n).
        stds: Vector de desviaciones estándar de longitud n.

    Returns:
        Matriz de covarianza de forma (n, n).
    """
    D = np.diag(stds)
    return D @ corr @ D


def generar_features_clinicas(df_base: pd.DataFrame) -> pd.DataFrame:
    """Genera biomarcadores hematológicos y features radiómicas para cada paciente.

    Requiere como mínimo las columnas ``Age`` y ``Diagnosis``. Si existen
    ``Gender``, ``Smoking_History``, ``Inflammatory_Bowel_Disease`` y
    ``Genetic_Mutation``, las usa para ajustar los valores generados.
    Inyecta un 12% de ruido biológico cruzado para evitar separabilidad perfecta.

    Args:
        df_base: DataFrame base con al menos las columnas ``Age`` y ``Diagnosis``.

    Returns:
        DataFrame con 13 columnas: ``Patient_ID``, 11 features clínicas/radiómicas
        y ``Diagnosis``.
    """
    df = df_base.copy()
    n  = len(df)

    if "Patient_ID" not in df.columns:
        df.insert(0, "Patient_ID", [f"PT-{i:05d}" for i in range(n)])

    ages    = df["Age"].to_numpy(float)
    gender  = df.get("Gender",                     pd.Series(np.ones(n,  int))).to_numpy(int)
    smoking = df.get("Smoking_History",            pd.Series(np.zeros(n, int))).to_numpy(int)
    ibd     = df.get("Inflammatory_Bowel_Disease", pd.Series(np.zeros(n, int))).to_numpy(int)
    genetic = df.get("Genetic_Mutation",           pd.Series(np.zeros(n, int))).to_numpy(int)
    diag    = df["Diagnosis"].to_numpy(int)

    cov_cancer  = corr_a_cov(CANCER_CORR,  CANCER_STDS)
    cov_healthy = corr_a_cov(HEALTHY_CORR, HEALTHY_STDS)

    cea      = np.zeros(n)
    hgb      = np.zeros(n)
    adc_mean = np.zeros(n)
    entropy  = np.zeros(n)
    contrast = np.zeros(n)

    # Grupo cáncer — estadificado T1→T4/M1
    cancer_idx = np.where(diag == 1)[0]
    n_c = len(cancer_idx)

    # Arrays de radiomics derivadas stage-aware (sólo para cáncer)
    hom_base_c = np.zeros(n_c)
    sph_mu_c   = np.zeros(n_c)
    skew_mu_c  = np.zeros(n_c)

    if n_c > 0:
        # Residuos correlacionados estándar: preservan estructura CANCER_CORR por estadio.
        L_cancer   = np.linalg.cholesky(CANCER_CORR)
        z_c        = RNG.standard_normal((n_c, 5))
        corr_res_c = z_c @ L_cancer.T          # (n_c, 5), media≈0, std≈1 por columna

        # Estadificación: 18% T1 · 27% T2 · 30% T3 · 25% T4/M1
        stages = RNG.choice([1, 2, 3, 4], size=n_c, p=[0.18, 0.27, 0.30, 0.25])

        for stage, (means_s, stds_s, hom_b, sph_m, skew_m) in STAGE_PARAMS.items():
            mask = stages == stage
            if not mask.any():
                continue
            r      = corr_res_c[mask]
            ages_s = ages[cancer_idx[mask]]
            exceso = np.maximum(ages_s - 70, 0.0)
            d_cea  = np.where(ages_s > 70, np.minimum(0.30 + 0.012 * exceso, 0.70), 0.0)
            d_hgb  = np.where(ages_s > 70, np.maximum(-0.80 - 0.04 * exceso, -2.00), 0.0)
            g_off  = np.where(gender[cancer_idx[mask]] == 1, 0.0, -1.5)

            cea[cancer_idx[mask]]      = np.exp(means_s[0] + stds_s[0] * r[:, 0] + d_cea)
            hgb[cancer_idx[mask]]      = means_s[1] + stds_s[1] * r[:, 1] + d_hgb + g_off
            adc_mean[cancer_idx[mask]] = means_s[2] + stds_s[2] * r[:, 2]
            entropy[cancer_idx[mask]]  = means_s[3] + stds_s[3] * r[:, 3]
            contrast[cancer_idx[mask]] = means_s[4] + stds_s[4] * r[:, 4]
            hom_base_c[mask] = hom_b
            sph_mu_c[mask]   = sph_m
            skew_mu_c[mask]  = skew_m

    # Grupo sano
    healthy_idx = np.where(diag == 0)[0]
    n_h = len(healthy_idx)
    if n_h > 0:
        # Tabaco, EII y mutaciones genéticas elevan el CEA basal en sanos (Duffy et al. 2021).
        cea_log_base = (
            HEALTHY_MEANS[0]
            + 0.85 * smoking[healthy_idx]
            + 0.30 * ibd[healthy_idx]
            + 0.20 * genetic[healthy_idx]
        )
        means_cero = HEALTHY_MEANS.copy()
        means_cero[0] = 0.0
        samples_h = RNG.multivariate_normal(means_cero, cov_healthy, size=n_h)

        offset_genero_h = np.where(gender[healthy_idx] == 1, 0.0, -1.5)

        cea[healthy_idx]      = np.exp(samples_h[:, 0] + cea_log_base)
        hgb[healthy_idx]      = samples_h[:, 1] + offset_genero_h
        adc_mean[healthy_idx] = samples_h[:, 2]
        entropy[healthy_idx]  = samples_h[:, 3]
        contrast[healthy_idx] = samples_h[:, 4]

    # ── Features radiómicas derivadas ─────────────────────────────────────────
    # ADC_Std: lognormal independiente del nivel ADC; cubre rangos reales inter-equipo.
    # Tumor (mediana ~200 µm²/s, p5~68, p95~480→clip): necrosis e hipoxia intra-tumoral.
    # Sano  (mediana ~82  µm²/s, p5~30, p95~195):      tejido uniforme, mínima dispersión.
    adc_std = np.where(
        diag == 1,
        np.clip(RNG.lognormal(np.log(200), 0.55, n), 5.0, 400.0),
        np.clip(RNG.lognormal(np.log(82),  0.48, n), 5.0, 400.0),
    )

    # Homogeneidad: gradiente T1(0.60)→T4(0.18) para cáncer; 0.72 para sano.
    hom_base_full = np.full(n, 0.72)
    hom_base_full[cancer_idx] = hom_base_c

    coef_c   = np.where(diag == 1, 0.003, 0.002)
    coef_e   = np.where(diag == 1, 0.016, 0.007)
    umbral_c = np.where(diag == 1, 30.0, 12.0)
    umbral_e = np.where(diag == 1, 5.0,   4.0)
    hom_raw = (
        RNG.normal(hom_base_full, 0.10, n)
        - coef_c * np.maximum(contrast - umbral_c, 0)
        - coef_e * np.maximum(entropy  - umbral_e, 0)
    )

    # Esfericidad: gradiente T1(regular)→T4(muy irregular); 0.73 para sano.
    sph_mu_full = np.full(n, 0.73)
    sph_mu_full[cancer_idx] = sph_mu_c
    sph_raw = RNG.normal(sph_mu_full, 0.12, n)

    # Asimetría: gradiente T1(leve)→T4(extrema por necrosis masiva); ~0 para sano.
    skew_mu_full = np.full(n, 0.05)
    skew_mu_full[cancer_idx] = skew_mu_c
    skew_raw = RNG.normal(skew_mu_full, 0.42, n)

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

    # ── Ruido biológico — zona gris clínica ────────────────────────────────────
    # Los estadios T1 tempranos están modelados explícitamente en el bloque de
    # estadificación (18% del grupo cáncer): no se sobreescriben aquí.

    # Inflamación severa multi-feature (~14% de sanos): EII activa, diverticulitis o
    # apendicitis eleva CEA, baja Hgb y restringe ADC simultáneamente (ESGAR 2022).
    # Al afectar MÚLTIPLES features a la vez, crea casos genuinamente ambiguos
    # que el modelo no puede resolver con certeza → probabilidades 25-55%.
    h_idx      = df_out.index[df_out["Diagnosis"] == 0].to_numpy()
    n_inflam   = max(1, int(len(h_idx) * 0.14))
    inflam_idx = RNG.choice(h_idx, size=n_inflam, replace=False)

    df_out.loc[inflam_idx, "CEA_Level_ng_mL"]          = RNG.lognormal(np.log(11.0), 0.60, n_inflam).clip(5.0, 55.0).round(2)
    df_out.loc[inflam_idx, "Hemoglobin_g_dL"]          = RNG.normal(11.2, 1.50, n_inflam).clip(7.5, 13.8).round(2)
    df_out.loc[inflam_idx, "PyRad_ADC_Mean"]           = RNG.normal(1200.0, 240.0, n_inflam).clip(700.0, 1750.0).round(2)
    df_out.loc[inflam_idx, "PyRad_ADC_Std"]            = RNG.normal(175.0,   60.0, n_inflam).clip(40.0, 380.0).round(2)
    df_out.loc[inflam_idx, "PyRad_Entropy"]            = RNG.normal(6.1,      1.1, n_inflam).clip(4.0, 9.5).round(4)
    df_out.loc[inflam_idx, "PyRad_GLCM_Contrast"]      = RNG.normal(46.0,    15.0, n_inflam).clip(20.0, 120.0).round(4)
    df_out.loc[inflam_idx, "PyRad_GLCM_Homogeneity"]   = RNG.normal(0.34,    0.11, n_inflam).clip(0.08, 0.56).round(4)
    df_out.loc[inflam_idx, "PyRad_Shape_Sphericity"]   = RNG.normal(0.61,    0.13, n_inflam).clip(0.25, 0.88).round(4)
    df_out.loc[inflam_idx, "PyRad_FirstOrder_Skewness"] = RNG.normal(0.35,   0.48, n_inflam).round(4)

    # Variabilidad de laboratorio e imagen (18% de sanos): artefactos y diferencias de equipo.
    sanos_idx = df_out.index[df_out["Diagnosis"] == 0].to_numpy()
    n_noise   = max(1, int(len(sanos_idx) * 0.18))
    noisy_idx = RNG.choice(sanos_idx, size=n_noise, replace=False)
    q = n_noise // 4
    idx_a = noisy_idx[:q]
    idx_b = noisy_idx[q:2*q]
    idx_c = noisy_idx[2*q:3*q]
    idx_d = noisy_idx[3*q:]

    df_out.loc[idx_a, "CEA_Level_ng_mL"] = (  # pico CEA benigno: tabaco activo o inflamación aguda
        RNG.lognormal(np.log(8.5), 0.60, len(idx_a)).clip(5.0, 40.0).round(2)
    )
    df_out.loc[idx_b, "Hemoglobin_g_dL"] = (  # anemia leve no oncológica (ferropénica o déficit B12)
        RNG.normal(11.5, 1.10, len(idx_b)).clip(8.5, 13.5).round(2)
    )
    adc_c = df_out.loc[idx_c, "PyRad_ADC_Mean"].to_numpy()  # variabilidad ADC por protocolo de adquisición
    df_out.loc[idx_c, "PyRad_ADC_Mean"] = (
        (adc_c * RNG.lognormal(0.0, 0.15, len(idx_c))).clip(800.0, 2500.0).round(2)
    )
    df_out.loc[idx_d, "PyRad_Entropy"] = (  # artefactos de textura: movimiento o distorsión B0
        (df_out.loc[idx_d, "PyRad_Entropy"] + RNG.normal(1.20, 0.45, len(idx_d))).clip(0.1, 10.0).round(4)
    )
    df_out.loc[idx_d, "PyRad_GLCM_Contrast"] = (
        (df_out.loc[idx_d, "PyRad_GLCM_Contrast"] + RNG.normal(10.0, 4.0, len(idx_d))).clip(0.1, 120.0).round(4)
    )

    print(f"Dataset generado: {len(df_out):,} pacientes | "
          f"{int(df_out['Diagnosis'].sum()):,} cancer / "
          f"{int((df_out['Diagnosis'] == 0).sum()):,} sanos")
    return df_out


def validar_distribuciones(df: pd.DataFrame) -> None:
    """Imprime media ± std por grupo diagnóstico para auditar plausibilidad clínica.

    Args:
        df: DataFrame generado por ``generar_features_clinicas``.

    Returns:
        None
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
