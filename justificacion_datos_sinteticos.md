# Justificación del Dataset Sintético: `clinical_data_generator.py`

> **Audiencia:** Comité de defensa de TFM, perfil técnico-médico.  
> **Contexto:** Script `Data_cleaning/clinical_data_generator.py`, generador del dataset `dataset_clinico_tumoral.csv`.

---

## 1. Por qué datos sintéticos y no un dataset público

La pregunta más obvia que hará el tribunal es: *"¿Por qué no usas un dataset real de Kaggle o del TCGA?"*

La respuesta tiene tres niveles:

**a) Disponibilidad real de datos radiómicos integrados con clínica:** Los datasets públicos de CRC (TCGA-COAD, NLST) contienen o bien imagen bruta sin features radiómicas extraídas, o bien datos clínicos sin radiología cuantitativa. Un dataset que combine simultáneamente biomarcadores séricos (CEA, Hgb) con features de PyRadiomics (ADC, entropía GLCM, esfericidad) y etiquetas diagnósticas limpias **no existe públicamente** a la fecha de este trabajo.

**b) Control del escenario de experimentación:** Para comparar modelos de forma justa y reproducible, el dataset debe tener propiedades conocidas y controladas: tamaño, balance de clases, correlaciones entre features. Un dataset real tiene sesgos de institución, protocolos de adquisición heterogéneos y valores faltantes no informativos que introducen varianza no atribuible al modelo.

**c) Privacidad y regulación:** Los datos de imagen oncológica están sujetos al RGPD y a acuerdos de uso de datos institucionales. Para un TFM académico sin convenio con un hospital, la síntesis controlada es la única vía legalmente limpia.

---

## 2. Los datos no son aleatorios: distribuciones ancladas en literatura clínica

La generación no parte de valores arbitrarios. Cada distribución está **calibrada sobre referencias clínicas publicadas**.

### 2.1 Biomarcadores séricos

| Feature | Referencia | Decisión de diseño |
|---|---|---|
| **CEA** | CEA log-normal: Duffy et al., *BMJ* 2021; NCCN Guidelines v2.2023 | Usamos `log(CEA)` con distr. normal, exponenciamos al final → distribución log-normal fidedigna |
| **Hemoglobina** | WHO Anaemia thresholds 2011; offset –1.5 g/dL en mujeres | Diferencial de sexo implementado (`offset_genero`) |
| **Ajuste por edad >70** | NCCN 2023: incremento de CEA basal y descenso de Hgb en mayores | `delta_cea_edad` y `delta_hgb_edad` con crecimiento lineal tras 70 años |
| **CEA en sanos fumadores** | Duffy et al. 2021: tabaco eleva CEA basal ~0.85 unidades log | `0.85 * smoking[idx]` en el offset de log(CEA) para controles |

### 2.2 Features radiómicas

| Feature | Referencia | Decisión de diseño |
|---|---|---|
| **ADC medio** | ESGAR Consensus 2022: ADC tumoral 800–1400 µm²/s vs. 1400–1800 µm²/s normal | Medias por estadio T1–T4; correlación negativa con CEA |
| **ADC_Std** | Lambregts 2013: dispersión intra-tumoral lognormal por necrosis e hipoxia | `lognormal(log(200), 0.55)` tumor vs. `lognormal(log(82), 0.48)` sano |
| **Entropía GLCM** | Davnall et al., *Insights Imaging* 2012; Hatt et al., *JNM* 2017 | Mayor entropía en tejido maligno por heterogeneidad de microarquitectura |
| **Contraste GLCM** | Gillies et al., *Cell* 2016 | Correlación positiva con entropía (r ≈ 0.48) en matriz de covarianza |
| **Esfericidad** | IBSI Standard v1.0: Shape_Sphericity | Gradiente T1(0.72)→T4(0.43) vs. sano(0.73) |

### 2.3 La matriz de covarianza: correlaciones con sentido fisiopatológico

El núcleo del generador no son distribuciones marginales independientes. Es una **distribución multivariante** con correlaciones basadas en la biología tumoral:

```
CANCER_CORR:
              log(CEA)   Hgb     ADC    Entropia  Contraste
log(CEA)    [  1.00,   -0.30,  -0.22,   0.32,    0.25  ]
Hgb         [ -0.30,    1.00,   0.28,  -0.25,   -0.18  ]
ADC         [ -0.22,    0.28,   1.00,  -0.38,   -0.30  ]
```

- **CEA ↑ → Hgb ↓ (r = -0.30):** Los tumores con alta carga antigénica causan pérdida oculta crónica → anemia ferropénica secundaria.
- **ADC ↓ → CEA ↑ (r = -0.22):** Difusión restringida indica alta celularidad → mayor secreción de CEA.
- **ADC ↓ → Entropía ↑ (r = -0.38):** Tejido compacto y heterogéneo al mismo tiempo (microarquitectura glandular desordenada).

En los controles sanos, las correlaciones son deliberadamente débiles (r < 0.20) porque sin tumor no existe el mecanismo fisiopatológico que acopla estas variables.

---

## 3. Estadificación T1–T4: el solapamiento clínico como característica de diseño

El solapamiento entre grupos no se introduce como ruido artificial: se modela desde la fisiopatología de cada estadio tumoral (Gollub 2018, Horvat 2019, NCCN 2023).

### 3.1 Distribución de estadios

| Estadio | Proporción | Descripción clínica |
|---|---|---|
| T1 (18 %) | Confinado a mucosa/submucosa | CEA casi normal (µ=1.9 ng/mL), ADC levemente restringido. Clínicamente casi indistinguible de tejido benigno. |
| T2 (27 %) | Invade muscular propia | CEA moderado (µ=5 ng/mL), ADC moderadamente restringido. Solapamiento considerable con sanos inflamados. |
| T3 (30 %) | Penetra subserosa | CEA elevado (µ=18 ng/mL), ADC claramente restringido, anemia establecida. Señal clara. |
| T4/M1 (25 %) | Perforación o metástasis | CEA muy alto (µ=90 ng/mL), ADC muy bajo, tumor necrótico. Inequívoco. |

### 3.2 Por qué el T1 debe solapar con sanos

Un 15–20 % de los CRC en estadio I presentan **CEA < 3 ng/mL y hemoglobina preservada** (NCCN 2023). El tumor existe, pero aún no ha generado la señal serológica ni la alteración radiológica suficiente. Forzar que todos los cancerosos tengan CEA alto sería biológicamente incorrecto y produciría un modelo sobreconfiado que fallará en detección temprana.

Por eso el T1 está parametrizado con medias próximas a los sanos y desviaciones estándar amplias:

```python
# STAGE_PARAMS en clinical_data_generator.py
1: ([np.log(1.9), 13.4, 1430., 4.92, 22.], [0.72, 2.00, 280., 1.35, 10.0], ...)
```

Esta decisión de diseño hace que el modelo aprenda a decir "57 %" en un T1 real, no "95 %". Eso es clínicamente correcto: un T1 temprano **requiere biopsia confirmatoria**, no certeza algoritmica.

### 3.3 Las desviaciones estándar amplias: heterogeneidad biológica real

El cáncer colorrectal no es una enfermedad uniforme. La variabilidad entre tumores (vascularización, necrosis, microentorno) produce rangos amplios en todos los biomarcadores:

| Parámetro | STD elegido | Justificación |
|---|---|---|
| log(CEA) cáncer | 1.55 | Variabilidad entre estadios y subtipos moleculares (KRAS/BRAF) |
| ADC medio cáncer | 360 µm²/s | Diferencias inter-equipo 1.5T vs 3T + variabilidad necrosis |
| Hemoglobina sanos | 2.10 g/dL | Variabilidad individual + diferencia de sexo real |

STDs estrechos (como en el diseño original, 1.10 / 270 / 1.80) producen separabilidad artificial: XGBoost alcanza AUC=0.992 pero los logits se saturan y el 82 % de los pacientes recibe probabilidad <5 % o >95 %. No es un modelo más potente; es un problema más fácil.

---

## 4. Sanos con inflamación severa: zona gris genuinamente ambigua

El 14 % de los sanos recibe una perturbación **multi-feature simultánea** que simula EII activa, diverticulitis o apendicitis (ESGAR 2022):

```python
# clinical_data_generator.py — bloque de inflamación
df_out.loc[inflam_idx, "CEA_Level_ng_mL"]  = lognormal(log(11.0), 0.60).clip(5, 55)
df_out.loc[inflam_idx, "Hemoglobin_g_dL"]  = normal(11.2, 1.50).clip(7.5, 13.8)
df_out.loc[inflam_idx, "PyRad_ADC_Mean"]   = normal(1200, 240).clip(700, 1750)
df_out.loc[inflam_idx, "PyRad_Entropy"]    = normal(6.1, 1.1).clip(4.0, 9.5)
```

La clave es que **varias features se elevan simultáneamente**. Un modelo que solo vea CEA alto puede ignorarlo. Un modelo que vea CEA=12, Hgb=11, ADC=1200, Entropía=6.1 simultáneamente — exactamente el perfil de un T2 temprano — genuinamente no puede resolverlo con certeza. Eso es lo que debe pasar: esos pacientes deben recibir probabilidades del 40–60 %, no del 5 % ni del 95 %.

---

## 5. Ruido de laboratorio e imagen: variabilidad instrumental (18 % de los sanos)

Además de la inflamación severa, el 18 % de los sanos recibe perturbaciones de cuatro tipos que simulan variabilidad en la cadena de adquisición:

| Tipo | Causa real simulada |
|---|---|
| Pico de CEA benigno | Tabaco activo, infección aguda, proceso autoinmune |
| Anemia leve | Ferropenia no oncológica, déficit de B12 |
| Variabilidad del ADC (×lognormal) | Diferencias de protocolo b-value entre escáneres 1.5T y 3T |
| Artefactos de textura | Movimiento respiratorio, distorsión de campo B0 en DW-MRI |

Esta capa de ruido **no invierte etiquetas**; solo perturba las features. Su función es asegurar que el modelo no aprenda diferencias sutiles de escáner o de laboratorio que no generalizarán en producción.

---

## 6. Reproducibilidad y auditabilidad

```python
RNG = np.random.default_rng(42)   # generador único — todo el script usa este RNG
```

El dataset es **100% reproducible** con esta semilla fija. Cualquier revisor puede re-ejecutar `clinical_data_generator.py` y obtener exactamente el mismo CSV. Esto satisface el criterio de reproducibilidad de la ESMO/ESGAR para estudios de validación de biomarcadores computacionales.

---

## 7. Resumen de la cadena de validación

```
Guías clínicas               Estadificación T1–T4          Dataset sintético
(NCCN, ESGAR, Gollub,    →   con STDs amplios          →   con solapamiento real
Lambregts, Horvat)           y correlaciones                T1≈sano, T4≠sano
                             fisiopatológicas                     ↓
                                                    Modelo con AUC=0.974 y
                                                    probabilidades 5–95%
                                                    clínicamente interpretables
```

El dataset no es realista por azar. Es realista porque fue **diseñado para serlo**, con cada parámetro justificado en una fuente clínica y cada capa de variabilidad modelando un fenómeno biológico o instrumental documentado.

---

## 1. Por qué datos sintéticos y no un dataset público

La pregunta más obvia que hará el tribunal es: *"¿Por qué no usas un dataset real de Kaggle o del TCGA?"*

La respuesta tiene tres niveles:

**a) Disponibilidad real de datos radiómicos integrados con clínica:** Los datasets públicos de CRC (TCGA-COAD, NLST) contienen o bien imagen bruta sin features radiómicas extraídas, o bien datos clínicos sin radiología cuantitativa. Un dataset que combine simultáneamente biomarcadores séricos (CEA, Hgb) con features de PyRadiomics (ADC, entropía GLCM, esfericidad) y etiquetas diagnósticas limpias **no existe públicamente** a la fecha de este trabajo.

**b) Control del escenario de experimentación:** Para comparar modelos de forma justa y reproducible, el dataset debe tener propiedades conocidas y controladas: tamaño, balance de clases, correlaciones entre features. Un dataset real tiene sesgos de institución, protocolos de adquisición heterogéneos y valores faltantes no informativos que introducen varianza no atribuible al modelo.

**c) Privacidad y regulación:** Los datos de imagen oncológica están sujetos al RGPD y a acuerdos de uso de datos institucionales. Para un TFM académico sin convenio con un hospital, la síntesis controlada es la única vía legalmente limpia.

---

## 2. Los datos no son aleatorios: distribuciones ancladas en literatura clínica

La generación no parte de valores arbitrarios. Cada distribución está **calibrada sobre referencias clínicas publicadas**.

### 2.1 Biomarcadores séricos

| Feature | Referencia | Decisión de diseño |
|---|---|---|
| **CEA** | CEA log-normal: Duffy et al., *BMJ* 2021; NCCN Guidelines v2.2023 | Usamos `log(CEA)` con distr. normal, exponenciamos al final → distribución log-normal fidedigna |
| **Hemoglobina** | WHO Anaemia thresholds 2011; offset –1.5 g/dL en mujeres | Diferencial de sexo implementado (`offset_genero`) |
| **Ajuste por edad >70** | NCCN 2023: incremento de CEA basal y descenso de Hgb en mayores | `delta_cea_edad` y `delta_hgb_edad` con crecimiento lineal tras 70 años |
| **CEA en sanos fumadores** | Duffy et al. 2021: tabaco eleva CEA basal ~0.85 unidades log | `0.85 * smoking[idx]` en el offset de log(CEA) para controles |

### 2.2 Features radiómicas

| Feature | Referencia | Decisión de diseño |
|---|---|---|
| **ADC medio** | ESGAR Consensus 2022: ADC tumoral 800–1400 µm²/s vs. 1400–1800 µm²/s normal | Medias 1240 vs. 1540 µm²/s; correlación negativa con CEA (tejido más celular = más difusión restringida) |
| **Entropía GLCM** | Davnall et al., *Insights Imaging* 2012; Hatt et al., *JNM* 2017 | Mayor entropía en tejido maligno por heterogeneidad de microarquitectura |
| **Contraste GLCM** | Gillies et al., *Cell* 2016 | Correlación positiva con entropía (r ≈ 0.48) implementada en la matriz de covarianza |
| **Esfericidad** | IBSI Standard v1.0: Shape_Sphericity | Tumores más irregulares (µ=0.62) vs. segmentos anatómicos normales (µ=0.73) |

### 2.3 La matriz de covarianza: correlaciones con sentido fisiopatológico

El núcleo del generador no son distribuciones marginales independientes. Es una **distribución multivariante** con correlaciones basadas en la biología tumoral:

```
CANCER_CORR:
              log(CEA)   Hgb     ADC    Entropia  Contraste
log(CEA)    [  1.00,   -0.30,  -0.22,   0.32,    0.25  ]
Hgb         [ -0.30,    1.00,   0.28,  -0.25,   -0.18  ]
ADC         [ -0.22,    0.28,   1.00,  -0.38,   -0.30  ]
```

- **CEA ↑ → Hgb ↓ (r = -0.30):** Los tumores con alta carga antigénica causan pérdida oculta crónica → anemia ferropénica secundaria.
- **ADC ↓ → CEA ↑ (r = -0.22):** Difusión restringida indica alta celularidad → mayor secreción de CEA.
- **ADC ↓ → Entropía ↑ (r = -0.38):** Tejido compacto y heterogéneo al mismo tiempo (microarquitectura glandular desordenada).

En los controles sanos, las correlaciones son deliberadamente débiles (r < 0.20) porque sin tumor no existe el mecanismo fisiopatológico que acopla estas variables.

---

## 3. El ruido deliberado: por qué el 12% de Target Noise no es un error

### 3.1 El problema del dataset "perfecto"

Si generáramos un dataset donde los grupos cáncer/sano están perfectamente separados, cualquier clasificador trivial alcanzaría AUC = 1.00. Eso no sería un dataset de entrenamiento; sería un **problema de juguete** que no prepara al modelo para la realidad clínica.

En la práctica oncológica, un AUC > 0.90 en un screening de CRC es excelente. Un AUC = 1.00 es una señal de alarma de data leakage o de un dataset irrealista.

### 3.2 El solapamiento biológico es el estado normal, no una excepción

Existen dos poblaciones clínicas bien documentadas que generan confusión diagnóstica:

#### A. Cánceres de estadio temprano (12% de los casos malignos)

```python
# clinical_data_generator.py – líneas 204-212
n_early   = max(1, int(len(c_idx) * 0.12))
early_idx = RNG.choice(c_idx, size=n_early, replace=False)
df_out.loc[early_idx, "CEA_Level_ng_mL"] = RNG.lognormal(np.log(1.7), 0.45, n_early).clip(0.5, 3.0)
df_out.loc[early_idx, "Hemoglobin_g_dL"] = RNG.normal(14.2, 1.00, n_early).clip(13.5, 17.5)
```

Un 15-20% de los CRC en estadio I presentan **CEA < 3 ng/mL y hemoglobina preservada** (NCCN 2023). El tumor existe, pero aún no ha generado la señal serológica ni la alteración radiológica suficiente para separarlos de controles sanos. Forzar que todos los cancerosos tengan CEA alto sería biológicamente incorrecto.

#### B. Sanos con inflamación severa (12% de los controles)

```python
# clinical_data_generator.py – líneas 217-226
n_inflam   = max(1, int(len(h_idx) * 0.12))
df_out.loc[inflam_idx, "CEA_Level_ng_mL"] = RNG.lognormal(np.log(12.0), 0.55, n_inflam).clip(8.0, 60.0)
df_out.loc[inflam_idx, "PyRad_ADC_Mean"]  = RNG.normal(1180.0, 220.0, n_inflam).clip(700.0, 1700.0)
```

La Enfermedad Inflamatoria Intestinal (EII) activa y la diverticulitis aguda pueden elevar el CEA hasta 60 ng/mL y restringir el ADC en la pared colónica inflamada (ESGAR 2022). Un modelo sin exposición a estos casos aprenderá la regla "CEA > 5 → cáncer" y fallará sistemáticamente en urgencias inflamatorias.

### 3.3 El Target Noise: inversión de etiqueta para simular la ambigüedad diagnóstica irreductible

```python
# clinical_data_generator.py – últimas líneas de generar_features_clinicas()
swap_mask = np.random.rand(len(df_out)) < 0.12
df_out.loc[swap_mask, "Diagnosis"] = 1 - df_out.loc[swap_mask, "Diagnosis"]
```

Este bloque invierte la etiqueta diagnóstica de un 12% aleatorio de pacientes. Modela la **incertidumbre diagnóstica residual** que existe incluso en datos reales bien curados:

- Casos en los que el diagnóstico histológico llegó tarde y el baseline clínico fue reclasificado.
- Pacientes con hallazgos radiológicos borderline (KRAS/BRAF wild-type con ADC ambiguo).
- Errores de anotación inherentes a cualquier cohorte multicéntrica.

**El efecto matemático intencionado:** este ruido fija un **techo teórico de AUC ≈ 0.88** para cualquier clasificador óptimo. Si un modelo supera consistentemente ese techo, es señal de sobreajuste o de que ha "memorizado" los IDs de pacientes ruidosos. Si se queda muy por debajo (< 0.80), el modelo no ha capturado las correlaciones reales.

El resultado obtenido (AUC entre 0.84 y 0.88) está exactamente en la banda esperada, lo que confirma que el modelo aprende el patrón real sin memorizar el ruido.

### 3.4 Por qué 12% y no otro valor

El 12% no es arbitrario. Proviene de la estimación de la tasa de reclasificación diagnóstica en estudios multicéntricos de CRC:

- Lieberman et al. (*Gastroenterology*, 2012): ~10-15% de lesiones T1 reclasificadas en revisión multidisciplinar.
- Colorectal Cancer Screening Metrics, EUREF 2022: ~13% de falsos positivos histológicos en screening poblacional.

El valor de 12% es el punto medio conservador de ese rango, suficiente para romper la separabilidad lineal perfecta sin hacer el problema artificialmente difícil.

---

## 4. Ruido de laboratorio e imagen: variabilidad instrumental (18% de los controles)

Además del target noise, el 18% de los sanos recibe perturbaciones de cuatro tipos que simulan variabilidad en la cadena de adquisición de datos:

| Tipo | Causa real simulada |
|---|---|
| Pico de CEA benigno | Tabaco activo, infección aguda, proceso autoinmune |
| Anemia leve | Ferropenia no oncológica, déficit de B12 |
| Variabilidad del ADC (×lognormal) | Diferencias de protocolo b-value entre escáneres 1.5T y 3T |
| Artefactos de textura | Movimiento respiratorio, distorsión de campo B0 en DW-MRI |

Esta capa de ruido diferente al target noise **no invierte etiquetas**; solo perturba las features. Su función es asegurar que el modelo no aprenda diferencias sutiles de escáner o de laboratorio que no generalizarán en producción.

---

## 5. Reproducibilidad y auditabilidad

```python
RNG = np.random.default_rng(42)   # generador principal
np.random.seed(42)                # semilla secundaria para target noise
```

El dataset es **100% reproducible** con estas dos semillas fijas. Cualquier revisor puede re-ejecutar `clinical_data_generator.py` y obtener exactamente el mismo CSV. Esto satisface el criterio de reproducibilidad de la ESMO/ESGAR para estudios de validación de biomarcadores computacionales.

---

## 6. Resumen de la cadena de validación

```
Guías clínicas         Matrices de covarianza       Dataset sintético
(NCCN, ESGAR, OMS)  →  calibradas por grupo      →  con ruido realista
                        diagnóstico                  (biológico + instrumental)
                                                          ↓
                                              Modelo entrenado en zona gris
                                              real → generalización robusta
```

El dataset no es realista por azar. Es realista porque fue **diseñado para serlo**, con cada parámetro justificado en una fuente clínica y cada capa de ruido modelando un fenómeno biológico o instrumental documentado.
