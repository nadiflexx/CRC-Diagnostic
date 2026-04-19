# CRC-Diagnostic — Módulo de Diagnóstico Clínico-Tumoral

Sistema de ayuda al diagnóstico de **Cáncer Colorrectal (CCR)** basado en XGBoost, optimización bayesiana con Optuna, calibración por **Temperature Scaling** y explicabilidad médica con SHAP. Opera sobre biomarcadores hematológicos (CEA, Hemoglobina) y features radiómicas (ADC, Entropía, GLCM…) generadas sintéticamente con estadificación T1–T4 calibrada con guías clínicas reales (NCCN 2023, ESGAR 2022, Gollub 2018). El resultado se expone a través de un dashboard interactivo en Streamlit.

---

## Objetivos

- **Generar datos sintéticos** biológicamente realistas con distribuciones multivariantes **estadificadas por estadio T1–T4**, solapamiento clínico real en estadios tempranos y casos inflamatorios benignos multi-feature.
- **Justificar la elección de XGBoost** mediante benchmark empírico frente a Regresión Logística, KNN, Random Forest y MLP.
- **Entrenar un clasificador XGBoost de producción** optimizado con Optuna, umbral de decisión ajustado para Recall ≥ 0.90, y **calibración de probabilidades por Temperature Scaling** (Guo et al., ICML 2017).
- **Exportar un paquete de inferencia** (`.pkl`) listo para producción con modelo calibrado, modelo raw para SHAP, umbral y nombres de features.
- **Exponer el modelo** en un dashboard interactivo (Streamlit) con diagnóstico en vivo, probabilidades clínicamente interpretables y explicabilidad SHAP por paciente.

---

## Métricas del Modelo Final

| Métrica | Valor |
|---|---|
| ROC-AUC | 0.9743 |
| Recall (Sensibilidad) | 0.9784 |
| Precisión | 0.7800 |
| F1-Score | 0.8653 |
| Umbral de decisión | 0.20 |
| Temperatura de calibración | 1.50 |
| Rango de probabilidades | 5 % – 95 % |

El AUC de 0.974 refleja un dataset con solapamiento clínico real entre estadios T1/T2 y tejido benigno inflamado. Ver [`justificacion_datos_sinteticos.md`](justificacion_datos_sinteticos.md) para la justificación completa.

---

## Estructura del Proyecto

```
CRC-Diagnostic/
├── app.py                                   # Dashboard Streamlit (Endo-AID)
├── requirements.txt
├── explicacion_optuna_shap.md               # Documentación: Optuna, SHAP y Temperature Scaling
├── justificacion_datos_sinteticos.md        # Documentación: estadificación T1-T4 y realismo clínico
├── justificacion_xgboost_vs_alternativas.md # Documentación: benchmark de modelos
├── Data/
│   ├── processed/
│   │   └── dataset_clinico_tumoral.csv      # ~335 000 pacientes sintéticos (11 features) [gitignored]
│   └── raw/
│       └── colorectal_cancer_full_dataset_v1.csv  # Dataset base de entrada
├── Data_cleaning/
│   └── clinical_data_generator.py          # Motor de síntesis multivariante con staging T1-T4
└── model/
    ├── calibration.py                       # Temperature Scaling — NECESARIO para deserializar el PKL
    ├── inspect_inference_package.py         # Regenera plots/métricas para la app (llamado por app.py)
    ├── xgb_clinical_model.py                # Pipeline completo: Optuna → XGBoost → calibración → PKL
    └── artifacts/                           # Generados en runtime [gitignored]
        ├── xgb_clinical_model.json          # Pesos XGBoost exportados
        ├── xgb_tumoral_model_package.pkl     # Paquete de inferencia: model + model_raw + threshold + feature_names
        └── inspection_plots/                # Gráficos de evaluación + inspection_metrics.json
```

---

## Guía de Ejecución

### 0. Instalación del entorno

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

### 1. Generación del dataset sintético

Genera `Data/processed/dataset_clinico_tumoral.csv` con ~335 000 pacientes a partir de distribuciones multivariantes **estadificadas por T1–T4** calibradas con NCCN 2023, ESGAR 2022 y Gollub 2018.

```bash
python Data_cleaning/clinical_data_generator.py
```

Produce un dataset balanceado (50/50) con solapamiento clínico real: T1 solapado con sanos, T2 con zona gris, T3/T4 claramente separados. El 14 % de los sanos recibe perturbaciones multi-feature de inflamación severa (EII, diverticulitis).

---

### 2. Entrenamiento MLOps

Ejecuta el pipeline completo en un solo comando:

1. Optuna TPE — 30 trials, CV 5-fold estratificado → hiperparámetros óptimos
2. XGBoost final sobre el 68 % de los datos (fit set)
3. **Temperature Scaling** — encuentra T óptima sobre el 12 % de calibración (mínimo clínico T=1.5)
4. Búsqueda de umbral óptimo con `recall_objetivo=0.90`, `umbral_maximo=0.20`
5. Evaluación en test set (20 %) + gráficos SHAP
6. Exportación del paquete de inferencia

```bash
python model/xgb_clinical_model.py
```

> Salida: `model/artifacts/xgb_tumoral_model_package.pkl` con claves `model` (calibrado), `model_raw` (para SHAP), `threshold` y `feature_names`.

---

### 3. Dashboard interactivo

`app.py` arranca autónomamente: si los plots de inspección no existen, los regenera llamando a `inspect_inference_package.py` antes de mostrar la interfaz.

```bash
streamlit run app.py
```

Abre `http://localhost:8501`. El dashboard tiene tres pestañas:

| Pestaña | Contenido |
|---|---|
| Diagnóstico en Vivo | Formulario de 11 variables clínicas · probabilidad calibrada (5–95 %) · gráfico SHAP local |
| Análisis y Rendimiento | Métricas del modelo · galería de gráficos de evaluación · distribución de probabilidades |
| Documentación Técnica | Estadificación T1–T4 · Temperature Scaling · Optuna · SHAP · referencias |

---

## Flujo de scripts

```
clinical_data_generator.py
    └─ Lee Data/raw/*.csv
    └─ Genera Data/processed/dataset_clinico_tumoral.csv
           │
           ▼
xgb_clinical_model.py
    └─ Optuna 30 trials (CV 5-fold)  →  mejores hiperparámetros
    └─ XGBoost fit (68 % datos)
    └─ Temperature Scaling (12 % calibración)  →  T=1.5
    └─ Búsqueda umbral (umbral_maximo=0.20)   →  threshold=0.20
    └─ SHAP beeswarm + barplot
    └─ PKL: {model, model_raw, threshold, feature_names}
           │
           ▼
app.py  (Streamlit)
    └─ Importa calibration.ModeloCalibraado  (necesario antes de joblib.load)
    └─ Si faltan plots → llama inspect_inference_package.main()
    └─ Carga PKL  →  model (calibrado), model_raw (SHAP)
    └─ Tab 1: predicción en vivo con SHAP local
    └─ Tab 2: métricas + plots de inspección
    └─ Tab 3: documentación técnica
```

---

## Stack Tecnológico

| Componente | Librería | Versión mínima |
|---|---|---|
| Clasificador | XGBoost | 2.0 |
| Calibración | scipy (minimize_scalar) | 1.11 |
| Optimización | Optuna (TPE) | 3.6 |
| Explicabilidad | SHAP | 0.45 |
| ML general | scikit-learn | 1.4 |
| Datos | pandas / numpy | 2.0 / 1.26 |
| Serialización | joblib | 1.3 |
| Visualización | matplotlib | 3.8 |
| Dashboard | Streamlit | 1.33 |

---

## Documentación de Diseño

- [`justificacion_datos_sinteticos.md`](justificacion_datos_sinteticos.md) — Estadificación T1–T4, solapamiento clínico real y variabilidad biológica.
- [`justificacion_xgboost_vs_alternativas.md`](justificacion_xgboost_vs_alternativas.md) — Comparativa cuantitativa de los cinco modelos del benchmark.
- [`explicacion_optuna_shap.md`](explicacion_optuna_shap.md) — Optuna TPE, SHAP TreeExplainer y Temperature Scaling

- [`justificacion_datos_sinteticos.md`](justificacion_datos_sinteticos.md) — Estadificación T1–T4, solapamiento clínico real y variabilidad biológica.
- [`justificacion_xgboost_vs_alternativas.md`](justificacion_xgboost_vs_alternativas.md) — Comparativa cuantitativa de los cinco modelos del benchmark.
- [`explicacion_optuna_shap.md`](explicacion_optuna_shap.md) — Optuna TPE, SHAP TreeExplainer y Temperature Scaling.

