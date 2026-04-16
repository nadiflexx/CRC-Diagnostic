# CRC-Diagnostic — Módulo de Diagnóstico Clínico-Tumoral

Sistema de ayuda al diagnóstico de **Cáncer Colorrectal (CCR)** basado en XGBoost, optimización bayesiana con Optuna y explicabilidad médica con SHAP. Opera sobre biomarcadores hematológicos (CEA, Hemoglobina) y features radiómicas (ADC, Entropía, GLCM…) generadas sintéticamente a partir de distribuciones multivariantes calibradas con guías clínicas reales (NCCN 2023, ESGAR 2022). El resultado se expone a través de un dashboard interactivo en Streamlit.

---

## Objetivos

- **Generar datos sintéticos** biológicamente realistas con distribuciones multivariantes e inyección de un 12 % de ruido biológico (Falsos Negativos y Positivos clínicos), evitando la separabilidad perfecta y el data leakage.
- **Justificar la elección de XGBoost** mediante benchmark empírico frente a Regresión Logística, KNN, Random Forest y MLP.
- **Entrenar un clasificador XGBoost de producción** optimizado con Optuna y umbral de decisión ajustado para maximizar Recall ≥ 0.90.
- **Exportar un paquete de inferencia** (`.pkl`) listo para producción, verificado automáticamente antes del despliegue.
- **Exponer el modelo** en un dashboard interactivo (Streamlit) con diagnóstico en vivo y explicabilidad SHAP por paciente.

---

## Métricas del Modelo Final

| Métrica | Valor |
|---|---|
| ROC-AUC | 0.8782 |
| Recall (Sensibilidad) | 0.8965 |
| Precisión | 0.7940 |
| F1-Score | 0.8421 |
| Especificidad (TNR) | 0.7676 |
| Umbral de decisión | 0.20 |

El techo de ~0.88 AUC no es un fallo: está diseñado deliberadamente para reflejar la ambigüedad clínica real. Ver [`justificacion_datos_sinteticos.md`](justificacion_datos_sinteticos.md) para la justificación completa.

---

## Estructura del Proyecto

```
CRC-Diagnostic/
├── app.py                                   # Dashboard Streamlit (Endo-AID)
├── requirements.txt
├── explicacion_optuna_shap.md               # Documentación: Optuna y SHAP
├── justificacion_datos_sinteticos.md        # Documentación: techo biológico 88%
├── justificacion_xgboost_vs_alternativas.md # Documentación: benchmark de modelos
├── Data/
│   ├── processed/
│   │   └── dataset_clinico_tumoral.csv      # ~335 000 pacientes sintéticos (11 features)
│   └── raw/                                 # Datasets base originales
├── Data_cleaning/
│   └── clinical_data_generator.py          # Motor de síntesis multivariante
└── model/
    ├── artifacts/
    │   ├── xgb_clinical_model.json          # Hiperparámetros óptimos Optuna
    │   ├── xgb_inference_package.pkl        # Paquete de inferencia: modelo + umbral + feature_names
    │   └── inspection_plots/                # Gráficos de evaluación + inspection_metrics.json
    ├── check_features.py                    # Auditoría anti-data leakage
    ├── compare_models.py                    # Benchmark 5 clasificadores (justifica XGBoost)
    ├── inspect_inference_package.py         # Verificación del paquete de producción
    └── xgb_clinical_model.py                # Pipeline MLOps completo (Optuna → SHAP → PKL)
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

Genera `Data/processed/dataset_clinico_tumoral.csv` con ~335 000 pacientes a partir de distribuciones multivariantes calibradas con NCCN 2023 y ESGAR 2022.

```bash
python Data_cleaning/clinical_data_generator.py
```

---

### 2. Auditoría anti-data leakage

Comprueba que ninguna feature separa perfectamente a sanos de enfermos por sí sola.

```bash
python model/check_features.py
```

> Resultado esperado: cero variables con separación perfecta.

---

### 3. Benchmark de clasificadores

Evalúa cinco algoritmos (Regresión Logística, KNN, Random Forest, MLP, XGBoost) mediante validación cruzada estratificada (5-fold) y justifica empíricamente la elección de XGBoost.

```bash
python model/compare_models.py
```

> Ver [`justificacion_xgboost_vs_alternativas.md`](justificacion_xgboost_vs_alternativas.md) para los resultados y el análisis completo.

---

### 4. Entrenamiento MLOps

Ejecuta el pipeline completo: búsqueda bayesiana de hiperparámetros (Optuna, 30 trials, 5-fold CV), entrenamiento final, optimización de umbral, gráficos SHAP y exportación del paquete de inferencia.

```bash
python model/xgb_clinical_model.py
```

> Salida: `model/artifacts/xgb_inference_package.pkl` y `model/artifacts/xgb_clinical_model.json`.

---

### 5. Verificación del paquete de producción

Valida la integridad del PKL, recalcula métricas sobre el test set y genera los gráficos de evaluación.

```bash
python model/inspect_inference_package.py
```

> Salida: `model/artifacts/inspection_plots/` con cuatro PNGs y `inspection_metrics.json`.

---

### 6. Dashboard interactivo

```bash
streamlit run app.py
```

Abre `http://localhost:8501`. El dashboard tiene tres pestañas:

| Pestaña | Contenido |
|---|---|
| Diagnóstico en Vivo | Formulario de 11 variables clínicas · indicador de riesgo · gráfico SHAP local por paciente |
| Análisis y Rendimiento | Métricas del modelo · galería de 4 gráficos de evaluación |
| Documentación Técnica | Justificación del techo biológico · hiperparámetros Optuna · explicación SHAP · referencias |

---

## Stack Tecnológico

| Componente | Librería | Versión mínima |
|---|---|---|
| Clasificador | XGBoost | 2.0 |
| Optimización | Optuna (TPE) | 3.6 |
| Explicabilidad | SHAP | 0.45 |
| ML general | scikit-learn | 1.4 |
| Datos | pandas / numpy | 2.0 / 1.26 |
| Serialización | joblib | 1.3 |
| Visualización | matplotlib | 3.8 |
| Dashboard | Streamlit | 1.33 |

---

## Documentación de Diseño

- [`justificacion_datos_sinteticos.md`](justificacion_datos_sinteticos.md) — Por qué el modelo toca un techo de ~0.88 AUC y qué certifica ese límite.
- [`justificacion_xgboost_vs_alternativas.md`](justificacion_xgboost_vs_alternativas.md) — Comparativa cuantitativa de los cinco modelos del benchmark.
- [`explicacion_optuna_shap.md`](explicacion_optuna_shap.md) — Funcionamiento interno de la búsqueda bayesiana y los SHAP values.

