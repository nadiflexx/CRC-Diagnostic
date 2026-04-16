"""
app.py  —  Endo-AID: Tumor Diagnostic Module
Dashboard Streamlit para el modelo XGBoost de diagnóstico de Cáncer Colorrectal.

Estructura:
  Tab 1 — Diagnóstico en Vivo   : formulario clínico + indicador de riesgo + SHAP local
  Tab 2 — Análisis y Rendimiento : galería de plots + tabla de métricas
  Tab 3 — Documentación Técnica  : techo biológico + Optuna + SHAP

Uso:
    streamlit run app.py
"""

import json
import importlib
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Configuración de página
st.set_page_config(
    page_title="Endo-AID · Tumor Diagnostic",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Rutas absolutas
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PKL_PATH    = os.path.join(BASE_DIR, "model", "artifacts", "xgb_inference_package.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "artifacts", "inspection_plots", "inspection_metrics.json")
PLOTS_DIR   = os.path.join(BASE_DIR, "model", "artifacts", "inspection_plots")


# Genera los plots y métricas de inspección si alguno falta
_EXPECTED_PLOTS = [
    "inspection_metrics.json",
    "panel_evaluacion_inferencia.png",
    "feature_importances_top20.png",
    "shap_beeswarm.png",
    "shap_bar.png",
]

if not all(os.path.exists(os.path.join(PLOTS_DIR, f)) for f in _EXPECTED_PLOTS):
    _model_dir = os.path.join(BASE_DIR, "model")
    if _model_dir not in sys.path:
        sys.path.insert(0, _model_dir)
    _iip = importlib.import_module("inspect_inference_package")
    _iip.main()


# Recursos cacheados
@st.cache_resource(show_spinner="Cargando modelo XGBoost...")
def load_package() -> dict:
    """Carga el paquete de inferencia XGBoost desde disco.

    Returns:
        dict con claves ``model``, ``threshold`` y ``feature_names``.

    Raises:
        SystemExit: si el PKL no existe en la ruta esperada.
    """
    if not os.path.exists(PKL_PATH):
        st.error(f"No se encontró el paquete de inferencia en:\n`{PKL_PATH}`\n\nEjecuta primero `xgb_clinical_model.py`.")
        st.stop()
    return joblib.load(PKL_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    """Carga las métricas de evaluación desde el JSON de inspeccion.

    Returns:
        dict con las métricas, o dict vacío si el JSON no existe.
    """
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


pkg           = load_package()
model         = pkg["model"]
threshold     = float(pkg.get("threshold", 0.2))
feature_names: list = pkg["feature_names"]

metrics_data  = load_metrics()
metrics       = metrics_data.get("metrics_test_recomputed", {})


# CSS — paleta neutral fija, independiente del modo claro/oscuro del navegador.
# Los !important anulan hojas de Streamlit y prefers-color-scheme del sistema.
st.markdown("""
<style>
/* ── Reset de modo oscuro del navegador ── */
:root {
    color-scheme: light only;
}

/* ── Fondo global neutro (gris pizarra frío) ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
[data-testid="block-container"],
section.main,
div.block-container {
    background-color: #1e2533 !important;
    color: #dce3ef !important;
}

/* Barra superior de Streamlit */
[data-testid="stHeader"] {
    background-color: #1e2533 !important;
    border-bottom: 1px solid #2d3548 !important;
}

/* Sidebar (por si se activa) */
[data-testid="stSidebar"] {
    background-color: #161c28 !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    color: #8fa3c8 !important;
    background: transparent !important;
    border-bottom: 2px solid transparent;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #7eb8f7 !important;
    border-bottom: 2px solid #4a90d9 !important;
}

/* Inputs numéricos y selectbox */
input[type="number"],
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div {
    background-color: #252d3d !important;
    color: #dce3ef !important;
    border: 1px solid #3a4560 !important;
    border-radius: 6px !important;
}

/* Métricas */
[data-testid="stMetric"] {
    background-color: #252d3d !important;
    border-radius: 8px !important;
    padding: .6rem 1rem !important;
    border: 1px solid #3a4560 !important;
}
[data-testid="stMetricLabel"]  { color: #8fa3c8 !important; }
[data-testid="stMetricValue"]  { color: #eaf0fc !important; }

/* Dataframes */
[data-testid="stDataFrame"] {
    background-color: #252d3d !important;
    border-radius: 8px !important;
}

/* Divider */
hr { border-color: #2d3548 !important; }

/* Botón primario */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #1a5fa8 0%, #0d2b4e 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: .02em;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2276d2 0%, #173f72 100%) !important;
}

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background-color: #4a90d9 !important;
}

/* ── Cabecera principal ── */
.endo-header {
    background: linear-gradient(135deg, #0d2b4e 0%, #1a5fa8 100%);
    padding: 1.4rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 16px rgba(0,0,0,.35);
}
.endo-header h1 { color: #ffffff; font-size: 1.9rem; margin: 0 0 .25rem 0; }
.endo-header p  { color: #b8d0f0; margin: 0; font-size: .92rem; }

/* ── Títulos de sección ── */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #7eb8f7;
    border-bottom: 2px solid #2d5a99;
    padding-bottom: .3rem;
    margin-bottom: 1rem;
}

/* ── Veredictos ── */
.verdict-high {
    background: #2e1515;
    border: 2px solid #dc2626;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    text-align: center;
}
.verdict-low {
    background: #122212;
    border: 2px solid #16a34a;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    text-align: center;
}

/* ── Tarjetas de información ── */
.info-card {
    background: #252d3d;
    border: 1px solid #3a4560;
    border-radius: 10px;
    padding: 1.2rem;
    margin-bottom: .8rem;
    color: #c8d6ee;
    line-height: 1.65;
}

/* ── Caption / notas pequeñas ── */
[data-testid="stCaptionContainer"] p {
    color: #7a90b8 !important;
}
</style>
""", unsafe_allow_html=True)


# Cabecera principal
st.markdown("""
<div class="endo-header">
    <h1>🔬 Endo-AID: Tumor Diagnostic Module</h1>
    <p>Sistema de Ayuda al Diagnóstico · Cáncer Colorrectal (CCR) · XGBoost + SHAP · v1.0</p>
</div>
""", unsafe_allow_html=True)


# Pestañas
tab1, tab2, tab3 = st.tabs([
    "🩺  Diagnóstico en Vivo",
    "📊  Análisis y Rendimiento",
    "📄  Documentación Técnica",
])


# TAB 1 — DIAGNÓSTICO EN VIVO
with tab1:
    # Configuración de cada feature: etiqueta, rango, valor por defecto, ayuda
    FEATURE_CFG = {
        "Age": {
            "label":   "Edad (años)",
            "min":     20.0, "max": 90.0, "default": 55.0, "step": 1.0,
            "help":    "Edad del paciente en el momento de la prueba.",
        },
        "Smoking_History": {
            "label":   "Historia tabáquica",
            "type":    "select",
            "options": [0, 1],
            "fmt":     {0: "No fumador (0)", 1: "Fumador (1)"},
            "default": 0,
            "help":    "El tabaquismo eleva el CEA basal en personas sanas.",
        },
        "CEA_Level_ng_mL": {
            "label":   "CEA (ng/mL)",
            "min":     0.1, "max": 500.0, "default": 2.5, "step": 0.1,
            "help":    "Antígeno carcinoembrionario. Valor normal < 3 ng/mL. "
                       "Tumores avanzados pueden superar 20 ng/mL.",
        },
        "Hemoglobin_g_dL": {
            "label":   "Hemoglobina (g/dL)",
            "min":     5.0, "max": 20.0, "default": 13.5, "step": 0.1,
            "help":    "Normal: hombres 13.5–17.5 g/dL · mujeres 12.0–15.5 g/dL. "
                       "Los tumores con sangrado crónico causan anemia.",
        },
        "PyRad_ADC_Mean": {
            "label":   "ADC Medio (µm²/s)",
            "min":     200.0, "max": 2500.0, "default": 1500.0, "step": 10.0,
            "help":    "Coeficiente de Difusión Aparente (MRI-DWI). "
                       "Tejido tumoral: < 1300 µm²/s (difusión restringida).",
        },
        "PyRad_ADC_Std": {
            "label":   "ADC Desv. Estándar",
            "min":     5.0, "max": 400.0, "default": 150.0, "step": 5.0,
            "help":    "Heterogeneidad del ADC dentro del volumen de interés. "
                       "Mayor en tejido maligno por necrosis e hipoxia.",
        },
        "PyRad_Entropy": {
            "label":   "Entropía GLCM",
            "min":     0.1, "max": 10.0, "default": 5.0, "step": 0.1,
            "help":    "Desorden textural de la imagen. "
                       "Valores > 6 son más frecuentes en lesiones malignas.",
        },
        "PyRad_GLCM_Contrast": {
            "label":   "Contraste GLCM",
            "min":     0.1, "max": 200.0, "default": 21.0, "step": 0.5,
            "help":    "Variación local de intensidades en la textura. "
                       "Más alto en tumor que en mucosa sana.",
        },
        "PyRad_GLCM_Homogeneity": {
            "label":   "Homogeneidad GLCM",
            "min":     0.01, "max": 1.0, "default": 0.70, "step": 0.01,
            "help":    "Uniformidad textural. Rango 0–1. "
                       "Tejido tumoral: 0.25–0.45 · Tejido sano: 0.60–0.85.",
        },
        "PyRad_Shape_Sphericity": {
            "label":   "Esfericidad (forma)",
            "min":     0.25, "max": 1.0, "default": 0.73, "step": 0.01,
            "help":    "Regularidad geométrica del VOI. Rango 0–1. "
                       "Los tumores son irregulares (< 0.65).",
        },
        "PyRad_FirstOrder_Skewness": {
            "label":   "Asimetría (1er orden)",
            "min":     -3.0, "max": 3.0, "default": 0.1, "step": 0.05,
            "help":    "Asimetría de la distribución de intensidades del vóxel. "
                       "Positiva y elevada en tejido tumoral por necrosis.",
        },
    }

    col_form, col_result = st.columns([1, 1.7], gap="large")

    # Formulario de entrada
    with col_form:
        st.markdown('<p class="section-title">🧬 Datos Clínicos del Paciente</p>',
                    unsafe_allow_html=True)

        input_values: dict[str, float] = {}

        for feat in feature_names:
            cfg = FEATURE_CFG.get(feat)
            if cfg is None:
                input_values[feat] = st.number_input(feat, value=0.0, key=feat)
                continue

            if cfg.get("type") == "select":
                sel = st.selectbox(
                    cfg["label"],
                    options=cfg["options"],
                    index=cfg["options"].index(cfg["default"]),
                    format_func=lambda x, m=cfg["fmt"]: m[x],
                    help=cfg["help"],
                    key=feat,
                )
                input_values[feat] = float(sel)
            else:
                input_values[feat] = st.number_input(
                    cfg["label"],
                    min_value=float(cfg["min"]),
                    max_value=float(cfg["max"]),
                    value=float(cfg["default"]),
                    step=float(cfg["step"]),
                    help=cfg["help"],
                    key=feat,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button(
            "🔍  Calcular Riesgo Oncológico",
            type="primary",
            use_container_width=True,
        )

    # Panel de resultados
    with col_result:
        st.markdown('<p class="section-title">📋 Resultado del Análisis</p>',
                    unsafe_allow_html=True)

        if not run_btn:
            st.info(
                "Completa el formulario con los datos del paciente y pulsa "
                "**Calcular Riesgo Oncológico** para obtener el diagnóstico."
            )
            st.markdown("""
<div class="info-card">
<b>Notas clínicas</b><br>
• El umbral de decisión es <b>0.20</b> para maximizar Sensibilidad/Recall.<br>
• Cualquier probabilidad ≥ 20 % activa la alerta de <b>ALTO RIESGO</b>.<br>
• El gráfico SHAP explica qué variables impulsan la predicción de este paciente.<br>
• Este sistema es de <b>apoyo a la decisión</b>: no reemplaza al criterio clínico.
</div>
""", unsafe_allow_html=True)
        else:
            X_input = pd.DataFrame(
                [[input_values[f] for f in feature_names]],
                columns=feature_names,
            )
            prob = float(model.predict_proba(X_input)[0, 1])
            pct  = prob * 100

            # Indicador de riesgo
            col_pct, col_thr = st.columns(2)
            col_pct.metric("Probabilidad de Malignidad", f"{pct:.1f} %")
            col_thr.metric("Umbral de Decisión",         f"{threshold * 100:.0f} %")

            st.progress(min(prob, 1.0))

            # Veredicto
            if prob >= threshold:
                st.markdown(f"""
<div class="verdict-high">
    <h2 style="color:#f87171;margin:0">🚨 ALTO RIESGO</h2>
    <p style="color:#fca5a5;font-size:1.05rem;margin:.6rem 0 0 0">
        <b>Derivar a Nivel 3 — Oncología / Endoscopia Avanzada</b><br>
        <small>P(Cáncer) = {pct:.1f}% &nbsp;≥&nbsp; Umbral {threshold*100:.0f}%</small>
    </p>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="verdict-low">
    <h2 style="color:#4ade80;margin:0">✅ BAJO RIESGO</h2>
    <p style="color:#86efac;font-size:1.05rem;margin:.6rem 0 0 0">
        <b>Sin indicios clínicos de malignidad significativa</b><br>
        <small>P(Cáncer) = {pct:.1f}% &nbsp;&lt;&nbsp; Umbral {threshold*100:.0f}%</small>
    </p>
</div>""", unsafe_allow_html=True)

            # SHAP local
            st.markdown("<br>**Explicación SHAP — Contribución de cada variable**",
                        unsafe_allow_html=True)

            with st.spinner("Calculando impacto SHAP..."):
                explainer = shap.TreeExplainer(model)
                shap_raw  = explainer.shap_values(X_input)

                if isinstance(shap_raw, list):
                    sv = np.array(shap_raw[1]).flatten()
                    base_raw = explainer.expected_value
                    base_val = float(base_raw[1]) if hasattr(base_raw, "__len__") else float(base_raw)
                else:
                    sv = np.array(shap_raw).flatten()
                    base_raw = explainer.expected_value
                    base_val = float(base_raw[1]) if hasattr(base_raw, "__len__") else float(base_raw)

                order    = np.argsort(np.abs(sv))[::-1][:10]
                top_feats = [feature_names[i] for i in order]
                top_vals  = sv[order]
                top_raw   = [input_values[f] for f in top_feats]

                labels = [f"{f}\n= {v:.2f}" for f, v in zip(top_feats, top_raw)]
                colors = ["#ef4444" if v > 0 else "#22c55e" for v in top_vals]

                fig, ax = plt.subplots(figsize=(7, 4.2))
                ax.barh(
                    range(len(labels)),
                    top_vals[::-1],
                    color=colors[::-1],
                    edgecolor="white",
                    linewidth=0.5,
                )
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels[::-1], fontsize=8.5)
                ax.axvline(0, color="#374151", linewidth=1.0)
                ax.set_xlabel("Impacto SHAP (log-odds)")
                ax.set_title(
                    f"Top 10 features · Base del modelo: {base_val:.3f}",
                    fontsize=9.5,
                )
                ax.grid(axis="x", alpha=0.25)
                fig.patch.set_facecolor("#ffffff")
                ax.set_facecolor("#fafafa")
                plt.tight_layout()
                st.pyplot(fig, width='stretch')
                plt.close(fig)

            st.caption(
                "🔴 Rojo = variable que **aumenta** el riesgo de malignidad  "
                "· 🟢 Verde = variable que **reduce** el riesgo"
            )


# TAB 2 — ANÁLISIS Y RENDIMIENTO
with tab2:

    # Métricas resumen
    st.markdown('<p class="section-title">📈 Métricas de Rendimiento (Test Set)</p>',
                unsafe_allow_html=True)

    if metrics:
        cm_data = metrics.get("confusion_matrix", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ROC-AUC",         f"{metrics.get('roc_auc', 0):.4f}")
        c2.metric("Recall (Sensib.)", f"{metrics.get('recall', 0):.4f}")
        c3.metric("Precision",        f"{metrics.get('precision', 0):.4f}")
        c4.metric("F1-Score",         f"{metrics.get('f1', 0):.4f}")
        c5.metric("Especificidad",    f"{metrics.get('specificity', 0):.4f}")

        st.markdown("<br>", unsafe_allow_html=True)
        ci1, ci2, ci3, ci4 = st.columns(4)
        ci1.metric("Umbral decisión",  f"{metrics_data.get('threshold', 0.2):.2f}")
        ci2.metric("Verdaderos Pos.", cm_data.get("tp", "—"))
        ci3.metric("Falsos Negativos", cm_data.get("fn", "—"),
                   delta=f"−{cm_data.get('fn', 0)} FN",
                   delta_color="inverse")
        ci4.metric("Falsos Positivos", cm_data.get("fp", "—"))
    else:
        st.warning("No se encontró `inspection_metrics.json`. Ejecuta `inspect_inference_package.py` primero.")

    st.divider()

    # Tabla comparativa de métricas
    if metrics:
        st.markdown('<p class="section-title">🗂️ Tabla Comparativa de Métricas</p>',
                    unsafe_allow_html=True)

        rows = [
            {"Métrica":       "ROC-AUC",
             "Valor":         round(metrics.get("roc_auc", 0), 4),
             "Interpretación": "Capacidad discriminativa global (1 = perfecto)"},
            {"Métrica":       "Average Precision",
             "Valor":         round(metrics.get("average_precision", 0), 4),
             "Interpretación": "Área bajo la curva Precisión-Recall"},
            {"Métrica":       "Recall (Sensibilidad)",
             "Valor":         round(metrics.get("recall", 0), 4),
             "Interpretación": "% de cánceres detectados — objetivo ≥ 0.90"},
            {"Métrica":       "Precisión",
             "Valor":         round(metrics.get("precision", 0), 4),
             "Interpretación": "% de alarmas que son realmente cáncer"},
            {"Métrica":       "F1-Score",
             "Valor":         round(metrics.get("f1", 0), 4),
             "Interpretación": "Media armónica Recall–Precisión"},
            {"Métrica":       "Especificidad (TNR)",
             "Valor":         round(metrics.get("specificity", 0), 4),
             "Interpretación": "% de sanos correctamente identificados"},
        ]
        st.dataframe(
            pd.DataFrame(rows),
            width='stretch',
            hide_index=True,
        )

    st.divider()

    # Galería de evaluación
    st.markdown('<p class="section-title">🖼️ Galería de Evaluación del Modelo</p>',
                unsafe_allow_html=True)

    GALLERY = [
        (
            "Panel de Evaluación Global",
            "panel_evaluacion_inferencia.png",
            "Matriz de confusión · Curva ROC · Curva Precisión-Recall · "
            "Distribución de probabilidades por clase (umbral = 0.20).",
        ),
        (
            "SHAP Beeswarm — Impacto Biológico",
            "shap_beeswarm.png",
            "Cada punto representa un paciente. El eje X es el impacto en log-odds. "
            "El color indica el valor de la feature (rojo = alto · azul = bajo).",
        ),
        (
            "Importancias del Modelo (Top 20)",
            "feature_importances_top20.png",
            "Peso relativo de cada feature según `feature_importances_` de XGBoost "
            "(ganancia media en splits).",
        ),
        (
            "SHAP Barplot — Importancia Media",
            "shap_bar.png",
            "Ranking de features por valor |SHAP| medio absoluto "
            "calculado sobre una submuestra de 2 000 pacientes.",
        ),
    ]

    for i in range(0, len(GALLERY), 2):
        cols = st.columns(2, gap="medium")
        for j, col in enumerate(cols):
            if i + j < len(GALLERY):
                title, fname, caption = GALLERY[i + j]
                img_path = os.path.join(PLOTS_DIR, fname)
                with col:
                    st.markdown(f"**{title}**")
                    if os.path.exists(img_path):
                        st.image(img_path, width='stretch')
                        st.caption(caption)
                    else:
                        st.warning(
                            f"Imagen no encontrada: `{fname}`\n\n"
                            "Ejecuta `inspect_inference_package.py` para generarla."
                        )


# TAB 3 — DOCUMENTACIÓN TÉCNICA
with tab3:
    st.markdown('<p class="section-title">📚 Documentación Técnica del Sistema</p>',
                unsafe_allow_html=True)

    col_doc1, col_doc2 = st.columns(2, gap="large")

    with col_doc1:
        st.markdown("### 🧱 El Techo Biológico del 88%")
        st.markdown("""
<div class="info-card">
El modelo <b>no puede superar un ROC-AUC de ≈ 0.88</b> por diseño deliberado.
Este límite matemático se introduce durante la generación de datos sintéticos
para replicar la ambigüedad clínica real:

<ul>
  <li><b>12% de ruido biológico</b>: se inyectan falsos negativos clínicos —cánceres en
      estadio temprano con CEA &lt; 3 ng/mL y hemoglobina preservada— y falsos positivos
      —pacientes sanos con biomarcadores elevados por tabaquismo o enfermedad inflamatoria
      intestinal (EII)—.</li>
  <li><b>Distribuciones solapadas</b>: las matrices de covarianza multivariante de los grupos
      <i>cáncer</i> y <i>sano</i> se calibran para generar zonas de ambigüedad irreducible,
      replicando la variabilidad real descrita en NCCN 2023 y Duffy et al. 2021.</li>
  <li><b>Verificación anti-data leakage</b>: <code>check_features.py</code> confirma que
      ninguna variable separa perfectamente a los enfermos de los sanos por sí sola.</li>
</ul>

Este techo del 88% <b>certifica que el dataset es un reto real</b> y que el modelo
no memoriza patrones triviales.
</div>
""", unsafe_allow_html=True)

        st.markdown("### 🧪 Pipeline de Features (11 variables)")
        feat_types = [
            ("Age",                       "Clínica demográfica",    "Registro del paciente"),
            ("Smoking_History",           "Clínica hábito",         "Factor de riesgo CEA"),
            ("CEA_Level_ng_mL",           "Hematológica",           "Analítica de sangre"),
            ("Hemoglobin_g_dL",           "Hematológica",           "Analítica de sangre"),
            ("PyRad_ADC_Mean",            "Radiológica",            "MRI-DWI"),
            ("PyRad_ADC_Std",             "Radiológica",            "MRI-DWI"),
            ("PyRad_Entropy",             "Radiológica",            "PyRadiomics GLCM"),
            ("PyRad_GLCM_Contrast",       "Radiológica",            "PyRadiomics GLCM"),
            ("PyRad_GLCM_Homogeneity",    "Radiológica",            "PyRadiomics GLCM"),
            ("PyRad_Shape_Sphericity",    "Radiológica",            "PyRadiomics Shape"),
            ("PyRad_FirstOrder_Skewness", "Radiológica",            "PyRadiomics 1er orden"),
        ]
        df_feat = pd.DataFrame(feat_types, columns=["Feature", "Categoría", "Fuente"])
        st.dataframe(df_feat, width='stretch', hide_index=True)

    with col_doc2:
        st.markdown("### ⚡ Optuna — Búsqueda Bayesiana de Hiperparámetros")
        st.markdown("""
<div class="info-card">
En lugar de un <code>GridSearch</code> exhaustivo, el sistema usa <b>Optuna</b> con
algoritmo <b>TPE</b> (<i>Tree-structured Parzen Estimator</i>):

<ol>
  <li><b>Espacio de búsqueda</b>: 9 hiperparámetros — <code>n_estimators</code>,
      <code>max_depth</code>, <code>learning_rate</code>, <code>subsample</code>,
      <code>colsample_bytree</code>, <code>min_child_weight</code>, <code>gamma</code>,
      <code>reg_alpha</code>, <code>reg_lambda</code>.</li>
  <li><b>30 trials</b> con validación cruzada <b>K-Fold estratificado (k = 5)</b>.
      Cada trial maximiza el ROC-AUC medio sobre los 5 pliegues.</li>
  <li><b>Hiperparámetros óptimos encontrados:</b></li>
</ol>
</div>
""", unsafe_allow_html=True)

        model_params = metrics_data.get("model_params", {})
        if model_params:
            df_params = pd.DataFrame(
                [
                    {"Hiperparámetro": k, "Valor": round(v, 6) if isinstance(v, float) else v}
                    for k, v in model_params.items()
                ]
            )
            st.dataframe(df_params, width='stretch', hide_index=True)

        st.markdown("""
**Umbral de decisión optimizado:**
Tras el entrenamiento se busca el mayor umbral que garantiza `Recall ≥ 0.90`
sobre el conjunto de entrenamiento. El resultado es **threshold = 0.20**,
priorizando la sensibilidad clínica (minimizar falsos negativos en oncología).
        """)

        st.markdown("### 🔎 SHAP — Explicabilidad Médica Certificable")
        st.markdown("""
<div class="info-card">
<b>SHAP</b> (<i>SHapley Additive exPlanations</i>) descompone la predicción de cada
paciente en la contribución individual de cada variable:

<ul>
  <li><b>TreeExplainer</b>: calcula SHAP values de forma <i>exacta</i> para modelos de
      árboles, sin aproximaciones. Coste O(TLD): T = árboles, L = hojas, D = profundidad.</li>
  <li><b>Gráfico global (Beeswarm)</b>: cada punto es un paciente.
      El eje X muestra el impacto en log-odds. El color representa el valor de la feature.</li>
  <li><b>Gráfico local (Tab 1)</b>: para cada caso concreto, muestra qué variables aumentan
      o reducen la probabilidad de malignidad respecto al valor base del modelo.</li>
</ul>

Esta capa de explicabilidad permite al clínico <b>justificar y auditar</b> cada predicción,
cumpliendo con los principios de IA responsable en entornos médicos (EU AI Act, MDR 2017/745).
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📖 Referencias Científicas")
    st.markdown("""
| Referencia | Relevancia |
|---|---|
| NCCN Clinical Practice Guidelines — Colorectal Cancer, v2.2023 | Rangos clínicos CEA y criterios de derivación |
| ESGAR Consensus Statement on Rectal MRI, 2022 | Parámetros ADC y criterios DWI |
| Duffy et al., *CEA as a marker for colorectal cancer*, EJCA 2021 | Efectos tabaquismo / EII sobre CEA basal |
| Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017 | Base teórica SHAP |
| Akiba et al., *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD 2019 | Base teórica búsqueda bayesiana |
    """)
