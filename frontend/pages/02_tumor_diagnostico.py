"""
Tumor Diagnostic — Clinical + Radiomic Risk Assessment.
"""

from typing import TypedDict

from components.banners import show_empty_state, show_result_banner
from components.cards import page_header
from components.sidebar import render_sidebar
import streamlit as st
from utils.api_client import get_patients, run_diagnosis
from utils.helpers import apply_custom_css, build_patient_options


class FeatureConfig(TypedDict, total=False):
    label: str
    min: float
    max: float
    default: float
    step: float
    help: str
    type: str
    options: list[int]
    fmt: dict[int, str]


st.set_page_config(
    page_title="Diagnóstico Tumoral · Endo-AID",
    page_icon="🌿",
    layout="wide",
)
apply_custom_css()
render_sidebar()

# ── Feature configuration ────────────────────────────────────────────────────
FEATURE_CFG: dict[str, FeatureConfig] = {
    "age_value": {
        "label": "Edad (años)",
        "min": 18.0,
        "max": 100.0,
        "default": 55.0,
        "step": 1.0,
        "help": "Edad del paciente en el momento del estudio.",
    },
    "smoking_history": {
        "label": "Historia tabáquica",
        "type": "select",
        "options": [0, 1],
        "fmt": {0: "No fumador (0)", 1: "Fumador / Ex-fumador (1)"},
        "default": 0,
        "help": "El tabaquismo eleva el CEA basal en personas sanas (Duffy et al. 2021).",
    },
    "cea_level_ng_ml": {
        "label": "CEA (ng/mL)",
        "min": 0.1,
        "max": 500.0,
        "default": 2.1,
        "step": 0.1,
        "help": "Antígeno carcinoembrionario. Normal < 3 ng/mL. "
        "Tumores avanzados pueden superar 20 ng/mL.",
    },
    "hemoglobin_g_dl": {
        "label": "Hemoglobina (g/dL)",
        "min": 5.0,
        "max": 20.0,
        "default": 13.5,
        "step": 0.1,
        "help": "Normal: hombres 13.5–17.5 · mujeres 12.0–15.5 g/dL. "
        "Los tumores con sangrado crónico causan anemia.",
    },
    "pyrad_adc_mean": {
        "label": "ADC Medio (µm²/s)",
        "min": 200.0,
        "max": 2500.0,
        "default": 1540.0,
        "step": 10.0,
        "help": "Coeficiente de Difusión Aparente (MRI-DWI). "
        "Tejido tumoral: < 1 300 µm²/s (difusión restringida).",
    },
    "pyrad_adc_std": {
        "label": "ADC Desv. Estándar",
        "min": 5.0,
        "max": 400.0,
        "default": 80.0,
        "step": 5.0,
        "help": "Heterogeneidad del ADC dentro del VOI. "
        "Mayor en tejido maligno por necrosis e hipoxia.",
    },
    "pyrad_entropy": {
        "label": "Entropía GLCM",
        "min": 0.1,
        "max": 10.0,
        "default": 4.8,
        "step": 0.1,
        "help": "Desorden textural. Valores > 6 son más frecuentes en lesiones malignas.",
    },
    "pyrad_glcm_contrast": {
        "label": "Contraste GLCM",
        "min": 0.1,
        "max": 200.0,
        "default": 21.0,
        "step": 0.5,
        "help": "Variación local de intensidades. Más alto en tumor que en mucosa sana.",
    },
    "pyrad_glcm_homogeneity": {
        "label": "Homogeneidad GLCM",
        "min": 0.01,
        "max": 1.0,
        "default": 0.72,
        "step": 0.01,
        "help": "Uniformidad textural (0–1). Tumoral: 0.25–0.45 · Sano: 0.60–0.85.",
    },
    "pyrad_shape_sphericity": {
        "label": "Esfericidad (forma)",
        "min": 0.25,
        "max": 1.0,
        "default": 0.73,
        "step": 0.01,
        "help": "Regularidad geométrica del VOI. Los tumores son irregulares (< 0.65).",
    },
    "pyrad_firstorder_skewness": {
        "label": "Asimetría (1er orden)",
        "min": -3.0,
        "max": 3.0,
        "default": 0.05,
        "step": 0.05,
        "help": "Asimetría de la distribución de intensidades del vóxel. "
        "Positiva y elevada en tejido tumoral por necrosis.",
    },
}

# Ordered list of feature keys (matches CLINICAL_NUMERIC_FEATURES order)
FEATURE_ORDER = [
    "age_value",
    "smoking_history",
    "cea_level_ng_ml",
    "hemoglobin_g_dl",
    "pyrad_adc_mean",
    "pyrad_adc_std",
    "pyrad_entropy",
    "pyrad_glcm_contrast",
    "pyrad_glcm_homogeneity",
    "pyrad_shape_sphericity",
    "pyrad_firstorder_skewness",
]


def render() -> None:
    page_header(
        "Diagnóstico Tumoral CRC",
        "Análisis predictivo mediante biomarcadores hematológicos y features "
        "radiómicas (XGBoost + SHAP · NCCN 2023 · ESGAR 2022).",
        icon="🧬",
    )

    patients = get_patients()
    if not patients:
        show_empty_state(
            "🧬",
            "No hay pacientes disponibles",
            "Registre al menos un paciente antes de ejecutar el diagnóstico tumoral.",
        )
        return

    opts = build_patient_options(patients)
    col_sel, _ = st.columns([1, 2])
    with col_sel:
        pid = opts[st.selectbox("Paciente Evaluado", list(opts.keys()))]

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input form ───────────────────────────────────────────────────────────
    with st.form("tumor_diagnostic_form"):
        st.markdown(
            "<div class='section-title'>🩻 Biomarcadores Hematológicos y Features Radiómicas</div>",
            unsafe_allow_html=True,
        )

        # Render in two-column grid
        input_values: dict[str, float] = {}
        keys = FEATURE_ORDER
        for i in range(0, len(keys), 2):
            cols = st.columns(2, gap="medium")
            for j, col in enumerate(cols):
                if i + j >= len(keys):
                    break
                feat = keys[i + j]
                cfg = FEATURE_CFG[feat]
                with col:
                    if cfg.get("type") == "select":
                        sel = st.selectbox(
                            cfg["label"],
                            options=cfg["options"],
                            index=cfg["options"].index(int(cfg["default"])),
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
        submitted = st.form_submit_button(
            "🧠 Calcular Riesgo Oncológico", type="primary", use_container_width=True
        )

    # ── Results ──────────────────────────────────────────────────────────────
    if submitted:
        with st.spinner("Computando modelo XGBoost + SHAP…"):
            payload = {
                "patient_id": pid,
                "clinical_data": input_values,
            }
            res = run_diagnosis(payload)

        if res is None:
            st.error("❌ Error de conexión con el backend.")
            return

        if "risk_level" not in res:
            st.warning("⚠️ Respuesta inesperada del servidor.")
            return

        score = res["risk_level"]["score"]
        level_name = res["risk_level"]["level"]
        lvl = "green" if score < 0.3 else "orange" if score < 0.5 else "red"
        show_result_banner(
            f"Nivel de Riesgo: {level_name}",
            f"{score:.1%}",
            lvl,
        )

        # ── Tabular detail ───────────────────────────────────────────────────
        tab_res = res.get("tabular_analysis")
        if tab_res:
            st.markdown(
                "<div class='section-title'>📋 Análisis Clínico-Radiómico</div>",
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Probabilidad de Malignidad",
                f"{tab_res['prediction_score']:.1%}",
            )
            col2.metric(
                "Diagnóstico",
                "⚠️ ALTO RIESGO" if tab_res["high_risk"] else "✅ BAJO RIESGO",
            )
            col3.metric(
                "CEA",
                f"{input_values.get('cea_level_ng_ml', 0):.1f} ng/mL",
                delta="Elevado"
                if input_values.get("cea_level_ng_ml", 0) > 5
                else "Normal",
            )

            # Risk factors
            factors = tab_res.get("top_risk_factors", [])
            if factors:
                st.markdown("**Factores de riesgo detectados:**")
                for name, val in factors:
                    st.markdown(f"- 🔴 **{name}**: `{val:.2f}`")

        # ── Radiomic summary ─────────────────────────────────────────────────
        with st.expander("🔬 Detalle de Features Radiómicas", expanded=False):
            r1, r2, r3, r4 = st.columns(4)
            r1.metric(
                "ADC Medio",
                f"{input_values.get('pyrad_adc_mean', 0):.0f} µm²/s",
                delta="Restringido"
                if input_values.get("pyrad_adc_mean", 0) < 1300
                else "Normal",
                delta_color="inverse",
            )
            r2.metric(
                "Entropía GLCM",
                f"{input_values.get('pyrad_entropy', 0):.2f}",
                delta="Alta" if input_values.get("pyrad_entropy", 0) > 6 else "Normal",
                delta_color="inverse",
            )
            r3.metric(
                "Homogeneidad",
                f"{input_values.get('pyrad_glcm_homogeneity', 0):.3f}",
                delta="Baja"
                if input_values.get("pyrad_glcm_homogeneity", 0) < 0.45
                else "Normal",
                delta_color="inverse",
            )
            r4.metric(
                "Esfericidad",
                f"{input_values.get('pyrad_shape_sphericity', 0):.3f}",
                delta="Irregular"
                if input_values.get("pyrad_shape_sphericity", 0) < 0.65
                else "Regular",
                delta_color="inverse",
            )

        # ── Recommendations ──────────────────────────────────────────────────
        recs = res.get("recommendations", [])
        if recs:
            st.markdown(
                "<div class='section-title'>📌 Recomendaciones Clínicas</div>",
                unsafe_allow_html=True,
            )
            for rec in recs:
                st.markdown(rec)

        # ── Reference note ───────────────────────────────────────────────────
        st.caption(
            "Modelo XGBoost entrenado sobre dataset sintético CRC (n=334.994) · "
            "AUC-ROC 0.877 · Recall ≥ 0.90 · "
            "Referencias: NCCN 2023, ESGAR 2022, Duffy et al. 2021"
        )


render()
