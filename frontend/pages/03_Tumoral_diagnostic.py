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
    page_title="Onologic Diagnositc — CRC · Endo-AID",
    page_icon="🌿",
    layout="wide",
)
apply_custom_css()
render_sidebar()

FEATURE_CFG: dict[str, FeatureConfig] = {
    "age_value": {
        "label": "Age (years)",
        "min": 18.0,
        "max": 100.0,
        "default": 55.0,
        "step": 1.0,
        "help": "Age of the patient at the time of the study.",
    },
    "smoking_history": {
        "label": "Smoking History",
        "type": "select",
        "options": [0, 1],
        "fmt": {0: "Non-smoker (0)", 1: "Smoker / Ex-smoker (1)"},
        "default": 0,
        "help": "Smoking increases baseline CEA levels in healthy individuals (Duffy et al. 2021).",
    },
    "cea_level_ng_ml": {
        "label": "CEA (ng/mL)",
        "min": 0.1,
        "max": 500.0,
        "default": 2.1,
        "step": 0.1,
        "help": "Carcinoembryonic Antigen. Normal < 3 ng/mL. "
        "Advanced tumors may exceed 20 ng/mL.",
    },
    "hemoglobin_g_dl": {
        "label": "Hemoglobin (g/dL)",
        "min": 5.0,
        "max": 20.0,
        "default": 13.5,
        "step": 0.1,
        "help": "Normal: Man 13.5–17.5 · Female 12.0–15.5 g/dL. "
        "Advanced tumors may cause anemia.",
    },
    "pyrad_adc_mean": {
        "label": "ADC Mean (µm²/s)",
        "min": 200.0,
        "max": 2500.0,
        "default": 1540.0,
        "step": 10.0,
        "help": "Coefficient of Apparent Diffusion (MRI-DWI). "
        "Tumoral tissue: < 1 300 µm²/s (restricted diffusion).",
    },
    "pyrad_adc_std": {
        "label": "ADC Standard Deviation",
        "min": 5.0,
        "max": 400.0,
        "default": 80.0,
        "step": 5.0,
        "help": "Heterogeneity of ADC within the VOI. "
        "Higher in malignant tissue due to necrosis and hypoxia.",
    },
    "pyrad_entropy": {
        "label": "Entropy GLCM",
        "min": 0.1,
        "max": 10.0,
        "default": 4.8,
        "step": 0.1,
        "help": "Textural disorder. Values > 6 are more frequent in malignant lesions.",
    },
    "pyrad_glcm_contrast": {
        "label": "Contrast GLCM",
        "min": 0.1,
        "max": 200.0,
        "default": 21.0,
        "step": 0.5,
        "help": "Local intensity variation. Higher in tumors than in healthy mucosa.",
    },
    "pyrad_glcm_homogeneity": {
        "label": "Homogeneity GLCM",
        "min": 0.01,
        "max": 1.0,
        "default": 0.72,
        "step": 0.01,
        "help": "Textural uniformity (0–1). Tumoral: 0.25–0.45 · Healthy: 0.60–0.85.",
    },
    "pyrad_shape_sphericity": {
        "label": "Sphericity (shape)",
        "min": 0.25,
        "max": 1.0,
        "default": 0.73,
        "step": 0.01,
        "help": "Geometric regularity of the VOI. Tumors are irregular (< 0.65).",
    },
    "pyrad_firstorder_skewness": {
        "label": "Skewness (1st order)",
        "min": -3.0,
        "max": 3.0,
        "default": 0.05,
        "step": 0.05,
        "help": "Asymmetry of the voxel intensity distribution. "
        "Positive and elevated in tumoral tissue due to necrosis.",
    },
}

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
        "Tumoral Diagnostic · CRC",
        "Predictive Analysis using Hematological Biomarkers and Radiomic Features "
        "(XGBoost + SHAP · NCCN 2023 · ESGAR 2022).",
        icon="🧬",
    )

    patients = get_patients()
    if not patients:
        show_empty_state(
            "🧬",
            "No patients available",
            "Register at least one patient before running the tumoral diagnostic.",
        )
        return

    opts = build_patient_options(patients)
    col_sel, _ = st.columns([1, 2])
    with col_sel:
        pid = opts[st.selectbox("Patient Evaluated", list(opts.keys()))]

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("tumor_diagnostic_form"):
        st.markdown(
            "<div class='section-title'>🩻 Hematological Biomarkers and Radiomic Features</div>",
            unsafe_allow_html=True,
        )

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
            "🧠 Calculate Oncological Risk", type="primary", width="stretch"
        )

    if submitted:
        with st.spinner("Computing XGBoost + SHAP model…"):
            payload = {
                "patient_id": pid,
                "clinical_data": input_values,
            }
            res = run_diagnosis(payload)

        if res is None:
            st.error("❌ Connection error with the backend.")
            return

        if "risk_level" not in res:
            st.warning("⚠️ Unexpected response from the server.")
            return

        score = res["risk_level"]["score"]
        level_name = res["risk_level"]["level"]
        lvl = "green" if score < 0.3 else "orange" if score < 0.5 else "red"
        show_result_banner(
            f"Oncological Risk Level: {level_name}",
            f"{score:.1%}",
            lvl,
        )

        tab_res = res.get("tabular_analysis")
        if tab_res:
            st.markdown(
                "<div class='section-title'>📋 Predictive Analysis</div>",
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Malignancy Probability",
                f"{tab_res['prediction_score']:.1%}",
            )
            col2.metric(
                "Diagnosis",
                "⚠️ HIGH RISK" if tab_res["high_risk"] else "✅ LOW RISK",
            )
            col3.metric(
                "CEA",
                f"{input_values.get('cea_level_ng_ml', 0):.1f} ng/mL",
                delta="Elevated"
                if input_values.get("cea_level_ng_ml", 0) > 5
                else "Normal",
            )

            factors = tab_res.get("top_risk_factors", [])
            if factors:
                st.markdown("**Detected Risk Factors:**")
                for name, val in factors:
                    st.markdown(f"- 🔴 **{name}**: `{val:.2f}`")

        with st.expander("🔬 Radiomic Features Detail", expanded=False):
            r1, r2, r3, r4 = st.columns(4)
            r1.metric(
                "ADC Mean",
                f"{input_values.get('pyrad_adc_mean', 0):.0f} µm²/s",
                delta="Restricted"
                if input_values.get("pyrad_adc_mean", 0) < 1300
                else "Normal",
                delta_color="inverse",
            )
            r2.metric(
                "Entropy GLCM",
                f"{input_values.get('pyrad_entropy', 0):.2f}",
                delta="High" if input_values.get("pyrad_entropy", 0) > 6 else "Normal",
                delta_color="inverse",
            )
            r3.metric(
                "Homogeneity",
                f"{input_values.get('pyrad_glcm_homogeneity', 0):.3f}",
                delta="Low"
                if input_values.get("pyrad_glcm_homogeneity", 0) < 0.45
                else "Normal",
                delta_color="inverse",
            )
            r4.metric(
                "Sphericity",
                f"{input_values.get('pyrad_shape_sphericity', 0):.3f}",
                delta="Irregular"
                if input_values.get("pyrad_shape_sphericity", 0) < 0.65
                else "Regular",
                delta_color="inverse",
            )

        recs = res.get("recommendations", [])
        if recs:
            st.markdown(
                "<div class='section-title'>📌 Clinical Recommendations</div>",
                unsafe_allow_html=True,
            )
            for rec in recs:
                st.markdown(rec)

        st.caption(
            "XGBoost Model + Temperature Scaling trained with sintétic dataset CRC · "
            "Stadification T1–T4 (NCCN 2023 · ESGAR 2022 · Duffy et al. 2021) · "
            "Recall ≥ 0.80 garanted over calibration set"
        )


render()
