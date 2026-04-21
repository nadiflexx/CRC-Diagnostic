"""
Smoking Risk Triage · Endo-AID

Indirect detection of smoking habits from haematological and anthropometric
biomarkers without directly questioning the patient. Supports endoscopic
screening prioritisation decisions.
"""

from __future__ import annotations

from components.banners import show_result_banner
from components.cards import page_header
from components.sidebar import render_sidebar
import streamlit as st
from utils.api_client import run_smoking_triage
from utils.helpers import apply_custom_css

st.set_page_config(
    page_title="Smoking Risk Triage · Endo-AID",
    page_icon="🌿",
    layout="wide",
)
apply_custom_css()
render_sidebar()

_RISK_ICON = {"LOW": "🟢", "MODERATE": "🟠", "HIGH": "🔴"}

_RISK_LABEL = {
    "LOW": "Low Risk — No significant smoking indicators",
    "MODERATE": "Moderate Risk — Possible tobacco exposure",
    "HIGH": "High Risk — Profile consistent with active smoking",
}

_RISK_LEVEL_MAP = {
    "LOW": "green",
    "MODERATE": "orange",
    "HIGH": "red",
}

_FEATURE_HELP = {
    "sex": "Patient's biological sex.",
    "age": "Age in full years.",
    "height": "Height in centimetres.",
    "weight": "Body weight in kilograms.",
    "BMI": "Body Mass Index (kg/m²). Auto-calculated from weight and height.",
    "waistline": "Waist circumference in cm measured at the umbilical level.",
    "triglyceride": "Blood triglycerides (mg/dL). Smoking raises this marker.",
    "HDL_chole": "HDL-Cholesterol (mg/dL). Smoking reduces the protective cholesterol.",
    "LDL_chole": "LDL-Cholesterol (mg/dL).",
    "hemoglobin": "Haemoglobin (g/dL). Smokers frequently show elevated values.",
    "waist_height_ratio": "Waist-to-height ratio. Leave at 0 to calculate automatically.",
    "hemoglobin_per_height": "Haemoglobin-to-height ratio. Leave at 0 to calculate automatically.",
}


def render() -> None:
    """
    Main entry point for the Smoking Risk Triage page.

    Renders the clinical context banner, the biomarker input form, and
    triggers inference via the backend when the form is submitted.
    """
    page_header(
        "Smoking Risk Triage",
        "Indirect detection of smoking habit from haematological and "
        "anthropometric biomarkers. Supports endoscopic screening decisions "
        "without direct patient questioning.",
        icon="🫁",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    _render_clinical_context()
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("smoking_triage_form"):
        _render_form_header()
        input_values = _render_input_form()
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🔍 Run Smoking Triage",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        _run_triage(input_values)


def _render_clinical_context() -> None:
    """
    Render an informational banner explaining the clinical purpose of the module.

    Describes the relationship between smoking and colorectal cancer risk,
    the indirect detection approach, and model performance metrics.
    """
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #f0f7f4 0%, #e4f2ed 100%);
            border: 1px solid #a8d5c2;
            border-left: 5px solid #5a8d7f;
            border-radius: 10px;
            padding: 1.1rem 1.5rem;
            font-size: 0.88rem;
            color: #2d5a4e;
            line-height: 1.7;
        ">
            <b>🎯 Clinical Objective:</b> Smoking increases the risk of colorectal
            adenomas and colon cancer. This module analyses the patient's
            <b>biometric and metabolic profile</b> to estimate the probability
            of active smoking, supporting <b>endoscopic screening prioritisation</b>
            for patients who do not disclose or minimise their smoking habit.<br><br>
            <b>Model:</b> Reverse Logic Tabular (XGBoost · Random Forest · LightGBM)
            trained on a Korean population dataset (n ≈ 991,000).
            F1 = 0.79 · AUC-ROC = 0.83 on the held-out test set.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_form_header() -> None:
    """Render the section title and caption above the biomarker input form."""
    st.markdown(
        "<div class='section-title'>🩺 Patient Clinical Biomarkers</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Enter values obtained from routine blood work. "
        "Fields marked with * are required. "
        "Derived features are calculated automatically when left at 0."
    )
    st.markdown("<br>", unsafe_allow_html=True)


def _render_input_form() -> dict:
    """
    Render the three-column biomarker input form.

    Columns cover demographics / anthropometrics, metabolic markers,
    and optional derived engineered features. BMI is pre-calculated
    from weight and height when not manually overridden.

    Returns:
        Dict mapping feature names to their float values, ready to POST
        to the ``/diagnosis/smoking-triage`` endpoint.
    """
    col_demo, col_meta, col_eng = st.columns(3, gap="large")

    with col_demo:
        st.markdown(
            "<p style='font-weight:700; color:#2d5a4e; font-size:0.9rem;"
            "border-bottom:2px solid #a8d5c2; padding-bottom:0.4rem;"
            "margin-bottom:1rem;'>👤 Demographics & Anthropometrics</p>",
            unsafe_allow_html=True,
        )
        sex = st.selectbox(
            "Sex *",
            options=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male",
            help=_FEATURE_HELP["sex"],
        )
        age = st.number_input(
            "Age (years) *",
            min_value=18,
            max_value=120,
            value=50,
            step=1,
            help=_FEATURE_HELP["age"],
        )
        height = st.number_input(
            "Height (cm) *",
            min_value=100.0,
            max_value=250.0,
            value=170.0,
            step=0.5,
            help=_FEATURE_HELP["height"],
        )
        weight = st.number_input(
            "Weight (kg) *",
            min_value=30.0,
            max_value=200.0,
            value=75.0,
            step=0.5,
            help=_FEATURE_HELP["weight"],
        )
        bmi_auto = round(weight / ((height / 100) ** 2), 2)
        BMI = st.number_input(
            "BMI (kg/m²)",
            min_value=10.0,
            max_value=70.0,
            value=bmi_auto,
            step=0.01,
            help=_FEATURE_HELP["BMI"],
        )

    with col_meta:
        st.markdown(
            "<p style='font-weight:700; color:#2d5a4e; font-size:0.9rem;"
            "border-bottom:2px solid #a8d5c2; padding-bottom:0.4rem;"
            "margin-bottom:1rem;'>🧪 Metabolic Markers</p>",
            unsafe_allow_html=True,
        )
        waistline = st.number_input(
            "Waist circumference (cm) *",
            min_value=40.0,
            max_value=180.0,
            value=88.0,
            step=0.5,
            help=_FEATURE_HELP["waistline"],
        )
        triglyceride = st.number_input(
            "Triglycerides (mg/dL) *",
            min_value=20.0,
            max_value=1000.0,
            value=120.0,
            step=1.0,
            help=_FEATURE_HELP["triglyceride"],
        )
        HDL_chole = st.number_input(
            "HDL-Cholesterol (mg/dL) *",
            min_value=5.0,
            max_value=200.0,
            value=55.0,
            step=0.5,
            help=_FEATURE_HELP["HDL_chole"],
        )
        LDL_chole = st.number_input(
            "LDL-Cholesterol (mg/dL) *",
            min_value=10.0,
            max_value=400.0,
            value=110.0,
            step=0.5,
            help=_FEATURE_HELP["LDL_chole"],
        )
        hemoglobin = st.number_input(
            "Haemoglobin (g/dL) *",
            min_value=4.0,
            max_value=25.0,
            value=15.0,
            step=0.1,
            help=_FEATURE_HELP["hemoglobin"],
        )

    with col_eng:
        st.markdown(
            "<p style='font-weight:700; color:#2d5a4e; font-size:0.9rem;"
            "border-bottom:2px solid #a8d5c2; padding-bottom:0.4rem;"
            "margin-bottom:1rem;'>⚙️ Engineered Features</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:0.82rem; color:#6c757d; margin-bottom:1rem;'>"
            "💡 Leave at <b>0</b> to calculate automatically from the values "
            "entered in the previous columns.</p>",
            unsafe_allow_html=True,
        )
        waist_height_ratio = st.number_input(
            "Waist-to-Height Ratio",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.001,
            format="%.3f",
            help=_FEATURE_HELP["waist_height_ratio"],
        )
        hemoglobin_per_height = st.number_input(
            "Haemoglobin / Height",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.0001,
            format="%.4f",
            help=_FEATURE_HELP["hemoglobin_per_height"],
        )

        whr_preview = (
            waist_height_ratio if waist_height_ratio > 0 else waistline / height
        )
        hph_preview = (
            hemoglobin_per_height if hemoglobin_per_height > 0 else hemoglobin / height
        )
        st.markdown(
            f"""
            <div style="
                background:#f8f9fa; border:1px solid #dee2e6;
                border-radius:8px; padding:0.85rem 1rem;
                margin-top:1rem; font-size:0.82rem; color:#495057;
            ">
                <b>Calculated values preview:</b><br>
                <span style="font-family:monospace;">
                    WHR = {whr_preview:.4f}<br>
                    Hgb/H = {hph_preview:.5f}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "sex": int(sex),
        "age": float(age),
        "height": float(height),
        "weight": float(weight),
        "BMI": float(BMI),
        "waistline": float(waistline),
        "triglyceride": float(triglyceride),
        "HDL_chole": float(HDL_chole),
        "LDL_chole": float(LDL_chole),
        "hemoglobin": float(hemoglobin),
        "waist_height_ratio": float(waist_height_ratio),
        "hemoglobin_per_height": float(hemoglobin_per_height),
    }


def _run_triage(input_values: dict) -> None:
    """
    Call the backend smoking triage endpoint and render the results.

    Displays an error message if the backend is unreachable or returns None.

    Args:
        input_values: Dict of feature names to float values as returned by
            ``_render_input_form``.
    """
    with st.spinner("Analysing biometric profile with Reverse Logic model…"):
        res = run_smoking_triage(input_values)

    if res is None:
        st.error(
            "❌ Could not obtain a response from the server. "
            "Please verify the backend is running."
        )
        return

    _render_results(res, input_values)


def _render_results(res: dict, input_values: dict) -> None:
    """
    Render the full triage results panel.

    Displays the result banner, probability metrics, visual gauge,
    clinical indicators table, input summary, clinical recommendation,
    and a methodological footnote.

    Args:
        res: Response dict from the ``/diagnosis/smoking-triage`` endpoint.
        input_values: Original input dict used to derive computed features
            for the summary table.
    """
    st.markdown("<br>", unsafe_allow_html=True)

    risk_level = res["risk_level"]
    icon = _RISK_ICON.get(risk_level, "⚪")
    label = _RISK_LABEL.get(risk_level, "Result")
    banner_level = _RISK_LEVEL_MAP.get(risk_level, "green")

    show_result_banner(f"{icon} {label}", res["confidence"], banner_level)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'>📊 Model Probabilities</div>",
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Smoking Probability", f"{res['smoking_probability']:.1%}")
    col2.metric("Non-Smoking Probability", f"{res['non_smoking_probability']:.1%}")
    col3.metric(
        "Prediction",
        "🚬 Smoker" if res["predicted_smoker"] else "✅ Non-Smoker",
    )
    col4.metric("Inference Backend", f"⚡ {res['backend']}")

    st.markdown("<br>", unsafe_allow_html=True)

    _render_probability_gauge(res["smoking_probability"], res["risk_color"], risk_level)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")
    with col_left:
        _render_clinical_indicators(res["top_indicators"])
    with col_right:
        _render_input_summary(input_values)

    st.markdown("<br>", unsafe_allow_html=True)

    _render_clinical_recommendation(res)

    st.markdown("<br>", unsafe_allow_html=True)

    st.caption(
        "Reverse Logic Model · XGBoost + Random Forest + LightGBM · "
        "Dataset: National Health Insurance Service – Health Screening (NHIS-HNS) · "
        "F1 = 0.79 · AUC-ROC = 0.83 · "
        "This analysis is a clinical decision-support tool and does not "
        "replace professional medical judgement."
    )


def _render_probability_gauge(prob: float, color: str, level: str) -> None:
    """
    Render a visual horizontal gauge bar for the smoking probability.

    The bar is overlaid on three coloured zones (LOW / MODERATE / HIGH)
    with a vertical marker at the current probability position.

    Args:
        prob: Smoking probability in [0, 1].
        color: Hex colour string corresponding to the risk level.
        level: Risk level string (``"LOW"``, ``"MODERATE"``, or ``"HIGH"``).
    """
    st.markdown(
        "<div class='section-title'>📈 Smoking Risk Level</div>",
        unsafe_allow_html=True,
    )
    pct = prob * 100
    st.markdown(
        f"""
        <div style="position:relative; margin-bottom:0.5rem;">
            <div style="
                display:flex; height:28px; border-radius:14px;
                overflow:hidden; border:1px solid #dee2e6;
            ">
                <div style="width:45%; background:rgba(56,142,60,0.25);"></div>
                <div style="width:25%; background:rgba(245,124,0,0.25);"></div>
                <div style="width:30%; background:rgba(211,47,47,0.25);"></div>
            </div>
            <div style="
                position:absolute; top:0; left:0;
                width:{pct:.1f}%; height:28px; border-radius:14px;
                background:{color}; opacity:0.85;
                transition:width 0.5s ease;
            "></div>
            <div style="
                position:absolute; top:-6px;
                left:calc({pct:.1f}% - 1px);
                width:2px; height:40px;
                background:{color};
            "></div>
        </div>
        <div style="
            display:flex; justify-content:space-between;
            font-size:0.75rem; color:#6c757d;
            margin-top:0.25rem; padding:0 2px;
        ">
            <span>🟢 Low (&lt;45%)</span>
            <span>🟠 Moderate (45–70%)</span>
            <span>🔴 High (&gt;70%)</span>
        </div>
        <div style="text-align:center; margin-top:0.5rem;">
            <span style="font-size:1.4rem; font-weight:700; color:{color};">
                {pct:.1f}%
            </span>
            <span style="font-size:0.85rem; color:#6c757d; margin-left:0.5rem;">
                probability of active smoking
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_clinical_indicators(indicators: list[dict]) -> None:
    """
    Render a list of clinical indicator rows with flag icons and values.

    Each indicator is displayed as a coloured card: red when the flag is
    raised (abnormal range associated with smoking), green otherwise.

    Args:
        indicators: List of indicator dicts, each containing ``name``,
            ``value``, ``flag`` (bool), and ``note`` keys.
    """
    st.markdown(
        "<div class='section-title'>🔬 Analysed Clinical Indicators</div>",
        unsafe_allow_html=True,
    )

    for ind in indicators:
        flagged = ind.get("flag", False)
        flag_icon = "🔴" if flagged else "🟢"
        flag_color = "#D32F2F" if flagged else "#388E3C"
        flag_bg = "rgba(211,47,47,0.06)" if flagged else "rgba(56,142,60,0.06)"
        flag_border = "#D32F2F" if flagged else "#388E3C"

        st.markdown(
            f"""
            <div style="
                background:{flag_bg};
                border-left: 4px solid {flag_border};
                border-radius: 6px;
                padding: 0.55rem 0.9rem;
                margin-bottom: 0.45rem;
                display: flex;
                align-items: center;
                gap: 0.75rem;
            ">
                <span style="font-size:1.1rem;">{flag_icon}</span>
                <div style="flex:1;">
                    <div style="font-weight:600; font-size:0.86rem; color:#2d3748;">
                        {ind["name"]}
                    </div>
                    <div style="font-size:0.8rem; color:#6c757d;">{ind["note"]}</div>
                </div>
                <div style="
                    font-weight:700; font-size:0.9rem;
                    color:{flag_color}; min-width:80px; text-align:right;
                ">{ind["value"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_input_summary(input_values: dict) -> None:
    """
    Render a compact read-only table summarising the submitted input values.

    Derived features (WHR, Hgb/H) are recalculated here if they were
    left at 0 in the form.

    Args:
        input_values: Dict of feature names to float values as submitted.
    """
    st.markdown(
        "<div class='section-title'>📋 Submitted Data</div>",
        unsafe_allow_html=True,
    )

    whr = (
        input_values["waist_height_ratio"]
        if input_values["waist_height_ratio"] > 0
        else input_values["waistline"] / input_values["height"]
    )
    hph = (
        input_values["hemoglobin_per_height"]
        if input_values["hemoglobin_per_height"] > 0
        else input_values["hemoglobin"] / input_values["height"]
    )

    rows = [
        ("Sex", "Male" if input_values["sex"] == 1 else "Female"),
        ("Age", f"{int(input_values['age'])} years"),
        ("Height", f"{input_values['height']:.1f} cm"),
        ("Weight", f"{input_values['weight']:.1f} kg"),
        ("BMI", f"{input_values['BMI']:.2f} kg/m²"),
        ("Waist", f"{input_values['waistline']:.1f} cm"),
        ("Triglycerides", f"{input_values['triglyceride']:.1f} mg/dL"),
        ("HDL-Chol.", f"{input_values['HDL_chole']:.1f} mg/dL"),
        ("LDL-Chol.", f"{input_values['LDL_chole']:.1f} mg/dL"),
        ("Haemoglobin", f"{input_values['hemoglobin']:.2f} g/dL"),
        ("WHR (calc.)", f"{whr:.4f}"),
        ("Hgb/H (calc.)", f"{hph:.5f}"),
    ]

    table_rows = "".join(
        f"""
        <tr>
            <td style="padding:0.3rem 0.6rem; font-size:0.82rem;
                       color:#6c757d; font-weight:500;">{label}</td>
            <td style="padding:0.3rem 0.6rem; font-size:0.82rem;
                       color:#2d3748; font-weight:600;
                       text-align:right;">{value}</td>
        </tr>
        """
        for label, value in rows
    )

    st.markdown(
        f"""
        <div style="
            background:#f8f9fa; border:1px solid #dee2e6;
            border-radius:10px; overflow:hidden;
        ">
            <table style="width:100%; border-collapse:collapse;">
                <tbody>{table_rows}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_clinical_recommendation(res: dict) -> None:
    """
    Render a colour-coded clinical recommendation box based on the risk level.

    The box title and text colour reflect the urgency:
    HIGH = red, MODERATE = orange, LOW = green.

    Args:
        res: Response dict from the smoking triage endpoint, containing
            ``risk_level`` and ``recommendation`` keys.
    """
    level = res["risk_level"]
    color_map = {
        "HIGH": ("#FFEBEE", "#D32F2F", "🔴", "Clinical Recommendation — HIGH PRIORITY"),
        "MODERATE": ("#FFF3E0", "#F57C00", "🟠", "Clinical Recommendation — FOLLOW-UP"),
        "LOW": ("#E8F5E9", "#388E3C", "🟢", "Clinical Recommendation — ROUTINE"),
    }
    bg, border, icon, title = color_map.get(level, color_map["LOW"])

    st.markdown(
        f"""
        <div style="
            background:{bg}; border:2px solid {border};
            border-radius:12px; padding:1.25rem 1.5rem;
        ">
            <div style="
                font-weight:700; font-size:0.95rem; color:{border};
                margin-bottom:0.75rem; display:flex;
                align-items:center; gap:0.5rem;
            ">
                {icon} {title}
            </div>
            <p style="margin:0; font-size:0.88rem; color:#333; line-height:1.7;">
                {res["recommendation"]}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


render()
