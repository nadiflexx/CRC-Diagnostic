"""
Endo-AID main dashboard.
"""

import streamlit as st

from components.cards import module_card, page_header, stat_card
from components.loading import show_splash_screen
from components.sidebar import render_sidebar
from utils.api_client import get_patients
from utils.helpers import apply_custom_css

st.set_page_config(
    page_title="Endo-AID",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()

if "app_loaded" not in st.session_state:
    show_splash_screen()
    st.rerun()

render_sidebar()


def main() -> None:
    page_header(
        "Clinical Decision Support System",
        "AI-powered gastrointestinal analysis: image classification, risk scoring, patient tracking, and model analytics.",
        icon="🌿",
    )

    patients = get_patients()
    num_patients = str(len(patients)) if patients else "0"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(stat_card("👥", num_patients, "Patients"), unsafe_allow_html=True)
    with col2:
        st.markdown(stat_card("🔬", "3", "AI Models"), unsafe_allow_html=True)
    with col3:
        st.markdown(stat_card("🧪", "7", "Biomarkers"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Explore Modules")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            module_card(
                "👥",
                "Patients & History",
                "Clinical database, patient profiles, and longitudinal AI risk tracking.",
            ),
            unsafe_allow_html=True,
        )
        st.page_link("pages/01_pacientes.py", label="Open Patient Records", icon="📋")

    with c2:
        st.markdown(
            module_card(
                "🧪",
                "Clinical Screening",
                "Predictive tabular analysis using tumor markers and blood counts.",
            ),
            unsafe_allow_html=True,
        )
        st.page_link("pages/02_cribado.py", label="Run Screening", icon="🔍")

    c4, c5 = st.columns(2)
    with c4:
        st.markdown(
            module_card(
                "🔬",
                "Endoscopy AI",
                "Deep learning classification, Grad-CAM explainability, and U-Net segmentation.",
            ),
            unsafe_allow_html=True,
        )
        st.page_link("pages/03_endoscopia.py", label="Launch Inference", icon="🧠")

    with c5:
        st.markdown(
            module_card(
                "📊",
                "Model Analysis",
                "Comparative visual analytics for colon tabular models, risk trends by country, and diagnostic explainability plots.",
            ),
            unsafe_allow_html=True,
        )
        st.page_link("pages/04_analysis.py", label="Open Analysis Dashboard", icon="📉")

    st.markdown(
        """
        <div style="text-align:center; color:#9dceca; font-size:0.78rem;
             padding:3rem 0 1rem; border-top:1px solid #d4eceb; margin-top:3rem;">
            Endo-AID Clinical Suite · v2.0 · For Research & Educational Use Only
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
