"""
🌿 Endo-AID — Main Entry Point
"""

from components.cards import module_card, page_header, stat_card
from components.loading import show_splash_screen
from components.sidebar import render_sidebar
import streamlit as st
from utils.api_client import get_patients
from utils.helpers import apply_custom_css

st.set_page_config(
    page_title="Endo-AID",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()

# ── Splash Screen ──
if st.session_state.get("backend_unreachable", False):
    st.error(
        "🚫 Could not initialize the application. "
        "Ensure the backend is running on http://localhost:8000 and refresh."
    )
    st.stop()

if "app_loaded" not in st.session_state:
    ok = show_splash_screen()
    if not ok:
        st.stop()
    else:
        st.rerun()

# ── Shared Sidebar ──
render_sidebar()


# ── Main Content ──
def main() -> None:
    page_header(
        "Clinical Decision Support System",
        "AI-powered gastrointestinal analysis: image classification, "
        "risk scoring, and patient tracking.",
        icon="🌿",
    )

    patients = get_patients()
    num_patients = str(len(patients)) if patients else "0"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(stat_card("👥", num_patients, "Patients"), unsafe_allow_html=True)
    with col2:
        st.markdown(stat_card("🔬", "3", "AI Models"), unsafe_allow_html=True)
    with col3:
        st.markdown(stat_card("🧪", "7", "Biomarkers"), unsafe_allow_html=True)
    with col4:
        st.markdown(stat_card("📈", "98.2%", "Accuracy"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Explore Modules")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            module_card(
                "👥",
                "Patients & History",
                "Clinical database, patient profiles, and longitudinal "
                "AI risk tracking.",
            ),
            unsafe_allow_html=True,
        )
        st.page_link("pages/01_Patients.py", label="Open Patient Records", icon="📋")

    with c2:
        st.markdown(
            module_card(
                "🫁",
                "Smoking Triage",
                "Indirect detection of smoking habit using clinical biomarkers. "
                "Prioritization of endoscopic screening without direct patient questioning.",
            ),
            unsafe_allow_html=True,
        )
        st.page_link(
            "pages/02_Smoking_triage.py",
            label="Execute Smoking Triage",
            icon="🫁",
        )

    with c3:
        st.markdown(
            module_card(
                "🧬",
                "Oncological Diagnosis CRC",
                "Predictive analysis of CRC: hematological biomarkers + "
                "radiomic features (XGBoost · SHAP · NCCN 2023).",
            ),
            unsafe_allow_html=True,
        )
        st.page_link(
            "pages/03_Tumoral_diagnostic.py",
            label="Execute Oncological Diagnosis",
            icon="🧬",
        )

    with c4:
        st.markdown(
            module_card(
                "🔬",
                "Endoscopic AI Analysis",
                "Deep learning classification, Grad-CAM explainability "
                "and U-Net polyp segmentation.",
            ),
            unsafe_allow_html=True,
        )
        st.page_link(
            "pages/04_Endoscopy_analyzer.py",
            label="Execute Endoscopic AI Analysis",
            icon="🔬",
        )


if __name__ == "__main__":
    main()
