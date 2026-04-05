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
if "app_loaded" not in st.session_state:
    show_splash_screen()
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

    c1, c2, c3 = st.columns(3)

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

    with c3:
        st.markdown(
            module_card(
                "📋",
                "General Screening",
                "Generic risk assessment using clinical and demographic factors. "
                "AI-powered CRC prediction.",
            ),
            unsafe_allow_html=True,
        )
        st.page_link("pages/02_Cribado_General.py", label="Run General Screening", icon="🧠")

    st.markdown("<br>", unsafe_allow_html=True)
    
    c4 = st.columns(1)[0]
    with c4:
        st.markdown(
            module_card(
                "🔬",
                "Endoscopy AI",
                "Deep learning classification, Grad-CAM explainability, "
                "and U-Net segmentation.",
            ),
            unsafe_allow_html=True,
        )
        st.page_link("pages/03_endoscopia.py", label="Launch Inference", icon="🧠")

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
