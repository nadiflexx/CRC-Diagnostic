"""
Clinical Screening & Laboratory Analysis.
"""

from components.banners import show_empty_state, show_result_banner
from components.cards import page_header
from components.sidebar import render_sidebar
import streamlit as st
from utils.api_client import get_patients, run_diagnosis
from utils.helpers import apply_custom_css, build_patient_options

st.set_page_config(page_title="Cribado · Endo-AID", page_icon="🌿", layout="wide")
apply_custom_css()
render_sidebar()


def render() -> None:
    page_header(
        "Cribado Clínico y Laboratorio",
        "Análisis tabular predictivo mediante marcadores tumorales y hemograma.",
        icon="🧪",
    )

    patients = get_patients()

    if not patients:
        show_empty_state(
            "🧪",
            "No hay pacientes disponibles",
            "Registre al menos un paciente antes de ejecutar el cribado.",
        )
        return

    opts = build_patient_options(patients)

    col_sel, _ = st.columns([1, 2])
    with col_sel:
        pid = opts[st.selectbox("Paciente Evaluado", list(opts.keys()))]

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("screening_form"):
        st.markdown(
            "<div class='section-title'>🩸 Marcadores y Hemograma</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        cea = c1.number_input(
            "CEA [ng/mL]", 0.0, 200.0, 2.5, help="Antígeno carcinoembrionario"
        )
        ca19 = c2.number_input(
            "CA 19-9 [U/mL]",
            0.0,
            500.0,
            15.0,
            help="Marcador tumoral gastrointestinal",
        )
        hemo = c3.number_input("Hemoglobina [g/dL]", 0.0, 25.0, 14.5)
        hema = c4.number_input("Hematocrito [%]", 0.0, 70.0, 42.0)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            "<div class='section-title'>🧫 Tests de Detección</div>",
            unsafe_allow_html=True,
        )

        f1, f2, _, _ = st.columns(4)
        fit = f1.checkbox("Test Inmunoquímico (FIT) Positivo")
        fobt = f2.checkbox("Test Sangre Oculta (FOBT) Positivo")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🧠 Ejecutar Modelo Predictivo", type="primary"
        )

    if submitted:
        with st.spinner("Computando modelo de riesgo tabular..."):
            body = {
                "patient_id": pid,
                "clinical_data": {
                    "hemoglobin": hemo,
                    "hematocrit": hema,
                    "wbc_count": 7.0,
                    "platelet_count": 250.0,
                    "ferritin": 100.0,
                    "crp": 0.5,
                    "cea": cea,
                    "ca19_9": ca19,
                    "fobt_positive": fobt,
                    "fit_positive": fit,
                },
            }
            res = run_diagnosis(body)

        if res and "risk_level" in res:
            score = res["risk_level"]["score"]
            level_name = res["risk_level"]["level"]
            lvl = "green" if score < 0.3 else "orange" if score < 0.5 else "red"

            show_result_banner(f"Nivel de Riesgo: {level_name}", f"{score:.1%}", lvl)

            st.markdown(
                "<div class='section-title'>📋 Detalle del Análisis</div>",
                unsafe_allow_html=True,
            )

            d1, d2, d3 = st.columns(3)
            d1.metric("CEA", f"{cea} ng/mL", delta="Elevado" if cea > 5 else "Normal")
            d2.metric(
                "CA 19-9",
                f"{ca19} U/mL",
                delta="Elevado" if ca19 > 37 else "Normal",
            )
            d3.metric(
                "Hemoglobina",
                f"{hemo} g/dL",
                delta="Baja" if hemo < 12 else "Normal",
            )

        elif res is None:
            st.error("❌ Error de conexión con el backend.")
        else:
            st.warning("⚠️ Respuesta inesperada del servidor.")


render()
