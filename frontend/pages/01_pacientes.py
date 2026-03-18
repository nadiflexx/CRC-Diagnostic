"""
Patient Management & Clinical History.
"""

from datetime import date

from components.banners import show_empty_state
from components.cards import page_header, render_profile_card, render_section_card
from components.charts import plot_risk_timeline
from components.sidebar import render_sidebar
import pandas as pd
import streamlit as st
from utils.api_client import get_patient_history, get_patients
from utils.helpers import apply_custom_css

st.set_page_config(page_title="Pacientes · Endo-AID", page_icon="🌿", layout="wide")
apply_custom_css()
render_sidebar()


def render() -> None:
    page_header(
        "Pacientes e Historial",
        "Base de datos clínica y seguimiento longitudinal del riesgo IA.",
        icon="👥",
    )

    tab_search, tab_new = st.tabs(["📋 Búsqueda y Perfil", "➕ Nuevo Registro"])

    with tab_search:
        patients = get_patients()

        if not patients:
            show_empty_state(
                "👥",
                "Sin pacientes registrados",
                "Utilice la pestaña 'Nuevo Registro' para añadir el primer paciente.",
            )
            return

        opts = {
            f"{p['first_name']} {p['last_name']}  ·  ID {p['id']}": p for p in patients
        }

        col_sel, _ = st.columns([1, 3])
        with col_sel:
            selected_key = st.selectbox("Buscar Paciente", list(opts.keys()))
        patient = opts[selected_key]

        st.markdown("<br>", unsafe_allow_html=True)

        age = (date.today() - date.fromisoformat(patient["date_of_birth"])).days // 365

        render_profile_card(
            title="🏥 Ficha del Paciente",
            fields=[
                ("Nombre Completo", f"{patient['first_name']} {patient['last_name']}"),
                ("Edad", f"{age} años"),
                ("Sexo", patient.get("gender", "N/A").title()),
                ("IMC", f"{patient.get('bmi', 0):.1f}"),
                ("Tabaco", patient.get("smoking_status", "N/A").title()),
                ("Alcohol", patient.get("alcohol_consumption", "N/A").title()),
            ],
        )

        history = get_patient_history(patient["id"])

        if history:
            df = pd.DataFrame(history)

            render_section_card(
                title="📈 Evolución de Riesgo (IA)",
                inner_html="<div id='risk-chart-placeholder'></div>",
            )

            fig = plot_risk_timeline(df)
            st.plotly_chart(fig, use_container_width=True)

            latest_score = df.iloc[-1]["multimodal_score"]
            color = (
                "#067a5f"
                if latest_score < 0.3
                else "#F59E0B"
                if latest_score < 0.5
                else "#EF4444"
            )
            st.markdown(
                f"""
                <div style="text-align:right; margin-top:-0.5rem;">
                    <span style="
                        background:{color}15; color:{color}; font-weight:700;
                        padding: 4px 14px; border-radius: 20px; font-size: 0.85rem;
                        border: 1px solid {color}30;
                    ">Último Score: {latest_score:.1%}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            show_empty_state(
                "📊",
                "Sin historial registrado",
                "Los diagnósticos aparecerán aquí como línea temporal.",
            )

    with tab_new:
        st.info("🚧 Formulario de registro — integración pendiente con API de alta.")

        with st.form("new_patient_form"):
            c1, c2 = st.columns(2)
            c1.text_input("Nombre", placeholder="María")
            c2.text_input("Apellidos", placeholder="García López")

            c3, c4, c5 = st.columns(3)
            c3.date_input("Fecha de Nacimiento")
            c4.selectbox("Sexo", ["Masculino", "Femenino", "Otro"])
            c5.number_input("IMC", 10.0, 60.0, 24.0, step=0.1)

            c6, c7 = st.columns(2)
            c6.selectbox("Tabaquismo", ["Nunca", "Ex-fumador", "Activo"])
            c7.selectbox("Consumo Alcohol", ["Nunca", "Moderado", "Alto"])

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Registrar Paciente", type="primary")
            if submitted:
                st.success("✅ Paciente registrado correctamente (demo).")


render()
