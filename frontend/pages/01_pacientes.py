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
from utils.api_client import create_patient, get_patient_history, get_patients
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

    # ── TAB BÚSQUEDA ──
    with tab_search:
        patients = get_patients()

        if not patients:
            show_empty_state(
                "👥",
                "Sin pacientes registrados",
                "Utilice la pestaña 'Nuevo Registro' para añadir el primer paciente.",
            )

        else:
            opts = {
                f"{p['first_name']} {p['last_name']}  ·  ID {p['id']}": p
                for p in patients
            }

            col_sel, _ = st.columns([1, 3])
            with col_sel:
                selected_key = st.selectbox("Buscar Paciente", list(opts.keys()))
            patient = opts[selected_key]

            st.markdown("<br>", unsafe_allow_html=True)

            age = (
                date.today() - date.fromisoformat(patient["date_of_birth"])
            ).days // 365

            render_profile_card(
                title="🏥 Ficha del Paciente",
                fields=[
                    (
                        "Nombre Completo",
                        f"{patient['first_name']} {patient['last_name']}",
                    ),
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
                st.plotly_chart(fig, width="stretch")

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
                            padding: 4px 14px; border-radius: 20px;
                            font-size: 0.85rem;
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

    # ── TAB NUEVO REGISTRO ──
    with tab_new, st.form("new_patient_form"):
        st.subheader("📝 Datos del Paciente")

        c1, c2 = st.columns(2)
        first_name = c1.text_input("Nombre", placeholder="María")
        last_name = c2.text_input("Apellidos", placeholder="García López")

        c3, c4 = st.columns(2)
        date_of_birth = c3.date_input(
            "Fecha de Nacimiento",
            value=date(1980, 1, 1),
            min_value=date(1900, 1, 1),
            max_value=date.today(),
        )
        gender = c4.selectbox("Sexo", ["male", "female", "other"])

        c5, c6 = st.columns(2)
        height_cm = c5.number_input("Altura (cm)", 100.0, 250.0, 170.0, step=0.5)
        weight_kg = c6.number_input("Peso (kg)", 30.0, 300.0, 70.0, step=0.5)

        # Mostrar BMI calculado en tiempo real
        if height_cm > 0:
            bmi_preview = weight_kg / ((height_cm / 100) ** 2)
            st.caption(f"📊 IMC estimado: **{bmi_preview:.1f}**")

        c7, c8 = st.columns(2)
        smoking = c7.selectbox(
            "Tabaquismo",
            ["never", "former", "current"],
            format_func=lambda x: {
                "never": "Nunca",
                "former": "Ex-fumador",
                "current": "Activo",
            }[x],
        )
        alcohol = c8.selectbox(
            "Consumo Alcohol",
            ["never", "moderate", "heavy"],
            format_func=lambda x: {
                "never": "Nunca",
                "moderate": "Moderado",
                "heavy": "Alto",
            }[x],
        )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Registrar Paciente", type="primary")

        if submitted:
            if not first_name.strip() or not last_name.strip():
                st.error("❌ Nombre y apellidos son obligatorios.")
            else:
                patient_data = {
                    "first_name": first_name.strip(),
                    "last_name": last_name.strip(),
                    "date_of_birth": date_of_birth.isoformat(),
                    "gender": gender,
                    "height_cm": height_cm,
                    "weight_kg": weight_kg,
                    "smoking_status": smoking,
                    "alcohol_consumption": alcohol,
                }

                result = create_patient(patient_data)

                if result:
                    st.success(
                        f"✅ Paciente **{first_name} {last_name}** "
                        f"registrado con ID **{result.get('id', '?')}**."
                    )
                    st.balloons()


render()
