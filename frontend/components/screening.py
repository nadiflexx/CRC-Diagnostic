import requests  # type: ignore
import streamlit as st

from frontend.assets_ui import page_header, show_banner

API_URL = "http://localhost:8000"


def render():
    page_header(
        "Cribado Clínico y Laboratorio",
        "Análisis tabular predictivo mediante marcadores tumorales y hemograma.",
    )

    try:
        patients = requests.get(f"{API_URL}/patients/").json()
    except requests.exceptions.ConnectionError:
        patients = []

    if not patients:
        st.warning("⚠️ No hay pacientes.")
        return

    opts = {
        f"{p['first_name']} {p['last_name']} (ID: {p['id']})": p["id"] for p in patients
    }

    col_sel, _ = st.columns([1, 2])
    with col_sel:
        pid = opts[st.selectbox("Paciente Evaluado", list(opts.keys()))]

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("tabular_form"):
        st.markdown(
            "<div class='section-title'>Inputs Médicos</div>", unsafe_allow_html=True
        )
        c1, c2, c3, c4 = st.columns(4)
        cea = c1.number_input("CEA [ng/mL]", 0.0, 200.0, 2.5)
        ca19 = c2.number_input("CA 19-9 [U/mL]", 0.0, 500.0, 15.0)
        hemo = c3.number_input("Hemoglobina [g/dL]", 0.0, 25.0, 14.5)
        hema = c4.number_input("Hematocrito [%]", 0.0, 70.0, 42.0)

        st.markdown("<br>", unsafe_allow_html=True)
        f1, f2 = st.columns(2)
        fit = f1.checkbox("Test Inmunoquímico (FIT) Positivo")
        fobt = f2.checkbox("Test Sangre Oculta (FOBT) Positivo")

        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("Ejecutar Modelo Predictivo", type="primary")

    if submit:
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
        res = requests.post(f"{API_URL}/diagnosis/run", json=body).json()

        score = res.get("risk_level", {}).get("score", 0)
        lvl = "green" if score < 0.3 else "orange" if score < 0.5 else "red"
        show_banner(
            f"Nivel de Riesgo Tabular: {res['risk_level']['level']}",
            f"{score:.1%}",
            lvl,
        )
