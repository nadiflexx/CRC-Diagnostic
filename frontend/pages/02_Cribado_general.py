"""
Cribado Clínico · Riesgo Epidemiológico (CSV MLP model).

Collects 13 clinically accessible fields. Three additional inputs required by
the model (early_detection, incidence_rate, mortality_rate) are auto-filled
inside CsvTabularPreprocessor.build_feature_row() so the form stays clean.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from components.banners import show_empty_state, show_result_banner
from components.cards import page_header
from components.sidebar import render_sidebar
import streamlit as st
from utils.api_client import get_patients, run_diagnosis
from utils.helpers import apply_custom_css, build_patient_options

st.set_page_config(page_title="Cribado · Endo-AID", page_icon="🌿", layout="wide")
apply_custom_css()
render_sidebar()


@st.cache_resource(show_spinner="Cargando modelo de riesgo…")
def _load_csv_mlp():
    from src.config.paths import paths
    from src.data.processing.tabular_preprocessor import CsvTabularPreprocessor
    from src.models.tabular_model import CsvMlpModel

    model = CsvMlpModel.load(paths.CSV_MLP_MODEL_DIR)
    preprocessor = CsvTabularPreprocessor()
    return model, preprocessor


# ── Option constants ─────────────────────────────────────────────────────────

_YES_NO    = ["No", "Yes"]
_GENDER    = ["M", "F"]
_OBESITY   = ["Normal", "Overweight", "Obese"]
_DIET      = ["Low", "Moderate", "High"]
_ACTIVITY  = ["Low", "Moderate", "High"]
_SCREENING = ["Never", "Irregular", "Regular"]
_URBAN     = ["Urban", "Rural"]

_LABELS_GENDER   = {"M": "Hombre", "F": "Mujer"}
_LABELS_OBESITY  = {"Normal": "Normal", "Overweight": "Sobrepeso", "Obese": "Obeso"}
_LABELS_DIET     = {"Low": "Bajo (saludable)", "Moderate": "Moderado", "High": "Alto riesgo"}
_LABELS_ACTIVITY = {"Low": "Baja", "Moderate": "Moderada", "High": "Alta"}
_LABELS_SCREEN   = {"Never": "Nunca", "Irregular": "Irregular", "Regular": "Regular"}
_LABELS_URBAN    = {"Urban": "Urbano", "Rural": "Rural"}
_LABELS_YESNO    = {"Yes": "Sí", "No": "No"}

_RISK_META = {
    "Bajo":     {"color": "green",  "icon": "✅", "threshold_lo": 0,    "threshold_hi": 0.35},
    "Moderado": {"color": "orange", "icon": "⚠️", "threshold_lo": 0.35, "threshold_hi": 0.60},
    "Alto":     {"color": "red",    "icon": "🔴", "threshold_lo": 0.60, "threshold_hi": 1.0},
}


def _risk_label(prob: float) -> str:
    if prob < 0.35: return "Bajo"
    if prob < 0.60: return "Moderado"
    return "Alto"


# ── Page ─────────────────────────────────────────────────────────────────────

def render() -> None:
    page_header(
        "Cribado Clínico · Riesgo Epidemiológico",
        "Predicción de riesgo de cáncer colorrectal mediante factores "
        "epidemiológicos y clínicos · Modelo MLP (167 k pacientes).",
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

    # ══════════════════════════════════════════════════════════════════════════
    # FORM  (13 fields — all clinically accessible in a standard consultation)
    # ══════════════════════════════════════════════════════════════════════════
    with st.form("screening_form"):

        # ── Sección 1: Datos demográficos ─────────────────────────────────────
        st.markdown(
            "<div class='section-title'>👤 Datos Demográficos</div>",
            unsafe_allow_html=True,
        )
        d1, d2, d3, d4 = st.columns(4)
        age     = d1.number_input("Edad", min_value=18, max_value=100, value=55, step=1)
        gender  = d2.selectbox("Género", _GENDER, format_func=_LABELS_GENDER.get)
        urban   = d3.selectbox("Entorno geográfico", _URBAN, format_func=_LABELS_URBAN.get)
        obesity = d4.selectbox("IMC / Peso", _OBESITY, format_func=_LABELS_OBESITY.get)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Sección 2: Hábitos ────────────────────────────────────────────────
        st.markdown(
            "<div class='section-title'>🚬 Hábitos y Estilo de Vida</div>",
            unsafe_allow_html=True,
        )
        h1, h2, h3, h4 = st.columns(4)
        smoking  = h1.selectbox("Tabaquismo", _YES_NO, format_func=_LABELS_YESNO.get)
        alcohol  = h2.selectbox("Consumo de alcohol", _YES_NO, format_func=_LABELS_YESNO.get)
        diet     = h3.selectbox("Calidad de la dieta", _DIET, format_func=_LABELS_DIET.get)
        activity = h4.selectbox("Actividad física", _ACTIVITY, format_func=_LABELS_ACTIVITY.get)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Sección 3: Antecedentes clínicos ──────────────────────────────────
        st.markdown(
            "<div class='section-title'>🏥 Antecedentes Clínicos</div>",
            unsafe_allow_html=True,
        )
        a1, a2, a3, a4 = st.columns(4)
        family_history = a1.selectbox(
            "Historia familiar de CRC", _YES_NO, format_func=_LABELS_YESNO.get,
            help="Familiar de primer grado con cáncer colorrectal",
        )
        diabetes = a2.selectbox("Diabetes tipo 2", _YES_NO, format_func=_LABELS_YESNO.get)
        ibd      = a3.selectbox(
            "Enf. Inflamatoria Intestinal", _YES_NO, format_func=_LABELS_YESNO.get,
            help="Colitis ulcerosa o enfermedad de Crohn diagnosticada",
        )
        genetic  = a4.selectbox(
            "Mutación genética conocida", _YES_NO, format_func=_LABELS_YESNO.get,
            help="Lynch, FAP u otra mutación hereditaria documentada",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Sección 4: Prevención ─────────────────────────────────────────────
        st.markdown(
            "<div class='section-title'>🔬 Historial de Cribado</div>",
            unsafe_allow_html=True,
        )
        scr_col, _ = st.columns([1, 3])
        screening = scr_col.selectbox(
            "Frecuencia de cribado previo",
            _SCREENING,
            format_func=_LABELS_SCREEN.get,
            help=(
                "Regular: colonoscopia o test cada 1-3 años · "
                "Irregular: algún cribado puntual · "
                "Nunca: sin cribado previo"
            ),
        )

        # Informational note about auto-filled fields
        st.caption(
            "ℹ️ La detección temprana y las tasas epidemiológicas regionales "
            "se calculan automáticamente a partir de los datos introducidos."
        )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🧠 Ejecutar Modelo Predictivo", type="primary")

    # ══════════════════════════════════════════════════════════════════════════
    # PREDICTION
    # ══════════════════════════════════════════════════════════════════════════
    if submitted:
        # Build the 13-field input dict — the preprocessor fills the rest
        inputs = {
            "age":              age,
            "gender":           gender,
            "family_history":   family_history,
            "smoking":          smoking,
            "alcohol":          alcohol,
            "obesity":          obesity,
            "diet_risk":        diet,
            "physical_activity": activity,
            "diabetes":         diabetes,
            "ibd":              ibd,
            "genetic":          genetic,
            "screening":        screening,
            "urban_rural":      urban,
            # early_detection, incidence_rate, mortality_rate → auto-filled
        }

        with st.spinner("Computando modelo de riesgo epidemiológico…"):
            try:
                model, preprocessor = _load_csv_mlp()
                df_row, risk_score, prevention_index, access_score, age_group, lifestyle = (
                    preprocessor.build_feature_row(inputs)
                )
                prob    = float(model.predict_proba(df_row)[0])
                risk_lv = _risk_label(prob)
            except Exception as exc:
                st.error(f"❌ Error al ejecutar el modelo: {exc}")
                return

        # ── Banner principal ───────────────────────────────────────────────
        meta = _RISK_META[risk_lv]
        show_result_banner(
            f"{meta['icon']} Nivel de Riesgo: {risk_lv}",
            f"P(cáncer colorrectal) = {prob:.1%}",
            meta["color"],
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Índices derivados ──────────────────────────────────────────────
        st.markdown(
            "<div class='section-title'>📊 Índices Calculados</div>",
            unsafe_allow_html=True,
        )
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric(
            "Risk Score", f"{risk_score:.1f} / 10",
            delta="↑ Alto" if risk_score >= 6 else "~ Moderado" if risk_score >= 3 else "↓ Bajo",
            delta_color="inverse",
        )
        s2.metric(
            "Prevention Index", f"{prevention_index:.1f} / 10",
            delta="↑ Bueno" if prevention_index >= 5 else "↓ Mejorable",
            delta_color="normal",
        )
        s3.metric("Acceso sanitario", "Urbano" if access_score == 1.0 else "Rural")
        s4.metric("Grupo de edad (riesgo)", age_group.replace("_", " "))
        s5.metric("Perfil de estilo de vida", lifestyle.replace("_", " "))

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Factores de riesgo ─────────────────────────────────────────────
        st.markdown(
            "<div class='section-title'>⚠️ Factores de Riesgo Detectados</div>",
            unsafe_allow_html=True,
        )
        risk_factors: list[str] = []
        if genetic == "Yes":                    risk_factors.append("🧬 Mutación genética documentada")
        if family_history == "Yes":             risk_factors.append("👨‍👩‍👧 Historia familiar de CRC")
        if age >= 75:                           risk_factors.append(f"🎂 Edad muy alta ({age} años)")
        elif age >= 60:                         risk_factors.append(f"🎂 Edad ≥ 60 años ({age})")
        elif age >= 45:                         risk_factors.append(f"🎂 Edad ≥ 45 años ({age})")
        if diabetes == "Yes":                   risk_factors.append("💉 Diabetes tipo 2")
        if ibd == "Yes":                        risk_factors.append("🔥 Enfermedad Inflamatoria Intestinal")
        if obesity == "Obese":                  risk_factors.append("⚖️ Obesidad")
        elif obesity == "Overweight":           risk_factors.append("⚖️ Sobrepeso")
        if diet == "High":                      risk_factors.append("🍔 Dieta de alto riesgo")
        if activity == "Low":                   risk_factors.append("🛋️ Sedentarismo")
        if smoking == "Yes":                    risk_factors.append("🚬 Tabaquismo activo")
        if alcohol == "Yes":                    risk_factors.append("🍷 Consumo de alcohol")
        if screening == "Never":                risk_factors.append("🩺 Sin cribado previo")
        elif screening == "Irregular":          risk_factors.append("🩺 Cribado irregular")

        if risk_factors:
            cols = st.columns(min(len(risk_factors), 3))
            for i, label in enumerate(risk_factors):
                cols[i % 3].warning(label)
        else:
            st.success("✅ No se detectaron factores de riesgo significativos.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Factores protectores ───────────────────────────────────────────
        st.markdown(
            "<div class='section-title'>🛡️ Factores Protectores</div>",
            unsafe_allow_html=True,
        )
        protective: list[str] = []
        if screening == "Regular":   protective.append("🩺 Cribado regular y activo")
        if activity == "High":       protective.append("🏃 Alta actividad física")
        if diet == "Low":            protective.append("🥦 Dieta saludable")
        if obesity == "Normal":      protective.append("⚖️ Peso en rango normal")
        if smoking == "No":          protective.append("✅ No fumador/a")
        if alcohol == "No":          protective.append("✅ Sin consumo de alcohol")
        if diabetes == "No" and ibd == "No" and genetic == "No":
            protective.append("🏥 Sin comorbilidades de riesgo")

        if protective:
            cols = st.columns(min(len(protective), 4))
            for i, label in enumerate(protective):
                cols[i % 4].success(label)
        else:
            st.info("ℹ️ No se identificaron factores protectores destacados.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Recomendación clínica ──────────────────────────────────────────
        st.markdown(
            "<div class='section-title'>📋 Recomendación Clínica</div>",
            unsafe_allow_html=True,
        )
        if risk_lv == "Alto":
            st.error(
                "🔴 **Riesgo ALTO** — Derivar a gastroenterología para colonoscopia "
                "diagnóstica en las próximas semanas. Valorar estudio genético familiar "
                "si se confirma mutación hereditaria."
            )
        elif risk_lv == "Moderado":
            st.warning(
                "⚠️ **Riesgo MODERADO** — Programar colonoscopia de cribado en los "
                "próximos 6-12 meses. Insistir en modificación de hábitos y seguimiento anual."
            )
        else:
            st.success(
                "✅ **Riesgo BAJO** — Mantener cribado estándar según edad y guías clínicas. "
                "Control en 2-3 años o antes si aparecen síntomas."
            )

        # ── Guardar en historial (backend, no crítico) ────────────────────
        try:
            run_diagnosis({
                "patient_id": pid,
                "clinical_data": {
                    **inputs,
                    "csv_mlp_probability": prob,
                    "risk_score":          risk_score,
                    "prevention_index":    prevention_index,
                    "lifestyle_cluster":   lifestyle,
                },
            })
        except Exception:
            pass


render()