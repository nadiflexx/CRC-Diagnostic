"""
Colorectal Cancer — Aplicación de predicción individual
========================================================
Dependencias:
    pip install streamlit joblib scikit-learn pandas numpy

Uso:
    streamlit run app_prediccion.py

Requiere en la misma carpeta (o ajustar rutas en config_features.py):
    · data/outputs/nn_model/mlp_model.pkl
    · data/outputs/nn_model/scaler.pkl
    · data/outputs/nn_model/model_config.pkl
    · config_features.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from patient_features import build_feature_row

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE RUTAS  (mismas que en config_features.py)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR   = 'data/outputs/nn_model'
MODEL_PATH  = os.path.join(MODEL_DIR, 'mlp_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
CONFIG_PATH = os.path.join(MODEL_DIR, 'model_config.pkl')


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DEL MODELO  (cacheado para no recargar en cada interacción)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    config = joblib.load(CONFIG_PATH)
    return model, scaler, config


# ─────────────────────────────────────────────────────────────────────────────
# INTERFAZ STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='Colorectal Cancer Risk',
    page_icon='🔬',
    layout='wide',
)

# ── Cabecera ─────────────────────────────────────────────────────────────────
st.title('🔬 Predicción de Riesgo — Cáncer Colorrectal')
st.markdown(
    'Introduce los datos clínicos del paciente. El modelo calculará automáticamente '
    'los índices derivados y estimará la probabilidad de diagnóstico positivo.'
)
st.divider()

# ── Comprobación del modelo ───────────────────────────────────────────────────
model_loaded = all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, CONFIG_PATH])

if not model_loaded:
    st.error(
        f'⚠️ No se encontraron los archivos del modelo en `{MODEL_DIR}/`.\n\n'
        'Ejecuta primero `model_tabulardata.py` para entrenar y guardar el modelo.'
    )
    st.stop()

model, scaler, config = load_model()
threshold      = config['threshold']
feature_names  = config['features']

st.caption(f'✅ Modelo cargado correctamente — Umbral de decisión: **{threshold}**')
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# FORMULARIO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────
st.subheader('📋 Datos del Paciente')

with st.form('patient_form'):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('**Datos demográficos**')
        age    = st.number_input('Edad', min_value=18, max_value=100, value=60, step=1)
        gender = st.selectbox('Género', ['M', 'F'], format_func=lambda x: 'Masculino' if x == 'M' else 'Femenino')
        urban  = st.selectbox('Entorno', ['Urban', 'Rural'], format_func=lambda x: 'Urbano' if x == 'Urban' else 'Rural')

        st.markdown('**Epidemiología**')
        incidence_rate = st.number_input('Tasa de incidencia (por 100K)', min_value=0.0, max_value=200.0, value=25.0, step=0.5)
        mortality_rate = st.number_input('Tasa de mortalidad (por 100K)', min_value=0.0, max_value=100.0, value=10.0, step=0.5)

    with col2:
        st.markdown('**Historial clínico**')
        family_history = st.selectbox('Historial familiar de cáncer', ['No', 'Yes'], format_func=lambda x: 'Sí' if x == 'Yes' else 'No')
        genetic        = st.selectbox('Mutación genética conocida', ['No', 'Yes'], format_func=lambda x: 'Sí' if x == 'Yes' else 'No')
        diabetes       = st.selectbox('Diabetes', ['No', 'Yes'], format_func=lambda x: 'Sí' if x == 'Yes' else 'No')
        ibd            = st.selectbox('Enfermedad inflamatoria intestinal', ['No', 'Yes'], format_func=lambda x: 'Sí' if x == 'Yes' else 'No')
        screening      = st.selectbox('Historial de cribado', ['Never', 'Irregular', 'Regular'],
                                      format_func=lambda x: {'Never': 'Nunca', 'Irregular': 'Irregular', 'Regular': 'Regular'}[x])
        early_det      = st.selectbox('Detección temprana previa', ['No', 'Yes'], format_func=lambda x: 'Sí' if x == 'Yes' else 'No')

    with col3:
        st.markdown('**Estilo de vida**')
        smoking  = st.selectbox('Tabaquismo', ['No', 'Yes'], format_func=lambda x: 'Sí' if x == 'Yes' else 'No')
        alcohol  = st.selectbox('Consumo de alcohol', ['No', 'Yes'], format_func=lambda x: 'Sí' if x == 'Yes' else 'No')
        obesity  = st.selectbox('IMC / Obesidad', ['Normal', 'Overweight', 'Obese'],
                                format_func=lambda x: {'Normal': 'Normal', 'Overweight': 'Sobrepeso', 'Obese': 'Obeso'}[x])
        diet     = st.selectbox('Riesgo dietético', ['Low', 'Moderate', 'High'],
                                format_func=lambda x: {'Low': 'Bajo', 'Moderate': 'Moderado', 'High': 'Alto'}[x])
        activity = st.selectbox('Actividad física', ['Low', 'Moderate', 'High'],
                                format_func=lambda x: {'Low': 'Baja', 'Moderate': 'Moderada', 'High': 'Alta'}[x])

    submitted = st.form_submit_button('🔍 Calcular predicción', use_container_width=True, type='primary')


# ─────────────────────────────────────────────────────────────────────────────
# PREDICCIÓN Y RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
if submitted:

    inputs = {
        'age': age, 'gender': gender, 'family_history': family_history,
        'smoking': smoking, 'alcohol': alcohol, 'obesity': obesity,
        'diet_risk': diet, 'physical_activity': activity,
        'diabetes': diabetes, 'ibd': ibd, 'genetic': genetic,
        'screening': screening, 'early_detection': early_det,
        'incidence_rate': incidence_rate, 'mortality_rate': mortality_rate,
        'urban_rural': urban,
    }

    # 1. Construir la fila de features procesadas
    df_input, risk_score, prevention_index, access_score, age_risk_group, lc = build_feature_row(inputs)

    # 2. Reordenar columnas exactamente como el modelo fue entrenado
    df_input = df_input[feature_names]

    # 3. Escalar con el scaler del entrenamiento
    X_scaled = scaler.transform(df_input)

    # 4. Probabilidad y decisión con el umbral ajustado
    prob      = model.predict_proba(X_scaled)[0, 1]
    diagnosis = int(prob >= threshold)

    st.divider()
    st.subheader('📊 Resultados')

    # ── Resultado principal ───────────────────────────────────────────────────
    col_res, col_prob, col_gauge = st.columns([1.2, 1, 1.5])

    with col_res:
        if diagnosis == 1:
            st.error('### 🔴 RIESGO POSITIVO\nEl modelo indica **riesgo elevado** de cáncer colorrectal.')
        else:
            st.success('### 🟢 RIESGO BAJO\nEl modelo **no detecta** indicadores de alto riesgo.')

    with col_prob:
        st.metric('Probabilidad estimada', f'{prob * 100:.1f} %')
        st.metric('Umbral de decisión',    f'{threshold * 100:.0f} %')
        delta_pct = (prob - threshold) * 100
        st.metric('Margen sobre umbral',   f'{delta_pct:+.1f} pp',
                  delta=f'{delta_pct:+.1f}', delta_color='inverse')

    with col_gauge:
        # Barra de progreso visual
        st.markdown('**Nivel de probabilidad**')
        bar_color = '#DC2626' if diagnosis == 1 else '#16A34A'
        fill = int(prob * 100)
        st.markdown(
            f"""
            <div style="background:#e5e7eb; border-radius:8px; height:28px; width:100%;">
              <div style="background:{bar_color}; border-radius:8px; height:28px;
                          width:{fill}%; display:flex; align-items:center;
                          justify-content:center; color:white; font-weight:bold;
                          font-size:14px; transition:width 0.4s;">
                {fill}%
              </div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:12px;
                        color:#6b7280; margin-top:4px;">
              <span>0%</span><span>Umbral {int(threshold*100)}%</span><span>100%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Features derivadas calculadas ────────────────────────────────────────
    st.subheader('🧮 Índices calculados automáticamente')
    col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)

    risk_color   = '🔴' if risk_score >= 6 else ('🟡' if risk_score >= 3 else '🟢')
    prev_color   = '🟢' if prevention_index >= 6 else ('🟡' if prevention_index >= 3 else '🔴')
    access_label = 'Urbano' if access_score == 1 else 'Rural'

    col_d1.metric('Risk Score',        f'{risk_color} {risk_score} / 10')
    col_d2.metric('Prevention Index',  f'{prev_color} {prevention_index} / 10')
    col_d3.metric('Access Score',      f'{access_score} ({access_label})')
    col_d4.metric('Grupo de edad',     age_risk_group)
    col_d5.metric('Lifestyle Cluster', lc)

    # ── Fila de features enviada al modelo ───────────────────────────────────
    with st.expander('🔎 Ver fila de datos enviada al modelo (valores numéricos)'):
        st.dataframe(df_input, use_container_width=True)
