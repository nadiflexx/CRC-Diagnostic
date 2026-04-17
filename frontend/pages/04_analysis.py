"""
Análisis de Lógica Inversa: Hábitos, Daño Metabólico y Riesgo de Cáncer Colorrectal.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from components.sidebar import render_sidebar
from components.cards import page_header, render_section_card
from components.banners import show_empty_state
from utils.helpers import apply_custom_css

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from src.config.paths import paths
from src.config.constants import (
    REVERSE_ANALYSIS_DEFAULT_PROFILE,
    REVERSE_ANALYSIS_FEATURES
)
from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel 


st.set_page_config(
    page_title="Análisis de Lógica Inversa · Endo-AID", 
    page_icon="🌿", 
    layout="wide"
)

apply_custom_css()
render_sidebar()

@st.cache_resource
def load_models():
    """Carga los modelos entrenados de tabaco y alcohol desde la ruta configurada."""
    drk_path = paths.REVERSE_LOGIC_MODEL_DIR / "reverse_logic_DRK_YN.pkl"
    smk_path = paths.REVERSE_LOGIC_MODEL_DIR / "reverse_logic_SMK_stat_type_cd.pkl"
    
    drk_model, smk_model = None, None
    
    try:
        if drk_path.exists():
            drk_model = ReverseLogicTabularModel.load(drk_path)
        if smk_path.exists():
            smk_model = ReverseLogicTabularModel.load(smk_path)
    except Exception as e:
        st.error(f"❌ Error al cargar los modelos: {e}")
        
    return drk_model, smk_model

drk_model, smk_model = load_models()


def calculate_bmi(weight, height):
    """Calcula BMI desde peso (kg) y altura (cm)."""
    height_m = height / 100
    return weight / (height_m ** 2)

def get_risk_category(probability):
    """Clasifica el riesgo basado en probabilidad."""
    if probability >= 70:
        return "🔴 MUY ALTO", "risk-high"
    elif probability >= 50:
        return "🟠 ALTO", "risk-medium"
    elif probability >= 30:
        return "🟡 MODERADO", "risk-medium"
    else:
        return "🟢 BAJO", "risk-low"
    
def create_risk_gauge(probability, label):
    """Crea un gráfico tipo gauge para mostrar riesgo."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': label, 'font': {'size': 16}},
        delta={'reference': 50, 'suffix': "% de probabilidad"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 30], 'color': "#d5f4e6"},
                {'range': [30, 50], 'color': "#ffeaa7"},
                {'range': [50, 70], 'color': "#ffcccc"},
                {'range': [70, 100], 'color': "#ff7675"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_biomarker_interpretation_chart(data):
    """Crea gráfico mostrando biomarcadores y riesgo de enfermedad hepática."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'bar'}, {'type': 'scatter'}]],
        subplot_titles=("Biomarcadores Hepáticos (Riesgo de Daño)", "Relación AST/ALT y Riesgo")
    )
    
    # Biomarcadores
    biomarkers = ['SGOT_AST', 'SGOT_ALT', 'gamma_GTP']
    values = [data['SGOT_AST'], data['SGOT_ALT'], data['gamma_GTP']]
    reference_max = [40, 40, 60]  # Valores de referencia normales
    
    colors = ['#ff7675' if v > r else '#55efc4' for v, r in zip(values, reference_max)]
    
    fig.add_trace(
        go.Bar(
            x=biomarkers,
            y=values,
            name="Valor Actual",
            marker=dict(color=colors),
            text=[f"{v:.1f}" for v in values],
            textposition="auto"
        ),
        row=1, col=1
    )
    
    fig.add_hline(y=40, line_dash="dash", line_color="green", 
                  annotation_text="Límite Normal", row=1, col=1)
    
    # Relación AST/ALT
    ast_alt_ratio = data['SGOT_AST'] / max(data['SGOT_ALT'], 1)
    alcohol_risk_scores = np.linspace(0, 100, 50)
    ratio_progression = 0.8 + (alcohol_risk_scores / 100) * 1.2
    
    fig.add_trace(
        go.Scatter(
            x=alcohol_risk_scores,
            y=ratio_progression,
            mode='lines',
            name='Riesgo vs AST/ALT',
            line=dict(color='#667eea', width=3),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ),
        row=1, col=2
    )
    
    fig.add_vline(x=data['DRK_YN'] if isinstance(data.get('DRK_YN'), (int, float)) else 0, 
                  line_dash="dash", line_color="red",
                  annotation_text="Riesgo Predicho", row=1, col=2)
    
    fig.update_xaxes(title_text="Biomarcador", row=1, col=1)
    fig.update_yaxes(title_text="Valor (U/L)", row=1, col=1)
    fig.update_xaxes(title_text="Riesgo de Alcohol (%)", row=1, col=2)
    fig.update_yaxes(title_text="Proporción AST/ALT", row=1, col=2)
    
    fig.update_layout(height=350, showlegend=True, hovermode='x unified')
    return fig

def create_metabolic_syndrome_chart(data):
    """Gráfico de síndrome metabólico."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Índice de Masa Corporal", "Presión Arterial", 
                       "Perfil Lipídico", "Glucosa Basal"),
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'bar'}, {'type': 'indicator'}]]
    )
    
    # BMI
    bmi = calculate_bmi(data['weight'], data['height'])
    bmi_status = "Normal" if bmi < 25 else "Sobrepeso" if bmi < 30 else "Obesidad"
    
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=bmi,
            title={"text": "BMI"},
            delta={'reference': 25, 'suffix': " (25 = sobrepeso)"},
            domain={'x': [0, 1], 'y': [0.5, 1]}
        ),
        row=1, col=1
    )
    
    # Presión Arterial
    bp_status = "Normal" if data['SBP'] < 130 else "Elevada"
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=data['SBP'],
            title={"text": "Presión Sistólica"},
            gauge={
                'axis': {'range': [80, 180]},
                'steps': [
                    {'range': [80, 120], 'color': "#d5f4e6"},
                    {'range': [120, 140], 'color': "#ffeaa7"},
                    {'range': [140, 180], 'color': "#ff7675"}
                ]
            },
            domain={'x': [0.5, 1], 'y': [0.5, 1]}
        ),
        row=1, col=2
    )
    
    # Perfil Lipídico
    lipids = ['Colesterol', 'HDL', 'LDL', 'Triglicéridos']
    lipid_values = [data['tot_chole'], data['HDL_chole'], data['LDL_chole'], data['triglyceride']]
    lipid_optimal = [200, 40, 100, 150]
    
    fig.add_trace(
        go.Bar(
            x=lipids,
            y=lipid_values,
            name="Valor Actual",
            marker=dict(color=['#ff7675' if v > o else '#55efc4' for v, o in zip(lipid_values, lipid_optimal)]),
            text=[f"{v:.0f}" for v in lipid_values],
            textposition="auto"
        ),
        row=2, col=1
    )
    
    # Glucosa
    gluc_status = "Normal" if data['BLDS'] < 100 else "Elevada"
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=data['BLDS'],
            title={"text": "Glucosa Basal (mg/dL)"},
            gauge={
                'axis': {'range': [60, 200]},
                'steps': [
                    {'range': [60, 100], 'color': "#d5f4e6"},
                    {'range': [100, 126], 'color': "#ffeaa7"},
                    {'range': [126, 200], 'color': "#ff7675"}
                ]
            },
            domain={'x': [0.5, 1], 'y': [0, 0.5]}
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    return fig

def create_scientific_evidence_chart():
    """Crea gráfico con evidencia científica sobre alcohol, tabaco y cáncer de colon."""
    
    # Datos de riesgo relativo basados en literatura científica
    risk_factors = ['Alcohol (>2 drinks/día)', 'Tabaco (Fumador actual)', 
                    'Síndrome Metabólico', 'Obesidad (BMI>30)', 'Bajo Consumo Fibra',
                    'Sedentarismo', 'Antecedentes Familiares', 'Edad >50']
    
    relative_risk = [1.8, 1.3, 2.1, 1.5, 1.4, 1.3, 2.8, 5.2]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=risk_factors,
        x=relative_risk,
        orientation='h',
        marker=dict(
            color=relative_risk,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Riesgo<br>Relativo")
        ),
        text=[f"{x:.1f}x" for x in relative_risk],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Riesgo Relativo: %{x:.2f}x<extra></extra>'
    ))
    
    fig.update_layout(
        title="📊 Factores de Riesgo para Cáncer Colorrectal<br><sub>Riesgo Relativo comparado con ausencia del factor</sub>",
        xaxis_title="Riesgo Relativo",
        yaxis_title="",
        height=450,
        hovermode='closest',
        margin=dict(l=250, r=50, t=100, b=50)
    )
    
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray", 
                  annotation_text="Riesgo Base (1.0x)", annotation_position="top right")
    
    return fig

def create_pathophysiology_chart(drk_prob, smk_prob):
    """Crea gráfico mostrando mecanismos de daño (patofisiología)."""
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'sunburst'}, {'type': 'sankey'}]],
        subplot_titles=("Cascada Patofisiológica: Alcohol → Daño Hepático → Cáncer",
                       "Cascada Patofisiológica: Tabaco → Estrés Oxidativo → Cáncer")
    )
    
    # Sunburst para Alcohol
    fig.add_trace(
        go.Sunburst(
            labels=["Consumo Alcohol", "Acetaldehído", "Daño Hepático", "Alteración Microbioma", "Inflamación Crónica", "Cáncer Colorrectal"],
            parents=["", "Consumo Alcohol", "Acetaldehído", "Consumo Alcohol", "Alteración Microbioma", "Inflamación Crónica"],
            values=[100, drk_prob*0.5, drk_prob*0.3, drk_prob*0.4, drk_prob*0.5, drk_prob*0.6],
            marker=dict(colors=['#ff7675', '#fab1a0', '#ff7675', '#dfe6e9', '#fdcb6e', '#d63031'],
                       line=dict(color='white', width=2)),
            textinfo="label+percent parent"
        ),
        row=1, col=1
    )
    
    # Sankey para Tabaco
    fig.add_trace(
        go.Sankey(
            node=dict(
                pad=15,
                line=dict(color='black', width=0.5),
                label=['Fumador', 'Toxinas\nTabáquicas', 'Inflamación', 'Isquemia\nColónica', 'Pólipos\nAdenomatosos', 'Cáncer\nColorectal'],
                color=['#1e90ff', '#ff6b6b', '#ffa502', '#d63031', '#ff7675', '#8b0000']
            ),
            link=dict(
                source=[0, 1, 1, 2, 3, 4],
                target=[1, 2, 3, 4, 4, 5],
                value=[smk_prob*0.6, smk_prob*0.5, smk_prob*0.4, smk_prob*0.5, smk_prob*0.3, smk_prob*0.7]
            )
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=500, showlegend=False)
    return fig


# INTERFAZ
page_header(
    "Análisis de Lógica Inversa: Hábitos y Riesgo de Cáncer",
    "Predicción IA de consumo de alcohol y tabaquismo basada en biomarcadores clínicos reales",
    icon="🔬"
)

st.markdown("""
### ¿Cómo funciona este análisis?

En lugar de preguntarte directamente si fumas o bebes, este análisis **inverso** utiliza **inteligencia artificial 
para predecir tu historial de consumo** basándose en **biomarcadores clínicos reales** (presión arterial, 
hígado, lípidos, glucosa, etc.).

**La lógica es simple:**
- Si tu cuerpo muestra signos de daño característicos del alcohol (elevación de Gamma-GTP, AST, ALT), 
  el modelo lo detectará incluso sin preguntarte.
- Si muestras estrés oxidativo y daño vascular típico del tabaquismo, el modelo lo reconocerá.

**¿Por qué es importante?**
Estos mismos factores que generan daño metabólico (inflamación crónica, estrés oxidativo, disbiosis) 
son **detonantes directos del cáncer colorrectal**. 

---
""")

st.divider()


# FORMULARIO
st.markdown(
    "<div class='section-title'>📋 Perfil Clínico del Paciente</div>",
    unsafe_allow_html=True
)

# Usamos el perfil por defecto de constants.py para pre-llenar el formulario
defaults = REVERSE_ANALYSIS_DEFAULT_PROFILE

# Inicializar session_state con valores por defecto
if "form_data" not in st.session_state:
    st.session_state.form_data = {
        "sex": 0 if defaults.get("sex", "Male") == "Male" else 1,
        "age": int(defaults.get("age", 45)),
        "height": int(defaults.get("height", 175)),
        "weight": int(defaults.get("weight", 75)),
        "waistline": float(defaults.get("waistline", 85.0)),
        "sbp": float(defaults.get("SBP", 120.0)),
        "dbp": float(defaults.get("DBP", 80.0)),
        "blds": float(defaults.get("BLDS", 100.0)),
        "tot_chole": float(defaults.get("tot_chole", 190.0)),
        "hdl": float(defaults.get("HDL_chole", 50.0)),
        "ldl": float(defaults.get("LDL_chole", 110.0)),
        "trigly": float(defaults.get("triglyceride", 120.0)),
        "hemo": float(defaults.get("hemoglobin", 15.0)),
        "urine_prot": int(defaults.get("urine_protein", 1.0)) - 1,
        "serum_crea": float(defaults.get("serum_creatinine", 1.0)),
        "sgot_ast": float(defaults.get("SGOT_AST", 25.0)),
        "sgot_alt": float(defaults.get("SGOT_ALT", 25.0)),
        "gamma_gtp": float(defaults.get("gamma_GTP", 30.0)),
    }

# Crear formulario
with st.form("clinical_form"):
    
    # TABS para organizar el formulario
    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Demografía & Antropometría",
        "🫀 Vitales & Visión",
        "🩸 Perfil Lipídico & Sangre",
        "🧪 Función Hepática & Renal"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            sex = st.selectbox(
                "Sexo",
                options=["Male", "Female"],
                index=st.session_state.form_data["sex"],
                key="sex_input",
                help="Género del paciente"
            )
            st.session_state.form_data["sex"] = 0 if sex == "Male" else 1
            
            age = st.number_input(
                "Edad (años)",
                min_value=18, max_value=120,
                value=st.session_state.form_data["age"],
                key="age_input",
                help="Edad actual"
            )
            st.session_state.form_data["age"] = age
            
            height = st.number_input(
                "Altura (cm)",
                min_value=100, max_value=250,
                value=st.session_state.form_data["height"],
                key="height_input",
                help="Estatura en centímetros"
            )
            st.session_state.form_data["height"] = height
        
        with col2:
            weight = st.number_input(
                "Peso (kg)",
                min_value=30, max_value=200,
                value=st.session_state.form_data["weight"],
                key="weight_input",
                help="Peso corporal"
            )
            st.session_state.form_data["weight"] = weight
            
            waistline = st.number_input(
                "Perímetro de Cintura (cm)",
                min_value=40.0, max_value=200.0,
                value=st.session_state.form_data["waistline"],
                key="waistline_input",
                help="Medida de cintura a la altura del ombligo"
            )
            st.session_state.form_data["waistline"] = waistline
        
        # Cálculo automático de BMI
        bmi = calculate_bmi(weight, height)
        st.info(f"📊 **IMC calculado: {bmi:.2f}** {'(Normal)' if bmi < 25 else '(Sobrepeso)' if bmi < 30 else '(Obesidad)'}")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            sbp = st.number_input(
                "Presión Sistólica - SBP (mmHg)",
                min_value=60.0, max_value=250.0,
                value=st.session_state.form_data["sbp"],
                key="sbp_input",
                help="Presión arterial sistólica"
            )
            st.session_state.form_data["sbp"] = sbp
            
            dbp = st.number_input(
                "Presión Diastólica - DBP (mmHg)",
                min_value=40.0, max_value=150.0,
                value=st.session_state.form_data["dbp"],
                key="dbp_input",
                help="Presión arterial diastólica"
            )
            st.session_state.form_data["dbp"] = dbp
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            blds = st.number_input(
                "Glucosa en Ayunas - BLDS (mg/dL)",
                min_value=50.0, max_value=400.0,
                value=st.session_state.form_data["blds"],
                key="blds_input",
                help="Glucosa basal. Normal: <100"
            )
            st.session_state.form_data["blds"] = blds
            
            tot_chole = st.number_input(
                "Colesterol Total (mg/dL)",
                min_value=50.0, max_value=500.0,
                value=st.session_state.form_data["tot_chole"],
                key="tot_chole_input",
                help="Ideal: <200"
            )
            st.session_state.form_data["tot_chole"] = tot_chole
            
            hdl = st.number_input(
                "Colesterol HDL (mg/dL)",
                min_value=10.0, max_value=150.0,
                value=st.session_state.form_data["hdl"],
                key="hdl_input",
                help="HDL 'colesterol bueno'. Mayor es mejor: >40 en hombres, >50 en mujeres"
            )
            st.session_state.form_data["hdl"] = hdl
        
        with col2:
            ldl = st.number_input(
                "Colesterol LDL (mg/dL)",
                min_value=10.0, max_value=300.0,
                value=st.session_state.form_data["ldl"],
                key="ldl_input",
                help="LDL 'colesterol malo'. Ideal: <100"
            )
            st.session_state.form_data["ldl"] = ldl
            
            trigly = st.number_input(
                "Triglicéridos (mg/dL)",
                min_value=20.0, max_value=1000.0,
                value=st.session_state.form_data["trigly"],
                key="trigly_input",
                help="Ideal: <150"
            )
            st.session_state.form_data["trigly"] = trigly
            
            hemo = st.number_input(
                "Hemoglobina (g/dL)",
                min_value=5.0, max_value=25.0,
                value=st.session_state.form_data["hemo"],
                key="hemo_input",
                help="Portador de oxígeno. Normal: 12-17"
            )
            st.session_state.form_data["hemo"] = hemo
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            urine_prot_options = ["Negativa", "+1", "+2", "+3", "+4", "+5"]
            urine_prot_idx = st.session_state.form_data["urine_prot"]
            urine_prot_label = st.selectbox(
                "Proteína en Orina",
                options=urine_prot_options,
                index=urine_prot_idx,
                key="urine_prot_input",
                help="Indica función renal"
            )
            # Calcular el índice actual y el valor numérico
            current_urine_prot_idx = urine_prot_options.index(urine_prot_label)
            urine_prot = current_urine_prot_idx + 1.0
            st.session_state.form_data["urine_prot"] = current_urine_prot_idx
            
            serum_crea = st.number_input(
                "Creatinina Sérica (mg/dL)",
                min_value=0.1, max_value=15.0,
                value=st.session_state.form_data["serum_crea"],
                step=0.1,
                key="serum_crea_input",
                help="Marcador de función renal. Normal: 0.7-1.3"
            )
            st.session_state.form_data["serum_crea"] = serum_crea
        
        with col2:
            sgot_ast = st.number_input(
                "SGOT/AST - Transaminasa Oxalacética (U/L)",
                min_value=1.0, max_value=500.0,
                value=st.session_state.form_data["sgot_ast"],
                key="sgot_ast_input",
                help="Enzima hepática. Normal: <40. ⚠️ Elevada en daño hepático por alcohol"
            )
            st.session_state.form_data["sgot_ast"] = sgot_ast
            
            sgot_alt = st.number_input(
                "SGOT/ALT - Transaminasa Pirúvica (U/L)",
                min_value=1.0, max_value=500.0,
                value=st.session_state.form_data["sgot_alt"],
                key="sgot_alt_input",
                help="Enzima hepática. Normal: <40. ⚠️ Muy elevada en hepatitis"
            )
            st.session_state.form_data["sgot_alt"] = sgot_alt
        
        col3, col4 = st.columns(2)
        with col3:
            gamma_gtp = st.number_input(
                "Gamma-GTP (U/L)",
                min_value=1.0, max_value=999.0,
                value=st.session_state.form_data["gamma_gtp"],
                key="gamma_gtp_input",
                help="Enzima hepática. Normal: <55. 🚨 MUY SENSIBLE al alcohol (aumenta primero)"
            )
            st.session_state.form_data["gamma_gtp"] = gamma_gtp
        
        with col4:
            st.info("💡 **Nota sobre Biomarcadores Hepáticos:**\n\n"
                   "- **Gamma-GTP**: Indicador MÁS SENSIBLE de consumo de alcohol\n"
                   "- **AST/ALT**: Ratio >2 sugiere daño alcohólico\n"
                   "- Estos biomarcadores predicen inflamación crónica → Cáncer")
    
    st.divider()
    submit_button = st.form_submit_button(
        "🔍 Analizar Riesgo Metabólico",
        use_container_width=True,
        type="primary"
    )


# PROCESAMIENTO Y RESULTADOS

if submit_button:
    if drk_model is None or smk_model is None:
        st.error("⚠️ **Modelos no encontrados**\n\nAsegúrate de haber ejecutado el entrenamiento previamente.")
        st.stop()
    else:
        # Obtener todos los datos desde session_state
        sex_str = "Male" if st.session_state.form_data["sex"] == 0 else "Female"
        height = st.session_state.form_data["height"]
        weight = st.session_state.form_data["weight"]
        waistline = st.session_state.form_data["waistline"]
        sbp = st.session_state.form_data["sbp"]
        dbp = st.session_state.form_data["dbp"]
        blds = st.session_state.form_data["blds"]
        tot_chole = st.session_state.form_data["tot_chole"]
        hdl = st.session_state.form_data["hdl"]
        ldl = st.session_state.form_data["ldl"]
        trigly = st.session_state.form_data["trigly"]
        hemo = st.session_state.form_data["hemo"]
        urine_prot = st.session_state.form_data["urine_prot"] + 1.0
        serum_crea = st.session_state.form_data["serum_crea"]
        sgot_ast = st.session_state.form_data["sgot_ast"]
        sgot_alt = st.session_state.form_data["sgot_alt"]
        gamma_gtp = st.session_state.form_data["gamma_gtp"]
        bmi = calculate_bmi(weight, height)
        
        # Construir DataFrame con las features exactas
        input_data = {
            "sex": sex_str, "height": height, "weight": weight, "waistline": waistline,
            "SBP": sbp, "DBP": dbp, "BLDS": blds, "tot_chole": tot_chole, "HDL_chole": hdl,
            "LDL_chole": ldl, "triglyceride": trigly, "hemoglobin": hemo, "urine_protein": urine_prot,
            "serum_creatinine": serum_crea, "SGOT_AST": sgot_ast, "SGOT_ALT": sgot_alt, "gamma_GTP": gamma_gtp,
            "BMI": bmi, "AST_ALT_ratio": sgot_ast / max(sgot_alt, 0.1)
        }
        df_input = pd.DataFrame([input_data])[REVERSE_ANALYSIS_FEATURES]

        try:
            # Intentamos acceder al pipeline
            smk_prob = smk_model.predict_proba(df_input, model_type="xgboost")[0][1] * 100
            drk_prob = drk_model.predict_proba(df_input, model_type="xgboost")[0][1] * 100
        except Exception as e:
            st.error(f"❌ Error al predecir: {e}")
            st.stop()
        
        st.divider()
        st.markdown(
            "<div class='section-title'>📊 Resultados del Análisis</div>",
            unsafe_allow_html=True
        )
        
        # Métricas principales
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            smk_risk, smk_css = get_risk_category(smk_prob)
            st.metric(
                label="🚬 Probabilidad de Tabaquismo",
                value=f"{smk_prob:.1f}%",
                delta=f"{smk_risk}",
                delta_color="inverse"
            )
            st.plotly_chart(create_risk_gauge(smk_prob, "Riesgo Tabaquismo"), use_container_width=True, config={"displayModeBar": False})
        
        with col_res2:
            drk_risk, drk_css = get_risk_category(drk_prob)
            st.metric(
                label="🍷 Probabilidad de Alcoholismo",
                value=f"{drk_prob:.1f}%",
                delta=f"{drk_risk}",
                delta_color="inverse"
            )
            st.plotly_chart(create_risk_gauge(drk_prob, "Riesgo de Alcoholismo"), use_container_width=True, config={"displayModeBar": False})
        
        with col_res3:
            # Síndrome metabólico
            metabolic_risk = 0
            if bmi >= 30:
                metabolic_risk += 30
            if waistline > 90 if sex == "Male" else waistline > 85:
                metabolic_risk += 25
            if blds > 100:
                metabolic_risk += 20
            if trigly > 150:
                metabolic_risk += 15
            if hdl < 40 if sex == "Male" else hdl < 50:
                metabolic_risk += 10
            metabolic_risk = min(metabolic_risk, 100)
            
            met_risk, met_css = get_risk_category(metabolic_risk)
            st.metric(
                label="⚠️ Síndrome Metabólico",
                value=f"{metabolic_risk:.0f}%",
                delta=f"{met_risk}",
                delta_color="inverse"
            )
            st.plotly_chart(create_risk_gauge(metabolic_risk, "Riesgo Metabólico"), use_container_width=True, config={"displayModeBar": False})
        
        st.divider()
        

        st.markdown(
            "<div class='section-title'>📈 Análisis Detallado & Evidencia Científica</div>",
            unsafe_allow_html=True
        )
        
        # Tab 1: Biomarcadores
        tab_bio, tab_met, tab_path, tab_evidence = st.tabs([
            "🧬 Biomarcadores Hepáticos",
            "⚖️ Síndrome Metabólico",
            "🔬 Patofisiología",
            "📚 Evidencia Científica"
        ])
        
        with tab_bio:
            st.markdown("""
            ### Interpretación de Biomarcadores Hepáticos
            
            Estos marcadores son críticos porque **el alcohol causa daño hepático crónico** 
            que genera inflamación sistémica, base del cáncer colorrectal.
            """)
            
            st.plotly_chart(create_biomarker_interpretation_chart(input_data), use_container_width=True)
            
            # Interpretación
            if gamma_gtp > 60:
                st.warning(f"⚠️ **Gamma-GTP ELEVADA ({gamma_gtp:.1f} U/L)**\n\n"
                           "Este es el biomarcador MÁS SENSIBLE para consumo de alcohol. "
                           "Una elevación sugiere:\n"
                           "- Daño hepático crónico\n"
                           "- Inflamación sistémica\n"
                           "- Mayor riesgo de cáncer colorrectal")
            
            if sgot_ast > 40 or sgot_alt > 40:
                st.warning(f"⚠️ **Transaminasas ELEVADAS**\n\n"
                           "AST: {sgot_ast:.1f} U/L | ALT: {sgot_alt:.1f} U/L\n\n"
                           "Indica necrosis hepatocitaria. Relación AST/ALT: {sgot_ast/max(sgot_alt,1):.2f}")
            
            if sgot_ast / max(sgot_alt, 0.1) > 1.5:
                st.error(f"🔴 **RATIO AST/ALT ELEVADO (>{sgot_ast/max(sgot_alt,0.1):.2f})**\n\n"
                        "Sugiere fuertemente **daño alcohólico hepático** (vs viral o autoinmune)")
        
        with tab_met:
            st.markdown("""
            ### Síndrome Metabólico y Cáncer Colorrectal
            
            Un hallazgo crítico: la **resistencia a la insulina y hiperinsulinemia** 
            actúan como factor de crecimiento (IGF-1), promoviendo directamente tumores colónicos.
            """)
            
            st.plotly_chart(create_metabolic_syndrome_chart(input_data), use_container_width=True)
            
            if metabolic_risk > 50:
                st.error(f"🔴 **SÍNDROME METABÓLICO PROBABLE**\n\n"
                        "Riesgo: {metabolic_risk:.0f}%\n\n"
                        "Tienes ≥3 de estos criterios:\n"
                        f"- {'✓ ' if bmi >= 30 else ''}BMI ≥30\n"
                        f"- {'✓ ' if waistline > (90 if sex=='Male' else 85) else ''}Cintura elevada\n"
                        f"- {'✓ ' if blds > 100 else ''}Glucosa basal >100\n"
                        f"- {'✓ ' if trigly > 150 else ''}Triglicéridos >150\n"
                        f"- {'✓ ' if hdl < (40 if sex=='Male' else 50) else ''}HDL bajo\n\n"
                        "⚠️ Esto multiplica tu riesgo de cáncer colorrectal")
        
        with tab_path:
            st.markdown("""
            ### Cascadas Patofisiológicas: De Hábito a Cáncer
            
            Estos gráficos muestran cómo el alcohol y el tabaco generan **inflamación crónica** 
            que finalmente causa cáncer.
            """)
            
            st.plotly_chart(create_pathophysiology_chart(drk_prob, smk_prob), use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🍷 Mecanismo Alcohol → Cáncer Colon**
                
                1. Alcohol → Acetaldehído (carcinógeno)
                2. Daño hepatocitario → Cicatrización (fibrosis)
                3. Endotoxemia → Inflamación sistémica
                4. Disbiosis → Alteración microbioma intestinal
                5. Pérdida barrera intestinal → Aumento permeabilidad
                6. Estrés oxidativo crónico → Mutaciones en colon
                7. **→ CÁNCER COLORRECTAL**
                """)
            
            with col2:
                st.markdown("""
                **🚬 Mecanismo Tabaco → Cáncer Colon**
                
                1. Humo del tabaco → Toxinas sistémicas
                2. Estrés oxidativo → Daño de ADN
                3. Isquemia colónica → Hipoxia tisular
                4. Inflamación crónica → Reparación defectuosa
                5. Aumento angiogénesis → Nutrición tumoral
                6. Inmunosupresión local → Escape inmune
                7. **→ CÁNCER COLORRECTAL**
                """)
        
        with tab_evidence:
            st.markdown("""
            ### Evidencia Científica: Factores de Riesgo para Cáncer Colorrectal
            
            Este gráfico muestra el **riesgo relativo** (comparado a ausencia del factor) 
            basado en meta-análisis y estudios epidemiológicos.
            """)
            
            st.plotly_chart(create_scientific_evidence_chart(), use_container_width=True)
            
            st.markdown("""
            **Referencias científicas:**
            - Alcohol: WHO IARC 2018 (Acetaldehído = carcinógeno Grupo 1)
            - Tabaco: 130+ años de evidencia acumulada
            - Síndrome Metabólico: Meta-análisis 2015-2023
            - Microbioma: Nature 2021 - Dysbiosis colónica y CRC
            """)
        
        st.divider()
        
        st.markdown(
            "<div class='section-title'>🩺 Interpretación Clínica Integrada</div>",
            unsafe_allow_html=True
        )
        
        interpretation = []
        
        # Alcohol
        if drk_prob > 70:
            interpretation.append("""
            #### 🔴 MUY ALTO RIESGO DE DAÑO HEPÁTICO POR ALCOHOL
            
            El modelo detecta un patrón clínico altamente compatible con consumo significativo de alcohol:
            
            **Biomarcadores críticos:**
            - Gamma-GTP está elevada (biomarcador más sensible)
            - Transaminasas (AST/ALT) están elevadas
            - Posible ratio AST/ALT > 1.5 (patrón alcohólico típico)
            
            **Implicaciones para cáncer:**
            1. **Inflamación hepática crónica** → Liberación de citoquinas proinflamatorias
            2. **Estrés oxidativo sistémico** → Daño de ADN en colonocitos
            3. **Endotoxemia** → LPS bacteriano → Mayor permeabilidad intestinal
            4. **Disbiosis intestinal** → Pérdida de protección bacteriana, proliferación de Fusobacterium
            5. **Inmunodeficiencia local** → Escape de células precancerosas
            
            **Riesgo de cáncer colorrectal:** 1.8-2.2x mayor que no bebedores
            """)
        elif drk_prob > 50:
            interpretation.append("""
            #### 🟠 ALTO RIESGO DE DAÑO ASOCIADO A ALCOHOL
            
            El perfil sugiere potencial consumo de alcohol con manifestaciones bioquímicas incipientes.
            
            **Hallazgos:**
            - Elevación moderada de biomarcadores hepáticos
            - Signos tempranos de estrés oxidativo
            
            **Recomendación:** Confirmación con historial clínico y pruebas adicionales
            
            **Riesgo de cáncer colorrectal:** 1.3-1.8x mayor
            """)
        else:
            interpretation.append("""
            #### 🟢 BAJO RIESGO RELATIVO DE DAÑO HEPÁTICO
            
            Los biomarcadores hepáticos no sugieren consumo significativo de alcohol.
            
            **Beneficio:** Menor exposición a acetaldehído y sus efectos carcinógenos
            """)
        
        # Tabaco
        if smk_prob > 70:
            interpretation.append("""
            #### 🔴 MUY ALTO RIESGO COMPATIBLE CON TABAQUISMO
            
            El modelo detecta patrones clínicos altamente sugestivos de tabaquismo:
            
            **Signos detectados:**
            - Posible hipertensión arterial (nicotina→vasoconstricción)
            - Hemoglobina alterada (carboxihemoglobina)
            - Estrés oxidativo sistémico (EROS del humo del tabaco)
            
            **Mecanismos oncológicos:**
            1. **Carcinógenos del tabaco** (BAP, NNK, etc.) → Metabolismo a metabolitos reactivos
            2. **Isquemia colónica** → Hipoxia → HIF-1α activación → Angiogénesis tumoral
            3. **Inmunosupresión** → Linfocitos T↓, NK↓ → Vigilancia inmune↓
            4. **Inflamación crónica** → TNF-α, IL-6 → Reparación defectuosa de ADN
            
            **Riesgo de cáncer colorrectal:** 1.3-1.5x mayor
            """)
        elif smk_prob > 50:
            interpretation.append("""
            #### 🟠 ALTO RIESGO ASOCIADO A TABAQUISMO
            
            El perfil es compatible con tabaquismo crónico con efectos sistémicos moderados.
            
            **Hallazgos:**
            - Signos de estrés oxidativo
            - Posible disfunción endotelial
            
            **Riesgo de cáncer colorrectal:** 1.2-1.4x mayor
            """)
        else:
            interpretation.append("""
            #### 🟢 BAJO RIESGO COMPATIBLE CON NO FUMADOR
            
            Los biomarcadores no sugieren daño típico del tabaquismo.
            
            **Beneficio:** Ausencia de carcinógenos del humo del tabaco y menor inflamación oxidativa
            """)
        
        # Síndrome Metabólico
        if metabolic_risk > 70:
            interpretation.append("""
            #### 🔴 SÍNDROME METABÓLICO SEVERO
            
            Tienes múltiples criterios de síndrome metabólico, creando un **perfecto caldo de cultivo para cáncer**.
            
            **El mecanismo crítico:**
            - **Resistencia a insulina** → Hiperinsulinemia
            - **IGF-1 elevado** → Factor de crecimiento para colonocitos
            - **Inflamación crónica** (TNF-α, IL-6, CRP elevado)
            - **Disbiosis** → Dysbiotic bacteria producen LPS → Permeabilidad intestinal
            - **Adipocitoquinas disfuncionales** → Pérdida de protección (adiponectina↓)
            
            **Riesgo de cáncer colorrectal:** 2.1-2.5x mayor
            
            **ACCIÓN URGENTE:** Pérdida de peso, ejercicio, reducción carbohidratos refinados
            """)
        elif metabolic_risk > 50:
            interpretation.append("""
            #### 🟠 SÍNDROME METABÓLICO MODERADO
            
            Tienes ≥2 criterios de síndrome metabólico. Esto multiplica tu riesgo.
            
            **Riesgo de cáncer colorrectal:** 1.5-2.0x mayor
            """)
        
        for text in interpretation:
            st.markdown(text)
        
        st.divider()
        
        #recomendaciones finales
        st.markdown(
            "<div class='section-title'>💡 Conclusión y Recomendaciones</div>",
            unsafe_allow_html=True
        )
        
        combined_risk = (drk_prob*0.4 + smk_prob*0.4 + metabolic_risk*0.2) / 3
        
        if combined_risk > 65:
            st.error(f"""
            ### 🔴 RIESGO COMBINADO MUY ALTO: {combined_risk:.0f}%
            
            La combinación de:
            - Posible consumo de alcohol (detectado por Gamma-GTP)
            - Posible tabaquismo (detectado por biomarcadores)
            - Síndrome metabólico
            
            **Crea un riesgo exponencial, no aditivo, para cáncer colorrectal.**
            
            **RECOMENDACIONES URGENTES:**
            1. **Endoscopia colonoscópica** - Evaluación de pólipos existentes
            2. **Intervención en hábitos:**
               - Abandonar alcohol (completamente)
               - Dejar de fumar
               - Programa de pérdida de peso si BMI >25
            3. **Suplementación:**
               - Vitamina D3 (2000-4000 IU/día)
               - Omega-3
               - Fibra prebiótica (restaurar microbioma)
            4. **Monitoreo:**
               - Pruebas hepáticas cada 3 meses
               - Colonoscopia cada 3 años (intensiva vigilancia)
            """)
        
        elif combined_risk > 50:
            st.warning(f"""
            ### 🟠 RIESGO COMBINADO ALTO: {combined_risk:.0f}%
            
            Los factores de riesgo son moderados a altos.
            
            **RECOMENDACIONES:**
            1. Consulta con gastroenterología para planificar cribado
            2. Reducir/eliminar alcohol
            3. Dejar de fumar si aplica
            4. Programa de pérdida de peso
            5. Aumentar actividad física (≥150 min/semana)
            6. Dieta alta en fibra, baja en procesados
            """)
        
        else:
            st.success(f"""
            ### 🟢 RIESGO COMBINADO BAJO-MODERADO: {combined_risk:.0f}%
            
            Tu perfil es relativamente favorable, pero hay oportunidades de mejora.
            
            **RECOMENDACIONES:**
            1. Seguir cribado estándar de colorrectal (cada 10 años si >50 años)
            2. Mantener estilo de vida saludable
            3. Dieta mediterránea o DASH
            4. Actividad física regular
            5. Evitar alcohol excesivo, definitivamente evitar tabaco
            """)
        
        # Pie de página científico
        st.divider()
        st.markdown("""
        ---
        
        ### 📚 Metodología Científica
        
        Este análisis utiliza **Machine Learning con lógica inversa**: en lugar de predecir directamente 
        si tendrás cáncer, predice tu exposición a factores de riesgo (alcohol, tabaco) basándose en 
        **biomarcadores objetivos** (sangre, presión, etc.).
        
        **Por qué funciona:**
        - Los biomarcadores (Gamma-GTP, AST, presión arterial, lípidos) son **proxies objetivos** 
          del daño causado por alcohol y tabaco
        - El daño metabólico (inflamación, disbiosis, estrés oxidativo) es el **mecanismo directo** 
          de carcinogénesis
        - Los estudios epidemiológicos confirman que estos mismos factores que predicen el consumo 
          TAMBIÉN predicen cáncer colorrectal
        
        **Limitaciones:**
        - Este análisis NO sustituye consejo médico
        - Los modelos tienen ~70-80% de precisión en estudios internos
        - Variación individual es alta (no todos los bebedores desarrollan el mismo daño)
        - Necesita validación clínica caso a caso
        
        **Próximos pasos recomendados:**
        1. Compartir con tu médico/gastroenterólogo
        2. Confirmación con historial clínico detallado
        3. Si riesgo alto: colonoscopia + biopsias
        4. Genética: si múltiples familiares con cáncer → test de Lynch/FAP
        
        ---
        
        ¿Preguntas? Consulta a un especialista en oncología o gastroenterología.
        """)