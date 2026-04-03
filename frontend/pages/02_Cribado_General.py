"""
Generic Tabular Risk Assessment - Colorectal Cancer Prediction.
Formulario para cribado genérico basado en datos tabulares.
"""

import pickle
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

from components.banners import show_empty_state, show_result_banner
from components.cards import page_header, render_section_card
from components.sidebar import render_sidebar
from utils.helpers import apply_custom_css

# Agregar src al path para importaciones
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.index_utils import calculate_risk_indices

st.set_page_config(page_title="Cribado General · Endo-AID", page_icon="📋", layout="wide")
apply_custom_css()
render_sidebar()


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN Y CARGA DE MODELOS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_artifacts():
    """Carga el modelo, scaler y configuración guardados."""
    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / "models" / "saved" / "generic_tabular"
    
    model_path = model_dir / "mlp_model.pkl"
    scaler_path = model_dir / "scaler.pkl"
    config_path = model_dir / "model_config.pkl"
    
    if not model_path.exists():
        st.error(f"❌ Modelo no encontrado en {model_path}")
        st.error(f"📂 Directorio existe: {model_dir.exists()}")
        if model_dir.exists():
            st.error(f"📋 Archivos en {model_dir}: {list(model_dir.glob('*'))}")
        return None, None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    return model, scaler, config


def get_feature_descriptions():
    """Retorna descripción de cada feature."""
    return {
        'Age': ('Edad del paciente', 'años', 18, 90, 45),
        'Gender': ('Género', ['Male', 'Female', 'Other']),
        'Family_History': ('Historia familiar de cáncer colorrectal', ['Yes', 'No']),
        'Smoking_History': ('Antecedentes de tabaquismo', ['Yes', 'No']),
        'Alcohol_Consumption': ('Consumo de alcohol regular', ['Yes', 'No']),
        'Obesity_BMI': ('Índice de masa corporal (BMI)', 'kg/m²', 15.0, 50.0, 25.0),
        'Diet_Risk': ('Dieta de alto riesgo (carnes rojas, procesadas)', ['Yes', 'No']),
        'Physical_Activity': ('Actividad física regular (≥150 min/semana)', ['Yes', 'No']),
        'Diabetes': ('Diagnóstico de diabetes', ['Yes', 'No']),
        'Inflammatory_Bowel_Disease': ('Enfermedad inflamatoria intestinal', ['Yes', 'No']),
        'Genetic_Mutation': ('Mutación genética conocida (Lynch, FAP, etc)', ['Yes', 'No']),
        'Screening_History': ('Historial de cribado previo', ['Yes', 'No']),
        'Early_Detection': ('Detección temprana en cribados previos', ['Yes', 'No']),
        'Urban_or_Rural': ('Localización de residencia', ['Urban', 'Rural']),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE ESCALAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
# ⭐ SOLUCIÓN A: Escalar índices del formulario al rango del dataset
# 
# El modelo fue entrenado con:
#   - Risk_Score: 0-10 (media ~3.35, std ~1.52)
#   - Prevention_Index: 0-10 (media ~5.14, std ~2.42)
#   - Access_Score: 0-1 (media ~0.70, std ~0.46)
#
# El formulario genera:
#   - Risk_Score: 0-150 (suma de factores: 20+20+20+...)
#   - Prevention_Index: 0-100 (suma de factores: 20+20+20+...)
#   - Access_Score: 50 o 70 (discreto: rural=50, urban=70)
#
# Factores de diferencia: 15x, 10x, 100x respectivamente
# Por eso el modelo siempre predice 100% sin este escalamiento.

DATASET_RISK_MAX = 10.0
DATASET_PREVENTION_MAX = 10.0
FORM_RISK_MAX = 150.0
FORM_PREVENTION_MAX = 100.0


def calculate_age_risk_group(age):
    """Clasifica edad en grupos de riesgo."""
    if age < 40:
        return 'Low_Risk'
    elif age < 50:
        return 'Medium_Risk'
    elif age < 60:
        return 'High_Risk'
    else:
        return 'Very_High_Risk'


def calculate_lifestyle_categories(data_dict):
    """Calcula las categorías de estilo de vida (one-hot encoded)."""
    lifestyle = {
        'LC_Dietary': 1 if data_dict.get('Diet_Risk') == 'Yes' else 0,
        'LC_Healthy': 1 if data_dict.get('Physical_Activity') == 'Yes' else 0,
        'LC_High_Risk': 1 if data_dict.get('Smoking_History') == 'Yes' or data_dict.get('Alcohol_Consumption') == 'Yes' else 0,
        'LC_Sedentary': 1 if data_dict.get('Physical_Activity') == 'No' else 0,
    }
    return lifestyle


# ─────────────────────────────────────────────────────────────────────────────
# ⭐ NUEVA FUNCIÓN: ESCALAR ÍNDICES AL RANGO DEL DATASET
# ─────────────────────────────────────────────────────────────────────────────
def scale_indices_to_dataset_range(risk_score, prevention_index, access_score):
    """
    ⭐ CRÍTICO: Escala los índices del formulario al rango del dataset.
    
    Esta es la SOLUCIÓN al problema del 100% en predicción.
    
    Conversiones:
      Risk_Score:       0-150 → 0-10 (divide por 15)
      Prevention_Index: 0-100 → 0-10 (divide por 10)
      Access_Score:     50-70 → 0-1  (normaliza a 0 o 1)
    
    Args:
        risk_score (float): Valor en rango 0-150
        prevention_index (float): Valor en rango 0-100
        access_score (float): Valor en rango 50-70
    
    Returns:
        tuple: (risk_scaled, prevention_scaled, access_scaled) en rango del dataset
    """
    # Escalar Risk_Score de 0-150 a 0-10
    risk_score_scaled = (risk_score / FORM_RISK_MAX) * DATASET_RISK_MAX
    
    # Escalar Prevention_Index de 0-100 a 0-10
    prevention_index_scaled = (prevention_index / FORM_PREVENTION_MAX) * DATASET_PREVENTION_MAX
    
    # Convertir Access_Score de 50-70 a 0-1
    # En el dataset: 0.0 (rural) o 1.0 (urban), con media 0.70
    access_score_scaled = 1.0 if access_score >= 60 else 0.0
    
    return risk_score_scaled, prevention_index_scaled, access_score_scaled


def prepare_features_for_model(data_dict, feature_names, n_features_expected):
    """Prepara el vector de features en el orden correcto del modelo según scaler."""
    feature_values = []
    
    # Excluir 'Diagnosis' del procesamiento (es el target, no una feature)
    features_to_use = [f for f in feature_names if f != 'Diagnosis']
    
    # Asegurar que usamos el número correcto de features
    max_features = min(len(features_to_use), n_features_expected)
    
    for i, feat in enumerate(features_to_use[:max_features]):
        if feat in data_dict:
            val = data_dict[feat]
            # Convertir categóricas a numéricas
            if isinstance(val, str):
                if val == 'Yes':
                    feature_values.append(1)
                elif val == 'No':
                    feature_values.append(0)
                elif val == 'Male':
                    feature_values.append(1)
                elif val == 'Female':
                    feature_values.append(0)
                elif val == 'Other':
                    feature_values.append(0.5)
                elif val == 'Urban':
                    feature_values.append(1)
                elif val == 'Rural':
                    feature_values.append(0)
                elif val in ['Low_Risk', 'Medium_Risk', 'High_Risk', 'Very_High_Risk']:
                    risk_mapping = {
                        'Low_Risk': 1, 'Medium_Risk': 2, 'High_Risk': 3, 'Very_High_Risk': 4
                    }
                    feature_values.append(risk_mapping.get(val, 2))
                else:
                    feature_values.append(0)
            else:
                feature_values.append(float(val))
        else:
            feature_values.append(0)
    
    # Rellenar con ceros si hace falta llegar al número esperado
    while len(feature_values) < n_features_expected:
        feature_values.append(0)
    
    return np.array(feature_values[:n_features_expected], dtype=float).reshape(1, -1)


# ─────────────────────────────────────────────────────────────────────────────
# INTERFAZ PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    page_header(
        "Cribado General — Evaluación Tabular",
        "Predicción de riesgo de cáncer colorrectal basada en factores genéricos.",
        icon="📋",
    )
    
    # Cargar modelos
    model, scaler, config = load_model_artifacts()
    
    if model is None:
        show_empty_state(
            "⚠️",
            "Modelo no disponible",
            "El modelo genérico tabular no se encuentra en models/saved/generic_tabular. "
            "Ejecute primero src/training/train_generic_tabular.py",
        )
        return
    
    feature_names = config.get('feature_names', [])
    best_threshold = config.get('best_threshold', 0.5)
    
    # ── Sección 1: Formulario de entrada ────────────────────────────────────
    st.markdown("<div class='section-title'>📝 Datos del Paciente</div>", unsafe_allow_html=True)
    
    with st.form("generic_tabular_form"):
        # ── Datos Demográficos ──
        st.markdown("**👤 Demografía**")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Edad", 18, 90, 45, key="age")
        
        with col2:
            gender = st.selectbox("Género", ['Male', 'Female', 'Other'], key="gender")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ── Historial Médico ──
        st.markdown("**🏥 Historial Médico y Factores de Riesgo**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            family_history = st.checkbox("Historia familiar", key="family_history")
        with col2:
            diabetes = st.checkbox("Diabetes", key="diabetes")
        with col3:
            ibd = st.checkbox("Enfermedad inflamatoria intestinal", key="ibd")
        with col4:
            genetic_mutation = st.checkbox("Mutación genética conocida", key="genetic_mutation")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ── Hábitos y Estilo de Vida ──
        st.markdown("**🏃 Hábitos y Estilo de Vida**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            smoking = st.checkbox("Antecedentes de tabaquismo", key="smoking")
        with col2:
            alcohol = st.checkbox("Consumo regular de alcohol", key="alcohol")
        with col3:
            diet_risk = st.checkbox("Dieta de alto riesgo", key="diet_risk")
        with col4:
            physical_activity = st.checkbox("Actividad física regular", key="activity")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ── Medidas de Salud ──
        st.markdown("**📊 Medidas de Salud**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bmi = st.slider("BMI (Índice de masa corporal)", 15.0, 50.0, 25.0, 0.1, key="bmi")
        
        with col2:
            urban_rural = st.selectbox("Localización", ['Urban', 'Rural'], key="urban_rural")
        
        with col3:
            pass  # Espacio en blanco
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ── Historial de Cribado ──
        st.markdown("**🔍 Historial de Cribado**")
        col1, col2 = st.columns(2)
        
        with col1:
            screening_history = st.checkbox("Historial de cribado previo", key="screening_history")
        
        with col2:
            early_detection = st.checkbox("Detección temprana previa", key="early_detection")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        submitted = st.form_submit_button("🧠 Predecir Riesgo", type="primary", use_container_width=True)
    
    # ── Predicción ──
    if submitted:
        # Construir diccionario de datos
        data_dict = {
            'Age': age,
            'Gender': gender,
            'Family_History': 'Yes' if family_history else 'No',
            'Smoking_History': 'Yes' if smoking else 'No',
            'Alcohol_Consumption': 'Yes' if alcohol else 'No',
            'Obesity_BMI': bmi,
            'Diet_Risk': 'Yes' if diet_risk else 'No',
            'Physical_Activity': 'Yes' if physical_activity else 'No',
            'Diabetes': 'Yes' if diabetes else 'No',
            'Inflammatory_Bowel_Disease': 'Yes' if ibd else 'No',
            'Genetic_Mutation': 'Yes' if genetic_mutation else 'No',
            'Screening_History': 'Yes' if screening_history else 'No',
            'Early_Detection': 'Yes' if early_detection else 'No',
            'Urban_or_Rural': urban_rural,
        }
        
        # Calcular índices (en escala del formulario: 0-150, 0-100, 50-70)
        risk_score, prevention_index, access_score = calculate_risk_indices(data_dict)
        
        # ⭐ SOLUCIÓN A: Escalar los índices al rango del dataset (0-10, 0-10, 0-1)
        # Esta es la corrección definitiva para el problema del 100% en predicción
        risk_score, prevention_index, access_score = scale_indices_to_dataset_range(
            risk_score, prevention_index, access_score
        )
        
        age_risk_group = calculate_age_risk_group(age)
        lifestyle_categories = calculate_lifestyle_categories(data_dict)
        
        # Completar dict con índices ESCALADOS (ahora en rango del dataset)
        data_dict.update({
            'Risk_Score': risk_score,              # ⭐ 0-10
            'Prevention_Index': prevention_index,  # ⭐ 0-10
            'Access_Score': access_score,          # ⭐ 0-1
            'Age_Risk_Group': age_risk_group,
            **lifestyle_categories,
        })
        
        # Preparar features para el modelo
        # El scaler tiene información sobre cuántas features espera
        n_features_expected = getattr(scaler, 'n_features_in_', 22)
        X_input = prepare_features_for_model(data_dict, feature_names, n_features_expected)
        X_scaled = scaler.transform(X_input)
        
        # ⭐ ARREGLADO: Agregar validación de escala
        max_scaled_val = np.abs(X_scaled).max()
        if max_scaled_val > 20:
            st.warning(
                f"⚠️ **Advertencia técnica:** Vector escalado con valores extremos (max={max_scaled_val:.2f}). "
                f"Esto podría afectar la precisión de la predicción. "
                f"Se recomienda revisar la configuración del modelo."
            )
        
        # Predicción
        y_prob = model.predict_proba(X_scaled)[0, 1]
        y_pred = 1 if y_prob >= best_threshold else 0
        
        # ── DEBUG INFO ──
        with st.expander("🔍 Debug: Información Técnica"):
            features_used = [f for f in feature_names if f != 'Diagnosis']
            st.write(f"**Features en modelo:** {len(feature_names)} (incluye Diagnosis)")
            st.write(f"**Features utilizadas para entrada:** {len(features_used)}")
            st.write(f"**Features entrada prepared:** {X_input.shape[1]}")
            st.write(f"**Features esperadas por scaler:** {n_features_expected}")
            st.write(f"**Umbral decisión:** {best_threshold:.4f}")
            st.write(f"**Probabilidad CRC (raw):** {y_prob:.6f}")
            st.write(f"**Predicción:** {'⚠️ ALTO RIESGO' if y_pred == 1 else '✅ BAJO RIESGO'}")
            
            st.write("\n**Índices (Formulario → Dataset):**")
            st.write(f"- Risk_Score: {risk_score:.1f} (0-150) → {data_dict.get('Risk_Score', 0):.4f} (0-10)")
            st.write(f"- Prevention_Index: {prevention_index:.1f} (0-100) → {data_dict.get('Prevention_Index', 0):.4f} (0-10)")
            st.write(f"- Access_Score: {access_score:.1f} (50-70) → {data_dict.get('Access_Score', 0):.4f} (0-1)")
            
            with st.expander("📋 Ver valores de features (primeras 10)"):
                df_features = pd.DataFrame({
                    'Feature': features_used[:10],
                    'Valor Original': X_input[0, :10],
                    'Valor Escalado': X_scaled[0, :10],
                })
                st.dataframe(df_features)
        
        # ── RESULTADOS ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📊 Resultados de Predicción</div>", unsafe_allow_html=True)
        
        # Usar banner de resultado
        if y_pred == 1:
            show_result_banner(
                "⚠️  RIESGO ALTO",
                f"Probabilidad de poder tener cáncer de colon: {y_prob*100:.1f}%",
                "red",
            )
            st.warning("Se recomienda evaluación inmediata con colonoscopia y consulta con gastroenterólogo.")
        else:
            show_result_banner(
                "✅ RIESGO BAJO",
                f"Probabilidad de poder tener cáncer de colon: {y_prob*100:.1f}%",
                "green",
            )
            st.success("Continúe con cribado preventivo según directrices clínicas.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ── Gráficas de resultados ──
        col_chart1, col_chart2 = st.columns(2)
        
        # Gráfica 1: Probabilidad
        with col_chart1:
            fig, ax = plt.subplots(figsize=(8, 5))
            categories = ['Sin CRC', 'Con CRC']
            probs = [1 - y_prob, y_prob]
            colors = ['#388E3C', '#D32F2F']
            bars = ax.bar(categories, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            ax.set_ylabel('Probabilidad', fontsize=11, fontweight='bold')
            ax.set_title('Distribución Predicha', fontsize=13, fontweight='bold')
            ax.set_ylim([0, 1])
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Gráfica 2: Factores de riesgo vs protección
        with col_chart2:
            fig, ax = plt.subplots(figsize=(8, 5))
            factors = ['Risk\nScore', 'Prevention\nIndex']
            values = [risk_score, prevention_index]
            colors_risk = ['#D32F2F', '#388E3C']
            bars = ax.bar(factors, values, color=colors_risk, alpha=0.7, edgecolor='black', linewidth=2)
            ax.set_ylabel('Puntuación', fontsize=11, fontweight='bold')
            ax.set_title('Factores de Riesgo vs Protección (Escalas Originales)', fontsize=13, fontweight='bold')
            ax.set_ylim([0, 110])
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ── Detalles de factores ──
        st.markdown("<div class='section-title'>🔍 Detalles de Evaluación</div>", unsafe_allow_html=True)
        
        col_factors1, col_factors2 = st.columns(2)
        
        with col_factors1:
            st.markdown("**Risk Factors (❌ Factores de Riesgo):**")
            st.markdown(f"- **Risk Score:** {risk_score:.1f}/150")
            st.markdown(f"- **Edad:** {age} años ({age_risk_group})")
            if family_history:
                st.markdown("- ✗ Historia familiar positiva")
            if smoking:
                st.markdown("- ✗ Antecedentes de tabaquismo")
            if alcohol:
                st.markdown("- ✗ Consumo de alcohol")
            if diabetes:
                st.markdown("- ✗ Diabetes")
            if ibd:
                st.markdown("- ✗ Enfermedad inflamatoria intestinal")
            if genetic_mutation:
                st.markdown("- ✗ Mutación genética")
            if diet_risk:
                st.markdown("- ✗ Dieta de alto riesgo")
            if bmi > 30:
                st.markdown(f"- ✗ Obesidad (BMI={bmi:.1f})")
        
        with col_factors2:
            st.markdown("**Protective Factors (✅ Factores Protectores):**")
            st.markdown(f"- **Prevention Index:** {prevention_index:.1f}/100")
            if physical_activity:
                st.markdown("- ✓ Actividad física regular")
            if screening_history:
                st.markdown("- ✓ Historial de cribado")
            if early_detection:
                st.markdown("- ✓ Detección temprana previa")
            st.markdown(f"- **Access Score:** {access_score:.1f}/100")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ── Recomendaciones ──
        st.markdown("<div class='section-title'>💊 Recomendaciones Clínicas</div>", unsafe_allow_html=True)
        
        if y_pred == 1:
            st.warning(
                "🏥 **EVALUACIÓN INMEDIATA RECOMENDADA**\n\n"
                "- Derivación a gastroenterología\n"
                "- Colonoscopia de evaluación\n"
                "- Análisis sanguíneo completo (hemograma, marcadores tumorales)\n"
                "- Seguimiento intensificado"
            )
        else:
            if risk_score > 60:
                st.info(
                    "🔍 **SEGUIMIENTO RECOMENDADO**\n\n"
                    "- Cribado colonoscópico cada 5-10 años (según directrices)\n"
                    "- Modificación de estilos de vida\n"
                    "- Reevaluación anual del riesgo"
                )
            else:
                st.success(
                    "✅ **CRIBADO PREVENTIVO ESTÁNDAR**\n\n"
                    "- Mantenerse activo físicamente\n"
                    "- Dieta rica en fibra\n"
                    "- Colonoscopia cada 10 años (según edad y directrices)"
                )


if __name__ == "__main__":
    render()