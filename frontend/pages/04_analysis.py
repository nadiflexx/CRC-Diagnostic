import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN DE RUTAS (Path Injection Dinámico)
# ────────────────────────────────────────────────────────────────────────────────
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.config.constants import REVERSE_ANALYSIS_SMOKE_FEATURES
    from src.config.paths import paths
    
    MODEL_FILE_PATH = f"{paths.REVERSE_LOGIC_MODEL_DIR}/reverse_logic_smk_stat_type_cd.pkl"
except ImportError as e:
    st.error(f"❌ Error crítico de importación: {e}")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# 2. VALORES POR DEFECTO (Profile de prueba)
# ────────────────────────────────────────────────────────────────────────────────
DEFAULT_VALUES = {
    "sex": 1,
    "age": 45,
    "height": 175.0,
    "weight": 75.0,
    "waistline": 85.0,
    "BMI": 24.49,
    "triglyceride": 120.0,
    "HDL_chole": 50.0,
    "LDL_chole": 110.0,
    "hemoglobin": 15.0,
    "waist_height_ratio": 0.0,  # Se calculará automáticamente
    "hemoglobin_per_height": 0.0  # Se calculará automáticamente
}

# ────────────────────────────────────────────────────────────────────────────────
# 3. INTERFAZ GRÁFICA
# ────────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Test Modelo Smoking", layout="wide")

st.title("🧪 Prueba Real - Modelo Reverse Logic (Smoking)")
st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────────
# 4. CARGA DEL MODELO
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Carga el modelo pkl desde la ruta configurada"""
    try:
        model_file = Path(MODEL_FILE_PATH)
        if not model_file.exists():
            raise FileNotFoundError(f"El modelo NO se encontró en:\n{model_file}")
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        return model
    except Exception as e:
        st.error(f"⚠️ Error al cargar el modelo: {e}")
        return None

model = load_model()

if model:
    st.success("✅ Modelo cargado exitosamente.")
    
    # Mostrar features requeridas
    with st.expander("ℹ️ Features requeridas por el modelo"):
        st.code(", ".join(REVERSE_ANALYSIS_SMOKE_FEATURES))
    
    # ────────────────────────────────────────────────────────────────────────────
    # 5. FORMULARIO DE ENTRADA
    # ────────────────────────────────────────────────────────────────────────────
    
    st.subheader("📝 Datos Clínicos Requeridos")
    
    col_demographics, col_metabolic, col_engineered = st.columns(3)
    
    with col_demographics:
        st.info("**Demografía Básica**")
        sex = st.selectbox(
            "Sexo", 
            options=[0, 1], 
            format_func=lambda x: "Femenino" if x == 0 else "Masculino",
            index=DEFAULT_VALUES["sex"]
        )
        age = st.number_input(
            "Edad (años)", 
            min_value=18, 
            max_value=120, 
            value=DEFAULT_VALUES["age"]
        )
        height = st.number_input(
            "Altura (cm)", 
            min_value=100.0, 
            max_value=250.0, 
            value=DEFAULT_VALUES["height"]
        )
        weight = st.number_input(
            "Peso (kg)", 
            min_value=30.0, 
            max_value=200.0, 
            value=DEFAULT_VALUES["weight"]
        )
        BMI = st.number_input(
            "BMI", 
            min_value=15.0, 
            max_value=50.0, 
            value=DEFAULT_VALUES["BMI"],
            help="Índice de Masa Corporal"
        )
    
    with col_metabolic:
        st.info("**Marcadores Metabólicos**")
        waistline = st.number_input(
            "Perímetro de Cintura (cm)", 
            min_value=40.0, 
            max_value=150.0, 
            value=DEFAULT_VALUES["waistline"]
        )
        triglyceride = st.number_input(
            "Triglicéridos (mg/dL)", 
            min_value=20.0, 
            max_value=600.0, 
            value=DEFAULT_VALUES["triglyceride"]
        )
        HDL_chole = st.number_input(
            "HDL Colesterol (mg/dL)", 
            min_value=10.0, 
            max_value=150.0, 
            value=DEFAULT_VALUES["HDL_chole"]
        )
        LDL_chole = st.number_input(
            "LDL Colesterol (mg/dL)", 
            min_value=20.0, 
            max_value=300.0, 
            value=DEFAULT_VALUES["LDL_chole"]
        )
        hemoglobin = st.number_input(
            "Hemoglobina (g/dL)", 
            min_value=5.0, 
            max_value=25.0, 
            value=DEFAULT_VALUES["hemoglobin"]
        )
    
    with col_engineered:
        st.info("**Features Ingenierizadas**")
        st.caption("💡 Dejar en 0 para calcular automáticamente")
        
        waist_height_ratio = st.number_input(
            "Ratio Cintura/Altura", 
            min_value=0.0, 
            max_value=1.5, 
            step=0.01, 
            value=0.0,
            help="Se calculará automáticamente si se deja en 0 (waistline/height)"
        )
        
        hemoglobin_per_height = st.number_input(
            "Hemoglobina/Altura", 
            min_value=0.0, 
            max_value=0.5, 
            step=0.001, 
            value=0.0,
            help="Se calculará automáticamente si se deja en 0 (hemoglobin/height)"
        )

    # ────────────────────────────────────────────────────────────────────────────
    # 6. FUNCIÓN PARA CALCULAR FEATURES DERIVADAS
    # ────────────────────────────────────────────────────────────────────────────
    
    def calculate_engineered_features(data):
        """Calcula features derivadas si no se proporcionaron"""
        if data['waist_height_ratio'] == 0:
            data['waist_height_ratio'] = data['waistline'] / data['height']
        
        if data['hemoglobin_per_height'] == 0:
            data['hemoglobin_per_height'] = data['hemoglobin'] / data['height']
        
        return data

    # ────────────────────────────────────────────────────────────────────────────
    # 7. BOTÓN DE PREDICCIÓN
    # ────────────────────────────────────────────────────────────────────────────
    
    st.markdown("---")
    
    if st.button("🔮 Ejecutar Predicción", type="primary", use_container_width=True):
        
        # Crear diccionario con datos
        input_data = {
            "sex": sex,
            "age": age,
            "height": height,
            "BMI": BMI,
            "weight": weight,
            "waistline": waistline,
            "triglyceride": triglyceride,
            "HDL_chole": HDL_chole,
            "LDL_chole": LDL_chole,
            "hemoglobin": hemoglobin,
            "waist_height_ratio": waist_height_ratio,
            "hemoglobin_per_height": hemoglobin_per_height
        }
        
        # Calcular features derivadas
        input_data = calculate_engineered_features(input_data)
        
        # Crear DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Reordenar columnas según REVERSE_ANALYSIS_SMOKE_FEATURES
        df_input = df_input[REVERSE_ANALYSIS_SMOKE_FEATURES]
        
        # Mostrar datos procesados
        with st.expander("📊 Ver datos de entrada procesados"):
            st.dataframe(df_input.style.format("{:.4f}"))
            st.caption(f"Features calculadas: waist_height_ratio={input_data['waist_height_ratio']:.4f}, hemoglobin_per_height={input_data['hemoglobin_per_height']:.4f}")
        
        try:
            # Ejecutar predicción
            prediction = model.predict(df_input)[0]
            proba = model.predict_proba(df_input)[0]
            
            # Obtener resultados
            result_class = "NO FUMADOR" if prediction == 0 else "FUMADOR"
            prob_no_smoke = proba[0]
            prob_smoke = proba[1]
            max_confidence = max(proba)
            
            # ────────────────────────────────────────────────────────────────────
            # 8. MOSTRAR RESULTADOS
            # ────────────────────────────────────────────────────────────────────
            
            st.markdown("---")
            st.subheader("🎯 Resultados de la Predicción")
            
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.metric(
                    label="Predicción",
                    value=result_class,
                    delta=None
                )
            
            with m2:
                st.metric(
                    label="Probabilidad NO FUMADOR",
                    value=f"{prob_no_smoke:.2%}"
                )
            
            with m3:
                st.metric(
                    label="Probabilidad FUMADOR",
                    value=f"{prob_smoke:.2%}"
                )
            
            # ────────────────────────────────────────────────────────────────────
            # 9. VALIDACIÓN DEL RESULTADO
            # ────────────────────────────────────────────────────────────────────
            
            st.markdown("---")
            st.subheader("✅ Validación del Resultado")
            
            if max_confidence >= 0.80:
                st.success(f"🟢 **RESULTADO VÁLIDO** - Alta confianza ({max_confidence:.2%})")
                st.info("El modelo tiene alta certeza en esta predicción.")
            elif max_confidence >= 0.60:
                st.warning(f"🟡 **RESULTADO MODERADO** - Confianza media ({max_confidence:.2%})")
                st.info("El modelo tiene confianza moderada. Se recomienda validación adicional.")
            else:
                st.error(f"🔴 **RESULTADO INCIERTO** - Baja confianza ({max_confidence:.2%})")
                st.info("El modelo tiene baja certeza. Se recomienda evaluación clínica detallada.")
            
            # ────────────────────────────────────────────────────────────────────
            # 10. ANÁLISIS DE FEATURES
            # ────────────────────────────────────────────────────────────────────
            
            st.markdown("---")
            st.subheader("📈 Análisis de Características Clave")
            
            col_feat1, col_feat2 = st.columns(2)
            
            with col_feat1:
                st.markdown("**Features más importantes:**")
                st.write(f"• Hemoglobina/Altura: **{input_data['hemoglobin_per_height']:.4f}**")
                st.write(f"• Ratio Cintura/Altura: **{input_data['waist_height_ratio']:.4f}**")
                st.write(f"• Hemoglobina: **{input_data['hemoglobin']:.2f}** g/dL")
            
            with col_feat2:
                st.markdown("**Marcadores adicionales:**")
                st.write(f"• Cintura: **{input_data['waistline']:.1f}** cm")
                st.write(f"• Triglicéridos: **{input_data['triglyceride']:.1f}** mg/dL")
                st.write(f"• BMI: **{input_data['BMI']:.2f}**")
            
            # Información adicional
            with st.expander("ℹ️ Interpretación de resultados"):
                st.markdown("""
                ### Niveles de confianza:
                
                - **🟢 Alta (>80%)**: El modelo está muy seguro de la predicción
                - **🟡 Media (60-80%)**: Predicción probable pero requiere validación
                - **🔴 Baja (<60%)**: Incertidumbre alta, se necesita más información
                
                ### Features más relevantes para detectar tabaquismo:
                
                1. **Hemoglobina/Altura** (peso: 0.25) - Indicador más fuerte
                2. **Ratio Cintura/Altura** (peso: 0.22) - Indicador metabólico
                3. **Hemoglobina** (peso: 0.20) - Marcador directo
                4. **Cintura** (peso: 0.18) - Indicador metabólico
                5. **Triglicéridos** (peso: 0.15) - Marcador metabólico
                
                ⚠️ **Nota:** Este modelo es una herramienta de apoyo y no sustituye el diagnóstico médico profesional.
                """)
                
        except ValueError as ve:
            st.error(f"❌ Error en la predicción: {ve}")
            st.info("Verifica que todos los campos sean válidos.")
        except Exception as e:
            st.error(f"❌ Error inesperado: {e}")
            st.exception(e)

else:
    st.error("❌ No se pudo cargar el modelo")
    st.info(f"**Ruta esperada:** `{MODEL_FILE_PATH}`")

# ────────────────────────────────────────────────────────────────────────────────
# FOOTER
# ────────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>🏥 Sistema de Predicción de Tabaquismo basado en Biomarcadores Clínicos</p>
    <p style='font-size: 0.8em;'>Modelo: Reverse Logic | Features: 12 variables clínicas</p>
</div>
""", unsafe_allow_html=True)