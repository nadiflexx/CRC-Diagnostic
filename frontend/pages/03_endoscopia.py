"""
Endoscopy AI Inference.
"""

from components.banners import show_empty_state, show_result_banner
from components.cards import (
    image_grid_card,
    page_header,
    render_segmentation,
    render_xai_grid,
)
from components.sidebar import render_sidebar
import streamlit as st
from utils.api_client import get_patients, run_diagnosis, upload_colonoscopy_image
from utils.helpers import apply_custom_css, build_patient_options, get_image_base64

st.set_page_config(page_title="Endoscopia IA · Endo-AID", page_icon="🌿", layout="wide")
apply_custom_css()
render_sidebar()


def render() -> None:
    page_header(
        "Inferencia Visual (Endoscopia IA)",
        "Clasificación, Ensemble Adaptativo y Segmentación por U-Net.",
        icon="🔬",
    )

    patients = get_patients()

    if not patients:
        show_empty_state(
            "🔬",
            "No hay pacientes registrados",
            "Registre pacientes antes de ejecutar la inferencia visual.",
        )
        return

    opts = build_patient_options(patients)

    col_sel, _ = st.columns([1, 2])
    with col_sel:
        pid = opts[st.selectbox("Paciente", list(opts.keys()))]

    st.markdown("<br>", unsafe_allow_html=True)

    col_up, col_preview = st.columns([2, 1], gap="large")

    with col_up:
        uploaded = st.file_uploader(
            "Subir Fotograma Endoscópico",
            type=["jpg", "png", "jpeg"],
            help="Imagen de colonoscopia en formato JPG o PNG.",
        )

    with col_preview:
        if uploaded:
            st.markdown(
                "<p style='font-weight:600; color:#5a8d7f; font-size:0.85rem;'>"
                "Previsualización</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="preview-card">
                    <img src="{get_image_base64(uploaded)}" alt="Preview">
                </div>
                """,
                unsafe_allow_html=True,
            )

    if uploaded:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧠 Procesar Fotograma con IA", type="primary"):
            _run_pipeline(pid, uploaded)


def _run_pipeline(pid: int, uploaded) -> None:
    """Executes the full AI inference pipeline."""

    placeholder = st.empty()
    placeholder.markdown(
        """
        <div class="loader-container">
            <div class="loader-spinner">&#8203;</div>
            <div class="loader-text-primary">
                Computando Modelos Neuronales...
            </div>
            <div class="loader-text-secondary">
                Generando Grad-CAMs y Segmentación U-Net
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload_res = upload_colonoscopy_image(pid, uploaded)
    if not upload_res:
        placeholder.empty()
        st.error("❌ Error al subir la imagen. Verifique la conexión.")
        return

    result = run_diagnosis(
        {
            "patient_id": pid,
            "clinical_data": {},
            "image_path": upload_res["saved_path"],
        }
    )

    placeholder.empty()

    if not result or not result.get("image_analysis"):
        st.error("❌ No se recibió respuesta del modelo.")
        return

    img = result["image_analysis"]
    cls = img["prediction_class"]
    score = img["prediction_score"]

    class_map = {
        "polyp": ("Pólipo Detectado", "red"),
        "inflammation": ("Inflamación / Colitis", "orange"),
    }
    label, level = class_map.get(cls, ("Mucosa Sana", "green"))
    show_result_banner(f"Diagnóstico: {label}", f"{score:.1%}", level)

    cards_html = ""
    cards_html += image_grid_card(
        get_image_base64(uploaded), "Original", "Imagen de entrada"
    )

    if img.get("gradcam_fusion"):
        alpha = img.get("alpha_used", 0)
        beta = img.get("beta_used", 0)
        cards_html += image_grid_card(
            img["gradcam_a"], "Modelo A", f"Atención Contexto ({alpha:.0%})"
        )
        cards_html += image_grid_card(
            img["gradcam_b"], "Modelo B", f"Atención Tejido ({beta:.0%})"
        )
        cards_html += image_grid_card(
            img["gradcam_fusion"], "Fusión Ensemble", "Decisión Final", special=True
        )
    else:
        cards_html += '<div style="grid-column: span 3;"></div>'

    render_xai_grid(cards_html)

    if img.get("report_path"):
        render_segmentation(img["report_path"])

    with st.expander("📊 Detalle de Confianza del Modelo", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Clase Predicha", cls.title())
        c2.metric("Confianza", f"{score:.1%}")
        c3.metric(
            "Ensemble Weights",
            f"α={img.get('alpha_used', 0):.2f} / β={img.get('beta_used', 0):.2f}",
        )
        st.info(
            "ℹ️ Los pesos del ensemble se adaptan dinámicamente según la entropía "
            "de cada modelo para cada imagen individual."
        )


render()
