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
    probs = img.get("probabilities", {})  # dict clase→probabilidad

    class_map = {
        "polyp": ("Pólipo Detectado", "red"),
        "inflammation": ("Inflamación / Colitis", "orange"),
    }
    label, level = class_map.get(cls, ("Mucosa Sana", "green"))
    show_result_banner(f"Diagnóstico: {label}", f"{score:.1%}", level)

    # ── Grid de imágenes ──────────────────────────────────────
    _render_image_grid(uploaded, img)

    # ── Segmentación ──────────────────────────────────────────
    if img.get("report_path"):
        render_segmentation(img["report_path"])

    # ── Expander de detalle ───────────────────────────────────
    _render_detail_expander(img, cls, score, probs)


def _render_image_grid(uploaded, img: dict) -> None:
    """
    Renderiza el grid de imágenes:
      - Siempre: imagen original
      - Si hay Grad-CAM (modo PyTorch): A, B, Fusión
      - Si no hay Grad-CAM (modo ONNX): aviso informativo
    """
    cards_html = ""
    cards_html += image_grid_card(
        get_image_base64(uploaded), "Original", "Imagen de entrada"
    )

    has_gradcam = bool(img.get("gradcam_fusion") or img.get("gradcam_a"))

    if has_gradcam:
        alpha = img.get("alpha_used", 0)
        beta = img.get("beta_used", 0)

        # Grad-CAM A (puede existir aunque B no)
        if img.get("gradcam_a"):
            cards_html += image_grid_card(
                img["gradcam_a"],
                "Modelo A",
                f"Atención Contexto ({alpha:.0%})",
            )
        else:
            cards_html += _placeholder_card("Modelo A", "No disponible")

        # Grad-CAM B
        if img.get("gradcam_b"):
            cards_html += image_grid_card(
                img["gradcam_b"],
                "Modelo B",
                f"Atención Tejido ({beta:.0%})",
            )
        else:
            cards_html += _placeholder_card("Modelo B", "No disponible")

        # Fusión
        if img.get("gradcam_fusion"):
            cards_html += image_grid_card(
                img["gradcam_fusion"],
                "Fusión Ensemble",
                "Decisión Final",
                special=True,
            )

    else:
        # Modo ONNX: sin Grad-CAM, mostrar aviso
        cards_html += _onnx_notice_card()

    render_xai_grid(cards_html)


def _placeholder_card(title: str, subtitle: str) -> str:
    """Card vacía con texto informativo."""
    return f"""
    <div style="
        background: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        color: #6c757d;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 200px;
    ">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔲</div>
        <div style="font-weight: 600; font-size: 0.9rem;">{title}</div>
        <div style="font-size: 0.8rem; margin-top: 0.25rem;">{subtitle}</div>
    </div>
    """


def _onnx_notice_card() -> str:
    """Card informativa cuando se usa backend ONNX (sin Grad-CAM)."""
    return """
    <div style="
        grid-column: span 3;
        background: linear-gradient(135deg, #e8f4f8 0%, #d1ecf1 100%);
        border: 1px solid #bee5eb;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        color: #0c5460;
    ">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚡</div>
        <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 0.4rem;">
            Backend ONNX Runtime activo
        </div>
        <div style="font-size: 0.85rem; opacity: 0.85;">
            La inferencia ONNX no genera mapas Grad-CAM (requiere backend PyTorch).
            La clasificación y segmentación son equivalentes.
        </div>
    </div>
    """


def _render_detail_expander(
    img: dict,
    cls: str,
    score: float,
    probs: dict,
) -> None:
    """
    Expander con detalle completo:
      - Métricas principales
      - Distribución de probabilidades de TODAS las clases
      - Info del ensemble
    """
    with st.expander("📊 Detalle de Confianza del Modelo", expanded=False):
        # ── Fila 1: métricas principales ──────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Clase Predicha", cls.title())
        c2.metric("Confianza", f"{score:.1%}")
        c3.metric(
            "Ensemble Weights",
            f"α={img.get('alpha_used', 0):.2f} / β={img.get('beta_used', 0):.2f}",
        )

        st.markdown("---")

        # ── Fila 2: probabilidades de todas las clases ─────────
        if probs:
            st.markdown(
                "<p style='font-weight:600; margin-bottom:0.5rem;'>"
                "📈 Distribución de probabilidades</p>",
                unsafe_allow_html=True,
            )

            # Ordenar de mayor a menor
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

            for class_name, prob in sorted_probs:
                is_predicted = class_name.lower() == cls.lower()

                # Color según clase
                color_map = {
                    "polyp": "#D32F2F",
                    "inflammation": "#F57C00",
                    "normal": "#388E3C",
                }
                bar_color = color_map.get(class_name.lower(), "#5a8d7f")
                star = " ⭐" if is_predicted else ""

                # Etiqueta con estrella si es la clase predicha
                label = f"<b>{class_name.title()}</b>{star}"

                col_label, col_bar, col_pct = st.columns([2, 5, 1])

                with col_label:
                    st.markdown(
                        f"<p style='margin:0; padding-top:6px; font-size:0.9rem;'>"
                        f"{label}</p>",
                        unsafe_allow_html=True,
                    )

                with col_bar:
                    # Barra de progreso coloreada via HTML
                    bar_width = f"{prob * 100:.1f}%"
                    opacity = "1.0" if is_predicted else "0.5"
                    st.markdown(
                        f"""
                        <div style="
                            background: #f0f0f0;
                            border-radius: 6px;
                            height: 22px;
                            margin-top: 4px;
                            overflow: hidden;
                        ">
                            <div style="
                                width: {bar_width};
                                background: {bar_color};
                                opacity: {opacity};
                                height: 100%;
                                border-radius: 6px;
                                transition: width 0.3s ease;
                            "></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col_pct:
                    st.markdown(
                        f"<p style='margin:0; padding-top:6px; font-size:0.9rem; "
                        f"font-weight: {'700' if is_predicted else '400'};'>"
                        f"{prob:.1%}</p>",
                        unsafe_allow_html=True,
                    )

                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        st.markdown("---")

        # ── Fila 3: detalle ensemble ───────────────────────────
        backend = img.get("ensemble_mode", "unknown")
        has_gradcam = bool(img.get("gradcam_fusion") or img.get("gradcam_a"))

        col_info1, col_info2 = st.columns(2)

        with col_info1:
            st.markdown(
                f"""
                <div style="
                    background: #f8f9fa;
                    border-radius: 8px;
                    padding: 0.75rem 1rem;
                    font-size: 0.85rem;
                ">
                    <b>Backend:</b> {"⚡ ONNX Runtime" if backend == "onnx" else "🔥 PyTorch"}<br>
                    <b>Ensemble:</b> {"✅ Activo" if img.get("ensemble_used") else "❌ Solo Modelo A"}<br>
                    <b>Grad-CAM:</b> {"✅ Disponible" if has_gradcam else "⚠️ No disponible (ONNX)"}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_info2:
            attention = img.get("attention_ratio")
            st.markdown(
                f"""
                <div style="
                    background: #f8f9fa;
                    border-radius: 8px;
                    padding: 0.75rem 1rem;
                    font-size: 0.85rem;
                ">
                    <b>Attention Ratio:</b> {f"{attention:.3f}" if attention else "N/A (ONNX)"}<br>
                    <b>α (Modelo A):</b> {img.get("alpha_used", 0):.3f}<br>
                    <b>β (Modelo B):</b> {img.get("beta_used", 0):.3f}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.info(
            "ℹ️ Los pesos del ensemble se adaptan dinámicamente según la entropía "
            "de cada modelo para cada imagen individual."
        )


render()
