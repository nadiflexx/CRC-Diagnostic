# frontend/pages/endoscopy.py
"""
Endoscopy AI Inference — Image & Video modes.
"""

from __future__ import annotations

import io

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
from utils.helpers import (
    apply_custom_css,
    build_patient_options,
    get_image_base64,
    get_pil_image_base64,
)
from utils.video_processor import (
    FRAME_INTERVAL_SECONDS,
    MAX_VIDEO_DURATION_SECONDS,
    extract_frames_from_video,
)

st.set_page_config(
    page_title="Endoscopia IA · Endo-AID",
    page_icon="🌿",
    layout="wide",
)
apply_custom_css()
render_sidebar()

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_CLASS_MAP = {
    "polyp": ("Pólipo Detectado", "red"),
    "inflammation": ("Inflamación / Colitis", "orange"),
}

_COLOR_MAP = {
    "polyp": "#D32F2F",
    "inflammation": "#F57C00",
    "normal": "#388E3C",
}

_SUMMARY_ICONS = {
    "polyp": "🔴",
    "inflammation": "🟠",
    "normal": "🟢",
}


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


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

    # ── Mode selector ──────────────────────────────────────────────────────
    _render_mode_tabs(pid)


# ─────────────────────────────────────────────────────────────────────────────
#  MODE TABS
# ─────────────────────────────────────────────────────────────────────────────


def _render_mode_tabs(pid: int) -> None:
    """Render Image / Video tab selector."""
    st.markdown(
        """
        <style>
        div[data-testid="stTabs"] button {
            font-size: 0.95rem;
            font-weight: 600;
            padding: 0.5rem 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    tab_image, tab_video = st.tabs(["🖼️  Imagen Endoscópica", "🎥  Vídeo Endoscópico"])

    with tab_image:
        _render_image_mode(pid)

    with tab_video:
        _render_video_mode(pid)


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE MODE  (original behaviour, unchanged)
# ─────────────────────────────────────────────────────────────────────────────


def _render_image_mode(pid: int) -> None:
    """Single-frame upload and inference (original mode)."""
    st.markdown("<br>", unsafe_allow_html=True)
    col_up, col_preview = st.columns([2, 1], gap="large")

    with col_up:
        uploaded = st.file_uploader(
            "Subir Fotograma Endoscópico",
            type=["jpg", "png", "jpeg"],
            help="Imagen de colonoscopia en formato JPG o PNG.",
            key="img_uploader",
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
        if st.button("🧠 Procesar Fotograma con IA", type="primary", key="btn_img"):
            _run_single_pipeline(pid, uploaded)


# ─────────────────────────────────────────────────────────────────────────────
#  VIDEO MODE
# ─────────────────────────────────────────────────────────────────────────────


def _render_video_mode(pid: int) -> None:
    """Video upload → frame extraction → batch inference → summary."""
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Info banner ────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #e8f4f8 0%, #d1ecf1 100%);
            border: 1px solid #bee5eb;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin-bottom: 1.5rem;
            color: #0c5460;
            font-size: 0.88rem;
        ">
            <b>🎥 Modo Vídeo Endoscópico</b><br>
            Sube un vídeo de hasta <b>{MAX_VIDEO_DURATION_SECONDS} segundos</b>.
            El sistema extraerá automáticamente un fotograma cada
            <b>{FRAME_INTERVAL_SECONDS} segundos</b>, analizará cada uno con los modelos
            de IA y generará un <b>resumen clínico global</b> con todos los hallazgos.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_up, col_cfg = st.columns([2, 1], gap="large")

    with col_up:
        video_file = st.file_uploader(
            "Subir Vídeo Endoscópico",
            type=["mp4", "avi", "mov", "mkv"],
            help=f"Vídeo de colonoscopia. Máximo {MAX_VIDEO_DURATION_SECONDS}s.",
            key="video_uploader",
        )

    with col_cfg:
        st.markdown(
            "<p style='font-weight:600; color:#5a8d7f; font-size:0.85rem; "
            "margin-bottom:0.25rem;'>Configuración de Extracción</p>",
            unsafe_allow_html=True,
        )
        interval = st.slider(
            "Intervalo entre fotogramas (s)",
            min_value=1,
            max_value=10,
            value=FRAME_INTERVAL_SECONDS,
            step=1,
            key="frame_interval",
            help="Cada cuántos segundos se extrae un fotograma del vídeo.",
        )
        st.markdown(
            "<p style='font-size:0.8rem; color:#6c757d; margin-top:0.25rem;'>"
            f"⏱ Con {interval}s de intervalo y {MAX_VIDEO_DURATION_SECONDS}s "
            f"de vídeo → máx. "
            f"<b>{MAX_VIDEO_DURATION_SECONDS // interval + 1} fotogramas</b>."
            "</p>",
            unsafe_allow_html=True,
        )

    if video_file:
        # Preview
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("▶️  Previsualizar Vídeo", expanded=False):
            st.video(video_file)
            video_file.seek(0)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(
            "🧠 Analizar Vídeo con IA",
            type="primary",
            key="btn_video",
        ):
            _run_video_pipeline(pid, video_file, interval)


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE IMAGE PIPELINE  (unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────


def _run_single_pipeline(pid: int, uploaded) -> None:
    """Full AI inference pipeline for a single frame."""
    placeholder = st.empty()
    placeholder.markdown(
        _loader_html(
            "Computando Modelos Neuronales...",
            "Generando Grad-CAMs y Segmentación U-Net",
        ),
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
    probs = img.get("probabilities", {})

    label, level = _CLASS_MAP.get(cls, ("Mucosa Sana", "green"))
    show_result_banner(f"Diagnóstico: {label}", f"{score:.1%}", level)

    _render_image_grid(uploaded, img)

    if img.get("report_path"):
        render_segmentation(img["report_path"])

    _render_detail_expander(img, cls, score, probs)


# ─────────────────────────────────────────────────────────────────────────────
#  VIDEO PIPELINE
# ─────────────────────────────────────────────────────────────────────────────


def _run_video_pipeline(pid: int, video_file, interval: int) -> None:
    """
    Full video inference pipeline:
    1. Extract frames
    2. Upload + infer each frame
    3. Display per-frame results
    4. Show global summary
    """

    # ── Step 1: Extract frames ─────────────────────────────────────────────
    extract_placeholder = st.empty()
    extract_placeholder.markdown(
        _loader_html(
            "Extrayendo Fotogramas del Vídeo...",
            f"Un fotograma cada {interval}s",
        ),
        unsafe_allow_html=True,
    )

    try:
        video_bytes = video_file.read()
        frames = extract_frames_from_video(
            video_bytes,
            interval_seconds=float(interval),
            max_duration=float(MAX_VIDEO_DURATION_SECONDS),
        )
    except Exception as e:
        extract_placeholder.empty()
        st.error(f"❌ Error al procesar el vídeo: {e}")
        return

    extract_placeholder.empty()

    if not frames:
        st.warning("⚠️ No se pudieron extraer fotogramas del vídeo.")
        return

    st.markdown(
        f"""
        <div style="
            background: #f8f9fa;
            border-left: 4px solid #5a8d7f;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 1.5rem;
            font-size: 0.88rem;
            color: #2d5a4e;
        ">
            ✅ <b>{len(frames)} fotogramas extraídos</b> correctamente del vídeo.
            Iniciando análisis con IA…
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Step 2: Infer each frame ───────────────────────────────────────────
    frame_results: list[dict] = []
    progress_bar = st.progress(0, text="Analizando fotogramas…")
    status_text = st.empty()

    for i, frame in enumerate(frames):
        status_text.markdown(
            f"<p style='color:#5a8d7f; font-size:0.85rem;'>"
            f"🔬 Analizando fotograma {i + 1}/{len(frames)} "
            f"(t={frame['timestamp']}s)…</p>",
            unsafe_allow_html=True,
        )

        # ── Construir _FrameFile desde los bytes del PIL guardado ──────
        buf = io.BytesIO()
        frame["image"].save(buf, format="JPEG", quality=92)
        frame_bytes = buf.getvalue()  # bytes completos, seguros

        frame_file = _FrameFile(
            data=frame_bytes,
            name=frame["filename"],
        )

        upload_res = upload_colonoscopy_image(pid, frame_file)
        if not upload_res:
            frame_results.append(
                {
                    "timestamp": frame["timestamp"],
                    "pil_image": frame["image"],
                    "error": True,
                    "result": None,
                }
            )
            progress_bar.progress((i + 1) / len(frames))
            continue

        result = run_diagnosis(
            {
                "patient_id": pid,
                "clinical_data": {},
                "image_path": upload_res["saved_path"],
            }
        )

        frame_results.append(
            {
                "timestamp": frame["timestamp"],
                "pil_image": frame["image"],
                "error": result is None or not result.get("image_analysis"),
                "result": result,
            }
        )
        progress_bar.progress((i + 1) / len(frames))

    progress_bar.empty()
    status_text.empty()

    # ── Step 3: Render per-frame results ───────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _render_video_frame_results(frame_results)

    # ── Step 4: Global summary ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _render_video_summary(frame_results)


# ─────────────────────────────────────────────────────────────────────────────
#  VIDEO RESULTS — PER FRAME
# ─────────────────────────────────────────────────────────────────────────────


def _render_video_frame_results(frame_results: list[dict]) -> None:
    """Render an expandable card for each analysed frame."""
    st.markdown(
        """
        <div style="
            font-size: 1.05rem;
            font-weight: 700;
            color: #2d5a4e;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        ">
            🎞️ Resultados por Fotograma
        </div>
        """,
        unsafe_allow_html=True,
    )

    for fr in frame_results:
        ts = fr["timestamp"]
        pil_img = fr["pil_image"]
        img_b64 = get_pil_image_base64(pil_img)

        if fr["error"] or not fr["result"]:
            with st.expander(f"⏱ t={ts}s — ⚠️ Error en el análisis", expanded=False):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.markdown(
                        f'<div class="preview-card">'
                        f'<img src="{img_b64}" alt="Frame {ts}s">'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with col_b:
                    st.error(
                        "No se pudo obtener respuesta del modelo para este fotograma."
                    )
            continue

        img_analysis = fr["result"]["image_analysis"]
        cls = img_analysis["prediction_class"]
        score = img_analysis["prediction_score"]
        probs = img_analysis.get("probabilities", {})
        label, _ = _CLASS_MAP.get(cls, ("Mucosa Sana", "green"))
        color = _COLOR_MAP.get(cls, "#388E3C")
        icon = _SUMMARY_ICONS.get(cls, "🟢")

        expander_label = f"{icon}  t={ts}s — {label}  ({score:.1%})"

        with st.expander(expander_label, expanded=False):
            col_img, col_info = st.columns([1, 2], gap="large")

            with col_img:
                # Original frame
                st.markdown(
                    f'<div class="preview-card">'
                    f'<img src="{img_b64}" alt="Frame {ts}s">'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p style='text-align:center; font-size:0.78rem; "
                    f"color:#6c757d; margin-top:0.25rem;'>t = {ts}s</p>",
                    unsafe_allow_html=True,
                )

            with col_info:
                # Diagnosis badge
                st.markdown(
                    f"""
                    <div style="
                        background: {color}18;
                        border: 2px solid {color};
                        border-radius: 10px;
                        padding: 0.6rem 1rem;
                        margin-bottom: 0.75rem;
                        display: flex;
                        align-items: center;
                        gap: 0.5rem;
                    ">
                        <span style="font-size:1.3rem;">{icon}</span>
                        <div>
                            <div style="font-weight:700; color:{color}; font-size:0.95rem;">
                                {label}
                            </div>
                            <div style="font-size:0.82rem; color:#555;">
                                Confianza: <b>{score:.1%}</b>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Probability bars
                if probs:
                    sorted_probs = sorted(
                        probs.items(), key=lambda x: x[1], reverse=True
                    )
                    for class_name, prob in sorted_probs:
                        bar_color = _COLOR_MAP.get(class_name.lower(), "#5a8d7f")
                        is_pred = class_name.lower() == cls.lower()
                        star = " ⭐" if is_pred else ""
                        st.markdown(
                            f"""
                            <div style="display:flex; align-items:center;
                                        gap:0.5rem; margin-bottom:4px;">
                                <div style="width:90px; font-size:0.8rem;
                                            font-weight:{"700" if is_pred else "400"};
                                            color:#333;">
                                    {class_name.title()}{star}
                                </div>
                                <div style="flex:1; background:#f0f0f0;
                                            border-radius:5px; height:16px;
                                            overflow:hidden;">
                                    <div style="
                                        width:{prob * 100:.1f}%;
                                        background:{bar_color};
                                        opacity:{"1" if is_pred else "0.5"};
                                        height:100%;
                                        border-radius:5px;
                                    "></div>
                                </div>
                                <div style="width:42px; text-align:right;
                                            font-size:0.8rem;
                                            font-weight:{"700" if is_pred else "400"};">
                                    {prob:.1%}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # GradCAM + segmentation inside expander
            has_gradcam = bool(
                img_analysis.get("gradcam_fusion") or img_analysis.get("gradcam_a")
            )

            if has_gradcam:
                st.markdown(
                    "<p style='font-weight:600; color:#5a8d7f; "
                    "font-size:0.82rem; margin-top:0.75rem;'>Mapas de Activación (Grad-CAM)</p>",
                    unsafe_allow_html=True,
                )
                cards_html = image_grid_card(img_b64, "Original", "Fotograma")

                alpha = img_analysis.get("alpha_used", 0)
                beta = img_analysis.get("beta_used", 0)

                if img_analysis.get("gradcam_a"):
                    cards_html += image_grid_card(
                        img_analysis["gradcam_a"],
                        "Modelo A",
                        f"Atención Contexto ({alpha:.0%})",
                    )
                if img_analysis.get("gradcam_b"):
                    cards_html += image_grid_card(
                        img_analysis["gradcam_b"],
                        "Modelo B",
                        f"Atención Tejido ({beta:.0%})",
                    )
                if img_analysis.get("gradcam_fusion"):
                    cards_html += image_grid_card(
                        img_analysis["gradcam_fusion"],
                        "Fusión Ensemble",
                        "Decisión Final",
                        special=True,
                    )
                render_xai_grid(cards_html)

            if img_analysis.get("report_path"):
                render_segmentation(img_analysis["report_path"])


# ─────────────────────────────────────────────────────────────────────────────
#  VIDEO RESULTS — GLOBAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────


def _render_video_summary(frame_results: list[dict]) -> None:
    """Compute and render global findings summary from all analysed frames."""
    valid = [
        fr
        for fr in frame_results
        if not fr["error"] and fr["result"] and fr["result"].get("image_analysis")
    ]

    if not valid:
        st.warning("⚠️ No hay fotogramas válidos para generar el resumen.")
        return

    # ── Aggregate stats ────────────────────────────────────────────────────
    class_counts: dict[str, int] = {"polyp": 0, "inflammation": 0, "normal": 0}
    polyp_timestamps: list[float] = []
    inflammation_timestamps: list[float] = []
    max_polyp_score: float = 0.0
    max_inflammation_score: float = 0.0

    for fr in valid:
        img = fr["result"]["image_analysis"]
        cls = img["prediction_class"].lower()
        score = img["prediction_score"]
        key = cls if cls in class_counts else "normal"
        class_counts[key] += 1
        if key == "polyp":
            polyp_timestamps.append(fr["timestamp"])
            max_polyp_score = max(max_polyp_score, score)
        elif key == "inflammation":
            inflammation_timestamps.append(fr["timestamp"])
            max_inflammation_score = max(max_inflammation_score, score)

    total = len(valid)
    polyp_pct = class_counts["polyp"] / total
    inflammation_pct = class_counts["inflammation"] / total
    normal_pct = class_counts["normal"] / total
    avg_confidence = (
        sum(fr["result"]["image_analysis"]["prediction_score"] for fr in valid) / total
    )

    # ── Overall risk level ─────────────────────────────────────────────────
    if class_counts["polyp"] > 0:
        overall_risk = "ALTO"
        risk_color = "#D32F2F"
        overall_icon = "🔴"
        overall_label = "Pólipo detectado — Requiere atención urgente"
        risk_bg = "rgba(211,47,47,0.08)"
    elif class_counts["inflammation"] > 0:
        overall_risk = "MODERADO"
        risk_color = "#F57C00"
        overall_icon = "🟠"
        overall_label = "Signos inflamatorios — Seguimiento recomendado"
        risk_bg = "rgba(245,124,0,0.08)"
    else:
        overall_risk = "BAJO"
        risk_color = "#388E3C"
        overall_icon = "🟢"
        overall_label = "Mucosa normal — Sin hallazgos significativos"
        risk_bg = "rgba(56,142,60,0.08)"

    # ── Render header card ─────────────────────────────────────────────────
    # Construimos el HTML en partes para evitar problemas con f-strings anidados
    header_html = (
        '<div style="'
        "background: linear-gradient(135deg, #f8fffe 0%, #edf7f4 100%);"
        "border: 2px solid #5a8d7f;"
        "border-radius: 16px;"
        "padding: 1.75rem 2rem;"
        "margin-bottom: 1.5rem;"
        'box-shadow: 0 4px 20px rgba(90,141,127,0.12);">'
        # Título
        '<div style="'
        "display: flex;"
        "align-items: center;"
        "gap: 0.75rem;"
        "margin-bottom: 1.25rem;"
        "border-bottom: 1px solid #d0e8e0;"
        'padding-bottom: 0.75rem;">'
        '<span style="font-size:1.8rem;">📋</span>'
        "<div>"
        '<div style="font-size:1.1rem; font-weight:700; color:#2d5a4e;">'
        "Resumen Clínico Global del Vídeo"
        "</div>"
        '<div style="font-size:0.82rem; color:#6c757d;">'
        f"{total} fotogramas analizados · Confianza media: <b>{avg_confidence:.1%}</b>"
        "</div>"
        "</div>"
        "</div>"
        # Risk badge — sin comentario HTML, colores via rgba()
        f'<div style="'
        f"background: {risk_bg};"
        f"border: 2px solid {risk_color};"
        "border-radius: 10px;"
        "padding: 0.75rem 1.25rem;"
        "margin-bottom: 0.25rem;"
        "display: flex;"
        "align-items: center;"
        'gap: 0.75rem;">'
        f'<span style="font-size:1.6rem;">{overall_icon}</span>'
        "<div>"
        f'<div style="font-weight:700; font-size:1rem; color:{risk_color};">'
        f"Riesgo {overall_risk}"
        "</div>"
        f'<div style="font-size:0.85rem; color:#555;">{overall_label}</div>'
        "</div>"
        "</div>"
        "</div>"
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # ── Metrics row ────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎞️ Fotogramas Analizados", total)
    col2.metric(
        "🔴 Fotogramas con Pólipo",
        class_counts["polyp"],
        delta=f"{polyp_pct:.0%} del total" if class_counts["polyp"] else None,
        delta_color="inverse",
    )
    col3.metric(
        "🟠 Con Inflamación",
        class_counts["inflammation"],
        delta=f"{inflammation_pct:.0%} del total"
        if class_counts["inflammation"]
        else None,
        delta_color="inverse",
    )
    col4.metric("🟢 Mucosa Normal", class_counts["normal"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Distribution + detail ──────────────────────────────────────────────
    col_chart, col_detail = st.columns([1, 1], gap="large")

    with col_chart:
        st.markdown(
            "<p style='font-weight:600; color:#2d5a4e; font-size:0.9rem;"
            "margin-bottom:0.5rem;'>📊 Distribución de Hallazgos</p>",
            unsafe_allow_html=True,
        )
        _render_summary_bar(
            "🔴 Pólipo", polyp_pct, "#D32F2F", class_counts["polyp"], total
        )
        _render_summary_bar(
            "🟠 Inflamación",
            inflammation_pct,
            "#F57C00",
            class_counts["inflammation"],
            total,
        )
        _render_summary_bar(
            "🟢 Mucosa Normal", normal_pct, "#388E3C", class_counts["normal"], total
        )

    with col_detail:
        st.markdown(
            "<p style='font-weight:600; color:#2d5a4e; font-size:0.9rem;"
            "margin-bottom:0.5rem;'>🔍 Detalle de Hallazgos</p>",
            unsafe_allow_html=True,
        )
        _render_detail_findings(
            class_counts,
            polyp_timestamps,
            inflammation_timestamps,
            normal_pct,
            max_polyp_score,
            max_inflammation_score,
        )

    # ── Clinical recommendation ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _render_clinical_recommendation(
        class_counts, polyp_timestamps, inflammation_timestamps, total
    )

    # ── Timeline strip ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _render_timeline(frame_results)


def _render_detail_findings(
    class_counts: dict,
    polyp_timestamps: list,
    inflammation_timestamps: list,
    normal_pct: float,
    max_polyp_score: float,
    max_inflammation_score: float,
) -> None:
    """Render the findings detail column using st.container blocks (no f-string HTML issues)."""

    if polyp_timestamps:
        ts_fmt = ", ".join(f"{t}s" for t in polyp_timestamps)
        st.markdown(
            "<div style='"
            "background:rgba(211,47,47,0.07);"
            "border-left:4px solid #D32F2F;"
            "border-radius:6px;"
            "padding:0.6rem 0.9rem;"
            "margin-bottom:0.5rem;"
            "font-size:0.84rem;"
            "'>"
            "<b style='color:#D32F2F;'>🔴 Pólipos detectados</b><br>"
            f"Instantes: <code>{ts_fmt}</code><br>"
            f"Confianza máxima: <b>{max_polyp_score:.1%}</b>"
            "</div>",
            unsafe_allow_html=True,
        )

    if inflammation_timestamps:
        ts_fmt = ", ".join(f"{t}s" for t in inflammation_timestamps)
        st.markdown(
            "<div style='"
            "background:rgba(245,124,0,0.07);"
            "border-left:4px solid #F57C00;"
            "border-radius:6px;"
            "padding:0.6rem 0.9rem;"
            "margin-bottom:0.5rem;"
            "font-size:0.84rem;"
            "'>"
            "<b style='color:#F57C00;'>🟠 Inflamación detectada</b><br>"
            f"Instantes: <code>{ts_fmt}</code><br>"
            f"Confianza máxima: <b>{max_inflammation_score:.1%}</b>"
            "</div>",
            unsafe_allow_html=True,
        )

    if class_counts["normal"] > 0:
        st.markdown(
            "<div style='"
            "background:rgba(56,142,60,0.07);"
            "border-left:4px solid #388E3C;"
            "border-radius:6px;"
            "padding:0.6rem 0.9rem;"
            "margin-bottom:0.5rem;"
            "font-size:0.84rem;"
            "'>"
            "<b style='color:#388E3C;'>🟢 Mucosa normal</b><br>"
            f"{class_counts['normal']} fotogramas sin hallazgos significativos "
            f"({normal_pct:.0%} del total)."
            "</div>",
            unsafe_allow_html=True,
        )

    if not any([polyp_timestamps, inflammation_timestamps, class_counts["normal"]]):
        st.markdown(
            "<p style='color:#6c757d; font-size:0.84rem;'>Sin hallazgos registrables.</p>",
            unsafe_allow_html=True,
        )


def _render_summary_bar(
    label: str,
    pct: float,
    color: str,
    count: int,
    total: int,
) -> None:
    """Render a labeled horizontal progress bar for the summary."""
    st.markdown(
        f"""
        <div style="margin-bottom: 0.6rem;">
            <div style="
                display: flex;
                justify-content: space-between;
                font-size: 0.83rem;
                margin-bottom: 3px;
            ">
                <span style="font-weight:600;">{label}</span>
                <span style="color:#6c757d;">{count}/{total} · {pct:.0%}</span>
            </div>
            <div style="
                background: #f0f0f0;
                border-radius: 6px;
                height: 20px;
                overflow: hidden;
            ">
                <div style="
                    width: {pct * 100:.1f}%;
                    background: {color};
                    height: 100%;
                    border-radius: 6px;
                    transition: width 0.4s ease;
                "></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_clinical_recommendation(
    class_counts: dict,
    polyp_ts: list,
    inflammation_ts: list,
    total: int,
) -> None:
    """Render clinical recommendation box based on findings."""
    if class_counts["polyp"] > 0:
        bg = "#FFEBEE"
        border = "#D32F2F"
        icon = "🔴"
        title = "Recomendación Clínica — URGENTE"
        lines = [
            f"Se han detectado <b>pólipos</b> en {class_counts['polyp']} "
            f"de {total} fotogramas analizados.",
            f"Instantes de detección: <b>{', '.join(f'{t}s' for t in polyp_ts)}</b>.",
            "Se recomienda <b>polipectomía</b> y revisión por especialista "
            "en el menor tiempo posible.",
        ]
    elif class_counts["inflammation"] > 0:
        bg = "#FFF3E0"
        border = "#F57C00"
        icon = "🟠"
        title = "Recomendación Clínica — SEGUIMIENTO"
        lines = [
            f"Se han observado <b>signos inflamatorios</b> en "
            f"{class_counts['inflammation']} fotogramas.",
            f"Instantes: <b>{', '.join(f'{t}s' for t in inflammation_ts)}</b>.",
            "Se recomienda evaluación de <b>tratamiento antiinflamatorio</b> "
            "y seguimiento estrecho.",
        ]
    else:
        bg = "#E8F5E9"
        border = "#388E3C"
        icon = "🟢"
        title = "Recomendación Clínica — RUTINA"
        lines = [
            f"Los {total} fotogramas analizados muestran <b>mucosa sana</b>.",
            "No se detectaron pólipos ni signos de inflamación significativos.",
            "Continuar con el <b>programa de cribado habitual</b>.",
        ]

    body = "".join(f"<li style='margin-bottom:4px;'>{line}</li>" for line in lines)
    st.markdown(
        f"""
        <div style="
            background: {bg};
            border: 2px solid {border};
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
        ">
            <div style="
                font-weight: 700;
                font-size: 0.95rem;
                color: {border};
                margin-bottom: 0.75rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            ">
                {icon} {title}
            </div>
            <ul style="
                margin: 0;
                padding-left: 1.25rem;
                font-size: 0.87rem;
                color: #333;
                line-height: 1.7;
            ">
                {body}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_timeline(frame_results: list[dict]) -> None:
    """Render a visual timeline strip of all analysed frames."""
    valid = [
        fr
        for fr in frame_results
        if not fr["error"] and fr["result"] and fr["result"].get("image_analysis")
    ]
    if not valid:
        return

    st.markdown(
        "<p style='font-weight:600; color:#2d5a4e; font-size:0.9rem;"
        "margin-bottom:0.75rem;'>⏱ Línea de Tiempo del Análisis</p>",
        unsafe_allow_html=True,
    )

    # ── Construimos cada nodo por separado para evitar problemas ──────────
    connector = (
        "<div style='"
        "flex:1; height:2px; background:#dee2e6;"
        "align-self:center; margin-bottom:26px; min-width:8px;"
        "'></div>"
    )

    items: list[str] = []
    for i, fr in enumerate(frame_results):
        ts = fr["timestamp"]

        if fr["error"] or not fr["result"] or not fr["result"].get("image_analysis"):
            color = "#9E9E9E"
            dot_bg = "rgba(158,158,158,0.13)"
            icon = "⚠️"
            conf = "Error"
        else:
            cls = fr["result"]["image_analysis"]["prediction_class"].lower()
            color = _COLOR_MAP.get(cls, "#388E3C")
            icon = _SUMMARY_ICONS.get(cls, "🟢")
            score = fr["result"]["image_analysis"]["prediction_score"]
            conf = f"{score:.0%}"
            # rgba equivalente al hex + "22" (~13% opacidad)
            rgba_map = {
                "#D32F2F": "rgba(211,47,47,0.13)",
                "#F57C00": "rgba(245,124,0,0.13)",
                "#388E3C": "rgba(56,142,60,0.13)",
            }
            dot_bg = rgba_map.get(color, "rgba(90,141,127,0.13)")

        node = (
            "<div style='display:flex; flex-direction:column;"
            "align-items:center; gap:4px; min-width:56px;'>"
            "<div style='"
            f"width:36px; height:36px; border-radius:50%;"
            f"background:{dot_bg}; border:2px solid {color};"
            "display:flex; align-items:center;"
            f"justify-content:center; font-size:1rem;'>{icon}</div>"
            f"<div style='font-size:0.72rem; color:#666; font-weight:600;'>{ts}s</div>"
            f"<div style='font-size:0.7rem; color:{color}; font-weight:700;'>{conf}</div>"
            "</div>"
        )
        items.append(node)
        if i < len(frame_results) - 1:
            items.append(connector)

    timeline_html = (
        "<div style='"
        "display:flex; align-items:flex-start; overflow-x:auto;"
        "padding:1rem 0.5rem; background:#f8f9fa;"
        "border-radius:12px; border:1px solid #dee2e6; gap:0;'>"
        + "".join(items)
        + "</div>"
    )
    st.markdown(timeline_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED RENDERERS  (image pipeline helpers — unchanged)
# ─────────────────────────────────────────────────────────────────────────────


def _render_image_grid(uploaded, img: dict) -> None:
    cards_html = image_grid_card(
        get_image_base64(uploaded), "Original", "Imagen de entrada"
    )

    has_gradcam = bool(img.get("gradcam_fusion") or img.get("gradcam_a"))

    if has_gradcam:
        alpha = img.get("alpha_used", 0)
        beta = img.get("beta_used", 0)

        if img.get("gradcam_a"):
            cards_html += image_grid_card(
                img["gradcam_a"],
                "Modelo A",
                f"Atención Contexto ({alpha:.0%})",
            )
        else:
            cards_html += _placeholder_card("Modelo A", "No disponible")

        if img.get("gradcam_b"):
            cards_html += image_grid_card(
                img["gradcam_b"],
                "Modelo B",
                f"Atención Tejido ({beta:.0%})",
            )
        else:
            cards_html += _placeholder_card("Modelo B", "No disponible")

        if img.get("gradcam_fusion"):
            cards_html += image_grid_card(
                img["gradcam_fusion"],
                "Fusión Ensemble",
                "Decisión Final",
                special=True,
            )
    else:
        cards_html += _onnx_notice_card()

    render_xai_grid(cards_html)


def _render_detail_expander(
    img: dict,
    cls: str,
    score: float,
    probs: dict,
) -> None:
    with st.expander("📊 Detalle de Confianza del Modelo", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Clase Predicha", cls.title())
        c2.metric("Confianza", f"{score:.1%}")
        c3.metric(
            "Ensemble Weights",
            f"α={img.get('alpha_used', 0):.2f} / β={img.get('beta_used', 0):.2f}",
        )

        st.markdown("---")

        if probs:
            st.markdown(
                "<p style='font-weight:600; margin-bottom:0.5rem;'>"
                "📈 Distribución de probabilidades</p>",
                unsafe_allow_html=True,
            )
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

            for class_name, prob in sorted_probs:
                is_predicted = class_name.lower() == cls.lower()
                bar_color = _COLOR_MAP.get(class_name.lower(), "#5a8d7f")
                star = " ⭐" if is_predicted else ""
                label = f"<b>{class_name.title()}</b>{star}"

                col_label, col_bar, col_pct = st.columns([2, 5, 1])
                with col_label:
                    st.markdown(
                        f"<p style='margin:0; padding-top:6px; font-size:0.9rem;'>"
                        f"{label}</p>",
                        unsafe_allow_html=True,
                    )
                with col_bar:
                    bar_width = f"{prob * 100:.1f}%"
                    opacity = "1.0" if is_predicted else "0.5"
                    st.markdown(
                        f"""
                        <div style="background:#f0f0f0; border-radius:6px;
                                    height:22px; margin-top:4px; overflow:hidden;">
                            <div style="width:{bar_width}; background:{bar_color};
                                        opacity:{opacity}; height:100%;
                                        border-radius:6px;"></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_pct:
                    st.markdown(
                        f"<p style='margin:0; padding-top:6px; font-size:0.9rem; "
                        f"font-weight:{'700' if is_predicted else '400'};'>"
                        f"{prob:.1%}</p>",
                        unsafe_allow_html=True,
                    )
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        st.markdown("---")

        backend = img.get("ensemble_mode", "unknown")
        has_gradcam = bool(img.get("gradcam_fusion") or img.get("gradcam_a"))

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown(
                f"""
                <div style="background:#f8f9fa; border-radius:8px;
                            padding:0.75rem 1rem; font-size:0.85rem;">
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
                <div style="background:#f8f9fa; border-radius:8px;
                            padding:0.75rem 1rem; font-size:0.85rem;">
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


# ─────────────────────────────────────────────────────────────────────────────
#  HTML HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _loader_html(primary: str, secondary: str) -> str:
    return f"""
    <div class="loader-container">
        <div class="loader-spinner">&#8203;</div>
        <div class="loader-text-primary">{primary}</div>
        <div class="loader-text-secondary">{secondary}</div>
    </div>
    """


def _placeholder_card(title: str, subtitle: str) -> str:
    return f"""
    <div style="
        background:#f8f9fa; border:2px dashed #dee2e6; border-radius:12px;
        padding:2rem; text-align:center; color:#6c757d;
        display:flex; flex-direction:column; align-items:center;
        justify-content:center; min-height:200px;
    ">
        <div style="font-size:2rem; margin-bottom:0.5rem;">🔲</div>
        <div style="font-weight:600; font-size:0.9rem;">{title}</div>
        <div style="font-size:0.8rem; margin-top:0.25rem;">{subtitle}</div>
    </div>
    """


def _onnx_notice_card() -> str:
    return """
    <div style="
        grid-column:span 3;
        background:linear-gradient(135deg,#e8f4f8 0%,#d1ecf1 100%);
        border:1px solid #bee5eb; border-radius:12px;
        padding:1.5rem 2rem; text-align:center; color:#0c5460;
    ">
        <div style="font-size:1.5rem; margin-bottom:0.5rem;">⚡</div>
        <div style="font-weight:600; font-size:0.95rem; margin-bottom:0.4rem;">
            Backend ONNX Runtime activo
        </div>
        <div style="font-size:0.85rem; opacity:0.85;">
            La inferencia ONNX no genera mapas Grad-CAM (requiere backend PyTorch).
            La clasificación y segmentación son equivalentes.
        </div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS — file-like wrapper for extracted frames
# ─────────────────────────────────────────────────────────────────────────────


class _FrameFile:
    """
    Minimal file-like object that mimics a Streamlit UploadedFile
    so es compatible con ``upload_colonoscopy_image``.
    """

    def __init__(self, data: bytes, name: str) -> None:
        self._data = data
        self._buf = io.BytesIO(data)
        self.name = name
        self.type = "image/jpeg"

    def read(self, size: int = -1) -> bytes:
        return self._buf.read(size)

    def seek(self, pos: int) -> int:
        return self._buf.seek(pos)

    def tell(self) -> int:
        return self._buf.tell()

    def getvalue(self) -> bytes:
        return self._data


# ─────────────────────────────────────────────────────────────────────────────
render()
