"""
Endoscopic AI Analysis · Endo-AID

Deep learning classification of colonic mucosa (polyp / inflammation / normal),
Grad-CAM explainability maps, and U-Net polyp segmentation.
Supports single-image and video batch inference modes.
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
    page_title="Endoscopic AI Analysis · Endo-AID",
    page_icon="🌿",
    layout="wide",
)
apply_custom_css()
render_sidebar()

_CLASS_MAP = {
    "polyp": ("Polyp Detected", "red"),
    "inflammation": ("Inflammation / Colitis", "orange"),
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


def render() -> None:
    """
    Main entry point for the Endoscopic AI Analysis page.

    Renders the patient selector and delegates to the image / video
    tab renderer.
    """
    page_header(
        "Endoscopic AI Analysis",
        "Colonic mucosa classification, Grad-CAM explainability, "
        "and U-Net polyp segmentation.",
        icon="🔬",
    )

    patients = get_patients()
    if not patients:
        show_empty_state(
            "🔬",
            "No patients registered",
            "Register patients before running visual inference.",
        )
        return

    opts = build_patient_options(patients)
    col_sel, _ = st.columns([1, 2])
    with col_sel:
        pid = opts[st.selectbox("Patient", list(opts.keys()))]

    st.markdown("<br>", unsafe_allow_html=True)
    _render_mode_tabs(pid)


def _render_mode_tabs(pid: int) -> None:
    """
    Render the Image / Video inference tab selector.

    Args:
        pid: Selected patient ID passed down to each inference pipeline.
    """
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

    tab_image, tab_video = st.tabs(["🖼️  Endoscopic Image", "🎥  Endoscopic Video"])

    with tab_image:
        _render_image_mode(pid)

    with tab_video:
        _render_video_mode(pid)


def _render_image_mode(pid: int) -> None:
    """
    Render the single-frame upload and inference panel.

    Args:
        pid: Selected patient ID forwarded to the upload and diagnosis endpoints.
    """
    st.markdown("<br>", unsafe_allow_html=True)
    col_up, col_preview = st.columns([2, 1], gap="large")

    with col_up:
        uploaded = st.file_uploader(
            "Upload Endoscopic Frame",
            type=["jpg", "png", "jpeg"],
            help="Colonoscopy image in JPG or PNG format.",
            key="img_uploader",
        )

    with col_preview:
        if uploaded:
            st.markdown(
                "<p style='font-weight:600; color:#5a8d7f; font-size:0.85rem;'>"
                "Preview</p>",
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
        if st.button("🧠 Process Frame with AI", type="primary", key="btn_img"):
            _run_single_pipeline(pid, uploaded)


def _render_video_mode(pid: int) -> None:
    """
    Render the video upload, frame extraction configuration, and inference panel.

    Args:
        pid: Selected patient ID forwarded to the upload and diagnosis endpoints.
    """
    st.markdown("<br>", unsafe_allow_html=True)

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
            <b>🎥 Video Mode</b><br>
            Upload a video up to <b>{MAX_VIDEO_DURATION_SECONDS} seconds</b>.
            The system will automatically extract one frame every
            <b>{FRAME_INTERVAL_SECONDS} seconds</b>, analyse each with the AI
            models, and generate a <b>global clinical summary</b> of all findings.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_up, col_cfg = st.columns([2, 1], gap="large")

    with col_up:
        video_file = st.file_uploader(
            "Upload Endoscopic Video",
            type=["mp4", "avi", "mov", "mkv"],
            help=f"Colonoscopy video. Maximum {MAX_VIDEO_DURATION_SECONDS}s.",
            key="video_uploader",
        )

    with col_cfg:
        st.markdown(
            "<p style='font-weight:600; color:#5a8d7f; font-size:0.85rem; "
            "margin-bottom:0.25rem;'>Extraction Configuration</p>",
            unsafe_allow_html=True,
        )
        interval = st.slider(
            "Frame interval (s)",
            min_value=1,
            max_value=10,
            value=FRAME_INTERVAL_SECONDS,
            step=1,
            key="frame_interval",
            help="How many seconds between extracted frames.",
        )
        st.markdown(
            "<p style='font-size:0.8rem; color:#6c757d; margin-top:0.25rem;'>"
            f"⏱ With {interval}s interval and {MAX_VIDEO_DURATION_SECONDS}s "
            f"video → max. "
            f"<b>{MAX_VIDEO_DURATION_SECONDS // interval + 1} frames</b>."
            "</p>",
            unsafe_allow_html=True,
        )

    if video_file:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("▶️  Preview Video", expanded=False):
            st.video(video_file)
            video_file.seek(0)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧠 Analyse Video with AI", type="primary", key="btn_video"):
            _run_video_pipeline(pid, video_file, interval)


def _run_single_pipeline(pid: int, uploaded) -> None:
    """
    Execute the full AI inference pipeline for a single endoscopic frame.

    Uploads the image, calls the diagnosis endpoint, and renders the
    classification result, Grad-CAM maps, segmentation report, and
    confidence detail expander.

    Args:
        pid: Selected patient ID.
        uploaded: Streamlit UploadedFile object for the colonoscopy image.
    """
    placeholder = st.empty()
    placeholder.markdown(
        _loader_html(
            "Computing Neural Models…",
            "Generating Grad-CAMs and U-Net Segmentation",
        ),
        unsafe_allow_html=True,
    )

    upload_res = upload_colonoscopy_image(pid, uploaded)
    if not upload_res:
        placeholder.empty()
        st.error("❌ Error uploading the image. Please check the connection.")
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
        st.error("❌ No response received from the model.")
        return

    img = result["image_analysis"]
    cls = img["prediction_class"]
    score = img["prediction_score"]
    probs = img.get("probabilities", {})

    label, level = _CLASS_MAP.get(cls, ("Healthy Mucosa", "green"))
    show_result_banner(f"Diagnosis: {label}", f"{score:.1%}", level)

    _render_image_grid(uploaded, img)

    if img.get("report_path"):
        render_segmentation(img["report_path"])

    _render_detail_expander(img, cls, score, probs)


def _run_video_pipeline(pid: int, video_file, interval: int) -> None:
    """
    Execute the full video inference pipeline.

    Steps:
        1. Extract frames at the configured interval.
        2. Upload and infer each frame via the backend.
        3. Display per-frame expandable results.
        4. Render a global clinical summary with timeline.

    Args:
        pid: Selected patient ID.
        video_file: Streamlit UploadedFile or seekable file-like object.
        interval: Frame extraction interval in seconds.
    """
    extract_placeholder = st.empty()
    extract_placeholder.markdown(
        _loader_html(
            "Extracting Video Frames…",
            f"One frame every {interval}s",
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
        st.error(f"❌ Error processing video: {e}")
        return

    extract_placeholder.empty()

    if not frames:
        st.warning("⚠️ No frames could be extracted from the video.")
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
            ✅ <b>{len(frames)} frames extracted</b> successfully.
            Starting AI analysis…
        </div>
        """,
        unsafe_allow_html=True,
    )

    frame_results: list[dict] = []
    progress_bar = st.progress(0, text="Analysing frames…")
    status_text = st.empty()

    for i, frame in enumerate(frames):
        status_text.markdown(
            f"<p style='color:#5a8d7f; font-size:0.85rem;'>"
            f"🔬 Analysing frame {i + 1}/{len(frames)} "
            f"(t={frame['timestamp']}s)…</p>",
            unsafe_allow_html=True,
        )

        buf = io.BytesIO()
        frame["image"].save(buf, format="JPEG", quality=92)
        frame_bytes = buf.getvalue()

        frame_file = _FrameFile(data=frame_bytes, name=frame["filename"])

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

    st.markdown("<br>", unsafe_allow_html=True)
    _render_video_frame_results(frame_results)

    st.markdown("<br>", unsafe_allow_html=True)
    _render_video_summary(frame_results)


def _render_video_frame_results(frame_results: list[dict]) -> None:
    """
    Render an expandable card for each analysed video frame.

    Each card shows the original frame, the classification badge,
    probability bars, and Grad-CAM / segmentation outputs when available.
    Frames that failed inference display an error notice.

    Args:
        frame_results: List of dicts with keys ``timestamp``, ``pil_image``,
            ``error``, and ``result``.
    """
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
            🎞️ Per-Frame Results
        </div>
        """,
        unsafe_allow_html=True,
    )

    for fr in frame_results:
        ts = fr["timestamp"]
        pil_img = fr["pil_image"]
        img_b64 = get_pil_image_base64(pil_img)

        if fr["error"] or not fr["result"]:
            with st.expander(f"⏱ t={ts}s — ⚠️ Analysis error", expanded=False):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.markdown(
                        f'<div class="preview-card">'
                        f'<img src="{img_b64}" alt="Frame {ts}s">'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with col_b:
                    st.error("Could not obtain a model response for this frame.")
            continue

        img_analysis = fr["result"]["image_analysis"]
        cls = img_analysis["prediction_class"]
        score = img_analysis["prediction_score"]
        probs = img_analysis.get("probabilities", {})
        label, _ = _CLASS_MAP.get(cls, ("Healthy Mucosa", "green"))
        color = _COLOR_MAP.get(cls, "#388E3C")
        icon = _SUMMARY_ICONS.get(cls, "🟢")

        expander_label = f"{icon}  t={ts}s — {label}  ({score:.1%})"

        with st.expander(expander_label, expanded=False):
            col_img, col_info = st.columns([1, 2], gap="large")

            with col_img:
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
                                Confidence: <b>{score:.1%}</b>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

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

            has_gradcam = bool(
                img_analysis.get("gradcam_fusion") or img_analysis.get("gradcam_a")
            )

            if has_gradcam:
                st.markdown(
                    "<p style='font-weight:600; color:#5a8d7f; "
                    "font-size:0.82rem; margin-top:0.75rem;'>"
                    "Activation Maps (Grad-CAM)</p>",
                    unsafe_allow_html=True,
                )
                cards_html = image_grid_card(img_b64, "Original", "Frame")

                alpha = img_analysis.get("alpha_used", 0)
                beta = img_analysis.get("beta_used", 0)

                if img_analysis.get("gradcam_a"):
                    cards_html += image_grid_card(
                        img_analysis["gradcam_a"],
                        "Model A",
                        f"Context Attention ({alpha:.0%})",
                    )
                if img_analysis.get("gradcam_b"):
                    cards_html += image_grid_card(
                        img_analysis["gradcam_b"],
                        "Model B",
                        f"Tissue Attention ({beta:.0%})",
                    )
                if img_analysis.get("gradcam_fusion"):
                    cards_html += image_grid_card(
                        img_analysis["gradcam_fusion"],
                        "Ensemble Fusion",
                        "Final Decision",
                        special=True,
                    )
                render_xai_grid(cards_html)

            if img_analysis.get("report_path"):
                render_segmentation(img_analysis["report_path"])


def _render_video_summary(frame_results: list[dict]) -> None:
    """
    Compute and render the global clinical summary for all analysed frames.

    Aggregates per-frame classifications, computes polyp and inflammation
    timestamps, determines the overall risk level, and renders metrics,
    distribution bars, findings detail, clinical recommendation, and timeline.

    Args:
        frame_results: List of per-frame result dicts from the inference loop.
    """
    valid = [
        fr
        for fr in frame_results
        if not fr["error"] and fr["result"] and fr["result"].get("image_analysis")
    ]

    if not valid:
        st.warning("⚠️ No valid frames available to generate the summary.")
        return

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

    if class_counts["polyp"] > 0:
        overall_risk = "HIGH"
        risk_color = "#D32F2F"
        overall_icon = "🔴"
        overall_label = "Polyp detected — Urgent attention required"
        risk_bg = "rgba(211,47,47,0.08)"
    elif class_counts["inflammation"] > 0:
        overall_risk = "MODERATE"
        risk_color = "#F57C00"
        overall_icon = "🟠"
        overall_label = "Inflammatory signs — Follow-up recommended"
        risk_bg = "rgba(245,124,0,0.08)"
    else:
        overall_risk = "LOW"
        risk_color = "#388E3C"
        overall_icon = "🟢"
        overall_label = "Normal mucosa — No significant findings"
        risk_bg = "rgba(56,142,60,0.08)"

    header_html = (
        '<div style="'
        "background: linear-gradient(135deg, #f8fffe 0%, #edf7f4 100%);"
        "border: 2px solid #5a8d7f;"
        "border-radius: 16px;"
        "padding: 1.75rem 2rem;"
        "margin-bottom: 1.5rem;"
        'box-shadow: 0 4px 20px rgba(90,141,127,0.12);">'
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
        "Global Clinical Video Summary"
        "</div>"
        '<div style="font-size:0.82rem; color:#6c757d;">'
        f"{total} frames analysed · Average confidence: <b>{avg_confidence:.1%}</b>"
        "</div>"
        "</div>"
        "</div>"
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
        f"{overall_risk} Risk"
        "</div>"
        f'<div style="font-size:0.85rem; color:#555;">{overall_label}</div>'
        "</div>"
        "</div>"
        "</div>"
    )
    st.markdown(header_html, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎞️ Frames Analysed", total)
    col2.metric(
        "🔴 Frames with Polyp",
        class_counts["polyp"],
        delta=f"{polyp_pct:.0%} of total" if class_counts["polyp"] else None,
        delta_color="inverse",
    )
    col3.metric(
        "🟠 With Inflammation",
        class_counts["inflammation"],
        delta=f"{inflammation_pct:.0%} of total"
        if class_counts["inflammation"]
        else None,
        delta_color="inverse",
    )
    col4.metric("🟢 Normal Mucosa", class_counts["normal"])

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart, col_detail = st.columns([1, 1], gap="large")

    with col_chart:
        st.markdown(
            "<p style='font-weight:600; color:#2d5a4e; font-size:0.9rem;"
            "margin-bottom:0.5rem;'>📊 Findings Distribution</p>",
            unsafe_allow_html=True,
        )
        _render_summary_bar(
            "🔴 Polyp", polyp_pct, "#D32F2F", class_counts["polyp"], total
        )
        _render_summary_bar(
            "🟠 Inflammation",
            inflammation_pct,
            "#F57C00",
            class_counts["inflammation"],
            total,
        )
        _render_summary_bar(
            "🟢 Normal Mucosa", normal_pct, "#388E3C", class_counts["normal"], total
        )

    with col_detail:
        st.markdown(
            "<p style='font-weight:600; color:#2d5a4e; font-size:0.9rem;"
            "margin-bottom:0.5rem;'>🔍 Findings Detail</p>",
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

    st.markdown("<br>", unsafe_allow_html=True)
    _render_clinical_recommendation(
        class_counts, polyp_timestamps, inflammation_timestamps, total
    )

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
    """
    Render the findings detail column with colour-coded summary cards.

    Args:
        class_counts: Dict mapping ``"polyp"``, ``"inflammation"``, ``"normal"``
            to their respective frame counts.
        polyp_timestamps: List of timestamps (seconds) where polyps were detected.
        inflammation_timestamps: List of timestamps where inflammation was detected.
        normal_pct: Fraction of frames classified as normal mucosa.
        max_polyp_score: Highest polyp confidence score across all frames.
        max_inflammation_score: Highest inflammation confidence score across all frames.
    """
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
            "<b style='color:#D32F2F;'>🔴 Polyps detected</b><br>"
            f"Timestamps: <code>{ts_fmt}</code><br>"
            f"Maximum confidence: <b>{max_polyp_score:.1%}</b>"
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
            "<b style='color:#F57C00;'>🟠 Inflammation detected</b><br>"
            f"Timestamps: <code>{ts_fmt}</code><br>"
            f"Maximum confidence: <b>{max_inflammation_score:.1%}</b>"
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
            "<b style='color:#388E3C;'>🟢 Normal mucosa</b><br>"
            f"{class_counts['normal']} frames with no significant findings "
            f"({normal_pct:.0%} of total)."
            "</div>",
            unsafe_allow_html=True,
        )

    if not any([polyp_timestamps, inflammation_timestamps, class_counts["normal"]]):
        st.markdown(
            "<p style='color:#6c757d; font-size:0.84rem;'>No recordable findings.</p>",
            unsafe_allow_html=True,
        )


def _render_summary_bar(
    label: str,
    pct: float,
    color: str,
    count: int,
    total: int,
) -> None:
    """
    Render a labelled horizontal progress bar for the video summary section.

    Args:
        label: Display label shown above the bar (e.g. ``"🔴 Polyp"``).
        pct: Fraction in [0, 1] that determines the bar fill width.
        color: Hex colour string for the filled portion of the bar.
        count: Absolute count used in the right-side label.
        total: Total frame count used in the right-side label.
    """
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
    """
    Render a colour-coded clinical recommendation box based on video findings.

    Priority: polyp > inflammation > normal.

    Args:
        class_counts: Dict of classification counts per class.
        polyp_ts: Timestamps (seconds) where polyps were detected.
        inflammation_ts: Timestamps where inflammation was detected.
        total: Total number of analysed frames.
    """
    if class_counts["polyp"] > 0:
        bg, border, icon = "#FFEBEE", "#D32F2F", "🔴"
        title = "Clinical Recommendation — URGENT"
        lines = [
            f"<b>Polyps</b> detected in {class_counts['polyp']} "
            f"of {total} analysed frames.",
            f"Detection timestamps: <b>{', '.join(f'{t}s' for t in polyp_ts)}</b>.",
            "<b>Polypectomy</b> and specialist review are recommended "
            "as soon as possible.",
        ]
    elif class_counts["inflammation"] > 0:
        bg, border, icon = "#FFF3E0", "#F57C00", "🟠"
        title = "Clinical Recommendation — FOLLOW-UP"
        lines = [
            f"<b>Inflammatory signs</b> observed in "
            f"{class_counts['inflammation']} frames.",
            f"Timestamps: <b>{', '.join(f'{t}s' for t in inflammation_ts)}</b>.",
            "Evaluation of <b>anti-inflammatory treatment</b> and "
            "close follow-up are recommended.",
        ]
    else:
        bg, border, icon = "#E8F5E9", "#388E3C", "🟢"
        title = "Clinical Recommendation — ROUTINE"
        lines = [
            f"All {total} analysed frames show <b>healthy mucosa</b>.",
            "No polyps or significant inflammatory signs were detected.",
            "Continue with the <b>standard screening programme</b>.",
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
    """
    Render a horizontal scrollable visual timeline of all analysed frames.

    Each frame is represented by a coloured dot with the timestamp and
    confidence score beneath it. Failed frames show a grey warning dot.
    Dots are connected by thin horizontal lines.

    Args:
        frame_results: Full list of per-frame result dicts, including
            failed frames.
    """
    valid = [
        fr
        for fr in frame_results
        if not fr["error"] and fr["result"] and fr["result"].get("image_analysis")
    ]
    if not valid:
        return

    st.markdown(
        "<p style='font-weight:600; color:#2d5a4e; font-size:0.9rem;"
        "margin-bottom:0.75rem;'>⏱ Analysis Timeline</p>",
        unsafe_allow_html=True,
    )

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


def _render_image_grid(uploaded, img: dict) -> None:
    """
    Render the Grad-CAM image grid for a single-frame inference result.

    Shows the original image alongside Model A, Model B, and fusion
    Grad-CAM cards. Displays an ONNX notice card when Grad-CAM is
    unavailable (ONNX backend active).

    Args:
        uploaded: Streamlit UploadedFile for the original image.
        img: Image analysis result dict from the diagnosis endpoint.
    """
    cards_html = image_grid_card(get_image_base64(uploaded), "Original", "Input Image")

    has_gradcam = bool(img.get("gradcam_fusion") or img.get("gradcam_a"))

    if has_gradcam:
        alpha = img.get("alpha_used", 0)
        beta = img.get("beta_used", 0)

        if img.get("gradcam_a"):
            cards_html += image_grid_card(
                img["gradcam_a"],
                "Model A",
                f"Context Attention ({alpha:.0%})",
            )
        else:
            cards_html += _placeholder_card("Model A", "Not available")

        if img.get("gradcam_b"):
            cards_html += image_grid_card(
                img["gradcam_b"],
                "Model B",
                f"Tissue Attention ({beta:.0%})",
            )
        else:
            cards_html += _placeholder_card("Model B", "Not available")

        if img.get("gradcam_fusion"):
            cards_html += image_grid_card(
                img["gradcam_fusion"],
                "Ensemble Fusion",
                "Final Decision",
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
    """
    Render the collapsible model confidence detail expander.

    Displays predicted class, confidence, ensemble weights, probability
    distribution bars, and backend / Grad-CAM availability info.

    Args:
        img: Full image analysis dict from the diagnosis endpoint.
        cls: Predicted class string (e.g. ``"polyp"``).
        score: Prediction confidence in [0, 1].
        probs: Dict mapping class names to their probabilities.
    """
    with st.expander("📊 Model Confidence Detail", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Class", cls.title())
        c2.metric("Confidence", f"{score:.1%}")
        c3.metric(
            "Ensemble Weights",
            f"α={img.get('alpha_used', 0):.2f} / β={img.get('beta_used', 0):.2f}",
        )

        st.markdown("---")

        if probs:
            st.markdown(
                "<p style='font-weight:600; margin-bottom:0.5rem;'>"
                "📈 Probability distribution</p>",
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
                    <b>Ensemble:</b> {"✅ Active" if img.get("ensemble_used") else "❌ Model A only"}<br>
                    <b>Grad-CAM:</b> {"✅ Available" if has_gradcam else "⚠️ Not available (ONNX)"}
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
                    <b>α (Model A):</b> {img.get("alpha_used", 0):.3f}<br>
                    <b>β (Model B):</b> {img.get("beta_used", 0):.3f}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.info(
            "ℹ️ Ensemble weights are dynamically adapted based on each "
            "model's entropy for every individual image."
        )


def _loader_html(primary: str, secondary: str) -> str:
    """
    Generate the HTML markup for the animated loading overlay.

    Args:
        primary: Main loading message displayed in large text.
        secondary: Subtitle displayed beneath the primary message.

    Returns:
        HTML string containing the loader container markup.
    """
    return f"""
    <div class="loader-container">
        <div class="loader-spinner">&#8203;</div>
        <div class="loader-text-primary">{primary}</div>
        <div class="loader-text-secondary">{secondary}</div>
    </div>
    """


def _placeholder_card(title: str, subtitle: str) -> str:
    """
    Generate an HTML placeholder card for unavailable Grad-CAM outputs.

    Args:
        title: Card title displayed in bold (e.g. ``"Model A"``).
        subtitle: Secondary line displayed below the title.

    Returns:
        HTML string for a dashed-border placeholder card.
    """
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
    """
    Generate an HTML notice card indicating that ONNX Runtime is active.

    Informs the user that Grad-CAM maps are unavailable when the ONNX
    backend is used, while classification and segmentation remain equivalent.

    Returns:
        HTML string for the full-width ONNX notice card.
    """
    return """
    <div style="
        grid-column:span 3;
        background:linear-gradient(135deg,#e8f4f8 0%,#d1ecf1 100%);
        border:1px solid #bee5eb; border-radius:12px;
        padding:1.5rem 2rem; text-align:center; color:#0c5460;
    ">
        <div style="font-size:1.5rem; margin-bottom:0.5rem;">⚡</div>
        <div style="font-weight:600; font-size:0.95rem; margin-bottom:0.4rem;">
            ONNX Runtime backend active
        </div>
        <div style="font-size:0.85rem; opacity:0.85;">
            ONNX inference does not generate Grad-CAM maps (requires PyTorch backend).
            Classification and segmentation are equivalent.
        </div>
    </div>
    """


class _FrameFile:
    """
    Minimal file-like wrapper that mimics a Streamlit UploadedFile.

    Used to pass extracted video frames to ``upload_colonoscopy_image``
    without re-wrapping them as real uploaded files.

    Attributes:
        name: Filename string used by the upload endpoint.
        type: MIME type string (always ``"image/jpeg"``).
    """

    def __init__(self, data: bytes, name: str) -> None:
        """
        Initialise the frame file wrapper.

        Args:
            data: Raw JPEG bytes of the extracted frame.
            name: Filename to report to the upload endpoint.
        """
        self._data = data
        self._buf = io.BytesIO(data)
        self.name = name
        self.type = "image/jpeg"

    def read(self, size: int = -1) -> bytes:
        """
        Read up to ``size`` bytes from the internal buffer.

        Args:
            size: Maximum number of bytes to read. ``-1`` reads all.

        Returns:
            Bytes read from the buffer.
        """
        return self._buf.read(size)

    def seek(self, pos: int) -> int:
        """
        Seek to the given position in the internal buffer.

        Args:
            pos: Byte offset from the start of the buffer.

        Returns:
            New absolute position in the buffer.
        """
        return self._buf.seek(pos)

    def tell(self) -> int:
        """
        Return the current position of the internal buffer pointer.

        Returns:
            Current byte offset from the start of the buffer.
        """
        return self._buf.tell()

    def getvalue(self) -> bytes:
        """
        Return the full raw byte content regardless of the buffer position.

        Returns:
            Complete JPEG byte string of the frame.
        """
        return self._data


render()
