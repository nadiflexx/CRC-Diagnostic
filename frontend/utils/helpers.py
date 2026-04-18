"""
General helper utilities.
"""

import base64
from pathlib import Path

import streamlit as st


def apply_custom_css() -> None:
    """Injects the global stylesheet into the Streamlit app."""
    css_path = Path(__file__).parent.parent / "assets" / "styles.css"
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")
        # Force cache busting with timestamp
        cache_buster = int(css_path.stat().st_mtime * 1000)
        st.markdown(
            f"<style>/* v{cache_buster} */{css_content}</style>",
            unsafe_allow_html=True,
        )


def get_image_base64(uploaded_file) -> str:
    """Converts a Streamlit UploadedFile to a base64 data URI."""
    b64 = base64.b64encode(uploaded_file.getvalue()).decode()
    # Detect MIME type based on file extension
    mime_type = "image/jpeg"
    if uploaded_file.name.lower().endswith((".png", ".png")):
        mime_type = "image/png"
    return f"data:{mime_type};base64,{b64}"


def build_patient_options(patients: list[dict]) -> dict[str, int]:
    """
    Builds a display-label → ID mapping for patient selectboxes.

    Returns:
        Dict mapping "FirstName LastName (ID: X)" → patient_id
    """
    return {
        f"{p['first_name']} {p['last_name']}  ·  ID {p['id']}": p["id"]
        for p in patients
    }


def get_pil_image_base64(pil_img) -> str:
    """
    Convert a PIL Image to a base64 data URI for inline HTML display.

    Args:
        pil_img: PIL Image object.

    Returns:
        str: data:image/jpeg;base64,... URI.
    """
    import base64
    import io

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"
