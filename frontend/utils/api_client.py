"""
Centralized API client for backend communication.
Handles all HTTP requests with error handling.
"""

from typing import Any

import requests  # type: ignore
import streamlit as st

API_BASE_URL = "http://localhost:8000"


def _handle_response(response: requests.Response) -> Any | None:
    """Parse response or return None on failure."""
    try:
        response.raise_for_status()
        return response.json()
    except (requests.HTTPError, requests.JSONDecodeError):
        return None


def get_patients() -> list[dict]:
    """Fetches all patients from the API."""
    try:
        resp = requests.get(f"{API_BASE_URL}/patients/", timeout=10)
        return _handle_response(resp) or []
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to the backend. Is the server running?")
        return []


def get_patient_history(patient_id: int) -> list[dict]:
    """Fetches diagnosis history for a patient."""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/diagnosis/history/{patient_id}", timeout=10
        )
        return _handle_response(resp) or []
    except requests.ConnectionError:
        return []


def upload_colonoscopy_image(patient_id: int, file) -> dict | None:
    """
    Uploads an endoscopy image (or extracted video frame) for a patient.

    Compatible with:
        - Streamlit UploadedFile (has .getvalue() and .name)
        - _FrameFile wrapper (has .getvalue() and .name)
    """
    try:
        if hasattr(file, "getvalue"):
            file_bytes = file.getvalue()
        elif hasattr(file, "read"):
            file_bytes = file.read()
            if hasattr(file, "seek"):
                file.seek(0)
        else:
            return None

        filename = getattr(file, "name", "frame.jpg")
        mime = getattr(file, "type", "image/jpeg")

        files = {"file": (filename, file_bytes, mime)}

        resp = requests.post(
            f"{API_BASE_URL}/uploads/colonoscopy/{patient_id}",
            files=files,
            timeout=30,
        )
        return _handle_response(resp)
    except requests.ConnectionError:
        return None


def run_diagnosis(payload: dict) -> dict | None:
    """Runs the AI diagnosis pipeline."""
    try:
        resp = requests.post(f"{API_BASE_URL}/diagnosis/run", json=payload, timeout=60)
        return _handle_response(resp)
    except requests.ConnectionError:
        return None


def create_patient(patient_data: dict) -> dict | None:
    """Creates a new patient via API."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/patients/",
            json=patient_data,
            timeout=15,
        )
        if resp.status_code in (200, 201):
            return resp.json()
        st.error(f"❌ Error del servidor: {resp.text}")
        return None
    except requests.ConnectionError:
        st.error("⚠️ No se pudo conectar con el servidor.")
        return None


def run_smoking_triage(payload: dict) -> dict | None:
    """Runs the smoking habit triage model via reverse logic."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/diagnosis/smoking-triage",
            json=payload,
            timeout=30,
        )
        return _handle_response(resp)
    except requests.ConnectionError:
        st.error("⚠️ No se pudo conectar con el servidor.")
        return None


def ensure_db_initialized() -> bool:
    """
    Calls the backend /health/init endpoint to guarantee that all
    database tables exist before the app starts serving requests.

    This is idempotent — safe to call on every app load.
    Tables are only created if they do not already exist.

    Returns:
        bool: True if initialization succeeded, False otherwise.
    """
    try:
        resp = requests.get(f"{API_BASE_URL}/health/init", timeout=10)
        resp.raise_for_status()
        data = resp.json()

        created = data.get("tables_created", [])
        if created:
            st.toast(
                f"✅ Database initialized — tables created: {', '.join(created)}",
                icon="🗄️",
            )
        return True

    except requests.ConnectionError:
        st.error(
            "⚠️ Cannot connect to the backend. "
            "Please ensure the server is running on http://localhost:8000"
        )
        return False
    except requests.HTTPError as e:
        st.error(f"❌ Backend initialization failed: {e}")
        return False
