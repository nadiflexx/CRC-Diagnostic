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
    """Uploads an endoscopy image for a patient."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
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
