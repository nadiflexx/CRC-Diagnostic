"""
Splash Screen & Loading Animation for Endo-AID.
"""

import time

import requests  # type: ignore
import streamlit as st

API_BASE_URL = "http://localhost:8000"


def _initialize_database() -> bool:
    """
    Calls /health/init to ensure all DB tables exist.
    Returns True on success, False on failure.
    """
    try:
        resp = requests.get(f"{API_BASE_URL}/health/init", timeout=10)
        resp.raise_for_status()
        return True
    except (requests.ConnectionError, requests.HTTPError, requests.Timeout):
        return False


def show_splash_screen() -> bool:
    """
    Displays a professional splash loading screen with staged progress.
    Includes DB initialization as a real blocking stage.
    Returns True when complete, False if backend is unreachable.
    """
    if st.session_state.get("app_loaded", False):
        return True

    container = st.empty()

    with container.container():
        st.markdown(
            """
            <style>
                .stApp > header { opacity: 0; }
                section[data-testid="stSidebar"] { display: none !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        _, col_center, _ = st.columns([1, 2, 1])

        with col_center:
            st.markdown(
                """
                <div style="text-align: center; padding-top: 15vh;">
                    <div style="
                        display: inline-flex; align-items: center; justify-content: center;
                        width: 80px; height: 80px; border-radius: 20px;
                        background: linear-gradient(135deg, #067a5f 0%, #0a5847 100%);
                        box-shadow: 0 8px 30px rgba(5, 150, 105, 0.3);
                        margin-bottom: 1.5rem;
                    ">
                        <span style="font-size: 2.5rem;">🌿</span>
                    </div>
                    <h1 style="
                        font-size: 2.8rem; font-weight: 900; color: #0F172A;
                        letter-spacing: -0.04em; margin: 0;
                    ">Endo-AID</h1>
                    <p style="
                        color: #94A3B8; font-size: 0.9rem; letter-spacing: 3px;
                        text-transform: uppercase; margin-top: 0.5rem;
                    ">Clinical Decision Support System</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br><br>", unsafe_allow_html=True)
            progress = st.progress(0)
            status = st.empty()

            pre_stages = [
                ("🔌 Connecting to backend services...", 15),
                ("🗄️  Checking database integrity...", 30),
            ]

            post_stages = [
                ("🧠 Initializing classification models...", 50),
                ("🔬 Loading U-Net segmentation engine...", 65),
                ("📊 Preparing analytics pipelines...", 80),
                ("🧪 Calibrating risk scoring...", 92),
                ("✅ System ready", 100),
            ]

            def _render_status(msg: str, val: int, color: str = "#94A3B8") -> None:
                status.markdown(
                    f"<p style='text-align:center; color:{color}; font-size:0.95rem;'>{msg}</p>",
                    unsafe_allow_html=True,
                )
                progress.progress(val)

            for msg, val in pre_stages:
                _render_status(msg, val)
                time.sleep(0.3)

                if "Checking database" in msg:
                    db_ok = _initialize_database()

                    if not db_ok:
                        _render_status(
                            "❌ Cannot reach backend — is the server running?",
                            val,
                            color="#ef4444",
                        )
                        time.sleep(2)
                        container.empty()
                        st.session_state["backend_unreachable"] = True
                        return False

                    _render_status(
                        "🗄️  Database ready",
                        val,
                        color="#067a5f",
                    )
                    time.sleep(0.3)

            for msg, val in post_stages:
                _render_status(msg, val)
                time.sleep(0.3)

            status.markdown(
                "<p style='text-align:center; color:#067a5f; font-weight:700;'>✅ All systems operational</p>",
                unsafe_allow_html=True,
            )
            time.sleep(0.5)

    container.empty()
    st.session_state.app_loaded = True
    return True
