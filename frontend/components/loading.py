"""
Splash Screen & Loading Animation for Endo-AID.
"""

import time

import streamlit as st


def show_splash_screen() -> bool:
    """
    Displays a professional splash loading screen with staged progress.
    Returns True when complete.
    """
    if st.session_state.get("app_loaded", False):
        return True

    container = st.empty()

    with container.container():
        # Hide sidebar during load
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

            stages = [
                ("🔌 Connecting to backend services...", 15),
                ("🗄️ Loading patient database...", 30),
                ("🧠 Initializing classification models...", 50),
                ("🔬 Loading U-Net segmentation engine...", 65),
                ("📊 Preparing analytics pipelines...", 80),
                ("🧪 Calibrating risk scoring...", 92),
                ("✅ System ready", 100),
            ]

            for msg, val in stages:
                status.markdown(
                    f"<p style='text-align:center; color:#94A3B8; font-size:0.95rem;'>{msg}</p>",
                    unsafe_allow_html=True,
                )
                progress.progress(val)
                time.sleep(0.3)

            status.markdown(
                "<p style='text-align:center; color:#067a5f; font-weight:700;'>✅ All systems operational</p>",
                unsafe_allow_html=True,
            )
            time.sleep(0.5)

    container.empty()
    st.session_state.app_loaded = True
    return True
