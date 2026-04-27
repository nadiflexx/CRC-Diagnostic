"""
Banner & notification components.
All HTML is self-contained in single st.markdown calls.
"""

import streamlit as st


def show_result_banner(title: str, score: str, level: str = "green") -> None:
    """Displays a result banner with traffic-light styling."""
    level_map = {
        "green": ("result-banner-success", "banner-score-green"),
        "orange": ("result-banner-warning", "banner-score-orange"),
        "red": ("result-banner-danger", "banner-score-red"),
    }
    banner_cls, score_cls = level_map.get(level, level_map["green"])

    st.markdown(
        f"""
        <div class="result-banner {banner_cls}">
            <div class="banner-label">{title}</div>
            <div class="banner-score {score_cls}">{score}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_empty_state(icon: str, title: str, message: str) -> None:
    """Displays a friendly empty-state placeholder. ONE call."""
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-icon">{icon}</div>
            <div class="empty-state-title">{title}</div>
            <div class="empty-state-msg">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
