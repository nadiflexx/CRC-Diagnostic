"""
Reusable card components.
"""

import streamlit as st


def page_header(title: str, subtitle: str, icon: str = "") -> None:
    """Renders a consistent page header."""
    icon_html = f"<span style='margin-right: 0.5rem;'>{icon}</span>" if icon else ""
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-header-title">{icon_html}{title}</div>
            <div class="page-header-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_profile_card(title: str, fields: list[tuple[str, str]]) -> None:
    """
    Renders a complete patient profile card in ONE st.markdown call.

    Args:
        title: Card section title (e.g., "🏥 Ficha del Paciente").
        fields: List of (label, value) tuples.
    """
    items_html = ""
    for label, value in fields:
        # Escape any special HTML characters in values
        safe_value = (
            str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        items_html += f'<div class="profile-item"><div class="profile-label">{label}</div><div class="profile-value">{safe_value}</div></div>'

    card_html = f'<div class="gastro-card"><div class="section-title">{title}</div><div class="profile-grid">{items_html}</div></div>'

    st.markdown(card_html, unsafe_allow_html=True)


def render_xai_grid(
    cards_html: str, title: str = "🧠 Explicabilidad del Modelo (XAI)"
) -> None:
    """
    Renders the complete XAI image grid inside a gastro-card.
    All HTML is emitted in ONE call.

    Args:
        cards_html: Pre-built HTML string of image cards.
        title: Section title.
    """
    st.markdown(
        f"""
        <div class="gastro-card">
            <div class="section-title">{title}</div>
            <div class="xai-grid">
                {cards_html}
        """,
        unsafe_allow_html=True,
    )


def render_segmentation(img_src: str) -> None:
    """Renders segmentation result inside a card, in ONE call."""
    st.markdown(
        f"""
        <div class="gastro-card">
            <div class="section-title">📍 Localización Exacta (Segmentación U-Net)</div>
            <div class="segmentation-result">
                <img src="{img_src}" alt="Segmentation Result">
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_card(title: str, inner_html: str) -> None:
    """
    Generic: renders any inner HTML inside a gastro-card with title.
    Everything in ONE st.markdown call.
    """
    st.markdown(
        f"""
        <div class="gastro-card">
            <div class="section-title">{title}</div>
            {inner_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_grid_card(
    img_src: str, title: str, subtitle: str = "", special: bool = False
) -> str:
    """
    Returns HTML string for ONE image card (to be combined in a grid).
    Does NOT call st.markdown — the caller assembles the full grid.
    """
    special_cls = " special" if special else ""
    return f"""
    <div class="img-grid-card{special_cls}">
        <img src="{img_src}" alt="{title}">
        <div class="img-grid-title">{title}</div>
        <div class="img-grid-sub">{subtitle}</div>
    </div>
    """


def stat_card(icon: str, value: str, label: str) -> str:
    """Returns HTML string for a stat mini-card (landing page)."""
    return f"""
    <div class="stat-card">
        <span class="stat-icon">{icon}</span>
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
    </div>
    """


def module_card(icon: str, title: str, description: str) -> str:
    """Returns HTML string for a landing page module card."""
    return f"""
    <div class="module-card">
        <span class="module-icon">{icon}</span>
        <div class="module-title">{title}</div>
        <div class="module-desc">{description}</div>
    </div>
    """
