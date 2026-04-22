"""
Patient Management & Clinical History Dashboard — Endo-AID.

Redesigned for professional clinical presentation:
  • Compact patient profile header with risk flags
  • Visit type badges (Endoscopy / Tumoral / Full / Smoking)
  • Visit-type-aware table and charts
  • Improved KPI cards with trend indicators
  • Tabbed analytics section
"""

from __future__ import annotations

from datetime import date

from components.banners import show_empty_state
from components.sidebar import render_sidebar
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from utils.api_client import create_patient, get_patient_history, get_patients
from utils.helpers import apply_custom_css

st.set_page_config(
    page_title="Patients · Endo-AID",
    page_icon="🌿",
    layout="wide",
)
apply_custom_css()
render_sidebar()

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

_C = {
    "green": "#388E3C",
    "orange": "#F57C00",
    "red": "#D32F2F",
    "teal": "#5a8d7f",
    "blue": "#1565C0",
    "purple": "#6A1B9A",
    "grey": "#9E9E9E",
    "dark": "#2d5a4e",
}

_DIAG_COLOR = {
    "negative": _C["green"],
    "suspicious": _C["orange"],
    "positive": _C["red"],
}

_CLASS_COLOR = {
    "polyp": _C["red"],
    "inflammation": _C["orange"],
    "normal": _C["green"],
}

_VISIT_META = {
    "endoscopy": ("🔬", "#1565C0", "#E3F2FD", "Endoscopy"),
    "tumor_analysis": ("🧬", "#6A1B9A", "#F3E5F5", "Tumoral"),
    "smoking_triage": ("🫁", "#F57C00", "#FFF3E0", "Smoking"),
    "full": ("⚕️", "#2d5a4e", "#E8F5E9", "Full"),
}


def _rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ─────────────────────────────────────────────────────────────────────────────
# Badge helpers
# ─────────────────────────────────────────────────────────────────────────────


def _badge(label: str, color: str, bg: str, border: str = "") -> str:
    border_css = f"border:1px solid {border};" if border else ""
    return (
        f"<span style='background:{bg}; color:{color}; font-weight:700;"
        f"padding:2px 9px; border-radius:10px; font-size:0.75rem;"
        f"{border_css}'>{label}</span>"
    )


def _risk_badge(score: float | None) -> str:
    if score is None:
        return _badge("N/A", _C["grey"], "#f5f5f5")
    if score >= 0.7:
        color, label = _C["red"], f"HIGH ({score:.0%})"
    elif score >= 0.4:
        color, label = _C["orange"], f"MOD ({score:.0%})"
    else:
        color, label = _C["green"], f"LOW ({score:.0%})"
    return _badge(label, color, _rgba(color, 0.12), _rgba(color, 0.35))


def _diag_badge(diagnosis: str | None) -> str:
    if not diagnosis:
        return _badge("N/A", _C["grey"], "#f5f5f5")
    color = _DIAG_COLOR.get(diagnosis, _C["grey"])
    label = {
        "negative": "Negative",
        "suspicious": "Suspicious",
        "positive": "Positive",
    }.get(diagnosis, diagnosis.title())
    return _badge(label, color, _rgba(color, 0.12), _rgba(color, 0.35))


def _visit_type_badge(visit_type: str | None) -> str:
    meta = _VISIT_META.get(visit_type or "full", _VISIT_META["full"])
    icon, color, bg, label = meta
    return _badge(f"{icon} {label}", color, bg, _rgba(color, 0.3))


def _fmt(val, fmt: str = ".2f", fallback: str = "—") -> str:
    if val is None:
        return fallback
    return format(val, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Patient profile header
# ─────────────────────────────────────────────────────────────────────────────


def _mini_stat(icon: str, label: str, value: str) -> str:
    """Generate a single mini stat card as an HTML string."""
    color = _C["dark"]  # extract first — no dict access inside f-string
    return (
        '<div style="background:white; border:1px solid #d0e8e0;'
        "border-radius:10px; padding:0.45rem 0.75rem; min-width:90px;"
        'box-shadow:0 1px 4px rgba(0,0,0,0.05);">'
        '<div style="font-size:0.7rem; color:#9E9E9E; margin-bottom:1px;">'
        f"{icon} {label}"
        "</div>"
        f'<div style="font-size:0.92rem; font-weight:700; color:{color};">'
        f"{value}"
        "</div>"
        "</div>"
    )


def _render_patient_header(patient: dict, df_raw: pd.DataFrame) -> None:
    """Full-width professional patient profile card."""

    # ── Compute stats ─────────────────────────────────────────────────────
    age = (date.today() - date.fromisoformat(patient["date_of_birth"])).days // 365
    total_visits = len(df_raw)

    last_diag = df_raw.iloc[0]["diagnosis"] if total_visits else None
    last_score = df_raw.iloc[0]["multimodal_score"] if total_visits else None
    last_date = str(df_raw.iloc[0]["date"])[:10] if total_visits else "—"
    colon_count = (
        int(df_raw["colonoscopy_performed"].sum())
        if total_visits and "colonoscopy_performed" in df_raw.columns
        else 0
    )

    diag_color = _DIAG_COLOR.get(last_diag or "", _C["grey"])
    diag_label = {
        "negative": "Negative",
        "suspicious": "Suspicious",
        "positive": "Positive",
    }.get(last_diag or "", "No data yet")

    score_color = (
        _C["red"]
        if (last_score or 0) >= 0.7
        else _C["orange"]
        if (last_score or 0) >= 0.4
        else _C["green"]
    )
    score_str = f"{last_score:.1%}" if last_score is not None else "N/A"

    # ── Lifestyle maps ────────────────────────────────────────────────────
    smoking_map = {
        "never": "🚭 Never",
        "former": "🚬 Former",
        "current": "🔴 Current",
    }
    alcohol_map = {
        "never": "❌ Never",
        "moderate": "🍷 Moderate",
        "heavy": "⚠️ Heavy",
    }
    smoking_val = smoking_map.get(patient.get("smoking_status", ""), "—")
    alcohol_val = alcohol_map.get(patient.get("alcohol_consumption", ""), "—")

    bmi_val = patient.get("bmi")
    bmi_str = f"{bmi_val:.1f}" if bmi_val is not None else "—"
    height_val = patient.get("height_cm")
    height_str = f"{height_val:.0f} cm" if height_val is not None else "—"
    weight_val = patient.get("weight_kg")
    weight_str = f"{weight_val:.0f} kg" if weight_val is not None else "—"

    # ── Risk flags ────────────────────────────────────────────────────────
    flag_items = []
    if patient.get("family_history_ccr"):
        flag_items.append("👨‍👩‍👧 Family CRC")
    if patient.get("has_ibd"):
        flag_items.append("🔴 IBD")
    if patient.get("previous_polyps"):
        flag_items.append("🟠 Prev. Polyps")
    if patient.get("previous_cancer"):
        flag_items.append("⚠️ Prev. Cancer")

    flag_badge_style = (
        "background:#fff3cd; color:#856404; border:1px solid #ffc107;"
        "border-radius:8px; padding:1px 8px; font-size:0.75rem; margin-right:4px;"
    )
    if flag_items:
        flags_html = "".join(
            f'<span style="{flag_badge_style}">{f}</span>' for f in flag_items
        )
    else:
        flags_html = '<span style="color:#9E9E9E; font-size:0.8rem;">No notable risk flags</span>'

    stats_html = "".join(
        [
            _mini_stat("📅", "Total Visits", str(total_visits)),
            _mini_stat("📏", "Height", height_str),
            _mini_stat("⚖️", "Weight", weight_str),
            _mini_stat("📊", "BMI", bmi_str),
            _mini_stat("🚬", "Smoking", smoking_val),
            _mini_stat("🍷", "Alcohol", alcohol_val),
            _mini_stat("🔬", "Colonoscopies", str(colon_count)),
        ]
    )

    dark = _C["dark"]
    grey6 = "#6c757d"
    grey9 = "#9E9E9E"

    card = (
        '<div style="'
        "background:linear-gradient(135deg,#f8fffe 0%,#edf7f4 100%);"
        "border:1.5px solid #a8d5c2; border-radius:16px;"
        "padding:1.4rem 1.8rem; margin-bottom:0.5rem;"
        'box-shadow:0 2px 12px rgba(90,141,127,0.10);">'
        '<div style="display:flex; justify-content:space-between;'
        "align-items:flex-start; flex-wrap:wrap; gap:0.5rem;"
        'margin-bottom:0.9rem;">'
        "<div>"
        f'<div style="font-size:1.35rem; font-weight:800; color:{dark};'
        'line-height:1.2;">'
        f"{patient['first_name']} {patient['last_name']}"
        "</div>"
        f'<div style="font-size:0.82rem; color:{grey6}; margin-top:2px;">'
        f"ID {patient['id']} &nbsp;·&nbsp; "
        f"{patient.get('gender', '').title()} &nbsp;·&nbsp; "
        f"{age} years &nbsp;·&nbsp; DOB {patient['date_of_birth']}"
        "</div>"
        "</div>"
        '<div style="text-align:right;">'
        f'<div style="font-size:0.75rem; color:{grey9}; margin-bottom:2px;">'
        f"Last visit &nbsp;·&nbsp; {last_date}"
        "</div>"
        f'<div style="font-size:1.6rem; font-weight:800; color:{diag_color};">'
        f"{diag_label}"
        "</div>"
        f'<div style="font-size:0.78rem; color:{score_color}; font-weight:600;">'
        f"Risk score: {score_str}"
        "</div>"
        "</div>"
        "</div>"
        '<div style="display:flex; gap:1rem; flex-wrap:wrap;'
        'margin-bottom:0.9rem;">'
        f"{stats_html}"
        "</div>"
        '<div style="display:flex; align-items:center; gap:0.4rem; flex-wrap:wrap;">'
        f'<span style="font-size:0.78rem; color:{grey6}; font-weight:600;'
        'margin-right:4px;">Risk flags:</span>'
        f"{flags_html}"
        "</div>"
        "</div>"
    )

    st.markdown(card, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# KPI cards
# ─────────────────────────────────────────────────────────────────────────────


def _render_kpis(df: pd.DataFrame) -> None:
    total = len(df)
    polyps = int(
        sum(
            1
            for _, r in df.iterrows()
            if (r.get("ai_snapshot") or {}).get("image", {}).get("prediction_class")
            == "polyp"
        )
    )
    colons = (
        int(df["colonoscopy_performed"].sum())
        if "colonoscopy_performed" in df.columns
        else 0
    )
    high_risk = int(
        sum(1 for _, r in df.iterrows() if (r.get("multimodal_score") or 0) >= 0.7)
    )

    last_score = df.iloc[0]["multimodal_score"] if total else None
    trend_delta, trend_color = None, "off"
    if total >= 2:
        prev = df.iloc[1]["multimodal_score"]
        curr = df.iloc[0]["multimodal_score"]
        if prev is not None and curr is not None:
            trend_delta = f"{curr - prev:+.1%}"
            trend_color = "inverse"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📅 Visits", total)
    c2.metric(
        "🎯 Latest Score",
        f"{last_score:.1%}" if last_score is not None else "N/A",
        delta=trend_delta,
        delta_color=trend_color,
    )
    c3.metric("🔴 Polyp Visits", polyps)
    c4.metric("🔬 Colonoscopies", colons)
    c5.metric("⚠️ High-Risk Visits", high_risk)


# ─────────────────────────────────────────────────────────────────────────────
# Visit history table
# ─────────────────────────────────────────────────────────────────────────────


def _render_visit_table(df: pd.DataFrame) -> None:
    st.markdown(
        "<p style='font-weight:700; color:#2d5a4e; font-size:0.95rem;"
        "margin-bottom:0.5rem;'>📋 Visit History</p>",
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        "<div style='"
        "display:grid;"
        "grid-template-columns:88px 110px 108px 120px 120px 110px 120px 110px;"
        "gap:6px; padding:7px 14px;"
        "background:#edf7f4; border-radius:10px 10px 0 0;"
        "font-size:0.75rem; font-weight:700; color:#2d5a4e;'>"
        "<div>Date</div><div>Type</div><div>Diagnosis</div>"
        "<div>Multimodal</div><div>Image Score</div>"
        "<div>Tumoral</div><div>AI Finding</div><div>Colonoscopy</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    for _, row in df.sort_values("date", ascending=False).iterrows():
        date_str = str(row.get("date", ""))[:10]
        diag = row.get("diagnosis") or ""
        mm_score = row.get("multimodal_score")
        img_score = row.get("image_score")
        tab_score = row.get("tabular_score")
        visit_type = row.get("visit_type") or "full"
        colon = "✅ Yes" if row.get("colonoscopy_performed") else "➖ No"

        snap = row.get("ai_snapshot") or {}
        img_snap = snap.get("image", {})
        cls = img_snap.get("prediction_class", "") or ""
        cls_color = _CLASS_COLOR.get(cls.lower(), _C["grey"])
        cls_label = {
            "polyp": "🔴 Polyp",
            "inflammation": "🟠 Inflam.",
            "normal": "🟢 Normal",
        }.get(cls.lower(), cls.title() or "—")

        row_bg = (
            _rgba(_C["red"], 0.04)
            if diag == "positive"
            else _rgba(_C["orange"], 0.03)
            if diag == "suspicious"
            else "white"
        )

        st.markdown(
            f"<div style='"
            f"display:grid;"
            f"grid-template-columns:88px 110px 108px 120px 120px 110px 120px 110px;"
            f"gap:6px; padding:8px 14px;"
            f"background:{row_bg}; border-bottom:1px solid #f0f5f3;"
            f"font-size:0.8rem; align-items:center;'>"
            f"<div style='color:#555; font-weight:600;'>{date_str}</div>"
            f"<div>{_visit_type_badge(visit_type)}</div>"
            f"<div>{_diag_badge(diag)}</div>"
            f"<div>{_risk_badge(mm_score)}</div>"
            f"<div>{_risk_badge(img_score) if img_score is not None else _badge('—', _C['grey'], '#f5f5f5')}</div>"
            f"<div>{_risk_badge(tab_score) if tab_score is not None else _badge('—', _C['grey'], '#f5f5f5')}</div>"
            f"<div><span style='color:{cls_color}; font-weight:600;'>{cls_label}</span></div>"
            f"<div style='color:#555;'>{colon}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='border-radius:0 0 10px 10px; background:#edf7f4;"
        "height:5px;'></div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────


def _chart_risk_timeline(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    for y0, y1, color, label in [
        (0.0, 0.4, _rgba(_C["green"], 0.07), "Low"),
        (0.4, 0.7, _rgba(_C["orange"], 0.07), "Moderate"),
        (0.7, 1.0, _rgba(_C["red"], 0.07), "High"),
    ]:
        fig.add_hrect(
            y0=y0,
            y1=y1,
            fillcolor=color,
            line_width=0,
            annotation_text=label,
            annotation_position="right",
            annotation_font_size=10,
            annotation_font_color="#888",
        )

    for y, color in [(0.4, _C["orange"]), (0.7, _C["red"])]:
        fig.add_hline(y=y, line_dash="dot", line_color=color, line_width=1, opacity=0.5)

    traces = [
        ("multimodal_score", "Multimodal", _C["teal"], 3, "circle"),
        ("image_score", "Image", _C["blue"], 2, "diamond"),
        ("tabular_score", "Tumoral", _C["purple"], 2, "square"),
    ]
    for col, name, color, width, symbol in traces:
        if col in df.columns and df[col].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[col],
                    name=name,
                    mode="lines+markers",
                    line={"color": color, "width": width},
                    marker={
                        "color": color,
                        "size": 8,
                        "symbol": symbol,
                        "line": {"color": "white", "width": 1.5},
                    },
                    hovertemplate=f"<b>{name}</b><br>%{{x|%d/%m/%Y}}<br>%{{y:.1%}}<extra></extra>",
                )
            )

    for _, row in df.iterrows():
        diag = row.get("diagnosis")
        score = row.get("multimodal_score")
        if diag and score is not None:
            color = _DIAG_COLOR.get(diag, _C["grey"])
            fig.add_trace(
                go.Scatter(
                    x=[row["date"]],
                    y=[score],
                    mode="markers",
                    marker={
                        "color": color,
                        "size": 13,
                        "symbol": "star",
                        "line": {"color": "white", "width": 1.5},
                    },
                    showlegend=False,
                    hovertemplate=f"<b>Dx: {diag.title()}</b><br>%{{y:.1%}}<extra></extra>",
                )
            )

    fig.update_layout(
        title={
            "text": "Longitudinal AI Risk Evolution",
            "font": {"size": 14, "color": _C["dark"]},
        },
        xaxis={"title": "Visit Date", "tickformat": "%d/%m/%Y", "showgrid": False},
        yaxis={
            "title": "Risk Score",
            "tickformat": ".0%",
            "range": [0, 1.05],
            "gridcolor": "#f0f0f0",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=370,
        margin={"l": 10, "r": 80, "t": 55, "b": 10},
        hovermode="x unified",
    )
    return fig


def _chart_visit_type_distribution(df: pd.DataFrame) -> go.Figure:
    """Donut chart — visits by type."""
    counts = df["visit_type"].value_counts()
    labels, values, colors = [], [], []
    for vt, meta in _VISIT_META.items():
        if vt in counts:
            icon, color, _, label = meta
            labels.append(f"{icon} {label}")
            values.append(int(counts[vt]))
            colors.append(color)

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker={"colors": colors, "line": {"color": "white", "width": 2}},
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value} visits<extra></extra>",
        )
    )
    fig.update_layout(
        title={
            "text": "Visit Type Distribution",
            "font": {"size": 13, "color": _C["dark"]},
        },
        showlegend=False,
        height=300,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        paper_bgcolor="white",
        annotations=[
            {
                "text": f"<b>{sum(values)}</b><br>total",
                "x": 0.5,
                "y": 0.5,
                "font_size": 14,
                "showarrow": False,
                "font_color": _C["dark"],
            }
        ],
    )
    return fig


def _chart_class_distribution(df: pd.DataFrame) -> go.Figure:
    counts: dict[str, int] = {"polyp": 0, "inflammation": 0, "normal": 0}
    for _, row in df.iterrows():
        cls = (
            (row.get("ai_snapshot") or {}).get("image", {}).get("prediction_class", "")
        ).lower()
        if cls in counts:
            counts[cls] += 1

    total_img = sum(counts.values())
    labels = ["Polyp", "Inflammation", "Normal"]
    values = [counts["polyp"], counts["inflammation"], counts["normal"]]
    colors = [_C["red"], _C["orange"], _C["green"]]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker={"colors": colors, "line": {"color": "white", "width": 2}},
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value} visits<extra></extra>",
        )
    )
    fig.update_layout(
        title={
            "text": "Endoscopic Findings",
            "font": {"size": 13, "color": _C["dark"]},
        },
        showlegend=False,
        height=300,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        paper_bgcolor="white",
        annotations=[
            {
                "text": f"<b>{total_img}</b><br>images",
                "x": 0.5,
                "y": 0.5,
                "font_size": 14,
                "showarrow": False,
                "font_color": _C["dark"],
            }
        ],
    )
    return fig


def _chart_probability_heatmap(df: pd.DataFrame) -> go.Figure | None:
    rows = []
    for _, row in df.iterrows():
        probs = (row.get("ai_snapshot") or {}).get("image", {}).get("probabilities", {})
        if probs:
            rows.append(
                {
                    "date": row["date"],
                    "polyp": probs.get("polyp", 0),
                    "inflammation": probs.get("inflammation", 0),
                    "normal": probs.get("normal", 0),
                }
            )
    if not rows:
        return None

    heat_df = pd.DataFrame(rows).sort_values("date")
    dates = [str(d)[:10] for d in heat_df["date"]]
    z = [
        heat_df["polyp"].tolist(),
        heat_df["inflammation"].tolist(),
        heat_df["normal"].tolist(),
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=dates,
            y=["Polyp", "Inflammation", "Normal"],
            colorscale=[
                [0.0, "#E8F5E9"],
                [0.4, "#FFF9C4"],
                [0.7, "#FFECB3"],
                [1.0, "#B71C1C"],
            ],
            zmin=0,
            zmax=1,
            text=[[f"{v:.0%}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate="<b>%{y}</b> · %{x}<br>%{z:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title={
            "text": "Probability Heatmap per Visit",
            "font": {"size": 13, "color": _C["dark"]},
        },
        height=240,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        paper_bgcolor="white",
        xaxis={"tickangle": -30, "tickfont": {"size": 10}},
    )
    return fig


def _chart_biomarkers(df: pd.DataFrame) -> go.Figure | None:
    sub = df[["date", "cea", "hemoglobin"]].dropna(subset=["date"])
    sub = sub[sub[["cea", "hemoglobin"]].notna().any(axis=1)]
    if sub.empty:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if sub["cea"].notna().any():
        fig.add_trace(
            go.Bar(
                x=sub["date"],
                y=sub["cea"],
                name="CEA (ng/mL)",
                marker_color=[
                    _C["red"] if v and v > 5 else _rgba(_C["teal"], 0.6)
                    for v in sub["cea"]
                ],
                hovertemplate="CEA: %{y:.1f} ng/mL<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_hline(
            y=5,
            line_dash="dash",
            line_color=_C["orange"],
            line_width=1,
            opacity=0.7,
            annotation_text="Ref=5",
            annotation_font_size=9,
            secondary_y=False,
        )

    if sub["hemoglobin"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=sub["date"],
                y=sub["hemoglobin"],
                name="Haemoglobin (g/dL)",
                mode="lines+markers",
                line={"color": _C["purple"], "width": 2},
                marker={"size": 7, "color": _C["purple"]},
                hovertemplate="Hb: %{y:.1f} g/dL<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.add_hline(
            y=12,
            line_dash="dash",
            line_color=_C["purple"],
            line_width=1,
            opacity=0.5,
            annotation_text="Ref=12",
            annotation_font_size=9,
            secondary_y=True,
        )

    fig.update_layout(
        title={
            "text": "CEA & Haemoglobin over Time",
            "font": {"size": 13, "color": _C["dark"]},
        },
        height=300,
        margin={"l": 10, "r": 60, "t": 45, "b": 10},
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        hovermode="x unified",
        xaxis={"showgrid": False, "tickformat": "%d/%m/%Y"},
    )
    fig.update_yaxes(title_text="CEA (ng/mL)", secondary_y=False, gridcolor="#f0f0f0")
    fig.update_yaxes(title_text="Hb (g/dL)", secondary_y=True, showgrid=False)
    return fig


def _chart_ensemble_weights(df: pd.DataFrame) -> go.Figure | None:
    rows = []
    for _, row in df.iterrows():
        img = (row.get("ai_snapshot") or {}).get("image", {})
        alpha, beta = img.get("alpha_used"), img.get("beta_used")
        if alpha is not None and beta is not None:
            rows.append({"date": row["date"], "alpha": alpha, "beta": beta})
    if len(rows) < 2:
        return None

    w = pd.DataFrame(rows).sort_values("date")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=w["date"],
            y=w["alpha"],
            name="α Model A",
            fill="tozeroy",
            line={"color": _C["blue"], "width": 2},
            fillcolor=_rgba(_C["blue"], 0.12),
            hovertemplate="α=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=w["date"],
            y=w["beta"],
            name="β Model B",
            fill="tozeroy",
            line={"color": _C["purple"], "width": 2},
            fillcolor=_rgba(_C["purple"], 0.12),
            hovertemplate="β=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={
            "text": "Ensemble Weights per Visit",
            "font": {"size": 13, "color": _C["dark"]},
        },
        height=260,
        yaxis={"title": "Weight", "range": [0, 1], "gridcolor": "#f0f0f0"},
        xaxis={"showgrid": False, "tickformat": "%d/%m/%Y"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        hovermode="x unified",
    )
    return fig


def _chart_tumoral_scores(df: pd.DataFrame) -> go.Figure | None:
    """Bar chart of tumoral prediction scores for tumoral/full visits."""
    sub = df[df["tabular_score"].notna()][["date", "tabular_score"]].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("date")

    colors = [
        _C["red"] if v >= 0.5 else _C["orange"] if v >= 0.3 else _C["green"]
        for v in sub["tabular_score"]
    ]
    fig = go.Figure(
        go.Bar(
            x=sub["date"],
            y=sub["tabular_score"],
            marker_color=colors,
            hovertemplate="Tumoral score: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color=_C["red"],
        line_width=1,
        opacity=0.6,
        annotation_text="High risk",
        annotation_font_size=9,
    )
    fig.add_hline(
        y=0.3,
        line_dash="dot",
        line_color=_C["orange"],
        line_width=1,
        opacity=0.6,
        annotation_text="Suspicious",
        annotation_font_size=9,
    )
    fig.update_layout(
        title={
            "text": "Tumoral Model Scores per Visit",
            "font": {"size": 13, "color": _C["dark"]},
        },
        yaxis={
            "title": "Score",
            "tickformat": ".0%",
            "range": [0, 1.05],
            "gridcolor": "#f0f0f0",
        },
        xaxis={"showgrid": False, "tickformat": "%d/%m/%Y"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=280,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Risk factors
# ─────────────────────────────────────────────────────────────────────────────


def _render_risk_factors(df: pd.DataFrame) -> None:
    factor_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        factors = (
            (row.get("ai_snapshot") or {})
            .get("tabular", {})
            .get("top_risk_factors", [])
        )
        for item in factors:
            name = item[0] if isinstance(item, (list, tuple)) else str(item)
            factor_counts[name] = factor_counts.get(name, 0) + 1

    if not factor_counts:
        st.markdown(
            "<p style='color:#9E9E9E; font-size:0.85rem;'>"
            "No risk factors recorded yet.</p>",
            unsafe_allow_html=True,
        )
        return

    total = len(df)
    for name, count in sorted(factor_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / total
        color = _C["red"] if pct >= 0.5 else _C["orange"] if pct >= 0.25 else _C["teal"]
        st.markdown(
            f"<div style='margin-bottom:7px;'>"
            f"<div style='display:flex; justify-content:space-between;"
            f"font-size:0.82rem; margin-bottom:3px;'>"
            f"<span style='font-weight:600; color:{_C['dark']};'>{name}</span>"
            f"<span style='color:#6c757d;'>{count}/{total}</span>"
            f"</div>"
            f"<div style='background:#f0f0f0; border-radius:5px;"
            f"height:13px; overflow:hidden;'>"
            f"<div style='width:{pct * 100:.0f}%; background:{color};"
            f"height:100%; border-radius:5px;'></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Filters
# ─────────────────────────────────────────────────────────────────────────────


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander("🔽 Filters", expanded=False):
        c1, c2, c3, c4 = st.columns(4)

        dates = pd.to_datetime(df["date"]).dt.date
        min_d, max_d = dates.min(), dates.max()
        with c1:
            date_range = st.date_input(
                "Date range",
                value=(min_d, max_d),
                min_value=min_d,
                max_value=max_d,
                key="filter_dates",
            )
        diag_opts = ["All"] + sorted(df["diagnosis"].dropna().unique().tolist())
        with c2:
            diag_filter = st.selectbox("Diagnosis", diag_opts, key="filter_diag")

        vtype_opts = ["All"] + sorted(df["visit_type"].dropna().unique().tolist())
        with c3:
            vtype_filter = st.selectbox("Visit Type", vtype_opts, key="filter_vtype")

        with c4:
            colon_filter = st.selectbox(
                "Colonoscopy",
                ["All", "With colonoscopy", "Without colonoscopy"],
                key="filter_colon",
            )

    filtered = df.copy()
    filtered["_date"] = pd.to_datetime(filtered["date"]).dt.date

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        filtered = filtered[
            (filtered["_date"] >= date_range[0]) & (filtered["_date"] <= date_range[1])
        ]
    if diag_filter != "All":
        filtered = filtered[filtered["diagnosis"] == diag_filter]
    if vtype_filter != "All":
        filtered = filtered[filtered["visit_type"] == vtype_filter]
    if colon_filter == "With colonoscopy":
        filtered = filtered[filtered["colonoscopy_performed"].astype(bool)]
    elif colon_filter == "Without colonoscopy":
        filtered = filtered[~filtered["colonoscopy_performed"].astype(bool)]

    return filtered.drop(columns=["_date"])


# ─────────────────────────────────────────────────────────────────────────────
# Section title helper
# ─────────────────────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    st.markdown(
        f"<div style='font-weight:700; color:{_C['dark']}; font-size:0.95rem;"
        f"border-left:4px solid {_C['teal']}; padding-left:0.6rem;"
        f"margin:1.2rem 0 0.6rem 0;'>{title}</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────


def render() -> None:
    st.markdown(
        f"""
        <div style="
            display:flex; align-items:center; gap:0.75rem;
            margin-bottom:0.25rem;
        ">
            <span style="font-size:2rem;">👥</span>
            <div>
                <div style="font-size:1.45rem; font-weight:800;
                            color:{_C["dark"]}; line-height:1.1;">
                    Patients & Clinical History
                </div>
                <div style="font-size:0.85rem; color:#6c757d;">
                    Clinical database · Longitudinal AI risk tracking
                </div>
            </div>
        </div>
        <hr style="border:none; border-top:1px solid #d0e8e0;
                   margin:0.6rem 0 1rem 0;">
        """,
        unsafe_allow_html=True,
    )

    tab_search, tab_new = st.tabs(["📋 Search & Profile", "➕ New Registration"])

    with tab_search:
        _render_search_tab()

    with tab_new:
        _render_new_patient_tab()


def _render_search_tab() -> None:
    patients = get_patients()
    if not patients:
        show_empty_state(
            "👥",
            "No patients registered",
            "Use 'New Registration' to add the first patient.",
        )
        return

    # Patient selector
    opts = {f"{p['first_name']} {p['last_name']}  ·  ID {p['id']}": p for p in patients}
    col_sel, _ = st.columns([1, 3])
    with col_sel:
        selected_key = st.selectbox("🔍 Search Patient", list(opts.keys()))
    patient = opts[selected_key]

    history = get_patient_history(patient["id"])
    if not history:
        _render_patient_header(patient, pd.DataFrame())
        show_empty_state(
            "📊",
            "No visits recorded",
            "AI analyses will appear here as a longitudinal timeline.",
        )
        return

    df_raw = pd.DataFrame(history)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw = df_raw.sort_values("date", ascending=False).reset_index(drop=True)

    if "colonoscopy_performed" in df_raw.columns:
        df_raw["colonoscopy_performed"] = (
            df_raw["colonoscopy_performed"].fillna(False).astype(bool)
        )

    # ── Profile header ────────────────────────────────────────────────────
    _render_patient_header(patient, df_raw)

    # ── KPIs ──────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _render_kpis(df_raw)

    # ── Filters ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    df = _apply_filters(df_raw)

    if df.empty:
        st.info("ℹ️ No visits match the selected filters.")
        return

    # ── Visit table ───────────────────────────────────────────────────────
    _section("📋 Visit History")
    _render_visit_table(df)

    # ── Analytics tabs ────────────────────────────────────────────────────
    _section("📊 Clinical Analytics")

    tab_trend, tab_endo, tab_tumor, tab_bio, tab_factors = st.tabs(
        [
            "📈 Risk Trend",
            "🔬 Endoscopy",
            "🧬 Tumoral",
            "🩸 Biomarkers",
            "⚠️ Risk Factors",
        ]
    )

    with tab_trend:
        has_scores = (
            df["multimodal_score"].notna().any()
            or df["image_score"].notna().any()
            or df["tabular_score"].notna().any()
        )
        if has_scores:
            st.plotly_chart(_chart_risk_timeline(df), use_container_width=True)
        else:
            st.info("No score data available for this selection.")

        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.plotly_chart(
                _chart_visit_type_distribution(df), use_container_width=True
            )
        with col_b:
            fig_ens = _chart_ensemble_weights(df)
            if fig_ens:
                st.plotly_chart(fig_ens, use_container_width=True)
            else:
                st.info(
                    "Not enough data for ensemble weight chart (≥ 2 endoscopy visits needed)."
                )

    with tab_endo:
        endo_df = df[df["visit_type"].isin(["endoscopy", "full"])]
        if endo_df.empty:
            st.info("No endoscopy visits in the current filter selection.")
        else:
            col_donut, col_heat = st.columns([1, 2], gap="large")
            with col_donut:
                st.plotly_chart(
                    _chart_class_distribution(endo_df), use_container_width=True
                )
            with col_heat:
                fig_heat = _chart_probability_heatmap(endo_df)
                if fig_heat:
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("No probability data available for heatmap.")

    with tab_tumor:
        tumor_df = df[df["visit_type"].isin(["tumor_analysis", "full"])]
        if tumor_df.empty:
            st.info("No tumoral analysis visits in the current filter selection.")
        else:
            fig_tumor = _chart_tumoral_scores(tumor_df)
            if fig_tumor:
                st.plotly_chart(fig_tumor, use_container_width=True)
            else:
                st.info("No tumoral score data available.")

    with tab_bio:
        fig_bio = _chart_biomarkers(df)
        if fig_bio:
            st.plotly_chart(fig_bio, use_container_width=True)
        else:
            st.info("No biomarker data (CEA / Haemoglobin) recorded.")

    with tab_factors:
        _render_risk_factors(df)


def _render_new_patient_tab() -> None:
    st.markdown(
        f"<div style='font-size:1.05rem; font-weight:700; color:{_C['dark']};"
        f"margin-bottom:1rem;'>📝 New Patient Registration</div>",
        unsafe_allow_html=True,
    )

    with st.form("new_patient_form"):
        c1, c2 = st.columns(2)
        first_name = c1.text_input("First Name *", placeholder="Mary")
        last_name = c2.text_input("Last Name *", placeholder="Smith")

        c3, c4 = st.columns(2)
        date_of_birth = c3.date_input(
            "Date of Birth *",
            value=date(1980, 1, 1),
            min_value=date(1900, 1, 1),
            max_value=date.today(),
        )
        gender = c4.selectbox("Sex *", ["male", "female", "other"])

        c5, c6 = st.columns(2)
        height_cm = c5.number_input("Height (cm)", 100.0, 250.0, 170.0, step=0.5)
        weight_kg = c6.number_input("Weight (kg)", 30.0, 300.0, 70.0, step=0.5)
        if height_cm > 0:
            bmi_val = weight_kg / ((height_cm / 100) ** 2)
            st.caption(f"📊 Estimated BMI: **{bmi_val:.1f}**")

        c7, c8 = st.columns(2)
        smoking = c7.selectbox(
            "Smoking Status",
            ["never", "former", "current"],
            format_func=lambda x: {
                "never": "Never",
                "former": "Former smoker",
                "current": "Current smoker",
            }[x],
        )
        alcohol = c8.selectbox(
            "Alcohol Consumption",
            ["never", "moderate", "heavy"],
            format_func=lambda x: {
                "never": "Never",
                "moderate": "Moderate",
                "heavy": "Heavy",
            }[x],
        )

        st.markdown("**Clinical Risk Flags**")
        cr1, cr2, cr3, cr4 = st.columns(4)
        fam_ccr = cr1.checkbox("Family history CRC")
        has_ibd = cr2.checkbox("IBD")
        prev_polyps = cr3.checkbox("Previous polyps")
        prev_cancer = cr4.checkbox("Previous cancer")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "✅ Register Patient", type="primary", use_container_width=True
        )

    if submitted:
        if not first_name.strip() or not last_name.strip():
            st.error("❌ First name and last name are required.")
        else:
            result = create_patient(
                {
                    "first_name": first_name.strip(),
                    "last_name": last_name.strip(),
                    "date_of_birth": date_of_birth.isoformat(),
                    "gender": gender,
                    "height_cm": height_cm,
                    "weight_kg": weight_kg,
                    "smoking_status": smoking,
                    "alcohol_consumption": alcohol,
                    "family_history_ccr": fam_ccr,
                    "has_ibd": has_ibd,
                    "previous_polyps": prev_polyps,
                    "previous_cancer": prev_cancer,
                }
            )
            if result:
                st.success(
                    f"✅ Patient **{first_name} {last_name}** "
                    f"registered with ID **{result.get('id', '?')}**."
                )
                st.balloons()
            else:
                st.error("❌ Could not register the patient. Check backend logs.")


render()
