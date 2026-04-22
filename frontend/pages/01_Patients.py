"""
Patient Management & Clinical History Dashboard.

Provides patient search, longitudinal risk tracking, clinical history
visualisation, and new patient registration through the Endo-AID backend.
"""

from __future__ import annotations

from datetime import date

from components.banners import show_empty_state
from components.cards import page_header, render_profile_card
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

_C = {
    "green": "#388E3C",
    "orange": "#F57C00",
    "red": "#D32F2F",
    "teal": "#5a8d7f",
    "blue": "#1565C0",
    "purple": "#6A1B9A",
    "grey": "#9E9E9E",
    "bg": "#f8fffe",
}

_DIAG_COLOR = {
    "negative": _C["green"],
    "suspicious": _C["orange"],
    "positive": _C["red"],
}

_RISK_COLOR = {
    "LOW": _C["green"],
    "MODERATE": _C["orange"],
    "HIGH": _C["red"],
    "VERY_HIGH": "#7B0000",
}

_CLASS_COLOR = {
    "polyp": _C["red"],
    "inflammation": _C["orange"],
    "normal": _C["green"],
}


def _rgba(hex_color: str, alpha: float = 0.15) -> str:
    """
    Convert a hex colour string to an rgba() CSS value.

    Args:
        hex_color: Colour in ``#RRGGBB`` format.
        alpha: Opacity level between 0.0 and 1.0.

    Returns:
        CSS rgba string, e.g. ``"rgba(56,142,60,0.15)"``.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _risk_badge(score: float | None) -> str:
    """
    Generate an HTML badge indicating the risk level for a given score.

    Args:
        score: Numeric risk score in [0, 1], or None if unavailable.

    Returns:
        HTML ``<span>`` string with colour-coded risk label and percentage.
        Returns a grey "N/A" span when score is None.
    """
    if score is None:
        return "<span style='color:#9E9E9E;'>N/A</span>"
    if score >= 0.7:
        color, label = _C["red"], "HIGH"
    elif score >= 0.4:
        color, label = _C["orange"], "MODERATE"
    else:
        color, label = _C["green"], "LOW"
    return (
        f"<span style='"
        f"background:{_rgba(color, 0.12)}; color:{color};"
        f"font-weight:700; padding:2px 10px; border-radius:12px;"
        f"font-size:0.8rem; border:1px solid {_rgba(color, 0.4)};'>"
        f"{label} ({score:.0%})</span>"
    )


def _diag_badge(diagnosis: str | None) -> str:
    """
    Generate an HTML badge for a diagnosis result value.

    Args:
        diagnosis: One of ``"negative"``, ``"suspicious"``, or ``"positive"``.
            Any other value is title-cased. None renders a grey "N/A" span.

    Returns:
        HTML ``<span>`` string with colour-coded diagnosis label.
    """
    if not diagnosis:
        return "<span style='color:#9E9E9E;'>N/A</span>"
    color = _DIAG_COLOR.get(diagnosis, _C["grey"])
    label_map = {
        "negative": "Negative",
        "suspicious": "Suspicious",
        "positive": "Positive",
    }
    label = label_map.get(diagnosis, diagnosis.title())
    return (
        f"<span style='"
        f"background:{_rgba(color, 0.12)}; color:{color};"
        f"font-weight:700; padding:2px 10px; border-radius:12px;"
        f"font-size:0.8rem; border:1px solid {_rgba(color, 0.4)};'>"
        f"{label}</span>"
    )


def _fmt(val, fmt: str = ".2f", fallback: str = "N/A") -> str:
    """
    Format a numeric value, returning a fallback string when the value is None.

    Args:
        val: Value to format. May be None.
        fmt: Python format specification string (default ``".2f"``).
        fallback: String returned when val is None (default ``"N/A"``).

    Returns:
        Formatted string or the fallback string.
    """
    if val is None:
        return fallback
    return format(val, fmt)


def _chart_risk_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Build a multi-line Plotly figure showing the longitudinal AI risk scores.

    Draws multimodal, image, and tabular scores over visit dates, with
    coloured background bands for LOW / MODERATE / HIGH risk zones and
    star markers indicating the final diagnosis for each visit.

    Args:
        df: DataFrame with columns ``date``, ``multimodal_score``,
            ``image_score``, ``tabular_score``, and ``diagnosis``.

    Returns:
        Plotly ``Figure`` object ready for ``st.plotly_chart``.
    """
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
        fig.add_hline(
            y=y,
            line_dash="dot",
            line_color=color,
            line_width=1,
            opacity=0.5,
        )

    traces = [
        ("multimodal_score", "Multimodal Score", _C["teal"], 3, "circle"),
        ("image_score", "Image Score", _C["blue"], 2, "diamond"),
        ("tabular_score", "Tabular Score", _C["purple"], 2, "square"),
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
                    hovertemplate=(
                        f"<b>{name}</b><br>"
                        "Date: %{x|%d/%m/%Y}<br>"
                        "Score: %{y:.1%}<extra></extra>"
                    ),
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
                        "size": 14,
                        "symbol": "star",
                        "line": {"color": "white", "width": 1.5},
                    },
                    name=f"Dx: {diag.title()}",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>Diagnosis: {diag.title()}</b><br>"
                        "Score: %{y:.1%}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title={
            "text": "Longitudinal AI Risk Evolution",
            "font": {"size": 14, "color": "#2d5a4e"},
        },
        xaxis={
            "title": "Visit Date",
            "tickformat": "%d/%m/%Y",
            "showgrid": False,
        },
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
        height=380,
        margin={"l": 10, "r": 80, "t": 60, "b": 10},
        hovermode="x unified",
    )
    return fig


def _chart_class_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Build a donut chart showing the distribution of image classifications.

    Counts polyp, inflammation, and normal classifications across all visits
    by inspecting the ``ai_snapshot.image.prediction_class`` field.

    Args:
        df: DataFrame with an ``ai_snapshot`` column containing nested dicts.

    Returns:
        Plotly ``Figure`` object ready for ``st.plotly_chart``.
    """
    counts: dict[str, int] = {"polyp": 0, "inflammation": 0, "normal": 0}

    for _, row in df.iterrows():
        snap = row.get("ai_snapshot") or {}
        img = snap.get("image", {})
        cls = img.get("prediction_class", "").lower()
        if cls in counts:
            counts[cls] += 1

    labels = ["Polyp", "Inflammation", "Normal Mucosa"]
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
            "text": "Endoscopic Findings Distribution",
            "font": {"size": 13, "color": "#2d5a4e"},
        },
        showlegend=False,
        height=320,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        paper_bgcolor="white",
        annotations=[
            {
                "text": f"<b>{sum(values)}</b><br>visits",
                "x": 0.5,
                "y": 0.5,
                "font_size": 14,
                "showarrow": False,
                "font_color": "#2d5a4e",
            }
        ],
    )
    return fig


def _chart_probability_heatmap(df: pd.DataFrame) -> go.Figure | None:
    """
    Build a heatmap of per-class AI probabilities across all visit dates.

    Only includes visits that contain probability data in the AI snapshot.
    Returns None when no probability data is available.

    Args:
        df: DataFrame with ``date`` and ``ai_snapshot`` columns.

    Returns:
        Plotly ``Figure`` object, or None if no data is available.
    """
    rows = []
    for _, row in df.iterrows():
        snap = row.get("ai_snapshot") or {}
        img = snap.get("image", {})
        probs = img.get("probabilities", {})
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
    classes = ["Polyp", "Inflammation", "Normal"]
    z = [
        heat_df["polyp"].tolist(),
        heat_df["inflammation"].tolist(),
        heat_df["normal"].tolist(),
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=dates,
            y=classes,
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
            hoverongaps=False,
            hovertemplate=(
                "<b>%{y}</b><br>Visit: %{x}<br>Prob: %{z:.1%}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title={
            "text": "Probability Heatmap per Visit",
            "font": {"size": 13, "color": "#2d5a4e"},
        },
        height=260,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        paper_bgcolor="white",
        xaxis={"tickangle": -30, "tickfont": {"size": 10}},
    )
    return fig


def _chart_biomarkers(df: pd.DataFrame) -> go.Figure | None:
    """
    Build a dual-axis chart showing CEA (bar) and Haemoglobin (line) over time.

    Clinical reference lines are added at CEA = 5 ng/mL and Hb = 12 g/dL.
    Returns None when neither biomarker has data.

    Args:
        df: DataFrame with ``date``, ``cea``, and ``hemoglobin`` columns.

    Returns:
        Plotly ``Figure`` object, or None if no biomarker data is available.
    """
    sub_df = df[["date", "cea", "hemoglobin"]].dropna(subset=["date"])
    sub_df = sub_df[sub_df[["cea", "hemoglobin"]].notna().any(axis=1)]

    if sub_df.empty:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if sub_df["cea"].notna().any():
        fig.add_trace(
            go.Bar(
                x=sub_df["date"],
                y=sub_df["cea"],
                name="CEA (ng/mL)",
                marker_color=[
                    _C["red"] if v and v > 5 else _rgba(_C["teal"], 0.6)
                    for v in sub_df["cea"]
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
            annotation_text="Ref. CEA=5",
            annotation_font_size=9,
            secondary_y=False,
        )

    if sub_df["hemoglobin"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=sub_df["date"],
                y=sub_df["hemoglobin"],
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
            annotation_text="Ref. Hb=12",
            annotation_font_size=9,
            secondary_y=True,
        )

    fig.update_layout(
        title={
            "text": "Analytical Biomarkers (CEA · Haemoglobin)",
            "font": {"size": 13, "color": "#2d5a4e"},
        },
        height=320,
        margin={"l": 10, "r": 60, "t": 50, "b": 10},
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
    """
    Build a stacked area chart of adaptive ensemble weights (alpha / beta) per visit.

    Requires at least two visits with weight data. Returns None otherwise.

    Args:
        df: DataFrame with ``date`` and ``ai_snapshot`` columns. The snapshot
            must contain ``image.alpha_used`` and ``image.beta_used``.

    Returns:
        Plotly ``Figure`` object, or None if fewer than two data points exist.
    """
    rows = []
    for _, row in df.iterrows():
        snap = row.get("ai_snapshot") or {}
        img = snap.get("image", {})
        alpha = img.get("alpha_used")
        beta = img.get("beta_used")
        if alpha is not None and beta is not None:
            rows.append({"date": row["date"], "alpha": alpha, "beta": beta})

    if len(rows) < 2:
        return None

    w_df = pd.DataFrame(rows).sort_values("date")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=w_df["date"],
            y=w_df["alpha"],
            name="α Model A (Context)",
            fill="tozeroy",
            line={"color": _C["blue"], "width": 2},
            fillcolor=_rgba(_C["blue"], 0.15),
            hovertemplate="α = %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=w_df["date"],
            y=w_df["beta"],
            name="β Model B (Tissue)",
            fill="tozeroy",
            line={"color": _C["purple"], "width": 2},
            fillcolor=_rgba(_C["purple"], 0.15),
            hovertemplate="β = %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={
            "text": "Adaptive Ensemble Weights per Visit",
            "font": {"size": 13, "color": "#2d5a4e"},
        },
        height=280,
        yaxis={"title": "Weight", "range": [0, 1], "gridcolor": "#f0f0f0"},
        xaxis={"showgrid": False, "tickformat": "%d/%m/%Y"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
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


def _render_visit_table(df: pd.DataFrame) -> None:
    """
    Render an HTML visit history table with inline risk and diagnosis badges.

    Each row displays visit date, diagnosis, multimodal score, image score,
    colonoscopy flag, and the AI classification result.

    Args:
        df: Filtered visit DataFrame sorted in any order; this function
            re-sorts by date descending before rendering.
    """
    st.markdown(
        "<p style='font-weight:700; color:#2d5a4e; font-size:0.95rem;"
        "margin-bottom:0.5rem;'>📋 Visit History</p>",
        unsafe_allow_html=True,
    )

    header = (
        "<div style='"
        "display:grid; grid-template-columns:90px 110px 130px 130px 110px 120px;"
        "gap:8px; padding:6px 12px;"
        "background:#edf7f4; border-radius:8px 8px 0 0;"
        "font-size:0.78rem; font-weight:700; color:#2d5a4e;'>"
        "<div>Date</div>"
        "<div>Diagnosis</div>"
        "<div>Multimodal Score</div>"
        "<div>Image Score</div>"
        "<div>Colonoscopy</div>"
        "<div>AI Result</div>"
        "</div>"
    )
    st.markdown(header, unsafe_allow_html=True)

    for _, row in df.sort_values("date", ascending=False).iterrows():
        date_str = str(row.get("date", ""))[:10]
        diag = row.get("diagnosis") or ""
        mm_score = row.get("multimodal_score")
        img_score = row.get("image_score")
        colon = "✅ Yes" if row.get("colonoscopy_performed") else "➖ No"

        snap = row.get("ai_snapshot") or {}
        img_snap = snap.get("image", {})
        cls = img_snap.get("prediction_class", "")
        cls_color = _CLASS_COLOR.get(cls.lower(), _C["grey"])
        cls_label = {
            "polyp": "Polyp",
            "inflammation": "Inflammation",
            "normal": "Normal",
        }.get(cls.lower(), cls.title() or "N/A")

        row_bg = (
            _rgba(_C["red"], 0.04)
            if diag == "positive"
            else _rgba(_C["orange"], 0.04)
            if diag == "suspicious"
            else "white"
        )

        row_html = (
            f"<div style='"
            f"display:grid; grid-template-columns:90px 110px 130px 130px 110px 120px;"
            f"gap:8px; padding:7px 12px;"
            f"background:{row_bg}; border-bottom:1px solid #f0f0f0;"
            f"font-size:0.82rem; align-items:center;'>"
            f"<div style='color:#555;'>{date_str}</div>"
            f"<div>{_diag_badge(diag)}</div>"
            f"<div>{_risk_badge(mm_score)}</div>"
            f"<div>{_risk_badge(img_score)}</div>"
            f"<div style='color:#555;'>{colon}</div>"
            f"<div><span style='color:{cls_color}; font-weight:600;"
            f"font-size:0.8rem;'>● {cls_label}</span></div>"
            f"</div>"
        )
        st.markdown(row_html, unsafe_allow_html=True)

    st.markdown(
        "<div style='border-radius:0 0 8px 8px; border:1px solid #edf7f4; "
        "height:4px;'></div>",
        unsafe_allow_html=True,
    )


def _render_risk_factors(df: pd.DataFrame) -> None:
    """
    Aggregate risk factors detected across all visits and display as progress bars.

    Counts occurrences of each named risk factor stored in
    ``ai_snapshot.tabular.top_risk_factors`` and renders a frequency bar for each.
    Bar colour encodes frequency relative to total visits.

    Args:
        df: Filtered visit DataFrame with an ``ai_snapshot`` column.
    """
    factor_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        snap = row.get("ai_snapshot") or {}
        tab = snap.get("tabular", {})
        factors = tab.get("top_risk_factors", [])
        for item in factors:
            name = item[0] if isinstance(item, (list, tuple)) else str(item)
            factor_counts[name] = factor_counts.get(name, 0) + 1

    if not factor_counts:
        st.markdown(
            "<p style='color:#9E9E9E; font-size:0.85rem;'>"
            "No risk factors have been recorded.</p>",
            unsafe_allow_html=True,
        )
        return

    total_visits = len(df)
    sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)

    for name, count in sorted_factors:
        pct = count / total_visits
        color = _C["red"] if pct >= 0.5 else _C["orange"] if pct >= 0.25 else _C["teal"]
        st.markdown(
            f"<div style='margin-bottom:6px;'>"
            f"<div style='display:flex; justify-content:space-between;"
            f"font-size:0.82rem; margin-bottom:2px;'>"
            f"<span style='font-weight:600;'>{name}</span>"
            f"<span style='color:#6c757d;'>{count}/{total_visits} visits</span>"
            f"</div>"
            f"<div style='background:#f0f0f0; border-radius:5px; height:14px; overflow:hidden;'>"
            f"<div style='width:{pct * 100:.0f}%; background:{color}; height:100%; border-radius:5px;'></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )


def _render_kpis(df: pd.DataFrame, patient: dict) -> None:
    """
    Render top-level KPI metric cards for the selected patient.

    Displays total visits, latest multimodal score with trend delta,
    polyp visit count, and colonoscopy count.

    Args:
        df: Full (unfiltered) visit DataFrame sorted by date descending.
        patient: Patient dict as returned by the API, used for context.
    """
    total_visits = len(df)
    last_score = df.iloc[0]["multimodal_score"] if total_visits else None
    polyp_visits = sum(
        1
        for _, r in df.iterrows()
        if (r.get("ai_snapshot") or {}).get("image", {}).get("prediction_class")
        == "polyp"
    )
    colon_visits = (
        df["colonoscopy_performed"].sum() if "colonoscopy_performed" in df else 0
    )

    trend_delta = None
    trend_delta_color = "off"
    if total_visits >= 2:
        prev = df.iloc[1]["multimodal_score"]
        curr = df.iloc[0]["multimodal_score"]
        if prev is not None and curr is not None:
            trend_delta = f"{curr - prev:+.1%}"
            trend_delta_color = "inverse"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📅 Total Visits", total_visits)
    c2.metric(
        "🎯 Latest Multimodal Score",
        f"{last_score:.1%}" if last_score is not None else "N/A",
        delta=trend_delta,
        delta_color=trend_delta_color,
    )
    c3.metric("🔴 Visits with Polyp", polyp_visits)
    c4.metric("🔬 Colonoscopies", int(colon_visits))


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Render filter controls inside an expander and return the filtered DataFrame.

    Filters available: date range, diagnosis result, and colonoscopy presence.

    Args:
        df: Full visit DataFrame to filter.

    Returns:
        Filtered copy of the input DataFrame. May be empty if no rows match.
    """
    with st.expander("🔽 Filters", expanded=False):
        col_f1, col_f2, col_f3 = st.columns(3)

        dates = pd.to_datetime(df["date"]).dt.date
        min_d, max_d = dates.min(), dates.max()
        with col_f1:
            date_range = st.date_input(
                "Date range",
                value=(min_d, max_d),
                min_value=min_d,
                max_value=max_d,
                key="filter_dates",
            )

        diag_opts = ["All"] + sorted(df["diagnosis"].dropna().unique().tolist())
        with col_f2:
            diag_filter = st.selectbox("Diagnosis", diag_opts, key="filter_diag")

        with col_f3:
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

    if colon_filter == "With colonoscopy":
        filtered = filtered[filtered["colonoscopy_performed"]]
    elif colon_filter == "Without colonoscopy":
        filtered = filtered[~filtered["colonoscopy_performed"]]

    return filtered.drop(columns=["_date"])


def render() -> None:
    page_header(
        "Patients & Clinical History",
        "Clinical database and longitudinal AI risk tracking.",
        icon="👥",
    )

    tab_search, tab_new = st.tabs(["📋 Search & Profile", "➕ New Registration"])

    with tab_search:
        _render_search_tab()

    with tab_new:
        _render_new_patient_tab()


def _render_search_tab() -> None:
    """Search & Profile tab — safe to return early without killing tab_new."""
    patients = get_patients()

    if not patients:
        show_empty_state(
            "👥",
            "No patients registered",
            "Use the 'New Registration' tab to add the first patient.",
        )
        return

    opts = {f"{p['first_name']} {p['last_name']}  ·  ID {p['id']}": p for p in patients}
    col_sel, _ = st.columns([1, 3])
    with col_sel:
        selected_key = st.selectbox("Search Patient", list(opts.keys()))
    patient = opts[selected_key]

    st.markdown("<br>", unsafe_allow_html=True)

    age = (date.today() - date.fromisoformat(patient["date_of_birth"])).days // 365

    render_profile_card(
        title="🏥 Patient Record",
        fields=[
            ("Full Name", f"{patient['first_name']} {patient['last_name']}"),
            ("Age", f"{age} years"),
            ("Sex", patient.get("gender", "N/A").title()),
            ("BMI", f"{patient.get('bmi', 0):.1f}"),
            ("Smoking", patient.get("smoking_status", "N/A").title()),
            ("Alcohol", patient.get("alcohol_consumption", "N/A").title()),
        ],
    )

    history = get_patient_history(patient["id"])

    if not history:
        show_empty_state(
            "📊",
            "No history recorded",
            "Diagnoses will appear here as a longitudinal timeline.",
        )
        return

    df_raw = pd.DataFrame(history)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw = df_raw.sort_values("date", ascending=False).reset_index(drop=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _render_kpis(df_raw, patient)

    st.markdown("<br>", unsafe_allow_html=True)
    df = _apply_filters(df_raw)

    if df.empty:
        st.info("ℹ️ No visits match the selected filters.")
        return

    st.markdown("<br>", unsafe_allow_html=True)
    _render_visit_table(df)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-weight:700; color:#2d5a4e; font-size:0.95rem;'>"
        "📊 Visual History Analysis</p>",
        unsafe_allow_html=True,
    )

    if df["multimodal_score"].notna().any() or df["image_score"].notna().any():
        st.plotly_chart(_chart_risk_timeline(df), width="stretch")

    col_donut, col_heat = st.columns([1, 2], gap="large")
    with col_donut:
        st.plotly_chart(_chart_class_distribution(df), width="stretch")
    with col_heat:
        fig_heat = _chart_probability_heatmap(df)
        if fig_heat:
            st.plotly_chart(fig_heat, width="stretch")
        else:
            st.info("No probability data available for the heatmap.")

    col_bio, col_ens = st.columns([3, 2], gap="large")
    with col_bio:
        fig_bio = _chart_biomarkers(df)
        if fig_bio:
            st.plotly_chart(fig_bio, width="stretch")
        else:
            st.info("No biomarker data available (CEA / Haemoglobin).")

    with col_ens:
        fig_ens = _chart_ensemble_weights(df)
        if fig_ens:
            st.plotly_chart(fig_ens, width="stretch")
        else:
            st.info("Insufficient ensemble weight data.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-weight:700; color:#2d5a4e; font-size:0.95rem;'>"
        "⚠️ Cumulative Risk Factors</p>",
        unsafe_allow_html=True,
    )
    _render_risk_factors(df)


def _render_new_patient_tab() -> None:
    """New patient registration tab — completely isolated from tab_search."""
    with st.form("new_patient_form"):
        st.subheader("📝 Patient Data")

        c1, c2 = st.columns(2)
        first_name = c1.text_input("First Name", placeholder="Mary")
        last_name = c2.text_input("Last Name", placeholder="Smith")

        c3, c4 = st.columns(2)
        date_of_birth = c3.date_input(
            "Date of Birth",
            value=date(1980, 1, 1),
            min_value=date(1900, 1, 1),
            max_value=date.today(),
        )
        gender = c4.selectbox("Sex", ["male", "female", "other"])

        c5, c6 = st.columns(2)
        height_cm = c5.number_input("Height (cm)", 100.0, 250.0, 170.0, step=0.5)
        weight_kg = c6.number_input("Weight (kg)", 30.0, 300.0, 70.0, step=0.5)

        if height_cm > 0:
            bmi_preview = weight_kg / ((height_cm / 100) ** 2)
            st.caption(f"📊 Estimated BMI: **{bmi_preview:.1f}**")

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

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("✅ Register Patient", type="primary")

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
                }
            )
            if result:
                st.success(
                    f"✅ Patient **{first_name} {last_name}** "
                    f"registered with ID **{result.get('id', '?')}**."
                )
                st.balloons()
            else:
                st.error("❌ Could not register the patient. Check the backend logs.")


render()
