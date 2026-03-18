"""
Plotly chart components for Endo-AID.
"""

import pandas as pd
import plotly.graph_objects as go

EMERALD = "#067a5f"
EMERALD_LIGHT = "rgba(5, 150, 105, 0.1)"


def plot_risk_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Creates a professional risk evolution line chart.

    Args:
        df: DataFrame with columns 'date' and 'multimodal_score'.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%d/%m/%Y")

    fig = go.Figure()

    # Risk zones (background bands)
    fig.add_hrect(y0=0, y1=0.3, fillcolor="#DCFCE7", opacity=0.4, line_width=0)
    fig.add_hrect(y0=0.3, y1=0.5, fillcolor="#FEF3C7", opacity=0.4, line_width=0)
    fig.add_hrect(y0=0.5, y1=1.0, fillcolor="#FECACA", opacity=0.3, line_width=0)

    # Main line
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["multimodal_score"],
            mode="lines+markers",
            name="Risk Score",
            line={"color": EMERALD, "width": 3, "shape": "spline"},
            marker={
                "size": 9,
                "color": "white",
                "line": {"width": 2.5, "color": EMERALD},
            },
            fill="tozeroy",
            fillcolor=EMERALD_LIGHT,
            hovertemplate="<b>%{x}</b><br>Risk: %{y:.2%}<extra></extra>",
        )
    )

    # Threshold line
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="#EF4444",
        line_width=1.5,
        annotation_text="High Risk Threshold",
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color="#EF4444",
    )

    fig.update_layout(
        height=280,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        yaxis={
            "range": [0, 1],
            "tickformat": ".0%",
            "showgrid": True,
            "gridcolor": "#F1F5F9",
            "zeroline": False,
        },
        xaxis={"showgrid": False},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hovermode="x unified",
    )

    return fig
