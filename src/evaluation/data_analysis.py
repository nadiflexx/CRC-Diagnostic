"""
src/evaluation/data_analysis.py
================================
Análisis exploratorio y correlaciones del dataset procesado de CRC.

Genera tres visualizaciones y guarda un CSV limpio:
  1. Heatmap de correlación (triángulo inferior)
  2. Top correlaciones con Mortality y Survival_5_years (barras)
  3. Pairplot de variables clave (muestra 3 000 filas)
  4. Distribución de variables continuas por Mortalidad

Uso:
    python -m src.evaluation.data_analysis

También puede importarse como módulo:
    from src.evaluation.data_analysis import run_data_analysis
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from src.config.logger import get_logger
from src.config.paths import FULL_TABULAR_CSV, ANALYSIS_OUTPUT_DIR

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG  = "#0f1117"
CARD_BG  = "#1a1d27"
TEXT     = "#e8eaf0"
ACCENT   = "#6c8ebf"
RED_COOL = "#e05c5c"
GREEN_OK = "#5cb85c"

_RCPARAMS = {
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   CARD_BG,
    "axes.edgecolor":   "#2e3248",
    "axes.labelcolor":  TEXT,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "text.color":       TEXT,
    "grid.color":       "#2e3248",
    "font.family":      "DejaVu Sans",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER DE CODIFICACIÓN (sobre el CSV crudo, no el procesado)
# ─────────────────────────────────────────────────────────────────────────────


def _encode_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica el CSV crudo original a valores numéricos [0-1] para el análisis
    de correlaciones. No se usa el CSV procesado porque este análisis necesita
    variables como Cancer_Stage, Tumor_Size_mm, etc., que se eliminan en el
    pipeline de entrenamiento.
    """
    df = df.copy()

    binary_yes_no = [
        "Family_History", "Smoking_History", "Alcohol_Consumption", "Diabetes",
        "Inflammatory_Bowel_Disease", "Genetic_Mutation", "Early_Detection",
        "Survival_5_years", "Mortality", "Survival_Prediction",
    ]
    for col in binary_yes_no:
        if col in df.columns:
            df[col] = (df[col] == "Yes").astype(float)

    if "Gender" in df.columns:
        df["Gender"] = (df["Gender"] == "M").astype(float)
    if "Urban_or_Rural" in df.columns:
        df["Urban_or_Rural"] = (df["Urban_or_Rural"] == "Urban").astype(float)
    if "Economic_Classification" in df.columns:
        df["Economic_Classification"] = (df["Economic_Classification"] == "Developed").astype(float)
    if "Insurance_Status" in df.columns:
        df["Insurance_Status"] = (df["Insurance_Status"] == "Insured").astype(float)

    ordinal_maps = {
        "Cancer_Stage":      {"Localized": 0.0, "Regional": 0.5, "Metastatic": 1.0},
        "Obesity_BMI":       {"Normal": 0.0, "Overweight": 0.5, "Obese": 1.0},
        "Diet_Risk":         {"Low": 0.0, "Moderate": 0.5, "High": 1.0},
        "Physical_Activity": {"High": 0.0, "Moderate": 0.5, "Low": 1.0},
        "Screening_History": {"Regular": 0.0, "Irregular": 0.5, "Never": 1.0},
        "Healthcare_Access": {"High": 0.0, "Moderate": 0.5, "Low": 1.0},
    }
    for col, mapping in ordinal_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    nominal_cols = [c for c in ["Country", "Treatment_Type"] if c in df.columns]
    if nominal_cols:
        df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(float)

    # Normalización de continuas
    continuous = [
        c for c in ["Age", "Tumor_Size_mm", "Healthcare_Costs",
                    "Incidence_Rate_per_100K", "Mortality_Rate_per_100K"]
        if c in df.columns
    ]
    if continuous:
        scaler = MinMaxScaler()
        df[continuous] = scaler.fit_transform(df[continuous])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────


def _plot_heatmap(corr: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(20, 16))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(
        corr, mask=mask,
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 7.5, "color": TEXT},
        linewidths=0.4, linecolor="#0f1117",
        square=True, ax=ax,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    )
    ax.set_title("Heatmap de Correlacion — Colorectal Cancer Dataset",
                 fontsize=16, fontweight="bold", color=TEXT, pad=18)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    plt.tight_layout()
    out = output_dir / "heatmap_correlacion.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap guardado → {out}")


def _plot_top_correlations(corr: pd.DataFrame, output_dir: Path) -> None:
    targets = ["Mortality", "Survival_5_years"]
    available = [t for t in targets if t in corr.columns]
    if not available:
        logger.warning("No se encontraron las columnas Mortality / Survival_5_years para el gráfico.")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(18, 8))
    if len(available) == 1:
        axes = [axes]
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Top Correlaciones con Mortalidad y Supervivencia",
                 fontsize=15, fontweight="bold", color=TEXT, y=1.01)

    for ax, target in zip(axes, available):
        ax.set_facecolor(CARD_BG)
        vals = corr[target].drop([t for t in available]).sort_values()
        vals = pd.concat([vals.head(12), vals.tail(12)]).sort_values()
        colors = [RED_COOL if v > 0 else GREEN_OK for v in vals]
        bars = ax.barh(vals.index, vals.values, color=colors, edgecolor="none", height=0.7)
        ax.axvline(0, color="#555", lw=1)
        ax.set_xlabel("Pearson r", fontsize=11)
        ax.set_title(f"<-> {target}", fontsize=13, fontweight="bold", color=TEXT)
        ax.set_xlim(-1, 1)
        for bar, v in zip(bars, vals.values):
            ax.text(
                v + (0.02 if v >= 0 else -0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8, color=TEXT,
            )
        ax.legend(
            handles=[
                mpatches.Patch(color=RED_COOL, label="Correlacion positiva"),
                mpatches.Patch(color=GREEN_OK, label="Correlacion negativa"),
            ],
            fontsize=9, facecolor=CARD_BG, edgecolor="#2e3248", labelcolor=TEXT,
        )

    plt.tight_layout()
    out = output_dir / "top_correlaciones.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    logger.info(f"Top correlaciones guardado → {out}")


def _plot_pairplot(df: pd.DataFrame, output_dir: Path) -> None:
    key_vars = [
        c for c in ["Age", "Tumor_Size_mm", "Cancer_Stage", "Genetic_Mutation",
                    "Smoking_History", "Screening_History", "Mortality"]
        if c in df.columns
    ]
    if "Mortality" not in key_vars:
        logger.warning("Columna Mortality no disponible; se omite el pairplot.")
        return

    sample = df[key_vars].sample(min(3000, len(df)), random_state=42)
    g = sns.pairplot(
        sample, hue="Mortality", diag_kind="kde",
        palette={0: ACCENT, 1: RED_COOL},
        plot_kws={"alpha": 0.25, "s": 12},
        diag_kws={"fill": True, "alpha": 0.5},
    )
    g.figure.patch.set_facecolor(DARK_BG)
    for ax in g.axes.flatten():
        if ax:
            ax.set_facecolor(CARD_BG)
            ax.tick_params(colors=TEXT, labelsize=7)
            ax.xaxis.label.set_color(TEXT)
            ax.yaxis.label.set_color(TEXT)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2e3248")

    g.figure.legend(
        handles=[
            mpatches.Patch(color=ACCENT,   label="Mortality = No"),
            mpatches.Patch(color=RED_COOL, label="Mortality = Yes"),
        ],
        loc="upper right",
        facecolor=CARD_BG, edgecolor="#2e3248", labelcolor=TEXT, fontsize=10,
    )
    g.figure.suptitle("Pairplot — Variables Clave (muestra 3 000)",
                      y=1.01, fontsize=14, fontweight="bold", color=TEXT)
    out = output_dir / "pairplot_clave.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"Pairplot guardado → {out}")


def _plot_continuous_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    cont_plot = [
        c for c in ["Age", "Tumor_Size_mm", "Healthcare_Costs",
                    "Incidence_Rate_per_100K", "Mortality_Rate_per_100K"]
        if c in df.columns
    ]
    if "Mortality" not in df.columns or not cont_plot:
        return

    fig, axes = plt.subplots(1, len(cont_plot), figsize=(22, 5))
    if len(cont_plot) == 1:
        axes = [axes]
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Distribucion de Variables Continuas por Mortalidad",
                 fontsize=14, fontweight="bold", color=TEXT)

    for ax, col in zip(axes, cont_plot):
        ax.set_facecolor(CARD_BG)
        for val, color, label in [(0, ACCENT, "No"), (1, RED_COOL, "Yes")]:
            data = df.loc[df["Mortality"] == val, col]
            ax.hist(data, bins=40, alpha=0.55, color=color,
                    label=f"Mortality={label}", density=True, edgecolor="none")
        ax.set_title(col, fontsize=10, color=TEXT)
        ax.set_xlabel("Valor normalizado", fontsize=8)
        ax.legend(fontsize=8, facecolor=CARD_BG, edgecolor="#2e3248", labelcolor=TEXT)

    plt.tight_layout()
    out = output_dir / "distribucion_continuas.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Distribución continuas guardado → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────


def run_data_analysis(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> None:
    """
    Ejecuta el análisis exploratorio completo y guarda los gráficos.

    Parámetros
    ----------
    input_path : CSV crudo (default: paths.FULL_TABULAR_CSV)
    output_dir : directorio de salida (default: paths.ANALYSIS_OUTPUT_DIR)
    """
    src = Path(input_path) if input_path else FULL_TABULAR_CSV
    dest = Path(output_dir) if output_dir else ANALYSIS_OUTPUT_DIR
    dest.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(_RCPARAMS)

    logger.info(f"Cargando dataset desde {src}")
    df = pd.read_csv(src)
    logger.info(f"Shape cargado: {df.shape}")

    df_enc = _encode_raw(df)

    # Guardar CSV limpio
    clean_path = dest / "colorectal_cancer_clean.csv"
    df_enc.to_csv(clean_path, index=False)
    logger.info(f"Dataset limpio guardado → {clean_path}")

    # Correlaciones (solo columnas core, sin one-hots de País/Tratamiento)
    core_cols = [c for c in df_enc.columns
                 if not c.startswith("Country_") and not c.startswith("Treatment_Type_")]
    corr = df_enc[core_cols].corr()

    _plot_heatmap(corr, dest)
    _plot_top_correlations(corr, dest)
    _plot_pairplot(df_enc, dest)
    _plot_continuous_distributions(df_enc, dest)

    logger.info(f"Análisis completado. Resultados en '{dest}/'")


if __name__ == "__main__":
    run_data_analysis()
