"""
src/evaluation/tabular_plots.py
================================
Gráficos de evaluación del modelo tabular MLP.

Genera un panel de 6 subplots:
  A · Matriz de Confusión
  B · Curva ROC con punto del umbral elegido
  C · Curva Precision-Recall
  D · Distribución de probabilidades por clase
  E · Métricas vs Umbral (threshold sweep en validación)
  F · Feature Importance (permutation importance sobre Recall)

Utilizado por: train_tabular.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.config.constants import (
    RANDOM_SEED,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
    TRAIN_TEST_SIZE,
    TRAIN_VAL_SIZE,
)
from src.config.logger import get_logger
from src.models.tabular_model import TabularModel

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PALETA DE COLORES
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "primary": "#2563EB",
    "secondary": "#DC2626",
    "accent": "#16A34A",
    "warn": "#D97706",
    "bg": "#F8FAFC",
    "grid": "#E2E8F0",
    "text": "#1E293B",
}

_RCPARAMS = {
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor": PALETTE["bg"],
    "axes.edgecolor": PALETTE["grid"],
    "axes.labelcolor": PALETTE["text"],
    "xtick.color": PALETTE["text"],
    "ytick.color": PALETTE["text"],
    "grid.color": PALETTE["grid"],
    "text.color": PALETTE["text"],
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
}


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────


def generate_evaluation_plots(
    model: TabularModel,
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Genera el panel completo de evaluación y lo guarda como PNG.

    Parámetros
    ----------
    model      : TabularModel ya entrenado
    df         : DataFrame completo (con columna 'Diagnosis')
    output_dir : directorio de salida (default: paths.TABULAR_PLOTS_DIR)

    Retorna
    -------
    Path del archivo PNG generado
    """
    assert model.model is not None, "El modelo no ha sido entrenado."
    assert model.metrics is not None, "No hay métricas disponibles."

    dest = Path(output_dir) if output_dir else Path("data/outputs/tabular")
    dest.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(_RCPARAMS)

    # ── Re-construir particiones (idéntico a TabularModel.fit) ───────────────
    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]

    val_ratio = TRAIN_VAL_SIZE / (1 - TRAIN_TEST_SIZE)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=RANDOM_SEED
    )

    X_val_s = model.scaler.transform(X_val)
    X_test_s = model.scaler.transform(X_test)
    X_train_s = model.scaler.transform(X_train)

    y_prob_test = model.model.predict_proba(X_test_s)[:, 1]
    y_pred_test = (y_prob_test >= model.threshold).astype(int)
    y_prob_val = model.model.predict_proba(X_val_s)[:, 1]

    m = model.metrics
    feature_names = model.feature_names

    # ── Threshold sweep (para panel E) ──────────────────────────────────────
    from sklearn.metrics import f1_score, precision_score, recall_score

    results_thresh = []
    for t in np.arange(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP):
        y_pred_t = (y_prob_val >= t).astype(int)
        results_thresh.append({
            "threshold": round(t, 2),
            "recall": recall_score(y_val, y_pred_t, zero_division=0),
            "precision": precision_score(y_val, y_pred_t, zero_division=0),
            "f1": f1_score(y_val, y_pred_t, zero_division=0),
        })
    df_thr = pd.DataFrame(results_thresh)

    # ── Permutation Importance (para panel F) ────────────────────────────────
    logger.info("Calculando permutation importance…")
    perm = permutation_importance(
        model.model,
        X_test_s,
        y_test,
        n_repeats=5,
        random_state=RANDOM_SEED,
        scoring="recall",
        n_jobs=-1,
    )
    feat_imp = (
        pd.DataFrame({
            "feature": feature_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    # ── Construcción del panel ───────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 22))
    fig.suptitle(
        "Colorectal Cancer Prediction — Red Neuronal\n"
        f"Arquitectura: {len(feature_names)} → 256 → 128 → 64 → 1"
        f"   |   Umbral: {model.threshold}   |   Test n={len(y_test):,}",
        fontsize=15,
        fontweight="bold",
        color=PALETTE["text"],
        y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    # A · Matriz de Confusión
    ax_cm = fig.add_subplot(gs[0, 0])
    cmd = ConfusionMatrixDisplay(
        confusion_matrix=m.confusion_matrix,
        display_labels=["Sin cáncer", "Cáncer"],
    )
    cmd.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title("A · Matriz de Confusión")
    ax_cm.set_xlabel("Predicción")
    ax_cm.set_ylabel("Real")

    # B · Curva ROC
    ax_roc = fig.add_subplot(gs[0, 1])
    RocCurveDisplay.from_predictions(
        y_test, y_prob_test,
        ax=ax_roc,
        color=PALETTE["primary"],
        name=f"MLP (AUC={m.roc_auc:.3f})",
    )
    ax_roc.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Random")
    fpr_c, tpr_c, thresh_c = roc_curve(y_test, y_prob_test)
    idx = np.argmin(np.abs(thresh_c - model.threshold))
    ax_roc.scatter(
        fpr_c[idx], tpr_c[idx], s=120, zorder=5,
        color=PALETTE["secondary"], label=f"Umbral {model.threshold}",
    )
    ax_roc.set_title("B · Curva ROC")
    ax_roc.legend(fontsize=9)
    ax_roc.grid(True, alpha=0.4)

    # C · Curva Precision-Recall
    ax_pr = fig.add_subplot(gs[0, 2])
    PrecisionRecallDisplay.from_predictions(
        y_test, y_prob_test,
        ax=ax_pr,
        color=PALETTE["accent"],
        name=f"MLP (PR-AUC={m.pr_auc:.3f})",
    )
    ax_pr.axhline(y=m.precision, color=PALETTE["secondary"], linestyle="--", alpha=0.7,
                  label=f"Precision actual ({m.precision:.3f})")
    ax_pr.axvline(x=m.recall, color=PALETTE["warn"], linestyle="--", alpha=0.7,
                  label=f"Recall actual ({m.recall:.3f})")
    ax_pr.set_title("C · Curva Precision-Recall")
    ax_pr.legend(fontsize=9)
    ax_pr.grid(True, alpha=0.4)

    # D · Distribución de probabilidades
    ax_dist = fig.add_subplot(gs[1, :2])
    prob0 = y_prob_test[y_test == 0]
    prob1 = y_prob_test[y_test == 1]
    ax_dist.hist(prob0, bins=60, alpha=0.65, color=PALETTE["accent"],
                 label="Sin cáncer (0)", density=True)
    ax_dist.hist(prob1, bins=60, alpha=0.65, color=PALETTE["secondary"],
                 label="Cáncer (1)", density=True)
    ax_dist.axvline(x=model.threshold, color=PALETTE["primary"], linewidth=2,
                    linestyle="--", label=f"Umbral = {model.threshold}")
    ax_dist.set_xlabel("Probabilidad predicha P(cáncer)")
    ax_dist.set_ylabel("Densidad")
    ax_dist.set_title("D · Distribución de Probabilidades por Clase")
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.4)

    # E · Threshold sweep
    ax_thr = fig.add_subplot(gs[1, 2])
    ax_thr.plot(df_thr["threshold"], df_thr["recall"],
                color=PALETTE["secondary"], linewidth=2, label="Recall")
    ax_thr.plot(df_thr["threshold"], df_thr["precision"],
                color=PALETTE["primary"], linewidth=2, label="Precision")
    ax_thr.plot(df_thr["threshold"], df_thr["f1"],
                color=PALETTE["accent"], linewidth=2, label="F1")
    ax_thr.axvline(x=model.threshold, color="gray", linestyle="--", linewidth=1.5,
                   label=f"Elegido: {model.threshold}")
    ax_thr.set_xlabel("Umbral")
    ax_thr.set_ylabel("Score")
    ax_thr.set_title("E · Métricas vs Umbral (Validación)")
    ax_thr.legend(fontsize=9)
    ax_thr.grid(True, alpha=0.4)

    # F · Feature Importance
    ax_fi = fig.add_subplot(gs[2, :])
    top_n = 15
    fi_top = feat_imp.head(top_n)
    colors_fi = [PALETTE["secondary"] if i < 3 else PALETTE["primary"] for i in range(top_n)]
    ax_fi.barh(
        fi_top["feature"][::-1],
        fi_top["importance_mean"][::-1],
        xerr=fi_top["importance_std"][::-1],
        color=colors_fi[::-1],
        edgecolor="white",
        height=0.7,
        error_kw={"elinewidth": 1.2, "capsize": 3, "alpha": 0.7},
    )
    ax_fi.set_xlabel("Reducción de Recall (importancia por permutación)")
    ax_fi.set_title(f"F · Top-{top_n} Features — Impacto en Recall")
    ax_fi.grid(True, alpha=0.4, axis="x")
    ax_fi.legend(
        handles=[
            Patch(facecolor=PALETTE["secondary"], label="Top 3 features"),
            Patch(facecolor=PALETTE["primary"], label=f"Resto del top-{top_n}"),
        ],
        fontsize=9,
        loc="lower right",
    )

    # Panel resumen de métricas
    metrics_text = (
        f"  RECALL        {m.recall:.4f}  ★\n"
        f"  PRECISION     {m.precision:.4f}\n"
        f"  F1-SCORE      {m.f1:.4f}\n"
        f"  SPECIFICITY   {m.specificity:.4f}\n"
        f"  NPV           {m.npv:.4f}\n"
        f"  ROC-AUC       {m.roc_auc:.4f}\n"
        f"  PR-AUC        {m.pr_auc:.4f}\n"
        f"  CV Recall     {m.cv_mean:.4f}±{m.cv_std:.4f}\n"
        f"  Umbral        {m.threshold}\n"
        f"  TP / FN       {m.tp:,} / {m.fn:,}"
    )
    fig.text(
        0.77, 0.36, metrics_text,
        fontsize=10.5, family="monospace",
        verticalalignment="top", horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="white",
            edgecolor=PALETTE["primary"],
            linewidth=1.5,
        ),
        color=PALETTE["text"],
    )
    fig.text(0.77, 0.65, "RESUMEN\nMÉTRICAS",
             fontsize=12, fontweight="bold", color=PALETTE["primary"],
             ha="left", va="top")

    out_path = dest / "nn_cancer_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    logger.info(f"Gráfico guardado: {out_path}")
    return out_path
