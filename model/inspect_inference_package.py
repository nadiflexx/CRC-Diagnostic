"""inspect_inference_package.py

Verifica la integridad y el rendimiento del paquete de inferencia exportado
(``xgb_tumoral_model_package.pkl``) antes de desplegarlo en producción.

Flujo:
    1. Carga el paquete PKL y extrae modelo, umbral y nombres de features.
    2. Reconstruye el mismo split test que usó el entrenamiento (stratify, seed=42).
    3. Calcula métricas sobre el test set y las guarda en ``inspection_metrics.json``.
    4. Genera los gráficos de evaluación, importancias y SHAP.

Uso:
    python model/inspect_inference_package.py
"""

import json
import importlib
import os
import pprint

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

if importlib.util.find_spec("shap") is not None:
    shap = importlib.import_module("shap")
    HAS_SHAP = True
else:
    shap = None
    HAS_SHAP = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, "artifacts", "xgb_tumoral_model_package.pkl")
CSV_PATH = os.path.join(BASE_DIR, "..", "Data", "processed", "dataset_clinico_tumoral.csv")
OUT_DIR = os.path.join(BASE_DIR, "artifacts", "inspection_plots")


def cargar_paquete(pkl_path: str) -> dict:
    """Carga el paquete de inferencia serializado desde disco.

    Args:
        pkl_path: Ruta al archivo ``.pkl`` generado por ``xgb_clinical_model.py``.

    Returns:
        Diccionario con las claves ``model``, ``threshold`` y ``feature_names``.

    Raises:
        SystemExit: Si el archivo no existe.
        ValueError: Si el contenido no es un diccionario válido.
    """
    if not os.path.exists(pkl_path):
        print("NO_PKL", pkl_path)
        raise SystemExit(1)
    # Importar ModeloCalibraado ANTES de joblib.load para que pueda deserializar el PKL
    import sys as _sys
    _model_dir = os.path.dirname(os.path.abspath(pkl_path + "/../.."))
    _this_dir  = os.path.dirname(os.path.abspath(__file__))
    if _this_dir not in _sys.path:
        _sys.path.insert(0, _this_dir)
    importlib.import_module("calibration")
    pkg = joblib.load(pkl_path)
    if not isinstance(pkg, dict):
        raise ValueError("El archivo PKL no contiene un diccionario valido")
    return pkg


def reconstruir_test_set(csv_path: str, feature_names: list):
    """Reconstruye el conjunto de test con el mismo split usado en el entrenamiento.

    Replica el ``train_test_split`` estratificado con ``test_size=0.20`` y
    ``random_state=42`` para garantizar que las métricas se calculan sobre
    exactamente los mismos pacientes que el modelo nunca ha visto.

    Args:
        csv_path: Ruta al CSV del dataset clínico-tumoral procesado.
        feature_names: Lista de nombres de features en el orden del modelo.
            Si está vacía, se usan todas las columnas numéricas disponibles.

    Returns:
        Tupla (X_train, X_test, y_train, y_test) como DataFrames y Series.

    Raises:
        FileNotFoundError: Si el CSV no existe en la ruta indicada.
        ValueError: Si alguna feature del paquete no está en el CSV.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No existe CSV para evaluar: {csv_path}")

    df = pd.read_csv(csv_path)
    X = df.drop(columns=[c for c in ["Patient_ID", "Diagnosis"] if c in df.columns])
    y = df["Diagnosis"]
    X = X.select_dtypes(include=[np.number])

    if feature_names:
        faltantes = [c for c in feature_names if c not in X.columns]
        if faltantes:
            raise ValueError(
                "Faltan features del paquete en el CSV: " + ", ".join(faltantes)
            )
        X = X[feature_names]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


def generar_metricas(y_test, y_prob, threshold: float) -> dict:
    """Calcula el conjunto completo de métricas de clasificación binaria.

    Args:
        y_test: Array-like de etiquetas reales (0/1).
        y_prob: Array-like de probabilidades predichas para la clase positiva.
        threshold: Umbral de decisión aplicado sobre ``y_prob``.

    Returns:
        Diccionario con las claves: ``threshold``, ``roc_auc``,
        ``average_precision``, ``recall``, ``precision``, ``f1``,
        ``specificity`` y ``confusion_matrix``.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def plot_panel_evaluacion(y_test, y_prob, threshold: float, out_dir: str) -> None:
    """Genera y guarda el panel de evaluación 2×2 como PNG.

    El panel incluye: matriz de confusión, curva ROC, curva
    precisión-recall y distribución de probabilidades por clase.

    Args:
        y_test: Array-like de etiquetas reales (0/1).
        y_prob: Array-like de probabilidades predichas para la clase positiva.
        threshold: Umbral de decisión utilizado para binarizar las predicciones.
        out_dir: Directorio donde se guardará ``panel_evaluacion_inferencia.png``.

    Returns:
        None
    """
    os.makedirs(out_dir, exist_ok=True)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Evaluacion de inferencia (umbral={threshold:.2f})", fontsize=14)

    ax = axes[0, 0]
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Matriz de confusion")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Sano", "Cancer"])
    ax.set_yticklabels(["Sano", "Cancer"])
    fig.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax = axes[0, 1]
    ax.plot(fpr, tpr, lw=2, color="darkorange", label=f"ROC-AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="navy")
    ax.set_title("Curva ROC")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.grid(alpha=0.3)
    ax.legend()

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax = axes[1, 0]
    ax.plot(rec, prec, lw=2, color="green", label=f"AP={ap:.3f}")
    ax.set_title("Curva Precision-Recall")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.hist(y_prob[y_test == 0], bins=30, alpha=0.6, color="steelblue", label="Sano")
    ax.hist(y_prob[y_test == 1], bins=30, alpha=0.6, color="tomato", label="Cancer")
    ax.axvline(threshold, color="black", linestyle="--", lw=2, label=f"Umbral={threshold:.2f}")
    ax.set_title("Distribucion de probabilidades")
    ax.set_xlabel("P(Cancer)")
    ax.set_ylabel("Frecuencia")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "panel_evaluacion_inferencia.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_importancias_modelo(model, feature_names: list, out_dir: str) -> None:
    """Guarda un barplot horizontal con el top-20 de features por importancia.

    Usa ``feature_importances_`` de XGBoost (ganancia media en splits).
    Si el modelo no expone ese atributo, la función retorna sin error.

    Args:
        model: Clasificador ``XGBClassifier`` ajustado.
        feature_names: Lista de nombres de features en el orden del modelo.
        out_dir: Directorio donde se guardará ``feature_importances_top20.png``.

    Returns:
        None
    """
    os.makedirs(out_dir, exist_ok=True)
    if not hasattr(model, "feature_importances_"):
        return

    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    top_k = min(20, len(order))
    idx = order[:top_k]

    labels = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]
    vals = imp[idx]

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_k), vals[::-1], color="teal")
    plt.yticks(range(top_k), labels[::-1])
    plt.xlabel("Importancia")
    plt.title("Top features por feature_importances_")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importances_top20.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_si_disponible(model, X_test: pd.DataFrame, feature_names: list, out_dir: str) -> None:
    """Genera gráficos SHAP globales si la librería está disponible.

    Produce un beeswarm y un barplot sobre una submuestra de hasta 2 000 pacientes.
    Si ``shap`` no está instalado, imprime un aviso y retorna sin error.

    Args:
        model: Clasificador ``XGBClassifier`` ajustado.
        X_test: DataFrame de features del conjunto de test.
        feature_names: Lista de nombres de features en el orden del modelo.
        out_dir: Directorio donde se guardarán ``shap_beeswarm.png`` y ``shap_bar.png``.

    Returns:
        None
    """
    if not HAS_SHAP:
        print("SHAP no disponible: se omiten graficos SHAP")
        return

    os.makedirs(out_dir, exist_ok=True)
    n = min(2000, len(X_test))
    idx = np.random.choice(len(X_test), size=n, replace=False)
    X_sub = X_test.iloc[idx].copy()
    if feature_names and len(feature_names) == X_sub.shape[1]:
        X_sub.columns = feature_names

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)
    sv = shap_values if not isinstance(shap_values, list) else shap_values[1]

    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_sub, feature_names=list(X_sub.columns), show=False)
    plt.title("SHAP beeswarm")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, X_sub, feature_names=list(X_sub.columns), plot_type="bar", show=False)
    plt.title("SHAP importancia media")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Punto de entrada: orquesta la inspección completa del paquete de inferencia.

    Carga el PKL, reconstruye el test set, calcula métricas, las serializa
    en JSON y genera los gráficos de evaluación e interpretabilidad.

    Returns:
        None
    """
    pkg = cargar_paquete(PKL_PATH)
    model     = pkg.get("model")       # ModeloCalibraado — para predict_proba calibrado
    model_raw = pkg.get("model_raw", model)  # XGBClassifier puro — para get_params, importancias, SHAP
    threshold = float(pkg.get("threshold", 0.5))
    feature_names = pkg.get("feature_names") or []

    if model is None:
        raise ValueError("El paquete no contiene la clave 'model'")

    X_train, X_test, y_train, y_test = reconstruir_test_set(CSV_PATH, feature_names)
    y_prob = model.predict_proba(X_test)[:, 1]
    metricas = generar_metricas(y_test, y_prob, threshold)

    info = {
        "threshold": threshold,
        "train_shape": (int(X_train.shape[0]), int(X_train.shape[1])),
        "test_shape": (int(X_test.shape[0]), int(X_test.shape[1])),
        "metrics_test_recomputed": metricas,
        "model_params": {
            "n_estimators": model_raw.get_params().get("n_estimators"),
            "max_depth": model_raw.get_params().get("max_depth"),
            "learning_rate": model_raw.get_params().get("learning_rate"),
        },
    }
    pprint.pprint(info)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "inspection_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    plot_panel_evaluacion(y_test, y_prob, threshold, OUT_DIR)
    plot_importancias_modelo(model_raw, feature_names or list(X_test.columns), OUT_DIR)
    plot_shap_si_disponible(model_raw, X_test, feature_names or list(X_test.columns), OUT_DIR)

    print("\nGraficos y metricas guardados en:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
