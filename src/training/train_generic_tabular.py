"""
Entrenamiento del Modelo de Predicción de Cáncer Colorrectal
Modelo: Red Neuronal Multicapa (MLP)
=================================================================
Este módulo entrena y evalúa un modelo MLP para predicción de cáncer
colorrectal basado en características tabulares genéricas.

Uso:
    python train_generic_tabular.py
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score,
    ConfusionMatrixDisplay, RocCurveDisplay,
    PrecisionRecallDisplay, roc_curve
)
from sklearn.inspection import permutation_importance

# Agregar src al path para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.config.paths import paths
from src.config.generic_tabular_features import (
    INPUT_CSV_TABULAR_PROCESSED,
)
from src.evaluation.explainability import GenericTabularExplainer

matplotlib.use('Agg')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
SEED        = 42
TEST_SIZE   = 0.30
VAL_SIZE    = 0.10

# Usar rutas del sistema centralizado
MODEL_DIR = str(paths.GENERIC_TABULAR_MODEL_DIR)
MODEL_PATH = str(paths.GENERIC_TABULAR_MODEL_PATH)
SCALER_PATH = str(paths.GENERIC_TABULAR_SCALER_PATH)
CONFIG_PATH = str(paths.GENERIC_TABULAR_CONFIG_PATH)
ANALYSIS_DIR = str(paths.ANALYSIS)

np.random.seed(SEED)


def load_and_prepare_data():
    """Carga el dataset y separa features del target."""
    print("=" * 60)
    print("  COLORECTAL CANCER — RED NEURONAL")
    print("=" * 60)
    
    df = pd.read_csv(INPUT_CSV_TABULAR_PROCESSED)
    print(f"\n[DATA] Registros: {len(df):,}  |  Features: {df.shape[1]-1}")
    print(f"[DATA] Distribución Diagnosis → {df['Diagnosis'].value_counts().to_dict()}")
    
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']
    feature_names = X.columns.tolist()
    
    return X, y, feature_names


def split_data(X, y):
    """Divide los datos en train, validación y test con estratificación."""
    # Primero separamos test (30%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    # Luego validación (10% del total = 12.5% del temp restante)
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=SEED
    )
    
    print(f"\n[SPLIT] Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, val_ratio





def scale_data(X_train, X_val, X_test):
    """Escala los datos usando StandardScaler (fit solo en train)."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    
    return X_train_s, X_val_s, X_test_s, scaler


def create_and_train_model(X_train_s, X_val_s, y_train, y_val, feature_names, val_ratio):
    """Crea y entrena el modelo MLP."""
    mlp_final = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        learning_rate='adaptive',
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=val_ratio,
        n_iter_no_change=20,
        random_state=SEED,
        verbose=False,
    )
    
    print(f"\n[MODEL] Arquitectura: Input({len(feature_names)}) → 256 → 128 → 64 → Output(1)  [{len(feature_names)} features]")
    print("[MODEL] Entrenando... (puede tardar ~1-2 min)")
    
    t0 = time.time()
    X_fulltrain = np.vstack([X_train_s, X_val_s])
    y_fulltrain = np.concatenate([y_train, y_val])
    mlp_final.fit(X_fulltrain, y_fulltrain)
    elapsed = time.time() - t0
    
    print(f"[MODEL] Entrenamiento completado en {elapsed:.1f}s")
    print(f"[MODEL] Épocas ejecutadas: {mlp_final.n_iter_}")
    print(f"[MODEL] Loss final: {mlp_final.loss_:.4f}")
    
    return mlp_final


def find_optimal_threshold(model, X_val_s, y_val):
    """Busca el umbral óptimo priorizando Recall >= 0.90."""
    y_prob_val = model.predict_proba(X_val_s)[:, 1]
    
    best_thresh  = 0.50
    best_recall  = 0.0
    best_f1      = 0.0
    results_thresh = []
    
    for t in np.arange(0.20, 0.75, 0.01):
        y_pred_t = (y_prob_val >= t).astype(int)
        recall_t = recall_score(y_val, y_pred_t, zero_division=0)
        prec_t   = precision_score(y_val, y_pred_t, zero_division=0)
        f1_t     = f1_score(y_val, y_pred_t, zero_division=0)
        
        results_thresh.append({
            'threshold': t,
            'recall': recall_t,
            'precision': prec_t,
            'f1': f1_t,
        })
        
        # Actualizar si el recall está bueno y el f1 mejora
        if recall_t >= 0.88 and f1_t > best_f1:
            best_thresh = t
            best_recall = recall_t
            best_f1 = f1_t
    
    print(f"\n[THRESHOLD] Umbral óptimo seleccionado: {best_thresh}")
    print(f"[THRESHOLD] Recall en validación: {best_recall:.4f} | F1: {best_f1:.4f}")
    
    return best_thresh, results_thresh


def evaluate_model(model, X_test_s, y_test, best_thresh):
    """Evalúa el modelo en el conjunto test."""
    y_prob_test = model.predict_proba(X_test_s)[:, 1]
    y_pred_test = (y_prob_test >= best_thresh).astype(int)
    
    recall    = recall_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    f1        = f1_score(y_test, y_pred_test)
    roc_auc   = roc_auc_score(y_test, y_prob_test)
    pr_auc    = average_precision_score(y_test, y_prob_test)
    cm        = confusion_matrix(y_test, y_pred_test)
    
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print("\n" + "=" * 60)
    print("  MÉTRICAS FINALES — CONJUNTO TEST")
    print("=" * 60)
    print(f"\n  ★ RECALL      (Sensibilidad) : {recall:.4f}  ← MÉTRICA PRINCIPAL")
    print(f"    PRECISION                  : {precision:.4f}")
    print(f"    F1-SCORE                   : {f1:.4f}")
    print(f"    SPECIFICITY  (Especif.)    : {specificity:.4f}")
    print(f"    NPV          (Val.Pred.Neg): {npv:.4f}")
    print(f"    ROC-AUC                    : {roc_auc:.4f}")
    print(f"    PR-AUC                     : {pr_auc:.4f}")
    print(f"\n  Umbral de decisión           : {best_thresh}")
    print(f"\n  Matriz de Confusión:")
    print(f"    TP (cáncer correcto)  : {tp:>7,}")
    print(f"    FP (falso positivo)   : {fp:>7,}")
    print(f"    TN (sano correcto)    : {tn:>7,}")
    print(f"    FN (cáncer no detect.): {fn:>7,}  ← minimizar")
    
    print("\n" + "-" * 60)
    print(classification_report(y_test, y_pred_test,
                                target_names=['Sin cáncer (0)', 'Cáncer (1)']))
    
    metrics = {
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'specificity': specificity,
        'npv': npv,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'cm': cm,
        'y_prob_test': y_prob_test,
        'y_pred_test': y_pred_test,
    }
    
    return metrics


def cross_validation(model, X_train_s, y_train):
    """Realiza validación cruzada 5-fold."""
    print("\n[CV] Validación cruzada 5-fold (recall)...")
    cv_model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        learning_rate='adaptive',
        learning_rate_init=1e-3,
        max_iter=100,
        early_stopping=False,
        random_state=SEED,
        verbose=False,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(cv_model, X_train_s, y_train,
                                 cv=skf, scoring='recall', n_jobs=-1)
    print(f"[CV] Recall por fold: {[round(s, 4) for s in cv_scores]}")
    print(f"[CV] Recall medio: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return cv_scores


def compute_feature_importance(model, X_test_s, y_test, feature_names):
    """Calcula la importancia de features por permutación."""
    print("\n[IMPORTANCE] Calculando importancia de features...")
    perm = permutation_importance(
        model, X_test_s, y_test,
        n_repeats=5, random_state=SEED, scoring='recall', n_jobs=-1
    )
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm.importances_mean,
        'importance_std':  perm.importances_std
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
    
    print("\n[IMPORTANCE] Top-10 features por impacto en Recall:")
    print(feat_imp.head(10).to_string(index=False))
    
    return feat_imp


def generate_explanations(model, X_test_s, y_test, feature_names, X_train_s):
    """Crea y aplica el explainer tabular genérico."""
    print("\n[EXPLAINABILITY] Inicializando GenericTabularExplainer...")
    
    explainer = GenericTabularExplainer(model, feature_names, X_background=X_train_s)
    explainer.fit(X_train_s)
    
    # Generar explicación individual para una muestra CRC y una sin cáncer
    indices_crc = np.where(y_test == 1)[0]
    indices_healthy = np.where(y_test == 0)[0]
    
    explanations = {
        'explainer': explainer,
        'index_crc': indices_crc[0] if len(indices_crc) > 0 else None,
        'index_healthy': indices_healthy[0] if len(indices_healthy) > 0 else None,
    }
    
    if explanations['index_crc'] is not None:
        print(f"[EXPLAINABILITY] Generando explicación para muestra CRC (índice {explanations['index_crc']})")
        sample_crc = X_test_s[explanations['index_crc']]
        exp_crc = explainer.explain(sample_crc)
        explanations['exp_crc'] = exp_crc
        explanations['sample_crc'] = sample_crc
        print(f"  → Probabilidad predicha: {exp_crc['prediction_proba']:.2%}")
        print(f"  → Método: {exp_crc['method']}")
        
    if explanations['index_healthy'] is not None:
        print(f"[EXPLAINABILITY] Generando explicación para muestra Healthy (índice {explanations['index_healthy']})")
        sample_healthy = X_test_s[explanations['index_healthy']]
        exp_healthy = explainer.explain(sample_healthy)
        explanations['exp_healthy'] = exp_healthy
        explanations['sample_healthy'] = sample_healthy
        print(f"  → Probabilidad predicha: {exp_healthy['prediction_proba']:.2%}")
        print(f"  → Método: {exp_healthy['method']}")
    
    return explanations


def plot_explanations(explanations, analysis_dir):
    """Genera gráficas de explicabilidad."""
    print("\n[EXPLAINABILITY] Generando gráficas de explicaciones...")
    
    explainer = explanations['explainer']
    
    # Gráfica para muestra CRC
    if 'sample_crc' in explanations and 'exp_crc' in explanations:
        out_path_crc = os.path.join(analysis_dir, 'explanation_crc_sample.png')
        explainer.plot_explanation(
            explanations['sample_crc'].reshape(1, -1),
            y_pred_proba=np.array([[1 - explanations['exp_crc']['prediction_proba'], 
                                   explanations['exp_crc']['prediction_proba']]]),
            save_path=out_path_crc,
            figsize=(12, 8)
        )
        print(f"  ✓ Explicación CRC: {out_path_crc}")
    
    # Gráfica para muestra Healthy
    if 'sample_healthy' in explanations and 'exp_healthy' in explanations:
        out_path_healthy = os.path.join(analysis_dir, 'explanation_healthy_sample.png')
        explainer.plot_explanation(
            explanations['sample_healthy'].reshape(1, -1),
            y_pred_proba=np.array([[1 - explanations['exp_healthy']['prediction_proba'], 
                                   explanations['exp_healthy']['prediction_proba']]]),
            save_path=out_path_healthy,
            figsize=(12, 8)
        )
        print(f"  ✓ Explicación Healthy: {out_path_healthy}")
    
    print("[EXPLAINABILITY] Gráficas de explicación guardadas")


def save_model_and_artifacts(model, scaler, config_data):
    """Guarda el modelo, scaler y configuración."""
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(CONFIG_PATH, 'wb') as f:
        pickle.dump(config_data, f)
    
    print(f"\n[SAVED] Modelo: {MODEL_PATH}")
    print(f"[SAVED] Scaler: {SCALER_PATH}")
    print(f"[SAVED] Config: {CONFIG_PATH}")


def plot_results(y_test, y_prob_test, y_pred_test, metrics, results_thresh, 
                 feat_imp, feature_names, best_thresh, cv_scores):
    """Genera las gráficas de evaluación del modelo."""
    PALETTE = {
        'primary':   '#2563EB',
        'secondary': '#DC2626',
        'accent':    '#16A34A',
        'warn':      '#D97706',
        'bg':        '#F8FAFC',
        'grid':      '#E2E8F0',
        'text':      '#1E293B',
    }
    
    plt.rcParams.update({
        'figure.facecolor':  PALETTE['bg'],
        'axes.facecolor':    PALETTE['bg'],
        'axes.edgecolor':    PALETTE['grid'],
        'axes.labelcolor':   PALETTE['text'],
        'xtick.color':       PALETTE['text'],
        'ytick.color':       PALETTE['text'],
        'grid.color':        PALETTE['grid'],
        'text.color':        PALETTE['text'],
        'font.family':       'DejaVu Sans',
        'font.size':         11,
        'axes.titlesize':    13,
        'axes.titleweight':  'bold',
    })
    
    cm = metrics['cm']
    recall = metrics['recall']
    precision = metrics['precision']
    f1 = metrics['f1']
    specificity = metrics['specificity']
    npv = metrics['npv']
    roc_auc = metrics['roc_auc']
    pr_auc = metrics['pr_auc']
    
    tn, fp, fn, tp = cm.ravel()
    
    fig = plt.figure(figsize=(20, 22))
    fig.suptitle(
        'Colorectal Cancer Prediction — Red Neuronal\n'
        f'Arquitectura: {len(feature_names)} → 256 → 128 → 64 → 1   |   Umbral: {best_thresh}   |   Test n={len(y_test):,}',
        fontsize=15, fontweight='bold', color=PALETTE['text'], y=0.98
    )
    
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)
    
    # ── (A) Matriz de Confusión ────────────────────────────────────────────────
    ax_cm = fig.add_subplot(gs[0, 0])
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Sin cáncer', 'Cáncer'])
    cmd.plot(ax=ax_cm, colorbar=False, cmap='Blues')
    ax_cm.set_title('A · Matriz de Confusión')
    ax_cm.set_xlabel('Predicción'); ax_cm.set_ylabel('Real')
    
    # ── (B) Curva ROC ─────────────────────────────────────────────────────────
    ax_roc = fig.add_subplot(gs[0, 1])
    RocCurveDisplay.from_predictions(y_test, y_prob_test,
                                      ax=ax_roc, color=PALETTE['primary'],
                                      name=f'MLP (AUC={roc_auc:.3f})')
    ax_roc.plot([0,1],[0,1], '--', color='gray', alpha=0.5, label='Random')
    fpr_c, tpr_c, thresh_c = roc_curve(y_test, y_prob_test)
    idx = np.argmin(np.abs(thresh_c - best_thresh))
    ax_roc.scatter(fpr_c[idx], tpr_c[idx], s=120, zorder=5,
                   color=PALETTE['secondary'], label=f'Umbral {best_thresh}')
    ax_roc.set_title('B · Curva ROC')
    ax_roc.legend(fontsize=9); ax_roc.grid(True, alpha=0.4)
    
    # ── (C) Curva Precision-Recall ────────────────────────────────────────────
    ax_pr = fig.add_subplot(gs[0, 2])
    PrecisionRecallDisplay.from_predictions(y_test, y_prob_test,
                                             ax=ax_pr, color=PALETTE['accent'],
                                             name=f'MLP (PR-AUC={pr_auc:.3f})')
    ax_pr.axhline(y=precision, color=PALETTE['secondary'], linestyle='--',
                  alpha=0.7, label=f'Precision actual ({precision:.3f})')
    ax_pr.axvline(x=recall, color=PALETTE['warn'], linestyle='--',
                  alpha=0.7, label=f'Recall actual ({recall:.3f})')
    ax_pr.set_title('C · Curva Precision-Recall')
    ax_pr.legend(fontsize=9); ax_pr.grid(True, alpha=0.4)
    
    # ── (D) Distribución de probabilidades ───────────────────────────────────
    ax_dist = fig.add_subplot(gs[1, :2])
    prob0 = y_prob_test[y_test == 0]
    prob1 = y_prob_test[y_test == 1]
    ax_dist.hist(prob0, bins=60, alpha=0.65, color=PALETTE['accent'],
                 label='Sin cáncer (0)', density=True)
    ax_dist.hist(prob1, bins=60, alpha=0.65, color=PALETTE['secondary'],
                 label='Cáncer (1)', density=True)
    ax_dist.axvline(x=best_thresh, color=PALETTE['primary'], linewidth=2,
                    linestyle='--', label=f'Umbral = {best_thresh}')
    ax_dist.set_xlabel('Probabilidad predicha (P(cáncer))')
    ax_dist.set_ylabel('Densidad')
    ax_dist.set_title('D · Distribución de Probabilidades por Clase')
    ax_dist.legend(); ax_dist.grid(True, alpha=0.4)
    
    # ── (E) Threshold sweep ───────────────────────────────────────────────────
    ax_thr = fig.add_subplot(gs[1, 2])
    df_thr = pd.DataFrame(results_thresh)
    ax_thr.plot(df_thr['threshold'], df_thr['recall'],    color=PALETTE['secondary'],
                linewidth=2, label='Recall')
    ax_thr.plot(df_thr['threshold'], df_thr['precision'], color=PALETTE['primary'],
                linewidth=2, label='Precision')
    ax_thr.plot(df_thr['threshold'], df_thr['f1'],        color=PALETTE['accent'],
                linewidth=2, label='F1')
    ax_thr.axvline(x=best_thresh, color='gray', linestyle='--', linewidth=1.5,
                   label=f'Elegido: {best_thresh}')
    ax_thr.set_xlabel('Umbral'); ax_thr.set_ylabel('Score')
    ax_thr.set_title('E · Métricas vs Umbral (Validación)')
    ax_thr.legend(fontsize=9); ax_thr.grid(True, alpha=0.4)
    
    # ── (F) Feature Importance ────────────────────────────────────────────────
    ax_fi = fig.add_subplot(gs[2, :])
    top_n = 15
    fi_top = feat_imp.head(top_n)
    colors_fi = [PALETTE['secondary'] if i < 3 else PALETTE['primary']
                 for i in range(top_n)]
    bars = ax_fi.barh(fi_top['feature'][::-1], fi_top['importance_mean'][::-1],
                      xerr=fi_top['importance_std'][::-1],
                      color=colors_fi[::-1], edgecolor='white', height=0.7,
                      error_kw={'elinewidth': 1.2, 'capsize': 3, 'alpha': 0.7})
    ax_fi.set_xlabel('Reducción de Recall (importancia por permutación)')
    ax_fi.set_title(f'F · Top-{top_n} Features — Impacto en Recall')
    ax_fi.grid(True, alpha=0.4, axis='x')
    from matplotlib.patches import Patch
    legend_elem = [Patch(facecolor=PALETTE['secondary'], label='Top 3 features'),
                   Patch(facecolor=PALETTE['primary'],   label='Resto del top-15')]
    ax_fi.legend(handles=legend_elem, fontsize=9, loc='lower right')
    
    # ── Panel de resumen de métricas ──────────────────────────────────────────
    metrics_text = (
        f"  RECALL        {recall:.4f}  ★\n"
        f"  PRECISION     {precision:.4f}\n"
        f"  F1-SCORE      {f1:.4f}\n"
        f"  SPECIFICITY   {specificity:.4f}\n"
        f"  NPV           {npv:.4f}\n"
        f"  ROC-AUC       {roc_auc:.4f}\n"
        f"  PR-AUC        {pr_auc:.4f}\n"
        f"  CV Recall     {cv_scores.mean():.4f}±{cv_scores.std():.4f}\n"
        f"  Umbral        {best_thresh}\n"
        f"  TP / FN       {tp:,} / {fn:,}"
    )
    fig.text(0.77, 0.36, metrics_text,
             fontsize=10.5, family='monospace',
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='white',
                       edgecolor=PALETTE['primary'], linewidth=1.5),
             color=PALETTE['text'])
    
    fig.text(0.77, 0.65, 'RESUMEN\nMÉTRICAS',
             fontsize=12, fontweight='bold', color=PALETTE['primary'],
             ha='left', va='top')
    
    out_fig = os.path.join(ANALYSIS_DIR, 'generic_tabular_results.png')
    plt.savefig(out_fig, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    print(f"\n[PLOT] Gráfico guardado: {out_fig}")
    plt.close()


def main():
    """Función principal: ejecuta el pipeline completo de entrenamiento."""
    # 1. Carga y preparación
    X, y, feature_names = load_and_prepare_data()
    
    # 2. Split
    X_train, X_val, X_test, y_train, y_val, y_test, val_ratio = split_data(X, y)
    
    # 3. Escalado
    X_train_s, X_val_s, X_test_s, scaler = scale_data(X_train, X_val, X_test)
    
    # 4. Entrenamiento
    model = create_and_train_model(X_train_s, X_val_s, y_train, y_val, feature_names, val_ratio)
    
    # 5. Búsqueda de umbral óptimo
    best_thresh, results_thresh = find_optimal_threshold(model, X_val_s, y_val)
    
    # 6. Evaluación
    metrics = evaluate_model(model, X_test_s, y_test, best_thresh)
    
    # 7. Validación cruzada (15 - 18 minutos aprox)
    #cv_scores = cross_validation(model, X_train_s, y_train)
    
    # 8. Importancia de features
    feat_imp = compute_feature_importance(model, X_test_s, y_test, feature_names)
    
    # 9. Explicabilidad
    explanations = generate_explanations(model, X_test_s, y_test, feature_names, X_train_s)
    plot_explanations(explanations, ANALYSIS_DIR)
    
    # 10. Guarda artefactos
    config_data = {
        'feature_names': feature_names,
        'best_threshold': best_thresh,
        'metrics': metrics,
        'seed': SEED,
    }
    save_model_and_artifacts(model, scaler, config_data)
    
    # 11. Genera gráficas
    plot_results(
        y_test, metrics['y_prob_test'], metrics['y_pred_test'], metrics,
        results_thresh, feat_imp, feature_names, best_thresh, cv_scores
    )
    
    print("\n" + "=" * 60)
    print("  ENTRENAMIENTO COMPLETADO")
    print("=" * 60)


if __name__ == '__main__':
    main()
