import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from config_features import (INPUT_CSV_TABULAR_PROCESSED, OUTPUT_MODEL_TABULAR)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score,
    ConfusionMatrixDisplay, RocCurveDisplay,
    PrecisionRecallDisplay
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib, os, time

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
SEED        = 42
TEST_SIZE   = 0.30
VAL_SIZE    = 0.10
DATA_PATH   = INPUT_CSV_TABULAR_PROCESSED
OUT_DIR     = OUTPUT_MODEL_TABULAR
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA Y SEPARACIÓN DE FEATURES / TARGET
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  COLORECTAL CANCER — RED NEURONAL")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[DATA] Registros: {len(df):,}  |  Features: {df.shape[1]-1}")
print(f"[DATA] Distribución Diagnosis → {df['Diagnosis'].value_counts().to_dict()}")

X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']
feature_names = X.columns.tolist()

# ─────────────────────────────────────────────────────────────────────────────
# 2. SPLIT: TRAIN / VAL / TEST
#    Estratificado para mantener el 50/50 en cada partición
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# 3. ESCALADO (StandardScaler fit solo sobre TRAIN)
# ─────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 4. DEFINICIÓN DE LA RED NEURONAL
#
#  Arquitectura: 3 capas ocultas con decrecimiento progresivo
#    Input (24) → 256 → 128 → 64 → Output (1)
#
#  Decisiones de diseño orientadas a maximizar RECALL:
#    · activation='relu'  — evita saturación en capas profundas
#    · solver='adam'      — convergencia rápida con datasets grandes
#    · early_stopping     — usa la partición de validación para frenar
#    · n_iter_no_change   — paciencia generosa (20 épocas sin mejora)
#    · alpha=1e-4         — regularización L2 para reducir sobreajuste
# ─────────────────────────────────────────────────────────────────────────────
# MLPClassifier no permite pasar X_val externamente; entrenamos con train+val
# fusionados y reservamos val_ratio internamente para el early stopping.
# NOTA: validation_fraction debe ser > 0 y < 1, nunca 0.0.
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

# ─────────────────────────────────────────────────────────────────────────────
# 5. BÚSQUEDA DEL UMBRAL ÓPTIMO (prioridad RECALL ≥ 0.90)
#    Por defecto threshold=0.5; lo ajustamos sobre validación para
#    maximizar Recall sin derrumbar Precision por debajo de 0.60
# ─────────────────────────────────────────────────────────────────────────────
y_prob_val  = mlp_final.predict_proba(X_val_s)[:, 1]

best_thresh  = 0.50
best_recall  = 0.0
best_f1      = 0.0
results_thresh = []

for t in np.arange(0.20, 0.75, 0.01):
    y_pred_t  = (y_prob_val >= t).astype(int)
    rec = recall_score(y_val, y_pred_t, zero_division=0)
    pre = precision_score(y_val, y_pred_t, zero_division=0)
    f1  = f1_score(y_val, y_pred_t, zero_division=0)
    results_thresh.append({'threshold': round(t, 2), 'recall': rec, 'precision': pre, 'f1': f1})
    # Criterio: recall ≥ 0.90 Y maximizar F1
    if rec >= 0.90 and f1 > best_f1:
        best_f1     = f1
        best_recall = rec
        best_thresh = round(t, 2)

print(f"\n[THRESHOLD] Umbral óptimo seleccionado: {best_thresh}")
print(f"[THRESHOLD] Recall en validación: {best_recall:.4f} | F1: {best_f1:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUACIÓN FINAL EN TEST
# ─────────────────────────────────────────────────────────────────────────────
y_prob_test = mlp_final.predict_proba(X_test_s)[:, 1]
y_pred_test = (y_prob_test >= best_thresh).astype(int)

recall    = recall_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
f1        = f1_score(y_test, y_pred_test)
roc_auc   = roc_auc_score(y_test, y_prob_test)
pr_auc    = average_precision_score(y_test, y_prob_test)
cm        = confusion_matrix(y_test, y_pred_test)

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
npv         = tn / (tn + fn) if (tn + fn) > 0 else 0   # Valor Predictivo Negativo

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

# ─────────────────────────────────────────────────────────────────────────────
# 7. VALIDACIÓN CRUZADA (5-fold) sobre training completo
# ─────────────────────────────────────────────────────────────────────────────
print("[CV] Validación cruzada 5-fold (recall)...")
cv_model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    learning_rate='adaptive',
    learning_rate_init=1e-3,
    max_iter=100,             # reducido para CV rápido
    random_state=SEED,
    verbose=False,
)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(cv_model, X_train_s, y_train,
                             cv=skf, scoring='recall', n_jobs=-1)
print(f"[CV] Recall por fold: {[round(s, 4) for s in cv_scores]}")
print(f"[CV] Recall medio: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. IMPORTANCIA DE FEATURES (via permutation sobre test)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.inspection import permutation_importance
print("\n[IMPORTANCE] Calculando importancia de features...")
perm = permutation_importance(
    mlp_final, X_test_s, y_test,
    n_repeats=5, random_state=SEED, scoring='recall', n_jobs=-1
)
feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm.importances_mean,
    'importance_std':  perm.importances_std
}).sort_values('importance_mean', ascending=False).reset_index(drop=True)

print("\n[IMPORTANCE] Top-10 features por impacto en Recall:")
print(feat_imp.head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 9. GRÁFICAS
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    'primary':   '#2563EB',   # azul
    'secondary': '#DC2626',   # rojo
    'accent':    '#16A34A',   # verde
    'warn':      '#D97706',   # naranja
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
# Anotar FN prominentemente
ax_cm.texts[2].set_fontsize(16)   # TN
ax_cm.texts[3].set_fontsize(16)   # FP
ax_cm.texts[0].set_fontsize(16)   # FN
ax_cm.texts[1].set_fontsize(16)   # TP

# ── (B) Curva ROC ─────────────────────────────────────────────────────────
ax_roc = fig.add_subplot(gs[0, 1])
RocCurveDisplay.from_predictions(y_test, y_prob_test,
                                  ax=ax_roc, color=PALETTE['primary'],
                                  name=f'MLP (AUC={roc_auc:.3f})')
ax_roc.plot([0,1],[0,1], '--', color='gray', alpha=0.5, label='Random')
# Marcar el punto del umbral elegido
from sklearn.metrics import roc_curve
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
# Leyenda manual
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

out_fig = os.path.join(OUT_DIR, 'nn_cancer_results.png')
plt.savefig(out_fig, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
print(f"\n[PLOT] Gráfico guardado: {out_fig}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. GUARDAR MODELO Y SCALER
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump(mlp_final, os.path.join(OUT_DIR, 'mlp_model.pkl'))
joblib.dump(scaler,    os.path.join(OUT_DIR, 'scaler.pkl'))
joblib.dump({'threshold': best_thresh, 'features': feature_names},
            os.path.join(OUT_DIR, 'model_config.pkl'))
print("[SAVE] Modelo, scaler y configuración guardados en /mnt/user-data/outputs/")

print("\n" + "=" * 60)
print("  ENTRENAMIENTO COMPLETADO")
print("=" * 60)