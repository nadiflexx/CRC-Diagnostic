"""
Colorectal Cancer Dataset — Limpieza y Análisis de Correlaciones
=================================================================
Dependencias:
    pip install pandas numpy matplotlib seaborn scikit-learn

Uso:
    python colorectal_analysis.py

    Por defecto espera el CSV en la misma carpeta que este script.
    Puedes cambiar INPUT_CSV y OUTPUT_DIR abajo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# ── Configuración de rutas ────────────────────────────────────────────────────
INPUT_CSV  = "data/colorectal_cancer_full_dataset.csv"
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
print(f"Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CODIFICACIÓN DE VARIABLES CATEGÓRICAS → [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

# Binarias Yes/No → 1/0
binary_yes_no = [
    'Family_History', 'Smoking_History', 'Alcohol_Consumption', 'Diabetes',
    'Inflammatory_Bowel_Disease', 'Genetic_Mutation', 'Early_Detection',
    'Survival_5_years', 'Mortality', 'Survival_Prediction',
]
for col in binary_yes_no:
    df[col] = (df[col] == 'Yes').astype(float)

# Binarias con semántica propia
df['Gender']                  = (df['Gender'] == 'M').astype(float)         # M=1, F=0
df['Urban_or_Rural']          = (df['Urban_or_Rural'] == 'Urban').astype(float)
df['Economic_Classification'] = (df['Economic_Classification'] == 'Developed').astype(float)
df['Insurance_Status']        = (df['Insurance_Status'] == 'Insured').astype(float)

# Ordinales → 0 / 0.5 / 1  (dirección: mayor valor = mayor riesgo)
ordinal_maps = {
    'Cancer_Stage':      {'Localized': 0.0, 'Regional': 0.5, 'Metastatic': 1.0},
    'Obesity_BMI':       {'Normal': 0.0, 'Overweight': 0.5, 'Obese': 1.0},
    'Diet_Risk':         {'Low': 0.0, 'Moderate': 0.5, 'High': 1.0},
    'Physical_Activity': {'High': 0.0, 'Moderate': 0.5, 'Low': 1.0},  # baja actividad = mayor riesgo
    'Screening_History': {'Regular': 0.0, 'Irregular': 0.5, 'Never': 1.0},
    'Healthcare_Access': {'High': 0.0, 'Moderate': 0.5, 'Low': 1.0},
}
for col, mapping in ordinal_maps.items():
    df[col] = df[col].map(mapping)

# Nominales con más de 2 niveles → one-hot encoding
df = pd.get_dummies(df, columns=['Country', 'Treatment_Type'], drop_first=False)

# Aseguramos que todo sea float (pandas puede crear booleanos en get_dummies)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(float)

# ─────────────────────────────────────────────────────────────────────────────
# 3. NORMALIZACIÓN DE VARIABLES CONTINUAS → MinMaxScaler [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
continuous = [
    'Age', 'Tumor_Size_mm', 'Healthcare_Costs',
    'Incidence_Rate_per_100K', 'Mortality_Rate_per_100K',
]
scaler = MinMaxScaler()
df[continuous] = scaler.fit_transform(df[continuous])

# ─────────────────────────────────────────────────────────────────────────────
# 4. GUARDAR DATASET LIMPIO
# ─────────────────────────────────────────────────────────────────────────────
clean_path = OUTPUT_DIR / "colorectal_cancer_clean.csv"
df.to_csv(clean_path, index=False)
print(f"Dataset limpio guardado -> {clean_path}  ({df.shape[0]:,} filas x {df.shape[1]} columnas)")

# ─────────────────────────────────────────────────────────────────────────────
# PALETA Y ESTILO GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG  = '#0f1117'
CARD_BG  = '#1a1d27'
TEXT     = '#e8eaf0'
ACCENT   = '#6c8ebf'
RED_COOL = '#e05c5c'
GREEN_OK = '#5cb85c'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   CARD_BG,
    'axes.edgecolor':   '#2e3248',
    'axes.labelcolor':  TEXT,
    'xtick.color':      TEXT,
    'ytick.color':      TEXT,
    'text.color':       TEXT,
    'grid.color':       '#2e3248',
    'font.family':      'DejaVu Sans',
})

# Columnas "core" (excluye one-hots de Country y Treatment_Type)
core_cols = [c for c in df.columns
             if not c.startswith('Country_') and not c.startswith('Treatment_Type_')]

corr = df[core_cols].corr()

# ─────────────────────────────────────────────────────────────────────────────
# 5. HEATMAP DE CORRELACIÓN
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 16))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(CARD_BG)

cmap = sns.diverging_palette(220, 10, as_cmap=True)   # azul → blanco → rojo

# Máscara para mostrar solo el triángulo inferior
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(
    corr, mask=mask, cmap=cmap, center=0,
    vmin=-1, vmax=1,
    annot=True, fmt='.2f', annot_kws={'size': 7.5, 'color': TEXT},
    linewidths=0.4, linecolor='#0f1117',
    square=True, ax=ax,
    cbar_kws={'shrink': 0.75, 'label': 'Pearson r'},
)
ax.set_title('Heatmap de Correlacion — Colorectal Cancer Dataset',
             fontsize=16, fontweight='bold', color=TEXT, pad=18)
ax.tick_params(axis='x', rotation=45, labelsize=9)
ax.tick_params(axis='y', rotation=0,  labelsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'heatmap_correlacion.png', dpi=160, bbox_inches='tight')
plt.close()
print("Heatmap guardado")

# ─────────────────────────────────────────────────────────────────────────────
# 6. TOP CORRELACIONES CON MORTALITY Y SURVIVAL_5_YEARS
# ─────────────────────────────────────────────────────────────────────────────
targets = ['Mortality', 'Survival_5_years']
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle('Top Correlaciones con Mortalidad y Supervivencia',
             fontsize=15, fontweight='bold', color=TEXT, y=1.01)

for ax, target in zip(axes, targets):
    ax.set_facecolor(CARD_BG)
    vals = corr[target].drop(targets).sort_values()
    vals = pd.concat([vals.head(12), vals.tail(12)]).sort_values()
    colors = [RED_COOL if v > 0 else GREEN_OK for v in vals]
    bars = ax.barh(vals.index, vals.values, color=colors, edgecolor='none', height=0.7)
    ax.axvline(0, color='#555', lw=1)
    ax.set_xlabel('Pearson r', fontsize=11)
    ax.set_title(f'<-> {target}', fontsize=13, fontweight='bold', color=TEXT)
    ax.set_xlim(-1, 1)
    for bar, v in zip(bars, vals.values):
        ax.text(
            v + (0.02 if v >= 0 else -0.02),
            bar.get_y() + bar.get_height() / 2,
            f'{v:.2f}', va='center',
            ha='left' if v >= 0 else 'right',
            fontsize=8, color=TEXT,
        )
    pos_patch = mpatches.Patch(color=RED_COOL, label='Correlacion positiva')
    neg_patch = mpatches.Patch(color=GREEN_OK, label='Correlacion negativa')
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9,
              facecolor=CARD_BG, edgecolor='#2e3248', labelcolor=TEXT)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'top_correlaciones.png', dpi=160, bbox_inches='tight')
plt.close()
print("Top correlaciones guardado")

# ─────────────────────────────────────────────────────────────────────────────
# 7. PAIRPLOT DE VARIABLES CLAVE
# ─────────────────────────────────────────────────────────────────────────────
key_vars = [
    'Age', 'Tumor_Size_mm', 'Cancer_Stage', 'Genetic_Mutation',
    'Smoking_History', 'Screening_History', 'Mortality',
]
# Muestra de 3000 filas para que sea rapido
sample = df[key_vars].sample(3000, random_state=42)

g = sns.pairplot(
    sample, hue='Mortality', diag_kind='kde',
    palette={0: ACCENT, 1: RED_COOL},
    plot_kws={'alpha': 0.25, 's': 12},
    diag_kws={'fill': True, 'alpha': 0.5},
)
g.figure.patch.set_facecolor(DARK_BG)
for ax in g.axes.flatten():
    if ax:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT, labelsize=7)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2e3248')

handles = [
    mpatches.Patch(color=ACCENT,   label='Mortality = No'),
    mpatches.Patch(color=RED_COOL, label='Mortality = Yes'),
]
g.figure.legend(handles=handles, loc='upper right',
                facecolor=CARD_BG, edgecolor='#2e3248', labelcolor=TEXT, fontsize=10)
g.figure.suptitle('Pairplot — Variables Clave (muestra 3 000)',
                  y=1.01, fontsize=14, fontweight='bold', color=TEXT)
plt.savefig(OUTPUT_DIR / 'pairplot_clave.png', dpi=130, bbox_inches='tight')
plt.close()
print("Pairplot guardado")

# ─────────────────────────────────────────────────────────────────────────────
# 8. DISTRIBUCIÓN DE VARIABLES CONTINUAS POR MORTALIDAD
# ─────────────────────────────────────────────────────────────────────────────
cont_plot = [
    'Age', 'Tumor_Size_mm', 'Healthcare_Costs',
    'Incidence_Rate_per_100K', 'Mortality_Rate_per_100K',
]
fig, axes = plt.subplots(1, len(cont_plot), figsize=(22, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle('Distribucion de Variables Continuas por Mortalidad',
             fontsize=14, fontweight='bold', color=TEXT)

for ax, col in zip(axes, cont_plot):
    ax.set_facecolor(CARD_BG)
    for val, color, label in [(0, ACCENT, 'No'), (1, RED_COOL, 'Yes')]:
        data = df.loc[df['Mortality'] == val, col]
        ax.hist(data, bins=40, alpha=0.55, color=color,
                label=f'Mortality={label}', density=True, edgecolor='none')
    ax.set_title(col, fontsize=10, color=TEXT)
    ax.set_xlabel('Valor normalizado', fontsize=8)
    ax.legend(fontsize=8, facecolor=CARD_BG, edgecolor='#2e3248', labelcolor=TEXT)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'distribucion_continuas.png', dpi=150, bbox_inches='tight')
plt.close()
print("Distribucion continuas guardado")

print(f"\nTodo completado. Resultados en la carpeta '{OUTPUT_DIR}/'")