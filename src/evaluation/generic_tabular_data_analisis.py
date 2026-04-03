"""
Colorectal Cancer Dataset — Limpieza y Análisis de Correlaciones
=================================================================
Dependencias:
    pip install pandas numpy matplotlib seaborn scikit-learn

Uso:
    python src/evaluation/generic_tabular_data_analisis.py

    Utiliza rutas del sistema centralizado de configuración.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# Agregar src al path para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.config.paths import paths

# ── Configuración de rutas ────────────────────────────────────────────────────
INPUT_CSV  = str(paths.PROCESSED_TABULAR / "colorectal_cancer_full_dataset.csv")
OUTPUT_DIR = paths.ANALYSIS

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
print(f"Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CODIFICACIÓN DE VARIABLES CATEGÓRICAS → [0, 1]
#    NOTA: El dataset ya viene procesado por generic_tabular_synthetic.py,
#    solo necesitamos convertir columnas categóricas restantes si existen
# ─────────────────────────────────────────────────────────────────────────────

# Binarias Yes/No → 1/0 (solo si existen en el dataset)
binary_yes_no = [
    'Family_History', 'Smoking_History', 'Alcohol_Consumption', 'Diabetes',
    'Inflammatory_Bowel_Disease', 'Genetic_Mutation', 'Early_Detection',
]
for col in binary_yes_no:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = (df[col] == 'Yes').astype(float)

# Binarias con semántica propia (solo si existen)
if 'Gender' in df.columns and df['Gender'].dtype == 'object':
    df['Gender'] = (df['Gender'] == 'M').astype(float)
if 'Urban_or_Rural' in df.columns and df['Urban_or_Rural'].dtype == 'object':
    df['Urban_or_Rural'] = (df['Urban_or_Rural'] == 'Urban').astype(float)

# Ordinales → 0 / 0.5 / 1  (dirección: mayor valor = mayor riesgo)
ordinal_maps = {
    'Obesity_BMI':       {'Normal': 0.0, 'Overweight': 0.5, 'Obese': 1.0},
    'Diet_Risk':         {'Low': 0.0, 'Moderate': 0.5, 'High': 1.0},
    'Physical_Activity': {'High': 0.0, 'Moderate': 0.5, 'Low': 1.0},
    'Screening_History': {'Regular': 0.0, 'Irregular': 0.5, 'Never': 1.0},
    'Age_Risk_Group':    {'Low': 0.0, 'Medium': 0.5, 'High': 0.75, 'Very_High': 1.0},
}
for col, mapping in ordinal_maps.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# ─────────────────────────────────────────────────────────────────────────────
# 3. NORMALIZACIÓN DE VARIABLES CONTINUAS → MinMaxScaler [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
continuous = [
    'Age', 'Tumor_Size_mm', 'Healthcare_Costs',
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
# 6. TOP CORRELACIONES CON DIAGNOSIS (target principal)
# ─────────────────────────────────────────────────────────────────────────────
if 'Diagnosis' in df.columns:
    targets = ['Diagnosis']
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle('Top Correlaciones con Diagnosis (Cáncer Colorrectal)',
                 fontsize=15, fontweight='bold', color=TEXT, y=1.01)

    ax = axes
    ax.set_facecolor(CARD_BG)
    vals = corr['Diagnosis'].drop('Diagnosis').sort_values()
    vals = pd.concat([vals.head(12), vals.tail(12)]).sort_values()
    colors = [RED_COOL if v > 0 else GREEN_OK for v in vals]
    bars = ax.barh(vals.index, vals.values, color=colors, edgecolor='none', height=0.7)
    ax.axvline(0, color='#555', lw=1)
    ax.set_xlabel('Pearson r', fontsize=11)
    ax.set_title('Features correlacionadas con Diagnosis', fontsize=13, fontweight='bold', color=TEXT)
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
# 7. PAIRPLOT DE VARIABLES CLAVE CON DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────
key_vars = [
    'Diagnosis', 'Age', 'Tumor_Size_mm', 'Cancer_Stage', 'Genetic_Mutation',
    'Smoking_History', 'Screening_History',
]
# Filtrar solo columnas que existen en el dataset
key_vars_existing = [col for col in key_vars if col in df.columns]

if len(key_vars_existing) > 1:
    # Muestra de 3000 filas para que sea rapido
    sample = df[key_vars_existing].sample(min(3000, len(df)), random_state=42)

    g = sns.pairplot(
        sample, hue='Diagnosis', diag_kind='kde',
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
        mpatches.Patch(color=ACCENT,   label='Diagnosis = Healthy'),
        mpatches.Patch(color=RED_COOL, label='Diagnosis = CRC'),
    ]
    g.figure.legend(handles=handles, loc='upper right',
                    facecolor=CARD_BG, edgecolor='#2e3248', labelcolor=TEXT, fontsize=10)
    g.figure.suptitle('Pairplot — Variables Clave con Diagnosis (muestra 3 000)',
                      y=1.01, fontsize=14, fontweight='bold', color=TEXT)
    plt.savefig(OUTPUT_DIR / 'pairplot_clave.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("Pairplot guardado")
else:
    print("⚠️  Insuficientes columnas para pairplot")

# ─────────────────────────────────────────────────────────────────────────────
# 8. DISTRIBUCIÓN DE VARIABLES CONTINUAS POR DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────
cont_plot = [
    'Age', 'Tumor_Size_mm', 'Healthcare_Costs',
]
# Filtrar solo columnas que existen
cont_plot = [col for col in cont_plot if col in df.columns]

if len(cont_plot) > 0 and 'Diagnosis' in df.columns:
    fig, axes = plt.subplots(1, len(cont_plot), figsize=(16, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle('Distribucion de Variables Continuas por Diagnosis',
                 fontsize=14, fontweight='bold', color=TEXT)

    for ax, col in zip(axes, cont_plot):
        ax.set_facecolor(CARD_BG)
        for val, color, label in [(0, ACCENT, 'Healthy'), (1, RED_COOL, 'CRC')]:
            data = df.loc[df['Diagnosis'] == val, col]
            ax.hist(data.dropna(), bins=40, alpha=0.55, color=color,
                    label=f'Diagnosis={label}', density=True, edgecolor='none')
        ax.set_title(col, fontsize=10, color=TEXT)
        ax.set_xlabel('Valor normalizado', fontsize=8)
        ax.legend(fontsize=8, facecolor=CARD_BG, edgecolor='#2e3248', labelcolor=TEXT)
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2e3248')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distribucion_continuas.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Distribucion continuas guardado")
else:
    print("⚠️  No hay variables continuas para plotear")

print(f"\nTodo completado. Resultados en la carpeta '{OUTPUT_DIR}/'")