"""
notebooks/eda_processed_datasets.py

EDA del dataset PROCESADO (colon_processed/) — lo que el modelo realmente ve.

Análisis:
  1. Inventario: imágenes por clase, por fuente, por split
  2. Balance: distribución de clases en train/val/test
  3. Source mixing: verificación anti-leakage por split
  4. Resoluciones: ¿todas son 384×384 post-preprocessing?
  5. Color post-CLAHE: histogramas RGB/HSV después de normalizar
  6. Artefactos residuales: % negro restante post-crop
  7. Domain gap residual: ¿el preprocesamiento cerró el gap?
  8. Brillo/contraste/saturación por clase (¿hay sesgo?)
  9. Brillo/contraste/saturación por fuente (¿hay shortcut?)
  10. Muestras visuales por clase × fuente
  11. Comparativa RAW vs PROCESSED (antes/después)
  12. Tissue-only: análisis si existe colon_processed_tissue_only/
  13. Resumen clínico y recomendaciones

Genera:
  notebooks/eda_processed_output/
  ├── 01_inventory_splits.png
  ├── 02_class_balance.png
  ├── 03_source_mixing.png
  ├── 04_resolutions_check.png
  ├── 05_color_post_clahe.png
  ├── 06_residual_artifacts.png
  ├── 07_domain_gap_residual.png
  ├── 08_features_by_class.png
  ├── 09_features_by_source.png
  ├── 10_samples_class_source.png
  ├── 11_raw_vs_processed.png
  ├── 12_tissue_only_comparison.png
  ├── eda_processed_report.txt
  └── eda_processed_stats.json
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import warnings

import cv2
import matplotlib

from src.config.paths import paths
from src.database.connection import get_db
from src.database.repositories import TrainingImageRepository

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")


OUTPUT_DIR = paths.NOTEBOOKS / "eda_processed_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = paths.DATA / "colon_processed"
CLEAN_DIR = paths.DATA / "colon_clean"
TISSUE_DIR = paths.DATA / "colon_processed_tissue_only"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SOURCE_COLORS = {
    "hyperkvasir": "#3498db",
    "cvc_clinicdb": "#e74c3c",
    "limuc": "#2ecc71",
    "curated_colon": "#f39c12",
    "unknown": "#95a5a6",
}

CLASS_COLORS = {
    "normal": "#3498db",
    "polyp": "#e74c3c",
    "inflammation": "#f39c12",
}

SPLIT_COLORS = {
    "train": "#2ecc71",
    "val": "#f39c12",
    "test": "#e74c3c",
}


# ═══════════════════════════════════════════════════════════
#  UTILIDADES
# ═══════════════════════════════════════════════════════════


def detect_source(filepath: str) -> str:
    """Detecta fuente a partir del nombre."""
    stem = Path(filepath).stem
    if stem.startswith("hk_"):
        return "hyperkvasir"
    elif stem.startswith("cvc_"):
        return "cvc_clinicdb"
    elif stem.startswith("limuc_"):
        return "limuc"
    elif stem.startswith("curated_"):
        return "curated_colon"
    return "unknown"


def load_db_records() -> list[dict]:
    """Carga todos los records de la DB con atributos extraídos."""
    with get_db() as db:
        repo = TrainingImageRepository(db)
        all_records = []
        for split in ["train", "val", "test"]:
            data = repo.get_by_split(split)
            for d in data:
                all_records.append(
                    {
                        "file_path": str(d.file_path),
                        "label": int(d.label),
                        "split": str(d.split),
                        "dataset_source": str(d.dataset_source)
                        if d.dataset_source
                        else "unknown",
                        "mask_path": str(d.mask_path) if d.mask_path else None,
                        "width": int(d.width) if d.width else None,
                        "height": int(d.height) if d.height else None,
                    }
                )
    return all_records


def compute_image_stats(img_path: str) -> dict | None:
    """Estadísticas de una imagen procesada."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        brightness = float(gray.mean())
        contrast = float(gray.std())
        saturation = float(hsv[:, :, 1].mean())

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())

        black_pct = float((gray < 15).sum() / gray.size * 100)

        green_lower = np.array([30, 50, 50])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_pct = float(green_mask.sum() / (255.0 * gray.size) * 100)

        r_mean = float(img[:, :, 2].mean())
        g_mean = float(img[:, :, 1].mean())
        b_mean = float(img[:, :, 0].mean())

        hue_mean = float(hsv[:, :, 0].mean())
        val_mean = float(hsv[:, :, 2].mean())

        return {
            "width": w,
            "height": h,
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "saturation": round(saturation, 2),
            "sharpness": round(sharpness, 2),
            "black_pct": round(black_pct, 2),
            "green_pct": round(green_pct, 2),
            "r_mean": round(r_mean, 2),
            "g_mean": round(g_mean, 2),
            "b_mean": round(b_mean, 2),
            "hue_mean": round(hue_mean, 2),
            "val_mean": round(val_mean, 2),
            "file_size_kb": round(Path(img_path).stat().st_size / 1024, 1),
        }
    except Exception:
        return None


def sample_indices(n: int, k: int, seed: int = 42) -> list[int]:
    """Muestra aleatoria de k índices de n."""
    rng = np.random.RandomState(seed)
    if n <= k:
        return list(range(n))
    return sorted(rng.choice(n, k, replace=False).tolist())


# ═══════════════════════════════════════════════════════════
#  CARGA DE DATOS
# ═══════════════════════════════════════════════════════════


def load_and_enrich_records() -> list[dict]:
    """Carga records del DB y añade source + class_name."""
    label_to_class = {0: "normal", 1: "polyp", 2: "inflammation"}

    # Intentar cargar class_mapping
    for p in [
        PROCESSED_DIR / "class_mapping.json",
        CLEAN_DIR / "class_mapping.json",
    ]:
        if p.exists():
            with open(p) as f:
                mapping = json.load(f)
            label_to_class = {int(k): v for k, v in mapping.get("classes", {}).items()}
            break

    records = load_db_records()

    for r in records:
        r["source"] = detect_source(r["file_path"])
        r["class_name"] = label_to_class.get(r["label"], f"class_{r['label']}")
        r["exists"] = Path(r["file_path"]).exists()

    return records


# ═══════════════════════════════════════════════════════════
#  1. INVENTARIO POR SPLIT
# ═══════════════════════════════════════════════════════════


def analyze_inventory(records: list[dict]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("1. INVENTARIO: IMÁGENES PROCESADAS POR SPLIT")
    lines.append("=" * 80)

    total = len(records)
    existing = sum(1 for r in records if r["exists"])
    missing = total - existing
    lines.append(f"\n  Total en DB: {total}")
    lines.append(f"  Existen en disco: {existing}")
    if missing > 0:
        lines.append(f"  ⚠️  Faltan: {missing}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Por split
    split_counts = Counter(r["split"] for r in records)
    ax = axes[0]
    splits = ["train", "val", "test"]
    counts = [split_counts.get(s, 0) for s in splits]
    colors = [SPLIT_COLORS[s] for s in splits]
    bars = ax.bar(splits, counts, color=colors, edgecolor="white")
    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center",
            fontweight="bold",
        )
    ax.set_title("Imágenes por Split")
    ax.set_ylabel("Cantidad")

    for s in splits:
        pct = split_counts.get(s, 0) / total * 100
        lines.append(f"  {s:6s}: {split_counts.get(s, 0):5d} ({pct:.1f}%)")

    # Por clase
    class_counts = Counter(r["class_name"] for r in records)
    ax = axes[1]
    classes = sorted(class_counts.keys())
    c_counts = [class_counts[c] for c in classes]
    c_colors = [CLASS_COLORS.get(c, "#95a5a6") for c in classes]
    bars = ax.bar(classes, c_counts, color=c_colors, edgecolor="white")
    for bar, count in zip(bars, c_counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(c_counts) * 0.01,
            str(count),
            ha="center",
            fontweight="bold",
        )
    ax.set_title("Imágenes por Clase")
    ax.set_ylabel("Cantidad")

    lines.append("\n  Por clase:")
    for c in classes:
        lines.append(f"    {c:15s}: {class_counts[c]:5d}")

    # Por fuente
    source_counts = Counter(r["source"] for r in records)
    ax = axes[2]
    sources = sorted(source_counts.keys())
    s_counts = [source_counts[s] for s in sources]
    s_colors = [SOURCE_COLORS.get(s, "#95a5a6") for s in sources]
    bars = ax.bar(sources, s_counts, color=s_colors, edgecolor="white")
    for bar, count in zip(bars, s_counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(s_counts) * 0.01,
            str(count),
            ha="center",
            fontweight="bold",
            fontsize=8,
        )
    ax.set_title("Imágenes por Fuente")
    ax.set_ylabel("Cantidad")
    ax.tick_params(axis="x", rotation=15)

    lines.append("\n  Por fuente:")
    for s in sources:
        lines.append(f"    {s:15s}: {source_counts[s]:5d}")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "01_inventory_splits.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    text = "\n".join(lines)
    print(text)
    return text


# ═══════════════════════════════════════════════════════════
#  2. BALANCE DE CLASES POR SPLIT
# ═══════════════════════════════════════════════════════════


def analyze_class_balance(records: list[dict]) -> str:
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("2. BALANCE DE CLASES POR SPLIT")
    lines.append("=" * 80)

    splits = ["train", "val", "test"]
    classes = sorted({r["class_name"] for r in records})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, split in enumerate(splits):
        split_records = [r for r in records if r["split"] == split]
        class_counts = Counter(r["class_name"] for r in split_records)

        ax = axes[idx]
        c_list = [class_counts.get(c, 0) for c in classes]
        c_colors = [CLASS_COLORS.get(c, "#95a5a6") for c in classes]

        bars = ax.bar(classes, c_list, color=c_colors, edgecolor="white")
        for bar, count in zip(bars, c_list, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(count),
                ha="center",
                fontsize=9,
            )
        ax.set_title(f"{split.upper()} ({len(split_records)})")
        ax.set_ylabel("Cantidad")

        lines.append(f"\n  {split.upper()} ({len(split_records)}):")
        total_split = len(split_records)
        for c in classes:
            cnt = class_counts.get(c, 0)
            pct = cnt / total_split * 100 if total_split > 0 else 0
            lines.append(f"    {c:15s}: {cnt:5d} ({pct:.1f}%)")

        # Imbalance ratio
        if c_list:
            ratio = max(c_list) / max(min(c_list), 1)
            status = "✅" if ratio < 1.5 else "⚠️"
            lines.append(f"    {status} Ratio max/min: {ratio:.2f}")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "02_class_balance.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    text = "\n".join(lines)
    print(text)
    return text


# ═══════════════════════════════════════════════════════════
#  3. SOURCE MIXING POR SPLIT × CLASE
# ═══════════════════════════════════════════════════════════


def analyze_source_mixing(records: list[dict]) -> str:
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("3. SOURCE MIXING (Anti-Leakage)")
    lines.append("=" * 80)
    lines.append("  Cada clase en cada split debe tener ≥2 fuentes")

    splits = ["train", "val", "test"]
    classes = sorted({r["class_name"] for r in records})
    sources = sorted({r["source"] for r in records})

    n_splits = len(splits)

    fig, axes = plt.subplots(n_splits, 1, figsize=(14, 4 * n_splits))
    if n_splits == 1:
        axes = [axes]

    issues = []

    for s_idx, split in enumerate(splits):
        ax = axes[s_idx]
        split_records = [r for r in records if r["split"] == split]

        lines.append(f"\n  {split.upper()}:")

        x = np.arange(len(classes))
        width = 0.8 / max(len(sources), 1)

        for src_idx, source in enumerate(sources):
            counts = []
            for cls in classes:
                cnt = sum(
                    1
                    for r in split_records
                    if r["class_name"] == cls and r["source"] == source
                )
                counts.append(cnt)

            color = SOURCE_COLORS.get(source, "#95a5a6")
            ax.bar(
                x + src_idx * width,
                counts,
                width,
                label=source,
                color=color,
                alpha=0.8,
            )

        ax.set_xticks(x + width * (len(sources) - 1) / 2)
        ax.set_xticklabels(classes)
        ax.set_title(f"{split.upper()}: Fuentes por Clase")
        ax.set_ylabel("Cantidad")
        ax.legend(fontsize=8)

        for cls in classes:
            cls_sources = {r["source"] for r in split_records if r["class_name"] == cls}
            n_sources = len(cls_sources)
            counts_by_src = Counter(
                r["source"] for r in split_records if r["class_name"] == cls
            )
            detail = ", ".join(f"{s}={c}" for s, c in sorted(counts_by_src.items()))
            status = "✅" if n_sources >= 2 else "⚠️"
            lines.append(f"    {status} {cls:15s}: {n_sources} fuente(s) ({detail})")
            if n_sources < 2:
                issues.append(f"{split}/{cls}")

    if issues:
        lines.append(f"\n  ⚠️  {len(issues)} grupo(s) con 1 sola fuente: {issues}")
    else:
        lines.append("\n  ✅ Todas las clases en todos los splits tienen ≥2 fuentes")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "03_source_mixing.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    text = "\n".join(lines)
    print(text)
    return text


# ═══════════════════════════════════════════════════════════
#  4. RESOLUCIONES POST-PREPROCESSING
# ═══════════════════════════════════════════════════════════


def analyze_resolutions(records: list[dict], img_stats: dict) -> str:
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("4. RESOLUCIONES POST-PREPROCESSING")
    lines.append("=" * 80)

    # Desde stats calculados
    all_widths = []
    all_heights = []

    for split_stats in img_stats.values():
        for s in split_stats:
            all_widths.append(s["width"])
            all_heights.append(s["height"])

    if not all_widths:
        lines.append("  Sin datos de resolución")
        return "\n".join(lines)

    res_counts = Counter(
        f"{w}x{h}" for w, h in zip(all_widths, all_heights, strict=True)
    )

    lines.append("\n  Resoluciones encontradas:")
    for res, cnt in res_counts.most_common():
        pct = cnt / len(all_widths) * 100
        status = "✅" if "384" in res else "⚠️"
        lines.append(f"    {status} {res}: {cnt} ({pct:.1f}%)")

    uniform = len(res_counts) == 1
    if uniform:
        lines.append("\n  ✅ Todas las imágenes tienen la misma resolución")
    else:
        lines.append(
            f"\n  ⚠️  {len(res_counts)} resoluciones diferentes "
            f"→ verificar preprocesamiento"
        )

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    resolutions = list(res_counts.keys())
    counts = list(res_counts.values())
    ax.bar(resolutions, counts, color="#3498db", edgecolor="white")
    ax.set_title("Resoluciones Post-Preprocessing")
    ax.set_ylabel("Cantidad")
    ax.set_xlabel("Resolución")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "04_resolutions_check.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    text = "\n".join(lines)
    print(text)
    return text


# ═══════════════════════════════════════════════════════════
#  5. COLOR POST-CLAHE
# ═══════════════════════════════════════════════════════════


def analyze_color_post_clahe(records: list[dict], img_stats: dict) -> str:
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("5. COLOR POST-CLAHE (¿Normalización efectiva?)")
    lines.append("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Brillo por clase
    ax = axes[0, 0]
    classes = sorted({r["class_name"] for r in records})

    for cls in classes:
        cls_records = [r for r in records if r["class_name"] == cls]
        cls_paths = {r["file_path"] for r in cls_records}

        brightness = []
        for split_stats in img_stats.values():
            for s in split_stats:
                if s.get("path") in cls_paths:
                    brightness.append(s["brightness"])

        if brightness:
            ax.hist(
                brightness,
                bins=30,
                alpha=0.5,
                label=f"{cls} (μ={np.mean(brightness):.0f})",
                color=CLASS_COLORS.get(cls, "#95a5a6"),
            )
            lines.append(
                f"  {cls}: brillo={np.mean(brightness):.1f}±{np.std(brightness):.1f}"
            )

    ax.set_xlabel("Brillo (0-255)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Brillo por Clase (post-CLAHE)")
    ax.legend()

    # Brillo por fuente
    ax = axes[0, 1]
    sources = sorted({r["source"] for r in records})

    for src in sources:
        src_records = [r for r in records if r["source"] == src]
        src_paths = {r["file_path"] for r in src_records}

        brightness = []
        for split_stats in img_stats.values():
            for s in split_stats:
                if s.get("path") in src_paths:
                    brightness.append(s["brightness"])

        if brightness:
            ax.hist(
                brightness,
                bins=30,
                alpha=0.5,
                label=f"{src[:8]} (μ={np.mean(brightness):.0f})",
                color=SOURCE_COLORS.get(src, "#95a5a6"),
            )

    ax.set_xlabel("Brillo (0-255)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Brillo por Fuente (post-CLAHE)")
    ax.legend(fontsize=8)

    # Saturación por clase
    ax = axes[1, 0]
    for cls in classes:
        cls_records = [r for r in records if r["class_name"] == cls]
        cls_paths = {r["file_path"] for r in cls_records}

        sat = []
        for split_stats in img_stats.values():
            for s in split_stats:
                if s.get("path") in cls_paths:
                    sat.append(s["saturation"])

        if sat:
            ax.hist(
                sat,
                bins=30,
                alpha=0.5,
                label=f"{cls} (μ={np.mean(sat):.0f})",
                color=CLASS_COLORS.get(cls, "#95a5a6"),
            )

    ax.set_xlabel("Saturación (0-255)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Saturación por Clase")
    ax.legend()

    # R vs G scatter por fuente
    ax = axes[1, 1]
    for src in sources:
        src_records = [r for r in records if r["source"] == src]
        src_paths = {r["file_path"] for r in src_records}

        r_vals, g_vals = [], []
        for split_stats in img_stats.values():
            for s in split_stats:
                if s.get("path") in src_paths:
                    r_vals.append(s["r_mean"])
                    g_vals.append(s["g_mean"])

        if r_vals:
            ax.scatter(
                r_vals,
                g_vals,
                alpha=0.3,
                s=10,
                label=src[:8],
                color=SOURCE_COLORS.get(src, "#95a5a6"),
            )

    ax.set_xlabel("Red mean")
    ax.set_ylabel("Green mean")
    ax.set_title("Domain Gap Residual: R vs G (post-CLAHE)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "05_color_post_clahe.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    text = "\n".join(lines)
    print(text)
    return text


# ═══════════════════════════════════════════════════════════
#  6. ARTEFACTOS RESIDUALES
# ═══════════════════════════════════════════════════════════


def analyze_residual_artifacts(records: list[dict], img_stats: dict) -> str:
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("6. ARTEFACTOS RESIDUALES POST-PREPROCESSING")
    lines.append("=" * 80)

    all_stats_flat = []
    for split_stats in img_stats.values():
        all_stats_flat.extend(split_stats)

    if not all_stats_flat:
        lines.append("  Sin datos")
        return "\n".join(lines)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # % negro residual
    black_vals = [s["black_pct"] for s in all_stats_flat]
    ax = axes[0]
    ax.hist(black_vals, bins=50, color="#e74c3c", alpha=0.7, edgecolor="white")
    ax.set_xlabel("% Negro residual")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Negro Residual Post-Crop")
    ax.axvline(
        x=np.mean(black_vals),
        color="blue",
        linestyle="--",
        label=f"Media={np.mean(black_vals):.1f}%",
    )
    ax.legend()

    n_clean = sum(1 for b in black_vals if b <= 1.0)
    n_some = sum(1 for b in black_vals if 1.0 < b <= 5.0)
    n_bad = sum(1 for b in black_vals if b > 5.0)

    lines.append("\n  Negro residual:")
    lines.append(f"    ≤1%:   {n_clean:5d} ({n_clean / len(black_vals) * 100:.1f}%)")
    lines.append(f"    1-5%:  {n_some:5d} ({n_some / len(black_vals) * 100:.1f}%)")
    lines.append(f"    >5%:   {n_bad:5d} ({n_bad / len(black_vals) * 100:.1f}%)")
    lines.append(f"    Media: {np.mean(black_vals):.2f}%")
    lines.append(f"    Max:   {max(black_vals):.2f}%")

    # % negro por fuente
    ax = axes[1]
    sources = sorted({r["source"] for r in records})
    data_by_src = []
    labels_src = []
    colors_src = []

    for src in sources:
        src_paths = {r["file_path"] for r in records if r["source"] == src}
        vals = [s["black_pct"] for s in all_stats_flat if s.get("path") in src_paths]
        if vals:
            data_by_src.append(vals)
            labels_src.append(src[:10])
            colors_src.append(SOURCE_COLORS.get(src, "#95a5a6"))

    if data_by_src:
        bp = ax.boxplot(data_by_src, labels=labels_src, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors_src, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax.set_ylabel("% Negro")
    ax.set_title("Negro por Fuente")
    ax.tick_params(axis="x", rotation=15)

    # % verde residual
    green_vals = [s["green_pct"] for s in all_stats_flat]
    ax = axes[2]
    ax.hist(green_vals, bins=50, color="#2ecc71", alpha=0.7, edgecolor="white")
    ax.set_xlabel("% Verde residual")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Verde Residual (Scope Guides)")
    ax.axvline(
        x=np.mean(green_vals),
        color="blue",
        linestyle="--",
        label=f"Media={np.mean(green_vals):.2f}%",
    )
    ax.legend()

    lines.append("\n  Verde residual (scope guides):")
    lines.append(f"    Media: {np.mean(green_vals):.3f}%")
    lines.append(f"    Max:   {max(green_vals):.3f}%")

    n_green = sum(1 for g in green_vals if g > 1.0)
    if n_green > 0:
        lines.append(f"    ⚠️  {n_green} imágenes con >1% verde residual")
    else:
        lines.append("    ✅ Sin scope guides residuales")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "06_residual_artifacts.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    text = "\n".join(lines)
    print(text)
    return text


# ═══════════════════════════════════════════════════════════
#  7. FEATURES POR CLASE (¿hay sesgo visual?)
# ═══════════════════════════════════════════════════════════


def analyze_features_by_class(records: list[dict], img_stats: dict) -> str:
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("7-8. FEATURES POR CLASE Y FUENTE")
    lines.append("=" * 80)
    lines.append("  Si una feature separa clases → modelo puede usarla como shortcut")
    lines.append("  Si una feature separa fuentes → domain gap no cerrado")

    all_stats_flat = []
    for split_stats in img_stats.values():
        all_stats_flat.extend(split_stats)

    if not all_stats_flat:
        return "\n".join(lines)

    classes = sorted({r["class_name"] for r in records})
    sources = sorted({r["source"] for r in records})
    features = ["brightness", "contrast", "saturation", "sharpness"]

    fig, axes = plt.subplots(2, len(features), figsize=(5 * len(features), 10))

    # Fila 1: Por clase
    for f_idx, feat in enumerate(features):
        ax = axes[0, f_idx]
        data = []
        labels = []
        colors = []

        for cls in classes:
            cls_paths = {r["file_path"] for r in records if r["class_name"] == cls}
            vals = [s[feat] for s in all_stats_flat if s.get("path") in cls_paths]
            if vals:
                data.append(vals)
                labels.append(cls[:8])
                colors.append(CLASS_COLORS.get(cls, "#95a5a6"))

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors, strict=True):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
        ax.set_title(f"{feat} (por clase)")
        ax.tick_params(axis="x", rotation=15)

    # Fila 2: Por fuente
    for f_idx, feat in enumerate(features):
        ax = axes[1, f_idx]
        data = []
        labels = []
        colors = []

        for src in sources:
            src_paths = {r["file_path"] for r in records if r["source"] == src}
            vals = [s[feat] for s in all_stats_flat if s.get("path") in src_paths]
            if vals:
                data.append(vals)
                labels.append(src[:8])
                colors.append(SOURCE_COLORS.get(src, "#95a5a6"))

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors, strict=True):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
        ax.set_title(f"{feat} (por fuente)")
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle(
        "Features por Clase (fila 1) y Fuente (fila 2)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "08_features_by_class_source.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    # Log estadísticas
    for feat in features:
        lines.append(f"\n  {feat}:")
        for cls in classes:
            cls_paths = {r["file_path"] for r in records if r["class_name"] == cls}
            vals = [s[feat] for s in all_stats_flat if s.get("path") in cls_paths]
            if vals:
                lines.append(f"    {cls:15s}: {np.mean(vals):.1f}±{np.std(vals):.1f}")

    text = "\n".join(lines)
    print(text)
    return text


# ═══════════════════════════════════════════════════════════
#  9. MUESTRAS VISUALES POR CLASE × FUENTE
# ═══════════════════════════════════════════════════════════


def plot_samples_grid(records: list[dict]):
    """Grid: filas=clases, columnas=fuentes."""
    classes = sorted({r["class_name"] for r in records})
    sources = sorted({r["source"] for r in records})

    samples_per_cell = 2
    n_rows = len(classes) * samples_per_cell
    n_cols = len(sources)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    rng = np.random.RandomState(42)

    row = 0
    for cls in classes:
        for sample_idx in range(samples_per_cell):
            for col, src in enumerate(sources):
                pool = [
                    r
                    for r in records
                    if r["class_name"] == cls and r["source"] == src and r["exists"]
                ]

                ax = axes[row, col]

                if sample_idx < len(pool):
                    r = pool[rng.randint(0, len(pool))]
                    img = cv2.imread(r["file_path"])
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        ax.imshow(img_rgb)
                        ax.set_title(
                            f"{cls}/{src[:6]}",
                            fontsize=8,
                        )
                    else:
                        ax.text(0.5, 0.5, "Error", ha="center", va="center")

                ax.axis("off")

            row += 1

    plt.suptitle(
        "Muestras: Clase × Fuente (procesadas)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "10_samples_class_source.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  ✅ 10_samples_class_source.png")


# ═══════════════════════════════════════════════════════════
#  10. TISSUE-ONLY COMPARISON (si existe)
# ═══════════════════════════════════════════════════════════


def analyze_tissue_only(records: list[dict]) -> str:
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("10. TISSUE-ONLY vs STANDARD")
    lines.append("=" * 80)

    if not TISSUE_DIR.exists():
        lines.append("  ℹ️  colon_processed_tissue_only/ no existe, saltando")
        text = "\n".join(lines)
        print(text)
        return text

    # Contar crops
    tissue_counts = defaultdict(int)
    for class_dir in sorted(TISSUE_DIR.iterdir()):
        if class_dir.is_dir() and class_dir.name != "masks":
            n = sum(
                1 for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
            )
            tissue_counts[class_dir.name] = n

    total_tissue = sum(tissue_counts.values())
    total_standard = len(records)

    lines.append(f"\n  Standard: {total_standard} imágenes")
    lines.append(f"  Tissue-only: {total_tissue} crops")
    lines.append(
        f"  Ratio: {total_tissue / max(total_standard, 1):.1f}x "
        f"(≈{total_tissue / max(total_standard, 1):.0f} crops/imagen)"
    )

    for cls, cnt in sorted(tissue_counts.items()):
        lines.append(f"    {cls}: {cnt} crops")

    # Comparar negro residual
    sample_standard = []
    sample_tissue = []

    # Standard samples
    existing = [r for r in records if r["exists"]]
    if existing:
        idxs = sample_indices(len(existing), 200)
        for i in idxs:
            img = cv2.imread(existing[i]["file_path"])
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sample_standard.append((gray < 15).sum() / gray.size * 100)

    # Tissue samples
    tissue_images = []
    for class_dir in sorted(TISSUE_DIR.iterdir()):
        if class_dir.is_dir() and class_dir.name != "masks":
            for f in class_dir.iterdir():
                if f.suffix.lower() in IMAGE_EXTENSIONS:
                    tissue_images.append(f)

    if tissue_images:
        idxs = sample_indices(len(tissue_images), 200)
        for i in idxs:
            img = cv2.imread(str(tissue_images[i]))
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sample_tissue.append((gray < 15).sum() / gray.size * 100)

    if sample_standard and sample_tissue:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.hist(
            sample_standard,
            bins=40,
            alpha=0.6,
            label=f"Standard (μ={np.mean(sample_standard):.1f}%)",
            color="#3498db",
        )
        ax.hist(
            sample_tissue,
            bins=40,
            alpha=0.6,
            label=f"Tissue-only (μ={np.mean(sample_tissue):.1f}%)",
            color="#e74c3c",
        )
        ax.set_xlabel("% Negro")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Negro Residual: Standard vs Tissue-Only")
        ax.legend()

        # Muestras lado a lado
        ax = axes[1]
        if existing and tissue_images:
            # Elegir una imagen y su tissue crop
            sample_rec = existing[0]
            stem = Path(sample_rec["file_path"]).stem
            cls = Path(sample_rec["file_path"]).parent.name
            crop0 = TISSUE_DIR / cls / f"{stem}_crop0.jpg"

            std_img = cv2.imread(sample_rec["file_path"])
            if std_img is not None and crop0.exists():
                tis_img = cv2.imread(str(crop0))
                if tis_img is not None:
                    combo = np.hstack(
                        [
                            cv2.cvtColor(
                                cv2.resize(std_img, (384, 384)),
                                cv2.COLOR_BGR2RGB,
                            ),
                            cv2.cvtColor(
                                cv2.resize(tis_img, (384, 384)),
                                cv2.COLOR_BGR2RGB,
                            ),
                        ]
                    )
                    ax.imshow(combo)
                    ax.set_title("Standard (izq) vs Tissue-Only (der)")
            ax.axis("off")

        lines.append("\n  Negro residual (muestra de 200):")
        lines.append(f"    Standard:    {np.mean(sample_standard):.2f}%")
        lines.append(f"    Tissue-only: {np.mean(sample_tissue):.2f}%")

        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / "12_tissue_only_comparison.png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()

    text = "\n".join(lines)
    print(text)
    return text


# ═══════════════════════════════════════════════════════════
#  11. RESUMEN Y RECOMENDACIONES
# ═══════════════════════════════════════════════════════════


def generate_summary(records: list[dict], img_stats: dict) -> str:
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("11. RESUMEN Y RECOMENDACIONES")
    lines.append("=" * 80)

    # Checks
    checks = []

    # Balance
    classes = sorted({r["class_name"] for r in records})
    class_counts = Counter(r["class_name"] for r in records)
    if class_counts:
        ratio = max(class_counts.values()) / max(min(class_counts.values()), 1)
        ok = ratio < 1.5
        checks.append(("Balance de clases", ok, f"ratio max/min={ratio:.2f}"))

    # Source mixing
    splits = ["train", "val", "test"]
    mixing_ok = True
    for split in splits:
        for cls in classes:
            n_sources = len(
                {
                    r["source"]
                    for r in records
                    if r["split"] == split and r["class_name"] == cls
                }
            )
            if n_sources < 2:
                mixing_ok = False
    checks.append(("Source mixing (≥2 fuentes/clase/split)", mixing_ok, ""))

    # Split proportions
    split_counts = Counter(r["split"] for r in records)
    total = len(records)
    train_pct = split_counts.get("train", 0) / total * 100
    checks.append(("Split 70/15/15", 65 <= train_pct <= 75, f"train={train_pct:.1f}%"))

    # No missing files
    missing = sum(1 for r in records if not r["exists"])
    checks.append(("Archivos completos", missing == 0, f"{missing} faltantes"))

    # Artifacts
    all_stats_flat = []
    for split_stats in img_stats.values():
        all_stats_flat.extend(split_stats)

    if all_stats_flat:
        avg_black = np.mean([s["black_pct"] for s in all_stats_flat])
        checks.append(("Negro residual <5%", avg_black < 5, f"media={avg_black:.1f}%"))

    for name, ok, detail in checks:
        status = "✅" if ok else "⚠️"
        lines.append(f"  {status} {name:45s} {detail}")

    # Score
    score = sum(1 for _, ok, _ in checks if ok)
    total_checks = len(checks)
    lines.append(f"\n  📊 Score: {score}/{total_checks} checks pasados")

    if score == total_checks:
        lines.append("  ✅ Dataset listo para entrenamiento")
    else:
        lines.append("  ⚠️  Revisar los checks que fallaron")

    text = "\n".join(lines)
    print(text)
    return text


# ═══════════════════════════════════════════════════════════
#  PIPELINE
# ═══════════════════════════════════════════════════════════


def run_eda():
    """EDA completo del dataset procesado."""
    print("=" * 80)
    print("EDA: DATASET PROCESADO (colon_processed/)")
    print(f"  Input: {PROCESSED_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 80)

    # ── 1. Cargar records ──
    print("\n📂 Cargando records de la DB...")
    records = load_and_enrich_records()
    print(f"  → {len(records)} records cargados")

    if not records:
        print("  ❌ Sin records en DB. Ejecuta organize_multi_dataset.py")
        return

    # ── 2. Calcular estadísticas de imágenes ──
    print("\n📊 Calculando estadísticas de imágenes procesadas...")
    img_stats: dict[str, list[dict]] = {}

    for split in ["train", "val", "test"]:
        split_records = [r for r in records if r["split"] == split and r["exists"]]
        idxs = sample_indices(len(split_records), 300)
        sampled = [split_records[i] for i in idxs]

        print(
            f"  {split}: analizando {len(sampled)} de {len(split_records)} imágenes..."
        )

        stats = []
        for r in tqdm(sampled, desc=f"    {split}"):
            s = compute_image_stats(r["file_path"])
            if s:
                s["path"] = r["file_path"]
                stats.append(s)

        img_stats[split] = stats

    # ── 3. Análisis ──
    report_parts = []

    report_parts.append(analyze_inventory(records))
    report_parts.append(analyze_class_balance(records))
    report_parts.append(analyze_source_mixing(records))
    report_parts.append(analyze_resolutions(records, img_stats))
    report_parts.append(analyze_color_post_clahe(records, img_stats))
    report_parts.append(analyze_residual_artifacts(records, img_stats))
    report_parts.append(analyze_features_by_class(records, img_stats))

    print("\n🖼️ Generando grid de muestras...")
    plot_samples_grid(records)

    report_parts.append(analyze_tissue_only(records))
    report_parts.append(generate_summary(records, img_stats))

    # ── 4. Guardar ──
    full_report = "\n".join(report_parts)
    report_path = OUTPUT_DIR / "eda_processed_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"\n📄 Reporte: {report_path}")

    # JSON stats
    json_out = {}
    for split, stats in img_stats.items():
        if not stats:
            continue
        json_out[split] = {
            "n_analyzed": len(stats),
            "brightness": {
                "mean": round(np.mean([s["brightness"] for s in stats]), 2),
                "std": round(np.std([s["brightness"] for s in stats]), 2),
            },
            "contrast": {
                "mean": round(np.mean([s["contrast"] for s in stats]), 2),
                "std": round(np.std([s["contrast"] for s in stats]), 2),
            },
            "black_pct": {
                "mean": round(np.mean([s["black_pct"] for s in stats]), 2),
                "max": round(max(s["black_pct"] for s in stats), 2),
            },
            "sharpness": {
                "mean": round(np.mean([s["sharpness"] for s in stats]), 1),
                "median": round(float(np.median([s["sharpness"] for s in stats])), 1),
            },
        }

    stats_path = OUTPUT_DIR / "eda_processed_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"📊 Stats: {stats_path}")

    # Resumen
    print("\n" + "=" * 80)
    print("✅ EDA PROCESADO COMPLETADO")
    print(f"  📁 Output: {OUTPUT_DIR}")
    print("  Archivos:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    ✅ {f.name:45s} ({size_kb:.0f} KB)")
    print("=" * 80)


if __name__ == "__main__":
    run_eda()
