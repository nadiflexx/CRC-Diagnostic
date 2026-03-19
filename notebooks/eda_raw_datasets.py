"""
notebooks/eda_raw_datasets.py

EDA completo de los 4 datasets RAW antes de organizar.

Análisis:
  1. Inventario: cuántas imágenes por dataset y subcarpeta
  2. Resoluciones: distribución de tamaños, aspect ratios
  3. Color: histogramas RGB/HSV por dataset (detectar domain gap)
  4. Intensidad: brillo, contraste, saturación
  5. Artefactos: % de negro (bordes), verde (scope guides)
  6. FOV: circular vs rectangular por dataset
  7. Calidad: varianza de Laplacian (nitidez/blur)
  8. Muestras visuales: grid por dataset y subcarpeta
  9. Comparativa inter-dataset: ¿qué tan diferentes son?
  10. Resumen clínico: mapeo a clases del proyecto

Genera:
  notebooks/eda_output/
  ├── 01_inventory.png
  ├── 02_resolutions.png
  ├── 03_aspect_ratios.png
  ├── 04_color_histograms.png
  ├── 05_hsv_distributions.png
  ├── 06_black_border_analysis.png
  ├── 07_green_artifact_analysis.png
  ├── 08_fov_types.png
  ├── 09_sharpness.png
  ├── 10_samples_grid.png
  ├── 11_domain_gap.png
  ├── 12_class_mapping.png
  ├── eda_report.txt
  └── eda_stats.json
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import warnings

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.config import constants
from src.config.paths import paths

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════


OUTPUT_DIR = paths.NOTEBOOKS / "eda_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  UTILIDADES
# ═══════════════════════════════════════════════════════════


def glob_images(directory: Path) -> list[Path]:
    """Busca imágenes recursivamente."""
    images: list[Path] = []
    if not directory.exists():
        return images
    for f in sorted(directory.rglob("*")):
        if f.is_file() and f.suffix.lower() in constants.IMAGE_EXTENSIONS:
            images.append(f)
    return images


def glob_images_flat(directory: Path) -> list[Path]:
    """Busca imágenes solo en el directorio (no recursivo)."""
    images: list[Path] = []
    if not directory.exists():
        return images
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix.lower() in constants.IMAGE_EXTENSIONS:
            images.append(f)
    return images


def sample_images(images: list[Path], n: int = 200, seed: int = 42) -> list[Path]:
    """Muestra aleatoria de N imágenes."""
    rng = np.random.RandomState(seed)
    if len(images) <= n:
        return images
    indices = rng.choice(len(images), n, replace=False)
    return [images[i] for i in sorted(indices)]


def compute_image_stats(img_path: Path) -> dict | None:
    """Calcula estadísticas de una imagen."""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Brillo y contraste
        brightness = float(gray.mean())
        contrast = float(gray.std())

        # Saturación
        saturation = float(hsv[:, :, 1].mean())

        # Nitidez (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())

        # % negro (bordes)
        black_pct = float((gray < 15).sum() / gray.size * 100)

        # % verde (scope guides)
        green_lower = np.array([30, 50, 50])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_pct = float(green_mask.sum() / (255.0 * gray.size) * 100)

        # FOV type
        cs_h = max(h // 10, 5)
        cs_w = max(w // 10, 5)
        corners = [
            gray[:cs_h, :cs_w],
            gray[:cs_h, w - cs_w :],
            gray[h - cs_h :, :cs_w],
            gray[h - cs_h :, w - cs_w :],
        ]
        dark_corners = sum(1 for c in corners if np.mean(c) < 15)
        fov_type = "circular" if dark_corners >= 3 else "rectangular"

        # Histograma RGB (simplificado: medias por canal)
        b_mean = float(img[:, :, 0].mean())
        g_mean = float(img[:, :, 1].mean())
        r_mean = float(img[:, :, 2].mean())

        # Hue dominante
        hue_mean = float(hsv[:, :, 0].mean())

        return {
            "path": str(img_path),
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 3),
            "pixels": w * h,
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "saturation": round(saturation, 2),
            "sharpness": round(sharpness, 2),
            "black_pct": round(black_pct, 2),
            "green_pct": round(green_pct, 2),
            "fov_type": fov_type,
            "r_mean": round(r_mean, 2),
            "g_mean": round(g_mean, 2),
            "b_mean": round(b_mean, 2),
            "hue_mean": round(hue_mean, 2),
            "file_size_kb": round(img_path.stat().st_size / 1024, 1),
        }
    except Exception as e:
        print(f"  ⚠️ Error: {img_path.name}: {e}")
        return None


# ═══════════════════════════════════════════════════════════
#  DESCUBRIMIENTO DE ESTRUCTURA
# ═══════════════════════════════════════════════════════════


def discover_dataset_structure(name: str, config: dict) -> dict:
    """Descubre la estructura real de un dataset."""
    base = config["base_path"]
    result = {
        "name": name,
        "exists": base.exists(),
        "base_path": str(base),
        "folders": {},
        "total_images": 0,
        "all_images": [],
    }

    if not base.exists():
        return result

    # Recorrer todas las subcarpetas
    for subdir in sorted(base.rglob("*")):
        if not subdir.is_dir():
            continue

        images = glob_images_flat(subdir)
        if not images:
            continue

        rel = str(subdir.relative_to(base))
        result["folders"][rel] = {
            "path": str(subdir),
            "n_images": len(images),
            "extensions": dict(Counter(f.suffix.lower() for f in images)),
        }
        result["total_images"] += len(images)
        result["all_images"].extend(images)

    # Imágenes en el directorio raíz
    root_images = glob_images_flat(base)
    if root_images:
        result["folders"]["."] = {
            "path": str(base),
            "n_images": len(root_images),
            "extensions": dict(Counter(f.suffix.lower() for f in root_images)),
        }
        result["total_images"] += len(root_images)
        result["all_images"].extend(root_images)

    return result


# ═══════════════════════════════════════════════════════════
#  ANÁLISIS 1: INVENTARIO
# ═══════════════════════════════════════════════════════════


def analyze_inventory(structures: dict) -> str:
    """Inventario completo de todos los datasets."""
    lines = []
    lines.append("=" * 80)
    lines.append("1. INVENTARIO DE DATASETS RAW")
    lines.append("=" * 80)

    total_all = 0

    for name, struct in structures.items():
        config = constants.DATASET_CONFIG[name]
        lines.append(f"\n{'─' * 60}")
        lines.append(f"📦 {config['description']}")
        lines.append(f"   Path: {struct['base_path']}")
        lines.append(f"   Existe: {'✅' if struct['exists'] else '❌'}")
        lines.append(f"   Total imágenes: {struct['total_images']:,}")

        if struct["folders"]:
            lines.append("   Subcarpetas:")
            # Ordenar por número de imágenes descendente
            sorted_folders = sorted(
                struct["folders"].items(),
                key=lambda x: x[1]["n_images"],
                reverse=True,
            )
            for folder, info in sorted_folders:
                exts = ", ".join(f"{k}={v}" for k, v in info["extensions"].items())
                lines.append(
                    f"     📁 {folder:50s} {info['n_images']:6d} imgs ({exts})"
                )

        total_all += struct["total_images"]

    lines.append(f"\n{'═' * 60}")
    lines.append(f"📊 GRAN TOTAL: {total_all:,} imágenes en raw")
    lines.append("═" * 60)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart: imágenes por dataset
    names = []
    counts = []
    colors = []
    for name, struct in structures.items():
        if struct["total_images"] > 0:
            names.append(name)
            counts.append(struct["total_images"])
            colors.append(constants.DATASET_CONFIG[name]["color"])

    axes[0].barh(names, counts, color=colors, edgecolor="white")
    for i, (_, c) in enumerate(zip(names, counts, strict=True)):
        axes[0].text(c + max(counts) * 0.01, i, f"{c:,}", va="center")
    axes[0].set_xlabel("Número de imágenes")
    axes[0].set_title("Imágenes por Dataset (RAW)")
    axes[0].invert_yaxis()

    # Pie chart: proporción
    axes[1].pie(
        counts,
        labels=names,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[1].set_title("Proporción de imágenes")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "01_inventory.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    text = "\n".join(lines)
    print(text)
    return text


# ═══════════════════════════════════════════════════════════
#  ANÁLISIS 2-3: RESOLUCIONES Y ASPECT RATIOS
# ═══════════════════════════════════════════════════════════


def analyze_resolutions(all_stats: dict) -> str:
    """Distribución de resoluciones y aspect ratios."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("2-3. RESOLUCIONES Y ASPECT RATIOS")
    lines.append("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for _, (dataset, stats) in enumerate(all_stats.items()):
        if not stats:
            continue

        widths = [s["width"] for s in stats]
        heights = [s["height"] for s in stats]
        ratios = [s["aspect_ratio"] for s in stats]

        color = constants.DATASET_CONFIG[dataset]["color"]

        lines.append(f"\n  {dataset}:")
        lines.append(
            f"    Ancho:  min={min(widths)}, max={max(widths)}, "
            f"media={np.mean(widths):.0f}"
        )
        lines.append(
            f"    Alto:   min={min(heights)}, max={max(heights)}, "
            f"media={np.mean(heights):.0f}"
        )
        lines.append(
            f"    Ratio:  min={min(ratios):.2f}, max={max(ratios):.2f}, "
            f"media={np.mean(ratios):.2f}"
        )

        unique_res = Counter(f"{w}x{h}" for w, h in zip(widths, heights, strict=True))
        top3 = unique_res.most_common(3)
        lines.append("    Top resoluciones: " + ", ".join(f"{r}({c})" for r, c in top3))

    # Plot resoluciones (scatter)
    ax = axes[0, 0]
    for dataset, stats in all_stats.items():
        if not stats:
            continue
        w = [s["width"] for s in stats]
        h = [s["height"] for s in stats]
        ax.scatter(
            w,
            h,
            alpha=0.3,
            s=10,
            label=dataset,
            color=constants.DATASET_CONFIG[dataset]["color"],
        )
    ax.set_xlabel("Ancho (px)")
    ax.set_ylabel("Alto (px)")
    ax.set_title("Resoluciones por Dataset")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot aspect ratios (histograma)
    ax = axes[0, 1]
    for dataset, stats in all_stats.items():
        if not stats:
            continue
        ratios = [s["aspect_ratio"] for s in stats]
        ax.hist(
            ratios,
            bins=30,
            alpha=0.5,
            label=dataset,
            color=constants.DATASET_CONFIG[dataset]["color"],
        )
    ax.set_xlabel("Aspect Ratio (ancho/alto)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de Aspect Ratios")
    ax.legend()
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.5, label="Cuadrado")

    # Plot megapíxeles
    ax = axes[1, 0]
    data_mp = []
    labels_mp = []
    colors_mp = []
    for dataset, stats in all_stats.items():
        if not stats:
            continue
        mp = [s["pixels"] / 1e6 for s in stats]
        data_mp.append(mp)
        labels_mp.append(dataset)
        colors_mp.append(constants.DATASET_CONFIG[dataset]["color"])
    if data_mp:
        bp = ax.boxplot(data_mp, labels=labels_mp, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors_mp, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax.set_ylabel("Megapíxeles")
    ax.set_title("Distribución de Tamaño (MP)")

    # Plot file size
    ax = axes[1, 1]
    data_fs = []
    labels_fs = []
    colors_fs = []
    for dataset, stats in all_stats.items():
        if not stats:
            continue
        fs = [s["file_size_kb"] for s in stats]
        data_fs.append(fs)
        labels_fs.append(dataset)
        colors_fs.append(constants.DATASET_CONFIG[dataset]["color"])
    if data_fs:
        bp = ax.boxplot(data_fs, labels=labels_fs, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors_fs, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax.set_ylabel("Tamaño archivo (KB)")
    ax.set_title("Distribución de Tamaño de Archivo")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "02_resolutions.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    text = "\n".join(lines)
    print(text)
    return text


# ═══════════════════════════════════════════════════════════
#  ANÁLISIS 4-5: COLOR (RGB + HSV)
# ═══════════════════════════════════════════════════════════


def analyze_color(all_stats: dict) -> str:
    """Distribución de color por dataset (domain gap)."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("4-5. ANÁLISIS DE COLOR (Domain Gap)")
    lines.append("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # R, G, B medias por dataset
    ax = axes[0, 0]
    for dataset, stats in all_stats.items():
        if not stats:
            continue
        r = [s["r_mean"] for s in stats]
        g = [s["g_mean"] for s in stats]
        b = [s["b_mean"] for s in stats]
        means = [np.mean(r), np.mean(g), np.mean(b)]
        stds = [np.std(r), np.std(g), np.std(b)]

        lines.append(f"\n  {dataset}:")
        lines.append(
            f"    R: {means[0]:.1f}±{stds[0]:.1f}  "
            f"G: {means[1]:.1f}±{stds[1]:.1f}  "
            f"B: {means[2]:.1f}±{stds[2]:.1f}"
        )

    # Scatter R vs G (domain gap visual)
    for dataset, stats in all_stats.items():
        if not stats:
            continue
        r = [s["r_mean"] for s in stats]
        g = [s["g_mean"] for s in stats]
        ax.scatter(
            r,
            g,
            alpha=0.3,
            s=10,
            label=dataset,
            color=constants.DATASET_CONFIG[dataset]["color"],
        )
    ax.set_xlabel("Red mean")
    ax.set_ylabel("Green mean")
    ax.set_title("Domain Gap: R vs G")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Brillo por dataset
    ax = axes[0, 1]
    data_br = []
    labels_br = []
    colors_br = []
    for dataset, stats in all_stats.items():
        if not stats:
            continue
        br = [s["brightness"] for s in stats]
        data_br.append(br)
        labels_br.append(dataset)
        colors_br.append(constants.DATASET_CONFIG[dataset]["color"])

        lines.append(f"    Brillo: {np.mean(br):.1f}±{np.std(br):.1f}")

    if data_br:
        bp = ax.boxplot(data_br, labels=labels_br, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors_br, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax.set_ylabel("Brillo (0-255)")
    ax.set_title("Distribución de Brillo")

    # Saturación por dataset
    ax = axes[1, 0]
    data_sat = []
    labels_sat = []
    colors_sat = []
    for dataset, stats in all_stats.items():
        if not stats:
            continue
        sat = [s["saturation"] for s in stats]
        data_sat.append(sat)
        labels_sat.append(dataset)
        colors_sat.append(constants.DATASET_CONFIG[dataset]["color"])

        lines.append(f"    Saturación: {np.mean(sat):.1f}±{np.std(sat):.1f}")

    if data_sat:
        bp = ax.boxplot(data_sat, labels=labels_sat, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors_sat, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax.set_ylabel("Saturación (0-255)")
    ax.set_title("Distribución de Saturación")

    # Hue por dataset
    ax = axes[1, 1]
    for dataset, stats in all_stats.items():
        if not stats:
            continue
        hue = [s["hue_mean"] for s in stats]
        ax.hist(
            hue,
            bins=30,
            alpha=0.5,
            label=dataset,
            color=constants.DATASET_CONFIG[dataset]["color"],
        )

        lines.append(f"    Hue: {np.mean(hue):.1f}±{np.std(hue):.1f}")

    ax.set_xlabel("Hue medio (0-180)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de Hue (tono dominante)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "04_color_histograms.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    text = "\n".join(lines)
    print(text)
    return text


# ═══════════════════════════════════════════════════════════
#  ANÁLISIS 6-7: ARTEFACTOS (negro + verde)
# ═══════════════════════════════════════════════════════════


def analyze_artifacts(all_stats: dict) -> str:
    """Análisis de artefactos: bordes negros y scope guides."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("6-7. ARTEFACTOS: Bordes Negros + Scope Guides")
    lines.append("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # % negro por dataset
    ax = axes[0, 0]
    data_black = []
    labels_black = []
    colors_black = []

    for dataset, stats in all_stats.items():
        if not stats:
            continue
        black = [s["black_pct"] for s in stats]
        data_black.append(black)
        labels_black.append(dataset)
        colors_black.append(constants.DATASET_CONFIG[dataset]["color"])

        n_significant = sum(1 for b in black if b > 5)
        lines.append(f"\n  {dataset}:")
        lines.append(f"    Negro: media={np.mean(black):.1f}%, max={max(black):.1f}%")
        lines.append(
            f"    Con >5% negro: {n_significant}/{len(black)} "
            f"({n_significant / len(black) * 100:.1f}%)"
        )

    if data_black:
        bp = ax.boxplot(data_black, labels=labels_black, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors_black, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax.set_ylabel("% Negro")
    ax.set_title("Bordes Negros por Dataset")
    ax.axhline(y=5, color="red", linestyle="--", alpha=0.5)

    # Histograma de % negro
    ax = axes[0, 1]
    for dataset, stats in all_stats.items():
        if not stats:
            continue
        black = [s["black_pct"] for s in stats]
        ax.hist(
            black,
            bins=50,
            alpha=0.5,
            label=dataset,
            color=str(constants.DATASET_CONFIG[dataset]["color"]),
        )
    ax.set_xlabel("% Negro")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de % Negro (bordes)")
    ax.legend()
    ax.set_xlim(0, 50)

    # % verde por dataset
    ax = axes[1, 0]
    data_green = []
    labels_green = []
    colors_green = []

    for dataset, stats in all_stats.items():
        if not stats:
            continue
        green = [s["green_pct"] for s in stats]
        data_green.append(green)
        labels_green.append(dataset)
        colors_green.append(constants.DATASET_CONFIG[dataset]["color"])

        n_green = sum(1 for g in green if g > 1)
        lines.append(f"    Verde: media={np.mean(green):.2f}%, max={max(green):.2f}%")
        lines.append(
            f"    Con scope guide (>1%): {n_green}/{len(green)} "
            f"({n_green / len(green) * 100:.1f}%)"
        )

    if data_green:
        bp = ax.boxplot(data_green, labels=labels_green, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors_green, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax.set_ylabel("% Verde")
    ax.set_title("Scope Guides (verde) por Dataset")

    # FOV types
    ax = axes[1, 1]
    fov_data = {}
    for dataset, stats in all_stats.items():
        if not stats:
            continue
        fov_counts = Counter(s["fov_type"] for s in stats)
        fov_data[dataset] = fov_counts
        total = len(stats)
        circ = fov_counts.get("circular", 0)
        rect = fov_counts.get("rectangular", 0)
        lines.append(
            f"    FOV: {circ} circular ({circ / total * 100:.1f}%), "
            f"{rect} rectangular ({rect / total * 100:.1f}%)"
        )

    # Stacked bar
    datasets_with_data = [d for d in fov_data if fov_data[d]]
    if datasets_with_data:
        circ_counts = [fov_data[d].get("circular", 0) for d in datasets_with_data]
        rect_counts = [fov_data[d].get("rectangular", 0) for d in datasets_with_data]
        x = np.arange(len(datasets_with_data))
        ax.bar(x, circ_counts, label="Circular", color="#e74c3c", alpha=0.7)
        ax.bar(
            x,
            rect_counts,
            bottom=circ_counts,
            label="Rectangular",
            color="#3498db",
            alpha=0.7,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(datasets_with_data, rotation=15)
        ax.set_ylabel("Número de imágenes")
        ax.set_title("Tipo de FOV por Dataset")
        ax.legend()

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "06_artifacts.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    text = "\n".join(lines)
    print(text)
    return text


# ═══════════════════════════════════════════════════════════
#  ANÁLISIS 8: NITIDEZ
# ═══════════════════════════════════════════════════════════


def analyze_sharpness(all_stats: dict) -> str:
    """Distribución de nitidez (Laplacian variance)."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("8. NITIDEZ (Laplacian Variance)")
    lines.append("=" * 80)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for dataset, stats in all_stats.items():
        if not stats:
            continue
        sharp = [min(s["sharpness"], 5000) for s in stats]  # Cap outliers
        ax.hist(
            sharp,
            bins=50,
            alpha=0.5,
            label=dataset,
            color=str(constants.DATASET_CONFIG[dataset]["color"]),
        )

        n_blurry = sum(1 for s in sharp if s < 100)
        lines.append(f"\n  {dataset}:")
        lines.append(
            f"    Nitidez: media={np.mean(sharp):.0f}, mediana={np.median(sharp):.0f}"
        )
        lines.append(
            f"    Borrosas (<100): {n_blurry}/{len(sharp)} "
            f"({n_blurry / len(sharp) * 100:.1f}%)"
        )

    ax.set_xlabel("Laplacian Variance (mayor = más nítida)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de Nitidez")
    ax.legend()
    ax.axvline(x=100, color="red", linestyle="--", alpha=0.5, label="Umbral blur")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "09_sharpness.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    text = "\n".join(lines)
    print(text)
    return text


# ═══════════════════════════════════════════════════════════
#  ANÁLISIS 9: MUESTRAS VISUALES
# ═══════════════════════════════════════════════════════════


def plot_sample_grid(structures: dict):
    """Grid de muestras por dataset."""
    datasets_with_images = {
        name: struct for name, struct in structures.items() if struct["all_images"]
    }

    n_datasets = len(datasets_with_images)
    if n_datasets == 0:
        return

    samples_per = 6
    fig, axes = plt.subplots(
        n_datasets, samples_per, figsize=(3 * samples_per, 3.5 * n_datasets)
    )

    if n_datasets == 1:
        axes = axes[np.newaxis, :]

    for row, (name, struct) in enumerate(datasets_with_images.items()):
        images = struct["all_images"]
        sampled = sample_images(images, samples_per, seed=123)

        for col in range(samples_per):
            ax = axes[row, col]
            if col < len(sampled):
                img = cv2.imread(str(sampled[col]))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Resize para display
                    h, w = img_rgb.shape[:2]
                    display_size = 300
                    scale = display_size / max(h, w)
                    img_display = cv2.resize(
                        img_rgb,
                        (int(w * scale), int(h * scale)),
                    )
                    ax.imshow(img_display)
                    ax.set_title(
                        f"{w}×{h}",
                        fontsize=8,
                    )
            ax.axis("off")

            if col == 0:
                ax.set_ylabel(name, fontsize=10, fontweight="bold")

    plt.suptitle(
        "Muestras por Dataset (RAW)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "10_samples_grid.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  ✅ 10_samples_grid.png")


# ═══════════════════════════════════════════════════════════
#  ANÁLISIS 10: DOMAIN GAP INTER-DATASET
# ═══════════════════════════════════════════════════════════


def analyze_domain_gap(all_stats: dict) -> str:
    """Cuantifica el domain gap entre datasets."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("10. DOMAIN GAP INTER-DATASET")
    lines.append("=" * 80)
    lines.append("  ¿Qué tan diferentes son las imágenes de cada fuente?")
    lines.append("  Domain gap alto = el modelo puede usar la fuente como shortcut")

    # Calcular centroides de features por dataset
    centroids = {}
    for dataset, stats in all_stats.items():
        if not stats:
            continue
        features = np.array(
            [
                [
                    s["brightness"],
                    s["contrast"],
                    s["saturation"],
                    s["r_mean"],
                    s["g_mean"],
                    s["b_mean"],
                    s["black_pct"],
                    s["sharpness"] / 100,  # Normalizar
                ]
                for s in stats
            ]
        )
        centroids[dataset] = {
            "mean": features.mean(axis=0),
            "std": features.std(axis=0),
        }

    # Distancia euclidiana entre centroides
    datasets = list(centroids.keys())
    lines.append("\n  Distancia entre centroides (features normalizadas):")

    # Normalizar
    all_means = np.array([centroids[d]["mean"] for d in datasets])
    global_mean = all_means.mean(axis=0)
    global_std = all_means.std(axis=0) + 1e-8

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap de distancias
    n = len(datasets)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            norm_i = (centroids[datasets[i]]["mean"] - global_mean) / global_std
            norm_j = (centroids[datasets[j]]["mean"] - global_mean) / global_std
            dist_matrix[i, j] = np.linalg.norm(norm_i - norm_j)

    ax = axes[0]
    im = ax.imshow(dist_matrix, cmap="YlOrRd")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_yticklabels(datasets)
    ax.set_title("Distancia entre Datasets")

    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{dist_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    plt.colorbar(im, ax=ax, shrink=0.8)

    lines.append(f"\n  {'':15s}" + "".join(f"{d:>15s}" for d in datasets))
    for i, d1 in enumerate(datasets):
        row = f"  {d1:15s}"
        for j, _ in enumerate(datasets):
            row += f"{dist_matrix[i, j]:15.2f}"
        lines.append(row)

    # Radar chart de features promedio
    ax = axes[1]
    feature_names = [
        "Brillo",
        "Contraste",
        "Saturación",
        "Red",
        "Green",
        "Blue",
        "% Negro",
        "Nitidez",
    ]

    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    for dataset in datasets:
        values = (centroids[dataset]["mean"] - global_mean) / global_std
        values = np.concatenate([values, [values[0]]])
        ax.plot(
            angles,
            values,
            "o-",
            label=dataset,
            color=constants.DATASET_CONFIG[dataset]["color"],
            linewidth=2,
        )
        ax.fill(
            angles,
            values,
            alpha=0.1,
            color=constants.DATASET_CONFIG[dataset]["color"],
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=8)
    ax.set_title("Perfil de Features (normalizado)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "11_domain_gap.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    text = "\n".join(lines)
    print(text)
    return text


# ═══════════════════════════════════════════════════════════
#  ANÁLISIS 11: MAPEO A CLASES DEL PROYECTO
# ═══════════════════════════════════════════════════════════


def analyze_class_mapping(structures: dict) -> str:
    """Mapeo de subcarpetas a clases del proyecto."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("11. MAPEO A CLASES DEL PROYECTO")
    lines.append("=" * 80)
    lines.append("  Clases: Normal (0) | Pólipo (1) | Inflamación (2)")

    class_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # HyperKvasir
    hk_struct = structures.get("hyperkvasir", {})
    if hk_struct.get("exists"):
        for class_name, subpaths in constants.DATASET_CONFIG["hyperkvasir"][
            "subcarpetas_interes"
        ].items():
            for sp in subpaths:
                for folder, info in hk_struct.get("folders", {}).items():
                    if sp in folder or folder.endswith(sp.split("/")[-1]):
                        class_totals[class_name]["hyperkvasir"] += info["n_images"]

    # CVC-ClinicDB
    cvc_struct = structures.get("cvc_clinicdb", {})
    if cvc_struct.get("exists"):
        class_totals["polyp"]["cvc_clinicdb"] = cvc_struct.get("total_images", 0)

    # LIMUC
    limuc_struct = structures.get("limuc", {})
    if limuc_struct.get("exists"):
        for folder, info in limuc_struct.get("folders", {}).items():
            # Detectar Mayo score
            for score in ["0"]:
                if f"/{score}" in folder or folder.endswith(f"/{score}"):
                    class_totals["normal"]["limuc"] += info["n_images"]
            for score in ["1", "2", "3"]:
                if f"/{score}" in folder or folder.endswith(f"/{score}"):
                    class_totals["inflammation"]["limuc"] += info["n_images"]

    # Curated Colon
    cc_struct = structures.get("curated_colon", {})
    if cc_struct.get("exists"):
        for folder, info in cc_struct.get("folders", {}).items():
            fl = folder.lower()
            if any(kw in fl for kw in ["polyp", "adenoma", "cancer", "tumor"]):
                class_totals["polyp"]["curated_colon"] += info["n_images"]
            elif any(kw in fl for kw in ["normal", "healthy", "benign", "negative"]):
                class_totals["normal"]["curated_colon"] += info["n_images"]

    # Tabla
    lines.append(
        f"\n  {'Clase':<15s} {'HyperKvasir':>12s} {'CVC':>12s} "
        f"{'LIMUC':>12s} {'Curated':>12s} {'TOTAL':>12s}"
    )
    lines.append("  " + "-" * 70)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    class_order = ["normal", "polyp", "inflammation"]
    source_order = ["hyperkvasir", "cvc_clinicdb", "limuc", "curated_colon"]
    source_colors = [constants.DATASET_CONFIG[s]["color"] for s in source_order]

    x = np.arange(len(class_order))
    width = 0.2

    for s_idx, source in enumerate(source_order):
        counts = [class_totals[cls].get(source, 0) for cls in class_order]
        ax.bar(
            x + s_idx * width,
            counts,
            width,
            label=source,
            color=source_colors[s_idx],
            alpha=0.8,
        )

    for cls in class_order:
        total = sum(class_totals[cls].values())
        hk = class_totals[cls].get("hyperkvasir", 0)
        cvc = class_totals[cls].get("cvc_clinicdb", 0)
        limuc = class_totals[cls].get("limuc", 0)
        curated = class_totals[cls].get("curated_colon", 0)
        lines.append(
            f"  {cls:<15s} {hk:>12d} {cvc:>12d} "
            f"{limuc:>12d} {curated:>12d} {total:>12d}"
        )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(class_order)
    ax.set_ylabel("Número de imágenes")
    ax.set_title("Imágenes por Clase × Fuente (estimado)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "12_class_mapping.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    # Alertas
    lines.append("")
    for cls in class_order:
        total = sum(class_totals[cls].values())
        n_sources = sum(1 for v in class_totals[cls].values() if v > 0)
        if n_sources < 2:
            lines.append(
                f"  ⚠️  {cls}: solo {n_sources} fuente(s) "
                f"→ riesgo de shortcut por artefactos"
            )
        else:
            lines.append(f"  ✅ {cls}: {n_sources} fuentes → mixing efectivo")

    text = "\n".join(lines)
    print(text)
    return text


# ═══════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════


def run_eda():
    """Pipeline EDA completo."""
    print("=" * 80)
    print("EDA COMPLETO: DATASETS RAW PARA CÁNCER DE COLON")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 80)

    # ── 1. Descubrir estructura ──
    print("\n📂 Descubriendo estructura de datasets...")
    structures = {}
    for name, config in constants.DATASET_CONFIG.items():
        print(f"  Escaneando {name}...")
        structures[name] = discover_dataset_structure(name, config)
        print(
            f"    → {structures[name]['total_images']:,} imágenes, "
            f"{len(structures[name]['folders'])} carpetas"
        )

    # ── 2. Inventario ──
    report_parts = []
    report_parts.append(analyze_inventory(structures))

    # ── 3. Calcular estadísticas por imagen ──
    print("\n📊 Calculando estadísticas de imágenes...")
    all_stats: dict[str, list[dict]] = {}

    for name, struct in structures.items():
        if not struct["all_images"]:
            all_stats[name] = []
            continue

        sampled = sample_images(struct["all_images"], n=300)
        print(
            f"  {name}: analizando {len(sampled)} de "
            f"{len(struct['all_images'])} imágenes..."
        )

        stats = []
        for img_path in tqdm(sampled, desc=f"    {name}"):
            s = compute_image_stats(img_path)
            if s:
                stats.append(s)

        all_stats[name] = stats
        print(f"    → {len(stats)} estadísticas calculadas")

    # ── 4. Análisis ──
    report_parts.append(analyze_resolutions(all_stats))
    report_parts.append(analyze_color(all_stats))
    report_parts.append(analyze_artifacts(all_stats))
    report_parts.append(analyze_sharpness(all_stats))

    print("\n🖼️ Generando grid de muestras...")
    plot_sample_grid(structures)

    report_parts.append(analyze_domain_gap(all_stats))
    report_parts.append(analyze_class_mapping(structures))

    # ── 5. Guardar reporte ──
    full_report = "\n".join(report_parts)
    report_path = OUTPUT_DIR / "eda_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"\n📄 Reporte: {report_path}")

    # ── 6. Guardar stats JSON ──
    json_stats = {}
    for dataset, stats in all_stats.items():
        if not stats:
            json_stats[dataset] = {"n_analyzed": 0}
            continue

        json_stats[dataset] = {
            "n_analyzed": len(stats),
            "n_total": structures[dataset]["total_images"],
            "resolution": {
                "width_mean": round(np.mean([s["width"] for s in stats]), 1),
                "height_mean": round(np.mean([s["height"] for s in stats]), 1),
                "width_min": min(s["width"] for s in stats),
                "width_max": max(s["width"] for s in stats),
            },
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
                "pct_over_5": round(
                    sum(1 for s in stats if s["black_pct"] > 5) / len(stats) * 100,
                    1,
                ),
            },
            "green_pct": {
                "mean": round(np.mean([s["green_pct"] for s in stats]), 2),
                "max": round(max(s["green_pct"] for s in stats), 2),
            },
            "fov_types": dict(Counter(s["fov_type"] for s in stats)),
            "sharpness": {
                "mean": round(np.mean([s["sharpness"] for s in stats]), 1),
                "median": round(
                    float(np.median([s["sharpness"] for s in stats])),
                    1,
                ),
            },
        }

    stats_path = OUTPUT_DIR / "eda_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(json_stats, f, indent=2, ensure_ascii=False)
    print(f"📊 Stats JSON: {stats_path}")

    # ── Resumen final ──
    print("\n" + "=" * 80)
    print("✅ EDA COMPLETADO")
    print(f"  📁 Output: {OUTPUT_DIR}")
    print("  Archivos generados:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {'✅'} {f.name:40s} ({size_kb:.0f} KB)")
    print("=" * 80)


if __name__ == "__main__":
    run_eda()
