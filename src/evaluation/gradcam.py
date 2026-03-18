"""
src/training/evaluate_gradcam.py

Grad-CAM para validar que el modelo mira el tejido, no los artefactos.
Versión multi-source: análisis por fuente (HyperKvasir vs CVC vs LIMUC).
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.config.constants import detect_source_from_stem
from src.config.paths import paths
from src.config.settings import model as model_cfg
from src.data.processing.image_preprocessor import get_val_transforms
from src.database.connection import get_db
from src.database.repositories import TrainingImageRepository
from src.evaluation.attention import compute_attention_stats, compute_pointing_accuracy
from src.models.image_classifier import ColonCancerClassifier

# ═══════════════════════════════════════════════════════════
#  GRAD-CAM
# ═══════════════════════════════════════════════════════════


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []

        self._hooks.append(target_layer.register_forward_hook(self._save_activation))
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Genera mapa Grad-CAM."""
        self.model.eval()

        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)

        pred_class = torch.argmax(logits, dim=1).item()
        if target_class is None:
            target_class = pred_class

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM: no se capturaron gradientes.")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)

        cam = cam.cpu().numpy()
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_class, logits.detach()

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()


# ═══════════════════════════════════════════════════════════
#  UTILIDADES
# ═══════════════════════════════════════════════════════════


def create_heatmap_overlay(img_rgb, cam, alpha=0.5):
    """Superpone heatmap sobre imagen."""
    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    img_float = img_rgb.astype(np.float32) / 255.0
    heat_float = heatmap_rgb.astype(np.float32) / 255.0

    overlay = heat_float * alpha + img_float * (1.0 - alpha)
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return overlay


def find_target_layer(model):
    """Encuentra última capa convolucional del backbone."""
    backbone = model.backbone

    for attr in ["conv_head", "bn2"]:
        if hasattr(backbone, attr):
            layer = getattr(backbone, attr)
            if isinstance(layer, torch.nn.Conv2d):
                print(f"  Target layer: backbone.{attr}")
                return layer

    if hasattr(backbone, "blocks"):
        blocks = backbone.blocks
        if len(blocks) > 0:
            last_conv = None
            name_found = ""
            for name, module in blocks[-1].named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
                    name_found = name
            if last_conv is not None:
                print(f"  Target layer: backbone.blocks[-1].{name_found}")
                return last_conv

    if hasattr(backbone, "layer4"):
        last_conv = None
        for _, module in backbone.layer4[-1].named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv

    last_conv = None
    last_name = ""
    for name, module in backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
            last_name = name

    if last_conv is not None:
        print(f"  Target layer: backbone.{last_name} (fallback)")
        return last_conv

    raise RuntimeError("No se encontró capa convolucional.")


# ═══════════════════════════════════════════════════════════
#  CARGA
# ═══════════════════════════════════════════════════════════


def load_model_and_config(device):
    """Carga modelo entrenado."""
    ckpt_path = paths.CLASSIFIER_CHECKPOINT
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No se encontró {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_name = ckpt.get("model_name", "efficientnet_b2")
    num_classes = ckpt.get("num_classes", 3)
    class_mapping = ckpt.get("class_mapping", {})
    temperature = ckpt.get("temperature", 1.0)

    class_names = {int(k): v for k, v in class_mapping.get("classes", {}).items()}

    model = ColonCancerClassifier(
        model_name=model_name,
        pretrained=False,
        num_classes=num_classes,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  Modelo: {model_name} ({num_classes} clases)")
    print(f"  F1: {ckpt.get('best_f1', 0):.4f} | Acc: {ckpt.get('best_acc', 0):.4f}")
    print(f"  Temperatura: {temperature:.4f}")
    print(f"  Clases: {list(class_names.values())}")

    return model, class_names, num_classes, temperature


def load_test_data():
    """Carga datos del test set."""
    with get_db() as db:
        repo = TrainingImageRepository(db)
        test = repo.get_by_split("test")
        paths_list = [str(d.file_path) for d in test]
        labels = [int(d.label) for d in test]

    if not paths_list:
        raise ValueError("No hay datos de test.")

    # Log distribución por fuente
    from collections import Counter

    source_counts = Counter(detect_source_from_stem(Path(p).stem) for p in paths_list)
    print(f"  Test set: {len(paths_list)} imágenes")
    print(f"  Fuentes: {dict(source_counts)}")

    return paths_list, labels


# ═══════════════════════════════════════════════════════════
#  ANÁLISIS PRINCIPAL
# ═══════════════════════════════════════════════════════════


def run_gradcam_analysis(samples_per_class=4, output_dir=None):
    """Análisis Grad-CAM completo con desglose por fuente."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir) if output_dir else paths.ROOT

    print("=" * 65)
    print("GRAD-CAM: ANÁLISIS MULTI-SOURCE")
    print(f"  Device: {device}")
    print(f"  Samples/clase: {samples_per_class}")
    print("=" * 65)

    model, class_names, num_classes, temperature = load_model_and_config(device)
    test_paths, test_labels = load_test_data()
    image_size = model_cfg.IMAGE_SIZE
    transform = get_val_transforms(image_size)

    target_layer = find_target_layer(model)
    grad_cam = GradCAM(model, target_layer)

    print("\nProcesando imágenes...")
    results = []

    for idx in range(len(test_paths)):
        path = test_paths[idx]
        real_label = test_labels[idx]

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        img_resized = cv2.resize(img_bgr, (image_size, image_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)

        try:
            cam, pred_class, logits = grad_cam.generate(tensor)
        except Exception as e:
            print(f"  ⚠️ Error en {Path(path).name}: {e}")
            continue

        probs = F.softmax(logits / temperature, dim=1)[0].cpu().numpy()
        overlay = create_heatmap_overlay(img_rgb, cam, alpha=0.45)

        center_attention, border_attention = compute_attention_stats(
            cam, center_ratio=0.6
        )
        max_in_center, center_strong_ratio = compute_pointing_accuracy(cam)

        results.append(
            {
                "idx": idx,
                "path": path,
                "source": detect_source_from_stem(Path(path).stem),
                "real_label": real_label,
                "pred_class": pred_class,
                "correct": real_label == pred_class,
                "confidence": float(probs[pred_class]),
                "probs": probs,
                "img_rgb": img_rgb,
                "cam": cam,
                "cam_shape": cam.shape,
                "overlay": overlay,
                "center_attention": center_attention,
                "border_attention": border_attention,
                "max_in_center": max_in_center,
                "center_strong_ratio": center_strong_ratio,
            }
        )

    print(f"  Procesadas: {len(results)} imágenes")
    if results:
        print(f"  CAM nativo: {results[0]['cam_shape']}")

    # 1. Grilla por clase
    print("\nGenerando grilla por clase...")
    _plot_per_class_grid(
        results,
        class_names,
        num_classes,
        samples_per_class,
        output_dir / "gradcam_per_class.png",
    )

    # 2. Grilla por fuente (NUEVO)
    print("\nGenerando grilla por fuente...")
    _plot_per_source_grid(
        results,
        class_names,
        num_classes,
        output_dir / "gradcam_per_source.png",
    )

    # 3. Errores
    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\nGenerando análisis de errores ({len(errors)} errores)...")
        _plot_errors(
            errors,
            class_names,
            num_classes,
            min(12, len(errors)),
            output_dir / "gradcam_errors.png",
        )

    # 4. Estadísticas (con análisis por fuente)
    print("\nEstadísticas de atención...")
    _save_attention_stats(
        results,
        class_names,
        num_classes,
        output_dir / "gradcam_attention_stats.txt",
    )

    grad_cam.remove_hooks()

    print(f"\n✅ Resultados en: {output_dir}")
    print("  📊 gradcam_per_class.png")
    print("  📊 gradcam_per_source.png")
    if errors:
        print("  📊 gradcam_errors.png")
    print("  📊 gradcam_attention_stats.txt")


# ═══════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════


def _plot_per_class_grid(
    results, class_names, num_classes, samples_per_class, output_path
):
    """Grilla: cada fila = una muestra, cols = original | overlay | probs."""
    selected = []
    for c in range(num_classes):
        class_results = [r for r in results if r["real_label"] == c]
        correct = [r for r in class_results if r["correct"]]
        incorrect = [r for r in class_results if not r["correct"]]
        pool = correct + incorrect
        n = min(samples_per_class, len(pool))
        if n > 0:
            indices = np.random.choice(len(pool), n, replace=False)
            selected.extend([(c, pool[i]) for i in indices])

    n_rows = len(selected)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (_, r) in enumerate(selected):
        real_name = class_names.get(r["real_label"], str(r["real_label"]))
        pred_name = class_names.get(r["pred_class"], str(r["pred_class"]))

        axes[row, 0].imshow(r["img_rgb"])
        axes[row, 0].set_title(
            f"Real: {real_name} [{r['source'][:3]}]",
            fontsize=10,
            fontweight="bold",
        )
        axes[row, 0].axis("off")

        axes[row, 1].imshow(r["overlay"])
        color = "#2ecc71" if r["correct"] else "#e74c3c"
        symbol = "✓" if r["correct"] else "✗"
        axes[row, 1].set_title(
            f"{symbol} Pred: {pred_name} ({r['confidence']:.1%})",
            fontsize=10,
            fontweight="bold",
            color=color,
        )
        axes[row, 1].axis("off")

        _plot_prob_bars(
            axes[row, 2],
            r["probs"],
            class_names,
            num_classes,
            r["real_label"],
            r["pred_class"],
        )

    plt.suptitle(
        "Grad-CAM por Clase [fuente]\n"
        "Rojo/Amarillo = alta atención | Azul = baja atención",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ {output_path}")


def _plot_per_source_grid(results, class_names, num_classes, output_path):
    """
    Grilla comparativa por fuente: misma clase, diferentes fuentes.
    Clave para verificar que el modelo mira tejido y no artefactos de fuente.
    """
    sources = sorted({r["source"] for r in results})
    samples_per_cell = 2

    n_rows = num_classes * samples_per_cell
    n_cols = len(sources) * 2  # overlay + original por fuente

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    row = 0
    for c in range(num_classes):
        class_name = class_names.get(c, str(c))

        for sample_idx in range(samples_per_cell):
            for col_base, source in enumerate(sources):
                pool = [
                    r for r in results if r["real_label"] == c and r["source"] == source
                ]

                col_orig = col_base * 2
                col_overlay = col_base * 2 + 1

                if row >= n_rows:
                    break

                if sample_idx < len(pool):
                    r = pool[sample_idx]

                    axes[row, col_orig].imshow(r["img_rgb"])
                    axes[row, col_orig].set_title(
                        f"{class_name}\n{source[:6]}",
                        fontsize=8,
                    )
                    axes[row, col_orig].axis("off")

                    axes[row, col_overlay].imshow(r["overlay"])
                    symbol = "✓" if r["correct"] else "✗"
                    color = "#2ecc71" if r["correct"] else "#e74c3c"
                    ratio = r["center_attention"] / max(r["border_attention"], 1e-8)
                    axes[row, col_overlay].set_title(
                        f"{symbol} c/b={ratio:.1f}",
                        fontsize=8,
                        color=color,
                    )
                    axes[row, col_overlay].axis("off")
                else:
                    axes[row, col_orig].axis("off")
                    axes[row, col_overlay].axis("off")

            row += 1

    plt.suptitle(
        "Grad-CAM por Fuente: ¿Misma atención independiente del endoscopio?\n"
        "c/b = ratio centro/borde (>1.5 = bueno)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ {output_path}")


def _plot_prob_bars(ax, probs, class_names, num_classes, real_label, pred_label):
    """Barras de probabilidad."""
    names = [class_names.get(i, str(i))[:12] for i in range(num_classes)]
    y_pos = np.arange(num_classes)
    colors = []
    for i in range(num_classes):
        if i == real_label and i == pred_label:
            colors.append("#2ecc71")
        elif i == pred_label:
            colors.append("#e74c3c")
        elif i == real_label:
            colors.append("#3498db")
        else:
            colors.append("#bdc3c7")

    ax.barh(y_pos, probs, color=colors, height=0.7, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Prob", fontsize=8)
    ax.invert_yaxis()
    for i, p in enumerate(probs):
        if p > 0.03:
            ax.text(p + 0.01, i, f"{p:.0%}", va="center", fontsize=7)


def _plot_errors(errors, class_names, num_classes, max_errors, output_path):
    """Errores ordenados por confianza con info de fuente."""
    errors_sorted = sorted(errors, key=lambda x: x["confidence"], reverse=True)
    errors_to_show = errors_sorted[:max_errors]
    n = len(errors_to_show)

    fig, axes = plt.subplots(n, 3, figsize=(16, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, r in enumerate(errors_to_show):
        real_name = class_names.get(r["real_label"], str(r["real_label"]))
        pred_name = class_names.get(r["pred_class"], str(r["pred_class"]))

        axes[row, 0].imshow(r["img_rgb"])
        axes[row, 0].set_title(f"Real: {real_name} [{r['source'][:6]}]", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(r["overlay"])
        axes[row, 1].set_title(
            f"✗ Pred: {pred_name} ({r['confidence']:.1%})",
            fontsize=10,
            color="#e74c3c",
            fontweight="bold",
        )
        axes[row, 1].axis("off")

        _plot_prob_bars(
            axes[row, 2],
            r["probs"],
            class_names,
            num_classes,
            r["real_label"],
            r["pred_class"],
        )

    plt.suptitle(
        "ERRORES con fuente (ordenados por confianza)",
        fontsize=14,
        fontweight="bold",
        color="#e74c3c",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ {output_path}")


# ═══════════════════════════════════════════════════════════
#  ESTADÍSTICAS (con análisis por fuente)
# ═══════════════════════════════════════════════════════════


def _save_attention_stats(results, class_names, num_classes, output_path):
    """Estadísticas completas con desglose por fuente."""
    lines = []
    lines.append("=" * 75)
    lines.append("ESTADÍSTICAS DE ATENCIÓN GRAD-CAM (MULTI-SOURCE)")
    lines.append("=" * 75)
    lines.append("")

    if results:
        lines.append(f"Resolución nativa del CAM: {results[0]['cam_shape']}")
    lines.append("Centro = 60% interior | Borde = 20% exterior por lado")
    lines.append("")

    # ── GLOBAL ──
    all_center = [r["center_attention"] for r in results]
    all_border = [r["border_attention"] for r in results]
    all_pointing = [r["max_in_center"] for r in results]
    all_strong = [r["center_strong_ratio"] for r in results]

    ratio = np.mean(all_center) / max(np.mean(all_border), 1e-8)
    pointing_acc = np.mean(all_pointing) * 100

    lines.append(f"GLOBAL ({len(results)} imágenes):")
    lines.append(f"  Atención centro: {np.mean(all_center):.4f}")
    lines.append(f"  Atención borde:  {np.mean(all_border):.4f}")
    lines.append(f"  Ratio centro/borde: {ratio:.2f}")
    lines.append(f"  Pointing accuracy: {pointing_acc:.1f}%")
    lines.append(f"  Activación fuerte en centro: {np.mean(all_strong) * 100:.1f}%")

    if ratio > 2.0:
        lines.append("  ✅ MUY BIEN: modelo enfocado en tejido central")
    elif ratio > 1.5:
        lines.append("  ✅ BIEN: modelo mira principalmente el tejido")
    elif ratio > 1.0:
        lines.append("  ⚠️  ACEPTABLE: mira más el centro pero no mucho más")
    else:
        lines.append("  🚨 PROBLEMA: el modelo mira más los bordes que el tejido")

    lines.append("")

    # ── POR CLASE ──
    lines.append("POR CLASE:")
    for c in range(num_classes):
        cr = [r for r in results if r["real_label"] == c]
        if not cr:
            continue

        name = class_names.get(c, str(c))
        centers = [r["center_attention"] for r in cr]
        borders = [r["border_attention"] for r in cr]
        pointing = [r["max_in_center"] for r in cr]
        strong = [r["center_strong_ratio"] for r in cr]
        correct = sum(1 for r in cr if r["correct"])
        total = len(cr)
        cls_ratio = np.mean(centers) / max(np.mean(borders), 1e-8)
        p_acc = np.mean(pointing) * 100

        status = "✅" if cls_ratio > 1.5 else ("⚠️ " if cls_ratio > 1.0 else "🚨")

        lines.append(
            f"  {status} {name:25s} | "
            f"centro={np.mean(centers):.3f} "
            f"borde={np.mean(borders):.3f} "
            f"ratio={cls_ratio:.2f} "
            f"pointing={p_acc:.0f}% "
            f"strong_center={np.mean(strong) * 100:.0f}% | "
            f"acc={correct}/{total} ({correct / max(total, 1):.0%})"
        )

    lines.append("")

    # ── POR FUENTE (NUEVO - clave para multi-source) ──
    lines.append("=" * 75)
    lines.append("POR FUENTE (¿El modelo se comporta igual con cada endoscopio?):")
    lines.append("=" * 75)

    sources = sorted({r["source"] for r in results})
    for source in sources:
        sr = [r for r in results if r["source"] == source]
        if not sr:
            continue

        centers = [r["center_attention"] for r in sr]
        borders = [r["border_attention"] for r in sr]
        pointing = [r["max_in_center"] for r in sr]
        strong = [r["center_strong_ratio"] for r in sr]
        correct = sum(1 for r in sr if r["correct"])
        total = len(sr)
        src_ratio = np.mean(centers) / max(np.mean(borders), 1e-8)
        p_acc = np.mean(pointing) * 100

        status = "✅" if src_ratio > 1.5 else ("⚠️ " if src_ratio > 1.0 else "🚨")

        lines.append(
            f"  {status} {source:15s} ({total:3d} imgs) | "
            f"centro={np.mean(centers):.3f} "
            f"borde={np.mean(borders):.3f} "
            f"ratio={src_ratio:.2f} "
            f"pointing={p_acc:.0f}% "
            f"strong={np.mean(strong) * 100:.0f}% | "
            f"acc={correct}/{total} ({correct / max(total, 1):.0%})"
        )

    lines.append("")

    # ── POR CLASE × FUENTE (NUEVO - diagnóstico fino) ──
    lines.append("=" * 75)
    lines.append("POR CLASE × FUENTE (desglose completo):")
    lines.append("=" * 75)

    for c in range(num_classes):
        class_name = class_names.get(c, str(c))
        lines.append(f"\n  {class_name.upper()}:")

        for source in sources:
            csr = [r for r in results if r["real_label"] == c and r["source"] == source]
            if not csr:
                continue

            centers = [r["center_attention"] for r in csr]
            borders = [r["border_attention"] for r in csr]
            pointing = [r["max_in_center"] for r in csr]
            correct = sum(1 for r in csr if r["correct"])
            total = len(csr)
            cs_ratio = np.mean(centers) / max(np.mean(borders), 1e-8)
            p_acc = np.mean(pointing) * 100

            status = "✅" if cs_ratio > 1.5 else ("⚠️ " if cs_ratio > 1.0 else "🚨")

            lines.append(
                f"    {status} {source:15s} ({total:3d}) | "
                f"ratio={cs_ratio:.2f} "
                f"pointing={p_acc:.0f}% | "
                f"acc={correct}/{total} ({correct / max(total, 1):.0%})"
            )

    # ── DIAGNÓSTICO DE CONSISTENCIA ENTRE FUENTES ──
    lines.append("")
    lines.append("=" * 75)
    lines.append("DIAGNÓSTICO DE CONSISTENCIA:")
    lines.append("=" * 75)

    # ¿Las fuentes tienen ratios similares? Si no, hay source bias
    source_ratios = {}
    for source in sources:
        sr = [r for r in results if r["source"] == source]
        if sr:
            c = np.mean([r["center_attention"] for r in sr])
            b = np.mean([r["border_attention"] for r in sr])
            source_ratios[source] = c / max(b, 1e-8)

    if len(source_ratios) >= 2:
        ratios_list = list(source_ratios.values())
        ratio_std = np.std(ratios_list)
        ratio_range = max(ratios_list) - min(ratios_list)

        lines.append(
            "  Ratios por fuente: "
            + ", ".join(f"{s}={r:.2f}" for s, r in sorted(source_ratios.items()))
        )
        lines.append(f"  Rango de ratios: {ratio_range:.2f}")
        lines.append(f"  Desv. estándar: {ratio_std:.2f}")

        if ratio_range < 0.5:
            lines.append(
                "  ✅ CONSISTENTE: El modelo mira de forma similar en todas las fuentes"
            )
            lines.append("  → Los artefactos de marco/endoscopio NO influyen")
        elif ratio_range < 1.0:
            lines.append("  ⚠️  ACEPTABLE: Diferencias moderadas entre fuentes")
            lines.append("  → Puede haber algo de sesgo por fuente pero no crítico")
        else:
            lines.append(
                "  🚨 INCONSISTENTE: El modelo se comporta muy "
                "diferente según la fuente"
            )
            lines.append("  → Posible shortcut por artefactos de fuente")

    lines.append("")

    # ── CORRECTOS vs INCORRECTOS ──
    lines.append("CORRECTOS vs INCORRECTOS:")

    for label, subset in [
        ("Correctos", [r for r in results if r["correct"]]),
        ("Incorrectos", [r for r in results if not r["correct"]]),
    ]:
        if not subset:
            continue
        c = np.mean([r["center_attention"] for r in subset])
        b = np.mean([r["border_attention"] for r in subset])
        p = np.mean([r["max_in_center"] for r in subset]) * 100

        # Desglose por fuente
        source_detail = {}
        for r in subset:
            source_detail[r["source"]] = source_detail.get(r["source"], 0) + 1
        src_str = ", ".join(f"{s}={c}" for s, c in sorted(source_detail.items()))

        lines.append(
            f"  {label} ({len(subset)}): "
            f"centro={c:.3f} borde={b:.3f} "
            f"ratio={c / max(b, 1e-8):.2f} "
            f"pointing={p:.0f}% "
            f"[{src_str}]"
        )

    # ── CONFUSIONES ──
    lines.append("")
    lines.append("CONFUSIONES MÁS FRECUENTES (con fuente):")
    confusion = {}
    for r in results:
        if not r["correct"]:
            key = (
                class_names.get(r["real_label"], str(r["real_label"])),
                class_names.get(r["pred_class"], str(r["pred_class"])),
                r["source"],
            )
            confusion[key] = confusion.get(key, 0) + 1

    for (real, pred, source), count in sorted(
        confusion.items(), key=lambda x: x[1], reverse=True
    )[:15]:
        lines.append(f"  {real:15s} → {pred:15s} [{source:10s}]: {count} veces")

    text = "\n".join(lines)
    print(text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n  ✅ {output_path}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_gradcam_analysis(
        samples_per_class=args.samples,
        output_dir=args.output,
    )
