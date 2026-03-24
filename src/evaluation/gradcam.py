"""
src/evaluation/gradcam.py

Wrapper moderno de Grad-CAM usando pytorch-grad-cam.
Soporta múltiples variantes: GradCAM, GradCAM++, HiResCAM, ScoreCAM, etc.
"""

from typing import Literal

import cv2
import numpy as np
from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    ScoreCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
import torch.nn as nn

from src.config.logger import log as logger

# ═══════════════════════════════════════════════════════════
#  TIPO DE CAM
# ═══════════════════════════════════════════════════════════

CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "hirescam": HiResCAM,
    "scorecam": ScoreCAM,
}

CamMethod = Literal["gradcam", "gradcam++", "hirescam", "scorecam"]


# ═══════════════════════════════════════════════════════════
#  BÚSQUEDA AUTOMÁTICA DE CAPA TARGET
# ═══════════════════════════════════════════════════════════


def find_target_layer(model: nn.Module) -> nn.Module:
    """
    Encuentra la última capa convolucional del backbone.

    Compatible con:
    - EfficientNet (timm)
    - ResNet
    - ConvNeXt
    - Vision Transformers (limitado)

    Returns:
        nn.Module: Última capa Conv2d encontrada
    """
    backbone = model.backbone if hasattr(model, "backbone") else model

    # ── Estrategia 1: Nombres conocidos ──
    for attr in ["conv_head", "head", "norm"]:
        if hasattr(backbone, attr):
            layer = getattr(backbone, attr)
            if isinstance(layer, nn.Conv2d):
                logger.info(f"  🎯 Target: backbone.{attr}")
                return layer

    # ── Estrategia 2: Bloques tipo EfficientNet ──
    if hasattr(backbone, "blocks"):
        blocks = backbone.blocks
        if len(blocks) > 0:
            last_conv = None
            name_found = ""
            for name, module in blocks[-1].named_modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
                    name_found = name
            if last_conv is not None:
                logger.info(f"  🎯 Target: backbone.blocks[-1].{name_found}")
                return last_conv

    # ── Estrategia 3: ResNet layer4 ──
    if hasattr(backbone, "layer4"):
        last_conv = None
        for _, module in backbone.layer4[-1].named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            logger.info("  🎯 Target: backbone.layer4[-1] (ResNet)")
            return last_conv

    # ── Estrategia 4: Búsqueda genérica (última Conv2d) ──
    last_conv = None
    last_name = ""
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
            last_name = name

    if last_conv is not None:
        logger.info(f"  🎯 Target: backbone.{last_name} (fallback)")
        return last_conv

    raise RuntimeError("❌ No se encontró capa convolucional en el modelo")


# ═══════════════════════════════════════════════════════════
#  GENERACIÓN DE GRAD-CAM
# ═══════════════════════════════════════════════════════════


def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
    method: CamMethod = "gradcam++",
    target_layer: nn.Module | None = None,
) -> tuple[np.ndarray, int]:
    """
    Genera mapa Grad-CAM usando pytorch-grad-cam.

    Args:
        model: Modelo PyTorch
        input_tensor: Tensor (1, C, H, W) normalizado
        target_class: Clase objetivo (None = predicción del modelo)
        method: Tipo de CAM ("gradcam", "gradcam++", "hirescam", "scorecam")
        target_layer: Capa objetivo (None = búsqueda automática)

    Returns:
        cam: Mapa de calor [0-1] shape (H, W)
        pred_class: Clase predicha por el modelo
    """
    model.eval()

    # ── Seleccionar capa ──
    if target_layer is None:
        target_layer = find_target_layer(model)

    # ── Predicción ──
    with torch.no_grad():
        logits = model(input_tensor)
        pred_class = int(torch.argmax(logits, dim=1).item())

    if target_class is None:
        target_class = pred_class

    # ── Grad-CAM ──
    cam_algorithm = CAM_METHODS[method]
    targets = [ClassifierOutputTarget(target_class)]

    with cam_algorithm(model=model, target_layers=[target_layer]) as cam:
        # input_tensor debe estar en rango [0, 1] para pytorch-grad-cam
        # Si está normalizado con ImageNet stats, desnormalizar primero
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # grayscale_cam shape: (batch, H, W)
    cam_map = grayscale_cam[0]  # (H, W) en [0, 1]

    return cam_map, pred_class


# ═══════════════════════════════════════════════════════════
#  OVERLAY
# ═══════════════════════════════════════════════════════════


def create_heatmap_overlay(
    img_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Superpone heatmap Grad-CAM sobre imagen RGB.

    Args:
        img_rgb: Imagen RGB uint8 (H, W, 3)
        cam: Mapa de calor [0-1] (H_cam, W_cam)
        alpha: Transparencia del heatmap (0=transparente, 1=opaco)
        colormap: Mapa de colores OpenCV

    Returns:
        overlay: Imagen RGB uint8 con heatmap superpuesto
    """
    h, w = img_rgb.shape[:2]

    # Resize CAM a tamaño de imagen
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    # Usar utilidad de pytorch-grad-cam (más robusta)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    overlay = show_cam_on_image(
        img_normalized,
        cam_resized,
        use_rgb=True,
        colormap=colormap,
        image_weight=1.0 - alpha,
    )

    return overlay  # uint8 RGB


# ═══════════════════════════════════════════════════════════
#  ESTADÍSTICAS DE ATENCIÓN
# ═══════════════════════════════════════════════════════════


def compute_attention_stats(
    cam: np.ndarray,
    center_ratio: float = 0.6,
) -> tuple[float, float]:
    """
    Calcula atención en centro vs borde.

    Args:
        cam: Mapa [0-1] shape (H, W)
        center_ratio: Proporción del centro (0.6 = 60% central)

    Returns:
        center_attention: Activación media en centro
        border_attention: Activación media en borde
    """
    h, w = cam.shape
    margin_h = int(h * (1 - center_ratio) / 2)
    margin_w = int(w * (1 - center_ratio) / 2)

    center = cam[margin_h : h - margin_h, margin_w : w - margin_w]
    border_mask = np.ones_like(cam, dtype=bool)
    border_mask[margin_h : h - margin_h, margin_w : w - margin_w] = False

    center_attention = float(center.mean()) if center.size > 0 else 0.0
    border_attention = float(cam[border_mask].mean()) if border_mask.sum() > 0 else 0.0

    return center_attention, border_attention


def compute_pointing_accuracy(
    cam: np.ndarray, center_ratio: float = 0.6
) -> tuple[bool, float]:
    """
    Verifica si el máximo de atención está en el centro.

    Args:
        cam: Mapa [0-1] shape (H, W)
        center_ratio: Proporción del centro

    Returns:
        max_in_center: True si máximo está en centro
        center_strong_ratio: % de píxeles fuertes (>90 percentil) en centro
    """
    h, w = cam.shape
    margin_h = int(h * (1 - center_ratio) / 2)
    margin_w = int(w * (1 - center_ratio) / 2)

    center_mask = np.zeros((h, w), dtype=bool)
    center_mask[margin_h : h - margin_h, margin_w : w - margin_w] = True

    max_pos = np.unravel_index(cam.argmax(), cam.shape)
    max_in_center = bool(center_mask[max_pos[0], max_pos[1]])

    threshold = np.percentile(cam, 90)
    strong_pixels = cam >= threshold
    strong_in_center = (strong_pixels & center_mask).sum()
    strong_total = strong_pixels.sum()

    center_strong_ratio = float(strong_in_center) / max(float(strong_total), 1.0)

    return max_in_center, center_strong_ratio
