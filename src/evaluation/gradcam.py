"""
src/evaluation/gradcam.py

Modern Grad-CAM wrapper using pytorch-grad-cam.
Supports multiple variants: GradCAM, GradCAM++, HiResCAM, ScoreCAM, etc.
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
#  CAM METHOD TYPE
# ═══════════════════════════════════════════════════════════

CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "hirescam": HiResCAM,
    "scorecam": ScoreCAM,
}

CamMethod = Literal["gradcam", "gradcam++", "hirescam", "scorecam"]


# ═══════════════════════════════════════════════════════════
#  AUTOMATIC TARGET LAYER DISCOVERY
# ═══════════════════════════════════════════════════════════


def find_target_layer(model: nn.Module) -> nn.Module:
    """
    Locate the last convolutional layer of the model backbone.

    Tries multiple discovery strategies in priority order so that it works
    across common architectures (EfficientNet, ResNet, ConvNeXt, and
    Vision Transformers with limited support).

    Strategy order:
        1. Named attributes: ``conv_head``, ``head``, ``norm`` on the
           backbone — returned only if the attribute is an ``nn.Conv2d``.
        2. EfficientNet-style ``blocks``: iterates ``backbone.blocks[-1]``
           and returns the last ``nn.Conv2d`` found.
        3. ResNet-style ``layer4``: iterates ``backbone.layer4[-1]`` and
           returns the last ``nn.Conv2d`` found.
        4. Generic fallback: iterates all named modules of the backbone
           and returns the last ``nn.Conv2d`` found anywhere.

    Args:
        model (nn.Module): Model instance. If the model has a ``backbone``
            attribute it is used as the search root; otherwise the model
            itself is searched.

    Returns:
        nn.Module: The target ``nn.Conv2d`` layer for Grad-CAM hook
            registration.

    Raises:
        RuntimeError: If no ``nn.Conv2d`` layer can be found in the model.
    """
    backbone = model.backbone if hasattr(model, "backbone") else model

    for attr in ["conv_head", "head", "norm"]:
        if hasattr(backbone, attr):
            layer = getattr(backbone, attr)
            if isinstance(layer, nn.Conv2d):
                logger.info(f"  🎯 Target: backbone.{attr}")
                return layer

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

    if hasattr(backbone, "layer4"):
        last_conv = None
        for _, module in backbone.layer4[-1].named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            logger.info("  🎯 Target: backbone.layer4[-1] (ResNet)")
            return last_conv

    last_conv = None
    last_name = ""
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
            last_name = name

    if last_conv is not None:
        logger.info(f"  🎯 Target: backbone.{last_name} (fallback)")
        return last_conv

    raise RuntimeError("❌ No convolutional layer found in the model.")


# ═══════════════════════════════════════════════════════════
#  GRAD-CAM GENERATION
# ═══════════════════════════════════════════════════════════


def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
    method: CamMethod = "gradcam++",
    target_layer: nn.Module | None = None,
) -> tuple[np.ndarray, int]:
    """
    Generate a Grad-CAM activation map using pytorch-grad-cam.

    The model is set to evaluation mode before the forward pass. If no
    target class is specified the class with the highest logit is used.
    The CAM algorithm is instantiated as a context manager so that hooks
    are cleanly removed after the call.

    Args:
        model (nn.Module): PyTorch model to explain. Must have at least
            one ``nn.Conv2d`` layer reachable by ``find_target_layer``.
        input_tensor (torch.Tensor): Pre-processed input batch of shape
            (1, C, H, W) already moved to the correct device. Should be
            normalised with the same statistics used during training.
        target_class (int | None): Index of the class to explain. If
            ``None`` the predicted class (argmax of logits) is used.
            Default is ``None``.
        method (CamMethod): CAM algorithm to use. One of ``"gradcam"``,
            ``"gradcam++"``, ``"hirescam"``, or ``"scorecam"``. Default
            is ``"gradcam++"``.
        target_layer (nn.Module | None): Specific layer to attach the
            hooks to. If ``None``, ``find_target_layer`` is called to
            discover it automatically. Default is ``None``.

    Returns:
        tuple[np.ndarray, int]:
            - Grayscale activation map of shape (H, W) with values
              normalised to [0, 1] by pytorch-grad-cam.
            - Predicted class index (argmax of the model logits before
              any CAM computation).
    """
    model.eval()

    if target_layer is None:
        target_layer = find_target_layer(model)

    with torch.no_grad():
        logits = model(input_tensor)
        pred_class = int(torch.argmax(logits, dim=1).item())

    if target_class is None:
        target_class = pred_class

    cam_algorithm = CAM_METHODS[method]
    targets = [ClassifierOutputTarget(target_class)]

    with cam_algorithm(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    cam_map = grayscale_cam[0]

    return cam_map, pred_class


# ═══════════════════════════════════════════════════════════
#  HEATMAP OVERLAY
# ═══════════════════════════════════════════════════════════


def create_heatmap_overlay(
    img_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap over an RGB image.

    The CAM map is resized to match the image spatial dimensions and then
    composited using ``show_cam_on_image`` from pytorch-grad-cam, which
    handles the colour mapping and alpha blending internally.

    Args:
        img_rgb (np.ndarray): Source image in RGB format with dtype
            ``uint8`` and shape (H, W, 3). Values must be in [0, 255].
        cam (np.ndarray): Grayscale activation map of shape
            (H_cam, W_cam) with values in [0, 1]. Resized to (H, W)
            before blending.
        alpha (float): Opacity of the heatmap layer. A value of 0 makes
            the heatmap invisible (only the original image is shown); a
            value of 1 makes only the heatmap visible. Passed as
            ``1.0 - alpha`` to the ``image_weight`` parameter of
            ``show_cam_on_image``. Default is 0.5.
        colormap (int): OpenCV colormap constant applied to the
            normalised activation map. Default is ``cv2.COLORMAP_JET``.

    Returns:
        np.ndarray: Blended image in RGB format with dtype ``uint8`` and
            shape (H, W, 3).
    """
    h, w = img_rgb.shape[:2]

    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

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
#  ATTENTION STATISTICS
# ═══════════════════════════════════════════════════════════


def compute_attention_stats(
    cam: np.ndarray,
    center_ratio: float = 0.6,
) -> tuple[float, float]:
    """
    Compute mean activation in the central region versus the border region.

    Divides the CAM map into a rectangular centre crop (defined by
    ``center_ratio``) and the surrounding border. Returns the mean
    activation in each zone, useful for detecting whether the model
    attends to clinically relevant central content or peripheral
    artifacts.

    Args:
        cam (np.ndarray): Activation map of shape (H, W) with values in
            [0, 1].
        center_ratio (float): Fraction of the spatial dimensions occupied
            by the central zone. A value of 0.6 means the central 60% of
            both height and width is considered the centre. Default is
            0.6.

    Returns:
        tuple[float, float]:
            - ``center_attention``: Mean activation inside the central
              rectangle. Returns 0.0 if the centre region is empty.
            - ``border_attention``: Mean activation in the surrounding
              border pixels. Returns 0.0 if no border pixels exist.
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
    Determine whether the peak activation falls inside the central region.

    Also computes what fraction of the strongest activations (top 10th
    percentile) are located within the centre, providing a soft measure
    of how well-localised the model's attention is.

    Args:
        cam (np.ndarray): Activation map of shape (H, W) with values in
            [0, 1].
        center_ratio (float): Fraction of the spatial dimensions that
            define the central zone. Default is 0.6.

    Returns:
        tuple[bool, float]:
            - ``max_in_center`` (bool): ``True`` if the pixel with the
              highest activation value lies inside the central rectangle.
            - ``center_strong_ratio`` (float): Proportion of pixels
              above the 90th-percentile threshold that fall within the
              central rectangle. Range [0, 1].
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
