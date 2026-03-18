"""
Attention analysis utilities extracted from evaluate_gradcam.
Used by DiagnosisEngine and EnsemblePredictor for runtime attention gating.
"""

import numpy as np


def compute_attention_stats(
    cam: np.ndarray, center_ratio: float = 0.6
) -> tuple[float, float]:
    """
    Measures center vs border attention at native CAM resolution.

    Returns:
        (center_attention, border_attention)
    """
    h, w = cam.shape
    margin_h = max(1, int(h * (1.0 - center_ratio) / 2))
    margin_w = max(1, int(w * (1.0 - center_ratio) / 2))

    center_mask = np.zeros((h, w), dtype=bool)
    center_mask[margin_h : h - margin_h, margin_w : w - margin_w] = True

    center_pixels = cam[center_mask]
    border_pixels = cam[~center_mask]

    center_att = float(center_pixels.mean()) if len(center_pixels) > 0 else 0.0
    border_att = float(border_pixels.mean()) if len(border_pixels) > 0 else 0.0

    return center_att, border_att


def compute_pointing_accuracy(
    cam: np.ndarray, cam_threshold: float = 0.5
) -> tuple[bool, float]:
    """
    Pointing Game (Selvaraju 2017).

    Returns:
        (max_in_center, center_strong_ratio)
    """
    h, w = cam.shape
    margin_h = max(1, int(h * 0.2))
    margin_w = max(1, int(w * 0.2))

    max_y, max_x = np.unravel_index(cam.argmax(), cam.shape)
    max_in_center = (
        margin_h <= max_y < h - margin_h and margin_w <= max_x < w - margin_w
    )

    strong_mask = cam > cam_threshold
    if strong_mask.sum() == 0:
        return max_in_center, 0.5

    center_mask = np.zeros((h, w), dtype=bool)
    center_mask[margin_h : h - margin_h, margin_w : w - margin_w] = True
    center_strong = (strong_mask & center_mask).sum()
    center_strong_ratio = float(center_strong / strong_mask.sum())

    return max_in_center, center_strong_ratio
