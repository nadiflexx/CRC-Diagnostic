"""
Polyp segmenter: U-Net with EfficientNet-B4 encoder.
"""

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.interfaces import BaseSegmenter


class ColonPolypSegmenter(BaseSegmenter):
    """
    U-Net segmentation model with an EfficientNet-B4 encoder.

    Wraps ``segmentation_models_pytorch.Unet`` and adds convenience methods
    for inference with and without test-time augmentation (TTA), as well as
    post-processing utilities for connected-component instance filtering.
    """

    def __init__(
        self,
        encoder_name="efficientnet-b4",
        pretrained="imagenet",
    ):
        """
        Initialise the U-Net segmentation model.

        Args:
            encoder_name (str): Name of the encoder backbone supported by
                ``segmentation_models_pytorch``. Default is
                ``"efficientnet-b4"``.
            pretrained (str): Pre-trained weight source for the encoder.
                Default is ``"imagenet"``.
        """
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=pretrained,
            in_channels=3,
            classes=1,
            activation=None,
        )

    def forward(self, x):
        """
        Perform a forward pass through the U-Net.

        Args:
            x (torch.Tensor): Input image batch of shape (N, 3, H, W).

        Returns:
            torch.Tensor: Raw logit map of shape (N, 1, H, W).
        """
        return self.model(x)

    def predict_mask(self, x, threshold=0.5):
        """
        Predict a binary segmentation mask for the given input batch.

        The model is set to evaluation mode and gradients are disabled.
        Logits are passed through a sigmoid activation and then
        thresholded to produce a binary mask.

        Args:
            x (torch.Tensor): Input image batch of shape (N, 3, H, W).
            threshold (float): Probability threshold for binarisation.
                Pixels with sigmoid output above this value are set to 1.
                Default is 0.5.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Binary mask tensor of shape (N, 1, H, W) with values
                  in {0.0, 1.0}.
                - Sigmoid probability map of shape (N, 1, H, W) in
                  range [0, 1].
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            masks = (probs > threshold).float()
        return masks, probs

    def predict_mask_tta(self, x, threshold=0.5):
        """
        Predict a binary segmentation mask using test-time augmentation (TTA).

        Four augmentation variants are evaluated:
            1. Original image.
            2. Horizontal flip (reversed before averaging).
            3. Vertical flip (reversed before averaging).
            4. Both horizontal and vertical flip (reversed before averaging).

        The four probability maps are averaged and then thresholded.

        Args:
            x (torch.Tensor): Input image batch of shape (N, 3, H, W).
            threshold (float): Probability threshold for binarisation.
                Default is 0.5.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Binary mask tensor of shape (N, 1, H, W) with values
                  in {0.0, 1.0}.
                - Averaged sigmoid probability map of shape (N, 1, H, W)
                  in range [0, 1].
        """
        self.eval()
        with torch.no_grad():
            probs_list = [torch.sigmoid(self.forward(x))]

            x_hflip = torch.flip(x, dims=[3])
            p_hflip = torch.sigmoid(self.forward(x_hflip))
            probs_list.append(torch.flip(p_hflip, dims=[3]))

            x_vflip = torch.flip(x, dims=[2])
            p_vflip = torch.sigmoid(self.forward(x_vflip))
            probs_list.append(torch.flip(p_vflip, dims=[2]))

            x_both = torch.flip(x, dims=[2, 3])
            p_both = torch.sigmoid(self.forward(x_both))
            probs_list.append(torch.flip(p_both, dims=[2, 3]))

        probs = torch.stack(probs_list).mean(dim=0)
        masks = (probs > threshold).float()
        return masks, probs

    @staticmethod
    def postprocess_instances(mask_np, min_area=100):
        """
        Clean and label connected components in a binary segmentation mask.

        Applies morphological opening followed by closing to remove noise
        and fill gaps, then performs connected-component analysis. Components
        smaller than ``min_area`` pixels are discarded. Surviving components
        are relabelled with consecutive integer IDs starting from 1.

        Args:
            mask_np (np.ndarray): Binary mask array of shape (H, W) with
                float values in [0, 1]. Internally converted to uint8.
            min_area (int): Minimum number of pixels required for a
                connected component to be retained. Default is 100.

        Returns:
            tuple[np.ndarray, int]:
                - Instance-labelled mask of shape (H, W) where each
                  valid polyp region is assigned a unique integer ID
                  (background = 0).
                - Total number of valid polyp instances detected.
        """
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_clean, connectivity=8
        )
        valid_mask = np.zeros_like(mask_np)
        polyp_id = 0
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                polyp_id += 1
                valid_mask[labels == i] = polyp_id
        return valid_mask, polyp_id


class DiceBCELoss(nn.Module):
    """
    Combined Dice + Binary Cross-Entropy + Boundary loss for binary
    segmentation.

    The total loss is computed as a weighted sum of three components:

        loss = dice_weight * Dice_loss
             + bce_weight * BCE_loss
             + boundary_weight * Boundary_loss

    The boundary loss up-weights pixels near the mask boundary using a
    morphologically derived weight map, encouraging the model to produce
    sharper and more accurate contour predictions.
    """

    def __init__(
        self, dice_weight=0.5, bce_weight=0.3, boundary_weight=0.2, smooth=1.0
    ):
        """
        Initialise the combined segmentation loss.

        Args:
            dice_weight (float): Weight applied to the Dice loss component.
                Default is 0.5.
            bce_weight (float): Weight applied to the binary cross-entropy
                loss component. Default is 0.3.
            boundary_weight (float): Weight applied to the boundary-aware
                loss component. Set to 0.0 to disable boundary loss.
                Default is 0.2.
            smooth (float): Smoothing constant added to the numerator and
                denominator of the Dice score to prevent division by zero.
                Default is 1.0.
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight
        self.smooth = smooth

    def _compute_boundary_mask(self, targets, kernel_size=3):
        """
        Derive a boundary mask from a binary target map.

        Boundary pixels are identified as the difference between a dilated
        and an eroded version of the target, both computed with
        ``nn.MaxPool2d``.

        Args:
            targets (torch.Tensor): Binary ground-truth mask of shape
                (N, 1, H, W) with float values in {0.0, 1.0}.
            kernel_size (int): Spatial size of the pooling kernel used for
                morphological dilation and erosion. Default is 3.

        Returns:
            torch.Tensor: Boundary mask of shape (N, 1, H, W) with values
                clamped to [0, 1]. Non-zero pixels lie on or near the
                mask boundary.
        """
        pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        dilated = pool(targets)
        eroded = 1.0 - pool(1.0 - targets)
        return (dilated - eroded).clamp(0, 1)

    def forward(self, inputs, targets):
        """
        Compute the combined Dice + BCE + Boundary loss.

        If ``targets`` has three dimensions (H, W, or N, H, W without a
        channel axis), an unsqueeze is applied to ensure the shape is
        (N, 1, H, W) before loss computation.

        Args:
            inputs (torch.Tensor): Raw logit predictions of shape
                (N, 1, H, W). Sigmoid is applied internally.
            targets (torch.Tensor): Binary ground-truth masks of shape
                (N, 1, H, W) or (N, H, W). Float conversion is applied
                internally.

        Returns:
            torch.Tensor: Scalar loss tensor representing the weighted
                combination of Dice, BCE, and boundary losses.
        """
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(inputs, targets)

        probs = torch.sigmoid(inputs)
        batch_size = inputs.shape[0]
        dice_loss = 0.0
        for i in range(batch_size):
            pred_flat = probs[i].view(-1)
            target_flat = targets[i].view(-1)
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += 1.0 - dice_score
        dice_loss = dice_loss / batch_size

        boundary_loss = 0.0
        if self.boundary_weight > 0:
            boundary_mask = self._compute_boundary_mask(targets)
            weight_map = 1.0 + 2.0 * boundary_mask
            boundary_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, weight=weight_map
            )

        return (
            self.dice_weight * dice_loss
            + self.bce_weight * bce
            + self.boundary_weight * boundary_loss
        )
