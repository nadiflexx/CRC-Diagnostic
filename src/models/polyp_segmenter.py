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
    def __init__(
        self,
        encoder_name="efficientnet-b4",
        pretrained="imagenet",
    ):
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
        Forward pass through the model.

        :param x: Input tensor
        :return: Output tensor
        """
        return self.model(x)

    def predict_mask(self, x, threshold=0.5):
        """
        Predict the segmentation mask for the given input.

        :param x: Input tensor
        :param threshold: Threshold for binarizing the probabilities
        :return: Binarized mask and segmentation probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            masks = (probs > threshold).float()
        return masks, probs

    def predict_mask_tta(self, x, threshold=0.5):
        """
        Predict the segmentation mask for the given input using test-time augmentation (TTA).

        param: x: Input tensor
        param: threshold: Threshold for binarizing the probabilities
        return: Binarized mask and segmentation probabilities
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
        Post-process the segmentation instances.

        :param mask_np: Numpy array of the segmentation mask
        :param min_area: Minimum area for valid instances
        :return: Processed mask and number of polyps
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
    Dice + BCE loss for binary segmentation.
    """

    def __init__(
        self, dice_weight=0.5, bce_weight=0.3, boundary_weight=0.2, smooth=1.0
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight
        self.smooth = smooth

    def _compute_boundary_mask(self, targets, kernel_size=3):
        """
        Compute the boundary mask for the given targets.

        :param targets: Target tensor
        :param kernel_size: Size of the kernel for morphological operations
        :return: Boundary mask
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
        Compute the loss for the given inputs and targets.

        :param inputs: Input tensor
        :param targets: Target tensor
        :return: Computed loss
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
