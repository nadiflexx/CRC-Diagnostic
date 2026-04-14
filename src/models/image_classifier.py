"""
Colon Cancer Classifier — 3 classes: Normal | Polyp | Inflammation.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.interfaces import BaseClassifier


class ColonCancerClassifier(BaseClassifier):
    """
    EfficientNetV2-S image classifier for 3-class colon cancer detection.

    Architecture:
        - Backbone: any Timm model (default ``tf_efficientnetv2_s.in21k``),
          instantiated with ``num_classes=0`` so that it outputs raw feature
          maps rather than class logits.
        - Classification head: ``AdaptiveAvgPool2d → Flatten → Dropout →
          Linear(feature_dim, 512) → BatchNorm1d → SiLU → Dropout/2 →
          Linear(512, num_classes)``.

    The backbone and head are trained end-to-end. The three output logits
    correspond to the classes Normal, Polyp, and Inflammation.
    """

    def __init__(
        self,
        model_name="tf_efficientnetv2_s.in21k",
        pretrained=True,
        dropout=0.4,
        num_classes=3,
    ):
        """
        Initialise the colon cancer classifier.

        Args:
            model_name (str): Timm model identifier. The backbone is
                created with ``num_classes=0`` to expose raw feature maps.
                Default is ``"tf_efficientnetv2_s.in21k"``.
            pretrained (bool): Whether to load ImageNet-21k pretrained
                weights for the backbone. Default is ``True``.
            dropout (float): Dropout probability applied after the first
                linear layer. Half of this value is applied before the
                final linear layer. Default is 0.4.
            num_classes (int): Number of output logits. Default is 3.
        """
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        """
        Perform a full forward pass through backbone and classification head.

        Args:
            x (torch.Tensor): Input image batch of shape (N, 3, H, W).

        Returns:
            torch.Tensor: Class logit tensor of shape (N, num_classes).
        """
        features = self.backbone.forward_features(x)
        return self.classifier(features)

    def predict_proba(self, x):
        """
        Return softmax class probabilities for the given input batch.

        The model is set to evaluation mode and gradients are disabled
        for the duration of this call.

        Args:
            x (torch.Tensor): Input image batch of shape (N, 3, H, W).

        Returns:
            torch.Tensor: Probability tensor of shape (N, num_classes)
                with values in [0, 1] summing to 1 along the class
                dimension.
        """
        self.eval()
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=1)


class FocalLoss(nn.Module):
    """
    Focal loss with label smoothing for multi-class classification.

    Focal loss down-weights well-classified examples so that the model
    focuses training signal on hard or misclassified samples.

    Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection".

    The label-smoothing term replaces the one-hot target with a soft
    distribution, reducing overconfidence:

        smooth_target[correct] = 1 - smoothing
        smooth_target[others]  = smoothing / (num_classes - 1)

    The final per-sample loss is:

        loss = -sum(smooth_target * focal_weight * log_probs)

    where ``focal_weight = alpha_t * (1 - p_t)^gamma``.
    """

    def __init__(self, alpha=None, gamma=2.0, num_classes=3, label_smoothing=0.1):
        """
        Initialise the focal loss.

        Args:
            alpha (list[float] | None): Per-class weighting factors of
                length ``num_classes``. Typically set to balanced class
                weights to counteract label imbalance. If ``None``, all
                classes are weighted equally. Default is ``None``.
            gamma (float): Focusing parameter that controls the rate at
                which easy examples are down-weighted. Higher values
                increase focus on hard samples. Default is 2.0.
            num_classes (int): Number of output classes. Used to construct
                the smooth label distribution. Default is 3.
            label_smoothing (float): Smoothing factor in [0, 1). A value
                of 0 corresponds to standard one-hot targets. Default is
                0.1.
        """
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.alpha = (
            torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None
        )

    def forward(self, inputs, targets):
        """
        Compute the focal loss for a batch of predictions.

        Constructs a smoothed target distribution, computes per-class
        log-softmax, applies the focal modulation factor
        ``(1 - p_t)^gamma``, optionally scales by class weights
        ``alpha``, and returns the mean loss over the batch.

        Args:
            inputs (torch.Tensor): Raw logit tensor of shape
                (N, num_classes).
            targets (torch.Tensor): Integer class label tensor of shape
                (N,) with values in ``[0, num_classes)``.

        Returns:
            torch.Tensor: Scalar mean focal loss over the batch.
        """
        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(self.label_smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_probs)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets].view(-1, 1)
            focal_weight = alpha_t * focal_weight

        loss = torch.sum(-true_dist * focal_weight * log_probs, dim=1)
        return loss.mean()
