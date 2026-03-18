"""
Colon Cancer Classifier — 3 classes: Normal | Polyp | Inflammation.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.interfaces import BaseClassifier


class ColonCancerClassifier(BaseClassifier):
    def __init__(
        self,
        model_name="tf_efficientnetv2_s.in21k",
        pretrained=True,
        dropout=0.4,
        num_classes=3,
    ):
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
        Forward pass through the model.

        :param x: Input tensor
        :return: Output tensor
        """
        features = self.backbone.forward_features(x)
        return self.classifier(features)

    def predict_proba(self, x):
        """
        Predict the probability of each class for the given input.

        :param x: Input tensor
        :return: Probability tensor
        """
        self.eval()
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=1)


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.
    """

    def __init__(self, alpha=None, gamma=2.0, num_classes=3, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.alpha = (
            torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None
        )

    def forward(self, inputs, targets):
        """
        Forward pass through the Focal loss.

        :param inputs: Input tensor
        :param targets: Target tensor
        :return: Loss tensor
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
