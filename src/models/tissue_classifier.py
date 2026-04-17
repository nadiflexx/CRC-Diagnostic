"""
Tissue-Only Classifier — SAME backbone as Model A.
Diversity comes from preprocessing, not architecture.
"""

import torch
import torch.nn.functional as F

from src.models.image_classifier import ColonCancerClassifier


class TissueOnlyClassifier(ColonCancerClassifier):
    """
    Tissue-only image classifier that inherits directly from
    ``ColonCancerClassifier``.

    Shares the exact same EfficientNetV2-S backbone and classifier head as
    Model A. The only source of diversity between the two models is the
    preprocessing strategy applied before training:
        - Model A: multi-source standardization with padding.
        - Model B (this class): tissue-only crops with no padding and no
          border artifacts.

    Inheriting from ``ColonCancerClassifier`` means all forward-pass logic,
    weight initialisation, and dropout configuration are reused without
    modification.
    """

    def __init__(
        self,
        model_name: str = "tf_efficientnetv2_s.in21k",
        pretrained: bool = True,
        dropout: float = 0.4,
        num_classes: int = 3,
    ):
        """
        Initialise the tissue-only classifier.

        Delegates entirely to ``ColonCancerClassifier.__init__`` so all
        backbone and head construction logic is shared.

        Args:
            model_name (str): Timm model identifier for the backbone.
                Default is ``"tf_efficientnetv2_s.in21k"``.
            pretrained (bool): Whether to initialise the backbone with
                ImageNet-21k pretrained weights. Default is ``True``.
            dropout (float): Dropout probability applied in the
                classification head. Default is 0.4.
            num_classes (int): Number of output classes. Default is 3.
        """
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            dropout=dropout,
            num_classes=num_classes,
        )

    def predict_proba_multicrop(
        self,
        crops: list[torch.Tensor],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute averaged class probabilities over a list of image crops.

        Each crop is forwarded independently through the model. The resulting
        softmax probability vectors are stacked and averaged to produce a
        single probability distribution per image, reducing the variance
        introduced by crop position or content variation.

        The model is set to evaluation mode and gradients are disabled
        for the duration of this call.

        Args:
            crops (list[torch.Tensor]): List of image tensors, each of
                shape (C, H, W) or (1, C, H, W). If a crop has three
                dimensions it is automatically unsqueezed to add the
                batch dimension.
            temperature (float): Scalar divisor applied to the logits
                before the softmax. Values greater than 1 soften the
                distribution; values less than 1 sharpen it. Default is
                1.0 (no scaling).

        Returns:
            torch.Tensor: 1-D probability tensor of shape (num_classes,)
                representing the mean softmax output across all crops.
        """
        self.eval()
        all_probs = []
        with torch.no_grad():
            for crop in crops:
                if crop.dim() == 3:
                    crop = crop.unsqueeze(0)
                logits = self.forward(crop) / temperature
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)
        stacked = torch.cat(all_probs, dim=0)
        return stacked.mean(dim=0)
