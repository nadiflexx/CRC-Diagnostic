"""
Tissue-Only Classifier — SAME backbone as Model A.
Diversity comes from preprocessing, not architecture.
"""

import torch
import torch.nn.functional as F

from src.models.image_classifier import ColonCancerClassifier


class TissueOnlyClassifier(ColonCancerClassifier):
    def __init__(
        self,
        model_name: str = "tf_efficientnetv2_s.in21k",
        pretrained: bool = True,
        dropout: float = 0.4,
        num_classes: int = 3,
    ):
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
        Predict the probability of each class for multiple crops.

        :param crops: List of image crops
        :param temperature: Temperature for softmax
        :return: Average probabilities across all crops
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
