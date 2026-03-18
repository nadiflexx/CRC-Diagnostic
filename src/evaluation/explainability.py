"""
Explainability for image and tabular models.
"""

import io
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.config.logger import log as logger


class ImageExplainer:
    def __init__(self, classifier, segmenter, device="cpu"):
        self.classifier = classifier
        self.segmenter = segmenter
        self.device = device
        self._gradients = None
        self._activations = None
        self._hook_registered = False

    def _register_hooks(self):
        """
        Register hooks for gradient and activation computation.
        """
        if self._hook_registered:
            return
        target_layer = None
        for _, module in self.classifier.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        if target_layer is None:
            logger.warning("No conv layer found for GradCAM")
            return

        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
        self._hook_registered = True

    def generate_gradcam(self, input_tensor, original_image):
        """
        Generate Grad-CAM heatmap for the input image.
        :param input_tensor: Input tensor
        :param original_image: Original image
        :return: Overlay image and prediction score
        """
        self._register_hooks()
        self.classifier.eval()
        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        logits = self.classifier(input_tensor)
        score = torch.sigmoid(logits).item()
        self.classifier.zero_grad()
        logits.backward(retain_graph=True)
        if self._gradients is None or self._activations is None:
            return original_image.copy(), score
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        return overlay, score

    def generate_segmentation_overlay(
        self, input_tensor, original_image, threshold=0.5
    ):
        """
        Generate segmentation overlay for the input image.
        :param input_tensor: Input tensor
        :param original_image: Original image
        :param threshold: Threshold for binary mask
        :return: Overlay image, binary mask and bounding boxes
        """
        self.segmenter.eval()
        with torch.no_grad():
            logits = self.segmenter(input_tensor.to(self.device))
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        h, w = original_image.shape[:2]
        mask = cv2.resize(probs, (w, h))
        binary_mask = (mask > threshold).astype(np.uint8)
        bboxes = self._extract_bounding_boxes(binary_mask)
        overlay = original_image.copy()
        green_mask = np.zeros_like(overlay)
        green_mask[:, :, 1] = 255
        overlay[binary_mask == 1] = cv2.addWeighted(
            overlay[binary_mask == 1], 0.5, green_mask[binary_mask == 1], 0.5, 0
        )
        for bbox in bboxes:
            x, y, bw, bh = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
            cv2.putText(
                overlay,
                f"{bbox['confidence']:.0%}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
        return overlay, binary_mask, bboxes

    def _extract_bounding_boxes(self, binary_mask):
        """
        Extract bounding boxes from the binary mask.
        :param binary_mask: Binary mask
        :return: List of bounding boxes
        """
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bboxes = []
        total_area = binary_mask.shape[0] * binary_mask.shape[1]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < total_area * 0.001:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            roi = binary_mask[y : y + bh, x : x + bw]
            confidence = roi.sum() / (bw * bh) if bw * bh > 0 else 0
            bboxes.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "width": int(bw),
                    "height": int(bh),
                    "area_px": int(area),
                    "area_pct": round(area / total_area * 100, 2),
                    "confidence": round(float(confidence), 3),
                }
            )
        return sorted(bboxes, key=lambda b: b["area_px"], reverse=True)

    def generate_full_report_image(
        self, input_tensor, original_image, classification_score, save_path=None
    ):
        """
        Generate a full report image with Grad-CAM and segmentation overlay.

        param: input_tensor: Input tensor
        param: original_image: Original image
        param: classification_score: Classification score
        param: save_path: Path to save the report image
        return: Report image
        """
        gradcam_img, _ = self.generate_gradcam(input_tensor, original_image)
        seg_img, mask, bboxes = self.generate_segmentation_overlay(
            input_tensor, original_image
        )
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(original_image)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(gradcam_img)
        axes[1].set_title(f"GradCAM ({classification_score:.2%})")
        axes[1].axis("off")
        axes[2].imshow(seg_img)
        axes[2].set_title(f"Segmentation ({len(bboxes)} lesions)")
        axes[2].axis("off")
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        report = cv2.imdecode(
            np.frombuffer(buf.read(), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        report = cv2.cvtColor(report, cv2.COLOR_BGR2RGB)
        plt.close(fig)
        return report


class TabularExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self._explainer = None
        self._fitted = False

    def fit(self, X_background):
        try:
            import shap

            try:
                self._explainer = shap.TreeExplainer(self.model.model)
            except Exception:
                self._explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    shap.sample(X_background, min(100, len(X_background))),
                )
            self._fitted = True
            logger.info("SHAP explainer initialized")
        except ImportError:
            logger.warning("SHAP not installed")

    def explain(self, X):
        """
        Explain the prediction for the given input.
        :param X: Input data
        :return: Explanation
        """
        if self._explainer is not None and self._fitted:
            return self._explain_shap(X)
        return self._explain_feature_importance(X)

    def _explain_shap(self, X):
        """
        Explain the prediction for the given input using SHAP.
        :param X: Input data
        :return: Explanation
        """
        shap_values = self._explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values
        if sv.ndim > 1:
            sv = sv[0]
        paired = list(zip(self.feature_names, sv.tolist(), strict=True))
        paired_sorted = sorted(paired, key=lambda p: abs(p[1]), reverse=True)
        return {
            "shap_values": np.array(sv),
            "feature_names": self.feature_names,
            "top_risk_factors": [(n, v) for n, v in paired_sorted if v > 0],
            "protective_factors": [(n, v) for n, v in paired_sorted if v < 0],
            "base_value": float(
                self._explainer.expected_value[1]
                if isinstance(self._explainer.expected_value, (list, np.ndarray))
                else self._explainer.expected_value
            ),
        }

    def _explain_feature_importance(self, X):
        """
        Explain the prediction for the given input using feature importance.
        :param X: Input data
        :return: Explanation
        """
        try:
            importances = self.model.model.feature_importances_
        except AttributeError:
            importances = np.zeros(len(self.feature_names))
        paired = sorted(
            zip(self.feature_names, importances.tolist(), strict=True),
            key=lambda p: abs(p[1]),
            reverse=True,
        )
        return {
            "shap_values": importances,
            "feature_names": self.feature_names,
            "top_risk_factors": paired[:10],
            "protective_factors": [],
            "base_value": 0.0,
        }

    def plot_explanation(self, X, save_path=None):
        """
        Plot the explanation for the given input.
        :param X: Input data
        :param save_path: Path to save the plot
        """
        explanation = self.explain(X)
        sv = explanation["shap_values"]
        fig, ax = plt.subplots(figsize=(10, 8))
        sorted_idx = np.argsort(np.abs(sv))[-15:]
        colors = ["#D32F2F" if v > 0 else "#388E3C" for v in sv[sorted_idx]]
        names = [self.feature_names[i] for i in sorted_idx]
        ax.barh(range(len(sorted_idx)), sv[sorted_idx], color=colors)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Risk contribution")
        ax.set_title("Patient Risk Factors")
        ax.axvline(x=0, color="black", linewidth=0.5)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig
