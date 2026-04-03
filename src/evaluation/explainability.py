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


class GenericTabularExplainer:
    """
    Explainer for GenericTabularMLPClassifier model using permutation importance.
    Provides explanations for individual predictions and visualization of risk factors.
    """

    def __init__(self, model, feature_names, X_background=None):
        """
        Initialize the explainer.

        :param model: GenericTabularMLPClassifier instance (fitted)
        :param feature_names: List of feature names
        :param X_background: Background data for SHAP (optional, for future use)
        """
        self.model = model
        self.feature_names = np.array(feature_names)
        self.X_background = X_background
        self._shap_explainer = None
        self._permutation_importances = None
        self._fitted = False

    def fit(self, X_train):
        """
        Fit the explainer with background data.

        :param X_train: Training data for computing permutation importance
        """
        # Siempre computar permutation importance como fallback principal
        self._compute_permutation_importance(X_train)
        
        try:
            import shap

            # Intentar usar KernelExplainer con muestras del background data
            background_sample = shap.sample(
                X_train, min(50, len(X_train)), random_state=42
            )
            self._shap_explainer = shap.KernelExplainer(
                self.model.predict_proba, background_sample
            )
            self._fitted = True
            logger.info("SHAP KernelExplainer initialized for GenericTabularMLPClassifier")
        except ImportError:
            logger.warning("SHAP not installed, using permutation importance only")
            self._fitted = True
        except Exception as e:
            logger.warning(f"SHAP initialization failed: {e}, using permutation importance only")
            self._fitted = True

    def _compute_permutation_importance(self, X_train):
        """
        Compute permutation importance for the features.

        :param X_train: Training data
        """
        from sklearn.inspection import permutation_importance

        # Compute permutation importance
        result = permutation_importance(
            self.model,
            X_train,
            self.model.predict(X_train),  # Use model predictions
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        self._permutation_importances = result.importances_mean
        self._fitted = True
        logger.info("Permutation importance computed for GenericTabularMLPClassifier")

    def explain(self, X, y_pred_proba=None):
        """
        Explain the prediction for the given input.

        :param X: Input data (single sample or batch)
        :param y_pred_proba: Pre-computed prediction probabilities (optional)
        :return: Explanation dictionary
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if y_pred_proba is None:
            y_pred_proba = self.model.predict_proba(X)

        # Priorizar permutation importance por robustez
        if self._permutation_importances is not None:
            return self._explain_permutation(X, y_pred_proba)
        elif self._shap_explainer is not None and self._fitted:
            return self._explain_shap(X, y_pred_proba)
        else:
            return self._explain_simple(X, y_pred_proba)

    def _explain_shap(self, X, y_pred_proba):
        """
        Explain using SHAP values.

        :param X: Input data
        :param y_pred_proba: Prediction probabilities
        :return: Explanation dictionary
        """
        try:
            shap_values = self._shap_explainer.shap_values(X)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classification returns list of [neg_class, pos_class]
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                sv = shap_values
            
            # Ensure sv is 1D array
            if hasattr(sv, 'ndim'):
                if sv.ndim > 1:
                    sv = sv[0]
            else:
                # Convert to numpy array if it's not
                sv = np.array(sv).flatten()
            
            # Ensure numeric values
            sv = np.asarray(sv, dtype=float)
            
            paired = list(zip(self.feature_names, sv.tolist(), strict=True))
            paired_sorted = sorted(paired, key=lambda p: abs(p[1]), reverse=True)

            return {
                "shap_values": sv,
                "feature_names": self.feature_names.tolist(),
                "top_risk_factors": [(n, float(v)) for n, v in paired_sorted if v > 0],
                "protective_factors": [(n, float(v)) for n, v in paired_sorted if v < 0],
                "prediction_proba": float(y_pred_proba[0, 1]),
                "base_value": float(
                    self._shap_explainer.expected_value[1]
                    if isinstance(self._shap_explainer.expected_value, (list, np.ndarray))
                    else self._shap_explainer.expected_value
                ),
                "method": "SHAP",
            }
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}, falling back to permutation importance")
            return self._explain_permutation(X, y_pred_proba)

    def _explain_permutation(self, X, y_pred_proba):
        """
        Explain using permutation importance.

        :param X: Input data
        :param y_pred_proba: Prediction probabilities
        :return: Explanation dictionary
        """
        if self._permutation_importances is None:
            return self._explain_simple(X, y_pred_proba)
        
        importances = self._permutation_importances
        
        # Ensure importances is numeric array
        if not isinstance(importances, np.ndarray):
            importances = np.array(importances, dtype=float)
        
        paired = sorted(
            zip(self.feature_names, importances.tolist(), strict=True),
            key=lambda p: abs(p[1]),
            reverse=True,
        )

        return {
            "shap_values": importances,
            "feature_names": self.feature_names.tolist(),
            "top_risk_factors": [(n, float(v)) for n, v in paired if v > 0][:10],
            "protective_factors": [(n, float(v)) for n, v in paired if v < 0][:10],
            "prediction_proba": float(y_pred_proba[0, 1]),
            "base_value": 0.0,
            "method": "Permutation Importance",
        }

    def _explain_simple(self, X, y_pred_proba):
        """
        Simple explanation based on feature values relative to background.

        :param X: Input data
        :param y_pred_proba: Prediction probabilities
        :return: Explanation dictionary
        """
        # Use mean of background data or zeros as baseline
        if self.X_background is not None:
            baseline = np.mean(self.X_background, axis=0)
        else:
            baseline = np.zeros(len(self.feature_names))

        # Simple contribution: difference from baseline
        contribution = np.abs(X[0] - baseline)
        paired = sorted(
            zip(self.feature_names, contribution.tolist(), strict=True),
            key=lambda p: abs(p[1]),
            reverse=True,
        )

        return {
            "shap_values": contribution,
            "feature_names": self.feature_names.tolist(),
            "top_risk_factors": paired[:10],
            "protective_factors": [],
            "prediction_proba": float(y_pred_proba[0, 1]),
            "base_value": 0.0,
            "method": "Feature Deviation",
        }

    def plot_explanation(
        self, X, y_pred_proba=None, top_n=15, save_path=None, figsize=(10, 8)
    ):
        """
        Plot explanation for a single sample.

        :param X: Input data (single sample)
        :param y_pred_proba: Pre-computed prediction probability (optional)
        :param top_n: Number of top features to display
        :param save_path: Path to save the plot
        :param figsize: Figure size
        :return: Figure object
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        explanation = self.explain(X, y_pred_proba)
        sv = explanation["shap_values"]
        method = explanation["method"]
        pred_prob = explanation["prediction_proba"]

        fig, ax = plt.subplots(figsize=figsize)
        sorted_idx = np.argsort(np.abs(sv))[-top_n:]
        colors = ["#D32F2F" if v > 0 else "#388E3C" for v in sv[sorted_idx]]
        names = [self.feature_names[i] for i in sorted_idx]

        bars = ax.barh(range(len(sorted_idx)), sv[sorted_idx], color=colors)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("Contribution / Importance", fontsize=11)
        ax.set_title(
            f"Risk Factors — CRC Probability: {pred_prob:.1%} ({method})",
            fontsize=12,
            fontweight="bold",
        )
        ax.axvline(x=0, color="black", linewidth=0.8)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, sv[sorted_idx])):
            x_pos = val + (max(sv[sorted_idx]) * 0.02 if val > 0 else -max(sv[sorted_idx]) * 0.02)
            ax.text(
                x_pos,
                i,
                f"{val:.3f}",
                va="center",
                ha="left" if val > 0 else "right",
                fontsize=9,
            )

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
            logger.info(f"Explanation plot saved to {save_path}")
        plt.close(fig)
        return fig

    def plot_batch_comparison(self, X, y_true=None, save_path=None):
        """
        Plot comparison of predictions and feature importances for multiple samples.

        :param X: Input data (multiple samples)
        :param y_true: True labels (optional, for accuracy visualization)
        :param save_path: Path to save the plot
        :return: Figure object
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        y_pred_proba = self.model.predict_proba(X)
        y_pred = self.model.predict(X)

        # Create subplots: probabilities + feature importance
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Top plot: Prediction probabilities
        ax_prob = axes[0]
        x_pos = np.arange(len(y_pred_proba))
        colors_pred = ["#D32F2F" if pred == 1 else "#388E3C" for pred in y_pred]
        bars = ax_prob.bar(x_pos, y_pred_proba[:, 1], color=colors_pred, alpha=0.7, edgecolor="black")

        # Add true labels if provided
        if y_true is not None:
            for i, (true_label, bar) in enumerate(zip(y_true, bars)):
                marker = "✓" if (true_label == y_pred[i]) else "✗"
                ax_prob.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    marker,
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                    color="green" if (true_label == y_pred[i]) else "red",
                )

        ax_prob.set_xlabel("Sample", fontsize=11)
        ax_prob.set_ylabel("CRC Probability", fontsize=11)
        ax_prob.set_title("Predicted Probabilities", fontsize=12, fontweight="bold")
        ax_prob.set_ylim([0, 1.1] if y_true is not None else [0, 1])
        ax_prob.grid(axis="y", alpha=0.3)

        # Bottom plot: Average feature importance
        if self._permutation_importances is not None:
            ax_imp = axes[1]
            importances = self._permutation_importances
            sorted_idx = np.argsort(np.abs(importances))[-15:]
            colors_imp = ["#D32F2F" if v > 0 else "#388E3C" for v in importances[sorted_idx]]
            ax_imp.barh(range(len(sorted_idx)), importances[sorted_idx], color=colors_imp)
            ax_imp.set_yticks(range(len(sorted_idx)))
            ax_imp.set_yticklabels([self.feature_names[i] for i in sorted_idx])
            ax_imp.set_xlabel("Importance", fontsize=11)
            ax_imp.set_title("Feature Importance (Permutation-based)", fontsize=12, fontweight="bold")
            ax_imp.axvline(x=0, color="black", linewidth=0.8)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
            logger.info(f"Batch comparison plot saved to {save_path}")
        plt.close(fig)
        return fig
