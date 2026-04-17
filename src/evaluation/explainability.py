"""
src/diagnosis/explainability.py

Explainability utilities for image and tabular models.

Image
-----
``ImageExplainer`` centralises all visualisation logic:
    - Grad-CAM heatmap composition (Model A, Model B, weighted fusion).
    - Segmentation mask post-processing and overlay persistence.

Tabular
-------
``TabularExplainer`` provides SHAP-based feature attribution with a
feature-importance fallback when SHAP is not installed.
"""

from datetime import datetime
from zipfile import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config.logger import log as logger
from src.config.paths import paths
from src.evaluation.gradcam import (
    CamMethod,
    create_heatmap_overlay,
    generate_gradcam,
)
from src.models.polyp_segmenter import ColonPolypSegmenter
from src.utils.ml_utils import numpy_to_base64


class ImageExplainer:
    """
    Visualisation helper for the image classification and segmentation
    pipeline.

    Provides:
    - Grad-CAM heatmap generation via ``src.evaluation.gradcam``.
    - Segmentation mask post-processing, overlay composition, and
      report persistence.

    Both methods are stateless with respect to model weights; they
    receive pre-computed tensors or masks and return visualisation
    artefacts.
    """

    def __init__(
        self,
        ensemble,
        img_transform,
        class_names: dict[int, str],
        device: str = "cpu",
        cam_method: CamMethod = "gradcam++",
    ):
        """
        Initialise the image explainer.

        Args:
            ensemble (EnsemblePredictor): Loaded ensemble used to access
                ``model_a``, ``model_b``, ``preprocessor_a``,
                ``preprocessor_b``, and ``ensemble_available``.
            img_transform: Albumentations validation transform applied
                before feeding images to the models.
            class_names (dict[int, str]): Mapping from class index to
                human-readable label.
            device (str): Target device for tensor operations. Default
                is ``"cpu"``.
            cam_method (CamMethod): Grad-CAM algorithm to use. One of
                ``"gradcam"``, ``"gradcam++"``, ``"hirescam"``, or
                ``"scorecam"``. Default is ``"gradcam++"``.
        """
        self.ensemble = ensemble
        self.img_transform = img_transform
        self.class_names = class_names
        self.device = device
        self.cam_method = cam_method

    # ─────────────────────────────────────────────
    #  Grad-CAM
    # ─────────────────────────────────────────────

    def generate_gradcam(
        self,
        image_bgr: np.ndarray,
        pred_idx: int,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> dict:
        """
        Generate Grad-CAM activation maps for Model A, Model B, and
        their weighted fusion.

        Available only when the PyTorch ``EnsemblePredictor`` is loaded
        (pure PyTorch mode or hybrid ONNX + PyTorch mode). The fusion
        heatmap uses the same ``alpha``/``beta`` weights that determined
        the prediction so that the visualisation faithfully reflects the
        ensemble decision.

        Args:
            image_bgr (np.ndarray): Original BGR image of shape
                (H, W, 3).
            pred_idx (int): Target class index for the Grad-CAM
                explanation.
            alpha (float): Weight of Model A in the fused heatmap.
                Default is 0.5.
            beta (float): Weight of Model B in the fused heatmap.
                Default is 0.5.

        Returns:
            dict: Dictionary with keys ``"gradcam_a"``,
                ``"gradcam_b"``, and ``"gradcam_fusion"``, each holding
                a base64 PNG data URI or ``None`` if unavailable.
        """
        result: dict[str, str | None] = {
            "gradcam_a": None,
            "gradcam_b": None,
            "gradcam_fusion": None,
        }

        preprocessed_a = self.ensemble.preprocessor_a.process_image(image_bgr)
        preprocessed_a_rgb = cv2.cvtColor(preprocessed_a, cv2.COLOR_BGR2RGB)
        tensor_a = (
            self.img_transform(image=preprocessed_a_rgb)["image"]
            .unsqueeze(0)
            .to(self.device)
        )

        cam_a = self._compute_cam_a(tensor_a, preprocessed_a_rgb, pred_idx, result)
        cam_b_resized = self._compute_cam_b(image_bgr, pred_idx, beta, cam_a, result)
        self._compute_fusion(
            preprocessed_a_rgb, cam_a, cam_b_resized, alpha, beta, result
        )

        return result

    def _compute_cam_a(
        self,
        tensor_a,
        preprocessed_a_rgb: np.ndarray,
        pred_idx: int,
        result: dict,
    ) -> np.ndarray | None:
        """
        Run Grad-CAM for Model A and store the overlay in ``result``.

        Args:
            tensor_a (torch.Tensor): Pre-processed tensor of shape
                (1, C, H, W) on the correct device.
            preprocessed_a_rgb (np.ndarray): RGB image used as the
                overlay background, shape (H, W, 3).
            pred_idx (int): Target class index.
            result (dict): Output dictionary; ``"gradcam_a"`` is
                populated in-place on success.

        Returns:
            np.ndarray | None: Raw CAM array of shape (H, W) normalised
                to [0, 1], or ``None`` if generation fails.
        """
        try:
            cam_a, _ = generate_gradcam(
                model=self.ensemble.model_a,
                input_tensor=tensor_a,
                target_class=pred_idx,
                method=self.cam_method,
            )
            overlay_a = create_heatmap_overlay(preprocessed_a_rgb, cam_a, alpha=0.45)
            result["gradcam_a"] = numpy_to_base64(overlay_a)
            return cam_a
        except Exception as exc:
            logger.warning(f"  ⚠️ Grad-CAM A: {exc}")
            return None

    def _compute_cam_b(
        self,
        image_bgr: np.ndarray,
        pred_idx: int,
        beta: float,
        cam_a: np.ndarray | None,
        result: dict,
    ) -> np.ndarray | None:
        """
        Run Grad-CAM for Model B (first tissue crop) and store the
        overlay in ``result``.

        Skipped when the ensemble is unavailable, ``model_b`` is
        ``None``, or ``beta`` is negligible (< 0.01).

        Args:
            image_bgr (np.ndarray): Original BGR image of shape
                (H, W, 3).
            pred_idx (int): Target class index.
            beta (float): Weight of Model B; used only to decide
                whether to skip computation.
            cam_a (np.ndarray | None): Model A CAM used to determine
                the target shape for resizing.
            result (dict): Output dictionary; ``"gradcam_b"`` is
                populated in-place on success.

        Returns:
            np.ndarray | None: Model B CAM resized to match ``cam_a``
                shape, or ``None`` if skipped or generation fails.
        """
        if (
            not self.ensemble.ensemble_available
            or self.ensemble.model_b is None
            or beta <= 0.01
        ):
            return None

        try:
            crops_bgr = self.ensemble.preprocessor_b.process_image(image_bgr)
            if not crops_bgr:
                return None

            crop_rgb = cv2.cvtColor(crops_bgr[0], cv2.COLOR_BGR2RGB)
            tensor_b = (
                self.img_transform(image=crop_rgb)["image"].unsqueeze(0).to(self.device)
            )
            cam_b, _ = generate_gradcam(
                model=self.ensemble.model_b,
                input_tensor=tensor_b,
                target_class=pred_idx,
                method=self.cam_method,
            )
            overlay_b = create_heatmap_overlay(crop_rgb, cam_b, alpha=0.45)
            result["gradcam_b"] = numpy_to_base64(overlay_b)

            if cam_a is not None:
                return cv2.resize(
                    cam_b,
                    (cam_a.shape[1], cam_a.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
        except Exception as exc:
            logger.warning(f"  ⚠️ Grad-CAM B: {exc}")

        return None

    def _compute_fusion(
        self,
        preprocessed_a_rgb: np.ndarray,
        cam_a: np.ndarray | None,
        cam_b_resized: np.ndarray | None,
        alpha: float,
        beta: float,
        result: dict,
    ) -> None:
        """
        Compute the weighted fusion heatmap and store it in ``result``.

        When Model B is unavailable or ``beta`` is negligible, Model A's
        CAM is used as the fusion map unmodified.

        Args:
            preprocessed_a_rgb (np.ndarray): RGB background image of
                shape (H, W, 3).
            cam_a (np.ndarray | None): Model A CAM of shape (H, W).
            cam_b_resized (np.ndarray | None): Model B CAM already
                resized to match ``cam_a``, or ``None``.
            alpha (float): Model A weight.
            beta (float): Model B weight.
            result (dict): Output dictionary; ``"gradcam_fusion"`` is
                populated in-place.
        """
        if cam_a is None:
            return

        if cam_b_resized is not None and beta > 0.01:
            cam_fusion = alpha * cam_a + beta * cam_b_resized
            rng = cam_fusion.max() - cam_fusion.min()
            cam_fusion = (cam_fusion - cam_fusion.min()) / rng if rng > 1e-8 else cam_a
        else:
            cam_fusion = cam_a

        overlay_fusion = create_heatmap_overlay(
            preprocessed_a_rgb, cam_fusion, alpha=0.45
        )
        result["gradcam_fusion"] = numpy_to_base64(overlay_fusion)

    # ─────────────────────────────────────────────
    #  Segmentation overlay
    # ─────────────────────────────────────────────

    def postprocess_and_save_mask(
        self,
        mask_np: np.ndarray,
        preprocessed: np.ndarray,
    ) -> tuple[bool, str | None, int]:
        """
        Post-process a binary mask and save the annotated overlay to
        disk.

        Applies morphological cleaning and connected-component labelling
        via ``ColonPolypSegmenter.postprocess_instances``. Draws a
        semi-transparent red highlight over detected regions and green
        contours around each instance. The overlay is saved as a
        timestamped PNG under ``paths.REPORTS``.

        Shared by both ONNX and PyTorch segmentation backends.

        Args:
            mask_np (np.ndarray): Binary float32 mask of shape (H, W)
                with values in {0.0, 1.0}.
            preprocessed (np.ndarray): BGR image of shape (H, W, 3)
                used as the drawing canvas.

        Returns:
            tuple[bool, str | None, int]:
                - ``True`` if at least one valid polyp instance was
                  found.
                - URL string pointing to the saved report image, or
                  ``None`` if no instances were found.
                - Number of valid polyp instances detected.
        """
        instance_mask, n_polyps = ColonPolypSegmenter.postprocess_instances(
            mask_np, min_area=100
        )
        if n_polyps == 0:
            return False, None, 0

        overlay = self._draw_segmentation_overlay(preprocessed, instance_mask, n_polyps)
        report_url = self._save_report(overlay)
        return True, report_url, n_polyps

    def _draw_segmentation_overlay(
        self,
        preprocessed: np.ndarray,
        instance_mask: np.ndarray,
        n_polyps: int,
    ) -> np.ndarray:
        """
        Compose the segmentation overlay image.

        Renders a semi-transparent red highlight over masked pixels and
        draws green contours around each individual polyp instance.

        Args:
            preprocessed (np.ndarray): BGR source image of shape
                (H, W, 3).
            instance_mask (np.ndarray): Integer mask of shape (H, W)
                where pixel value ``k`` belongs to instance ``k``
                (0 = background).
            n_polyps (int): Number of polyp instances (max label in
                ``instance_mask``).

        Returns:
            np.ndarray: Annotated BGR overlay image of shape (H, W, 3).
        """
        overlay = preprocessed.copy()
        h, w = overlay.shape[:2]

        # ── red semi-transparent highlight ───────────────────────────
        binary_mask = (instance_mask > 0).astype(np.float32)
        mask_resized = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        red_layer = overlay.copy()
        red_layer[:, :, 2] = np.clip(
            red_layer[:, :, 2].astype(int) + 100, 0, 255
        ).astype(np.uint8)
        red_layer[:, :, 0] = (red_layer[:, :, 0] * 0.5).astype(np.uint8)
        red_layer[:, :, 1] = (red_layer[:, :, 1] * 0.5).astype(np.uint8)

        idx = mask_resized > 0.5
        overlay[idx] = cv2.addWeighted(overlay, 0.5, red_layer, 0.5, 0)[idx]

        # ── green contours per instance ───────────────────────────────
        for pid in range(1, n_polyps + 1):
            single = (instance_mask == pid).astype(np.uint8) * 255
            single_rsz = cv2.resize(single, (w, h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(
                single_rsz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        return overlay

    def _save_report(self, overlay: np.ndarray) -> str:
        """
        Persist the overlay image to ``paths.REPORTS`` and return its
        URL.

        The file name includes a UTC timestamp to avoid collisions.

        Args:
            overlay (np.ndarray): BGR image to save.

        Returns:
            str: URL of the form
                ``http://localhost:8000/static/reports/<filename>``.
        """
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = paths.REPORTS
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"report_{ts}.png"
        cv2.imwrite(str(out_dir / filename), overlay)
        return f"http://localhost:8000/static/reports/{filename}"


# ══════════════════════════════════════════════════════════════════════
#  Tabular explainer
# ══════════════════════════════════════════════════════════════════════


class TabularExplainer:
    """
    SHAP-based explainability helper for the tabular cancer risk model.

    Attempts to use ``shap.TreeExplainer`` for gradient-boosted models
    (XGBoost / LightGBM) and falls back to ``shap.KernelExplainer`` for
    any other model type. If SHAP is not installed, feature importances
    from the underlying model are used instead.
    """

    def __init__(self, model, feature_names: list[str]):
        """
        Initialise the tabular explainer.

        Args:
            model: Trained model instance with a ``predict_proba``
                method and optionally a ``model`` attribute exposing
                ``feature_importances_``.
            feature_names (list[str]): Ordered list of feature names
                corresponding to the columns of the input matrix.
        """
        self.model = model
        self.feature_names = feature_names
        self._explainer = None
        self._fitted = False

    # ─────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────

    def fit(self, X_background) -> None:
        """
        Initialise the SHAP explainer using background data.

        Tries ``shap.TreeExplainer`` first (fast, exact for tree
        models). Falls back to ``shap.KernelExplainer`` with a random
        subsample of up to 100 background points if the tree explainer
        fails. If SHAP is not installed the method logs a warning and
        returns without raising.

        Args:
            X_background (array-like of shape (n_samples, n_features)):
                Background dataset used to marginalise features in
                ``KernelExplainer``.
        """
        try:
            import shap

            try:
                self._explainer = shap.TreeExplainer(self.model.model)
            except Exception:
                background_sample = shap.sample(
                    X_background, min(50, len(X_background)), random_state=42
                )
                self._explainer = shap.KernelExplainer(
                    self.model.predict_proba, background_sample
                )
            logger.info("SHAP explainer initialized")
        except ImportError:
            logger.warning("SHAP not installed – using permutation importance only")
        except Exception as e:
            logger.debug(f"SHAP initialization failed: {e}")

        self._fitted = True

    def explain(self, X, y_pred_proba=None) -> dict:
        """
        Generate a feature-level explanation for the given input.

        Delegates to ``_explain_shap`` if the SHAP explainer has been
        fitted, otherwise falls back to ``_explain_feature_importance``.

        Args:
            X (array-like of shape (1, n_features)): Single-sample
                input to explain.

        Returns:
            dict: Explanation dictionary.
        """
        if self._explainer is not None and self._fitted:
            return self._explain_shap(X)
        return self._explain_feature_importance(X)

    def plot_explanation(
        self, X, y_pred_proba=None, top_n=15, save_path=None, figsize=(10, 8)
    ):
        """
        Render a horizontal bar chart of the top-15 SHAP contributions.

        Positive SHAP values are drawn in red (``#D32F2F``); negative
        values in green (``#388E3C``).

        Args:
            X (array-like of shape (1, n_features)): Input to explain.
            save_path (str | None): Optional file path where the PNG
                will be saved at 150 DPI. Default is ``None``.

        Returns:
            matplotlib.figure.Figure: The closed figure object.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        explanation = self.explain(X, y_pred_proba)
        sv = explanation["shap_values"]
        method = explanation.get("method", "Unknown")
        pred_prob = explanation.get("prediction_proba", 0.0)

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

        Creates a two-panel visualization: top panel shows predicted probabilities,
        bottom panel shows feature importances.

        Args:
            X (array-like of shape (n_samples, n_features)): Input data.
            y_true (array-like, optional): True labels for accuracy marking.
                Default is ``None``.
            save_path (str | None): Optional file path to save the plot.
                Default is ``None``.

        Returns:
            matplotlib.figure.Figure: The closed figure object.
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

    # ─────────────────────────────────────────────
    #  Private helpers
    # ─────────────────────────────────────────────

    def _compute_permutation_importance(self, X_train):
        """
        Compute permutation importance for the features.

        Args:
            X_train (array-like of shape (n_samples, n_features)):
                Training data for computing importance.
        """
        from sklearn.inspection import permutation_importance

        # Compute permutation importance
        result = permutation_importance(
            self.model,
            X_train,
            self.model.predict(X_train),
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        self._permutation_importances = result.importances_mean
        logger.info("Permutation importance computed")

    def _explain_shap(self, X, y_pred_proba) -> dict:
        """
        Compute a SHAP-based explanation.

        Args:
            X (array-like of shape (1, n_features) or (n_samples, n_features)):
                Input(s) to explain.
            y_pred_proba (array-like): Prediction probabilities.

        Returns:
            dict: Explanation dictionary with SHAP values and metadata.
        """
        try:
            shap_values = self._explainer.shap_values(X)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classification returns list of [neg_class, pos_class]
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                sv = shap_values

            # Ensure sv is 1D array
            if hasattr(sv, "ndim"):
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
                    self._explainer.expected_value[1]
                    if isinstance(self._explainer.expected_value, (list, np.ndarray))
                    else self._explainer.expected_value
                ),
                "method": "SHAP",
            }
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}, falling back to permutation importance")
            return self._explain_permutation(X, y_pred_proba)

    def _explain_permutation(self, X, y_pred_proba) -> dict:
        """
        Explain using permutation importance.

        Args:
            X (array-like of shape (1, n_features) or (n_samples, n_features)):
                Input(s) to explain.
            y_pred_proba (array-like): Prediction probabilities.

        Returns:
            dict: Explanation dictionary with permutation importances.
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

    def _explain_simple(self, X, y_pred_proba) -> dict:
        """
        Simple explanation based on feature values relative to background.

        Args:
            X (array-like of shape (1, n_features) or (n_samples, n_features)):
                Input(s) to explain.
            y_pred_proba (array-like): Prediction probabilities.

        Returns:
            dict: Explanation dictionary with feature deviations.
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

    def _explain_feature_importance(self, X) -> dict:
        """
        Fallback explanation using built-in feature importances.

        Args:
            X (array-like): Input (not used; kept for API consistency).

        Returns:
            dict: Explanation dictionary with model feature importances.
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
            "feature_names": self.feature_names.tolist(),
            "top_risk_factors": paired[:10],
            "protective_factors": [],
            "base_value": 0.0,
        }