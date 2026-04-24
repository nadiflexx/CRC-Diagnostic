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

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

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

    Attempts to use ``shap.TreeExplainer`` for XGBoost models and falls
    back to ``shap.KernelExplainer`` for any other model type. If SHAP
    is not installed, feature importances from the underlying model are
    used instead.

    Provides both single-sample explanation (``explain``) and global
    visualisation plots (``plot_explanation``) including beeswarm and
    bar charts consistent with the clinical reporting style of the
    training pipeline.
    """

    def __init__(self, model, feature_names: list[str]):
        """
        Initialise the tabular explainer.

        Args:
            model: Trained ``TabularCancerModel`` instance with a
                ``predict_proba`` method and a ``model`` attribute
                exposing the underlying XGBoost classifier.
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
        Initialise the SHAP TreeExplainer using the fitted XGBoost model.

        Falls back to ``KernelExplainer`` with a random subsample of up
        to 100 background points if ``TreeExplainer`` fails. If SHAP is
        not installed the method logs a warning and returns without
        raising.

        Args:
            X_background (array-like of shape (n_samples, n_features)):
                Background dataset (training set recommended).
        """
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
            logger.warning("SHAP not installed – tabular explanations unavailable")

    def explain(self, X) -> dict:
        """
        Generate a feature-level explanation for the given input.

        Args:
            X (array-like of shape (1, n_features)): Single-sample input.

        Returns:
            dict: Explanation dictionary with keys ``shap_values``,
                ``feature_names``, ``top_risk_factors``,
                ``protective_factors``, ``base_value``.
        """
        if self._explainer is not None and self._fitted:
            return self._explain_shap(X)
        return self._explain_feature_importance(X)

    def plot_explanation(
        self,
        X,
        save_path: str | None = None,
        plot_type: str = "bar",
    ) -> "plt.Figure":
        """
        Render SHAP visualisations for the given input.

        When ``plot_type`` is ``"bar"`` a horizontal bar chart of the
        top-15 SHAP contributions is produced (positive = risk-increasing
        in red, negative = protective in green).

        When ``plot_type`` is ``"beeswarm"`` a global beeswarm summary
        plot is produced using the provided ``X`` as the evaluation set
        (recommended: pass ``X_test`` with multiple rows).

        Args:
            X (array-like): Input to explain. For ``"bar"`` pass a
                single row (1, n_features). For ``"beeswarm"`` pass the
                full test set.
            save_path (str | None): Optional file path where the PNG
                will be saved at 150 DPI. Default is ``None``.
            plot_type (str): One of ``"bar"`` or ``"beeswarm"``.
                Default is ``"bar"``.

        Returns:
            matplotlib.figure.Figure: The closed figure object.
        """
        import matplotlib.pyplot as plt

        if plot_type == "beeswarm":
            return self._plot_beeswarm(X, save_path)

        explanation = self.explain(X)
        sv = explanation["shap_values"]

        fig, ax = plt.subplots(figsize=(10, 8))
        sorted_idx = np.argsort(np.abs(sv))[-15:]
        colors = ["#D32F2F" if v > 0 else "#388E3C" for v in sv[sorted_idx]]
        names = [self.feature_names[i] for i in sorted_idx]

        ax.barh(range(len(sorted_idx)), sv[sorted_idx], color=colors)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Risk contribution (SHAP value)")
        ax.set_title("Patient Risk Factors — SHAP")
        ax.axvline(x=0, color="black", linewidth=0.5)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ─────────────────────────────────────────────
    #  Private helpers
    # ─────────────────────────────────────────────

    def _plot_beeswarm(self, X, save_path: str | None) -> "plt.Figure":
        """
        Produce a SHAP beeswarm summary plot over multiple samples.

        Uses up to 2 000 samples for performance. Saves two files when
        ``save_path`` is provided: the beeswarm PNG and a bar-importance
        PNG with ``_bar`` appended before the extension.

        Args:
            X (array-like of shape (n_samples, n_features)): Evaluation
                set, typically the test split.
            save_path (str | None): Base file path for PNG output.

        Returns:
            matplotlib.figure.Figure: Beeswarm figure (closed).
        """
        import matplotlib.pyplot as plt

        if self._explainer is None or not self._fitted:
            logger.warning("SHAP explainer not fitted — skipping beeswarm plot")
            return plt.figure()

        try:
            n = min(2000, len(X))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), size=n, replace=False)
            X_sub = pd.DataFrame(X[idx], columns=self.feature_names)

            shap_values = self._explainer.shap_values(X_sub)
            sv = shap_values if not isinstance(shap_values, list) else shap_values[1]

            fig_bee = plt.figure(figsize=(10, 7))
            shap.summary_plot(
                sv,
                X_sub,
                feature_names=self.feature_names,
                show=False,
            )
            plt.title("SHAP — Feature impact (beeswarm)")
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig_bee)

            fig_bar = plt.figure(figsize=(8, 6))
            shap.summary_plot(
                sv,
                X_sub,
                feature_names=self.feature_names,
                plot_type="bar",
                show=False,
            )
            plt.title("SHAP — Mean |SHAP| importance")
            plt.tight_layout()
            if save_path:
                from pathlib import Path as _Path

                p = _Path(save_path)
                bpath = p.parent / (p.stem + "_bar" + p.suffix)
                plt.savefig(str(bpath), dpi=150, bbox_inches="tight")
            plt.close(fig_bar)

            return fig_bee

        except Exception as exc:
            logger.warning(f"Beeswarm plot failed: {exc}")
            return plt.figure()

    def _explain_shap(self, X) -> dict:
        """
        Compute a SHAP-based explanation for a single sample.

        Args:
            X (array-like of shape (1, n_features)): Input to explain.

        Returns:
            dict: Keys ``shap_values``, ``feature_names``,
                ``top_risk_factors``, ``protective_factors``,
                ``base_value``.
        """
        shap_values = self._explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values
        if sv.ndim > 1:
            sv = sv[0]

        paired = sorted(
            zip(self.feature_names, sv.tolist(), strict=True),
            key=lambda p: abs(p[1]),
            reverse=True,
        )
        expected = self._explainer.expected_value
        base_value = float(
            expected[1] if isinstance(expected, (list, np.ndarray)) else expected
        )
        return {
            "shap_values": np.array(sv),
            "feature_names": self.feature_names,
            "top_risk_factors": [(n, v) for n, v in paired if v > 0],
            "protective_factors": [(n, v) for n, v in paired if v < 0],
            "base_value": base_value,
        }

    def _explain_feature_importance(self, X) -> dict:
        """
        Fallback explanation using built-in XGBoost feature importances.

        Args:
            X: Not used; kept for API consistency.

        Returns:
            dict: Same keys as ``_explain_shap`` with ``base_value=0.0``.
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
