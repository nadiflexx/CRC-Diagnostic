"""
src/diagnosis/engine.py

Multimodal diagnosis engine.
Manages: segmentation, tabular risk scoring, Grad-CAM visualisation,
and multimodal fusion.

Inference backend:
- ONNX Runtime (preferred): faster, no PyTorch dependency at runtime.
- PyTorch (fallback): original logic when ONNX models are not exported.
"""

from datetime import datetime
import json
from pathlib import Path
import traceback

import cv2
import joblib
import numpy as np
import pandas as pd
import torch

from src.config.constants import CLINICAL_DEFAULTS, DEFAULT_CLASS_NAMES
from src.config.logger import log as logger
from src.config.paths import paths
from src.config.settings import model as model_cfg
from src.data.processing.image_preprocessor import get_val_transforms
from src.evaluation.explainability import ImageExplainer
from src.evaluation.gradcam import CamMethod
from src.models.ensemble_predictor import EnsemblePredictor
from src.models.polyp_segmenter import ColonPolypSegmenter
from src.utils.ml_utils import encode_patient_row, softmax_np
from src.utils.onnx_session import ONNXSession

ONNX_DIR = paths.MODELS / "onnx"


class DiagnosisEngine:
    """
    Multimodal diagnosis engine with adaptive ensemble inference.

    Orchestrates three independent pipelines:
        - **Image**: endoscopic image classification + Grad-CAM +
          polyp segmentation.
        - **Tabular**: clinical risk prediction from structured patient
          data.
        - **Fusion**: heuristic combination of image and tabular signals
          into a final risk score.

    Each pipeline first attempts ONNX Runtime inference and falls back
    to PyTorch if no exported ONNX model is available or the session
    fails. Errors are logged and the affected component is disabled
    gracefully so that the engine can continue with the remaining
    pipelines.
    """

    def __init__(self, cam_method: CamMethod = "gradcam++"):
        """
        Initialise the diagnosis engine with ONNX session handles.

        ONNX sessions are created lazily; the underlying runtime is not
        initialised until ``load_models`` is called.

        Args:
            cam_method (CamMethod): Grad-CAM algorithm to use for
                visualisation. One of ``"gradcam"``, ``"gradcam++"``,
                ``"hirescam"``, or ``"scorecam"``. Default is
                ``"gradcam++"``.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_loaded = False
        self.cam_method: CamMethod = cam_method

        self.ensemble = EnsemblePredictor(device=self.device)
        self.image_segmenter = None
        self.tabular_model = None
        self.tabular_preprocessor = None

        self._onnx_classifier = ONNXSession(
            ONNX_DIR / "best_classifier.onnx", self.device
        )
        self._onnx_tissue = ONNXSession(
            ONNX_DIR / "best_tissue_classifier.onnx", self.device
        )
        self._onnx_segmenter = ONNXSession(
            ONNX_DIR / "best_segmenter.onnx", self.device
        )
        self._onnx_tabular = ONNXSession(ONNX_DIR / "tabular_model.onnx", self.device)

        self._use_onnx_classifier = False
        self._use_onnx_segmenter = False
        self._use_onnx_tabular = False

        self.img_transform = get_val_transforms(model_cfg.IMAGE_SIZE)

        self.class_names: dict[int, str] = {}
        self.clinical_info: dict = {}
        self.num_classes = 3

        self._tabular_threshold: float = 0.5
        self._tabular_feature_names: list[str] = []

        # Initialised in load_models() once ensemble and class_names
        # are available.
        self._image_explainer: ImageExplainer | None = None

    # ═══════════════════════════════════════════════
    #  MODEL LOADING
    # ═══════════════════════════════════════════════

    def load_models(self):
        """
        Load all models required by the diagnosis engine.

        For each component, ONNX Runtime is attempted first and PyTorch
        is used as a fallback. Errors are caught and logged; a failed
        component leaves the corresponding flag or attribute as its
        default (disabled) value so that the remaining components can
        still function.

        Sets ``self.models_loaded = True`` on completion regardless of
        which individual components succeeded.
        """
        logger.info("Loading diagnosis engine models…")
        self._load_classifier()
        self._load_segmenter()
        self._load_tabular()

        self._image_explainer = ImageExplainer(
            ensemble=self.ensemble,
            img_transform=self.img_transform,
            class_names=self.class_names,
            device=self.device,
            cam_method=self.cam_method,
        )

        self.models_loaded = True
        backend = "ONNX" if self._use_onnx_classifier else "PyTorch"
        logger.info(f"✅ Engine ready (backend={backend}, CAM={self.cam_method})")

    def _load_classifier(self):
        """
        Load the image classification backend.

        Tries ONNX (classifier + tissue model). On success, also loads
        the PyTorch ``EnsemblePredictor`` in parallel so that Grad-CAM
        remains available (hybrid mode). If ONNX fails, falls back to
        ``EnsemblePredictor`` PyTorch only.
        """
        if self._onnx_classifier.available:
            try:
                self._onnx_classifier._get()
                self._load_onnx_classifier_meta()

                if self._onnx_tissue.available:
                    self._onnx_tissue._get()
                    logger.info("  ✅ ONNX tissue classifier ready")

                self._use_onnx_classifier = True
                logger.info("  ✅ Classification: ONNX Runtime")
                self._try_load_pytorch_for_gradcam()
                return
            except Exception as e:
                logger.warning(f"  ⚠️  ONNX classifier failed ({e}), using PyTorch")

        self.ensemble.load_models()
        self.class_names = self.ensemble.class_names or DEFAULT_CLASS_NAMES
        self.num_classes = self.ensemble.num_classes
        self.clinical_info = self.ensemble.class_mapping.get("clinical_info", {})
        self._use_onnx_classifier = False
        logger.info("  ✅ Classification: PyTorch EnsemblePredictor")

    def _try_load_pytorch_for_gradcam(self):
        """
        Attempt to load the PyTorch ``EnsemblePredictor`` alongside
        ONNX.

        When successful, Grad-CAM generation is available even though
        forward passes are routed through ONNX Runtime (hybrid mode).
        Failure is non-blocking: a warning is logged and the engine
        continues without Grad-CAM support.
        """
        try:
            self.ensemble.load_models()
            logger.info("  ✅ PyTorch loaded for Grad-CAM (hybrid mode)")
        except Exception as e:
            logger.warning(f"  ⚠️  PyTorch Grad-CAM not available: {e}")

    def _load_onnx_classifier_meta(self):
        """
        Load class names and ``num_classes`` from the ONNX classifier
        metadata JSON produced by ``export_onnx.py``.

        Reads ``ONNX_DIR / best_classifier.json``. If the file does
        not exist, ``DEFAULT_CLASS_NAMES`` is used as a fallback and a
        warning is logged.
        """
        meta_path = ONNX_DIR / "best_classifier.json"
        if not meta_path.exists():
            self.class_names = DEFAULT_CLASS_NAMES
            logger.warning("  ⚠️  No ONNX metadata found, using default class names")
            return

        with open(meta_path) as f:
            meta = json.load(f)

        class_mapping = meta.get("class_mapping", {})
        self.num_classes = meta.get("num_classes", 3)
        self.class_names = {
            int(k): v for k, v in class_mapping.get("classes", {}).items()
        }
        self.clinical_info = class_mapping.get("clinical_info", {})
        logger.info(f"  📄 ONNX classes: {self.class_names}")

    def _load_segmenter(self):
        """
        Load the polyp segmentation backend.

        Tries the ONNX segmenter first. Falls back to
        ``ColonPolypSegmenter`` PyTorch loaded from
        ``paths.SEGMENTER_CHECKPOINT``. If neither is available,
        segmentation is disabled (``self.image_segmenter`` remains
        ``None``).
        """
        if self._onnx_segmenter.available:
            try:
                self._onnx_segmenter._get()
                self._use_onnx_segmenter = True
                logger.info("  ✅ Segmentation: ONNX Runtime")
                return
            except Exception as e:
                logger.warning(f"  ⚠️  ONNX segmenter failed ({e}), using PyTorch")

        seg_path = paths.SEGMENTER_CHECKPOINT
        if not seg_path.exists():
            logger.warning(f"  ⚠️ Segmenter not found: {seg_path}")
            return

        try:
            self.image_segmenter = ColonPolypSegmenter().to(self.device)
            ckpt = torch.load(seg_path, map_location=self.device, weights_only=False)
            self.image_segmenter.load_state_dict(ckpt["model_state_dict"])
            self.image_segmenter.eval()
            self._use_onnx_segmenter = False
            logger.info("  ✅ Segmentation: PyTorch")
        except Exception as e:
            logger.error(f"  ❌ PyTorch segmenter failed: {e}")

    def _load_tabular(self):
        """
        Load the clinical risk tabular model backend.

        Tries ONNX-ML first (exported XGBoost). Falls back to loading
        the joblib-serialised ``TabularCancerModel`` and its
        preprocessor from ``paths.TABULAR_MODEL_PATH`` and
        ``paths.TABULAR_PREPROCESSOR_PATH``. If neither is available,
        tabular analysis is disabled.
        """
        if self._onnx_tabular.available:
            try:
                self._onnx_tabular._get()
                self._load_onnx_tabular_meta()
                self._use_onnx_tabular = True
                logger.info("  ✅ Tabular: ONNX-ML Runtime")
                return
            except Exception as e:
                logger.warning(f"  ⚠️  ONNX tabular failed ({e}), using joblib")

        tab_path = paths.TABULAR_MODEL_PATH
        prep_path = paths.TABULAR_PREPROCESSOR_PATH
        if tab_path.exists() and prep_path.exists():
            try:
                self.tabular_model = joblib.load(tab_path)
                self.tabular_preprocessor = joblib.load(prep_path)
                self._use_onnx_tabular = False
                logger.info("  ✅ Tabular: joblib")
            except Exception as e:
                logger.error(f"  ❌ Tabular joblib failed: {e}")
        else:
            logger.warning("  ⚠️ Tabular model not found")

    def _load_onnx_tabular_meta(self):
        """
        Load the decision threshold and feature names from the tabular
        ONNX model metadata JSON.

        Reads ``ONNX_DIR / tabular_model.json``. The loaded values are
        stored in ``self._tabular_threshold`` and
        ``self._tabular_feature_names`` for use in
        ``_analyze_tabular_onnx``.
        """
        meta_path = ONNX_DIR / "tabular_model.json"
        if not meta_path.exists():
            return

        with open(meta_path) as f:
            meta = json.load(f)

        self._tabular_threshold = float(meta.get("best_threshold", 0.5))
        self._tabular_feature_names = meta.get("feature_names", [])
        logger.info(
            f"  📄 Tabular ONNX: {len(self._tabular_feature_names)} "
            f"features, threshold={self._tabular_threshold:.4f}"
        )

    # ═══════════════════════════════════════════════
    #  MAIN DIAGNOSIS
    # ═══════════════════════════════════════════════

    def diagnose(
        self,
        patient_data: dict,
        image_path: str | None = None,
        patient_id: int | None = None,
    ) -> dict:
        """
        Execute the full multimodal diagnosis pipeline.

        Combines endoscopic image analysis and clinical tabular data
        into a fused risk score using heuristic rules. Each modality is
        optional; if both are present the fusion weights depend on the
        predicted image class.

        Fusion logic:
            - ``"polyp"`` → ``final_score = max(0.85, image_prob)``.
            - ``"inflammation"`` → weighted mix (70 % image,
              30 % tabular).
            - ``"normal"`` → tabular score scaled by 0.4.
            - Image unavailable → ``final_score = tabular_prob``.

        Args:
            patient_data (dict): Clinical variables for the patient.
            image_path (str | None): Path to the colonoscopy image. If
                ``None`` or the file does not exist, image analysis is
                skipped. Default is ``None``.
            patient_id (int | None): Optional patient identifier
                included for traceability. Not used in computation.
                Default is ``None``.

        Returns:
            dict: Diagnosis result with keys ``"timestamp"``,
                ``"image_analysis"``, ``"tabular_analysis"``,
                ``"multimodal_result"``, ``"final_diagnosis"``,
                ``"confidence"``, ``"risk_level"``, and
                ``"recommendations"``.
        """
        if not self.models_loaded:
            self.load_models()

        result: dict = {
            "timestamp": datetime.utcnow().isoformat(),
            "image_analysis": None,
            "tabular_analysis": None,
            "multimodal_result": None,
            "final_diagnosis": "Normal",
            "confidence": 0.0,
            "risk_level": {
                "level": "LOW",
                "score": 0.0,
                "color": "#388E3C",
                "percentage": "0%",
            },
            "recommendations": [],
        }

        image_prob = None
        img_class = ""
        if image_path and Path(image_path).exists():
            img_res = self._analyze_image(image_path)
            if img_res:
                result["image_analysis"] = img_res
                image_prob = img_res["prediction_score"]
                img_class = img_res["prediction_class"]

        tabular_prob = None
        if patient_data and (self.tabular_model or self._use_onnx_tabular):
            tab_res = self._analyze_tabular(patient_data)
            if tab_res:
                result["tabular_analysis"] = tab_res
                tabular_prob = tab_res["prediction_score"]

        final_score = tabular_prob or 0.0
        if image_prob is not None:
            if img_class == "polyp":
                final_score = max(0.85, image_prob)
            elif img_class == "inflammation":
                final_score = (image_prob * 0.7) + ((tabular_prob or 0) * 0.3)
            else:
                final_score = (tabular_prob or 0.0) * 0.4
            result["multimodal_result"] = {"combined_score": float(final_score)}

        result["final_diagnosis"] = self._get_final_diagnosis_text(
            img_class, final_score
        )
        result["risk_level"] = self._get_risk_level(final_score)
        result["confidence"] = float(final_score)
        result["recommendations"] = self._get_recommendations(final_score, img_class)

        return result

    # ═══════════════════════════════════════════════
    #  IMAGE ANALYSIS
    # ═══════════════════════════════════════════════

    def _analyze_image(self, image_path: str) -> dict | None:
        """
        Route image analysis to the active backend.

        Reads the image from disk, selects ONNX or PyTorch inference
        based on ``self._use_onnx_classifier``, and returns the result.

        Args:
            image_path (str): Absolute path to the colonoscopy image.

        Returns:
            dict | None: Image analysis result dictionary, or ``None``
                if the image cannot be read or an unhandled exception
                occurs.
        """
        try:
            original = cv2.imread(image_path)
            if original is None:
                logger.error(f"Could not read image: {image_path}")
                return None
            return (
                self._analyze_image_onnx(original)
                if self._use_onnx_classifier
                else self._analyze_image_pytorch(original)
            )
        except Exception as e:
            logger.error(f"Image analysis error: {e}\n{traceback.format_exc()}")
            return None

    def _analyze_image_onnx(self, original: np.ndarray) -> dict | None:
        """
        Run image classification with ONNX Runtime plus adaptive
        ensemble logic from ``EnsemblePredictor``.

        Forward passes (logits) for both Model A and Model B are
        executed via ONNX for speed. The attention ratio and dynamic
        weight computation are delegated to ``EnsemblePredictor``
        exactly as in PyTorch mode, ensuring identical decision logic.

        Args:
            original (np.ndarray): BGR image array of shape (H, W, 3).

        Returns:
            dict | None: Image result dictionary as constructed by
                ``_build_image_result``, or ``None`` on failure.
        """
        preprocessed = self.ensemble.preprocessor_a.process_image(original)
        img_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
        augmented = self.img_transform(image=img_rgb)
        tensor_np = augmented["image"].numpy()[np.newaxis, ...]

        logits_a = self._onnx_classifier.run(tensor_np)
        probs_a = softmax_np(logits_a[0])

        attention_ratio = None
        if self.ensemble.use_adaptive and self._onnx_tissue.available:
            if self.ensemble.model_a is not None:
                tensor_pt = augmented["image"].unsqueeze(0).to(self.device)
                attention_ratio = self.ensemble.compute_attention_ratio(
                    tensor_pt, preprocessed
                )
            else:
                attention_ratio = 1.5

        if self._onnx_tissue.available:
            crops_bgr = self.ensemble.preprocessor_b.process_image(original)
            crop_probs = []
            for crop in crops_bgr:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_np = self.img_transform(image=crop_rgb)["image"].numpy()[
                    np.newaxis, ...
                ]
                logits_b = self._onnx_tissue.run(crop_np)
                crop_probs.append(softmax_np(logits_b[0]))
            probs_b = np.mean(crop_probs, axis=0)
        else:
            probs_b = probs_a

        probs, alpha_used, beta_used = self.ensemble.combine_predictions(
            probs_a, probs_b, attention_ratio
        )

        pred_idx = int(np.argmax(probs))
        pred_cls = self.class_names.get(pred_idx, str(pred_idx))
        score = float(probs[pred_idx])

        logger.info(
            f"  [ONNX] attention={attention_ratio}, "
            f"α={alpha_used:.2f}, β={beta_used:.2f} "
            f"→ {pred_cls} ({score:.1%})"
        )

        gradcam_data = self._generate_gradcam(original, pred_idx, alpha_used, beta_used)

        polyp_detected, report_url, lesion_count = False, None, 0
        if pred_cls == "polyp":
            polyp_detected, report_url, lesion_count = self._run_segmentation(original)

        return self._build_image_result(
            pred_cls,
            score,
            probs,
            probs_a,
            polyp_detected,
            report_url,
            lesion_count,
            ensemble_used=self._onnx_tissue.available,
            ensemble_mode="onnx",
            attention_ratio=attention_ratio,
            alpha_used=alpha_used,
            beta_used=beta_used,
            gradcam_data=gradcam_data,
        )

    def _analyze_image_pytorch(self, original: np.ndarray) -> dict | None:
        """
        Run image classification with the full PyTorch
        ``EnsemblePredictor``.

        Delegates the complete forward pass, attention ratio
        computation, and probability combination to
        ``self.ensemble.predict``.

        Args:
            original (np.ndarray): BGR image array of shape (H, W, 3).

        Returns:
            dict | None: Image result dictionary as constructed by
                ``_build_image_result``, or ``None`` on failure.
        """
        prediction = self.ensemble.predict(original)
        pred_idx = prediction["class_idx"]
        pred_class = prediction["class_name"]
        pred_score = prediction["confidence"]
        probs = np.array(prediction["probabilities"])
        probs_a = np.array(prediction["model_a_probs"])
        alpha_used = prediction["alpha_used"]
        beta_used = prediction["beta_used"]
        attention_ratio = prediction["attention_ratio"]

        logger.info(f"  [PyTorch] → {pred_class} ({pred_score:.1%})")

        gradcam_data = self._generate_gradcam(original, pred_idx, alpha_used, beta_used)

        polyp_detected, report_url, lesion_count = False, None, 0
        if pred_class == "polyp" and (self.image_segmenter or self._use_onnx_segmenter):
            polyp_detected, report_url, lesion_count = self._run_segmentation(original)

        return self._build_image_result(
            pred_class,
            pred_score,
            probs,
            probs_a,
            polyp_detected,
            report_url,
            lesion_count,
            ensemble_used=self.ensemble.ensemble_available,
            ensemble_mode=prediction["mode"],
            attention_ratio=attention_ratio,
            alpha_used=float(alpha_used),
            beta_used=float(beta_used),
            gradcam_data=gradcam_data,
        )

    def _build_image_result(
        self,
        pred_class: str,
        score: float,
        probs: np.ndarray,
        probs_a: np.ndarray,
        polyp_detected: bool,
        report_url: str | None,
        lesion_count: int,
        ensemble_used: bool,
        ensemble_mode: str,
        attention_ratio: float | None,
        alpha_used: float,
        beta_used: float,
        gradcam_data: dict,
    ) -> dict:
        """
        Construct the standardised image analysis result dictionary.

        Centralises result formatting so that both ONNX and PyTorch
        backends produce an identical output schema.

        Args:
            pred_class (str): Human-readable predicted class name.
            score (float): Confidence of the predicted class in [0, 1].
            probs (np.ndarray): Combined ensemble probability vector of
                shape (num_classes,).
            probs_a (np.ndarray): Model A probability vector.
            polyp_detected (bool): Whether the segmenter detected a
                polyp.
            report_url (str | None): URL of the saved segmentation
                overlay PNG, or ``None``.
            lesion_count (int): Number of polyp instances detected.
            ensemble_used (bool): Whether both models contributed.
            ensemble_mode (str): ``"onnx"``, ``"adaptive"``, or
                ``"fixed"``.
            attention_ratio (float | None): Computed attention ratio,
                or ``None`` if adaptive gating was not applied.
            alpha_used (float): Effective Model A weight.
            beta_used (float): Effective Model B weight.
            gradcam_data (dict): Dictionary with optional keys
                ``"gradcam_a"``, ``"gradcam_b"``, and
                ``"gradcam_fusion"``.

        Returns:
            dict: Fully populated image analysis result dictionary.
        """
        return {
            "prediction_class": pred_class,
            "prediction_score": score,
            "polyp_detected": polyp_detected,
            "lesion_count": lesion_count,
            "report_path": report_url,
            "probabilities": {
                self.class_names.get(i, f"class_{i}"): float(probs[i])
                for i in range(len(probs))
            },
            "ensemble_used": ensemble_used,
            "ensemble_mode": ensemble_mode,
            "attention_ratio": (
                round(attention_ratio, 3) if attention_ratio is not None else None
            ),
            "alpha_used": round(alpha_used, 3),
            "beta_used": round(beta_used, 3),
            "model_a_probs": {
                self.class_names.get(i, f"class_{i}"): float(probs_a[i])
                for i in range(len(probs_a))
            },
            "gradcam_a": gradcam_data.get("gradcam_a"),
            "gradcam_b": gradcam_data.get("gradcam_b"),
            "gradcam_fusion": gradcam_data.get("gradcam_fusion"),
        }

    # ═══════════════════════════════════════════════
    #  GRAD-CAM  (delegated to ImageExplainer)
    # ═══════════════════════════════════════════════

    def _generate_gradcam(
        self,
        image_bgr: np.ndarray,
        pred_idx: int,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> dict:
        """
        Delegate Grad-CAM generation to ``ImageExplainer``.

        Returns an empty result dict when the explainer is not yet
        initialised or ``model_a`` is unavailable.

        Args:
            image_bgr (np.ndarray): Original BGR image of shape
                (H, W, 3).
            pred_idx (int): Target class index for the explanation.
            alpha (float): Model A weight. Default is 0.5.
            beta (float): Model B weight. Default is 0.5.

        Returns:
            dict: Keys ``"gradcam_a"``, ``"gradcam_b"``,
                ``"gradcam_fusion"`` each holding a base64 PNG data
                URI or ``None``.
        """
        if self._image_explainer is None or self.ensemble.model_a is None:
            return {
                "gradcam_a": None,
                "gradcam_b": None,
                "gradcam_fusion": None,
            }
        return self._image_explainer.generate_gradcam(image_bgr, pred_idx, alpha, beta)

    # ═══════════════════════════════════════════════
    #  SEGMENTATION
    # ═══════════════════════════════════════════════

    def _run_segmentation(self, original: np.ndarray) -> tuple[bool, str | None, int]:
        """
        Route polyp segmentation to the active backend.

        Args:
            original (np.ndarray): BGR image array of shape (H, W, 3).

        Returns:
            tuple[bool, str | None, int]:
                - Whether at least one polyp instance was found.
                - URL of the saved overlay image, or ``None``.
                - Number of detected polyp instances.
        """
        if self._use_onnx_segmenter:
            return self._run_segmentation_onnx(original)
        if self.image_segmenter is not None:
            return self._run_segmentation_pytorch(original)
        return False, None, 0

    def _run_segmentation_onnx(
        self, original: np.ndarray
    ) -> tuple[bool, str | None, int]:
        """
        Run polyp segmentation with ONNX Runtime.

        Applies a sigmoid to the raw ONNX logits, thresholds at 0.5 to
        produce a binary mask, and delegates post-processing to
        ``ImageExplainer.postprocess_and_save_mask``.

        Args:
            original (np.ndarray): BGR image array of shape (H, W, 3).

        Returns:
            tuple[bool, str | None, int]: ``(polyp_detected,
                report_url, lesion_count)``. Returns
                ``(False, None, 0)`` on failure or if the mask contains
                fewer than 20 positive pixels.
        """
        try:
            preprocessed = self.ensemble.preprocessor_a.process_image(original)
            img_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
            tensor_np = self.img_transform(image=img_rgb)["image"].numpy()[
                np.newaxis, ...
            ]

            logits = self._onnx_segmenter.run(tensor_np)
            probs = 1.0 / (1.0 + np.exp(-logits))
            mask_np = (probs[0, 0] > 0.5).astype(np.float32)

            if mask_np.sum() <= 20:
                return False, None, 0

            return self._postprocess_and_save_mask(mask_np, preprocessed)
        except Exception as e:
            logger.error(f"ONNX segmentation failed: {e}")
            return False, None, 0

    def _run_segmentation_pytorch(
        self, original: np.ndarray
    ) -> tuple[bool, str | None, int]:
        """
        Run polyp segmentation with the PyTorch
        ``ColonPolypSegmenter``.

        Uses test-time augmentation (TTA) when the model exposes
        ``predict_mask_tta``; otherwise falls back to
        ``predict_mask``.

        Args:
            original (np.ndarray): BGR image array of shape (H, W, 3).

        Returns:
            tuple[bool, str | None, int]: ``(polyp_detected,
                report_url, lesion_count)``. Returns
                ``(False, None, 0)`` on failure or if the mask contains
                fewer than 20 positive pixels.
        """
        try:
            preprocessed = self.ensemble.preprocessor_a.process_image(original)
            img_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
            tensor = (
                self.img_transform(image=img_rgb)["image"].unsqueeze(0).to(self.device)
            )

            if self.image_segmenter is not None and hasattr(
                self.image_segmenter, "predict_mask_tta"
            ):
                masks = self.image_segmenter.predict_mask_tta(tensor, threshold=0.5)
            elif self.image_segmenter is not None:
                masks = self.image_segmenter.predict_mask(tensor, threshold=0.5)
            else:
                logger.warning("  ⚠️  Segmenter not available for PyTorch inference")
                return False, None, 0

            mask_np = masks[0, 0].cpu().numpy()
            if mask_np.sum() <= 20:
                return False, None, 0

            return self._postprocess_and_save_mask(mask_np, preprocessed)
        except Exception as e:
            logger.error(f"PyTorch segmentation failed: {e}\n{traceback.format_exc()}")
            return False, None, 0

    def _postprocess_and_save_mask(
        self,
        mask_np: np.ndarray,
        preprocessed: np.ndarray,
    ) -> tuple[bool, str | None, int]:
        """
        Delegate mask post-processing and overlay persistence to
        ``ImageExplainer``.

        Args:
            mask_np (np.ndarray): Binary float32 mask of shape (H, W)
                with values in {0.0, 1.0}.
            preprocessed (np.ndarray): BGR image of shape (H, W, 3)
                used as the drawing canvas.

        Returns:
            tuple[bool, str | None, int]: ``(polyp_detected,
                report_url, lesion_count)``. Returns
                ``(False, None, 0)`` when the explainer is not
                initialised.
        """
        if self._image_explainer is None:
            return False, None, 0
        return self._image_explainer.postprocess_and_save_mask(mask_np, preprocessed)

    # ═══════════════════════════════════════════════
    #  TABULAR ANALYSIS
    # ═══════════════════════════════════════════════

    def _analyze_tabular(self, patient_data: dict) -> dict | None:
        """
        Route tabular inference to the active backend.

        Args:
            patient_data (dict): Clinical variables for the patient.

        Returns:
            dict | None: Tabular analysis result with
                ``"prediction_score"``, ``"high_risk"``, and
                ``"top_risk_factors"``, or ``None`` if the analysis
                fails.
        """
        try:
            return (
                self._analyze_tabular_onnx(patient_data)
                if self._use_onnx_tabular
                else self._analyze_tabular_pytorch(patient_data)
            )
        except Exception as e:
            logger.error(f"Tabular analysis error: {e}")
            return None

    def _analyze_tabular_onnx(self, patient_data: dict) -> dict | None:
        """
        Run tabular risk inference with ONNX-ML Runtime.

        Uses ``encode_patient_row`` to convert the heterogeneous
        patient dictionary into a float32 feature vector in the correct
        column order. The ONNX session is expected to return either one
        output (predicted label) or two outputs (label + probability
        array).

        Args:
            patient_data (dict): Clinical variables for the patient.

        Returns:
            dict | None: Result with ``"prediction_score"``,
                ``"high_risk"``, and ``"top_risk_factors"``. Falls back
                to ``_analyze_tabular_pytorch`` if
                ``self._tabular_feature_names`` is empty.
        """
        if not self._tabular_feature_names:
            logger.warning("  ⚠️  No ONNX feature names → fallback to PyTorch tabular")
            return self._analyze_tabular_pytorch(patient_data)

        X = encode_patient_row(patient_data, self._tabular_feature_names)
        outputs = self._onnx_tabular.run_all(X)

        if len(outputs) > 1:
            proba_output = outputs[1]
            if proba_output.ndim == 1:
                prob = float(proba_output[0])
            elif proba_output.ndim == 2:
                prob = float(proba_output[0, 1])
            else:
                prob = 0.5
        else:
            label = int(outputs[0][0])
            prob = 0.75 if label == 1 else 0.25

        high_risk = prob >= self._tabular_threshold
        factors = self._extract_risk_factors(patient_data)
        logger.info(f"  [ONNX-ML] prob={prob:.3f}, high_risk={high_risk}")

        return {
            "prediction_score": prob,
            "high_risk": high_risk,
            "top_risk_factors": factors[:5],
        }

    def _analyze_tabular_pytorch(self, patient_data: dict) -> dict | None:
        """
        Run tabular risk inference with the joblib XGBoost model.

        Constructs a single-row DataFrame from ``patient_data``, fills
        in any missing columns with defaults from
        ``CLINICAL_DEFAULTS``, applies the preprocessor (if one was
        loaded), and calls ``predict_proba``.

        Args:
            patient_data (dict): Clinical variables for the patient.

        Returns:
            dict | None: Result with ``"prediction_score"``,
                ``"high_risk"``, and ``"top_risk_factors"``, or
                ``None`` if ``self.tabular_model`` is not loaded.
        """
        if self.tabular_model is None:
            return None

        df = pd.DataFrame([patient_data])
        expected_cols = (
            self.tabular_model.feature_names_in_
            if hasattr(self.tabular_model, "feature_names_in_")
            else []
        )
        for col in expected_cols:
            if col not in df.columns:
                df[col] = CLINICAL_DEFAULTS.get(col, 0)

        X = (
            df
            if isinstance(self.tabular_preprocessor, dict)
            else self.tabular_preprocessor.transform(df)
        )
        prob = (
            float(self.tabular_model.predict_proba(X)[0][1])
            if hasattr(self.tabular_model, "predict_proba")
            else 0.5
        )

        return {
            "prediction_score": prob,
            "high_risk": prob > 0.5,
            "top_risk_factors": self._extract_risk_factors(patient_data)[:5],
        }

    def _extract_risk_factors(self, patient_data: dict) -> list[tuple[str, float]]:
        """
        Extract clinically relevant risk factors from patient data.

        Applies simple threshold rules to identify elevated biomarkers
        and positive screening tests.

        Args:
            patient_data (dict): Clinical variables. Recognised keys:
                ``"cea"``, ``"fit_positive"``, ``"hemoglobin"``.

        Returns:
            list[tuple[str, float]]: List of ``(factor_name, value)``
                tuples for factors that exceed their respective
                thresholds.
        """
        factors = []
        if patient_data.get("cea", 0) > 5.0:
            factors.append(("Elevated CEA", float(patient_data["cea"])))
        if patient_data.get("fit_positive", False):
            factors.append(("FIT Positive", 1.0))
        if patient_data.get("hemoglobin", 15) < 12.0:
            factors.append(("Low Haemoglobin", float(patient_data["hemoglobin"])))
        return factors

    # ═══════════════════════════════════════════════
    #  BUSINESS UTILITIES
    # ═══════════════════════════════════════════════

    def _get_final_diagnosis_text(self, img_class: str, score: float) -> str:
        """
        Produce a human-readable diagnosis string.

        Args:
            img_class (str): Predicted image class.
            score (float): Fused risk score in [0, 1].

        Returns:
            str: Descriptive diagnosis text.
        """
        if img_class == "polyp":
            return "Adenomatous Polyp Detected"
        if img_class == "inflammation":
            return "Signs of Ulcerative Colitis"
        if score >= 0.7:
            return "High Clinical Risk"
        return "Healthy Mucosa"

    def _get_risk_level(self, score: float) -> dict:
        """
        Map a continuous risk score to a categorical risk level
        descriptor.

        Thresholds:
            - score ≥ 0.7 → ``"HIGH"`` (red).
            - 0.4 ≤ score < 0.7 → ``"MODERATE"`` (orange).
            - score < 0.4 → ``"LOW"`` (green).

        Args:
            score (float): Fused risk score in [0, 1].

        Returns:
            dict: Keys ``"level"``, ``"score"``, ``"color"``,
                ``"percentage"``.
        """
        if score >= 0.7:
            return {
                "level": "HIGH",
                "score": score,
                "color": "#D32F2F",
                "percentage": f"{score * 100:.1f}%",
            }
        if score >= 0.4:
            return {
                "level": "MODERATE",
                "score": score,
                "color": "#F57C00",
                "percentage": f"{score * 100:.1f}%",
            }
        return {
            "level": "LOW",
            "score": score,
            "color": "#388E3C",
            "percentage": f"{score * 100:.1f}%",
        }

    def _get_recommendations(self, score: float, img_class: str) -> list[str]:
        """
        Generate clinical action recommendations based on the
        diagnosis.

        Args:
            score (float): Fused risk score in [0, 1].
            img_class (str): Predicted image class.

        Returns:
            list[str]: Recommendation strings with urgency indicators.
        """
        if img_class == "polyp":
            return ["🔴 URGENT: Schedule polypectomy for polyp resection."]
        if img_class == "inflammation":
            return ["🟠 Evaluate anti-inflammatory treatment."]
        if score >= 0.7:
            return ["🔴 Urgent diagnostic colonoscopy (high blood-based risk)."]
        return ["🟢 Continue routine screening."]


# ═══════════════════════════════════════════════
#  SINGLETON
# ═══════════════════════════════════════════════

_engine: DiagnosisEngine | None = None


def get_diagnosis_engine() -> DiagnosisEngine:
    """
    Return the singleton ``DiagnosisEngine`` instance.

    Creates and fully initialises the engine on the first call.
    Subsequent calls return the same already-loaded instance, avoiding
    repeated model loading overhead.

    Returns:
        DiagnosisEngine: Shared engine instance with all models loaded.
    """
    global _engine
    if _engine is None:
        _engine = DiagnosisEngine()
        _engine.load_models()
    return _engine
