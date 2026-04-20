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
import numpy as np
import pandas as pd
import torch

from src.config.constants import CLINICAL_DEFAULTS, DEFAULT_CLASS_NAMES
from src.config.logger import log as logger
from src.config.paths import paths
from src.config.settings import model as model_cfg
from src.data.processing.image_preprocessor import get_val_transforms
from src.data.processing.tabular_preprocessor import TabularPreprocessor
from src.evaluation.explainability import ImageExplainer
from src.evaluation.gradcam import CamMethod
from src.models.ensemble_predictor import EnsemblePredictor
from src.models.polyp_segmenter import ColonPolypSegmenter
from src.models.tabular_model import TabularCancerModel
from src.utils.ml_utils import softmax_np
from src.utils.onnx_session import ONNXSession

ONNX_DIR = paths.MODELS / "onnx"

_API_TO_MODEL_FEATURES = {
    "age_value": "Age",
    "smoking_history": "Smoking_History",
    "cea_level_ng_ml": "CEA_Level_ng_mL",
    "hemoglobin_g_dl": "Hemoglobin_g_dL",
    "pyrad_adc_mean": "PyRad_ADC_Mean",
    "pyrad_adc_std": "PyRad_ADC_Std",
    "pyrad_entropy": "PyRad_Entropy",
    "pyrad_glcm_contrast": "PyRad_GLCM_Contrast",
    "pyrad_glcm_homogeneity": "PyRad_GLCM_Homogeneity",
    "pyrad_shape_sphericity": "PyRad_Shape_Sphericity",
    "pyrad_firstorder_skewness": "PyRad_FirstOrder_Skewness",
}


def _build_model_row(patient_data: dict) -> dict:
    """
    Build a single-row dict with the 11 model feature names from the
    raw patient_data dict received by the engine.

    Falls back to ``CLINICAL_DEFAULTS`` for any missing feature so that
    the preprocessor always receives a complete row.

    Args:
        patient_data (dict): Raw dict from ``routes.py``, containing
            both legacy fields and new ``ClinicalDataIn`` fields.

    Returns:
        dict: Dict with exactly the 11 keys expected by
            ``TabularPreprocessor`` (``CLINICAL_NUMERIC_FEATURES``).
    """
    row = {}
    for api_key, model_col in _API_TO_MODEL_FEATURES.items():
        value = patient_data.get(api_key)
        if value is None:
            value = CLINICAL_DEFAULTS.get(model_col, 0.0)
        row[model_col] = float(value)
    return row


class DiagnosisEngine:
    """
    Multimodal diagnosis engine with adaptive ensemble inference.

    Orchestrates three independent pipelines:
        - **Image**: endoscopic image classification + Grad-CAM +
          polyp segmentation.
        - **Tabular**: clinical risk prediction from structured patient
          data using the new CRC synthetic model (11 features).
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

        Args:
            cam_method (CamMethod): Grad-CAM algorithm to use for
                visualisation. Default is ``"gradcam++"``.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_loaded = False
        self.cam_method: CamMethod = cam_method

        self.ensemble = EnsemblePredictor(device=self.device)
        self.image_segmenter = None

        # Tabular — new model + preprocessor objects
        self.tabular_model: TabularCancerModel | None = None
        self.tabular_preprocessor: TabularPreprocessor | None = None

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

        self._image_explainer: ImageExplainer | None = None

    # ═══════════════════════════════════════════════
    #  MODEL LOADING
    # ═══════════════════════════════════════════════

    def load_models(self):
        """
        Load all models required by the diagnosis engine.

        For each component, ONNX Runtime is attempted first and PyTorch
        is used as a fallback. Sets ``self.models_loaded = True`` on
        completion regardless of which individual components succeeded.
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
        """Load image classification backend (ONNX → PyTorch fallback)."""
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
        """Attempt to load PyTorch alongside ONNX for Grad-CAM (hybrid mode)."""
        try:
            self.ensemble.load_models()
            logger.info("  ✅ PyTorch loaded for Grad-CAM (hybrid mode)")
        except Exception as e:
            logger.warning(f"  ⚠️  PyTorch Grad-CAM not available: {e}")

    def _load_onnx_classifier_meta(self):
        """Load class names from the ONNX classifier metadata JSON."""
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
        """Load polyp segmentation backend (ONNX → PyTorch fallback)."""
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
        Load the clinical risk tabular model and preprocessor.

        Tries ONNX-ML first. Falls back to loading the joblib-serialised
        ``TabularCancerModel`` + ``TabularPreprocessor`` from the paths
        defined in ``paths``. If neither is available, tabular analysis
        is disabled.
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
                self.tabular_model = TabularCancerModel.load(tab_path)
                self.tabular_preprocessor = TabularPreprocessor.load(prep_path)
                self._use_onnx_tabular = False
                logger.info("  ✅ Tabular: TabularCancerModel + TabularPreprocessor")
            except Exception as e:
                logger.error(f"  ❌ Tabular joblib failed: {e}")
        else:
            logger.warning("  ⚠️ Tabular model not found — tabular analysis disabled")

    def _load_onnx_tabular_meta(self):
        """Load threshold and feature names from tabular ONNX metadata JSON."""
        meta_path = ONNX_DIR / "tabular_model.json"
        if not meta_path.exists():
            return
        with open(meta_path) as f:
            meta = json.load(f)
        self._tabular_threshold = float(meta.get("best_threshold", 0.5))
        self._tabular_feature_names = meta.get("feature_names", [])
        logger.info(
            f"  📄 Tabular ONNX: {len(self._tabular_feature_names)} features, "
            f"threshold={self._tabular_threshold:.4f}"
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

        Fusion logic:
            - ``"polyp"``        → ``final_score = max(0.85, image_prob)``.
            - ``"inflammation"`` → weighted mix (70% image, 30% tabular).
            - ``"normal"``       → tabular score scaled by 0.4.
            - Image unavailable  → ``final_score = tabular_prob``.

        Args:
            patient_data (dict): Clinical variables for the patient.
                Must contain the new API field names (``age_value``,
                ``cea_level_ng_ml``, etc.) for tabular analysis.
            image_path (str | None): Path to the colonoscopy image.
            patient_id (int | None): Optional patient identifier.

        Returns:
            dict: Diagnosis result dictionary.
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

        # Image pipeline
        image_prob = None
        img_class = ""
        if image_path and Path(image_path).exists():
            img_res = self._analyze_image(image_path)
            if img_res:
                result["image_analysis"] = img_res
                image_prob = img_res["prediction_score"]
                img_class = img_res["prediction_class"]

        # Tabular pipeline
        tabular_prob = None
        if patient_data and (self.tabular_model is not None or self._use_onnx_tabular):
            tab_res = self._analyze_tabular(patient_data)
            if tab_res:
                result["tabular_analysis"] = tab_res
                tabular_prob = tab_res["prediction_score"]

        # Fusion
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
    #  IMAGE ANALYSIS  (unchanged)
    # ═══════════════════════════════════════════════

    def _analyze_image(self, image_path: str) -> dict | None:
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
            f"α={alpha_used:.2f}, β={beta_used:.2f} → {pred_cls} ({score:.1%})"
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
        pred_class,
        score,
        probs,
        probs_a,
        polyp_detected,
        report_url,
        lesion_count,
        ensemble_used,
        ensemble_mode,
        attention_ratio,
        alpha_used,
        beta_used,
        gradcam_data,
    ) -> dict:
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
    #  GRAD-CAM  (unchanged)
    # ═══════════════════════════════════════════════

    def _generate_gradcam(self, image_bgr, pred_idx, alpha=0.5, beta=0.5) -> dict:
        if self._image_explainer is None or self.ensemble.model_a is None:
            return {"gradcam_a": None, "gradcam_b": None, "gradcam_fusion": None}
        return self._image_explainer.generate_gradcam(image_bgr, pred_idx, alpha, beta)

    # ═══════════════════════════════════════════════
    #  SEGMENTATION  (unchanged)
    # ═══════════════════════════════════════════════

    def _run_segmentation(self, original) -> tuple[bool, str | None, int]:
        if self._use_onnx_segmenter:
            return self._run_segmentation_onnx(original)
        if self.image_segmenter is not None:
            return self._run_segmentation_pytorch(original)
        return False, None, 0

    def _run_segmentation_onnx(self, original) -> tuple[bool, str | None, int]:
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

    def _run_segmentation_pytorch(self, original) -> tuple[bool, str | None, int]:
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
                return False, None, 0
            mask_np = masks[0, 0].cpu().numpy()
            if mask_np.sum() <= 20:
                return False, None, 0
            return self._postprocess_and_save_mask(mask_np, preprocessed)
        except Exception as e:
            logger.error(f"PyTorch segmentation failed: {e}\n{traceback.format_exc()}")
            return False, None, 0

    def _postprocess_and_save_mask(
        self, mask_np, preprocessed
    ) -> tuple[bool, str | None, int]:
        if self._image_explainer is None:
            return False, None, 0
        return self._image_explainer.postprocess_and_save_mask(mask_np, preprocessed)

    # ═══════════════════════════════════════════════
    #  TABULAR ANALYSIS  (new logic)
    # ═══════════════════════════════════════════════

    def _analyze_tabular(self, patient_data: dict) -> dict | None:
        """
        Route tabular inference to the active backend.

        Args:
            patient_data (dict): Raw patient dict from the route,
                containing new API field names.

        Returns:
            dict | None: Tabular analysis result or ``None`` on failure.
        """
        try:
            return (
                self._analyze_tabular_onnx(patient_data)
                if self._use_onnx_tabular
                else self._analyze_tabular_joblib(patient_data)
            )
        except Exception as e:
            logger.error(f"Tabular analysis error: {e}\n{traceback.format_exc()}")
            return None

    def _analyze_tabular_onnx(self, patient_data: dict) -> dict | None:
        """
        Run tabular risk inference with ONNX-ML Runtime.

        Builds the model feature row from the API field names, runs the
        ONNX session, and returns the probability of the positive class.

        Args:
            patient_data (dict): Raw patient dict.

        Returns:
            dict | None: Result with ``prediction_score``,
                ``high_risk``, and ``top_risk_factors``.
        """
        if not self._tabular_feature_names:
            logger.warning("No ONNX feature names → fallback to joblib tabular")
            return self._analyze_tabular_joblib(patient_data)

        row = _build_model_row(patient_data)
        X = np.array([[row[f] for f in self._tabular_feature_names]], dtype=np.float32)
        outputs = self._onnx_tabular.run_all(X)

        if len(outputs) > 1:
            proba_output = outputs[1]
            prob = float(
                proba_output[0] if proba_output.ndim == 1 else proba_output[0, 1]
            )
        else:
            prob = 0.75 if int(outputs[0][0]) == 1 else 0.25

        high_risk = prob >= self._tabular_threshold
        factors = self._extract_risk_factors(patient_data)
        logger.info(f"  [ONNX-ML tabular] prob={prob:.3f}, high_risk={high_risk}")
        return {
            "prediction_score": prob,
            "high_risk": high_risk,
            "top_risk_factors": factors[:5],
        }

    def _analyze_tabular_joblib(self, patient_data: dict) -> dict | None:
        """
        Run tabular risk inference with the joblib ``TabularCancerModel``.

        Builds the 11-feature row from ``patient_data``, applies the
        fitted ``TabularPreprocessor``, and calls
        ``TabularCancerModel.predict_proba``.

        Args:
            patient_data (dict): Raw patient dict containing new API
                field names (``age_value``, ``cea_level_ng_ml``, etc.).

        Returns:
            dict | None: Result with ``prediction_score``,
                ``high_risk``, and ``top_risk_factors``, or ``None``
                if the model is not loaded.
        """
        if self.tabular_model is None or self.tabular_preprocessor is None:
            logger.warning("Tabular model not loaded — skipping tabular analysis")
            return None

        # Build model row with correct column names
        row = _build_model_row(patient_data)
        df = pd.DataFrame([row])

        # Ensure all expected feature columns are present
        for feat in self.tabular_preprocessor.feature_names:
            if feat not in df.columns:
                df[feat] = CLINICAL_DEFAULTS.get(feat, 0.0)

        X = self.tabular_preprocessor.transform(df)

        # predict_proba returns 1-D array of shape (n_samples,) — P(cancer)
        # TemperatureScaledModel clips output to [PROB_MIN, PROB_MAX] = [0.05, 0.95]
        prob_array = self.tabular_model.predict_proba(X)
        prob = float(prob_array[0])

        high_risk = prob >= self.tabular_model.best_threshold
        factors = self._extract_risk_factors(patient_data)

        logger.info(
            f"  [joblib tabular] prob={prob:.3f} "
            f"(calibrated via TemperatureScaling), "
            f"threshold={self.tabular_model.best_threshold:.2f}, "
            f"high_risk={high_risk}"
        )
        return {
            "prediction_score": prob,
            "high_risk": high_risk,
            "top_risk_factors": factors[:5],
        }

    def _extract_risk_factors(self, patient_data: dict) -> list[tuple[str, float]]:
        """
        Extract clinically relevant risk factors from patient data.

        Works with both old field names (legacy) and new field names
        (new model) by checking both.

        Args:
            patient_data (dict): Raw patient dict.

        Returns:
            list[tuple[str, float]]: Elevated risk factor tuples.
        """
        factors = []

        # CEA — check both old and new field names
        cea = patient_data.get("cea_level_ng_ml") or patient_data.get("cea", 0)
        if cea and float(cea) > 5.0:
            factors.append(("Elevated CEA", float(cea)))

        # Haemoglobin — check both field names
        hgb = patient_data.get("hemoglobin_g_dl") or patient_data.get("hemoglobin", 15)
        if hgb and float(hgb) < 12.0:
            factors.append(("Low Haemoglobin", float(hgb)))

        # Smoking history
        smoking = patient_data.get("smoking_history", 0)
        if smoking and int(smoking) == 1:
            factors.append(("Smoking History", 1.0))

        # Radiomic flags
        adc = patient_data.get("pyrad_adc_mean")
        if adc and float(adc) < 1300.0:
            factors.append(("Restricted ADC (tumour pattern)", float(adc)))

        entropy = patient_data.get("pyrad_entropy")
        if entropy and float(entropy) > 6.0:
            factors.append(("High Texture Entropy", float(entropy)))

        # Legacy screening tests
        if patient_data.get("fit_positive"):
            factors.append(("FIT Positive", 1.0))
        if patient_data.get("fobt_positive"):
            factors.append(("FOBT Positive", 1.0))

        return factors

    # ═══════════════════════════════════════════════
    #  BUSINESS UTILITIES  (unchanged)
    # ═══════════════════════════════════════════════

    def _get_final_diagnosis_text(self, img_class: str, score: float) -> str:
        if img_class == "polyp":
            return "Adenomatous Polyp Detected"
        if img_class == "inflammation":
            return "Signs of Ulcerative Colitis"
        if score >= 0.7:
            return "High Clinical Risk"
        return "Healthy Mucosa"

    def _get_risk_level(self, score: float) -> dict:
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
    Subsequent calls return the same already-loaded instance.

    Returns:
        DiagnosisEngine: Shared engine instance with all models loaded.
    """
    global _engine
    if _engine is None:
        _engine = DiagnosisEngine()
        _engine.load_models()
    return _engine
