"""
src/diagnosis/engine.py

Motor de diagnóstico multimodal con Ensemble.
Carga Model A + Model B (si existe) y combina predicciones.
Si Model B no existe, funciona solo con Model A (backward compatible).

ACTUALIZADO: Usa pytorch-grad-cam (GradCAM++, HiResCAM, etc.)
"""

import base64
from datetime import datetime
import json
from pathlib import Path
import traceback

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.config.constants import CLINICAL_DEFAULTS, DEFAULT_CLASS_NAMES
from src.config.logger import log as logger
from src.config.paths import paths
from src.config.settings import model as model_cfg
from src.data.processing.image_preprocessor import get_val_transforms
from src.data.processing.standardizer import MultiSourceStandardizer
from src.data.processing.tissue_only_preprocessor import (
    TissueOnlyPreprocessor,
)
from src.evaluation.gradcam import (
    CamMethod,
    create_heatmap_overlay,
    generate_gradcam,
)
from src.models.image_classifier import ColonCancerClassifier
from src.models.polyp_segmenter import ColonPolypSegmenter
from src.models.tissue_classifier import (
    TissueOnlyClassifier,
)


class DiagnosisEngine:
    """Motor de diagnóstico multimodal con ensemble adaptativo."""

    def __init__(self, cam_method: CamMethod = "gradcam++"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_loaded = False
        self.cam_method: CamMethod = cam_method

        # Model A (context-aware)
        self.image_classifier = None
        self.temp_a = 1.0

        # Model B (tissue-only) — opcional
        self.tissue_classifier = None
        self.temp_b = 1.0
        self.ensemble_alpha = 1.0  # Default: solo Model A
        self.ensemble_beta = 0.0
        self.use_uncertainty = False
        self.ensemble_available = False

        # Otros modelos
        self.image_segmenter = None
        self.tabular_model = None
        self.tabular_preprocessor = None

        # Transforms
        self.img_transform = get_val_transforms(model_cfg.IMAGE_SIZE)

        # Config del modelo (se carga desde checkpoint)
        self.class_names = {}
        self.clinical_info = {}
        self.model_name = None
        self.num_classes = 3

        # Preprocesadores
        self.standardizer = MultiSourceStandardizer(target_size=model_cfg.IMAGE_SIZE)
        self.tissue_preprocessor = None  # Se carga lazy

    # ═════════════════════════════════════════════
    #  CARGA DE MODELOS
    # ═════════════════════════════════════════════

    def load_models(self):
        """Carga todos los modelos del motor."""
        logger.info("Cargando modelos del motor de diagnóstico…")

        self._load_model_a()
        self._load_model_b()
        self._load_ensemble_config()
        self._load_segmenter()
        self._load_tabular()

        self.models_loaded = True

        if self.ensemble_available:
            logger.info(
                f"✅ Motor listo (ENSEMBLE: α={self.ensemble_alpha:.2f}, "
                f"β={self.ensemble_beta:.2f}, CAM={self.cam_method})"
            )
        else:
            logger.info(f"✅ Motor listo (Model A solo, CAM={self.cam_method})")

    def _load_model_a(self):
        """Carga Model A (clasificador principal)."""
        clf_path = paths.CLASSIFIER_CHECKPOINT
        if not clf_path.exists():
            logger.warning(f"  ⚠️ Clasificador no encontrado: {clf_path}")
            self.class_names = DEFAULT_CLASS_NAMES
            return

        ckpt = torch.load(clf_path, map_location=self.device, weights_only=False)

        self.model_name = ckpt.get("model_name", "tf_efficientnetv2_s.in21k")
        self.num_classes = ckpt.get("num_classes", 3)
        self.temp_a = ckpt.get("temperature", 1.0)

        class_mapping = ckpt.get("class_mapping", {})
        self.class_names = {
            int(k): v for k, v in class_mapping.get("classes", {}).items()
        }
        self.clinical_info = class_mapping.get("clinical_info", {})

        logger.info(f"  Model A: {self.model_name}")
        logger.info(f"  Clases: {self.class_names}")
        logger.info(f"  Temperatura A: {self.temp_a:.2f}")
        logger.info(
            f"  F1: {ckpt.get('best_f1', 0):.4f} | Acc: {ckpt.get('best_acc', 0):.4f}"
        )

        self.image_classifier = ColonCancerClassifier(
            model_name=self.model_name,
            pretrained=False,
            num_classes=self.num_classes,
        ).to(self.device)

        self.image_classifier.load_state_dict(ckpt["model_state_dict"])
        self.image_classifier.eval()
        logger.info("  ✅ Model A cargado")

    def _load_model_b(self):
        """Carga Model B (tissue-only) si existe."""
        tissue_path = paths.TISSUE_CLASSIFIER_CHECKPOINT
        if not tissue_path.exists():
            logger.info("  ℹ️  Model B no encontrado → usando solo Model A")
            return

        try:
            ckpt = torch.load(
                tissue_path,
                map_location=self.device,
                weights_only=False,
            )

            model_name_b = ckpt.get("model_name", "tf_efficientnetv2_s.in21k")
            self.temp_b = ckpt.get("temperature", 1.0)

            self.tissue_classifier = TissueOnlyClassifier(
                model_name=model_name_b,
                pretrained=False,
                num_classes=self.num_classes,
            ).to(self.device)

            self.tissue_classifier.load_state_dict(ckpt["model_state_dict"])
            self.tissue_classifier.eval()

            logger.info(
                f"  ✅ Model B: {model_name_b} "
                f"(F1={ckpt.get('best_f1', 0):.4f}, "
                f"T={self.temp_b:.2f})"
            )
            self.ensemble_available = True

        except Exception as e:
            logger.warning(f"  ⚠️ Error cargando Model B: {e}")
            self.tissue_classifier = None

    def _load_ensemble_config(self):
        """Carga pesos del ensemble si existen."""
        config_path = paths.ENSEMBLE_CONFIG_PATH
        if not config_path.exists():
            if self.ensemble_available:
                self.ensemble_alpha = 0.5
                self.ensemble_beta = 0.5
                logger.info("  ℹ️  Sin ensemble_config.json → α=0.5, β=0.5")
            return

        try:
            with open(config_path) as f:
                config = json.load(f)

            self.ensemble_alpha = config.get("alpha", 0.5)
            self.ensemble_beta = config.get("beta", 0.5)
            self.use_uncertainty = config.get("use_uncertainty", False)

            logger.info(
                f"  📄 Ensemble: α={self.ensemble_alpha:.2f}, "
                f"β={self.ensemble_beta:.2f}, "
                f"uncertainty={self.use_uncertainty}"
            )
        except Exception as e:
            logger.warning(f"  ⚠️ Error leyendo ensemble config: {e}")

    def _load_segmenter(self):
        """Carga segmentador de pólipos."""
        self.image_segmenter = ColonPolypSegmenter().to(self.device)
        seg_path = paths.SEGMENTER_CHECKPOINT
        if seg_path.exists():
            ckpt = torch.load(seg_path, map_location=self.device, weights_only=False)
            self.image_segmenter.load_state_dict(ckpt["model_state_dict"])
            self.image_segmenter.eval()
            logger.info("  ✅ Segmentador cargado")
        else:
            self.image_segmenter = None
            logger.warning(f"  ⚠️ Segmentador no encontrado: {seg_path}")

    def _load_tabular(self):
        """Carga modelo tabular."""
        tab_path = paths.TABULAR_MODEL_PATH
        prep_path = paths.TABULAR_PREPROCESSOR_PATH
        if tab_path.exists() and prep_path.exists():
            self.tabular_model = joblib.load(tab_path)
            self.tabular_preprocessor = joblib.load(prep_path)
            logger.info("  ✅ Modelo tabular cargado")

    # ═════════════════════════════════════════════
    #  PREPROCESAMIENTO
    # ═════════════════════════════════════════════

    def _preprocess_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """Preprocesa para Model A (inscribed 0.68)."""
        return self.standardizer.process_image(image_bgr)

    def _preprocess_tissue_only(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        """Preprocesa para Model B (tissue-only, multi-crop)."""
        if self.tissue_preprocessor is None:
            self.tissue_preprocessor = TissueOnlyPreprocessor(
                target_size=model_cfg.IMAGE_SIZE, n_crops=5
            )
        return self.tissue_preprocessor.process_image(image_bgr)

    # ═════════════════════════════════════════════
    #  PREDICCIÓN CON ENSEMBLE
    # ═════════════════════════════════════════════

    def _predict_model_a(self, image_bgr: np.ndarray) -> np.ndarray:
        """Predicción con Model A."""
        preprocessed = self._preprocess_image(image_bgr)
        preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)

        tensor = (
            self.img_transform(image=preprocessed_rgb)["image"]
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            logits = self.image_classifier(tensor) / self.temp_a
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        return probs

    def _predict_model_b(self, image_bgr: np.ndarray) -> np.ndarray:
        """Predicción con Model B (multi-crop promediado)."""
        crops_bgr = self._preprocess_tissue_only(image_bgr)

        all_probs = []
        with torch.no_grad():
            for crop in crops_bgr:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = (
                    self.img_transform(image=crop_rgb)["image"]
                    .unsqueeze(0)
                    .to(self.device)
                )
                logits = self.tissue_classifier(tensor) / self.temp_b
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                all_probs.append(probs)

        return np.mean(all_probs, axis=0)

    def _combine_predictions(
        self, probs_a: np.ndarray, probs_b: np.ndarray
    ) -> np.ndarray:
        """Combina predicciones de Model A y Model B."""
        if self.use_uncertainty:
            return self._combine_uncertainty_aware(probs_a, probs_b)
        else:
            return self.ensemble_alpha * probs_a + self.ensemble_beta * probs_b

    def _combine_uncertainty_aware(
        self, probs_a: np.ndarray, probs_b: np.ndarray
    ) -> np.ndarray:
        """Ponderación por incertidumbre (entropía)."""
        eps = 1e-10
        max_entropy = np.log(self.num_classes)

        entropy_a = -np.sum(probs_a * np.log(probs_a + eps)) / max_entropy
        entropy_b = -np.sum(probs_b * np.log(probs_b + eps)) / max_entropy

        conf_a = 1.0 - entropy_a
        conf_b = 1.0 - entropy_b

        w_a = self.ensemble_alpha * conf_a
        w_b = self.ensemble_beta * conf_b

        total = w_a + w_b + eps
        return (w_a / total) * probs_a + (w_b / total) * probs_b

    # ═════════════════════════════════════════════
    #  DIAGNÓSTICO PRINCIPAL
    # ═════════════════════════════════════════════

    def diagnose(
        self,
        patient_data: dict,
        image_path: str | None = None,
        patient_id: int | None = None,
    ) -> dict:
        """Ejecuta diagnóstico multimodal completo."""
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
                "level": "BAJO",
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
        if patient_data and self.tabular_model:
            tab_res = self._analyze_tabular(patient_data)
            if tab_res:
                result["tabular_analysis"] = tab_res
                tabular_prob = tab_res["prediction_score"]

        # Fusión
        final_score = tabular_prob if tabular_prob else 0.0

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

    def _analyze_image(self, image_path: str) -> dict | None:
        """Análisis con ensemble ADAPTATIVO."""
        try:
            original = cv2.imread(image_path)
            if original is None:
                logger.error(f"No se pudo leer: {image_path}")
                return None

            # Model A
            probs_a = self._predict_model_a(original)

            # Ensemble adaptativo
            attention_ratio = None
            alpha_used = 1.0
            beta_used = 0.0
            probs = probs_a

            if self.ensemble_available and self.tissue_classifier:
                probs_b = self._predict_model_b(original)

                # Grad-CAM rápido → attention ratio
                attention_ratio = self._get_attention_ratio(original)

                # Pesos dinámicos
                sigmoid_val = 1.0 / (1.0 + np.exp(-5.0 * (attention_ratio - 1.3)))
                alpha_used = 0.25 + 0.50 * sigmoid_val
                beta_used = 1.0 - alpha_used

                probs = alpha_used * probs_a + beta_used * probs_b

                logger.info(
                    f"  Attention: {attention_ratio:.2f} "
                    f"→ α={alpha_used:.2f}, β={beta_used:.2f}"
                )
                logger.info(
                    f"  A: {dict(zip(self.class_names.values(), probs_a.round(3).tolist(), strict=True))}"
                )
                logger.info(
                    f"  B: {dict(zip(self.class_names.values(), probs_b.round(3).tolist(), strict=True))}"
                )
                logger.info(
                    f"  → {dict(zip(self.class_names.values(), probs.round(3).tolist(), strict=True))}"
                )

            pred_idx = int(np.argmax(probs))
            pred_class = self.class_names.get(pred_idx, f"class_{pred_idx}")
            pred_score = float(probs[pred_idx])

            logger.info(f"  → {pred_class} ({pred_score:.1%})")

            gradcam_data = {}
            if self.image_classifier is not None:
                gradcam_data = self._generate_gradcam(
                    original, pred_idx, alpha_used, beta_used
                )

            polyp_detected = False
            report_url = None
            lesion_count = 0

            if pred_class == "polyp" and self.image_segmenter:
                polyp_detected, report_url, lesion_count = self._run_segmentation(
                    original
                )

            return {
                "prediction_class": pred_class,
                "prediction_score": pred_score,
                "polyp_detected": polyp_detected,
                "lesion_count": lesion_count,
                "report_path": report_url,
                "probabilities": {
                    self.class_names.get(i, f"class_{i}"): float(probs[i])
                    for i in range(len(probs))
                },
                "ensemble_used": self.ensemble_available,
                "ensemble_mode": "adaptive" if self.ensemble_available else "single",
                "attention_ratio": (
                    round(attention_ratio, 3) if attention_ratio is not None else None
                ),
                "alpha_used": round(float(alpha_used), 3),
                "beta_used": round(float(beta_used), 3),
                "model_a_probs": {
                    self.class_names.get(i, f"class_{i}"): float(probs_a[i])
                    for i in range(len(probs_a))
                },
                "gradcam_a": gradcam_data.get("gradcam_a"),
                "gradcam_b": gradcam_data.get("gradcam_b"),
                "gradcam_fusion": gradcam_data.get("gradcam_fusion"),
            }
        except Exception as e:
            logger.error(f"Error analizando imagen: {e}")
            logger.error(traceback.format_exc())
            return None

    # ═════════════════════════════════════════════
    #  ATTENTION RATIO (usa pytorch-grad-cam)
    # ═════════════════════════════════════════════

    def _get_attention_ratio(self, image_bgr: np.ndarray) -> float:
        """
        Calcula ratio de atención centro/borde usando pytorch-grad-cam.

        Usa GradCAM (vanilla) para velocidad en el cálculo adaptativo.
        Para visualización se usa el método configurado (gradcam++, etc).
        """
        try:
            preprocessed = self._preprocess_image(image_bgr)
            preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)

            tensor = (
                self.img_transform(image=preprocessed_rgb)["image"]
                .unsqueeze(0)
                .to(self.device)
            )

            # ═══ NUEVO: usa generate_gradcam ═══
            # Para attention ratio usamos "gradcam" (más rápido)
            cam, _ = generate_gradcam(
                model=self.image_classifier,
                input_tensor=tensor,
                target_class=None,
                method="gradcam",
            )

            h_cam, w_cam = cam.shape
            h_img, w_img = gray.shape

            margin_h = max(1, int(h_cam * 0.25))
            margin_w = max(1, int(w_cam * 0.25))

            center_mask = np.zeros((h_cam, w_cam), dtype=bool)
            center_mask[
                margin_h : h_cam - margin_h,
                margin_w : w_cam - margin_w,
            ] = True

            # Pointing
            max_pos = np.unravel_index(cam.argmax(), cam.shape)
            max_in_center = center_mask[max_pos[0], max_pos[1]]

            # Top 10%
            threshold_val = np.percentile(cam, 90)
            hot_pixels = cam >= threshold_val
            hot_total = hot_pixels.sum()

            if hot_total == 0:
                logger.info("  📊 Attention: sin activación → 1.5")
                return 1.5

            hot_in_center = (hot_pixels & center_mask).sum()
            hot_in_border = (hot_pixels & ~center_mask).sum()

            hot_ratio = float(hot_in_center) / max(float(hot_in_border), 1.0)

            # Resize CAM para comparar con imagen
            cam_resized = cv2.resize(
                cam,
                (w_img, h_img),
                interpolation=cv2.INTER_LINEAR,
            )
            hot_resized = cam_resized >= np.percentile(cam_resized, 90)

            img_margin_h = max(1, int(h_img * 0.25))
            img_margin_w = max(1, int(w_img * 0.25))
            center_mask_img = np.zeros((h_img, w_img), dtype=bool)
            center_mask_img[
                img_margin_h : h_img - img_margin_h,
                img_margin_w : w_img - img_margin_w,
            ] = True

            hot_border_mask = hot_resized & ~center_mask_img

            # ══════════════════════════════════════
            # SOLO penalizar si hay NEGRO donde mira
            # Si hay tejido (brillo > 60) → NO penalizar
            # ══════════════════════════════════════

            border_brightness = -1.0
            if hot_border_mask.sum() > 0:
                border_brightness = float(gray[hot_border_mask].mean())

                if border_brightness < 30:
                    hot_ratio *= 0.3
                    logger.info(
                        f"  📊 🚨 NEGRO en borde "
                        f"(brillo={border_brightness:.0f}) "
                        f"→ ratio×0.3"
                    )
                elif border_brightness < 60:
                    hot_ratio *= 0.6
                    logger.info(
                        f"  📊 ⚠️ Gris oscuro en borde "
                        f"(brillo={border_brightness:.0f}) "
                        f"→ ratio×0.6"
                    )
                else:
                    logger.info(
                        f"  📊 ✅ Tejido en borde "
                        f"(brillo={border_brightness:.0f}) "
                        f"→ sin penalización"
                    )

            # Verificar máximo
            max_brightness = -1.0
            if not max_in_center:
                max_y = int(max_pos[0] / h_cam * h_img)
                max_x = int(max_pos[1] / w_cam * w_img)
                max_y = min(max_y, h_img - 1)
                max_x = min(max_x, w_img - 1)

                wy1 = max(0, max_y - 5)
                wy2 = min(h_img, max_y + 6)
                wx1 = max(0, max_x - 5)
                wx2 = min(w_img, max_x + 6)
                max_brightness = float(gray[wy1:wy2, wx1:wx2].mean())

                if max_brightness < 30:
                    hot_ratio *= 0.3
                    logger.info(
                        f"  📊 🚨 Máximo en NEGRO "
                        f"(brillo={max_brightness:.0f}) "
                        f"→ ratio×0.3"
                    )
                elif max_brightness < 60:
                    hot_ratio *= 0.5
                    logger.info(
                        f"  📊 ⚠️ Máximo en gris "
                        f"(brillo={max_brightness:.0f}) "
                        f"→ ratio×0.5"
                    )
                else:
                    logger.info(
                        f"  📊 ✅ Máximo en tejido "
                        f"(brillo={max_brightness:.0f}) "
                        f"→ sin penalización"
                    )

            logger.info(
                f"  📊 FINAL: center={max_in_center}, "
                f"hot_c={hot_in_center}, "
                f"hot_b={hot_in_border}, "
                f"border_br={border_brightness:.0f}, "
                f"max_br={max_brightness:.0f}, "
                f"ratio={hot_ratio:.2f}"
            )

            return float(hot_ratio)

        except Exception as e:
            logger.warning(f"  ⚠️ Attention ratio FALLÓ: {e}")
            logger.warning(traceback.format_exc())
            return 1.5

    # ═════════════════════════════════════════════
    #  SEGMENTACIÓN
    # ═════════════════════════════════════════════

    def _run_segmentation(self, original: np.ndarray) -> tuple[bool, str | None, int]:
        """
        Ejecuta segmentación de pólipos.

        El overlay se hace sobre la imagen PROCESADA (post-crop),
        no sobre la original, porque la máscara del U-Net está
        alineada con la imagen procesada.
        """
        try:
            preprocessed = self._preprocess_image(original)
            preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
            tensor = (
                self.img_transform(image=preprocessed_rgb)["image"]
                .unsqueeze(0)
                .to(self.device)
            )

            if hasattr(self.image_segmenter, "predict_mask_tta"):
                masks, probs = self.image_segmenter.predict_mask_tta(
                    tensor, threshold=0.5
                )
            else:
                masks, probs = self.image_segmenter.predict_mask(tensor, threshold=0.5)

            mask_np = masks[0, 0].cpu().numpy()

            if mask_np.sum() <= 20:
                return False, None, 0

            if hasattr(self.image_segmenter, "postprocess_instances"):
                instance_mask, n_polyps = self.image_segmenter.postprocess_instances(
                    mask_np, min_area=100
                )
            else:
                instance_mask = mask_np
                n_polyps = 1 if mask_np.sum() > 20 else 0

            if n_polyps == 0:
                return False, None, 0

            overlay = preprocessed.copy()

            binary_mask = (instance_mask > 0).astype(np.float32)
            mask_resized = cv2.resize(
                binary_mask,
                (overlay.shape[1], overlay.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            red_overlay = overlay.copy()
            red_overlay[:, :, 2] = np.clip(
                red_overlay[:, :, 2].astype(int) + 100, 0, 255
            ).astype(np.uint8)
            red_overlay[:, :, 0] = (red_overlay[:, :, 0] * 0.5).astype(np.uint8)
            red_overlay[:, :, 1] = (red_overlay[:, :, 1] * 0.5).astype(np.uint8)

            idx = mask_resized > 0.5
            alpha = 0.5
            overlay[idx] = cv2.addWeighted(overlay, 1 - alpha, red_overlay, alpha, 0)[
                idx
            ]

            for polyp_id in range(1, n_polyps + 1):
                single_mask = (instance_mask == polyp_id).astype(np.uint8) * 255
                single_resized = cv2.resize(
                    single_mask,
                    (overlay.shape[1], overlay.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                contours, _ = cv2.findContours(
                    single_resized,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out_dir = paths.REPORTS
            out_dir.mkdir(parents=True, exist_ok=True)

            filename = f"report_{ts}.png"
            filepath = out_dir / filename
            cv2.imwrite(str(filepath), overlay)

            report_url = f"http://localhost:8000/static/reports/{filename}"
            return True, report_url, n_polyps

        except Exception as e:
            logger.error(f"Error en segmentación: {e}")
            logger.error(traceback.format_exc())
            return False, None, 0

    # ═════════════════════════════════════════════
    #  ANÁLISIS TABULAR
    # ═════════════════════════════════════════════

    def _analyze_tabular(self, patient_data: dict) -> dict | None:
        """Análisis de datos clínicos tabulares."""
        try:
            df = pd.DataFrame([patient_data])
            expected_cols = (
                self.tabular_model.feature_names_in_
                if hasattr(self.tabular_model, "feature_names_in_")
                else []
            )
            if len(expected_cols) > 0:
                for col in expected_cols:
                    if col not in df.columns:
                        df[col] = CLINICAL_DEFAULTS.get(col, 0)

            if isinstance(self.tabular_preprocessor, dict):
                X = df
            else:
                X = self.tabular_preprocessor.transform(df)

            if hasattr(self.tabular_model, "predict_proba"):
                prob = float(self.tabular_model.predict_proba(X)[0][1])
            else:
                prob = 0.5

            factors = []
            if patient_data.get("cea", 0) > 5.0:
                factors.append(("CEA Elevado", float(patient_data["cea"])))
            if patient_data.get("fit_positive", False):
                factors.append(("Test FIT (+)", 1.0))
            if patient_data.get("hemoglobin", 15) < 12.0:
                factors.append(
                    (
                        "Hemoglobina Baja",
                        float(patient_data["hemoglobin"]),
                    )
                )

            return {
                "prediction_score": prob,
                "high_risk": prob > 0.5,
                "top_risk_factors": factors[:5],
            }
        except Exception as e:
            logger.error(f"Error tabular: {e}")
            return None

    # ═════════════════════════════════════════════
    #  UTILIDADES
    # ═════════════════════════════════════════════

    def _get_final_diagnosis_text(self, img_class: str, score: float) -> str:
        """Texto de diagnóstico final."""
        if img_class == "polyp":
            return "Pólipo Adenomatoso Detectado"
        if img_class == "inflammation":
            return "Signos de Colitis Ulcerosa"
        if score >= 0.7:
            return "Alto Riesgo Clínico"
        return "Mucosa Sana"

    def _get_risk_level(self, score: float) -> dict:
        """Nivel de riesgo con color y porcentaje."""
        if score >= 0.7:
            return {
                "level": "ALTO",
                "score": score,
                "color": "#D32F2F",
                "percentage": f"{score * 100:.1f}%",
            }
        if score >= 0.4:
            return {
                "level": "MODERADO",
                "score": score,
                "color": "#F57C00",
                "percentage": f"{score * 100:.1f}%",
            }
        return {
            "level": "BAJO",
            "score": score,
            "color": "#388E3C",
            "percentage": f"{score * 100:.1f}%",
        }

    def _get_recommendations(self, score: float, img_class: str) -> list:
        """Recomendaciones clínicas."""
        recs = []
        if img_class == "polyp":
            recs.append("🔴 URGENTE: Programar polipectomía para resección de pólipo.")
        elif img_class == "inflammation":
            recs.append("🟠 Evaluar tratamiento antiinflamatorio.")
        elif score >= 0.7:
            recs.append("🔴 Colonoscopia diagnóstica urgente (Alto riesgo en sangre).")
        else:
            recs.append("🟢 Continuar con screening rutinario.")
        return recs

    # ═════════════════════════════════════════════
    #  GRAD-CAM PARA VISUALIZACIÓN (pytorch-grad-cam)
    # ═════════════════════════════════════════════

    def _generate_gradcam(
        self,
        image_bgr: np.ndarray,
        pred_idx: int,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> dict:
        """
        Genera Grad-CAM con los pesos ADAPTATIVOS de esta imagen.

        La fusión usa los MISMOS α, β que se usaron para la predicción,
        no los pesos base. Así el heatmap refleja fielmente la decisión.

        Usa el método CAM configurado (gradcam++, hirescam, etc.)
        para máxima calidad en la visualización.

        Args:
            image_bgr: Imagen original BGR
            pred_idx: Índice de la clase predicha
            alpha: Peso REAL de Model A para ESTA imagen
            beta: Peso REAL de Model B para ESTA imagen
        """
        result: dict[str, str | None] = {
            "gradcam_a": None,
            "gradcam_b": None,
            "gradcam_fusion": None,
        }

        # ── Preparar imagen para Model A ──
        preprocessed_a = self._preprocess_image(image_bgr)
        preprocessed_a_rgb = cv2.cvtColor(preprocessed_a, cv2.COLOR_BGR2RGB)
        tensor_a = (
            self.img_transform(image=preprocessed_a_rgb)["image"]
            .unsqueeze(0)
            .to(self.device)
        )

        # ── Grad-CAM Model A ──
        cam_a = None
        try:
            cam_a, _ = generate_gradcam(
                model=self.image_classifier,
                input_tensor=tensor_a,
                target_class=pred_idx,
                method=self.cam_method,
            )
            overlay_a = create_heatmap_overlay(preprocessed_a_rgb, cam_a, alpha=0.45)
            result["gradcam_a"] = self._numpy_to_base64(overlay_a)
        except Exception as e:
            logger.warning(f"  ⚠️ Grad-CAM A falló: {e}")

        # ── Grad-CAM Model B ──
        cam_b = None
        cam_b_resized = None
        if (
            self.ensemble_available
            and self.tissue_classifier is not None
            and beta > 0.01
        ):
            try:
                crops_bgr = self._preprocess_tissue_only(image_bgr)
                if crops_bgr:
                    crop_rgb = cv2.cvtColor(crops_bgr[0], cv2.COLOR_BGR2RGB)
                    tensor_b = (
                        self.img_transform(image=crop_rgb)["image"]
                        .unsqueeze(0)
                        .to(self.device)
                    )

                    cam_b, _ = generate_gradcam(
                        model=self.tissue_classifier,
                        input_tensor=tensor_b,
                        target_class=pred_idx,
                        method=self.cam_method,
                    )
                    overlay_b = create_heatmap_overlay(crop_rgb, cam_b, alpha=0.45)
                    result["gradcam_b"] = self._numpy_to_base64(overlay_b)

                    if cam_a is not None:
                        cam_b_resized = cv2.resize(
                            cam_b,
                            (cam_a.shape[1], cam_a.shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
            except Exception as e:
                logger.warning(f"  ⚠️ Grad-CAM B falló: {e}")

        # ── Fusión con pesos ADAPTATIVOS reales ──
        if cam_a is not None:
            if cam_b_resized is not None and beta > 0.01:
                cam_fusion = alpha * cam_a + beta * cam_b_resized

                cam_min = cam_fusion.min()
                cam_max = cam_fusion.max()
                if cam_max - cam_min > 1e-8:
                    cam_fusion = (cam_fusion - cam_min) / (cam_max - cam_min)
                else:
                    cam_fusion = cam_a

                logger.info(f"  Grad-CAM fusión: α={alpha:.2f}×A + β={beta:.2f}×B")
            else:
                cam_fusion = cam_a

            overlay_fusion = create_heatmap_overlay(
                preprocessed_a_rgb, cam_fusion, alpha=0.45
            )
            result["gradcam_fusion"] = self._numpy_to_base64(overlay_fusion)

        return result

    def _numpy_to_base64(self, img_rgb: np.ndarray) -> str:
        """Convierte imagen RGB numpy a data URI base64 PNG."""
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".png", img_bgr)
        b64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/png;base64,{b64}"


# ═════════════════════════════════════════════
#  SINGLETON
# ═════════════════════════════════════════════

_engine: DiagnosisEngine | None = None


def get_diagnosis_engine() -> DiagnosisEngine:
    """Obtiene instancia singleton del motor de diagnóstico."""
    global _engine
    if _engine is None:
        _engine = DiagnosisEngine()
        _engine.load_models()
    return _engine
