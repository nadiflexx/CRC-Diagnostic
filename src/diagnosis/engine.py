"""
src/diagnosis/engine.py

Motor de diagnóstico multimodal.
Gestiona: segmentación, tabular, Grad-CAM visualización, fusión multimodal.
"""

import base64
from datetime import datetime
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
from src.evaluation.gradcam import (
    CamMethod,
    create_heatmap_overlay,
    generate_gradcam,
)
from src.models.ensemble_predictor import EnsemblePredictor
from src.models.polyp_segmenter import ColonPolypSegmenter


class DiagnosisEngine:
    """Motor de diagnóstico multimodal con ensemble adaptativo."""

    def __init__(self, cam_method: CamMethod = "gradcam++"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_loaded = False
        self.cam_method: CamMethod = cam_method

        # ── Ensemble predictor (clasificación A + B) ──
        self.ensemble = EnsemblePredictor(device=self.device)

        # ── Otros modelos (propiedad del engine) ──
        self.image_segmenter = None
        self.tabular_model = None
        self.tabular_preprocessor = None

        # Transforms (para Grad-CAM visualización)
        self.img_transform = get_val_transforms(model_cfg.IMAGE_SIZE)

        # Metadata (se llena desde ensemble al cargar)
        self.class_names: dict[int, str] = {}
        self.clinical_info: dict = {}
        self.num_classes = 3

    # ═════════════════════════════════════════════
    #  CARGA DE MODELOS
    # ═════════════════════════════════════════════

    def load_models(self):
        """Carga todos los modelos del motor."""
        logger.info("Cargando modelos del motor de diagnóstico…")

        # ── Clasificación (delegada al ensemble) ──
        self.ensemble.load_models()

        # Copiar metadata del ensemble
        self.class_names = self.ensemble.class_names or DEFAULT_CLASS_NAMES
        self.num_classes = self.ensemble.num_classes
        self.clinical_info = self.ensemble.class_mapping.get("clinical_info", {})

        # ── Segmentación y tabular (propiedad del engine) ──
        self._load_segmenter()
        self._load_tabular()

        self.models_loaded = True

        if self.ensemble.ensemble_available:
            logger.info(f"✅ Motor listo (ENSEMBLE adaptativo, CAM={self.cam_method})")
        else:
            logger.info(f"✅ Motor listo (Model A solo, CAM={self.cam_method})")

    def _load_segmenter(self):
        """Carga segmentador de pólipos."""
        seg_path = paths.SEGMENTER_CHECKPOINT
        if not seg_path.exists():
            self.image_segmenter = None
            logger.warning(f"  ⚠️ Segmentador no encontrado: {seg_path}")
            return

        self.image_segmenter = ColonPolypSegmenter().to(self.device)
        ckpt = torch.load(seg_path, map_location=self.device, weights_only=False)
        self.image_segmenter.load_state_dict(ckpt["model_state_dict"])
        self.image_segmenter.eval()
        logger.info("  ✅ Segmentador cargado")

    def _load_tabular(self):
        """Carga modelo tabular."""
        tab_path = paths.TABULAR_MODEL_PATH
        prep_path = paths.TABULAR_PREPROCESSOR_PATH
        if tab_path.exists() and prep_path.exists():
            self.tabular_model = joblib.load(tab_path)
            self.tabular_preprocessor = joblib.load(prep_path)
            logger.info("  ✅ Modelo tabular cargado")

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

        # Fusión multimodal
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

    # ═════════════════════════════════════════════
    #  ANÁLISIS DE IMAGEN (delega al ensemble)
    # ═════════════════════════════════════════════

    def _analyze_image(self, image_path: str) -> dict | None:
        """Análisis de imagen con ensemble adaptativo."""
        try:
            original = cv2.imread(image_path)
            if original is None:
                logger.error(f"No se pudo leer: {image_path}")
                return None

            # ══ CLASIFICACIÓN → delegada al EnsemblePredictor ══
            prediction = self.ensemble.predict(original)

            pred_idx = prediction["class_idx"]
            pred_class = prediction["class_name"]
            pred_score = prediction["confidence"]
            probs = np.array(prediction["probabilities"])
            probs_a = np.array(prediction["model_a_probs"])
            alpha_used = prediction["alpha_used"]
            beta_used = prediction["beta_used"]
            attention_ratio = prediction["attention_ratio"]

            logger.info(f"  → {pred_class} ({pred_score:.1%})")

            # ══ GRAD-CAM VISUALIZACIÓN (responsabilidad del engine) ══
            gradcam_data = {}
            if self.ensemble.model_a is not None:
                gradcam_data = self._generate_gradcam(
                    original, pred_idx, alpha_used, beta_used
                )

            # ══ SEGMENTACIÓN ══
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
                "ensemble_used": self.ensemble.ensemble_available,
                "ensemble_mode": prediction["mode"],
                "attention_ratio": attention_ratio,
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
    #  GRAD-CAM VISUALIZACIÓN
    # ═════════════════════════════════════════════

    def _generate_gradcam(
        self,
        image_bgr: np.ndarray,
        pred_idx: int,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> dict:
        """
        Genera Grad-CAM para visualización.

        Usa los MISMOS α, β que se usaron en la predicción
        para que el heatmap refleje fielmente la decisión.
        """
        result: dict[str, str | None] = {
            "gradcam_a": None,
            "gradcam_b": None,
            "gradcam_fusion": None,
        }

        # ── Preparar imagen Model A ──
        preprocessed_a = self.ensemble.preprocessor_a.process_image(image_bgr)
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
                model=self.ensemble.model_a,
                input_tensor=tensor_a,
                target_class=pred_idx,
                method=self.cam_method,
            )
            overlay_a = create_heatmap_overlay(preprocessed_a_rgb, cam_a, alpha=0.45)
            result["gradcam_a"] = self._numpy_to_base64(overlay_a)
        except Exception as e:
            logger.warning(f"  ⚠️ Grad-CAM A falló: {e}")

        # ── Grad-CAM Model B ──
        cam_b_resized = None
        if (
            self.ensemble.ensemble_available
            and self.ensemble.model_b is not None
            and beta > 0.01
        ):
            try:
                crops_bgr = self.ensemble.preprocessor_b.process_image(image_bgr)
                if crops_bgr:
                    crop_rgb = cv2.cvtColor(crops_bgr[0], cv2.COLOR_BGR2RGB)
                    tensor_b = (
                        self.img_transform(image=crop_rgb)["image"]
                        .unsqueeze(0)
                        .to(self.device)
                    )

                    cam_b, _ = generate_gradcam(
                        model=self.ensemble.model_b,
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

        # ── Fusión con pesos adaptativos reales ──
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

    # ═════════════════════════════════════════════
    #  SEGMENTACIÓN
    # ═════════════════════════════════════════════

    def _run_segmentation(self, original: np.ndarray) -> tuple[bool, str | None, int]:
        """Ejecuta segmentación de pólipos."""
        try:
            preprocessed = self.ensemble.preprocessor_a.process_image(original)
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
            blend_alpha = 0.5
            overlay[idx] = cv2.addWeighted(
                overlay, 1 - blend_alpha, red_overlay, blend_alpha, 0
            )[idx]

            for polyp_id in range(1, n_polyps + 1):
                single_mask = (instance_mask == polyp_id).astype(np.uint8) * 255
                single_resized = cv2.resize(
                    single_mask,
                    (overlay.shape[1], overlay.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                contours, _ = cv2.findContours(
                    single_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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
                factors.append(("Hemoglobina Baja", float(patient_data["hemoglobin"])))

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
