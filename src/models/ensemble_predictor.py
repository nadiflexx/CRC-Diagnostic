"""
src/ml/ensemble_predictor.py

Ensemble Adaptativo con Attention-Gating.

Estrategia:
  - Pesos DINÁMICOS por imagen basados en Grad-CAM de Model A
  - Si Model A mira tejido (ratio alto) → confiar más en A
  - Si Model A mira bordes NEGROS (ratio bajo) → confiar más en B
  - Si Model A mira bordes con TEJIDO → confiar en A (NO penalizar)
  - Transición suave con sigmoid
"""

import json
from pathlib import Path
import platform

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config.logger import log as logger
from src.config.paths import paths
from src.data.processing.image_preprocessor import (
    ColonoscopyDataset,
    get_val_transforms,
)
from src.data.processing.standardizer import MultiSourceStandardizer
from src.data.processing.tissue_only_preprocessor import TissueOnlyPreprocessor
from src.evaluation.gradcam import generate_gradcam
from src.models.image_classifier import ColonCancerClassifier, FocalLoss
from src.models.tissue_classifier import TissueOnlyClassifier
from src.training.train_tissue_classifier import (
    TissueCropDataset,
    get_tissue_val_transforms,
    validate_multicrop,
)

TISSUE_DIR = paths.COLON_TISSUE_ONLY


class EnsemblePredictor:
    """
    Ensemble Adaptativo: Model A + Model B.

    Los pesos se ajustan POR IMAGEN según dónde mira Model A:
    - Mira tejido → confiar en A (tiene contexto útil)
    - Mira bordes negros → confiar en B (A está usando shortcuts)
    - Mira bordes con tejido → confiar en A (no es shortcut)
    """

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_a: ColonCancerClassifier | None = None
        self.model_b: TissueOnlyClassifier | None = None

        self.temp_a = 1.0
        self.temp_b = 1.0

        # Pesos BASE (se modulan por atención)
        self.alpha_base = 0.5
        self.beta_base = 0.5

        # Parámetros del sigmoid adaptativo
        self.sigmoid_center = 1.5
        self.sigmoid_slope = 3.0
        self.alpha_min = 0.20
        self.alpha_max = 0.80

        self.use_adaptive = True
        self.num_classes = 3
        self.class_names: dict[int, str] = {}
        self.class_mapping: dict = {}

        # Preprocesadores
        self.preprocessor_a = MultiSourceStandardizer(target_size=384)
        self.preprocessor_b = TissueOnlyPreprocessor(target_size=384, n_crops=5)

        self.transform = get_val_transforms(384)

        self._loaded = False

    # ═════════════════════════════════════════════
    #  CARGA
    # ═════════════════════════════════════════════

    def load_models(self) -> bool:
        """Carga modelos A y B + config ensemble."""
        success_a = self._load_model_a()
        success_b = self._load_model_b()
        self._load_ensemble_config()

        self._loaded = success_a and success_b

        if self._loaded:
            logger.info(
                f"  ✅ Ensemble ADAPTATIVO cargado "
                f"(base α={self.alpha_base:.2f}, β={self.beta_base:.2f})"
            )
        elif success_a:
            logger.warning("  ⚠️ Solo Model A disponible")
            self.alpha_base = 1.0
            self.beta_base = 0.0
            self.use_adaptive = False
            self._loaded = True
        else:
            logger.error("  ❌ No se pudo cargar ningún modelo")

        return self._loaded

    @property
    def ensemble_available(self) -> bool:
        """True si ambos modelos están cargados."""
        return self.model_a is not None and self.model_b is not None

    def _load_model_a(self) -> bool:
        """Carga Model A (clasificador principal)."""
        path = paths.CLASSIFIER_CHECKPOINT
        if not path.exists():
            return False

        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        model_name = ckpt.get("model_name", "tf_efficientnetv2_s.in21k")
        self.num_classes = ckpt.get("num_classes", 3)
        self.class_mapping = ckpt.get("class_mapping", {})
        self.class_names = {
            int(k): v for k, v in self.class_mapping.get("classes", {}).items()
        }
        self.temp_a = ckpt.get("temperature", 1.0)

        self.model_a = ColonCancerClassifier(
            model_name=model_name,
            pretrained=False,
            num_classes=self.num_classes,
        ).to(self.device)
        self.model_a.load_state_dict(ckpt["model_state_dict"])
        self.model_a.eval()

        logger.info(
            f"  ✅ Model A: {model_name} "
            f"(F1={ckpt.get('best_f1', 0):.4f}, T={self.temp_a:.2f})"
        )
        return True

    def _load_model_b(self) -> bool:
        """Carga Model B (tissue-only)."""
        path = paths.TISSUE_CLASSIFIER_CHECKPOINT
        if not path.exists():
            return False

        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        model_name = ckpt.get("model_name", "tf_efficientnetv2_s.in21k")
        self.temp_b = ckpt.get("temperature", 1.0)

        self.model_b = TissueOnlyClassifier(
            model_name=model_name,
            pretrained=False,
            num_classes=self.num_classes,
        ).to(self.device)
        self.model_b.load_state_dict(ckpt["model_state_dict"])
        self.model_b.eval()

        logger.info(
            f"  ✅ Model B: {model_name} "
            f"(F1={ckpt.get('best_f1', 0):.4f}, T={self.temp_b:.2f})"
        )
        return True

    def _load_ensemble_config(self):
        """Carga configuración del ensemble."""
        config_path = paths.ENSEMBLE_CONFIG_PATH
        if not config_path.exists():
            return

        try:
            with open(config_path) as f:
                config = json.load(f)

            self.alpha_base = config.get("alpha", 0.5)
            self.beta_base = config.get("beta", 0.5)
            self.use_adaptive = config.get("use_adaptive", True)
            self.sigmoid_center = config.get("sigmoid_center", 1.3)
            self.sigmoid_slope = config.get("sigmoid_slope", 5.0)
            self.alpha_min = config.get("alpha_min", 0.25)
            self.alpha_max = config.get("alpha_max", 0.75)

            logger.info(
                f"  📄 Config: α_base={self.alpha_base:.2f}, "
                f"adaptive={self.use_adaptive}"
            )
        except Exception as e:
            logger.warning(f"  ⚠️ Error config: {e}")

    # ═════════════════════════════════════════════
    #  ATTENTION RATIO (content-aware)
    # ═════════════════════════════════════════════

    def compute_attention_ratio(
        self,
        tensor_a: torch.Tensor,
        preprocessed_bgr: np.ndarray | None = None,
    ) -> float:
        """
        Analiza A QUÉ mira Model A (no DÓNDE).

        Criterio único: ¿Mira TEJIDO o ARTEFACTOS OSCUROS?

        Returns:
            > 2.0: Mira tejido iluminado → confiar en A
            ~ 1.0: Zona gris/ambigua → neutro
            < 0.5: Mira artefactos negros → confiar en B
        """
        try:
            cam, _ = generate_gradcam(
                model=self.model_a,
                input_tensor=tensor_a,
                target_class=None,
                method="gradcam",
            )

            if preprocessed_bgr is None:
                logger.info("  📊 Sin imagen BGR → ratio neutro 1.5")
                return 1.5

            # ═══════════════════════════════════════════════════════════
            # PASO 1: Identificar zonas de ALTA atención
            # ═══════════════════════════════════════════════════════════

            h_cam, w_cam = cam.shape
            threshold = np.percentile(cam, 90)
            hot_mask = cam >= threshold

            if hot_mask.sum() == 0:
                logger.info("  📊 Sin atención detectada → ratio neutro 1.5")
                return 1.5

            # ═══════════════════════════════════════════════════════════
            # PASO 2: Analizar el CONTENIDO de las zonas calientes
            # ═══════════════════════════════════════════════════════════

            gray = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2GRAY)
            h_img, w_img = gray.shape

            cam_resized = cv2.resize(
                cam, (w_img, h_img), interpolation=cv2.INTER_LINEAR
            )
            hot_mask_resized = cam_resized >= np.percentile(cam_resized, 90)

            attention_pixels = gray[hot_mask_resized]

            if len(attention_pixels) == 0:
                return 1.5

            # ═══════════════════════════════════════════════════════════
            # PASO 3: Clasificar el contenido por BRILLO
            # ═══════════════════════════════════════════════════════════

            # Estadísticas de brillo
            mean_brightness = float(np.mean(attention_pixels))
            median_brightness = float(np.median(attention_pixels))
            p25_brightness = float(np.percentile(attention_pixels, 25))
            p75_brightness = float(np.percentile(attention_pixels, 75))

            # Distribución global de la imagen
            img_p25 = np.percentile(gray, 25)
            img_p50 = np.percentile(gray, 50)
            img_p75 = np.percentile(gray, 75)

            logger.info(
                f"  📊 Imagen: p25={img_p25:.0f}, p50={img_p50:.0f}, p75={img_p75:.0f}"
            )
            logger.info(
                f"  📊 Atención: mean={mean_brightness:.0f}, "
                f"median={median_brightness:.0f}, "
                f"p25={p25_brightness:.0f}, p75={p75_brightness:.0f}"
            )

            # ═══════════════════════════════════════════════════════════
            # PASO 4: Calcular ratio basado en CALIDAD del contenido
            # ═══════════════════════════════════════════════════════════

            # Caso 1: Mira MAYORMENTE tejido brillante
            if median_brightness > img_p75:
                # Mira el cuartil superior → tejido bien iluminado
                ratio = 3.0
                logger.info(
                    f"  📊 ✅ TEJIDO BRILLANTE: median={median_brightness:.0f} "
                    f"> p75={img_p75:.0f} → ratio={ratio:.2f}"
                )

            # Caso 2: Mira tejido promedio
            elif median_brightness > img_p50:
                # Mira por encima de la mediana → tejido normal
                ratio = 2.0
                logger.info(
                    f"  📊 ✅ TEJIDO NORMAL: median={median_brightness:.0f} "
                    f"> p50={img_p50:.0f} → ratio={ratio:.2f}"
                )

            # Caso 3: Mira zona intermedia (puede ser tejido oscuro o artefacto)
            elif median_brightness > img_p25:
                # Entre p25 y p50 → zona gris
                dark_pixels = np.sum(attention_pixels < img_p25)
                dark_ratio = dark_pixels / len(attention_pixels)

                if dark_ratio > 0.5:
                    # Más del 50% mira zonas oscuras → sospechoso
                    ratio = 0.7
                    logger.info(
                        f"  📊 ⚠️  ZONA GRIS con {dark_ratio:.0%} oscura "
                        f"→ ratio={ratio:.2f}"
                    )
                else:
                    # Distribución mixta → neutro
                    ratio = 1.2
                    logger.info(f"  📊 ⚠️  ZONA GRIS mixta → ratio={ratio:.2f}")

            # Caso 4: Mira MAYORMENTE zonas oscuras (artefactos)
            else:
                # Mediana por debajo de p25 → definitivamente artefactos
                ratio = 0.3
                logger.info(
                    f"  📊 🚨 ARTEFACTO OSCURO: median={median_brightness:.0f} "
                    f"< p25={img_p25:.0f} → ratio={ratio:.2f}"
                )

            # ═══════════════════════════════════════════════════════════
            # PASO 5: Penalización adicional por outliers extremos
            # ═══════════════════════════════════════════════════════════

            # Si tiene píxeles MUY oscuros (posibles marcos negros)
            very_dark_pixels = np.sum(attention_pixels < 20)
            very_dark_ratio = very_dark_pixels / len(attention_pixels)

            if very_dark_ratio > 0.3:
                # Más del 30% de la atención está en píxeles < 20 → artefacto
                penalty = 0.5
                ratio *= penalty
                logger.info(
                    f"  📊 🚨 {very_dark_ratio:.0%} atención en NEGRO PURO "
                    f"(<20) → penalización ×{penalty} → ratio={ratio:.2f}"
                )

            # ═══════════════════════════════════════════════════════════
            # PASO 6: Verificación de coherencia espacial
            # ═══════════════════════════════════════════════════════════

            # Punto de máxima atención
            max_pos = np.unravel_index(cam.argmax(), cam.shape)
            max_y = int(max_pos[0] / h_cam * h_img)
            max_x = int(max_pos[1] / w_cam * w_img)

            # Brillo en el punto máximo (ventana 11x11)
            y1, y2 = max(0, max_y - 5), min(h_img, max_y + 6)
            x1, x2 = max(0, max_x - 5), min(w_img, max_x + 6)
            max_point_brightness = float(gray[y1:y2, x1:x2].mean())

            logger.info(
                f"  📊 Punto máximo: br={max_point_brightness:.0f} "
                f"en ({max_x}, {max_y})"
            )

            # Si el punto máximo es muy oscuro, penalizar incluso si el promedio es OK
            if max_point_brightness < 30 and ratio > 1.0:
                penalty = 0.6
                ratio *= penalty
                logger.info(
                    f"  📊 🚨 Máximo en NEGRO ({max_point_brightness:.0f}<30) "
                    f"→ penalización ×{penalty} → ratio={ratio:.2f}"
                )

            return float(ratio)

        except Exception as e:
            logger.warning(f"  ⚠️ Attention ratio falló: {e}")
        return 1.5

    # ═════════════════════════════════════════════
    #  COMBINACIÓN ADAPTATIVA
    # ═════════════════════════════════════════════

    def compute_adaptive_weights(self, attention_ratio: float) -> tuple[float, float]:
        """
        Calcula pesos dinámicos basados en attention ratio.

        Sigmoid suave:
          ratio=3.0 → α=0.75, β=0.25  (confiar en A, mira tejido)
          ratio=1.3 → α=0.50, β=0.50  (neutro)
          ratio=0.3 → α=0.26, β=0.74  (confiar en B, A mira artefactos)
        """
        sigmoid_val = 1.0 / (
            1.0 + np.exp(-self.sigmoid_slope * (attention_ratio - self.sigmoid_center))
        )

        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * sigmoid_val
        beta = 1.0 - alpha

        return round(float(alpha), 3), round(float(beta), 3)

    def combine_predictions(
        self,
        probs_a: np.ndarray,
        probs_b: np.ndarray,
        attention_ratio: float | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """
        Combina predicciones con pesos adaptativos.

        Returns:
            (probs_combined, alpha_used, beta_used)
        """
        if self.use_adaptive and attention_ratio is not None:
            alpha, beta = self.compute_adaptive_weights(attention_ratio)
        else:
            alpha = self.alpha_base
            beta = self.beta_base

        combined = alpha * probs_a + beta * probs_b
        return combined, alpha, beta

    # ═════════════════════════════════════════════
    #  PREDICCIÓN COMPLETA
    # ═════════════════════════════════════════════

    def predict(self, image: np.ndarray | str | Path) -> dict:
        """
        Predicción con ensemble adaptativo.

        1. Model A predice + preprocesa imagen
        2. Grad-CAM → attention ratio (content-aware)
        3. Model B predice (tissue-only multi-crop)
        4. Pesos dinámicos según attention ratio
        5. Combinar
        """
        if not self._loaded:
            raise RuntimeError("Modelos no cargados. Llama load_models() primero.")

        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError("No se pudo cargar la imagen")

        # ── Model A (devuelve también imagen preprocesada) ──
        probs_a, tensor_a, preprocessed_a = self._predict_model_a_with_tensor(image)

        # ── Attention ratio (content-aware) ──
        attention_ratio = None
        if self.use_adaptive and self.model_b is not None:
            attention_ratio = self.compute_attention_ratio(tensor_a, preprocessed_a)

        # ── Model B ──
        if self.model_b is not None and self.beta_base > 0:
            probs_b = self._predict_model_b(image)
        else:
            probs_b = probs_a

        # ── Combinar ──
        probs, alpha_used, beta_used = self.combine_predictions(
            probs_a, probs_b, attention_ratio
        )

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        logger.info(
            f"  Ensemble: attention={attention_ratio}, "
            f"α={alpha_used:.2f}, β={beta_used:.2f} "
            f"→ {self.class_names.get(pred_class, '?')} ({confidence:.1%})"
        )

        return {
            "class_idx": pred_class,
            "class_name": self.class_names.get(pred_class, str(pred_class)),
            "confidence": confidence,
            "probabilities": probs.tolist(),
            "model_a_probs": probs_a.tolist(),
            "model_b_probs": probs_b.tolist(),
            "attention_ratio": (
                round(attention_ratio, 3) if attention_ratio is not None else None
            ),
            "alpha_used": alpha_used,
            "beta_used": beta_used,
            "mode": "adaptive" if self.use_adaptive else "fixed",
        }

    def _predict_model_a_with_tensor(
        self, img_bgr: np.ndarray
    ) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
        """
        Predice con Model A.

        Returns:
            (probabilidades, tensor, imagen_preprocesada_bgr)
        """
        processed = self.preprocessor_a.process_image(img_bgr)
        img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=img_rgb)
        tensor = augmented["image"].unsqueeze(0).to(self.device)

        self.model_a.eval()
        with torch.no_grad():
            logits = self.model_a(tensor) / self.temp_a
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        return probs, tensor, processed

    def _predict_model_b(self, img_bgr: np.ndarray) -> np.ndarray:
        """Predicción con Model B (tissue-only multi-crop)."""
        crops_bgr = self.preprocessor_b.process_image(img_bgr)

        self.model_b.eval()
        all_probs = []

        with torch.no_grad():
            for crop in crops_bgr:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                augmented = self.transform(image=crop_rgb)
                t = augmented["image"].unsqueeze(0).to(self.device)
                logits = self.model_b(t) / self.temp_b
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                all_probs.append(probs)

        return np.mean(all_probs, axis=0)

    # ═════════════════════════════════════════════
    #  BATCH PREDICTION (para evaluación)
    # ═════════════════════════════════════════════

    def predict_batch_from_preprocessed(
        self,
        test_paths: list[str],
        test_labels: list[int],
        image_size: int = 384,
        batch_size: int = 16,
        n_crops: int = 5,
    ) -> dict:
        """Predicción batch con pesos fijos (para evaluación rápida)."""

        logger.info("  Model A predicciones...")
        test_ds_a = ColonoscopyDataset(
            test_paths,
            test_labels,
            transform=get_val_transforms(image_size),
            image_size=image_size,
        )

        is_win = platform.system() == "Windows"
        loader_a = DataLoader(
            test_ds_a,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0 if is_win else 4,
        )

        probs_a_list = []
        self.model_a.eval()
        with torch.no_grad():
            for images, _ in loader_a:
                images = images.to(self.device)
                logits = self.model_a(images) / self.temp_a
                probs = F.softmax(logits, dim=1).cpu().numpy()
                probs_a_list.append(probs)

        probs_a_all = np.vstack(probs_a_list)

        probs_b_all = probs_a_all
        if self.model_b is not None:
            logger.info("  Model B predicciones (multi-crop)...")

            crop_ds = TissueCropDataset(
                test_paths,
                test_labels,
                TISSUE_DIR,
                transform=get_tissue_val_transforms(image_size),
                image_size=image_size,
                n_crops=n_crops,
            )

            criterion = FocalLoss(num_classes=self.num_classes, label_smoothing=0.0)
            m_b = validate_multicrop(
                self.model_b,
                crop_ds,
                criterion,
                self.device,
                self.temp_b,
                batch_size * 2,
            )
            probs_b_all = m_b["probabilities"]

        logger.info("  Combinando ensemble...")
        n = min(len(probs_a_all), len(probs_b_all))
        probs_a_all = probs_a_all[:n]
        probs_b_all = probs_b_all[:n]
        labels = np.array(test_labels[:n])

        ensemble_probs = self.alpha_base * probs_a_all + self.beta_base * probs_b_all
        preds = np.argmax(ensemble_probs, axis=1)

        try:
            auc = roc_auc_score(
                labels, ensemble_probs, multi_class="ovr", average="macro"
            )
        except Exception:
            auc = 0.0

        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro", zero_division=0),
            "precision": precision_score(
                labels, preds, average="macro", zero_division=0
            ),
            "recall": recall_score(labels, preds, average="macro", zero_division=0),
            "auc": auc,
            "predictions": preds,
            "labels": labels,
            "probabilities": ensemble_probs,
            "probs_a": probs_a_all,
            "probs_b": probs_b_all,
        }

    # ═════════════════════════════════════════════
    #  OPTIMIZACIÓN
    # ═════════════════════════════════════════════

    def optimize_weights(
        self,
        val_paths: list[str],
        val_labels: list[int],
        image_size: int = 384,
        n_crops: int = 5,
        metric: str = "f1",
    ) -> tuple[float, float]:
        """Grid search de pesos base en validation set."""

        logger.info("\n═══ OPTIMIZANDO PESOS BASE ═══")

        val_ds_a = ColonoscopyDataset(
            val_paths,
            val_labels,
            transform=get_val_transforms(image_size),
            image_size=image_size,
        )

        is_win = platform.system() == "Windows"
        loader_a = DataLoader(
            val_ds_a,
            batch_size=32,
            shuffle=False,
            num_workers=0 if is_win else 4,
        )

        probs_a = []
        self.model_a.eval()
        with torch.no_grad():
            for images, _ in loader_a:
                images = images.to(self.device)
                logits = self.model_a(images) / self.temp_a
                p = F.softmax(logits, dim=1).cpu().numpy()
                probs_a.append(p)
        probs_a = np.vstack(probs_a)

        crop_ds = TissueCropDataset(
            val_paths,
            val_labels,
            TISSUE_DIR,
            transform=get_tissue_val_transforms(image_size),
            image_size=image_size,
            n_crops=n_crops,
        )
        criterion = FocalLoss(num_classes=self.num_classes, label_smoothing=0.0)
        m_b = validate_multicrop(
            self.model_b, crop_ds, criterion, self.device, self.temp_b
        )
        probs_b = m_b["probabilities"]

        n = min(len(probs_a), len(probs_b))
        probs_a = probs_a[:n]
        probs_b = probs_b[:n]
        labels = np.array(val_labels[:n])

        best_score = -1.0
        best_alpha = 0.5
        results = []

        for alpha_c in np.arange(0.0, 1.05, 0.05):
            beta_c = 1.0 - alpha_c
            combo = alpha_c * probs_a + beta_c * probs_b
            preds = np.argmax(combo, axis=1)

            if metric == "f1":
                score = f1_score(labels, preds, average="macro", zero_division=0)
            elif metric == "recall":
                score = recall_score(labels, preds, average="macro", zero_division=0)
            else:
                score = accuracy_score(labels, preds)

            results.append(
                {"alpha": round(alpha_c, 2), "beta": round(beta_c, 2), "score": score}
            )

            if score > best_score:
                best_score = score
                best_alpha = alpha_c

        self.alpha_base = best_alpha
        self.beta_base = 1.0 - best_alpha

        results.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"\n  Top 5 ({metric}):")
        for r in results[:5]:
            logger.info(
                f"    α={r['alpha']:.2f} β={r['beta']:.2f} → {metric}={r['score']:.4f}"
            )

        logger.info(
            f"\n  ✅ ÓPTIMO: α_base={self.alpha_base:.2f}, β_base={self.beta_base:.2f}"
        )

        self._save_ensemble_config()
        return self.alpha_base, self.beta_base

    def _save_ensemble_config(self):
        """Guarda configuración del ensemble."""
        config = {
            "alpha": round(self.alpha_base, 4),
            "beta": round(self.beta_base, 4),
            "use_adaptive": self.use_adaptive,
            "sigmoid_center": self.sigmoid_center,
            "sigmoid_slope": self.sigmoid_slope,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "temp_a": round(self.temp_a, 4),
            "temp_b": round(self.temp_b, 4),
            "model_a": "best_classifier.pth",
            "model_b": "best_tissue_classifier.pth",
        }

        config_path = paths.ENSEMBLE_CONFIG_PATH
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"  💾 Config: {config_path}")
