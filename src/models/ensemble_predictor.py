"""
src/ml/ensemble_predictor.py

Ensemble Adaptativo con Attention-Gating.

Estrategia:
  - Pesos DINÁMICOS por imagen basados en Grad-CAM de Model A
  - Si Model A mira tejido (ratio alto) → confiar más en A
  - Si Model A mira bordes/esquinas (ratio bajo) → confiar más en B
  - Transición suave con sigmoid

Refs:
  - Mehrtash et al. 2020: Preprocessing diversity ensembles
  - Lakshminarayanan et al. 2017: Uncertainty estimation
  - Selvaraju et al. 2017: Grad-CAM for attention analysis
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

from config.logger import log as logger
from config.paths import paths
from data.processing.image_preprocessor import (
    ColonoscopyDataset,
    get_val_transforms,
)
from data.processing.standardizer import (
    MultiSourceStandardizer,
)
from data.processing.tissue_only_preprocessor import (
    TissueOnlyPreprocessor,
)
from evaluation.gradcam import (
    GradCAM,
    find_target_layer,
)
from models.image_classifier import ColonCancerClassifier, FocalLoss
from models.tissue_classifier import TissueOnlyClassifier
from training.train_tissue_classifier import (
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
    - Mira bordes → confiar en B (A está usando shortcuts)
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
        self.sigmoid_center = 1.3
        self.sigmoid_slope = 5.0
        self.alpha_min = 0.25
        self.alpha_max = 0.75

        self.use_adaptive = True
        self.num_classes = 3
        self.class_names: dict[int, str] = {}
        self.class_mapping: dict = {}

        # Preprocesadores
        self.preprocessor_a = MultiSourceStandardizer(target_size=384)
        self.preprocessor_b = TissueOnlyPreprocessor(target_size=384, n_crops=5)

        self.transform = get_val_transforms(384)

        # Cache para Grad-CAM target layer
        self._target_layer_a = None

        self._loaded = False

    # ═════════════════════════════════════════════
    #  CARGA
    # ═════════════════════════════════════════════

    def load_models(self) -> bool:
        """
        Load the ensemble models.

        :return: True if all models are loaded successfully, False otherwise.
        """
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

    def _load_model_a(self) -> bool:
        """Load Model A."""
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
        """Load Model B."""
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
        """Load ensemble config."""
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
    #  ATTENTION RATIO (Grad-CAM rápido)
    # ═════════════════════════════════════════════

    def compute_attention_ratio(self, tensor_a: torch.Tensor) -> float:
        """
        Analiza si el foco PRINCIPAL de Model A está en bordes.

        Método: Pointing Game + distribución top-10%.

        Score alto → tejido → confiar en A
        Score bajo → bordes → confiar en B
        """
        try:
            if self._target_layer_a is None:
                self._target_layer_a = find_target_layer(self.model_a)

            grad_cam = GradCAM(self.model_a, self._target_layer_a)

            try:
                cam, _, _ = grad_cam.generate(tensor_a)

                h, w = cam.shape
                margin_h = max(1, int(h * 0.25))
                margin_w = max(1, int(w * 0.25))

                center_mask = np.zeros((h, w), dtype=bool)
                center_mask[
                    margin_h : h - margin_h,
                    margin_w : w - margin_w,
                ] = True

                # Pointing: ¿máximo en centro?
                max_pos = np.unravel_index(cam.argmax(), cam.shape)
                max_in_center = center_mask[max_pos[0], max_pos[1]]

                # Top 10% de activación
                threshold = np.percentile(cam, 90)
                hot_pixels = cam >= threshold
                hot_total = hot_pixels.sum()

                if hot_total == 0:
                    return 1.5

                hot_in_center = (hot_pixels & center_mask).sum()
                hot_in_border = (hot_pixels & ~center_mask).sum()

                hot_ratio = float(hot_in_center) / max(float(hot_in_border), 1.0)

                if not max_in_center:
                    hot_ratio *= 0.5

                border_hot_pct = hot_in_border / hot_total
                if border_hot_pct > 0.6:
                    hot_ratio *= 0.5

                return float(hot_ratio)

            finally:
                grad_cam.remove_hooks()

        except Exception:
            return 1.5

    # ═════════════════════════════════════════════
    #  COMBINACIÓN ADAPTATIVA
    # ═════════════════════════════════════════════

    def compute_adaptive_weights(self, attention_ratio: float) -> tuple[float, float]:
        """
        Calcula pesos dinámicos basados en attention ratio.

        Sigmoid suave:
          ratio=2.0 → α=0.74, β=0.26 (confiar en A)
          ratio=1.3 → α=0.50, β=0.50 (neutro)
          ratio=0.8 → α=0.27, β=0.73 (confiar en B)
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
        Combina predicciones de ambos modelos.

        Si use_adaptive=True y attention_ratio disponible:
          → pesos dinámicos por imagen
        Si no:
          → pesos fijos (alpha_base, beta_base)

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

        1. Model A predice + Grad-CAM → attention ratio
        2. Model B predice (tissue-only multi-crop)
        3. Pesos dinámicos según attention ratio
        4. Combinar
        """
        if not self._loaded:
            raise RuntimeError("Modelos no cargados")

        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError("No se pudo cargar imagen")

        # Model A
        probs_a, tensor_a = self._predict_model_a_with_tensor(image)

        # Attention ratio
        attention_ratio = None
        if self.use_adaptive and self.model_b is not None:
            attention_ratio = self.compute_attention_ratio(tensor_a)

        # Model B
        if self.model_b is not None and self.beta_base > 0:
            probs_b = self._predict_model_b(image)
        else:
            probs_b = probs_a

        # Combinar
        probs, alpha_used, beta_used = self.combine_predictions(
            probs_a, probs_b, attention_ratio
        )

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        return {
            "class_idx": pred_class,
            "class_name": self.class_names.get(pred_class, str(pred_class)),
            "confidence": confidence,
            "probabilities": probs.tolist(),
            "model_a_probs": probs_a.tolist(),
            "model_b_probs": probs_b.tolist(),
            "attention_ratio": attention_ratio,
            "alpha_used": alpha_used,
            "beta_used": beta_used,
            "mode": "adaptive" if self.use_adaptive else "fixed",
        }

    def _predict_model_a_with_tensor(
        self, img_bgr: np.ndarray
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Predice con Model A y retorna también el tensor (para Grad-CAM)."""
        processed = self.preprocessor_a.process_image(img_bgr)
        img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=img_rgb)
        tensor = augmented["image"].unsqueeze(0).to(self.device)

        self.model_a.eval()
        with torch.no_grad():
            logits = self.model_a(tensor) / self.temp_a
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        return probs, tensor

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

        # Model A
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

        # Model B
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

        # Combinar con pesos fijos (batch)
        logger.info("  Combinando ensemble...")
        n = min(len(probs_a_all), len(probs_b_all))
        probs_a_all = probs_a_all[:n]
        probs_b_all = probs_b_all[:n]
        labels = np.array(test_labels[:n])

        ensemble_probs = self.alpha_base * probs_a_all + self.beta_base * probs_b_all
        preds = np.argmax(ensemble_probs, axis=1)

        try:
            auc = roc_auc_score(
                labels,
                ensemble_probs,
                multi_class="ovr",
                average="macro",
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

        # Model A probs
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

        # Model B probs
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
            self.model_b,
            crop_ds,
            criterion,
            self.device,
            self.temp_b,
        )
        probs_b = m_b["probabilities"]

        n = min(len(probs_a), len(probs_b))
        probs_a = probs_a[:n]
        probs_b = probs_b[:n]
        labels = np.array(val_labels[:n])

        # Grid search
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
                {
                    "alpha": round(alpha_c, 2),
                    "beta": round(beta_c, 2),
                    "score": score,
                }
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
