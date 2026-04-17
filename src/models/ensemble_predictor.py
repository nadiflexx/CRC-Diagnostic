"""
src/ml/ensemble_predictor.py

Adaptive Ensemble with Attention-Gating.

Strategy:
    - DYNAMIC per-image weights derived from Model A's Grad-CAM activations.
    - If Model A attends to tissue (high ratio) → trust Model A more.
    - If Model A attends to BLACK borders (low ratio) → trust Model B more.
    - If Model A attends to borders WITH tissue → trust Model A (do not
      penalise).
    - Smooth transition between regimes via a sigmoid function.
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
    Adaptive ensemble that combines Model A and Model B predictions.

    Weights are adjusted per image based on where Model A's Grad-CAM
    activations fall:
        - Attends to tissue → rely on Model A (useful contextual signal).
        - Attends to black borders → rely on Model B (Model A is using
          shortcuts).
        - Attends to borders containing tissue → rely on Model A (not a
          shortcut).

    The transition between regimes is smooth, governed by a sigmoid
    function parameterised by ``sigmoid_center`` and ``sigmoid_slope``.
    """

    def __init__(self, device: str | None = None):
        """
        Initialise the ensemble predictor with default hyperparameters.

        Args:
            device (str | None): Target device. If ``None``, CUDA is used
                when available, otherwise CPU.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_a: ColonCancerClassifier | None = None
        self.model_b: TissueOnlyClassifier | None = None

        self.temp_a = 1.0
        self.temp_b = 1.0

        # Base weights (modulated by attention at inference time)
        self.alpha_base = 0.5
        self.beta_base = 0.5

        # Adaptive sigmoid parameters
        self.sigmoid_center = 1.5
        self.sigmoid_slope = 3.0
        self.alpha_min = 0.20
        self.alpha_max = 0.80

        self.use_adaptive = True
        self.num_classes = 3
        self.class_names: dict[int, str] = {}
        self.class_mapping: dict = {}

        # Preprocessors
        self.preprocessor_a = MultiSourceStandardizer(target_size=384)
        self.preprocessor_b = TissueOnlyPreprocessor(target_size=384, n_crops=5)

        self.transform = get_val_transforms(384)

        self._loaded = False

    # ═════════════════════════════════════════════
    #  LOADING
    # ═════════════════════════════════════════════

    def load_models(self) -> bool:
        """
        Load Model A, Model B, and the ensemble configuration from disk.

        Attempts to load both models independently. If only Model A is
        available the ensemble falls back to fixed weights of
        ``alpha=1.0, beta=0.0`` with adaptive gating disabled.

        Returns:
            bool: ``True`` if at least Model A was loaded successfully,
                ``False`` if no model could be loaded.
        """
        success_a = self._load_model_a()
        success_b = self._load_model_b()
        self._load_ensemble_config()

        self._loaded = success_a and success_b

        if self._loaded:
            logger.info(
                f"  ✅ Adaptive ensemble loaded "
                f"(base α={self.alpha_base:.2f}, β={self.beta_base:.2f})"
            )
        elif success_a:
            logger.warning("  ⚠️ Only Model A available")
            self.alpha_base = 1.0
            self.beta_base = 0.0
            self.use_adaptive = False
            self._loaded = True
        else:
            logger.error("  ❌ No model could be loaded")

        return self._loaded

    @property
    def ensemble_available(self) -> bool:
        """
        Check whether both models are loaded and ready for ensemble inference.

        Returns:
            bool: ``True`` if both ``model_a`` and ``model_b`` are not
                ``None``.
        """
        return self.model_a is not None and self.model_b is not None

    def _load_model_a(self) -> bool:
        """
        Load Model A (primary classifier) from its checkpoint file.

        Reads ``paths.CLASSIFIER_CHECKPOINT``, restores model weights,
        and populates ``self.num_classes``, ``self.class_mapping``,
        ``self.class_names``, and ``self.temp_a`` from the checkpoint
        metadata.

        Returns:
            bool: ``True`` if the checkpoint was found and loaded
                successfully, ``False`` otherwise.
        """
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
        """
        Load Model B (tissue-only classifier) from its checkpoint file.

        Reads ``paths.TISSUE_CLASSIFIER_CHECKPOINT`` and restores model
        weights and temperature into ``self.model_b`` and ``self.temp_b``.

        Returns:
            bool: ``True`` if the checkpoint was found and loaded
                successfully, ``False`` otherwise.
        """
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
        """
        Load adaptive ensemble hyperparameters from the JSON config file.

        Reads ``paths.ENSEMBLE_CONFIG_PATH`` and updates base weights,
        sigmoid parameters, and temperature values. If the file does not
        exist or cannot be parsed the method returns silently, leaving
        all parameters at their default values.
        """
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
            logger.warning(f"  ⚠️ Config error: {e}")

    # ═════════════════════════════════════════════
    #  ATTENTION RATIO (content-aware)
    # ═════════════════════════════════════════════

    def compute_attention_ratio(
        self,
        tensor_a: torch.Tensor,
        preprocessed_bgr: np.ndarray | None = None,
    ) -> float:
        """
        Compute a content-aware attention ratio from Model A's Grad-CAM map.

        Analyses WHAT Model A is looking at (tissue vs. dark artifacts),
        not WHERE in the image it looks. The ratio is used to decide how
        much to trust Model A versus Model B for a given image.

        Algorithm:
            1. Generate a Grad-CAM activation map for the predicted class.
            2. Identify the top-10% high-attention pixels.
            3. Measure the brightness distribution of those pixels in the
               preprocessed grayscale image.
            4. Classify the attention content into one of four regimes:
               bright tissue (ratio ≈ 3.0), normal tissue (≈ 2.0),
               grey zone (0.7–1.2), or dark artifacts (≈ 0.3).
            5. Apply an additional penalty if more than 30% of attended
               pixels are near-black (< 20 intensity).
            6. Apply a further penalty if the point of maximum activation
               is near-black (< 30 intensity) but the overall ratio is
               still high.

        Args:
            tensor_a (torch.Tensor): Preprocessed input tensor already on
                the correct device, shape (1, C, H, W). Used to generate
                the Grad-CAM map via Model A.
            preprocessed_bgr (np.ndarray | None): BGR image array produced
                by ``preprocessor_a.process_image`` of shape (H, W, 3).
                If ``None`` a neutral ratio of 1.5 is returned immediately.

        Returns:
            float: Attention ratio value:
                - > 2.0: Model A attends to well-lit tissue → trust Model A.
                - ~ 1.0: Ambiguous zone → neutral weighting.
                - < 0.5: Model A attends to dark artifacts → trust Model B.
                Returns 1.5 if Grad-CAM fails or no attended pixels are
                found.
        """
        try:
            cam, _ = generate_gradcam(
                model=self.model_a,
                input_tensor=tensor_a,
                target_class=None,
                method="gradcam",
            )

            if preprocessed_bgr is None:
                logger.info("  📊 No BGR image → neutral ratio 1.5")
                return 1.5

            h_cam, w_cam = cam.shape
            threshold = np.percentile(cam, 90)
            hot_mask = cam >= threshold

            if hot_mask.sum() == 0:
                logger.info("  📊 No attention detected → neutral ratio 1.5")
                return 1.5

            gray = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2GRAY)
            h_img, w_img = gray.shape

            cam_resized = cv2.resize(
                cam, (w_img, h_img), interpolation=cv2.INTER_LINEAR
            )
            hot_mask_resized = cam_resized >= np.percentile(cam_resized, 90)

            attention_pixels = gray[hot_mask_resized]

            if len(attention_pixels) == 0:
                return 1.5

            mean_brightness = float(np.mean(attention_pixels))
            median_brightness = float(np.median(attention_pixels))
            p25_brightness = float(np.percentile(attention_pixels, 25))
            p75_brightness = float(np.percentile(attention_pixels, 75))

            img_p25 = np.percentile(gray, 25)
            img_p50 = np.percentile(gray, 50)
            img_p75 = np.percentile(gray, 75)

            logger.info(
                f"  📊 Image: p25={img_p25:.0f}, p50={img_p50:.0f}, p75={img_p75:.0f}"
            )
            logger.info(
                f"  📊 Attention: mean={mean_brightness:.0f}, "
                f"median={median_brightness:.0f}, "
                f"p25={p25_brightness:.0f}, p75={p75_brightness:.0f}"
            )

            if median_brightness > img_p75:
                ratio = 3.0
                logger.info(
                    f"  📊 ✅ BRIGHT TISSUE: median={median_brightness:.0f} "
                    f"> p75={img_p75:.0f} → ratio={ratio:.2f}"
                )

            elif median_brightness > img_p50:
                ratio = 2.0
                logger.info(
                    f"  📊 ✅ NORMAL TISSUE: median={median_brightness:.0f} "
                    f"> p50={img_p50:.0f} → ratio={ratio:.2f}"
                )

            elif median_brightness > img_p25:
                dark_pixels = np.sum(attention_pixels < img_p25)
                dark_ratio = dark_pixels / len(attention_pixels)

                if dark_ratio > 0.5:
                    ratio = 0.7
                    logger.info(
                        f"  📊 ⚠️  GREY ZONE with {dark_ratio:.0%} dark "
                        f"→ ratio={ratio:.2f}"
                    )
                else:
                    ratio = 1.2
                    logger.info(f"  📊 ⚠️  MIXED GREY ZONE → ratio={ratio:.2f}")

            else:
                ratio = 0.3
                logger.info(
                    f"  📊 🚨 DARK ARTIFACT: median={median_brightness:.0f} "
                    f"< p25={img_p25:.0f} → ratio={ratio:.2f}"
                )

            very_dark_pixels = np.sum(attention_pixels < 20)
            very_dark_ratio = very_dark_pixels / len(attention_pixels)

            if very_dark_ratio > 0.3:
                penalty = 0.5
                ratio *= penalty
                logger.info(
                    f"  📊 🚨 {very_dark_ratio:.0%} attention on PURE BLACK "
                    f"(<20) → penalty ×{penalty} → ratio={ratio:.2f}"
                )

            max_pos = np.unravel_index(cam.argmax(), cam.shape)
            max_y = int(max_pos[0] / h_cam * h_img)
            max_x = int(max_pos[1] / w_cam * w_img)

            y1, y2 = max(0, max_y - 5), min(h_img, max_y + 6)
            x1, x2 = max(0, max_x - 5), min(w_img, max_x + 6)
            max_point_brightness = float(gray[y1:y2, x1:x2].mean())

            logger.info(
                f"  📊 Max point: brightness={max_point_brightness:.0f} "
                f"at ({max_x}, {max_y})"
            )

            if max_point_brightness < 30 and ratio > 1.0:
                penalty = 0.6
                ratio *= penalty
                logger.info(
                    f"  📊 🚨 Maximum at BLACK ({max_point_brightness:.0f}<30) "
                    f"→ penalty ×{penalty} → ratio={ratio:.2f}"
                )

            return float(ratio)

        except Exception as e:
            logger.warning(f"  ⚠️ Attention ratio failed: {e}")
        return 1.5

    # ═════════════════════════════════════════════
    #  ADAPTIVE COMBINATION
    # ═════════════════════════════════════════════

    def compute_adaptive_weights(self, attention_ratio: float) -> tuple[float, float]:
        """
        Derive dynamic per-image ensemble weights from the attention ratio.

        Maps the attention ratio through a sigmoid function centred at
        ``sigmoid_center`` with steepness ``sigmoid_slope``, then linearly
        scales the result into [``alpha_min``, ``alpha_max``].

        Reference values (default parameters):
            - ratio = 3.0 → α ≈ 0.75, β ≈ 0.25  (trust Model A).
            - ratio = 1.3 → α ≈ 0.50, β ≈ 0.50  (neutral).
            - ratio = 0.3 → α ≈ 0.26, β ≈ 0.74  (trust Model B).

        Args:
            attention_ratio (float): Content-aware attention ratio as
                returned by ``compute_attention_ratio``.

        Returns:
            tuple[float, float]: ``(alpha, beta)`` where
                ``alpha + beta = 1.0``. Both values are rounded to three
                decimal places.
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
        Linearly combine Model A and Model B probability vectors.

        If adaptive mode is enabled and an attention ratio is provided the
        weights are computed dynamically via ``compute_adaptive_weights``.
        Otherwise the fixed base weights ``alpha_base`` and ``beta_base``
        are used.

        Args:
            probs_a (np.ndarray): Softmax probability vector from Model A,
                shape (num_classes,).
            probs_b (np.ndarray): Softmax probability vector from Model B,
                shape (num_classes,).
            attention_ratio (float | None): Attention ratio from
                ``compute_attention_ratio``. If ``None`` or adaptive mode
                is disabled, fixed base weights are applied.

        Returns:
            tuple[np.ndarray, float, float]:
                - Combined probability vector of shape (num_classes,).
                - ``alpha`` weight actually applied to Model A.
                - ``beta`` weight actually applied to Model B.
        """
        if self.use_adaptive and attention_ratio is not None:
            alpha, beta = self.compute_adaptive_weights(attention_ratio)
        else:
            alpha = self.alpha_base
            beta = self.beta_base

        combined = alpha * probs_a + beta * probs_b
        return combined, alpha, beta

    # ═════════════════════════════════════════════
    #  FULL PREDICTION
    # ═════════════════════════════════════════════

    def predict(self, image: np.ndarray | str | Path) -> dict:
        """
        Run the full adaptive ensemble inference pipeline on a single image.

        Steps:
            1. Load the image from disk if a path is provided.
            2. Obtain Model A probabilities and the preprocessed tensor.
            3. Compute the content-aware attention ratio via Grad-CAM.
            4. Obtain Model B probabilities via multi-crop inference.
            5. Compute adaptive ensemble weights from the attention ratio.
            6. Combine both probability vectors and produce the final
               prediction.

        Args:
            image (np.ndarray | str | Path): Input image as a BGR NumPy
                array or a file path. If a path is given it is loaded with
                ``cv2.imread``.

        Returns:
            dict: Prediction results with the following keys:
                - ``"class_idx"`` (int): Predicted class index.
                - ``"class_name"`` (str): Human-readable predicted class
                  name.
                - ``"confidence"`` (float): Probability of the predicted
                  class.
                - ``"probabilities"`` (list[float]): Combined probability
                  vector.
                - ``"model_a_probs"`` (list[float]): Model A probabilities.
                - ``"model_b_probs"`` (list[float]): Model B probabilities.
                - ``"attention_ratio"`` (float | None): Computed attention
                  ratio, or ``None`` if adaptive mode is disabled.
                - ``"alpha_used"`` (float): Model A weight used.
                - ``"beta_used"`` (float): Model B weight used.
                - ``"mode"`` (str): ``"adaptive"`` or ``"fixed"``.

        Raises:
            RuntimeError: If ``load_models`` has not been called before
                ``predict``.
            ValueError: If the provided file path cannot be read by
                ``cv2.imread``.
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError("Could not load image.")

        probs_a, tensor_a, preprocessed_a = self._predict_model_a_with_tensor(image)

        attention_ratio = None
        if self.use_adaptive and self.model_b is not None:
            attention_ratio = self.compute_attention_ratio(tensor_a, preprocessed_a)

        if self.model_b is not None and self.beta_base > 0:
            probs_b = self._predict_model_b(image)
        else:
            probs_b = probs_a

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
        Run Model A on a BGR image and return probabilities, tensor, and
        the preprocessed image.

        The image is standardised by ``preprocessor_a``, converted to RGB,
        transformed to a tensor, and forwarded through Model A with
        temperature scaling.

        Args:
            img_bgr (np.ndarray): Raw BGR input image of shape (H, W, 3).

        Returns:
            tuple[np.ndarray, torch.Tensor, np.ndarray]:
                - Softmax probability vector of shape (num_classes,).
                - Input tensor on the target device, shape (1, C, H, W).
                  Required for subsequent Grad-CAM computation.
                - Preprocessed BGR image of shape (H, W, 3) used for
                  brightness analysis in ``compute_attention_ratio``.
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
        """
        Run Model B on a BGR image using multi-crop averaging.

        The image is processed by ``preprocessor_b`` to produce ``n_crops``
        tissue-only crops. Each crop is forwarded independently through
        Model B with temperature scaling, and the resulting probability
        vectors are averaged.

        Args:
            img_bgr (np.ndarray): Raw BGR input image of shape (H, W, 3).

        Returns:
            np.ndarray: Averaged softmax probability vector of shape
                (num_classes,).
        """
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
    #  BATCH PREDICTION (for evaluation)
    # ═════════════════════════════════════════════

    def predict_batch_from_preprocessed(
        self,
        test_paths: list[str],
        test_labels: list[int],
        image_size: int = 384,
        batch_size: int = 16,
        n_crops: int = 5,
    ) -> dict:
        """
        Run batch ensemble inference with fixed base weights.

        Designed for fast evaluation over a pre-processed dataset. Does not
        compute Grad-CAM or adaptive weights; uses ``alpha_base`` and
        ``beta_base`` directly.

        Model A predictions are obtained from a standard ``ColonoscopyDataset``
        loader. Model B predictions are obtained via multi-crop averaging
        through ``validate_multicrop``.

        Args:
            test_paths (list[str]): Absolute paths to the test images.
            test_labels (list[int]): Ground-truth integer labels aligned
                with ``test_paths``.
            image_size (int): Spatial resolution used when building
                datasets. Default is 384.
            batch_size (int): Number of samples per forward-pass batch for
                Model A. Model B uses ``batch_size * 2``. Default is 16.
            n_crops (int): Number of tissue-only crop variants per image
                for Model B. Default is 5.

        Returns:
            dict: Evaluation metrics and raw outputs with the following
                keys:
                - ``"accuracy"`` (float): Overall accuracy.
                - ``"f1"`` (float): Macro-averaged F1 score.
                - ``"precision"`` (float): Macro-averaged precision.
                - ``"recall"`` (float): Macro-averaged recall.
                - ``"auc"`` (float): Macro one-vs-rest ROC AUC.
                - ``"predictions"`` (np.ndarray): Predicted class per image.
                - ``"labels"`` (np.ndarray): Ground-truth labels.
                - ``"probabilities"`` (np.ndarray): Combined probability
                  matrix of shape (N, num_classes).
                - ``"probs_a"`` (np.ndarray): Model A probability matrix.
                - ``"probs_b"`` (np.ndarray): Model B probability matrix.
        """
        logger.info("  Model A predictions...")
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
            logger.info("  Model B predictions (multi-crop)...")

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

        logger.info("  Combining ensemble...")
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
    #  OPTIMISATION
    # ═════════════════════════════════════════════

    def optimize_weights(
        self,
        val_paths: list[str],
        val_labels: list[int],
        image_size: int = 384,
        n_crops: int = 5,
        metric: str = "f1",
    ) -> tuple[float, float]:
        """
        Find the optimal fixed base weights via grid search on the
        validation set.

        Evaluates all combinations of ``alpha`` in [0.0, 1.0] with step
        0.05, selecting the pair that maximises the specified metric. The
        best weights are stored in ``self.alpha_base`` and
        ``self.beta_base`` and persisted to disk via
        ``_save_ensemble_config``.

        Args:
            val_paths (list[str]): Absolute paths to validation images.
            val_labels (list[int]): Ground-truth labels aligned with
                ``val_paths``.
            image_size (int): Spatial resolution used for the validation
                datasets. Default is 384.
            n_crops (int): Number of tissue-only crop variants per image
                for Model B. Default is 5.
            metric (str): Optimisation target. One of ``"f1"``,
                ``"recall"``, or ``"accuracy"``. Default is ``"f1"``.

        Returns:
            tuple[float, float]: ``(alpha_base, beta_base)`` — the optimal
                base weights found by grid search.
        """
        logger.info("\n═══ OPTIMISING BASE WEIGHTS ═══")

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
            f"\n  ✅ OPTIMAL: α_base={self.alpha_base:.2f}, β_base={self.beta_base:.2f}"
        )

        self._save_ensemble_config()
        return self.alpha_base, self.beta_base

    def _save_ensemble_config(self):
        """
        Persist the current ensemble configuration to a JSON file.

        Writes all adaptive parameters (base weights, sigmoid settings,
        temperature values, and checkpoint file names) to
        ``paths.ENSEMBLE_CONFIG_PATH``. The parent directory is created
        if it does not exist.
        """
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

        logger.info(f"  💾 Config saved: {config_path}")
