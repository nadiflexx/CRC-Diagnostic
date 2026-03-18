"""
Multi-source standardizer for colonoscopy images.
Processes image + mask together with exact same geometry.
"""

import shutil

import cv2
import numpy as np
from tqdm import tqdm

from src.config.constants import IMAGE_GLOB_PATTERNS
from src.config.paths import paths


class MultiSourceStandardizer:
    """
    Standardizer for processing multiple image sources with consistent geometry.
    """

    def __init__(self, target_size: int = 384):
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def suppress_green(self, img: np.ndarray) -> np.ndarray:
        """
        Suppress green artifacts in the image.

        :param img: Input image.
        :return: Image with green artifacts suppressed.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        total_pixels = h * w
        green_lower = np.array([30, 50, 50])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_ratio = green_mask.sum() / (255.0 * total_pixels)
        if green_ratio < 0.003:
            return img
        min_area = int(total_pixels * 0.003)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            green_mask, connectivity=8
        )
        filtered_mask = np.zeros_like(green_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered_mask[labels == i] = 255
        if filtered_mask.sum() == 0:
            return img
        filtered_mask = cv2.dilate(
            filtered_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=2,
        )
        result = img.copy()
        result[filtered_mask > 0] = 0
        return result

    def detect_fov_type(self, img: np.ndarray) -> str:
        """
        Detect the type of field of view in the image.

        :param img: Input image.
        :return: Type of field of view ("circular" or "rectangular").
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cs_h, cs_w = max(h // 10, 5), max(w // 10, 5)
        corners = [
            gray[:cs_h, :cs_w],
            gray[:cs_h, w - cs_w :],
            gray[h - cs_h :, :cs_w],
            gray[h - cs_h :, w - cs_w :],
        ]
        dark = sum(1 for c in corners if np.mean(c) < 15)
        return "circular" if dark >= 3 else "rectangular"

    def _get_circular_crop_coords(self, img):
        """
        Get coordinates for circular cropping.

        :param img: Input image.
        :return: Coordinates for cropping (y1, y2, x1, x2).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.morph_kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.morph_kernel)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        cx, cy, radius = int(cx), int(cy), int(radius)
        if radius < 50:
            return None
        half_side = int(radius * 0.68)
        y1, y2 = max(0, cy - half_side), min(h, cy + half_side)
        x1, x2 = max(0, cx - half_side), min(w, cx + half_side)
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            return None
        return y1, y2, x1, x2

    def _get_rectangular_crop_coords(self, img):
        """
        Get coordinates for rectangular cropping.

        :param img: Input image.
        :return: Coordinates for cropping (y1, y2, x1, x2).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.morph_kernel)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(c)
        area_ratio = (cw * ch) / (w * h)
        if area_ratio < 0.10 or area_ratio > 0.95:
            return None
        my, mx = int(ch * 0.05), int(cw * 0.05)
        y1, y2 = y + my, y + ch - my
        x1, x2 = x + mx, x + cw - mx
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            return None
        return y1, y2, x1, x2

    def _get_crop_coords(self, img):
        """
        Get coordinates for image cropping.

        :param img: Input image.
        :return: Coordinates for cropping (y1, y2, x1, x2).
        """
        fov = self.detect_fov_type(img)
        if fov == "circular":
            return self._get_circular_crop_coords(img)
        return self._get_rectangular_crop_coords(img)

    def extract_tissue(self, img: np.ndarray) -> np.ndarray:
        """
        Extract the tissue region from the image.

        :param img: Input image.
        :return: Tissue region.
        """
        coords = self._get_crop_coords(img)
        if coords:
            y1, y2, x1, x2 = coords
            return img[y1:y2, x1:x2]
        return img

    def standardize_geometry(self, tissue: np.ndarray) -> np.ndarray:
        """
        Standardize the geometry of the tissue region.

        :param tissue: Tissue region.
        :return: Standardized tissue region.
        """
        h, w = tissue.shape[:2]
        if h != w:
            size = max(h, w)
            canvas = np.zeros((size, size, 3), dtype=np.uint8)
            yo, xo = (size - h) // 2, (size - w) // 2
            canvas[yo : yo + h, xo : xo + w] = tissue
            tissue = canvas
        return cv2.resize(
            tissue,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LANCZOS4,
        )

    def _standardize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Standardize the geometry of the mask.

        :param mask: Input mask.
        :return: Standardized mask.
        """
        h, w = mask.shape[:2]
        if h != w:
            size = max(h, w)
            canvas = np.zeros((size, size), dtype=np.uint8)
            yo, xo = (size - h) // 2, (size - w) // 2
            canvas[yo : yo + h, xo : xo + w] = mask
            mask = canvas
        return cv2.resize(
            mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST
        )

    def normalize_illumination(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize the illumination in the image.

        :param img: Input image.
        :return: Normalized image.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 5:
            return img
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        L = self.clahe.apply(L)
        return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)

    def process_image(self, img: np.ndarray) -> np.ndarray:
        """
        Process the image for standardization.

        :param img: Input image.
        :return: Processed image.
        """
        clean = self.suppress_green(img)
        tissue = self.extract_tissue(clean)
        standard = self.standardize_geometry(tissue)
        return self.normalize_illumination(standard)

    def process_image_and_mask(self, img, mask_gray):
        """
        Process the image and mask for standardization.

        :param img: Input image.
        :param mask_gray: Input mask.
        :return: Processed image and mask.
        """
        clean = self.suppress_green(img)
        coords = self._get_crop_coords(clean)
        if coords:
            y1, y2, x1, x2 = coords
            tissue_img = clean[y1:y2, x1:x2]
            tissue_mask = mask_gray[y1:y2, x1:x2]
        else:
            tissue_img = clean
            tissue_mask = mask_gray
        std_img = self.standardize_geometry(tissue_img)
        std_mask = self._standardize_mask(tissue_mask)
        final_img = self.normalize_illumination(std_img)
        return final_img, std_mask


def process_dataset(clean_dir=None, processed_dir=None, target_size=384):
    """
    Process the dataset for standardization.

    :param clean_dir: Path to the cleaned dataset directory.
    :param processed_dir: Path to the processed dataset directory.
    :param target_size: Target size for standardization.
    """
    clean_dir = clean_dir or paths.COLON_CLEAN
    processed_dir = processed_dir or paths.COLON_PROCESSED

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    mapping_src = clean_dir / "class_mapping.json"
    if mapping_src.exists():
        shutil.copy2(mapping_src, processed_dir / "class_mapping.json")

    preprocessor = MultiSourceStandardizer(target_size=target_size)

    mask_input_dir = clean_dir / "masks"
    available_masks = {}
    if mask_input_dir.exists():
        for f in mask_input_dir.iterdir():
            if f.is_file():
                available_masks[f.stem] = f

    mask_output_dir = processed_dir / "masks"
    if available_masks:
        mask_output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "failed": 0, "skipped": 0, "masks": 0}

    for class_dir in sorted(clean_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name == "masks":
            continue

        save_dir = processed_dir / class_dir.name
        save_dir.mkdir(parents=True, exist_ok=True)

        images = []
        for pattern in IMAGE_GLOB_PATTERNS:
            images.extend(class_dir.glob(pattern))
        images = sorted(images)

        for img_path in tqdm(images, desc=f"  {class_dir.name}"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    stats["skipped"] += 1
                    continue

                mask_file = available_masks.get(img_path.stem)
                mask = None
                if mask_file:
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

                if mask is not None:
                    processed, mask_processed = preprocessor.process_image_and_mask(
                        img, mask
                    )
                    cv2.imwrite(
                        str(mask_output_dir / (img_path.stem + ".png")), mask_processed
                    )
                    stats["masks"] += 1
                else:
                    processed = preprocessor.process_image(img)

                cv2.imwrite(
                    str(save_dir / (img_path.stem + ".jpg")),
                    processed,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                stats["processed"] += 1

            except Exception as e:
                print(f"  Error: {img_path.name}: {e}")
                stats["failed"] += 1

    print(f"Result: {stats['processed']} processed, {stats['masks']} masks aligned")


if __name__ == "__main__":
    process_dataset()
