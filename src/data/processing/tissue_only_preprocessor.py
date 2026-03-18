"""
Tissue-only preprocessor: SLIDE → SHRINK → RESIZE direct.
Zero padding, zero border artifacts.
"""

import shutil

import cv2
import numpy as np
from tqdm import tqdm

from src.config.constants import IMAGE_GLOB_PATTERNS
from src.config.paths import paths


class TissueOnlyPreprocessor:
    def __init__(
        self,
        target_size=384,
        black_threshold=15,
        max_black_pct=0.5,
        n_crops=5,
        min_tissue_ratio=0.80,
        shrink_step=0.03,
    ):
        self.target_size = target_size
        self.black_threshold = black_threshold
        self.max_black_pct = max_black_pct
        self.n_crops = n_crops
        self.min_tissue_ratio = min_tissue_ratio
        self.shrink_step = shrink_step
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def suppress_green(self, img):
        """
        Suppress green
        :param img:
        :return:
        """

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        total_pixels = h * w
        green_mask = cv2.inRange(hsv, np.array([30, 50, 50]), np.array([85, 255, 255]))
        if green_mask.sum() / (255.0 * total_pixels) < 0.003:
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

    def find_clean_square_crop(self, img):
        """
        Find clean square crop
        :param img:
        :return:
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        black_mask = gray < self.black_threshold
        total_black = black_mask.sum()
        if total_black == 0:
            side = min(h, w)
            y, x = (h - side) // 2, (w - side) // 2
            return img[y : y + side, x : x + side]
        tissue_ratio = 1.0 - total_black / gray.size
        if tissue_ratio < 0.05:
            side = min(h, w, 200)
            y, x = (h - side) // 2, (w - side) // 2
            return img[y : y + side, x : x + side]
        integral = cv2.integral(black_mask.astype(np.float64))
        side = min(h, w)
        min_side = max(100, self.target_size // 4)
        best_y, best_x, best_side = 0, 0, side
        while side >= min_side:
            result = self._search_best_position(integral, h, w, side)
            if result is None:
                side = int(side * (1 - self.shrink_step))
                continue
            pos_y, pos_x, black_count = result
            if black_count == 0:
                return img[pos_y : pos_y + side, pos_x : pos_x + side]
            pct = black_count / (side * side) * 100
            if pct <= self.max_black_pct:
                return img[pos_y : pos_y + side, pos_x : pos_x + side]
            best_y, best_x, best_side = pos_y, pos_x, side
            side = int(side * (1 - self.shrink_step))
        return img[best_y : best_y + best_side, best_x : best_x + best_side]

    def _search_best_position(self, integral, h, w, side):
        """
        Search best position
        :param integral:
        :param h:
        :param w:
        :param side:
        :return:
        """
        max_y, max_x = h - side, w - side
        if max_y < 0 or max_x < 0:
            return None
        coarse_step = max(1, max(max_y, max_x) // 30)
        ys = np.arange(0, max_y + 1, coarse_step)
        xs = np.arange(0, max_x + 1, coarse_step)
        if len(ys) > 0 and ys[-1] != max_y:
            ys = np.append(ys, max_y)
        if len(xs) > 0 and xs[-1] != max_x:
            xs = np.append(xs, max_x)
        if len(ys) == 0:
            ys = np.array([0])
        if len(xs) == 0:
            xs = np.array([0])
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        counts = (
            integral[Y + side, X + side]
            - integral[Y, X + side]
            - integral[Y + side, X]
            + integral[Y, X]
        )
        min_idx = np.argmin(counts)
        yi, xi = np.unravel_index(min_idx, counts.shape)
        coarse_y, coarse_x = int(ys[yi]), int(xs[xi])
        coarse_count = float(counts[yi, xi])
        if coarse_count == 0:
            return coarse_y, coarse_x, 0.0
        fine_ys = np.arange(
            max(0, coarse_y - coarse_step), min(max_y + 1, coarse_y + coarse_step + 1)
        )
        fine_xs = np.arange(
            max(0, coarse_x - coarse_step), min(max_x + 1, coarse_x + coarse_step + 1)
        )
        if len(fine_ys) == 0 or len(fine_xs) == 0:
            return coarse_y, coarse_x, coarse_count
        FY, FX = np.meshgrid(fine_ys, fine_xs, indexing="ij")
        fine_counts = (
            integral[FY + side, FX + side]
            - integral[FY, FX + side]
            - integral[FY + side, FX]
            + integral[FY, FX]
        )
        fine_min = np.argmin(fine_counts)
        fyi, fxi = np.unravel_index(fine_min, fine_counts.shape)
        return int(fine_ys[fyi]), int(fine_xs[fxi]), float(fine_counts[fyi, fxi])

    def generate_multi_crops(self, clean_square):
        """
        Generate multi crops
        :param clean_square:
        :return:
        """
        h, w = clean_square.shape[:2]
        side = min(h, w)
        if side < 150:
            return [clean_square.copy() for _ in range(self.n_crops)]
        crop_size = max(100, int(side * 0.88))
        max_offset = side - crop_size
        if max_offset <= 0:
            return [clean_square.copy() for _ in range(self.n_crops)]
        crops = []
        margin = max_offset // 2
        center = clean_square[margin : margin + crop_size, margin : margin + crop_size]
        crops.append(center)
        for _ in range(self.n_crops - 1):
            y = np.random.randint(0, max_offset + 1)
            x = np.random.randint(0, max_offset + 1)
            crop = clean_square[y : y + crop_size, x : x + crop_size]
            if self._is_valid_crop(crop):
                crops.append(crop)
        while len(crops) < self.n_crops:
            crops.append(center.copy())
        return crops[: self.n_crops]

    def _is_valid_crop(self, crop):
        """
        Is valid crop
        :param crop:
        :return:
        """
        if crop.size == 0:
            return False
        h, w = crop.shape[:2]
        if h < 100 or w < 100:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if (gray > self.black_threshold).sum() / gray.size < self.min_tissue_ratio:
            return False
        return np.var(gray) >= 100

    def resize_to_target(self, crop):
        """
        Resize to target
        :param crop:
        :return:
        """
        return cv2.resize(
            crop, (self.target_size, self.target_size), interpolation=cv2.INTER_LANCZOS4
        )

    def normalize_illumination(self, img):
        """
        Normalize illumination
        :param img:
        :return:
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 5:
            return img
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        L = self.clahe.apply(L)
        return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)

    def process_image(self, img):
        """
        Process image
        :param img:
        :return:
        """
        clean = self.suppress_green(img)
        clean_square = self.find_clean_square_crop(clean)
        crops = self.generate_multi_crops(clean_square)
        processed = []
        for crop in crops:
            resized = self.resize_to_target(crop)
            processed.append(self.normalize_illumination(resized))
        return processed

    def process_single(self, img):
        """
        Process single
        :param img:
        :return:
        """
        clean = self.suppress_green(img)
        clean_square = self.find_clean_square_crop(clean)
        resized = self.resize_to_target(clean_square)
        return self.normalize_illumination(resized)


def process_dataset_tissue_only(
    clean_dir=None, processed_dir=None, target_size=384, n_crops=5, max_black_pct=0.5
):
    """
    Process dataset tissue only
    :param clean_dir:
    :param processed_dir:
    :param target_size:
    :param n_crops:
    :param max_black_pct:
    :return:
    """
    clean_dir = clean_dir or paths.COLON_CLEAN
    processed_dir = processed_dir or paths.COLON_TISSUE_ONLY
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    mapping_src = clean_dir / "class_mapping.json"
    if mapping_src.exists():
        shutil.copy2(mapping_src, processed_dir / "class_mapping.json")
    preprocessor = TissueOnlyPreprocessor(
        target_size=target_size, max_black_pct=max_black_pct, n_crops=n_crops
    )
    stats = {"processed": 0, "failed": 0, "skipped": 0, "total_crops": 0}
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
                processed_crops = preprocessor.process_image(img)
                for crop_idx, crop in enumerate(processed_crops):
                    out_name = f"{img_path.stem}_crop{crop_idx}.jpg"
                    cv2.imwrite(
                        str(save_dir / out_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]
                    )
                    stats["total_crops"] += 1
                stats["processed"] += 1
            except Exception as e:
                print(f"  ⚠️ Error {img_path.name}: {e}")
                stats["failed"] += 1
    print(
        f"Result: {stats['processed']} processed, {stats['total_crops']} crops, {stats['failed']} failed"
    )


if __name__ == "__main__":
    process_dataset_tissue_only(target_size=384, n_crops=5, max_black_pct=0.5)
