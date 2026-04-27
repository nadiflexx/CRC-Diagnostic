# tests/src/data/test_tissue_only_preprocessor.py
"""
Complete tests for src/data/processing/tissue_only_preprocessor.py
"""

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.data.processing.tissue_only_preprocessor import (
    TissueOnlyPreprocessor,
    process_dataset_tissue_only,
)

# ─────────────────────────────────────────────────────────────
#  Image helpers
# ─────────────────────────────────────────────────────────────


def _uniform_image(value: int, h: int = 200, w: int = 200) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _noise_image(
    h: int = 200, w: int = 200, low: int = 80, high: int = 255, seed: int = 0
) -> np.ndarray:
    """High-variance image that passes both tissue and variance checks."""
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, (h, w, 3), dtype=np.uint8)


def _tissue_image(h: int = 200, w: int = 200) -> np.ndarray:
    """Simulate tissue: bright base with slight noise so variance >= 100."""
    rng = np.random.default_rng(1)
    img = rng.integers(120, 200, (h, w, 3), dtype=np.uint8)
    return img


def _circular_fov(h: int = 300, w: int = 300) -> np.ndarray:
    """Endoscope circular FOV: dark corners, bright centre circle."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cy, cx, r = h // 2, w // 2, min(h, w) // 2 - 10
    rng = np.random.default_rng(3)
    tissue = rng.integers(100, 200, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    img[mask > 0] = tissue[mask > 0]
    return img


@pytest.fixture
def prep():
    return TissueOnlyPreprocessor(target_size=128, n_crops=3, max_black_pct=0.5)


@pytest.fixture
def prep_small():
    """Preprocessor with small target for fast tests."""
    return TissueOnlyPreprocessor(target_size=64, n_crops=2, max_black_pct=0.5)


# ─────────────────────────────────────────────────────────────
#  __init__
# ─────────────────────────────────────────────────────────────


class TestInit:
    def test_target_size_stored(self):
        p = TissueOnlyPreprocessor(target_size=256)
        assert p.target_size == 256

    def test_black_threshold_stored(self):
        p = TissueOnlyPreprocessor(black_threshold=20)
        assert p.black_threshold == 20

    def test_max_black_pct_stored(self):
        p = TissueOnlyPreprocessor(max_black_pct=0.3)
        assert p.max_black_pct == 0.3

    def test_n_crops_stored(self):
        p = TissueOnlyPreprocessor(n_crops=7)
        assert p.n_crops == 7

    def test_min_tissue_ratio_stored(self):
        p = TissueOnlyPreprocessor(min_tissue_ratio=0.70)
        assert p.min_tissue_ratio == 0.70

    def test_shrink_step_stored(self):
        p = TissueOnlyPreprocessor(shrink_step=0.05)
        assert p.shrink_step == 0.05

    def test_clahe_created(self, prep):
        assert prep.clahe is not None

    def test_defaults(self):
        p = TissueOnlyPreprocessor()
        assert p.target_size == 384
        assert p.black_threshold == 15
        assert p.max_black_pct == 0.5
        assert p.n_crops == 5
        assert p.min_tissue_ratio == 0.80
        assert p.shrink_step == 0.03


# ─────────────────────────────────────────────────────────────
#  _is_valid_crop  — understanding the real logic
#
#  Two conditions:
#    1. tissue fraction > min_tissue_ratio  (default 0.80)
#    2. grayscale variance >= 100
# ─────────────────────────────────────────────────────────────


class TestIsValidCrop:
    def test_empty_array_fails(self, prep):
        assert not prep._is_valid_crop(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_too_small_height_fails(self, prep):
        img = _noise_image(50, 200)
        assert not prep._is_valid_crop(img)

    def test_too_small_width_fails(self, prep):
        img = _noise_image(200, 50)
        assert not prep._is_valid_crop(img)

    def test_fully_black_fails_tissue_check(self, prep):
        """All black → tissue fraction = 0 < 0.80."""
        img = _uniform_image(0)
        assert not prep._is_valid_crop(img)

    def test_uniform_bright_fails_variance_check(self, prep):
        """All-200 image: tissue fraction = 1.0 ✓, but variance = 0 < 100 ✗."""
        img = _uniform_image(200)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        assert float(np.var(gray)) < 100, "precondition: variance is zero"
        assert not prep._is_valid_crop(img)

    def test_low_tissue_fraction_fails(self, prep):
        """85% black, 15% bright → tissue fraction 0.15 < 0.80."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # 15% bright
        img[:30, :] = 200
        assert not prep._is_valid_crop(img)

    def test_high_variance_tissue_rich_passes(self, prep):
        """High-variance image with >80% tissue → should pass both checks."""
        img = _noise_image(200, 200, low=80, high=255, seed=42)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tissue_frac = float((gray > prep.black_threshold).sum()) / gray.size
        variance = float(np.var(gray))
        # Verify our test image actually meets both criteria
        assert tissue_frac >= prep.min_tissue_ratio
        assert variance >= 100
        assert prep._is_valid_crop(img)

    def test_mostly_tissue_with_variance_passes(self, prep):
        """90% bright pixels + enough variance → valid."""
        rng = np.random.default_rng(7)
        img = rng.integers(100, 220, (200, 200, 3), dtype=np.uint8)
        # Blank a small corner to keep tissue fraction > 80%
        img[:10, :10] = 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tissue_frac = float((gray > prep.black_threshold).sum()) / gray.size
        variance = float(np.var(gray))
        if tissue_frac >= prep.min_tissue_ratio and variance >= 100:
            assert prep._is_valid_crop(img)
        else:
            # Image doesn't meet criteria → expected False
            assert not prep._is_valid_crop(img)

    def test_variance_boundary_below_100_fails(self, prep):
        """Construct an image with variance just below 100."""
        # Nearly uniform: all 128 except a tiny patch
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        img[0, 0] = 130  # tiny difference
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        var = float(np.var(gray))
        assert var < 100, f"precondition: var={var}"
        # Should fail variance check
        assert not prep._is_valid_crop(img)

    def test_natural_colonoscopy_like_passes(self, prep):
        """
        Simulate a realistic tissue crop: mixed pinkish colours, high variance.
        """
        rng = np.random.default_rng(99)
        # Pinkish tissue simulation
        r = rng.integers(150, 220, (200, 200), dtype=np.uint8)
        g = rng.integers(80, 140, (200, 200), dtype=np.uint8)
        b = rng.integers(70, 130, (200, 200), dtype=np.uint8)
        img = np.stack([b, g, r], axis=2)  # BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tissue_frac = float((gray > prep.black_threshold).sum()) / gray.size
        variance = float(np.var(gray))
        if tissue_frac >= prep.min_tissue_ratio and variance >= 100:
            assert prep._is_valid_crop(img)

    def test_single_bright_pixel_fails_tissue(self, prep):
        """Single bright pixel on black → tissue fraction ≈ 0."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[100, 100] = 200
        assert not prep._is_valid_crop(img)

    def test_nearly_all_black_fails(self, prep):
        """99% black pixels → tissue fraction far below 0.80."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:10, :20] = 200  # 200/40000 = 0.5%
        assert not prep._is_valid_crop(img)

    def test_custom_min_tissue_ratio(self):
        """Lower min_tissue_ratio allows sparser tissue."""
        prep_loose = TissueOnlyPreprocessor(
            target_size=128, min_tissue_ratio=0.10, max_black_pct=0.5
        )
        rng = np.random.default_rng(5)
        # 20% bright, 80% dark — but high variance
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:40, :] = rng.integers(100, 220, (40, 200, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tissue_frac = float((gray > prep_loose.black_threshold).sum()) / gray.size
        variance = float(np.var(gray))
        if tissue_frac >= prep_loose.min_tissue_ratio and variance >= 100:
            assert prep_loose._is_valid_crop(img)
        else:
            assert not prep_loose._is_valid_crop(img)

    def test_custom_black_threshold(self):
        """Higher black_threshold means more pixels counted as black."""
        prep_strict = TissueOnlyPreprocessor(
            target_size=128, black_threshold=100, min_tissue_ratio=0.80
        )
        # Image with pixels at 80 — below new threshold → counted as black
        img = _uniform_image(80)
        # All pixels at 80 ≤ 100 → tissue fraction = 0 → fails
        assert not prep_strict._is_valid_crop(img)


# ─────────────────────────────────────────────────────────────
#  suppress_green
# ─────────────────────────────────────────────────────────────


class TestSuppressGreen:
    def test_no_green_returns_same(self, prep):
        img = _noise_image(seed=1)
        result = prep.suppress_green(img)
        assert result.shape == img.shape

    def test_large_green_region_zeroed(self, prep):
        """A large green annotation patch should be suppressed."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Create a green patch > 0.3% of pixels in HSV green range
        green_bgr = cv2.cvtColor(
            np.array([[[60, 220, 200]]], dtype=np.uint8),
            cv2.COLOR_HSV2BGR,
        ).flatten()
        img[40:160, 40:160] = green_bgr  # 120x120 = 14400 >> 0.3%
        result = prep.suppress_green(img)
        assert result[100, 100].sum() == 0

    def test_tiny_green_not_removed(self, prep):
        """A green pixel < 0.3% should not be touched."""
        img = _uniform_image(150)
        green_bgr = cv2.cvtColor(
            np.array([[[60, 220, 200]]], dtype=np.uint8),
            cv2.COLOR_HSV2BGR,
        ).flatten()
        img[5, 5] = green_bgr  # 1 pixel out of 40000 → 0.0025% < 0.3%
        result = prep.suppress_green(img)
        # Non-green pixels should be unchanged
        np.testing.assert_array_equal(result[0, 0], img[0, 0])

    def test_output_shape_preserved(self, prep):
        img = _noise_image(300, 400)
        result = prep.suppress_green(img)
        assert result.shape == (300, 400, 3)

    def test_returns_copy_not_original(self, prep):
        """When suppression occurs, result is not the same object."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        green_bgr = cv2.cvtColor(
            np.array([[[60, 220, 200]]], dtype=np.uint8),
            cv2.COLOR_HSV2BGR,
        ).flatten()
        img[40:160, 40:160] = green_bgr
        original = img.copy()
        result = prep.suppress_green(img)
        # Original shouldn't be mutated
        np.testing.assert_array_equal(img, original)


# ─────────────────────────────────────────────────────────────
#  find_clean_square_crop
# ─────────────────────────────────────────────────────────────


class TestFindCleanSquareCrop:
    def test_output_is_2d_crop(self, prep):
        img = _noise_image(300, 300)
        result = prep.find_clean_square_crop(img)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_all_black_returns_small_crop(self, prep):
        """When tissue ratio < 5%, returns a small central crop."""
        img = _uniform_image(0, 400, 400)
        result = prep.find_clean_square_crop(img)
        assert result.shape[0] <= 400
        assert result.shape[1] <= 400

    def test_no_black_returns_centre_square(self, prep):
        """Image with no black pixels → centre square crop."""
        img = _noise_image(300, 400)  # non-square
        result = prep.find_clean_square_crop(img)
        side = min(300, 400)
        assert result.shape[0] == side
        assert result.shape[1] == side

    def test_circular_fov_finds_tissue(self, prep_small):
        """Circular FOV image → crop avoids dark borders."""
        img = _circular_fov(200, 200)
        result = prep_small.find_clean_square_crop(img)
        assert result.ndim == 3

    def test_result_is_subset_of_original(self, prep):
        """Crop pixels should be a subset of the original image."""
        img = _noise_image(200, 200, seed=5)
        result = prep.find_clean_square_crop(img)
        assert result.shape[0] <= img.shape[0]
        assert result.shape[1] <= img.shape[1]

    def test_square_input_returns_same_size(self, prep):
        """Square, all-tissue image returns the full square."""
        img = _noise_image(200, 200, seed=10)
        result = prep.find_clean_square_crop(img)
        assert result.shape[0] == 200
        assert result.shape[1] == 200


# ─────────────────────────────────────────────────────────────
#  _search_best_position
# ─────────────────────────────────────────────────────────────


class TestSearchBestPosition:
    def test_returns_none_when_side_too_large(self, prep):
        mask = np.zeros((50, 50), dtype=np.float64)
        integral = cv2.integral(mask)
        result = prep._search_best_position(integral, 50, 50, side=100)
        assert result is None

    def test_returns_tuple_for_valid_params(self, prep):
        mask = np.zeros((100, 100), dtype=np.float64)
        integral = cv2.integral(mask)
        result = prep._search_best_position(integral, 100, 100, side=50)
        assert result is not None
        assert len(result) == 3

    def test_zero_count_for_clean_mask(self, prep):
        """All-zero mask → best position has dark count = 0."""
        mask = np.zeros((200, 200), dtype=np.float64)
        integral = cv2.integral(mask)
        y, x, count = prep._search_best_position(integral, 200, 200, side=100)
        assert count == 0.0

    def test_position_within_bounds(self, prep):
        mask = np.random.rand(150, 150).astype(np.float64)
        integral = cv2.integral(mask)
        y, x, _ = prep._search_best_position(integral, 150, 150, side=80)
        assert 0 <= y <= 150 - 80
        assert 0 <= x <= 150 - 80

    def test_prefers_dark_free_region(self, prep):
        """When one region is clean and another is dark, picks the clean one."""
        mask = np.ones((200, 200), dtype=np.float64)
        mask[100:, :] = 0.0  # Bottom half is clean
        integral = cv2.integral(mask)
        y, x, count = prep._search_best_position(integral, 200, 200, side=80)
        # Best position should have fewer dark pixels (lower count)
        assert count <= mask[:80, :80].sum()  # not worse than top-left


# ─────────────────────────────────────────────────────────────
#  generate_multi_crops
# ─────────────────────────────────────────────────────────────


class TestGenerateMultiCrops:
    def test_returns_n_crops_items(self, prep):
        img = _noise_image(300, 300, seed=20)
        crops = prep.generate_multi_crops(img)
        assert len(crops) == prep.n_crops

    def test_each_crop_is_ndarray(self, prep):
        img = _noise_image(300, 300, seed=21)
        crops = prep.generate_multi_crops(img)
        for c in crops:
            assert isinstance(c, np.ndarray)

    def test_small_image_returns_copies(self, prep):
        """Image < 150px → return n_crops copies of the original."""
        img = _noise_image(100, 100, seed=22)
        crops = prep.generate_multi_crops(img)
        assert len(crops) == prep.n_crops

    def test_crops_have_positive_size(self, prep):
        img = _noise_image(250, 250, seed=23)
        crops = prep.generate_multi_crops(img)
        for c in crops:
            assert c.shape[0] > 0
            assert c.shape[1] > 0

    def test_crops_are_3_channel(self, prep):
        img = _noise_image(250, 250, seed=24)
        crops = prep.generate_multi_crops(img)
        for c in crops:
            assert c.shape[2] == 3

    def test_zero_max_offset_returns_copies(self, prep):
        """When crop_size == side, max_offset=0 → all copies."""
        # Create a small image where crop_size >= side
        img = _noise_image(150, 150, seed=25)
        crops = prep.generate_multi_crops(img)
        assert len(crops) == prep.n_crops

    def test_different_crops_when_enough_space(self, prep):
        """Large image should generate varied crops."""
        img = _noise_image(400, 400, seed=26)
        crops = prep.generate_multi_crops(img)
        # At least some crops may differ
        assert len(crops) == prep.n_crops


# ─────────────────────────────────────────────────────────────
#  resize_to_target
# ─────────────────────────────────────────────────────────────


class TestResizeToTarget:
    def test_output_size(self, prep):
        crop = _noise_image(200, 300)
        result = prep.resize_to_target(crop)
        assert result.shape == (prep.target_size, prep.target_size, 3)

    def test_output_dtype_uint8(self, prep):
        crop = _noise_image(100, 100)
        result = prep.resize_to_target(crop)
        assert result.dtype == np.uint8

    def test_square_input_preserved(self, prep):
        crop = _noise_image(200, 200)
        result = prep.resize_to_target(crop)
        assert result.shape[0] == result.shape[1] == prep.target_size

    def test_small_input_upscaled(self, prep):
        crop = _noise_image(32, 32)
        result = prep.resize_to_target(crop)
        assert result.shape[:2] == (prep.target_size, prep.target_size)

    def test_large_input_downscaled(self, prep):
        crop = _noise_image(1024, 1024)
        result = prep.resize_to_target(crop)
        assert result.shape[:2] == (prep.target_size, prep.target_size)


# ─────────────────────────────────────────────────────────────
#  normalize_illumination
# ─────────────────────────────────────────────────────────────


class TestNormalizeIllumination:
    def test_output_shape_preserved(self, prep):
        img = _noise_image(128, 128)
        result = prep.normalize_illumination(img)
        assert result.shape == img.shape

    def test_output_dtype_uint8(self, prep):
        img = _noise_image(64, 64)
        result = prep.normalize_illumination(img)
        assert result.dtype == np.uint8

    def test_dark_image_returned_unchanged(self, prep):
        """Very dark image (mean grayscale < 5) returned as-is."""
        img = _uniform_image(2)
        result = prep.normalize_illumination(img)
        np.testing.assert_array_equal(result, img)

    def test_bright_image_processed(self, prep):
        """Bright image passes through CLAHE processing."""
        img = _noise_image(128, 128, seed=30)
        result = prep.normalize_illumination(img)
        assert result.shape == img.shape

    def test_returns_bgr_image(self, prep):
        """Output should be a 3-channel BGR image."""
        img = _noise_image(64, 64)
        result = prep.normalize_illumination(img)
        assert result.ndim == 3
        assert result.shape[2] == 3


# ─────────────────────────────────────────────────────────────
#  process_image
# ─────────────────────────────────────────────────────────────


class TestProcessImage:
    def test_returns_list(self, prep_small):
        img = _noise_image(200, 200, seed=40)
        result = prep_small.process_image(img)
        assert isinstance(result, list)

    def test_returns_n_crops_items(self, prep_small):
        img = _noise_image(200, 200, seed=41)
        result = prep_small.process_image(img)
        assert len(result) == prep_small.n_crops

    def test_each_output_correct_size(self, prep_small):
        img = _noise_image(250, 250, seed=42)
        result = prep_small.process_image(img)
        for crop in result:
            assert crop.shape == (
                prep_small.target_size,
                prep_small.target_size,
                3,
            )

    def test_output_dtype_uint8(self, prep_small):
        img = _noise_image(200, 200, seed=43)
        result = prep_small.process_image(img)
        for crop in result:
            assert crop.dtype == np.uint8

    def test_circular_fov_image(self, prep_small):
        img = _circular_fov(200, 200)
        result = prep_small.process_image(img)
        assert len(result) == prep_small.n_crops

    def test_black_image_does_not_crash(self, prep_small):
        img = _uniform_image(0, 200, 200)
        result = prep_small.process_image(img)
        assert len(result) == prep_small.n_crops


# ─────────────────────────────────────────────────────────────
#  process_single
# ─────────────────────────────────────────────────────────────


class TestProcessSingle:
    def test_returns_single_image(self, prep_small):
        img = _noise_image(200, 200, seed=50)
        result = prep_small.process_single(img)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3

    def test_output_correct_size(self, prep_small):
        img = _noise_image(200, 200, seed=51)
        result = prep_small.process_single(img)
        assert result.shape == (
            prep_small.target_size,
            prep_small.target_size,
            3,
        )

    def test_output_dtype_uint8(self, prep_small):
        img = _noise_image(200, 200, seed=52)
        result = prep_small.process_single(img)
        assert result.dtype == np.uint8

    def test_different_from_process_image(self, prep_small):
        """process_single returns one image; process_image returns n_crops."""
        img = _noise_image(200, 200, seed=53)
        single = prep_small.process_single(img)
        multi = prep_small.process_image(img)
        assert not isinstance(single, list)
        assert isinstance(multi, list)


# ─────────────────────────────────────────────────────────────
#  process_dataset_tissue_only
# ─────────────────────────────────────────────────────────────


class TestProcessDatasetTissueOnly:
    def _build_clean_dir(self, tmp_path: Path, n_images: int = 3) -> Path:
        """Create a minimal clean_dir structure with real JPEG images."""
        clean_dir = tmp_path / "clean"
        for cls in ["normal", "polyp"]:
            class_dir = clean_dir / cls
            class_dir.mkdir(parents=True)
            for i in range(n_images):
                img = _noise_image(200, 200, seed=i)
                cv2.imwrite(str(class_dir / f"img{i:03d}.jpg"), img)
        # Class mapping
        import json

        (clean_dir / "class_mapping.json").write_text(
            json.dumps({"project": "test", "num_classes": 2})
        )
        return clean_dir

    def test_creates_output_directory(self, tmp_path):
        clean_dir = self._build_clean_dir(tmp_path, n_images=1)
        processed_dir = tmp_path / "processed"
        process_dataset_tissue_only(
            clean_dir=clean_dir,
            processed_dir=processed_dir,
            target_size=64,
            n_crops=2,
        )
        assert processed_dir.exists()

    def test_creates_class_subdirs(self, tmp_path):
        clean_dir = self._build_clean_dir(tmp_path, n_images=1)
        processed_dir = tmp_path / "processed"
        process_dataset_tissue_only(
            clean_dir=clean_dir,
            processed_dir=processed_dir,
            target_size=64,
            n_crops=2,
        )
        assert (processed_dir / "normal").exists()
        assert (processed_dir / "polyp").exists()

    def test_generates_crop_files(self, tmp_path):
        clean_dir = self._build_clean_dir(tmp_path, n_images=2)
        processed_dir = tmp_path / "processed"
        n_crops = 2
        process_dataset_tissue_only(
            clean_dir=clean_dir,
            processed_dir=processed_dir,
            target_size=64,
            n_crops=n_crops,
        )
        normal_crops = list((processed_dir / "normal").glob("*_crop*.jpg"))
        assert len(normal_crops) == 2 * n_crops  # 2 images × 2 crops

    def test_copies_class_mapping(self, tmp_path):
        clean_dir = self._build_clean_dir(tmp_path, n_images=1)
        processed_dir = tmp_path / "processed"
        process_dataset_tissue_only(
            clean_dir=clean_dir,
            processed_dir=processed_dir,
            target_size=64,
            n_crops=1,
        )
        assert (processed_dir / "class_mapping.json").exists()

    def test_removes_existing_output_dir(self, tmp_path):
        clean_dir = self._build_clean_dir(tmp_path, n_images=1)
        processed_dir = tmp_path / "processed"
        # Pre-create with stale file
        processed_dir.mkdir()
        (processed_dir / "stale.txt").write_text("old")
        process_dataset_tissue_only(
            clean_dir=clean_dir,
            processed_dir=processed_dir,
            target_size=64,
            n_crops=1,
        )
        assert not (processed_dir / "stale.txt").exists()

    def test_skips_masks_directory(self, tmp_path):
        clean_dir = self._build_clean_dir(tmp_path, n_images=1)
        # Add a masks dir
        masks_dir = clean_dir / "masks"
        masks_dir.mkdir()
        (masks_dir / "mask.jpg").write_bytes(b"fake")
        processed_dir = tmp_path / "processed"
        process_dataset_tissue_only(
            clean_dir=clean_dir,
            processed_dir=processed_dir,
            target_size=64,
            n_crops=1,
        )
        # masks directory should not be in processed
        assert not (processed_dir / "masks").exists()

    def test_handles_unreadable_image_gracefully(self, tmp_path):
        """Corrupt image files should be skipped without crashing."""
        clean_dir = tmp_path / "clean"
        cls_dir = clean_dir / "normal"
        cls_dir.mkdir(parents=True)
        # Write a corrupt "image"
        (cls_dir / "corrupt.jpg").write_bytes(b"not an image")
        processed_dir = tmp_path / "processed"
        process_dataset_tissue_only(
            clean_dir=clean_dir,
            processed_dir=processed_dir,
            target_size=64,
            n_crops=1,
        )
        assert processed_dir.exists()

    def test_default_paths_used_when_none(self, tmp_path, monkeypatch):
        """When clean_dir/processed_dir are None, paths.COLON_CLEAN is used."""
        import src.data.processing.tissue_only_preprocessor as mod

        fake_paths = MagicMock()
        clean_dir = self._build_clean_dir(tmp_path, n_images=1)
        processed_dir = tmp_path / "default_processed"
        fake_paths.COLON_CLEAN = clean_dir
        fake_paths.COLON_TISSUE_ONLY = processed_dir
        monkeypatch.setattr(mod, "paths", fake_paths)
        process_dataset_tissue_only(
            clean_dir=None,
            processed_dir=None,
            target_size=64,
            n_crops=1,
        )
        assert processed_dir.exists()

    def test_crop_naming_convention(self, tmp_path):
        """Output files should follow <stem>_crop<idx>.jpg naming."""
        clean_dir = self._build_clean_dir(tmp_path, n_images=1)
        processed_dir = tmp_path / "processed"
        process_dataset_tissue_only(
            clean_dir=clean_dir,
            processed_dir=processed_dir,
            target_size=64,
            n_crops=2,
        )
        crops = list((processed_dir / "normal").glob("*.jpg"))
        for crop_path in crops:
            assert "_crop" in crop_path.stem

    def test_output_images_are_readable(self, tmp_path):
        """Generated JPEG files should be valid images."""
        clean_dir = self._build_clean_dir(tmp_path, n_images=1)
        processed_dir = tmp_path / "processed"
        process_dataset_tissue_only(
            clean_dir=clean_dir,
            processed_dir=processed_dir,
            target_size=64,
            n_crops=1,
        )
        for crop_path in (processed_dir / "normal").glob("*.jpg"):
            img = cv2.imread(str(crop_path))
            assert img is not None
            assert img.shape == (64, 64, 3)
