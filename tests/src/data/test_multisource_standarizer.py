# tests/test_multi_source_standardizer.py
"""
Tests for src/data/processing/standardizer.py (MultiSourceStandardizer).
"""

import cv2
import numpy as np
import pytest

from src.data.processing.standardizer import MultiSourceStandardizer

# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def std():
    return MultiSourceStandardizer(target_size=64)


def _white_image(h=200, w=200):
    return np.ones((h, w, 3), dtype=np.uint8) * 200


def _black_image(h=200, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _circular_fov_image(h=300, w=300):
    """Simulate a circular FOV image (dark corners)."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 150
    # Black out corners
    corner = h // 10
    img[:corner, :corner] = 0
    img[:corner, w - corner :] = 0
    img[h - corner :, :corner] = 0
    img[h - corner :, w - corner :] = 0
    return img


def _rect_fov_image(h=200, w=200):
    """Simulate a rectangular FOV (bright, uniform)."""
    return np.ones((h, w, 3), dtype=np.uint8) * 150


# ─────────────────────────────────────────────────────────────
#  __init__
# ─────────────────────────────────────────────────────────────


class TestInit:
    def test_target_size_stored(self):
        s = MultiSourceStandardizer(target_size=128)
        assert s.target_size == 128

    def test_clahe_created(self, std):
        assert std.clahe is not None

    def test_morph_kernel_created(self, std):
        assert std.morph_kernel is not None


# ─────────────────────────────────────────────────────────────
#  suppress_green
# ─────────────────────────────────────────────────────────────


class TestSuppressGreen:
    def test_returns_same_shape(self, std):
        img = _white_image()
        result = std.suppress_green(img)
        assert result.shape == img.shape

    def test_no_green_returns_original_like(self, std):
        img = _white_image()
        result = std.suppress_green(img)
        # Should be identical or very close (no green present)
        np.testing.assert_array_equal(result, img)

    def test_green_annotation_removed(self, std):
        """Large green region should be zeroed out."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Fill a large green region in HSV → BGR
        green_hsv = np.array([60, 200, 200], dtype=np.uint8)
        green_bgr = cv2.cvtColor(
            green_hsv.reshape(1, 1, 3), cv2.COLOR_HSV2BGR
        ).flatten()
        img[50:150, 50:150] = green_bgr  # 100×100 patch > 0.3%
        result = std.suppress_green(img)
        assert result[100, 100].sum() == 0

    def test_small_green_not_removed(self, std):
        """A green region smaller than 0.3% should not be removed."""
        img = _white_image(300, 300)
        # 1x1 green pixel
        green_bgr = cv2.cvtColor(
            np.array([[[60, 200, 200]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        ).flatten()
        img[5, 5] = green_bgr
        result = std.suppress_green(img)
        # White image unchanged in most areas
        np.testing.assert_array_equal(result[0, 0], img[0, 0])


# ─────────────────────────────────────────────────────────────
#  detect_fov_type
# ─────────────────────────────────────────────────────────────


class TestDetectFovType:
    def test_circular_detected(self, std):
        img = _circular_fov_image()
        result = std.detect_fov_type(img)
        assert result == "circular"

    def test_rectangular_detected(self, std):
        img = _rect_fov_image()
        result = std.detect_fov_type(img)
        assert result == "rectangular"

    def test_returns_string(self, std):
        img = _white_image()
        result = std.detect_fov_type(img)
        assert isinstance(result, str)
        assert result in ("circular", "rectangular")


# ─────────────────────────────────────────────────────────────
#  standardize_geometry
# ─────────────────────────────────────────────────────────────


class TestStandardizeGeometry:
    def test_output_size(self, std):
        tissue = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        result = std.standardize_geometry(tissue)
        assert result.shape == (64, 64, 3)

    def test_square_input_unchanged_size(self, std):
        tissue = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        result = std.standardize_geometry(tissue)
        assert result.shape == (64, 64, 3)

    def test_returns_uint8(self, std):
        tissue = _white_image(100, 100)
        result = std.standardize_geometry(tissue)
        assert result.dtype == np.uint8


# ─────────────────────────────────────────────────────────────
#  normalize_illumination
# ─────────────────────────────────────────────────────────────


class TestNormalizeIllumination:
    def test_output_shape_preserved(self, std):
        img = _white_image(64, 64)
        result = std.normalize_illumination(img)
        assert result.shape == img.shape

    def test_dark_image_returned_unchanged(self, std):
        """Very dark image (mean < 5) should be returned as-is."""
        img = _black_image(64, 64)
        result = std.normalize_illumination(img)
        np.testing.assert_array_equal(result, img)

    def test_output_dtype_uint8(self, std):
        img = _white_image(64, 64)
        result = std.normalize_illumination(img)
        assert result.dtype == np.uint8


# ─────────────────────────────────────────────────────────────
#  process_image
# ─────────────────────────────────────────────────────────────


class TestProcessImage:
    def test_output_shape(self, std):
        img = _white_image(200, 300)
        result = std.process_image(img)
        assert result.shape == (64, 64, 3)

    def test_output_dtype(self, std):
        img = _white_image()
        result = std.process_image(img)
        assert result.dtype == np.uint8

    def test_does_not_modify_input(self, std):
        img = _white_image()
        original = img.copy()
        std.process_image(img)
        np.testing.assert_array_equal(img, original)

    def test_handles_non_square_input(self, std):
        img = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)
        result = std.process_image(img)
        assert result.shape == (64, 64, 3)


# ─────────────────────────────────────────────────────────────
#  process_image_and_mask
# ─────────────────────────────────────────────────────────────


class TestProcessImageAndMask:
    def test_output_shapes(self, std):
        img = _white_image(200, 200)
        mask = np.ones((200, 200), dtype=np.uint8) * 255
        result_img, result_mask = std.process_image_and_mask(img, mask)
        assert result_img.shape == (64, 64, 3)
        assert result_mask.shape == (64, 64)

    def test_mask_values_binary(self, std):
        img = _white_image(100, 100)
        mask = np.where(np.random.rand(100, 100) > 0.5, 255, 0).astype(np.uint8)
        _, result_mask = std.process_image_and_mask(img, mask)
        unique = np.unique(result_mask)
        assert all(v in (0, 255) for v in unique)

    def test_full_mask_preserved(self, std):
        img = _white_image(100, 100)
        mask = np.full((100, 100), 255, dtype=np.uint8)
        _, result_mask = std.process_image_and_mask(img, mask)
        # After nearest-neighbour resize the mask should still be mostly 255
        assert result_mask.mean() > 200

    def test_empty_mask_preserved(self, std):
        img = _white_image(100, 100)
        mask = np.zeros((100, 100), dtype=np.uint8)
        _, result_mask = std.process_image_and_mask(img, mask)
        assert result_mask.sum() == 0

    def test_returns_tuple_of_two(self, std):
        img = _white_image()
        mask = np.zeros((200, 200), dtype=np.uint8)
        result = std.process_image_and_mask(img, mask)
        assert len(result) == 2
