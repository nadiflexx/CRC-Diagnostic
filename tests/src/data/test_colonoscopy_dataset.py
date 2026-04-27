# tests/test_colonoscopy_dataset.py
"""
Tests for src/data/datasets/colonoscopy_dataset.py
(ColonoscopyDataset + transform factory functions).
"""

import numpy as np
import torch

from src.data.processing.image_preprocessor import (
    ColonoscopyDataset,
    get_segmentation_train_transforms,
    get_segmentation_val_transforms,
    get_train_transforms,
    get_val_transforms,
)

# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────


def _make_fake_image(tmp_path, name="img.jpg"):
    """Write a 384×384 JPEG-like file using cv2."""
    import cv2

    img = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    p = tmp_path / name
    cv2.imwrite(str(p), img)
    return str(p)


def _make_fake_mask(tmp_path, name="mask.png"):
    """Write a binary 384×384 PNG mask."""
    import cv2

    mask = np.where(np.random.rand(384, 384) > 0.5, 255, 0).astype(np.uint8)
    p = tmp_path / name
    cv2.imwrite(str(p), mask)
    return str(p)


# ─────────────────────────────────────────────────────────────
#  Transform factories
# ─────────────────────────────────────────────────────────────


class TestGetTrainTransforms:
    def test_returns_compose(self):
        t = get_train_transforms(384)
        assert hasattr(t, "__call__")

    def test_output_is_tensor(self):
        t = get_train_transforms(384)
        img = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        result = t(image=img)["image"]
        assert isinstance(result, torch.Tensor)

    def test_output_shape(self):
        t = get_train_transforms(224)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = t(image=img)["image"]
        assert result.shape[0] == 3

    def test_custom_image_size_coarse_dropout_scaled(self):
        """Instantiation with a non-default size should not raise."""
        t = get_train_transforms(512)
        assert t is not None

    def test_output_dtype_float32(self):
        t = get_train_transforms(384)
        img = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        result = t(image=img)["image"]
        assert result.dtype == torch.float32


class TestGetValTransforms:
    def test_returns_callable(self):
        t = get_val_transforms(384)
        assert callable(t)

    def test_output_is_tensor(self):
        t = get_val_transforms(384)
        img = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        result = t(image=img)["image"]
        assert isinstance(result, torch.Tensor)

    def test_deterministic_output(self):
        """Same input → same output (no random ops)."""
        t = get_val_transforms(384)
        img = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        r1 = t(image=img)["image"]
        r2 = t(image=img)["image"]
        assert torch.equal(r1, r2)


class TestGetSegmentationTrainTransforms:
    def test_output_shapes_match(self):
        t = get_segmentation_train_transforms(256)
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
        out = t(image=img, mask=mask)
        img_t = out["image"]
        mask_t = out["mask"]
        # Both should be 256 after resize
        assert img_t.shape[1] == 256
        assert img_t.shape[2] == 256
        assert mask_t.shape[-2] == 256
        assert mask_t.shape[-1] == 256

    def test_returns_tensor_types(self):
        t = get_segmentation_train_transforms(128)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        out = t(image=img, mask=mask)
        assert isinstance(out["image"], torch.Tensor)


class TestGetSegmentationValTransforms:
    def test_deterministic(self):
        t = get_segmentation_val_transforms(128)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        out1 = t(image=img, mask=mask)
        out2 = t(image=img, mask=mask)
        assert torch.equal(out1["image"], out2["image"])

    def test_resize_applied(self):
        t = get_segmentation_val_transforms(64)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        out = t(image=img, mask=mask)
        assert out["image"].shape[1] == 64


# ─────────────────────────────────────────────────────────────
#  ColonoscopyDataset — classification mode
# ─────────────────────────────────────────────────────────────


class TestColonoscopyDatasetClassification:
    def test_len_matches_paths(self, tmp_path):
        paths = [_make_fake_image(tmp_path, f"img{i}.jpg") for i in range(5)]
        labels = [0, 1, 0, 1, 0]
        ds = ColonoscopyDataset(paths, labels)
        assert len(ds) == 5

    def test_getitem_returns_tuple(self, tmp_path):
        p = _make_fake_image(tmp_path)
        ds = ColonoscopyDataset([p], [1])
        sample = ds[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 2

    def test_getitem_label_preserved(self, tmp_path):
        p = _make_fake_image(tmp_path)
        ds = ColonoscopyDataset([p], [42])
        _, label = ds[0]
        assert label == 42

    def test_getitem_image_is_tensor(self, tmp_path):
        p = _make_fake_image(tmp_path)
        ds = ColonoscopyDataset([p], [0])
        img, _ = ds[0]
        assert isinstance(img, torch.Tensor)

    def test_getitem_image_shape_chw(self, tmp_path):
        p = _make_fake_image(tmp_path)
        ds = ColonoscopyDataset([p], [0], image_size=384)
        img, _ = ds[0]
        assert img.shape[0] == 3
        assert img.shape[1] == 384
        assert img.shape[2] == 384

    def test_missing_image_returns_zeros(self, tmp_path):
        ds = ColonoscopyDataset([str(tmp_path / "nope.jpg")], [0], image_size=64)
        img, _ = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)

    def test_transform_applied(self, tmp_path):
        p = _make_fake_image(tmp_path)
        t = get_val_transforms(384)
        ds = ColonoscopyDataset([p], [0], transform=t)
        img, _ = ds[0]
        assert isinstance(img, torch.Tensor)

    def test_no_transform_divides_by_255(self, tmp_path):
        p = _make_fake_image(tmp_path)
        ds = ColonoscopyDataset([p], [0], image_size=384, transform=None)
        img, _ = ds[0]
        assert img.max().item() <= 1.0 + 1e-6

    def test_multiple_labels(self, tmp_path):
        paths_list = [_make_fake_image(tmp_path, f"i{i}.jpg") for i in range(4)]
        labels = [0, 1, 2, 3]
        ds = ColonoscopyDataset(paths_list, labels)
        for i, (_, label) in enumerate(ds):
            assert label == labels[i]


# ─────────────────────────────────────────────────────────────
#  ColonoscopyDataset — segmentation mode
# ─────────────────────────────────────────────────────────────


class TestColonoscopyDatasetSegmentation:
    def test_getitem_returns_image_and_mask(self, tmp_path):
        img_path = _make_fake_image(tmp_path)
        mask_path = _make_fake_mask(tmp_path)
        ds = ColonoscopyDataset(
            [img_path],
            [0],
            mask_paths=[mask_path],
            image_size=384,
            mode="segmentation",
        )
        img, mask = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    def test_mask_shape_is_1hw(self, tmp_path):
        img_path = _make_fake_image(tmp_path)
        mask_path = _make_fake_mask(tmp_path)
        ds = ColonoscopyDataset(
            [img_path],
            [0],
            mask_paths=[mask_path],
            image_size=64,
            mode="segmentation",
        )
        _, mask = ds[0]
        assert mask.shape[0] == 1

    def test_mask_values_binary(self, tmp_path):
        img_path = _make_fake_image(tmp_path)
        mask_path = _make_fake_mask(tmp_path)
        ds = ColonoscopyDataset(
            [img_path],
            [0],
            mask_paths=[mask_path],
            image_size=64,
            mode="segmentation",
        )
        _, mask = ds[0]
        unique = torch.unique(mask)
        assert all(v.item() in (0.0, 1.0) for v in unique)

    def test_none_mask_path_falls_back_to_label(self, tmp_path):
        img_path = _make_fake_image(tmp_path)
        ds = ColonoscopyDataset(
            [img_path],
            [1],
            mask_paths=[None],
            image_size=64,
            mode="segmentation",
        )
        result = ds[0]
        # Falls back to (img, label) when mask_path is None
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_missing_mask_file_does_not_crash(self, tmp_path):
        img_path = _make_fake_image(tmp_path)
        ds = ColonoscopyDataset(
            [img_path],
            [0],
            mask_paths=[str(tmp_path / "nonexistent_mask.png")],
            image_size=64,
            mode="segmentation",
        )
        result = ds[0]
        assert result is not None

    def test_segmentation_with_transform(self, tmp_path):
        img_path = _make_fake_image(tmp_path)
        mask_path = _make_fake_mask(tmp_path)
        t = get_segmentation_val_transforms(64)
        ds = ColonoscopyDataset(
            [img_path],
            [0],
            mask_paths=[mask_path],
            transform=t,
            image_size=64,
            mode="segmentation",
        )
        img, mask = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
