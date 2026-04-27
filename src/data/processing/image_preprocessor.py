"""
Augmentation and Dataset for colonoscopy images.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config.constants import IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms(image_size=384):
    """
    Build the training augmentation pipeline for classification images.

    Applies spatial and colour augmentations followed by ImageNet
    normalisation and tensor conversion. CoarseDropout randomly erases
    rectangular regions to improve robustness to occlusion.

    Args:
        image_size (int): Used to scale the CoarseDropout patch
            dimensions relative to the input resolution. Default is 384.

    Returns:
        A.Compose: Albumentations composition containing:
            - Random rotation ±30° (p=0.5) with reflect border.
            - Horizontal flip (p=0.5).
            - Vertical flip (p=0.5).
            - One-of: brightness/contrast or hue/saturation jitter (p=0.5).
            - CoarseDropout of 1–3 patches (p=0.3).
            - ImageNet normalisation.
            - Conversion to PyTorch tensor.
    """
    return A.Compose(
        [
            A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15, contrast_limit=0.15, p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=15,
                        val_shift_limit=10,
                        p=1.0,
                    ),
                ],
                p=0.5,
            ),
            A.CoarseDropout(
                max_holes=3,
                max_height=image_size // 12,
                max_width=image_size // 12,
                min_holes=1,
                min_height=image_size // 20,
                min_width=image_size // 20,
                fill_value=0,
                p=0.3,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size=384):
    """
    Build the validation/test transform pipeline for classification images.

    No geometric or colour augmentation is applied. Only ImageNet
    normalisation and tensor conversion are performed to ensure
    deterministic evaluation.

    Args:
        image_size (int): Kept for API consistency with the training
            counterpart. Not used internally. Default is 384.

    Returns:
        A.Compose: Albumentations composition containing:
            - ImageNet normalisation.
            - Conversion to PyTorch tensor.
    """
    return A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_segmentation_train_transforms(image_size=384):
    """
    Build the training augmentation pipeline for segmentation images.

    Includes a mandatory ``Resize`` step so that both image and mask
    are brought to a consistent resolution before augmentation. Spatial
    transforms are applied identically to image and mask by Albumentations.

    Args:
        image_size (int): Target height and width after resizing.
            Default is 384.

    Returns:
        A.Compose: Albumentations composition containing:
            - Resize to ``(image_size, image_size)``.
            - Horizontal flip (p=0.5).
            - Vertical flip (p=0.5).
            - Random rotation ±15° (p=0.3) with reflect border.
            - Random brightness/contrast (p=0.3).
            - ImageNet normalisation.
            - Conversion to PyTorch tensor.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_REFLECT_101),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_segmentation_val_transforms(image_size=384):
    """
    Build the validation/test transform pipeline for segmentation images.

    Applies only a deterministic resize followed by normalisation and
    tensor conversion, with no augmentation.

    Args:
        image_size (int): Target height and width after resizing.
            Default is 384.

    Returns:
        A.Compose: Albumentations composition containing:
            - Resize to ``(image_size, image_size)``.
            - ImageNet normalisation.
            - Conversion to PyTorch tensor.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class ColonoscopyDataset(Dataset):
    """
    PyTorch Dataset for colonoscopy images supporting both classification
    and binary segmentation modes.

    In ``"classification"`` mode each ``__getitem__`` call returns an
    ``(image_tensor, label)`` tuple. In ``"segmentation"`` mode it
    returns ``(image_tensor, mask_tensor)`` where the mask tensor has
    shape (1, H, W) with float values in {0.0, 1.0}.

    Images and masks that cannot be read from disk are replaced by
    zero-filled arrays of the configured ``image_size``.
    """

    def __init__(
        self,
        image_paths,
        labels,
        mask_paths=None,
        transform=None,
        image_size=384,
        mode="classification",
    ):
        """
        Initialise the colonoscopy dataset.

        Args:
            image_paths (list[str]): Absolute paths to the image files.
            labels (list[int]): Integer class labels aligned with
                ``image_paths``.
            mask_paths (list[str | None] | None): Absolute paths to
                binary mask files aligned with ``image_paths``. May
                contain ``None`` entries for images without a mask.
                Ignored in ``"classification"`` mode. Default is ``None``.
            transform (A.Compose | None): Albumentations transform
                pipeline applied to each sample. When ``None`` the image
                is converted to a float tensor by dividing by 255.
                Default is ``None``.
            image_size (int): Expected spatial resolution in pixels.
                Images or masks that deviate are resized to
                ``(image_size, image_size)``. Default is 384.
            mode (str): Dataset operating mode. Use ``"classification"``
                to return ``(image, label)`` pairs or ``"segmentation"``
                to return ``(image, mask)`` pairs. Default is
                ``"classification"``.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.mask_paths = mask_paths

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Total number of image paths provided at construction.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load, optionally augment, and return a single sample.

        The image is read as BGR, converted to RGB, and resized if
        necessary. In segmentation mode the corresponding mask is loaded
        as grayscale, thresholded at 127, and resized with nearest-
        neighbour interpolation.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, int] | tuple[torch.Tensor, torch.Tensor]:
                - In ``"classification"`` mode: ``(image_tensor, label)``
                  where ``image_tensor`` has shape (C, H, W) and
                  ``label`` is an integer.
                - In ``"segmentation"`` mode: ``(image_tensor, mask_tensor)``
                  where ``mask_tensor`` has shape (1, H, W) with float
                  values in {0.0, 1.0}.
        """
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        if h != self.image_size or w != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))

        mask = None
        if self.mode == "segmentation" and self.mask_paths[idx]:
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = (mask > 127).astype(np.float32)
                if mask.shape[:2] != (self.image_size, self.image_size):
                    mask = cv2.resize(
                        mask,
                        (self.image_size, self.image_size),
                        interpolation=cv2.INTER_NEAREST,
                    )

        if self.transform:
            if mask is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=img)
                img = augmented["image"]
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        if self.mode == "segmentation" and mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            return img, mask

        return img, self.labels[idx]
