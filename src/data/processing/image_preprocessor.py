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
    Get the training transforms for the dataset.

    :param image_size: Size of the input images.
    :return: Composed augmentation transforms.
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
    Get the validation transforms for the dataset.

    :param image_size: Size of the input images.
    :return: Composed augmentation transforms.
    """
    return A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_segmentation_train_transforms(image_size=384):
    """
    Get the training transforms for the segmentation dataset.

    :param image_size: Size of the input images.
    :return: Composed augmentation transforms.
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
    Get the validation transforms for the segmentation dataset.

    :param image_size: Size of the input images.
    :return: Composed augmentation transforms.
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
    Dataset class for colonoscopy images.
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
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.mask_paths = mask_paths

    def __len__(self):
        """
        Get the length of the dataset.

        :return: Length of the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        :param idx: Index of the item to retrieve.
        :return: Tuple of (image, label) or (image, mask) depending on the mode.
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
