from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


ALLOWED_MODES = {"RGB", "RGBA", "L"}


def load_image_from_bytes(data: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(data))
        image.load()
        return image
    except Exception as exc:
        raise ValueError("Invalid or corrupt image file") from exc


def validate_image(image: Image.Image) -> bool:
    try:
        if image is None:
            return False
        if image.size[0] < 32 or image.size[1] < 32:
            return False
        if image.mode not in ALLOWED_MODES:
            return False
        return True
    except Exception:
        return False


def enhance_image_quality(image: Image.Image) -> Image.Image:
    try:
        opencv_image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)

        enhanced = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    except Exception:
        return image


def _base_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def preprocess_image(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    processed_image = enhance_image_quality(image)
    return _base_transform()(processed_image)


def preprocess_image_with_tta(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")

    enhanced_image = enhance_image_quality(image)
    transforms_list = [
        _base_transform(),
        transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomRotation(5),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    ]

    return torch.stack([transform(enhanced_image) for transform in transforms_list])
