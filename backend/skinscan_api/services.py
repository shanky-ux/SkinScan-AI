from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from io import BytesIO

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from skinscan_api.config import get_settings


DiseaseClasses = [
    "Actinic keratosis",
    "Atopic Dermatitis",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanocytic nevus",
    "Melanoma",
    "Squamous cell carcinoma",
    "Tinea Ringworm Candidiasis",
    "Vascular lesion",
]


DiseaseInfoMap: dict[str, dict[str, str]] = {
    "Actinic keratosis": {
        "description": "Precancerous skin lesions caused by sun damage",
        "severity": "Moderate risk - can develop into cancer",
        "recommendation": "Medical evaluation and treatment recommended",
        "severity_level": "moderate",
    },
    "Atopic Dermatitis": {
        "description": "Chronic inflammatory skin condition (eczema)",
        "severity": "Low to moderate risk",
        "recommendation": "Dermatologist consultation for treatment plan",
        "severity_level": "low",
    },
    "Benign keratosis": {
        "description": "Non-cancerous skin growths",
        "severity": "Low risk",
        "recommendation": "Regular monitoring recommended",
        "severity_level": "low",
    },
    "Dermatofibroma": {
        "description": "Benign fibrous skin tumor",
        "severity": "Low risk",
        "recommendation": "Usually no treatment needed unless bothersome",
        "severity_level": "low",
    },
    "Melanocytic nevus": {
        "description": "Common benign skin growths (moles)",
        "severity": "Generally benign",
        "recommendation": "Monitor for changes in size, color, or shape",
        "severity_level": "low",
    },
    "Melanoma": {
        "description": "A serious form of skin cancer",
        "severity": "High risk - requires immediate medical attention",
        "recommendation": "Consult a dermatologist immediately",
        "severity_level": "high",
    },
    "Squamous cell carcinoma": {
        "description": "Second most common type of skin cancer",
        "severity": "Moderate to high risk",
        "recommendation": "Immediate dermatologist consultation required",
        "severity_level": "high",
    },
    "Tinea Ringworm Candidiasis": {
        "description": "Fungal skin infections",
        "severity": "Low risk but contagious",
        "recommendation": "Antifungal treatment and medical consultation",
        "severity_level": "low",
    },
    "Vascular lesion": {
        "description": "Lesions involving blood vessels in the skin",
        "severity": "Generally low risk",
        "recommendation": "Medical evaluation for proper diagnosis",
        "severity_level": "low",
    },
}

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


class SkinDiseaseModel(nn.Module):
    def __init__(self, num_classes: int = 9, architecture: str = "efficientnet_b0"):
        super().__init__()
        self.architecture = architecture
        self.backbone = _build_backbone(architecture, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _build_backbone(architecture: str, num_classes: int) -> nn.Module:
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    return model


def _normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def _guess_architecture(state_dict: dict[str, torch.Tensor]) -> str:
    keys = list(state_dict.keys())
    if any(key.startswith("fc.") or key.startswith("layer1.") for key in keys):
        return "resnet18"
    return "efficientnet_b0"


def get_disease_classes() -> list[str]:
    settings = get_settings()
    model_path = settings.model_path_resolved
    if not model_path.exists():
        return DiseaseClasses.copy()

    try:
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        if isinstance(checkpoint, dict) and "class_names" in checkpoint:
            return list(checkpoint["class_names"])
    except Exception:
        pass

    return DiseaseClasses.copy()


def get_disease_info(disease_name: str) -> dict[str, str]:
    return DiseaseInfoMap.get(
        disease_name,
        {
            "description": "Skin condition requiring medical evaluation",
            "severity": "Unknown",
            "recommendation": "Consult a healthcare provider for proper diagnosis",
            "severity_level": "unknown",
        },
    )


@dataclass(slots=True)
class LoadedModel:
    model: nn.Module
    class_names: list[str]
    architecture: str
    mode: str
    checkpoint_path: Path


def _build_demo_model(class_names: list[str]) -> LoadedModel:
    model = _build_backbone("efficientnet_b0", len(class_names))
    model.eval()
    return LoadedModel(
        model=model,
        class_names=class_names,
        architecture="efficientnet_b0",
        mode="demo",
        checkpoint_path=get_settings().model_path_resolved,
    )


@lru_cache(maxsize=1)
def load_model() -> LoadedModel:
    settings = get_settings()
    class_names = get_disease_classes()
    model_path = settings.model_path_resolved

    if not model_path.exists():
        if settings.allow_demo_model:
            return _build_demo_model(class_names)
        raise FileNotFoundError(f"Model file not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        if settings.allow_demo_model:
            return _build_demo_model(class_names)
        raise ValueError("Invalid checkpoint format")

    state_dict = _normalize_state_dict(checkpoint["model_state_dict"])
    class_names = list(checkpoint.get("class_names", class_names))
    architecture = _guess_architecture(state_dict)

    model = _build_backbone(architecture, len(class_names))
    model_state = model.state_dict()
    filtered_state = {key: value for key, value in state_dict.items() if key in model_state and value.shape == model_state[key].shape}
    model_state.update(filtered_state)
    model.load_state_dict(model_state)
    model.eval()

    return LoadedModel(
        model=model,
        class_names=class_names,
        architecture=architecture,
        mode="checkpoint",
        checkpoint_path=model_path,
    )


def _confidence_level(probabilities: np.ndarray) -> str:
    max_prob = float(np.max(probabilities))
    if max_prob < 0.30:
        return "healthy"
    if max_prob >= 0.75:
        return "high"
    if max_prob >= 0.45:
        return "medium"
    return "low"


def predict_image(image: Image.Image) -> dict[str, Any]:
    loaded = load_model()

    if loaded.mode == "demo":
        class_count = max(len(loaded.class_names), 1)
        flat_probability = 100.0 / class_count
        return {
            "predicted_class": "Model unavailable",
            "confidence": 0.0,
            "confidence_level": "low",
            "probabilities": [
                {"class_name": class_name, "probability": flat_probability}
                for class_name in loaded.class_names
            ],
            "disease_info": {
                "description": "No trained checkpoint is loaded yet, so the backend is running in demo mode.",
                "severity": "Unknown",
                "recommendation": "Add the trained checkpoint file and restart the backend to get real predictions.",
                "severity_level": "unknown",
            },
            "is_healthy": False,
            "model_mode": loaded.mode,
            "model_architecture": loaded.architecture,
        }

    tta_images = preprocess_image_with_tta(image)

    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for image_tensor in tta_images:
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            logits = loaded.model(image_tensor)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
            predictions.append(probabilities)

    mean_probabilities = np.mean(predictions, axis=0)
    top_index = int(np.argmax(mean_probabilities))
    top_probability = float(mean_probabilities[top_index] * 100)
    confidence_level = _confidence_level(mean_probabilities)
    is_healthy = confidence_level == "healthy"

    if is_healthy:
        predicted_class = "Healthy Skin"
        disease_info = {
            "description": "No strong disease pattern detected by the model",
            "severity": "Low risk",
            "recommendation": "If symptoms persist, consult a clinician",
            "severity_level": "low",
        }
    else:
        predicted_class = loaded.class_names[top_index]
        disease_info = get_disease_info(predicted_class)

    probability_rows = [
        {"class_name": class_name, "probability": float(prob * 100)}
        for class_name, prob in zip(loaded.class_names, mean_probabilities)
    ]

    return {
        "predicted_class": predicted_class,
        "confidence": top_probability,
        "confidence_level": confidence_level,
        "probabilities": probability_rows,
        "disease_info": disease_info,
        "is_healthy": is_healthy,
        "model_mode": loaded.mode,
        "model_architecture": loaded.architecture,
    }
