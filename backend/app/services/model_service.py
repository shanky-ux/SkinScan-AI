from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from app.config import get_settings
from app.services.image_service import preprocess_image_with_tta


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
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        clean_key = key.replace("module.", "")
        normalized[clean_key] = value
    return normalized


def _guess_architecture(state_dict: dict[str, torch.Tensor]) -> str:
    keys = list(state_dict.keys())
    if any(key.startswith("fc.") or key.startswith("layer1.") for key in keys):
        return "resnet18"
    if any(key.startswith("classifier.") or key.startswith("features.") for key in keys):
        return "efficientnet_b0"
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


def predict_image(image) -> dict[str, Any]:
    loaded = load_model()
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
