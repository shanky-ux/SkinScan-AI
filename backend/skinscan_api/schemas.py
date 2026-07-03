from typing import Literal

from pydantic import BaseModel, Field


class DiseaseInfo(BaseModel):
    description: str
    severity: str
    recommendation: str
    severity_level: Literal["low", "moderate", "high", "unknown"] = "unknown"


class ProbabilityItem(BaseModel):
    class_name: str
    probability: float = Field(..., ge=0, le=100)


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(..., ge=0, le=100)
    confidence_level: Literal["high", "medium", "low", "healthy"]
    probabilities: list[ProbabilityItem]
    disease_info: DiseaseInfo
    is_healthy: bool = False
    model_mode: Literal["checkpoint", "demo"]
    model_architecture: str


class ClassItem(BaseModel):
    class_name: str
    disease_info: DiseaseInfo


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_mode: Literal["checkpoint", "demo", "unavailable"]
    model_architecture: str | None = None
    model_path: str
    classes_count: int = 0
