from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from app.schemas import ClassItem, DiseaseInfo, HealthResponse, PredictResponse, ProbabilityItem
from app.services.image_service import load_image_from_bytes, validate_image
from app.services.model_service import get_disease_classes, get_disease_info, load_model, predict_image


router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    try:
        loaded = load_model()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_mode=loaded.mode,
            model_architecture=loaded.architecture,
            model_path=str(loaded.checkpoint_path),
            classes_count=len(loaded.class_names),
        )
    except Exception as exc:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_mode="unavailable",
            model_architecture=None,
            model_path=str(exc),
            classes_count=len(get_disease_classes()),
        )


@router.get("/classes", response_model=list[ClassItem])
def classes() -> list[ClassItem]:
    class_names = get_disease_classes()
    return [
        ClassItem(class_name=class_name, disease_info=DiseaseInfo(**get_disease_info(class_name)))
        for class_name in class_names
    ]


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/tiff"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload JPEG, PNG, WebP, BMP, or TIFF.")

    try:
        data = await file.read()
        image = load_image_from_bytes(data)
        if not validate_image(image):
            raise HTTPException(status_code=400, detail="Image is too small or has an unsupported format.")
        result = predict_image(image)
        result["probabilities"] = [ProbabilityItem(**item) for item in result["probabilities"]]
        result["disease_info"] = DiseaseInfo(**result["disease_info"])
        return PredictResponse(**result)
    except HTTPException:
        raise
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
