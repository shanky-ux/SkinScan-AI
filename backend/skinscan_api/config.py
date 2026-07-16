from functools import lru_cache
from pathlib import Path
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "SkinScan-AI API"
    environment: str = "development"
    model_path: str = os.getenv("SKINSCAN_MODEL_PATH", "backend/attached_assets/skin_disease_model_1755756972916.pth")
    frontend_origin: str = os.getenv("SKINSCAN_FRONTEND_ORIGIN", "http://localhost:3000")
    allow_demo_model: bool = os.getenv("SKINSCAN_ALLOW_DEMO_MODEL", "true").lower() == "true"
    max_upload_mb: int = 10

    @property
    def model_path_resolved(self) -> Path:
        path = Path(self.model_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent.parent / path
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
