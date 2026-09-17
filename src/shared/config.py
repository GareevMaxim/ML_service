import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class settings(BaseSettings):
    PROJECT_NAME: str = "ML Service"
    ENV: str = "development"

    DATA_DIR: str = "data"
    MODELS_REGISTRY_DIR: str = "models_registry"
    CLASSIFICATION_THRESHOLD: float = 0.5

    @property
    def data_path(self) -> Path:
        return BASE_DIR / self.DATA_DIR
    @property
    def registry_path(self) -> Path:
        return BASE_DIR / self.MODELS_REGISTRY_DIR

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", 
        extra="ignore"
    )

settings = settings()