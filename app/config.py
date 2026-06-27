"""Application configuration using Pydantic Settings."""

import json
from pathlib import Path
from typing import Dict, Optional

from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = ConfigDict(
        protected_namespaces=(),
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Model paths
    model_dir: Path = Field(default=Path("./app/models"), alias="MODEL_DIR")

    # Database
    db_path: Path = Field(default=Path("./data/toxicity.db"), alias="DB_PATH")

    # Redis (optional - falls back to in-memory dict if empty)
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")

    # API limits
    max_batch: int = Field(default=100, alias="MAX_BATCH")

    # Logging
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # Language-specific thresholds (JSON string)
    thresholds_json: str = Field(
        default='{"default": 0.88, "ar": 0.85, "tr": 0.86, "ja": 0.87, "en": 0.88, "es": 0.88}',
        alias="THRESHOLDS"
    )

    @property
    def thresholds(self) -> Dict[str, float]:
        """Parse thresholds from JSON string."""
        return json.loads(self.thresholds_json)

    @property
    def svm_path(self) -> Path:
        return self.model_dir / "svm" / "svm_model.pkl"

    @property
    def mbert_path(self) -> Path:
        return self.model_dir / "mbert"

    @property
    def cnn_path(self) -> Path:
        return self.model_dir / "cnn" / "cnn_best.pt"


# Singleton settings instance
settings = Settings()