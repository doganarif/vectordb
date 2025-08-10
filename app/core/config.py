from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .constants import DistanceMetric, IndexAlgorithm


@dataclass
class Settings:
    """Application configuration settings."""

    environment: str = field(default_factory=lambda: os.getenv("ENV", "local"))
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", "data")).resolve()
    )
    cohere_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("COHERE_API_KEY")
    )
    default_metric: str = field(
        default_factory=lambda: os.getenv("DEFAULT_METRIC", DistanceMetric.COSINE.value)
    )
    default_index: str = field(
        default_factory=lambda: os.getenv("DEFAULT_INDEX", IndexAlgorithm.LINEAR.value)
    )

    # LSH configuration
    lsh_num_planes: int = field(
        default_factory=lambda: int(os.getenv("LSH_NUM_PLANES", "16"))
    )
    lsh_num_tables: int = field(
        default_factory=lambda: int(os.getenv("LSH_NUM_TABLES", "4"))
    )
    lsh_seed: int = field(default_factory=lambda: int(os.getenv("LSH_SEED", "42")))

    # Logging configuration
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        """Post-initialization to create directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()


def configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
