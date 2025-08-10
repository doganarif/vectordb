import logging
import os
from pathlib import Path
from typing import Optional


class Settings:
    def __init__(self) -> None:
        self.environment: str = os.getenv("ENV", "local")
        self.data_dir: Path = Path(os.getenv("DATA_DIR", "data")).resolve()
        self.cohere_api_key: Optional[str] = os.getenv("COHERE_API_KEY")
        self.default_metric: str = os.getenv("DEFAULT_METRIC", "cosine")
        self.default_index: str = os.getenv("DEFAULT_INDEX", "linear")
        # LSH configuration
        self.lsh_num_planes: int = int(os.getenv("LSH_NUM_PLANES", "16"))
        self.lsh_num_tables: int = int(os.getenv("LSH_NUM_TABLES", "4"))
        self.lsh_seed: int = int(os.getenv("LSH_SEED", "42"))

        # Logging configuration
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

        self.data_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()


def configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
