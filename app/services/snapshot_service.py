import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.core import settings
from app.repositories.base import VectorRepository
from app.services.index_service import IndexService


class SnapshotService:
    """Service for handling database snapshots (save/load operations)."""

    def __init__(
        self, repository: VectorRepository, index_service: IndexService
    ) -> None:
        self.repository = repository
        self.index_service = index_service
        self.logger = logging.getLogger(self.__class__.__name__)
        self._data_dir = settings.data_dir

    def save(self, path: Optional[Path] = None) -> Path:
        """Save database snapshot to disk.

        Saves both data and index metadata to a JSON file.

        Args:
            path: Optional path for the snapshot file.
                  Defaults to DATA_DIR/snapshot_YYYYMMDD_HHMMSS.json

        Returns:
            Path where the snapshot was saved
        """
        if path is None:
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self._data_dir / f"snapshot_{timestamp}.json"

        snapshot_data = self.repository.snapshot()
        index_metadata = self.index_service.get_index_metadata()

        data: dict[str, Any] = {
            **snapshot_data,
            "indices": index_metadata,
            "timestamp": datetime.now().isoformat(),
        }

        # Ensure the directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(json.dumps(data, indent=2, sort_keys=True))
        self.logger.info(f"Database saved to {path}")
        return path

    def load(self, path: Optional[Path] = None) -> None:
        """Load database snapshot from disk.

        Restores both data and rebuilds indices from saved metadata.

        Args:
            path: Optional path to the snapshot file.
                  Defaults to DATA_DIR/snapshot.json
        """
        path = path or settings.data_dir / "snapshot.json"

        if not path.exists():
            self.logger.info(f"No snapshot found at {path}")
            return

        try:
            data = json.loads(path.read_text())
            self.repository.load_snapshot(data)

            index_metadata = data.get("indices", {})
            self.index_service.rebuild_indices(index_metadata)

            self.logger.info(f"Database loaded from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load database: {e}")
            raise
