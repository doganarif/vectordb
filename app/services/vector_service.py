import json
import logging
from pathlib import Path
from typing import Any

from app.core import settings
from app.core.exceptions import (
    DimensionalityMismatchException,
    InvalidAlgorithmException,
    InvalidMetricException,
    ResourceNotFoundException,
)
from app.domain.models import Chunk, Document, Library
from app.repositories import InMemoryRepository
from app.repositories.base import VectorRepository
from app.services.chunk_service import ChunkService
from app.services.document_service import DocumentService
from app.services.index_service import IndexService
from app.services.library_service import LibraryService


class VectorDBService:
    def __init__(self, repo: VectorRepository | None = None) -> None:
        self.repo = repo or InMemoryRepository()
        self.library_service = LibraryService(self.repo)
        self.document_service = DocumentService(self.repo)
        self.chunk_service = ChunkService(self.repo)
        self.index_service = IndexService(self.repo)
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_library(
        self,
        name: str,
        description: str | None,
        metadata: dict[str, str],
    ) -> Library:
        return self.library_service.create_library(name, description, metadata)

    def list_libraries(self) -> list[Library]:
        return self.library_service.list_libraries()

    def get_library(self, library_id: str) -> Library | None:
        try:
            return self.library_service.get_library(library_id)
        except ResourceNotFoundException:
            return None

    def update_library(
        self,
        library_id: str,
        name: str | None,
        description: str | None,
        metadata: dict[str, str] | None,
    ) -> Library | None:
        try:
            return self.library_service.update_library(
                library_id, name, description, metadata
            )
        except ResourceNotFoundException:
            return None

    def delete_library(self, library_id: str) -> None:
        self.library_service.delete_library(library_id)
        self.index_service.clear_index(library_id)

    def create_document(
        self,
        library_id: str,
        title: str,
        description: str | None,
        metadata: dict[str, str],
    ) -> Document | None:
        try:
            return self.document_service.create_document(
                library_id, title, description, metadata
            )
        except ResourceNotFoundException:
            return None

    def list_documents(self, library_id: str) -> list[Document]:
        return self.document_service.list_documents(library_id)

    def update_document(
        self,
        document_id: str,
        title: str | None,
        description: str | None,
        metadata: dict[str, str] | None,
    ) -> Document | None:
        try:
            return self.document_service.update_document(
                document_id, title, description, metadata
            )
        except ResourceNotFoundException:
            return None

    def delete_document(self, document_id: str) -> None:
        self.document_service.delete_document(document_id)

    def create_chunk(
        self,
        library_id: str,
        document_id: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, str],
    ) -> Chunk | None:
        try:
            return self.chunk_service.create_chunk(
                library_id, document_id, text, embedding, metadata
            )
        except DimensionalityMismatchException as e:
            self.logger.error("Dimensionality mismatch: %s", e)
            raise ValueError(str(e))
        except ResourceNotFoundException:
            return None

    def list_chunks(self, library_id: str) -> list[Chunk]:
        return self.chunk_service.list_chunks(library_id)

    def update_chunk(
        self,
        chunk_id: str,
        text: str | None,
        embedding: list[float] | None,
        metadata: dict[str, str] | None,
    ) -> Chunk | None:
        try:
            return self.chunk_service.update_chunk(chunk_id, text, embedding, metadata)
        except DimensionalityMismatchException as e:
            self.logger.error("Dimensionality mismatch: %s", e)
            raise ValueError(str(e))
        except ResourceNotFoundException:
            return None

    def delete_chunk(self, chunk_id: str) -> None:
        self.chunk_service.delete_chunk(chunk_id)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        try:
            return self.chunk_service.get_chunk(chunk_id)
        except ResourceNotFoundException:
            return None

    def get_document(self, document_id: str) -> Document | None:
        try:
            return self.document_service.get_document(document_id)
        except ResourceNotFoundException:
            return None

    def build_index(self, library_id: str, algorithm: str, metric: str) -> None:
        try:
            self.index_service.build_index(library_id, algorithm, metric)
        except (InvalidAlgorithmException, InvalidMetricException) as e:
            self.logger.error("Invalid index configuration: %s", e)
            raise ValueError(str(e))

    def search(
        self,
        library_id: str,
        vector: list[float],
        k: int,
        metadata_filters: dict[str, str],
    ) -> list[tuple[str, float]]:
        return self.index_service.search(library_id, vector, k, metadata_filters)

    def get_index_info(self, library_id: str) -> dict[str, str]:
        return self.index_service.get_index_info(library_id)

    def save(self, path: Path | None = None) -> Path:
        path = path or settings.data_dir / "snapshot.json"

        snapshot_data = self.repo.snapshot()
        index_metadata = self.index_service.get_index_metadata()

        data: dict[str, Any] = {**snapshot_data, "indices": index_metadata}

        path.write_text(json.dumps(data, indent=2, sort_keys=True))
        self.logger.info("Database saved to %s", path)
        return path

    def load(self, path: Path | None = None) -> None:
        path = path or settings.data_dir / "snapshot.json"

        if not path.exists():
            self.logger.info("No snapshot found at %s", path)
            return

        try:
            data = json.loads(path.read_text())
            self.repo.load_snapshot(data)

            index_metadata = data.get("indices", {})
            self.index_service.rebuild_indices(index_metadata)

            self.logger.info("Database loaded from %s", path)
        except Exception as e:
            self.logger.error("Failed to load database: %s", e)
            raise
