import logging
from typing import Optional

from app.core.exceptions import (
    DimensionalityMismatchException,
    ResourceNotFoundException,
)
from app.domain.models import Chunk
from app.repositories.base import VectorRepository


class ChunkService:
    def __init__(self, repository: VectorRepository) -> None:
        self.repository = repository
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_chunk(
        self,
        library_id: str,
        document_id: str,
        text: str,
        embedding: list[float],
        metadata: Optional[dict[str, str]] = None,
    ) -> Chunk:
        document = self.repository.get_document(document_id)
        if not document:
            raise ResourceNotFoundException("Document", document_id)
        if document.library_id != library_id:
            raise ResourceNotFoundException("Document", document_id)

        # Validate and set embedding dimensions
        self._validate_embedding_dimensions(library_id, embedding)

        chunk = Chunk(
            document_id=document_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
        )
        created = self.repository.create_chunk(chunk)
        self.logger.info(f"Chunk created: {created.id} in document {document_id}")
        return created

    def get_chunk(self, chunk_id: str) -> Chunk:
        chunk = self.repository.get_chunk(chunk_id)
        if not chunk:
            raise ResourceNotFoundException("Chunk", chunk_id)
        return chunk

    def list_chunks(self, library_id: str) -> list[Chunk]:
        return self.repository.list_chunks(library_id)

    def update_chunk(
        self,
        chunk_id: str,
        text: Optional[str] = None,
        embedding: Optional[list[float]] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Chunk:
        chunk = self.get_chunk(chunk_id)

        if embedding is not None:
            document = self.repository.get_document(chunk.document_id)
            if document:
                self._validate_embedding_dimensions(document.library_id, embedding)
            chunk.embedding = embedding

        if text is not None:
            chunk.text = text
        if metadata is not None:
            chunk.metadata = metadata

        updated = self.repository.update_chunk(chunk)
        self.logger.info(f"Chunk updated: {updated.id}")
        return updated

    def delete_chunk(self, chunk_id: str) -> None:
        self.repository.delete_chunk(chunk_id)
        self.logger.info(f"Chunk deleted: {chunk_id}")

    def _validate_embedding_dimensions(
        self,
        library_id: str,
        embedding: list[float],
    ) -> None:
        if not embedding:
            return

        library = self.repository.get_library(library_id)
        if not library:
            return

        # Check library-level embedding_dim first
        if library.embedding_dim is not None:
            if len(embedding) != library.embedding_dim:
                raise DimensionalityMismatchException(
                    library.embedding_dim, len(embedding)
                )
        else:
            # Set library embedding_dim on first non-empty vector
            library.embedding_dim = len(embedding)
            self.repository.update_library(library)
            self.logger.info(
                f"Set library {library_id} embedding_dim to {len(embedding)}"
            )
