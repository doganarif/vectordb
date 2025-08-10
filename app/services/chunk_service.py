import logging

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
        metadata: dict[str, str] | None = None,
    ) -> Chunk:
        document = self.repository.get_document(document_id)
        if not document:
            raise ResourceNotFoundException(f"Document not found: {document_id}")
        if document.library_id != library_id:
            raise ResourceNotFoundException(
                f"Document {document_id} not in library {library_id}"
            )

        self._validate_embedding_dimensions(library_id, embedding)

        chunk = Chunk(
            document_id=document_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
        )
        created = self.repository.create_chunk(chunk)
        self.logger.info("Chunk created: %s in document %s", created.id, document_id)
        return created

    def get_chunk(self, chunk_id: str) -> Chunk:
        chunk = self.repository.get_chunk(chunk_id)
        if not chunk:
            raise ResourceNotFoundException(f"Chunk not found: {chunk_id}")
        return chunk

    def list_chunks(self, library_id: str) -> list[Chunk]:
        return self.repository.list_chunks(library_id)

    def update_chunk(
        self,
        chunk_id: str,
        text: str | None = None,
        embedding: list[float] | None = None,
        metadata: dict[str, str] | None = None,
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
        self.logger.info("Chunk updated: %s", updated.id)
        return updated

    def delete_chunk(self, chunk_id: str) -> None:
        self.repository.delete_chunk(chunk_id)
        self.logger.info("Chunk deleted: %s", chunk_id)

    def _validate_embedding_dimensions(
        self,
        library_id: str,
        embedding: list[float],
    ) -> None:
        if not embedding:
            return

        existing_chunks = self.repository.list_chunks(library_id)
        if existing_chunks and existing_chunks[0].embedding:
            expected_dim = len(existing_chunks[0].embedding)
            if len(embedding) != expected_dim:
                raise DimensionalityMismatchException(expected_dim, len(embedding))
