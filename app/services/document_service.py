import logging
from typing import Optional

from app.core.exceptions import ResourceNotFoundException
from app.domain.models import Document
from app.repositories.base import VectorRepository


class DocumentService:
    def __init__(self, repository: VectorRepository) -> None:
        self.repository = repository
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_document(
        self,
        library_id: str,
        title: str,
        description: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Document:
        if not self.repository.get_library(library_id):
            raise ResourceNotFoundException("Library", library_id)

        document = Document(
            library_id=library_id,
            title=title,
            description=description,
            metadata=metadata or {},
        )
        created = self.repository.create_document(document)
        self.logger.info(f"Document created: {created.id} in library {library_id}")
        return created

    def get_document(self, document_id: str) -> Document:
        document = self.repository.get_document(document_id)
        if not document:
            raise ResourceNotFoundException("Document", document_id)
        return document

    def list_documents(self, library_id: str) -> list[Document]:
        return self.repository.list_documents(library_id)

    def update_document(
        self,
        document_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Document:
        document = self.get_document(document_id)

        if title is not None:
            document.title = title
        if description is not None:
            document.description = description
        if metadata is not None:
            document.metadata = metadata

        updated = self.repository.update_document(document)
        self.logger.info(f"Document updated: {updated.id}")
        return updated

    def delete_document(self, document_id: str) -> None:
        self.repository.delete_document(document_id)
        self.logger.info(f"Document deleted: {document_id}")
