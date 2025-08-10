import logging
from typing import Optional

from app.core.exceptions import ResourceNotFoundException
from app.domain.models import Library
from app.repositories.base import VectorRepository


class LibraryService:
    def __init__(self, repository: VectorRepository, index_service=None) -> None:
        self.repository = repository
        self.index_service = index_service
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_library(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Library:
        library = Library(
            name=name,
            description=description,
            metadata=metadata or {},
        )
        created = self.repository.create_library(library)
        self.logger.info(f"Library created: {created.id}")
        return created

    def get_library(self, library_id: str) -> Library:
        library = self.repository.get_library(library_id)
        if not library:
            raise ResourceNotFoundException("Library", library_id)
        return library

    def list_libraries(self) -> list[Library]:
        return self.repository.list_libraries()

    def update_library(
        self,
        library_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Library:
        library = self.get_library(library_id)

        if name is not None:
            library.name = name
        if description is not None:
            library.description = description
        if metadata is not None:
            library.metadata = metadata

        updated = self.repository.update_library(library)
        self.logger.info(f"Library updated: {updated.id}")
        return updated

    def delete_library(self, library_id: str) -> None:
        self.repository.delete_library(library_id)
        self.logger.info(f"Library deleted: {library_id}")

    def delete_library_cascade(self, library_id: str) -> None:
        """Delete a library and clean up its index.

        This is a coordinated operation that requires both library
        deletion and index cleanup.
        """
        self.delete_library(library_id)
        if self.index_service:
            self.index_service.clear_index(library_id)
