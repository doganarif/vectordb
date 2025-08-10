import logging

from app.core.exceptions import ResourceNotFoundException
from app.domain.models import Library
from app.repositories.base import VectorRepository


class LibraryService:
    def __init__(self, repository: VectorRepository) -> None:
        self.repository = repository
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_library(
        self,
        name: str,
        description: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Library:
        library = Library(
            name=name,
            description=description,
            metadata=metadata or {},
        )
        created = self.repository.create_library(library)
        self.logger.info("Library created: %s", created.id)
        return created

    def get_library(self, library_id: str) -> Library:
        library = self.repository.get_library(library_id)
        if not library:
            raise ResourceNotFoundException(f"Library not found: {library_id}")
        return library

    def list_libraries(self) -> list[Library]:
        return self.repository.list_libraries()

    def update_library(
        self,
        library_id: str,
        name: str | None = None,
        description: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Library:
        library = self.get_library(library_id)

        if name is not None:
            library.name = name
        if description is not None:
            library.description = description
        if metadata is not None:
            library.metadata = metadata

        updated = self.repository.update_library(library)
        self.logger.info("Library updated: %s", updated.id)
        return updated

    def delete_library(self, library_id: str) -> None:
        self.repository.delete_library(library_id)
        self.logger.info("Library deleted: %s", library_id)
