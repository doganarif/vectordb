import logging
from typing import Optional

from app.repositories import InMemoryRepository
from app.repositories.base import VectorRepository
from app.services.chunk_service import ChunkService
from app.services.document_service import DocumentService
from app.services.index_service import IndexService
from app.services.library_service import LibraryService
from app.services.snapshot_service import SnapshotService


class VectorDBService:
    def __init__(self, repo: Optional[VectorRepository] = None) -> None:
        self.repository = repo or InMemoryRepository()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize services as public attributes
        self.indices = IndexService(self.repository)
        self.libraries = LibraryService(self.repository, self.indices)
        self.documents = DocumentService(self.repository)
        self.chunks = ChunkService(self.repository)
        self.snapshots = SnapshotService(self.repository, self.indices)
