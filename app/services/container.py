"""Dependency injection container for services.

Following Python's simplicity principle, we use a functional approach
with lru_cache for singleton behavior instead of complex patterns.
"""

from functools import lru_cache
from typing import Optional

from app.repositories import InMemoryRepository
from app.repositories.base import VectorRepository


class ServiceContainer:
    """Simple service container for dependency injection.

    This provides a clean, testable way to manage dependencies
    without using global state.
    """

    def __init__(self, repository: Optional[VectorRepository] = None):
        self.repository = repository or InMemoryRepository()

    @lru_cache(maxsize=1)
    def get_service(self):
        """Get the main VectorDB service instance."""
        from app.services.vector_service import VectorDBService

        return VectorDBService(repo=self.repository)

    def reset(self):
        """Reset cached service instance."""
        self.get_service.cache_clear()


# Default container instance
_default_container = ServiceContainer()


def get_repository() -> VectorRepository:
    """Get the current repository instance."""
    return _default_container.repository


def set_repository(repo: VectorRepository) -> None:
    """Set a custom repository instance."""
    _default_container.repository = repo
    _default_container.reset()


def get_service():
    """Get the main VectorDB service instance."""
    return _default_container.get_service()


def reset_service_cache() -> None:
    """Reset the service cache for testing."""
    _default_container.reset()
