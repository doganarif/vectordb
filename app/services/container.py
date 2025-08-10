from functools import lru_cache

from app.repositories import InMemoryRepository
from app.repositories.base import VectorRepository

_repository_instance: VectorRepository | None = None


def get_repository() -> VectorRepository:
    global _repository_instance
    if _repository_instance is None:
        _repository_instance = InMemoryRepository()
    return _repository_instance


def set_repository(repo: VectorRepository) -> None:
    global _repository_instance
    _repository_instance = repo


def reset_repository() -> None:
    global _repository_instance
    _repository_instance = None


@lru_cache(maxsize=1)
def get_service():
    from app.services.vector_service import VectorDBService

    return VectorDBService(repo=get_repository())


def reset_service_cache() -> None:
    get_service.cache_clear()
    reset_repository()
