"""Service layer."""

from app.services.container import get_service
from app.services.vector_service import VectorDBService

__all__ = ["VectorDBService", "get_service"]
