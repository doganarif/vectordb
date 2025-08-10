"""Base classes and utilities for vector indices."""

import math
from abc import ABC, abstractmethod


def dot(a: list[float], b: list[float]) -> float:
    """Calculate dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    """Calculate L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Returns 0.0 for zero vectors instead of using epsilon.
    """
    na = norm(a)
    nb = norm(b)

    # More Pythonic: handle edge case explicitly
    if na == 0.0 or nb == 0.0:
        return 0.0

    return dot(a, b) / (na * nb)


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Calculate Euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class VectorIndex(ABC):
    """Abstract base class for vector indices."""

    @abstractmethod
    def build(self, vectors: list[list[float]], ids: list[str]) -> None:
        """Build the index from vectors and IDs."""
        ...

    @abstractmethod
    def query(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        """Query the index for k nearest neighbors."""
        ...

    @abstractmethod
    def metric(self) -> str:
        """Return the distance metric used."""
        ...

    @abstractmethod
    def kind(self) -> str:
        """Return the index algorithm name."""
        ...

    def _validate_inputs(self, vectors: list[list[float]], ids: list[str]) -> None:
        """Validate input vectors and IDs."""
        if len(vectors) != len(ids):
            raise ValueError("Vectors and ids must have the same length")
        if vectors and any(len(vec) != len(vectors[0]) for vec in vectors):
            raise ValueError("All vectors must have the same dimensionality")

    def _validate_query_dim(
        self, vector: list[float], vectors: list[list[float]]
    ) -> None:
        """Validate query vector dimensions."""
        if vectors and len(vector) != len(vectors[0]):
            raise ValueError("Query vector dimensionality mismatch")
