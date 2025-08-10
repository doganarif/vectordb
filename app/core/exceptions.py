"""Custom exceptions for the vector database application."""


class VectorDBException(Exception):
    """Base exception for vector database operations."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ResourceNotFoundException(VectorDBException):
    """Raised when a requested resource is not found."""

    pass


class DimensionalityMismatchException(VectorDBException):
    """Raised when vector dimensions don't match."""

    def __init__(self, expected: int, got: int) -> None:
        message = f"Embedding dimensionality mismatch: expected {expected}, got {got}"
        super().__init__(message, {"expected": expected, "got": got})
        self.expected = expected
        self.got = got


class IndexNotBuiltException(VectorDBException):
    """Raised when trying to search without a built index."""

    pass


class InvalidMetricException(VectorDBException):
    """Raised when an invalid metric is specified for an index type."""

    def __init__(self, algorithm: str, metric: str, supported: list[str]) -> None:
        message = f"{algorithm} does not support metric '{metric}'. Supported: {', '.join(supported)}"
        super().__init__(
            message, {"algorithm": algorithm, "metric": metric, "supported": supported}
        )


class InvalidAlgorithmException(VectorDBException):
    """Raised when an unknown indexing algorithm is specified."""

    def __init__(self, algorithm: str, available: list[str]) -> None:
        message = (
            f"Unknown index algorithm '{algorithm}'. Available: {', '.join(available)}"
        )
        super().__init__(message, {"algorithm": algorithm, "available": available})
