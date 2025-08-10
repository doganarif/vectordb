"""Application constants."""

from __future__ import annotations

from enum import Enum

# Search constants
DEFAULT_SEARCH_MULTIPLIER = 3
MAX_SEARCH_BUFFER = 50
MIN_SEARCH_K = 1


# Index algorithms
class IndexAlgorithm(str, Enum):
    """Enumeration of available index algorithms."""

    LINEAR = "linear"
    KDTREE = "kdtree"
    LSH = "lsh"


# Distance metrics
class DistanceMetric(str, Enum):
    """Enumeration of available distance metrics."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


# Algorithm-metric compatibility
ALGORITHM_METRICS = {
    IndexAlgorithm.LINEAR: [DistanceMetric.COSINE, DistanceMetric.EUCLIDEAN],
    IndexAlgorithm.KDTREE: [DistanceMetric.EUCLIDEAN],
    IndexAlgorithm.LSH: [DistanceMetric.COSINE],
}

# HTTP configuration
HTTP_TIMEOUT = 30.0  # Default HTTP timeout in seconds
HTTP_POOL_SIZE = 10  # Maximum number of connections
HTTP_KEEPALIVE_CONNECTIONS = 5  # Maximum keepalive connections
HTTP_KEEPALIVE_EXPIRY = 30.0  # Keepalive expiry in seconds

# Embedding API configuration
EMBEDDING_MAX_RETRIES = 3
EMBEDDING_RETRY_DELAY = 1.0  # Initial retry delay in seconds
EMBEDDING_RETRY_BACKOFF = 2.0  # Exponential backoff multiplier

# Validation limits
MAX_TEXT_LENGTH = 10000
MIN_TEXT_LENGTH = 1
