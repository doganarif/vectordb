import logging
from contextlib import contextmanager
from typing import Iterator

from app.core import ReaderWriterLock, settings
from app.core.constants import (
    ALGORITHM_METRICS,
    AVAILABLE_ALGORITHMS,
    DEFAULT_SEARCH_MULTIPLIER,
    INDEX_KDTREE,
    INDEX_LINEAR,
    INDEX_LSH,
    MAX_SEARCH_BUFFER,
)
from app.core.exceptions import (
    InvalidAlgorithmException,
    InvalidMetricException,
)
from app.repositories.base import VectorRepository
from app.vector_index import KDTreeIndex, LinearIndex, LSHIndex, VectorIndex


class IndexService:
    def __init__(self, repository: VectorRepository) -> None:
        self.repository = repository
        self.logger = logging.getLogger(self.__class__.__name__)
        self._indices: dict[str, VectorIndex] = {}
        self._index_meta: dict[str, dict[str, str]] = {}
        self._lock = ReaderWriterLock()

    def build_index(
        self,
        library_id: str,
        algorithm: str,
        metric: str,
    ) -> None:
        algorithm = algorithm.lower()
        metric = metric.lower()

        index = self._create_index(algorithm, metric)

        chunks = self.repository.list_chunks(library_id)
        if chunks:
            vectors = [c.embedding for c in chunks]
            ids = [c.id for c in chunks]
            index.build(vectors, ids)

        with self._lock.write_lock():
            self._indices[library_id] = index
            self._index_meta[library_id] = {
                "algorithm": index.kind(),
                "metric": index.metric(),
            }

        self.logger.info(
            "Index built for library %s: algorithm=%s, metric=%s, chunks=%d",
            library_id,
            algorithm,
            metric,
            len(chunks),
        )

    def search(
        self,
        library_id: str,
        vector: list[float],
        k: int,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[tuple[str, float]]:
        if k <= 0:
            return []

        index = self._get_or_create_index(library_id)
        if not index:
            return []

        query_k = self._calculate_query_k(k)
        results = index.query(vector, query_k)

        if metadata_filters:
            results = self._apply_metadata_filters(results, metadata_filters)

        return results[:k]

    def get_index_info(self, library_id: str) -> dict[str, str]:
        with self._lock.read_lock():
            meta = self._index_meta.get(library_id)

        if not meta:
            return {
                "library_id": library_id,
                "algorithm": "none",
                "metric": settings.default_metric,
            }

        return {"library_id": library_id, **meta}

    def clear_index(self, library_id: str) -> None:
        with self._lock.write_lock():
            self._indices.pop(library_id, None)
            self._index_meta.pop(library_id, None)

        self.logger.info("Index cleared for library %s", library_id)

    def get_index_metadata(self) -> dict[str, dict[str, str]]:
        with self._lock.read_lock():
            return dict(self._index_meta)

    def rebuild_indices(self, metadata: dict[str, dict[str, str]]) -> None:
        for library_id, meta in metadata.items():
            try:
                self.build_index(
                    library_id,
                    meta.get("algorithm", settings.default_index),
                    meta.get("metric", settings.default_metric),
                )
            except Exception as e:
                self.logger.error(
                    "Failed to rebuild index for library %s: %s",
                    library_id,
                    e,
                )

    @contextmanager
    def read_lock(self) -> Iterator[None]:
        with self._lock.read_lock():
            yield

    def _create_index(self, algorithm: str, metric: str) -> VectorIndex:
        if algorithm not in AVAILABLE_ALGORITHMS:
            raise InvalidAlgorithmException(algorithm, AVAILABLE_ALGORITHMS)

        supported_metrics = ALGORITHM_METRICS.get(algorithm, [])
        if metric not in supported_metrics:
            raise InvalidMetricException(algorithm, metric, supported_metrics)

        if algorithm == INDEX_LINEAR:
            return LinearIndex(metric=metric)
        elif algorithm == INDEX_KDTREE:
            return KDTreeIndex()
        elif algorithm == INDEX_LSH:
            return LSHIndex()
        else:
            raise InvalidAlgorithmException(algorithm, AVAILABLE_ALGORITHMS)

    def _get_or_create_index(self, library_id: str) -> VectorIndex | None:
        with self._lock.read_lock():
            index = self._indices.get(library_id)

        if not index:
            chunks = self.repository.list_chunks(library_id)
            if not chunks:
                return None

            index = LinearIndex(metric=settings.default_metric)
            vectors = [c.embedding for c in chunks]
            ids = [c.id for c in chunks]
            index.build(vectors, ids)

        return index

    def _calculate_query_k(self, k: int) -> int:
        return max(k, min(k * DEFAULT_SEARCH_MULTIPLIER, k + MAX_SEARCH_BUFFER))

    def _apply_metadata_filters(
        self,
        results: list[tuple[str, float]],
        filters: dict[str, str],
    ) -> list[tuple[str, float]]:
        filtered = []
        for chunk_id, score in results:
            chunk = self.repository.get_chunk(chunk_id)
            if not chunk:
                continue

            if all(chunk.metadata.get(key) == value for key, value in filters.items()):
                filtered.append((chunk_id, score))

        return filtered
