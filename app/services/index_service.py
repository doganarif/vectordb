import logging
from contextlib import contextmanager
from typing import Iterator, Optional

from app.core import ReaderWriterLock, settings
from app.core.constants import (
    ALGORITHM_METRICS,
    DEFAULT_SEARCH_MULTIPLIER,
    MAX_SEARCH_BUFFER,
    DistanceMetric,
    IndexAlgorithm,
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
            # Filter out any chunks with empty embeddings (defensive)
            valid_pairs = [(c.embedding, c.id) for c in chunks if c.embedding]
            if valid_pairs:
                vectors, ids = zip(*valid_pairs)
                index.build(list(vectors), list(ids))
            else:
                # All embeddings are empty, build empty index
                index.build([], [])

        with self._lock.write_lock():
            self._indices[library_id] = index
            self._index_meta[library_id] = {
                "algorithm": index.kind(),
                "metric": index.metric(),
            }

        self.logger.info(
            f"Index built for library {library_id}: algorithm={algorithm}, metric={metric}, chunks={len(chunks)}"
        )

    def search(
        self,
        library_id: str,
        vector: list[float],
        k: int,
        metadata_filters: Optional[dict[str, str]] = None,
    ) -> list[tuple[str, float]]:
        if k <= 0:
            return []

        index = self._get_or_create_index(library_id)
        if not index:
            return []

        # Increase query_k when filters are present to ensure we get enough results
        query_k = self._calculate_query_k(k, has_filters=bool(metadata_filters))
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

        self.logger.info(f"Index cleared for library {library_id}")

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
                    f"Failed to rebuild index for library {library_id}: {e}"
                )

    @contextmanager
    def read_lock(self) -> Iterator[None]:
        with self._lock.read_lock():
            yield

    def _create_index(self, algorithm: str, metric: str) -> VectorIndex:
        # Convert string to enum, validate it exists
        try:
            algo_enum = IndexAlgorithm(algorithm)
        except ValueError:
            raise InvalidAlgorithmException(
                algorithm, [algo.value for algo in IndexAlgorithm]
            )

        # Convert string to enum for metric
        try:
            metric_enum = DistanceMetric(metric)
        except ValueError:
            supported = ALGORITHM_METRICS.get(algo_enum, [])
            supported_values = [m.value for m in supported]
            raise InvalidMetricException(algorithm, metric, supported_values)

        # Check compatibility
        supported_metrics = ALGORITHM_METRICS.get(algo_enum, [])
        if metric_enum not in supported_metrics:
            supported_values = [m.value for m in supported_metrics]
            raise InvalidMetricException(algorithm, metric, supported_values)

        # Create the appropriate index
        if algo_enum == IndexAlgorithm.LINEAR:
            return LinearIndex(metric=metric)
        elif algo_enum == IndexAlgorithm.KDTREE:
            return KDTreeIndex()
        elif algo_enum == IndexAlgorithm.LSH:
            return LSHIndex()
        else:
            raise InvalidAlgorithmException(
                algorithm, [algo.value for algo in IndexAlgorithm]
            )

    def _get_or_create_index(self, library_id: str) -> Optional[VectorIndex]:
        with self._lock.read_lock():
            index = self._indices.get(library_id)

        if not index:
            chunks = self.repository.list_chunks(library_id)
            if not chunks:
                return None

            # Filter out any chunks with empty embeddings (defensive)
            valid_pairs = [(c.embedding, c.id) for c in chunks if c.embedding]
            if not valid_pairs:
                return None

            index = LinearIndex(metric=settings.default_metric)
            vectors, ids = zip(*valid_pairs)
            index.build(list(vectors), list(ids))

            # Cache the fallback index
            with self._lock.write_lock():
                self._indices[library_id] = index
                self._index_meta[library_id] = {
                    "algorithm": index.kind(),
                    "metric": index.metric(),
                }

        return index

    def _calculate_query_k(self, k: int, has_filters: bool = False) -> int:
        multiplier = (
            DEFAULT_SEARCH_MULTIPLIER * 2 if has_filters else DEFAULT_SEARCH_MULTIPLIER
        )
        buffer = MAX_SEARCH_BUFFER * 2 if has_filters else MAX_SEARCH_BUFFER
        return max(k, min(k * multiplier, k + buffer))

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
