"""Linear search index implementation."""

from app.core.constants import DistanceMetric, IndexAlgorithm
from app.vector_index import VectorIndex, cosine_similarity, euclidean_distance


class LinearIndex(VectorIndex):
    """Linear search index supporting multiple metrics."""

    def __init__(self, metric: str = "cosine") -> None:
        # Store as enum value for consistency
        self._metric = DistanceMetric(metric).value
        self._vectors: list[list[float]] = []
        self._ids: list[str] = []

    def build(self, vectors: list[list[float]], ids: list[str]) -> None:
        self._validate_inputs(vectors, ids)
        self._vectors = vectors
        self._ids = ids

    def query(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        if not self._vectors or k <= 0:
            return []

        self._validate_query_dim(vector, self._vectors)
        scores: list[tuple[str, float]] = []
        if self._metric == DistanceMetric.COSINE.value:
            for cid, vec in zip(self._ids, self._vectors):
                scores.append((cid, cosine_similarity(vector, vec)))
            scores.sort(key=lambda x: x[1], reverse=True)
        else:
            for cid, vec in zip(self._ids, self._vectors):
                dist = euclidean_distance(vector, vec)
                sim = 1.0 / (1.0 + dist)
                scores.append((cid, sim))
            scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def metric(self) -> str:
        return self._metric

    def kind(self) -> str:
        return IndexAlgorithm.LINEAR.value
