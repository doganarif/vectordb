"""LSH (Locality Sensitive Hashing) index implementation for cosine similarity."""

import random

from app.core import settings
from app.core.constants import DistanceMetric, IndexAlgorithm
from app.vector_index import VectorIndex, cosine_similarity


class LSHIndex(VectorIndex):
    """LSH index using random hyperplanes for cosine similarity."""

    def __init__(
        self,
        num_planes: int = settings.lsh_num_planes,
        num_tables: int = settings.lsh_num_tables,
        seed: int = settings.lsh_seed,
    ) -> None:
        self._num_planes = num_planes
        self._num_tables = num_tables
        self._tables: list[dict[int, list[tuple[str, list[float]]]]] = []
        self._planes: list[list[list[float]]] = []
        self._seed = seed
        self._dim: int = 0

    def _hash(self, vec: list[float], planes: list[list[float]]) -> int:
        """Compute hash signature for a vector using hyperplanes.

        Uses bitwise operations for efficiency instead of string concatenation.
        """
        signature = 0
        for i, plane in enumerate(planes):
            if sum(v * p for v, p in zip(vec, plane)) >= 0:
                signature |= 1 << i
        return signature

    def build(self, vectors: list[list[float]], ids: list[str]) -> None:
        """Build LSH tables from vectors."""
        self._validate_inputs(vectors, ids)

        if not vectors:
            self._tables = []
            self._planes = []
            self._dim = 0
            return

        dim = len(vectors[0])
        self._dim = dim
        rng = random.Random(self._seed)

        # Generate random hyperplanes for each table
        self._planes = []
        for _ in range(self._num_tables):
            table_planes = []
            for _ in range(self._num_planes):
                # Generate random unit vector as hyperplane normal
                plane = [rng.gauss(0, 1) for _ in range(dim)]
                norm = sum(x * x for x in plane) ** 0.5
                plane = [x / norm for x in plane]
                table_planes.append(plane)
            self._planes.append(table_planes)

        # Build hash tables
        self._tables = [{} for _ in range(self._num_tables)]
        for vec, vec_id in zip(vectors, ids):
            for i, planes in enumerate(self._planes):
                signature = self._hash(vec, planes)
                if signature not in self._tables[i]:
                    self._tables[i][signature] = []
                self._tables[i][signature].append((vec_id, vec))

    def query(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        """Query for k nearest neighbors with multi-probe."""
        if k <= 0:
            return []

        if self._dim and len(vector) != self._dim:
            raise ValueError("Query vector dimensionality mismatch")

        # Collect candidates from all tables with multi-probe
        candidates: dict[str, list[float]] = {}
        for i, planes in enumerate(self._planes):
            signature = self._hash(vector, planes)

            # Query exact signature
            if signature in self._tables[i]:
                for vec_id, vec in self._tables[i][signature]:
                    if vec_id not in candidates:
                        candidates[vec_id] = vec

            # Multi-probe: flip 1-2 bits for better recall
            for bit_flip in range(min(2, self._num_planes)):
                # Flip bit at position bit_flip
                probed_sig = signature ^ (1 << bit_flip)
                if probed_sig in self._tables[i]:
                    for vec_id, vec in self._tables[i][probed_sig]:
                        if vec_id not in candidates:
                            candidates[vec_id] = vec

        # Score all candidates
        scores = [
            (vec_id, cosine_similarity(vector, vec))
            for vec_id, vec in candidates.items()
        ]

        # Sort by similarity and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def metric(self) -> str:
        """Return the distance metric."""
        return DistanceMetric.COSINE.value

    def kind(self) -> str:
        """Return the index type."""
        return IndexAlgorithm.LSH.value
