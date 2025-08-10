"""KD-Tree index implementation for Euclidean distance."""

from __future__ import annotations

from heapq import heappop, heappush
from typing import Optional

from app.core.constants import DistanceMetric, IndexAlgorithm
from app.vector_index import VectorIndex, euclidean_distance


class KDNode:
    """Node in a KD-Tree."""

    def __init__(
        self,
        point: list[float],
        point_id: str,
        axis: int,
        left: Optional["KDNode"],
        right: Optional["KDNode"],
    ) -> None:
        self.point = point
        self.point_id = point_id
        self.axis = axis
        self.left = left
        self.right = right


def build_kd(
    points: list[list[float]], ids: list[str], depth: int = 0
) -> Optional[KDNode]:
    """Recursively build a KD-Tree."""
    if not points:
        return None

    k = len(points[0])
    axis = depth % k

    # Sort points by the current axis
    combined = list(zip(points, ids))
    combined.sort(key=lambda x: x[0][axis])
    median = len(combined) // 2

    # Get median point
    median_point, median_id = combined[median]

    # Recursively build left and right subtrees
    left_points = [p for p, _ in combined[:median]]
    left_ids = [i for _, i in combined[:median]]
    left = build_kd(left_points, left_ids, depth + 1)

    right_points = [p for p, _ in combined[median + 1 :]]
    right_ids = [i for _, i in combined[median + 1 :]]
    right = build_kd(right_points, right_ids, depth + 1)

    return KDNode(median_point, median_id, axis, left, right)


def kd_query(
    node: Optional[KDNode],
    target: list[float],
    k: int,
    heap: list[tuple[float, str]],
) -> None:
    """Query a KD-Tree for k nearest neighbors."""
    if node is None:
        return

    # Calculate distance to current node
    dist = euclidean_distance(target, node.point)

    # Use max-heap (negative distance for max-heap behavior)
    heappush(heap, (-dist, node.point_id))
    if len(heap) > k:
        heappop(heap)

    # Determine which subtree to search first
    axis = node.axis
    diff = target[axis] - node.point[axis]

    if diff < 0:
        first, second = node.left, node.right
    else:
        first, second = node.right, node.left

    # Search the closer subtree first
    kd_query(first, target, k, heap)

    # Check if we need to search the other subtree
    if len(heap) < k or abs(diff) < -heap[0][0]:
        kd_query(second, target, k, heap)


class KDTreeIndex(VectorIndex):
    """KD-Tree index for Euclidean distance search."""

    def __init__(self) -> None:
        self._root: Optional[KDNode] = None
        self._dim: int = 0

    def build(self, vectors: list[list[float]], ids: list[str]) -> None:
        """Build the KD-Tree from vectors."""
        self._validate_inputs(vectors, ids)

        if not vectors:
            self._root = None
            self._dim = 0
            return

        self._dim = len(vectors[0])
        self._root = build_kd(vectors, ids)

    def query(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        """Query for k nearest neighbors."""
        if k <= 0:
            return []

        if self._dim and len(vector) != self._dim:
            raise ValueError("Query vector dimensionality mismatch")

        heap: list[tuple[float, str]] = []
        kd_query(self._root, vector, k, heap)

        # Sort by distance (remember we used negative distances)
        heap.sort(reverse=True)

        # Convert to similarity scores (inverse of distance)
        return [(pid, 1.0 / (1.0 + (-d))) for d, pid in heap]

    def metric(self) -> str:
        """Return the distance metric."""
        return DistanceMetric.EUCLIDEAN.value

    def kind(self) -> str:
        """Return the index type."""
        return IndexAlgorithm.KDTREE.value
