from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


class VectorDBClient:
    def __init__(
        self, base_url: str = "http://127.0.0.1:8000", timeout: int = 30
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        r = requests.request(
            method, url, json=json, params=params, timeout=self.timeout, headers=headers
        )
        r.raise_for_status()
        if r.content and r.headers.get("Content-Type", "").startswith(
            "application/json"
        ):
            return r.json()
        return None

    def create_library(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        payload = {"name": name, "description": description, "metadata": metadata or {}}
        return self._request("POST", "/libraries/", json=payload)

    def list_libraries(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/libraries")

    def get_library(self, library_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/libraries/{library_id}")

    def update_library(self, library_id: str, **fields: Any) -> Dict[str, Any]:
        return self._request("PATCH", f"/libraries/{library_id}", json=fields)

    def delete_library(self, library_id: str) -> None:
        self._request("DELETE", f"/libraries/{library_id}")

    def create_document(
        self,
        library_id: str,
        title: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "title": title,
            "description": description,
            "metadata": metadata or {},
        }
        return self._request("POST", f"/libraries/{library_id}/documents", json=payload)

    def list_documents(self, library_id: str) -> List[Dict[str, Any]]:
        return self._request("GET", f"/libraries/{library_id}/documents")

    def update_document(
        self, library_id: str, document_id: str, **fields: Any
    ) -> Dict[str, Any]:
        return self._request(
            "PATCH", f"/libraries/{library_id}/documents/{document_id}", json=fields
        )

    def delete_document(self, library_id: str, document_id: str) -> None:
        self._request("DELETE", f"/libraries/{library_id}/documents/{document_id}")

    def create_chunk(
        self,
        library_id: str,
        document_id: str,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "document_id": document_id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
        }
        return self._request("POST", f"/libraries/{library_id}/chunks", json=payload)

    def list_chunks(self, library_id: str) -> List[Dict[str, Any]]:
        return self._request("GET", f"/libraries/{library_id}/chunks")

    def update_chunk(
        self, library_id: str, chunk_id: str, **fields: Any
    ) -> Dict[str, Any]:
        return self._request(
            "PATCH", f"/libraries/{library_id}/chunks/{chunk_id}", json=fields
        )

    def delete_chunk(self, library_id: str, chunk_id: str) -> None:
        self._request("DELETE", f"/libraries/{library_id}/chunks/{chunk_id}")

    def build_index(
        self, library_id: str, algorithm: str, metric: str
    ) -> Dict[str, Any]:
        payload = {"algorithm": algorithm, "metric": metric}
        return self._request("PUT", f"/libraries/{library_id}/index", json=payload)

    def get_index(self, library_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/libraries/{library_id}/index")

    def delete_index(self, library_id: str) -> None:
        self._request("DELETE", f"/libraries/{library_id}/index")

    def search(
        self,
        library_id: str,
        vector: List[float],
        k: int = 10,
        metadata_filters: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Search for similar chunks using POST endpoint with JSON payload.

        Args:
            library_id: The library to search in
            vector: Query vector
            k: Number of results to return (default: 10)
            metadata_filters: Optional metadata filters

        Returns:
            Search response with matching chunks
        """
        payload = {
            "vector": vector,
            "k": k,
            "metadata_filters": metadata_filters or {},
        }

        return self._request(
            "POST", f"/libraries/{library_id}/chunks/search", json=payload
        )

    def create_snapshot(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Create a new database snapshot.

        Args:
            name: Optional name for the snapshot

        Returns:
            Snapshot details including ID and path
        """
        payload = {"name": name} if name else {}
        return self._request("POST", "/admin/snapshots", json=payload)

    def list_snapshots(self) -> Dict[str, Any]:
        """List all available snapshots.

        Returns:
            List of available snapshots with details
        """
        return self._request("GET", "/admin/snapshots")

    def restore_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Restore database from a specific snapshot.

        Args:
            snapshot_id: The snapshot ID to restore from

        Returns:
            Restore operation status
        """
        return self._request("POST", f"/admin/snapshots/{snapshot_id}/restore")

    def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete a specific snapshot.

        Args:
            snapshot_id: The snapshot ID to delete
        """
        self._request("DELETE", f"/admin/snapshots/{snapshot_id}")

    # Deprecated methods for backward compatibility
    def save(self) -> Dict[str, Any]:
        """Deprecated: Use create_snapshot() instead."""
        import warnings

        warnings.warn(
            "save() is deprecated, use create_snapshot() instead", DeprecationWarning
        )
        return self.create_snapshot()

    def load(self) -> Dict[str, Any]:
        """Deprecated: Use restore_snapshot() instead."""
        import warnings

        warnings.warn(
            "load() is deprecated, use restore_snapshot() instead", DeprecationWarning
        )
        # Try to restore the latest snapshot
        snapshots = self.list_snapshots()
        if snapshots and "snapshots" in snapshots and snapshots["snapshots"]:
            latest = snapshots["snapshots"][0]
            return self.restore_snapshot(latest["id"])
        raise ValueError("No snapshots available to restore")

    def embed_cohere(self, text: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"text": text}
        return self._request(
            "POST",
            "/embeddings",
            json=payload,
            headers={"Accept": "application/json"},
        )
