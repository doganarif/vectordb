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
        json: Any | None = None,
        headers: Dict[str, str] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        r = requests.request(
            method, url, json=json, timeout=self.timeout, headers=headers
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

    def search(
        self,
        library_id: str,
        vector: List[float],
        k: int = 5,
        metadata_filters: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        payload = {"vector": vector, "k": k, "metadata_filters": metadata_filters or {}}
        return self._request("POST", f"/libraries/{library_id}/searches", json=payload)

    def save(self) -> Dict[str, Any]:
        return self._request("POST", "/admin/snapshots")

    def load(self) -> Dict[str, Any]:
        return self._request("PUT", "/admin/snapshots")

    def embed_cohere(
        self, text: str, model: Optional[str] = None, input_type: Optional[str] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"text": text}
        if model is not None:
            payload["model"] = model
        if input_type is not None:
            payload["input_type"] = input_type
        return self._request(
            "POST",
            "/embeddings",
            json=payload,
            headers={"Accept": "application/json"},
        )
