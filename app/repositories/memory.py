from __future__ import annotations

from typing import Optional

from app.core import ReaderWriterLock
from app.domain.models import Chunk, Document, Library
from app.repositories.base import VectorRepository


class InMemoryRepository(VectorRepository):
    def __init__(self) -> None:
        self._libraries: dict[str, Library] = {}
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, Chunk] = {}
        self._rw = ReaderWriterLock()

    def create_library(self, library: Library) -> Library:
        with self._rw.write_lock():
            self._libraries[library.id] = library
            return library

    def get_library(self, library_id: str) -> Optional[Library]:
        with self._rw.read_lock():
            return self._libraries.get(library_id)

    def list_libraries(self) -> list[Library]:
        with self._rw.read_lock():
            return list(self._libraries.values())

    def update_library(self, library: Library) -> Library:
        with self._rw.write_lock():
            self._libraries[library.id] = library
            return library

    def delete_library(self, library_id: str) -> None:
        with self._rw.write_lock():
            doc_ids = {
                d.id for d in self._documents.values() if d.library_id == library_id
            }

            chunks_to_delete = [
                c.id for c in self._chunks.values() if c.document_id in doc_ids
            ]
            for chunk_id in chunks_to_delete:
                del self._chunks[chunk_id]

            for doc_id in doc_ids:
                del self._documents[doc_id]

            self._libraries.pop(library_id, None)

    def create_document(self, document: Document) -> Document:
        with self._rw.write_lock():
            self._documents[document.id] = document
            return document

    def get_document(self, document_id: str) -> Optional[Document]:
        with self._rw.read_lock():
            return self._documents.get(document_id)

    def list_documents(self, library_id: str) -> list[Document]:
        with self._rw.read_lock():
            return [d for d in self._documents.values() if d.library_id == library_id]

    def update_document(self, document: Document) -> Document:
        with self._rw.write_lock():
            self._documents[document.id] = document
            return document

    def delete_document(self, document_id: str) -> None:
        with self._rw.write_lock():
            chunks_to_delete = [
                c.id for c in self._chunks.values() if c.document_id == document_id
            ]
            for chunk_id in chunks_to_delete:
                del self._chunks[chunk_id]

            self._documents.pop(document_id, None)

    def create_chunk(self, chunk: Chunk) -> Chunk:
        with self._rw.write_lock():
            self._chunks[chunk.id] = chunk
            return chunk

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._rw.read_lock():
            return self._chunks.get(chunk_id)

    def list_chunks(self, library_id: str) -> list[Chunk]:
        with self._rw.read_lock():
            doc_ids = {
                d.id for d in self._documents.values() if d.library_id == library_id
            }
            return [c for c in self._chunks.values() if c.document_id in doc_ids]

    def update_chunk(self, chunk: Chunk) -> Chunk:
        with self._rw.write_lock():
            self._chunks[chunk.id] = chunk
            return chunk

    def delete_chunk(self, chunk_id: str) -> None:
        with self._rw.write_lock():
            self._chunks.pop(chunk_id, None)

    def snapshot(self) -> dict[str, list[dict]]:
        with self._rw.read_lock():
            return {
                "libraries": [lib.model_dump() for lib in self._libraries.values()],
                "documents": [d.model_dump() for d in self._documents.values()],
                "chunks": [c.model_dump() for c in self._chunks.values()],
            }

    def load_snapshot(self, data: dict[str, list[dict]]) -> None:
        with self._rw.write_lock():
            self._libraries = {
                lib_dict["id"]: Library(**lib_dict)
                for lib_dict in data.get("libraries", [])
            }
            self._documents = {
                d["id"]: Document(**d) for d in data.get("documents", [])
            }
            self._chunks = {c["id"]: Chunk(**c) for c in data.get("chunks", [])}
