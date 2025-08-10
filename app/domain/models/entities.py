from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    text: str
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    library_id: str
    title: str
    description: Optional[str] = None
    metadata: dict[str, str] = Field(default_factory=dict)


class Library(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    metadata: dict[str, str] = Field(default_factory=dict)
    embedding_dim: Optional[int] = None
