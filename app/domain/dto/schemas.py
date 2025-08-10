from pydantic import BaseModel, Field, field_validator
from pydantic.config import ConfigDict

from app.core.constants import MAX_TEXT_LENGTH, MIN_TEXT_LENGTH


def _validate_embedding(values: list[float]) -> list[float]:
    if not values:
        return values
    if not all(isinstance(x, (int, float)) for x in values):
        raise ValueError("Embedding must be a list of numbers")
    return [float(x) for x in values]


def _validate_optional_embedding(values: list[float] | None) -> list[float] | None:
    if values is None:
        return values
    return _validate_embedding(values)


def _sanitize_metadata(metadata: dict[str, str]) -> dict[str, str]:
    return {k.strip(): v.strip() for k, v in metadata.items() if k.strip()}


class CreateLibraryDTO(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(None, max_length=1000)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return v.strip()

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, str]) -> dict[str, str]:
        return _sanitize_metadata(v)


class UpdateLibraryDTO(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, max_length=1000)
    metadata: dict[str, str] | None = Field(None)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        return v.strip() if v else None

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        return _sanitize_metadata(v) if v else None


class CreateDocumentDTO(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(None, max_length=1000)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        return v.strip()

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, str]) -> dict[str, str]:
        return _sanitize_metadata(v)


class UpdateDocumentDTO(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, max_length=1000)
    metadata: dict[str, str] | None = Field(None)

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str | None) -> str | None:
        return v.strip() if v else None

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        return _sanitize_metadata(v) if v else None


class CreateChunkDTO(BaseModel):
    document_id: str = Field(...)
    text: str = Field(..., min_length=MIN_TEXT_LENGTH, max_length=MAX_TEXT_LENGTH)
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        return v.strip()

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: list[float]) -> list[float]:
        return _validate_embedding(v)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, str]) -> dict[str, str]:
        return _sanitize_metadata(v)

    model_config = ConfigDict(extra="ignore")


class UpdateChunkDTO(BaseModel):
    text: str | None = Field(
        None, min_length=MIN_TEXT_LENGTH, max_length=MAX_TEXT_LENGTH
    )
    embedding: list[float] | None = Field(None)
    metadata: dict[str, str] | None = Field(None)

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str | None) -> str | None:
        return v.strip() if v else None

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: list[float] | None) -> list[float] | None:
        return _validate_optional_embedding(v)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        return _sanitize_metadata(v) if v else None

    model_config = ConfigDict(extra="ignore")


class IndexBuildRequestDTO(BaseModel):
    algorithm: str = Field(...)
    metric: str = Field(...)

    @field_validator("algorithm", "metric")
    @classmethod
    def validate_lowercase(cls, v: str) -> str:
        return v.lower()


class SearchRequestDTO(BaseModel):
    vector: list[float] = Field(...)
    k: int = Field(..., ge=1, le=100)
    metadata_filters: dict[str, str] = Field(default_factory=dict)

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("Query vector cannot be empty")
        return _validate_embedding(v)

    @field_validator("metadata_filters")
    @classmethod
    def validate_filters(cls, v: dict[str, str]) -> dict[str, str]:
        return _sanitize_metadata(v)


class LibraryDTO(BaseModel):
    id: str
    name: str
    description: str | None
    metadata: dict[str, str]


class DocumentDTO(BaseModel):
    id: str
    library_id: str
    title: str
    description: str | None
    metadata: dict[str, str]


class ChunkDTO(BaseModel):
    id: str
    document_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, str]


class IndexInfoDTO(BaseModel):
    library_id: str
    algorithm: str
    metric: str


class SearchResultItemDTO(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    text: str
    metadata: dict[str, str]


class SearchResponseDTO(BaseModel):
    results: list[SearchResultItemDTO]
    metric: str | None
    algorithm: str | None
