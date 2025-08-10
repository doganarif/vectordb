from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.domain.dto import (
    ChunkDTO,
    CreateChunkDTO,
    CreateDocumentDTO,
    CreateLibraryDTO,
    DocumentDTO,
    IndexBuildRequestDTO,
    IndexInfoDTO,
    LibraryDTO,
    SearchRequestDTO,
    SearchResponseDTO,
    SearchResultItemDTO,
    UpdateChunkDTO,
    UpdateDocumentDTO,
    UpdateLibraryDTO,
)
from app.services import VectorDBService, get_service

router = APIRouter()


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=LibraryDTO)
def create_library(
    payload: CreateLibraryDTO,
    service: VectorDBService = Depends(get_service),
) -> LibraryDTO:
    lib = service.create_library(payload.name, payload.description, payload.metadata)
    return LibraryDTO(
        id=lib.id, name=lib.name, description=lib.description, metadata=lib.metadata
    )


@router.get("/", response_model=list[LibraryDTO])
def list_libraries(service: VectorDBService = Depends(get_service)) -> list[LibraryDTO]:
    libraries = service.list_libraries()
    return [
        LibraryDTO(
            id=lib.id, name=lib.name, description=lib.description, metadata=lib.metadata
        )
        for lib in libraries
    ]


@router.get("/{library_id}", response_model=LibraryDTO)
def get_library(
    library_id: str,
    service: VectorDBService = Depends(get_service),
) -> LibraryDTO:
    lib = service.get_library(library_id)
    if not lib:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return LibraryDTO(
        id=lib.id, name=lib.name, description=lib.description, metadata=lib.metadata
    )


@router.patch("/{library_id}", response_model=LibraryDTO)
def update_library(
    library_id: str,
    payload: UpdateLibraryDTO,
    service: VectorDBService = Depends(get_service),
) -> LibraryDTO:
    lib = service.update_library(
        library_id, payload.name, payload.description, payload.metadata
    )
    if not lib:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return LibraryDTO(
        id=lib.id, name=lib.name, description=lib.description, metadata=lib.metadata
    )


@router.delete(
    "/{library_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_library(
    library_id: str,
    service: VectorDBService = Depends(get_service),
) -> Response:
    service.delete_library(library_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{library_id}/documents",
    status_code=status.HTTP_201_CREATED,
    response_model=DocumentDTO,
)
def create_document(
    library_id: str,
    payload: CreateDocumentDTO,
    service: VectorDBService = Depends(get_service),
) -> DocumentDTO:
    doc = service.create_document(
        library_id, payload.title, payload.description, payload.metadata
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Library not found",
        )
    return DocumentDTO(
        id=doc.id,
        library_id=doc.library_id,
        title=doc.title,
        description=doc.description,
        metadata=doc.metadata,
    )


@router.get("/{library_id}/documents", response_model=list[DocumentDTO])
def list_documents(
    library_id: str,
    service: VectorDBService = Depends(get_service),
) -> list[DocumentDTO]:
    documents = service.list_documents(library_id)
    return [
        DocumentDTO(
            id=doc.id,
            library_id=doc.library_id,
            title=doc.title,
            description=doc.description,
            metadata=doc.metadata,
        )
        for doc in documents
    ]


@router.patch("/{library_id}/documents/{document_id}", response_model=DocumentDTO)
def update_document(
    library_id: str,
    document_id: str,
    payload: UpdateDocumentDTO,
    service: VectorDBService = Depends(get_service),
) -> DocumentDTO:
    existing = service.get_document(document_id)
    if not existing or existing.library_id != library_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    doc = service.update_document(
        document_id, payload.title, payload.description, payload.metadata
    )
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return DocumentDTO(
        id=doc.id,
        library_id=doc.library_id,
        title=doc.title,
        description=doc.description,
        metadata=doc.metadata,
    )


@router.delete(
    "/{library_id}/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_document(
    library_id: str,
    document_id: str,
    service: VectorDBService = Depends(get_service),
) -> Response:
    existing = service.get_document(document_id)
    if not existing or existing.library_id != library_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    service.delete_document(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{library_id}/chunks",
    status_code=status.HTTP_201_CREATED,
    response_model=ChunkDTO,
)
def create_chunk(
    library_id: str,
    payload: CreateChunkDTO,
    service: VectorDBService = Depends(get_service),
) -> ChunkDTO:
    try:
        chunk = service.create_chunk(
            library_id,
            payload.document_id,
            payload.text,
            payload.embedding,
            payload.metadata,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if not chunk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or mismatched library",
        )
    return ChunkDTO(
        id=chunk.id,
        document_id=chunk.document_id,
        text=chunk.text,
        embedding=chunk.embedding,
        metadata=chunk.metadata,
    )


@router.get("/{library_id}/chunks", response_model=list[ChunkDTO])
def list_chunks(
    library_id: str,
    service: VectorDBService = Depends(get_service),
) -> list[ChunkDTO]:
    chunks = service.list_chunks(library_id)
    return [
        ChunkDTO(
            id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            embedding=chunk.embedding,
            metadata=chunk.metadata,
        )
        for chunk in chunks
    ]


@router.patch("/{library_id}/chunks/{chunk_id}", response_model=ChunkDTO)
def update_chunk(
    library_id: str,
    chunk_id: str,
    payload: UpdateChunkDTO,
    service: VectorDBService = Depends(get_service),
) -> ChunkDTO:
    existing = service.get_chunk(chunk_id)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    parent_doc = service.get_document(existing.document_id)
    if not parent_doc or parent_doc.library_id != library_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    try:
        chunk = service.update_chunk(
            chunk_id, payload.text, payload.embedding, payload.metadata
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if not chunk:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return ChunkDTO(
        id=chunk.id,
        document_id=chunk.document_id,
        text=chunk.text,
        embedding=chunk.embedding,
        metadata=chunk.metadata,
    )


@router.delete(
    "/{library_id}/chunks/{chunk_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_chunk(
    library_id: str,
    chunk_id: str,
    service: VectorDBService = Depends(get_service),
) -> Response:
    existing = service.get_chunk(chunk_id)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    parent_doc = service.get_document(existing.document_id)
    if not parent_doc or parent_doc.library_id != library_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    service.delete_chunk(chunk_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put(
    "/{library_id}/index",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IndexInfoDTO,
)
def build_index(
    library_id: str,
    payload: IndexBuildRequestDTO,
    service: VectorDBService = Depends(get_service),
) -> IndexInfoDTO:
    try:
        service.build_index(library_id, payload.algorithm, payload.metric)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    info = service.get_index_info(library_id)
    return IndexInfoDTO(
        library_id=info["library_id"],
        algorithm=info["algorithm"],
        metric=info["metric"],
    )


@router.post("/{library_id}/searches", response_model=SearchResponseDTO)
def search(
    library_id: str,
    payload: SearchRequestDTO,
    service: VectorDBService = Depends(get_service),
) -> SearchResponseDTO:
    try:
        results = service.search(
            library_id, payload.vector, payload.k, payload.metadata_filters
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    items: list[SearchResultItemDTO] = []
    for chunk_id, score in results:
        chunk = service.get_chunk(chunk_id)
        if not chunk:
            continue

        items.append(
            SearchResultItemDTO(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                score=score,
                text=chunk.text,
                metadata=chunk.metadata,
            )
        )

    idx = service.get_index_info(library_id)
    return SearchResponseDTO(
        results=items,
        metric=idx.get("metric"),
        algorithm=idx.get("algorithm"),
    )
