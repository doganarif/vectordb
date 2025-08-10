from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.core.exceptions import (
    DimensionalityMismatchException,
    InvalidAlgorithmException,
    InvalidMetricException,
    ResourceNotFoundException,
)
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
    lib = service.libraries.create_library(
        payload.name, payload.description, payload.metadata
    )
    return LibraryDTO(
        id=lib.id, name=lib.name, description=lib.description, metadata=lib.metadata
    )


@router.get("/", response_model=list[LibraryDTO])
def list_libraries(service: VectorDBService = Depends(get_service)) -> list[LibraryDTO]:
    libraries = service.libraries.list_libraries()
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
    try:
        lib = service.libraries.get_library(library_id)
    except ResourceNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library not found: {library_id}",
        )
    return LibraryDTO(
        id=lib.id, name=lib.name, description=lib.description, metadata=lib.metadata
    )


@router.patch("/{library_id}", response_model=LibraryDTO)
def update_library(
    library_id: str,
    payload: UpdateLibraryDTO,
    service: VectorDBService = Depends(get_service),
) -> LibraryDTO:
    try:
        lib = service.libraries.update_library(
            library_id, payload.name, payload.description, payload.metadata
        )
    except ResourceNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library not found: {library_id}",
        )
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
    service.libraries.delete_library_cascade(library_id)
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
    try:
        doc = service.documents.create_document(
            library_id, payload.title, payload.description, payload.metadata
        )
    except ResourceNotFoundException:
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
    documents = service.documents.list_documents(library_id)
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
    try:
        existing = service.documents.get_document(document_id)
    except ResourceNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    if existing.library_id != library_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found in library {library_id}",
        )

    doc = service.documents.update_document(
        document_id, payload.title, payload.description, payload.metadata
    )
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
    try:
        existing = service.documents.get_document(document_id)
    except ResourceNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    if existing.library_id != library_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found in library {library_id}",
        )

    service.documents.delete_document(document_id)
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
        chunk = service.chunks.create_chunk(
            library_id,
            payload.document_id,
            payload.text,
            payload.embedding,
            payload.metadata,
        )
    except DimensionalityMismatchException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except ResourceNotFoundException:
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
    chunks = service.chunks.list_chunks(library_id)
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
    try:
        existing = service.chunks.get_chunk(chunk_id)
    except ResourceNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk not found: {chunk_id}",
        )

    try:
        parent_doc = service.documents.get_document(existing.document_id)
    except ResourceNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Parent document not found for chunk {chunk_id}",
        )

    if parent_doc.library_id != library_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk {chunk_id} not found in library {library_id}",
        )

    try:
        chunk = service.chunks.update_chunk(
            chunk_id, payload.text, payload.embedding, payload.metadata
        )
    except DimensionalityMismatchException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
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
    try:
        existing = service.chunks.get_chunk(chunk_id)
    except ResourceNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk not found: {chunk_id}",
        )

    try:
        parent_doc = service.documents.get_document(existing.document_id)
    except ResourceNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Parent document not found for chunk {chunk_id}",
        )

    if parent_doc.library_id != library_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk {chunk_id} not found in library {library_id}",
        )

    service.chunks.delete_chunk(chunk_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put(
    "/{library_id}/index",
    status_code=status.HTTP_200_OK,
    response_model=IndexInfoDTO,
)
def create_or_replace_index(
    library_id: str,
    payload: IndexBuildRequestDTO,
    service: VectorDBService = Depends(get_service),
) -> IndexInfoDTO:
    try:
        service.indices.build_index(library_id, payload.algorithm, payload.metric)
    except (InvalidAlgorithmException, InvalidMetricException) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    info = service.indices.get_index_info(library_id)
    return IndexInfoDTO(
        library_id=info["library_id"],
        algorithm=info["algorithm"],
        metric=info["metric"],
    )


@router.get(
    "/{library_id}/index",
    response_model=IndexInfoDTO,
)
def get_index(
    library_id: str,
    service: VectorDBService = Depends(get_service),
) -> IndexInfoDTO:
    info = service.indices.get_index_info(library_id)
    if info["algorithm"] == "none":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No index found for library {library_id}",
        )
    return IndexInfoDTO(
        library_id=info["library_id"],
        algorithm=info["algorithm"],
        metric=info["metric"],
    )


@router.delete(
    "/{library_id}/index",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_index(
    library_id: str,
    service: VectorDBService = Depends(get_service),
) -> Response:
    service.indices.clear_index(library_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{library_id}/chunks/search", response_model=SearchResponseDTO)
def search_chunks(
    library_id: str,
    request: SearchRequestDTO,
    service: VectorDBService = Depends(get_service),
) -> SearchResponseDTO:

    try:
        results = service.indices.search(
            library_id, request.vector, request.k, request.metadata_filters
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    items: list[SearchResultItemDTO] = []
    for chunk_id, score in results:
        try:
            chunk = service.chunks.get_chunk(chunk_id)
        except ResourceNotFoundException:
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

    idx = service.indices.get_index_info(library_id)
    return SearchResponseDTO(
        results=items,
        metric=idx.get("metric"),
        algorithm=idx.get("algorithm"),
    )
