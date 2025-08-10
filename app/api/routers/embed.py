"""Embedding API endpoints with retry logic and connection pooling."""

import asyncio
import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core import settings

router = APIRouter()
log = logging.getLogger(__name__)

# Global connection pool for better performance
_http_client: httpx.AsyncClient | None = None

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
RETRY_BACKOFF = 2.0  # exponential backoff multiplier


class EmbedText(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to embed")


class EmbeddingResponse(BaseModel):
    embedding: list[float] = Field(..., description="Vector embedding")


async def get_http_client() -> httpx.AsyncClient:
    """Get or create the global HTTP client with connection pooling.

    Returns:
        httpx.AsyncClient: The HTTP client instance
    """
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0,
            ),
        )
    return _http_client


async def close_http_client() -> None:
    """Close the global HTTP client (call on app shutdown)."""
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None


async def call_cohere_with_retry(
    url: str,
    payload: dict,
    headers: dict,
    max_retries: int = MAX_RETRIES,
) -> httpx.Response:
    """Call Cohere API with exponential backoff retry logic.

    Args:
        url: API endpoint URL
        payload: Request payload
        headers: Request headers
        max_retries: Maximum number of retry attempts

    Returns:
        httpx.Response: The API response

    Raises:
        HTTPException: If all retries fail
    """
    client = await get_http_client()
    last_error = None
    delay = RETRY_DELAY

    for attempt in range(max_retries):
        try:
            response = await client.post(url, json=payload, headers=headers)

            # Return successful responses immediately
            if response.status_code == 200:
                return response

            # Don't retry client errors (4xx)
            if 400 <= response.status_code < 500:
                return response

            # Server error (5xx) - retry
            last_error = f"Server error: {response.status_code}"
            log.warning(
                "Cohere API error (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                last_error,
            )

        except httpx.TimeoutException as e:
            last_error = f"Timeout: {e}"
            log.warning(
                "Cohere API timeout (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                e,
            )

        except httpx.RequestError as e:
            last_error = f"Request error: {e}"
            log.warning(
                "Cohere API request error (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                e,
            )

        # Wait before retrying (except on last attempt)
        if attempt < max_retries - 1:
            await asyncio.sleep(delay)
            delay *= RETRY_BACKOFF

    # All retries failed
    log.error("All retries failed for Cohere API: %s", last_error)

    if "timeout" in str(last_error).lower():
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Embedding service timeout after retries",
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Embedding service unavailable after retries",
        )


@router.post(
    "",
    summary="Create an embedding using Cohere v2",
    response_model=EmbeddingResponse,
    response_model_exclude_unset=True,
)
async def embed_with_cohere(body: EmbedText) -> dict[str, list[float]]:
    """Generate text embedding using Cohere API.

    Args:
        body: Text to embed

    Returns:
        Dictionary with embedding vector

    Raises:
        HTTPException: If API key missing or API call fails
    """
    api_key = settings.cohere_api_key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service not configured",
        )

    url = "https://api.cohere.com/v2/embed"
    payload = {
        "model": "embed-v4.0",
        "texts": [body.text],
        "input_type": "search_document",
        "embedding_types": ["float"],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Call API with retry logic
    resp = await call_cohere_with_retry(url, payload, headers)

    # Handle non-200 responses
    if resp.status_code != 200:
        try:
            err = resp.json()
        except ValueError:
            err = resp.text

        code = (
            resp.status_code
            if 400 <= resp.status_code < 500
            else status.HTTP_502_BAD_GATEWAY
        )
        raise HTTPException(status_code=code, detail=err)

    # Parse response
    try:
        data: Any = resp.json()
    except ValueError:
        log.error("Invalid JSON response from Cohere")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Invalid response from embedding service",
        )

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Invalid response format from embedding service",
        )

    embeddings = data.get("embeddings", {})
    floats = embeddings.get("float") if isinstance(embeddings, dict) else None

    # Support both [[...]] and [...] response formats
    vector: list[float] | None = None
    if isinstance(floats, list):
        if floats and isinstance(floats[0], list):
            vector = floats[0]
        elif floats and all(isinstance(x, (int, float)) for x in floats):
            vector = floats

    if vector is not None:
        return {"embedding": vector}

    log.error("Unexpected embeddings response structure: %s", data)
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Invalid embedding format received",
    )


# Note: The HTTP client cleanup should be handled in the main app's lifespan
# For now, the client will be cleaned up when the process exits
