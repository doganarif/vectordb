from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@patch("app.api.routers.embed.settings.cohere_api_key", new="testkey")
@patch("app.api.routers.embed.get_http_client")
def test_embed_success(mock_get_client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": {"float": [[0.1, 0.2, 0.3]]}}

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_get_client.return_value = mock_client

    response = client.post("/embeddings", json={"text": "hello"})

    assert response.status_code == 200
    assert response.json()["embedding"] == [0.1, 0.2, 0.3]


@patch("app.api.routers.embed.settings.cohere_api_key", new=None)
def test_embed_missing_api_key():
    response = client.post("/embeddings", json={"text": "hello"})
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


def test_embed_missing_text():
    response = client.post("/embeddings", json={})
    assert response.status_code == 422


def test_embed_empty_text():
    response = client.post("/embeddings", json={"text": ""})
    assert response.status_code == 422


@patch("app.api.routers.embed.settings.cohere_api_key", new="testkey")
@patch("app.api.routers.embed.get_http_client")
def test_embed_upstream_error(mock_get_client):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.json.return_value = {"message": "upstream error"}

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_get_client.return_value = mock_client

    response = client.post("/embeddings", json={"text": "hello"})

    assert response.status_code == 502


@patch("app.api.routers.embed.settings.cohere_api_key", new="testkey")
@patch("app.api.routers.embed.get_http_client")
def test_embed_timeout(mock_get_client):
    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.TimeoutException("Request timed out")
    mock_get_client.return_value = mock_client

    response = client.post("/embeddings", json={"text": "hello"})

    assert response.status_code == 504
    assert "timeout" in response.json()["detail"].lower()


@patch("app.api.routers.embed.settings.cohere_api_key", new="testkey")
@patch("app.api.routers.embed.get_http_client")
def test_embed_connection_error(mock_get_client):
    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.RequestError("Connection failed")
    mock_get_client.return_value = mock_client

    response = client.post("/embeddings", json={"text": "hello"})

    assert response.status_code == 502
    assert "unavailable" in response.json()["detail"].lower()


@patch("app.api.routers.embed.settings.cohere_api_key", new="testkey")
@patch("app.api.routers.embed.get_http_client")
def test_embed_invalid_response_format(mock_get_client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"unexpected": "format"}

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_get_client.return_value = mock_client

    response = client.post("/embeddings", json={"text": "hello"})

    assert response.status_code == 502
    assert "format" in response.json()["detail"].lower()


@patch("app.api.routers.embed.settings.cohere_api_key", new="testkey")
@patch("app.api.routers.embed.get_http_client")
def test_embed_client_error_no_retry(mock_get_client):
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {"error": "Bad request"}

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_get_client.return_value = mock_client

    response = client.post("/embeddings", json={"text": "hello"})

    assert response.status_code == 400
    mock_client.post.assert_called_once()


@patch("app.api.routers.embed.settings.cohere_api_key", new="testkey")
@patch("app.api.routers.embed.get_http_client")
@patch("app.api.routers.embed.asyncio.sleep", new_callable=AsyncMock)
def test_embed_retry_on_server_error(mock_sleep, mock_get_client):
    mock_response_fail = MagicMock()
    mock_response_fail.status_code = 500

    mock_response_success = MagicMock()
    mock_response_success.status_code = 200
    mock_response_success.json.return_value = {
        "embeddings": {"float": [[0.1, 0.2, 0.3]]}
    }

    mock_client = AsyncMock()
    mock_client.post.side_effect = [
        mock_response_fail,
        mock_response_fail,
        mock_response_success,
    ]
    mock_get_client.return_value = mock_client

    response = client.post("/embeddings", json={"text": "hello"})

    assert response.status_code == 200
    assert response.json()["embedding"] == [0.1, 0.2, 0.3]
    assert mock_client.post.call_count == 3
    assert mock_sleep.call_count == 2
