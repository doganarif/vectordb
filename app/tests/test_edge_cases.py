"""Tests for edge cases and error handling."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_empty_embedding_rejected():
    """Test that empty embeddings are rejected at DTO level."""
    # Create library and document
    r = client.post("/libraries/", json={"name": "test-empty"})
    assert r.status_code == 201
    lib_id = r.json()["id"]

    r = client.post(f"/libraries/{lib_id}/documents", json={"title": "doc"})
    assert r.status_code == 201
    doc_id = r.json()["id"]

    # Try to create chunk with empty embedding - should fail
    r = client.post(
        f"/libraries/{lib_id}/chunks",
        json={
            "document_id": doc_id,
            "text": "test",
            "embedding": [],  # Empty embedding
            "metadata": {},
        },
    )
    assert r.status_code == 422  # Validation error
    assert "min_length" in str(r.json()) or "at least 1" in str(r.json()).lower()

    # Try to create chunk without embedding field - should also fail
    r = client.post(
        f"/libraries/{lib_id}/chunks",
        json={"document_id": doc_id, "text": "test", "metadata": {}},
    )
    assert r.status_code == 422  # Validation error


def test_search_wrong_dimension():
    """Test that searching with wrong dimension vectors returns proper error."""
    # Create library with 3D vectors
    r = client.post("/libraries/", json={"name": "test-dims"})
    assert r.status_code == 201
    lib_id = r.json()["id"]

    r = client.post(f"/libraries/{lib_id}/documents", json={"title": "doc"})
    assert r.status_code == 201
    doc_id = r.json()["id"]

    # Add chunk with 3D vector
    r = client.post(
        f"/libraries/{lib_id}/chunks",
        json={
            "document_id": doc_id,
            "text": "test",
            "embedding": [1.0, 0.0, 0.0],  # 3D
            "metadata": {},
        },
    )
    assert r.status_code == 201

    # Build index
    r = client.put(
        f"/libraries/{lib_id}/index", json={"algorithm": "linear", "metric": "cosine"}
    )
    assert r.status_code == 200

    # Search with correct dimensions - should work
    r = client.post(
        f"/libraries/{lib_id}/chunks/search",
        json={
            "vector": [0.0, 1.0, 0.0],  # 3D - correct
            "k": 1,
            "metadata_filters": {},
        },
    )
    assert r.status_code == 200

    # Search with wrong dimensions - should fail
    r = client.post(
        f"/libraries/{lib_id}/chunks/search",
        json={"vector": [1.0, 0.0], "k": 1, "metadata_filters": {}},  # 2D - wrong!
    )
    assert r.status_code == 400  # Bad request
    assert "dimension" in str(r.json()).lower()

    # Search with empty vector - should fail at validation
    r = client.post(
        f"/libraries/{lib_id}/chunks/search",
        json={"vector": [], "k": 1, "metadata_filters": {}},  # Empty
    )
    assert r.status_code == 422  # Validation error


def test_snapshot_delete_then_restore():
    """Test that restoring a deleted snapshot returns 404."""
    # Create a snapshot
    r = client.post("/admin/snapshots", json={"name": "test-snapshot"})
    assert r.status_code == 201
    snapshot_id = r.json()["id"]

    # Verify snapshot exists
    r = client.get(f"/admin/snapshots/{snapshot_id}")
    assert r.status_code == 200

    # Delete the snapshot
    r = client.delete(f"/admin/snapshots/{snapshot_id}")
    assert r.status_code == 204

    # Try to restore deleted snapshot - should fail with 404
    r = client.post(f"/admin/snapshots/{snapshot_id}/restore")
    assert r.status_code == 404
    assert "not found" in str(r.json()).lower()

    # Try to get deleted snapshot - should also fail
    r = client.get(f"/admin/snapshots/{snapshot_id}")
    assert r.status_code == 404


def test_update_chunk_with_empty_embedding():
    """Test that updating a chunk with empty embedding is rejected."""
    # Create library and document
    r = client.post("/libraries/", json={"name": "test-update-empty"})
    assert r.status_code == 201
    lib_id = r.json()["id"]

    r = client.post(f"/libraries/{lib_id}/documents", json={"title": "doc"})
    assert r.status_code == 201
    doc_id = r.json()["id"]

    # Create chunk with valid embedding
    r = client.post(
        f"/libraries/{lib_id}/chunks",
        json={
            "document_id": doc_id,
            "text": "original",
            "embedding": [1.0, 0.0, 0.0],
            "metadata": {},
        },
    )
    assert r.status_code == 201
    chunk_id = r.json()["id"]

    # Try to update with empty embedding - should fail
    r = client.patch(
        f"/libraries/{lib_id}/chunks/{chunk_id}",
        json={"embedding": []},  # Empty embedding
    )
    assert r.status_code == 422  # Validation error

    # Update with valid embedding - should work
    r = client.patch(
        f"/libraries/{lib_id}/chunks/{chunk_id}",
        json={"embedding": [0.0, 1.0, 0.0]},  # Valid 3D
    )
    assert r.status_code == 200


def test_dimension_enforcement_on_second_chunk():
    """Test that library-level dimension is enforced after first chunk."""
    # Create library
    r = client.post("/libraries/", json={"name": "test-dim-enforce"})
    assert r.status_code == 201
    lib_id = r.json()["id"]

    r = client.post(f"/libraries/{lib_id}/documents", json={"title": "doc"})
    assert r.status_code == 201
    doc_id = r.json()["id"]

    # Create first chunk with 3D vector
    r = client.post(
        f"/libraries/{lib_id}/chunks",
        json={
            "document_id": doc_id,
            "text": "first",
            "embedding": [1.0, 0.0, 0.0],  # 3D
            "metadata": {},
        },
    )
    assert r.status_code == 201

    # Try to create second chunk with different dimension - should fail
    r = client.post(
        f"/libraries/{lib_id}/chunks",
        json={
            "document_id": doc_id,
            "text": "second",
            "embedding": [1.0, 0.0],  # 2D - wrong!
            "metadata": {},
        },
    )
    assert r.status_code == 400  # Bad request
    assert "dimension" in str(r.json()).lower()

    # Create second chunk with correct dimension - should work
    r = client.post(
        f"/libraries/{lib_id}/chunks",
        json={
            "document_id": doc_id,
            "text": "second",
            "embedding": [0.0, 1.0, 0.0],  # 3D - correct
            "metadata": {},
        },
    )
    assert r.status_code == 201
