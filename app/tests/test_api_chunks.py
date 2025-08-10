from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _setup() -> tuple[str, str]:
    r = client.post("/libraries/", json={"name": "lib-chunks"})
    lib_id = r.json()["id"]
    r = client.post(f"/libraries/{lib_id}/documents", json={"title": "doc"})
    doc_id = r.json()["id"]
    return lib_id, doc_id


def test_chunks_crud_flow_and_dimensionality_validation():
    lib_id, doc_id = _setup()

    # create chunk
    r = client.post(
        f"/libraries/{lib_id}/chunks",
        json={"document_id": doc_id, "text": "a", "embedding": [0.0, 1.0, 0.0]},
    )
    assert r.status_code == 201
    chunk_id = r.json()["id"]

    # list
    r = client.get(f"/libraries/{lib_id}/chunks")
    assert r.status_code == 200
    assert any(x["id"] == chunk_id for x in r.json())

    # update text
    r = client.patch(f"/libraries/{lib_id}/chunks/{chunk_id}", json={"text": "ax"})
    assert r.status_code == 200
    assert r.json()["text"] == "ax"

    # update embedding with wrong dimension -> 400
    r = client.patch(
        f"/libraries/{lib_id}/chunks/{chunk_id}", json={"embedding": [1.0, 2.0]}
    )
    assert r.status_code == 400

    # update embedding with correct dimension
    r = client.patch(
        f"/libraries/{lib_id}/chunks/{chunk_id}", json={"embedding": [1.0, 0.0, 0.0]}
    )
    assert r.status_code == 200
    assert r.json()["embedding"] == [1.0, 0.0, 0.0]

    # delete
    r = client.delete(f"/libraries/{lib_id}/chunks/{chunk_id}")
    assert r.status_code == 204

    # update after delete -> 404
    r = client.patch(f"/libraries/{lib_id}/chunks/{chunk_id}", json={"text": "z"})
    assert r.status_code == 404


def test_create_chunk_library_mismatch():
    _, doc_id = _setup()
    # create another library to mismatch
    r = client.post("/libraries/", json={"name": "lib2"})
    other_lib_id = r.json()["id"]
    r = client.post(
        f"/libraries/{other_lib_id}/chunks",
        json={"document_id": doc_id, "text": "x", "embedding": [0, 1, 0]},
    )
    assert r.status_code == 404
