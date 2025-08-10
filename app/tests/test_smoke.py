from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200


def test_crud_and_search_flow():
    # create library
    r = client.post("/libraries/", json={"name": "lib1"})
    assert r.status_code == 201
    lib = r.json()
    lib_id = lib["id"]

    # create document
    r = client.post(f"/libraries/{lib_id}/documents", json={"title": "doc1"})
    assert r.status_code == 201
    doc = r.json()

    # create chunks
    emb1 = [0.0, 1.0, 0.0]
    emb2 = [1.0, 0.0, 0.0]
    r = client.post(
        f"/libraries/{lib_id}/chunks",
        json={"document_id": doc["id"], "text": "hello world", "embedding": emb1},
    )
    assert r.status_code == 201
    r = client.post(
        f"/libraries/{lib_id}/chunks",
        json={"document_id": doc["id"], "text": "vector db", "embedding": emb2},
    )
    assert r.status_code == 201

    # build index
    r = client.put(
        f"/libraries/{lib_id}/index", json={"algorithm": "linear", "metric": "cosine"}
    )
    assert r.status_code == 200
    # verify response returns index info
    assert r.json()["algorithm"] == "linear"
    assert r.json()["metric"] == "cosine"

    # search using RESTful POST endpoint
    r = client.post(
        f"/libraries/{lib_id}/chunks/search",
        json={"vector": [0.0, 1.0, 0.0], "k": 1, "metadata_filters": {}},
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1
