from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _setup_lib_with_vectors(metric: str):
    r = client.post("/libraries/", json={"name": f"lib-{metric}"})
    lib_id = r.json()["id"]
    r = client.post(f"/libraries/{lib_id}/documents", json={"title": "doc"})
    doc_id = r.json()["id"]
    # two orthogonal-ish vectors
    v1 = [0.0, 1.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    client.post(
        f"/libraries/{lib_id}/chunks",
        json={"document_id": doc_id, "text": "a", "embedding": v1},
    )
    client.post(
        f"/libraries/{lib_id}/chunks",
        json={"document_id": doc_id, "text": "b", "embedding": v2},
    )
    return lib_id


def test_kdtree_euclidean():
    lib_id = _setup_lib_with_vectors("euclidean")
    r = client.put(
        f"/libraries/{lib_id}/index",
        json={"algorithm": "kdtree", "metric": "euclidean"},
    )
    assert r.status_code == 200
    r = client.post(
        f"/libraries/{lib_id}/chunks/search",
        json={"vector": [0.0, 1.0, 0.0], "k": 1, "metadata_filters": {}},
    )
    assert r.status_code == 200
    assert len(r.json()["results"]) == 1


def test_lsh_cosine():
    lib_id = _setup_lib_with_vectors("cosine")
    r = client.put(
        f"/libraries/{lib_id}/index", json={"algorithm": "lsh", "metric": "cosine"}
    )
    assert r.status_code == 200
    r = client.post(
        f"/libraries/{lib_id}/chunks/search",
        json={"vector": [1.0, 0.0, 0.0], "k": 1, "metadata_filters": {}},
    )
    assert r.status_code == 200
    assert len(r.json()["results"]) == 1


def test_metadata_filtering():
    r = client.post("/libraries/", json={"name": "lib-meta"})
    lib_id = r.json()["id"]
    r = client.post(f"/libraries/{lib_id}/documents", json={"title": "doc"})
    doc_id = r.json()["id"]
    client.post(
        f"/libraries/{lib_id}/chunks",
        json={
            "document_id": doc_id,
            "text": "x",
            "embedding": [0, 1, 0],
            "metadata": {"lang": "en"},
        },
    )
    client.post(
        f"/libraries/{lib_id}/chunks",
        json={
            "document_id": doc_id,
            "text": "y",
            "embedding": [1, 0, 0],
            "metadata": {"lang": "tr"},
        },
    )
    client.put(
        f"/libraries/{lib_id}/index", json={"algorithm": "linear", "metric": "cosine"}
    )
    r = client.post(
        f"/libraries/{lib_id}/chunks/search",
        json={
            "vector": [0.0, 1.0, 0.0],
            "k": 5,
            "metadata_filters": {"lang": "en"},
        },
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1
    assert results[0]["metadata"]["lang"] == "en"
