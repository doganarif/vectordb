from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _seed_vectors(metric: str = "cosine") -> tuple[str, str, str]:
    r = client.post("/libraries/", json={"name": f"lib-{metric}"})
    lib_id = r.json()["id"]
    r = client.post(f"/libraries/{lib_id}/documents", json={"title": "doc"})
    doc_id = r.json()["id"]
    r = client.post(
        f"/libraries/{lib_id}/chunks",
        json={"document_id": doc_id, "text": "a", "embedding": [0.0, 1.0, 0.0]},
    )
    c1 = r.json()["id"]
    r = client.post(
        f"/libraries/{lib_id}/chunks",
        json={"document_id": doc_id, "text": "b", "embedding": [1.0, 0.0, 0.0]},
    )
    c2 = r.json()["id"]
    return lib_id, c1, c2


def test_build_index_validations():
    lib_id, _, _ = _seed_vectors("cosine")

    # valid: linear+cosine
    r = client.put(
        f"/libraries/{lib_id}/index", json={"algorithm": "linear", "metric": "cosine"}
    )
    assert r.status_code == 200
    assert r.json()["algorithm"] == "linear"
    assert r.json()["metric"] == "cosine"

    # invalid combinations
    r = client.put(
        f"/libraries/{lib_id}/index", json={"algorithm": "kdtree", "metric": "cosine"}
    )
    assert r.status_code == 400

    r = client.put(
        f"/libraries/{lib_id}/index", json={"algorithm": "lsh", "metric": "euclidean"}
    )
    assert r.status_code == 400

    r = client.put(
        f"/libraries/{lib_id}/index", json={"algorithm": "unknown", "metric": "cosine"}
    )
    assert r.status_code == 400


def test_search_without_built_index_falls_back_and_respects_k():
    lib_id, _, _ = _seed_vectors("cosine")
    # no build call
    r = client.post(
        f"/libraries/{lib_id}/chunks/search",
        json={"vector": [0.0, 1.0, 0.0], "k": 1, "metadata_filters": {}},
    )
    assert r.status_code == 200
    assert len(r.json()["results"]) == 1


def test_search_with_metadata_filters():
    r = client.post("/libraries/", json={"name": "lib-meta2"})
    lib_id = r.json()["id"]
    r = client.post(f"/libraries/{lib_id}/documents", json={"title": "doc"})
    doc_id = r.json()["id"]
    client.post(
        f"/libraries/{lib_id}/chunks",
        json={
            "document_id": doc_id,
            "text": "x",
            "embedding": [0, 1, 0],
            "metadata": {"lang": "en", "topic": "a"},
        },
    )
    client.post(
        f"/libraries/{lib_id}/chunks",
        json={
            "document_id": doc_id,
            "text": "y",
            "embedding": [1, 0, 0],
            "metadata": {"lang": "tr", "topic": "b"},
        },
    )
    client.post(
        f"/libraries/{lib_id}/index", json={"algorithm": "linear", "metric": "cosine"}
    )
    r = client.post(
        f"/libraries/{lib_id}/chunks/search",
        json={
            "vector": [0.0, 1.0, 0.0],
            "k": 5,
            "metadata_filters": {"lang": "en", "topic": "a"},
        },
    )
    assert r.status_code == 200
    res = r.json()["results"]
    assert len(res) == 1
    assert res[0]["metadata"]["lang"] == "en"
    assert res[0]["metadata"]["topic"] == "a"
