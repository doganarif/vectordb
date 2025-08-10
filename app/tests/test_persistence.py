from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_persistence_save_load_and_search_restored(tmp_path: Path) -> None:
    r = client.post("/libraries/", json={"name": "lib-persist"})
    assert r.status_code == 201
    lib_id = r.json()["id"]

    r = client.post(f"/libraries/{lib_id}/documents", json={"title": "doc"})
    assert r.status_code == 201
    doc_id = r.json()["id"]

    v1 = [0.0, 1.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    r = client.post(
        f"/libraries/{lib_id}/chunks",
        json={"document_id": doc_id, "text": "a", "embedding": v1},
    )
    assert r.status_code == 201
    r = client.post(
        f"/libraries/{lib_id}/chunks",
        json={"document_id": doc_id, "text": "b", "embedding": v2},
    )
    assert r.status_code == 201

    r = client.put(
        f"/libraries/{lib_id}/index",
        json={"algorithm": "kdtree", "metric": "euclidean"},
    )
    assert r.status_code == 202
    info = r.json()
    assert info["algorithm"] == "kdtree"
    assert info["metric"] == "euclidean"

    r = client.post(f"/libraries/{lib_id}/searches", json={"vector": v1, "k": 1})
    assert r.status_code == 200
    before = r.json()["results"][0]["chunk_id"]

    snap_path = tmp_path / "snapshot.json"

    r = client.post("/admin/snapshots")
    assert r.status_code == 200
    saved_path = Path(r.json()["saved_to"]).resolve()
    snap_path.write_text(saved_path.read_text())

    r = client.put("/admin/snapshots")
    assert r.status_code == 200

    r = client.post(f"/libraries/{lib_id}/searches", json={"vector": v1, "k": 1})
    assert r.status_code == 200
    after = r.json()["results"][0]["chunk_id"]
    assert before == after

    r = client.put(
        f"/libraries/{lib_id}/index",
        json={"algorithm": "kdtree", "metric": "euclidean"},
    )
    assert r.status_code == 202
    info2 = r.json()
    assert info2["algorithm"] == "kdtree"
    assert info2["metric"] == "euclidean"
