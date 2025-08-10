from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_libraries_crud_flow():
    # create
    r = client.post(
        "/libraries/",
        json={"name": "lib-crud", "description": "d", "metadata": {"a": "b"}},
    )
    assert r.status_code == 201
    lib = r.json()
    lib_id = lib["id"]
    assert lib["name"] == "lib-crud"
    assert lib["description"] == "d"
    assert lib["metadata"]["a"] == "b"

    # list
    r = client.get("/libraries/")
    assert r.status_code == 200
    assert any(x["id"] == lib_id for x in r.json())

    # get
    r = client.get(f"/libraries/{lib_id}")
    assert r.status_code == 200
    assert r.json()["id"] == lib_id

    # update
    r = client.patch(
        f"/libraries/{lib_id}",
        json={"name": "lib-upd", "description": "dx", "metadata": {"k": "v"}},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "lib-upd"
    assert data["description"] == "dx"
    assert data["metadata"]["k"] == "v"

    # delete
    r = client.delete(f"/libraries/{lib_id}")
    assert r.status_code == 204

    # get after delete -> 404
    r = client.get(f"/libraries/{lib_id}")
    assert r.status_code == 404


def test_libraries_update_and_get_not_found():
    r = client.patch("/libraries/does-not-exist", json={"name": "x"})
    assert r.status_code == 404
    r = client.get("/libraries/does-not-exist")
    assert r.status_code == 404
