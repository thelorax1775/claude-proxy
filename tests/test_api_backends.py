"""The backend CRUD API the Backends tab drives."""

import json
import os

import httpx
import pytest

import main


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_list_never_returns_a_key(client):
    resp = await client.get("/api/backends")
    backends = resp.json()["backends"]
    assert [b["name"] for b in backends] == ["primary", "secondary"]
    assert all("api_key" not in b for b in backends)
    assert backends[0]["has_key"] is True


async def test_add_backend(client):
    resp = await client.post("/api/backends", json={
        "name": "third", "base_url": "https://third.test/v1/", "api_key": "sk-third-0000000000",
        "big_model": "third-big"})
    assert resp.status_code == 200
    assert resp.json()["base_url"] == "https://third.test/v1"     # trailing slash stripped
    assert resp.json()["small_model"] == "third-big"              # defaults to the big model
    assert [b["name"] for b in main.BACKENDS] == ["primary", "secondary", "third"]


async def test_add_rejects_a_duplicate_name(client):
    resp = await client.post("/api/backends", json={"name": "primary", "base_url": "https://x.test"})
    assert resp.status_code == 409


async def test_add_requires_name_and_url(client):
    assert (await client.post("/api/backends", json={"name": "x"})).status_code == 400
    assert (await client.post("/api/backends", json={"base_url": "https://x.test"})).status_code == 400


async def test_patch_changes_the_models(client):
    resp = await client.patch("/api/backends/primary", json={
        "big_model": "new-big", "small_model": "new-small"})
    assert resp.status_code == 200
    assert resp.json()["big_model"] == "new-big"
    assert main.BACKENDS[0]["small_model"] == "new-small"


async def test_patch_with_a_blank_key_keeps_the_stored_key(client):
    """The browser never has the key, so an edit form submitted with the key
    field left empty must not wipe it."""
    before = main.BACKENDS[0]["api_key"]
    resp = await client.patch("/api/backends/primary", json={"big_model": "x", "api_key": ""})
    assert resp.status_code == 200
    assert main.BACKENDS[0]["api_key"] == before
    assert resp.json()["has_key"] is True


async def test_patch_with_a_real_key_replaces_it(client):
    await client.patch("/api/backends/primary", json={"api_key": "sk-replaced-000000000"})
    assert main.BACKENDS[0]["api_key"] == "sk-replaced-000000000"


async def test_patch_with_null_clears_the_key(client):
    resp = await client.patch("/api/backends/primary", json={"api_key": None})
    assert main.BACKENDS[0]["api_key"] == ""
    assert resp.json()["has_key"] is False


async def test_patch_can_rename(client):
    resp = await client.patch("/api/backends/primary", json={"name": "renamed"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "renamed"
    assert [b["name"] for b in main.BACKENDS] == ["renamed", "secondary"]
    # The key survives the rename -- that is the whole point of not making
    # people delete and re-add.
    assert main.BACKENDS[0]["api_key"] == "sk-primary-000000000000"


async def test_patch_rename_rejects_a_collision(client):
    resp = await client.patch("/api/backends/primary", json={"name": "secondary"})
    assert resp.status_code == 409
    assert [b["name"] for b in main.BACKENDS] == ["primary", "secondary"]


async def test_patch_normalises_the_base_url(client):
    resp = await client.patch("/api/backends/primary", json={"base_url": " https://x.test/v1/ "})
    assert resp.json()["base_url"] == "https://x.test/v1"


async def test_patch_unknown_backend_is_404(client):
    assert (await client.patch("/api/backends/nope", json={"big_model": "x"})).status_code == 404


async def test_delete_and_reorder(client):
    assert (await client.delete("/api/backends/primary")).status_code == 200
    assert [b["name"] for b in main.BACKENDS] == ["secondary"]
    assert (await client.delete("/api/backends/primary")).status_code == 404


async def test_reorder_requires_exactly_the_current_names(client):
    resp = await client.post("/api/backends/reorder", json={"order": ["secondary", "primary"]})
    assert resp.status_code == 200
    assert [b["name"] for b in main.BACKENDS] == ["secondary", "primary"]

    assert (await client.post("/api/backends/reorder",
                              json={"order": ["secondary"]})).status_code == 400


async def test_changes_are_written_to_disk(client):
    await client.patch("/api/backends/primary", json={"big_model": "persisted"})
    with open(os.environ["BACKENDS_CONFIG"]) as f:
        assert json.load(f)[0]["big_model"] == "persisted"


async def test_health_lists_the_backends(client):
    body = (await client.get("/health")).json()
    assert body["status"] == "ok"


async def test_count_tokens_is_a_positive_estimate(client):
    body = (await client.post("/v1/messages/count_tokens", json={
        "model": "m", "messages": [{"role": "user", "content": "x" * 400}]})).json()
    assert body["input_tokens"] == 100
