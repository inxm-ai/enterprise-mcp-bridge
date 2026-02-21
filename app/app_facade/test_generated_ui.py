from contextlib import asynccontextmanager
import json
from typing import Any, Dict, Optional

import httpx
import pytest

from app.server import app
from app.app_facade.generated_service import Actor


class StubGeneratedService:
    def __init__(self):
        self.last_create: Optional[Dict[str, Any]] = None
        self.last_update: Optional[Dict[str, Any]] = None
        self.storage = type(
            "StorageStub",
            (),
            {"exists": lambda _self, _scope, _ui_id, _name: False},
        )()
        self.record = {
            "metadata": {
                "id": "dash1",
                "name": "overview",
                "scope": {"type": "user", "id": "user123"},
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "history": [],
            },
            "current": {
                "html": {
                    "page": "<html><body><div>Full Page</div></body></html>",
                    "snippet": "<div>Full Page</div>",
                },
                "metadata": {"components": ["pfusch-card"]},
            },
        }

    async def stream_update_ui(self, **kwargs):
        self.last_update = {
            "session": kwargs.get("session"),
            "scope": kwargs.get("scope"),
            "actor": kwargs.get("actor"),
            "ui_id": kwargs.get("ui_id"),
            "name": kwargs.get("name"),
            "prompt": kwargs.get("prompt"),
            "tools": list(kwargs.get("tools") or []),
            "access_token": kwargs.get("access_token"),
        }
        metadata = dict(self.record.get("metadata", {}))
        metadata["updated_at"] = "2024-01-02T00:00:00Z"
        record = {
            "metadata": metadata,
            "current": self.record.get("current", {}),
        }
        payload = json.dumps(
            {"status": "updated", "record": record}, ensure_ascii=False
        )
        yield f"event: done\\ndata: {payload}\\n\\n".encode("utf-8")

    def get_ui(self, *, scope, actor, ui_id, name, expand=False):
        return self.record

    async def stream_generate_ui(self, **kwargs):
        self.last_create = {
            "session": kwargs.get("session"),
            "scope": kwargs.get("scope"),
            "actor": kwargs.get("actor"),
            "ui_id": kwargs.get("ui_id"),
            "name": kwargs.get("name"),
            "prompt": kwargs.get("prompt"),
            "tools": list(kwargs.get("tools") or []),
            "access_token": kwargs.get("access_token"),
        }

        metadata = self.record.get("metadata", {})
        current = self.record.get("current", {})
        formatted = {
            "id": metadata.get("id"),
            "name": metadata.get("name"),
            "scope": metadata.get("scope"),
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
            "html": current.get("html"),
            "metadata": current.get("metadata"),
        }
        if metadata.get("history"):
            formatted["history"] = metadata.get("history")

        payload = json.dumps(formatted, ensure_ascii=False).encode("utf-8")
        yield payload


@pytest.fixture
def generated_setup(monkeypatch):
    stub_service = StubGeneratedService()

    monkeypatch.setenv("TGI_URL", "https://example.com/tgi")
    monkeypatch.setattr(
        "app.app_facade.route._get_generated_service", lambda: stub_service
    )
    monkeypatch.setattr(
        "app.app_facade.route._extract_actor",
        lambda _access_token: Actor(user_id="user123", groups=["group123"]),
    )

    @asynccontextmanager
    async def dummy_session_context(*_args, **_kwargs):
        yield object()

    monkeypatch.setattr(
        "app.app_facade.route.mcp_session_context", dummy_session_context
    )
    return stub_service


def _client():
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _setup_streaming_client(monkeypatch, stream_impl, actor_override=None):
    monkeypatch.setenv("TGI_URL", "https://example.com/tgi")
    monkeypatch.setattr(
        "app.app_facade.route._get_generated_service", lambda: stream_impl
    )
    if actor_override is None:
        monkeypatch.setattr(
            "app.app_facade.route._extract_actor",
            lambda _access_token: Actor(user_id="user123", groups=["group123"]),
        )
    else:
        monkeypatch.setattr(
            "app.app_facade.route._extract_actor", lambda _access_token: actor_override
        )

    @asynccontextmanager
    async def dummy_session_context(*_args, **_kwargs):
        yield object()

    monkeypatch.setattr(
        "app.app_facade.route.mcp_session_context", dummy_session_context
    )


@pytest.mark.asyncio
async def test_create_generated_ui(generated_setup):
    async with _client() as client:
        response = await client.post(
            "/app/_generated/user=user123",
            json={
                "id": "dash1",
                "name": "overview",
                "prompt": "Build a ui",
            },
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "dash1"
    assert payload["name"] == "overview"
    assert payload["scope"] == {"type": "user", "id": "user123"}
    assert payload["metadata"]["components"] == ["pfusch-card"]
    assert generated_setup.last_create["prompt"] == "Build a ui"
    assert generated_setup.last_create["name"] == "overview"


@pytest.mark.asyncio
async def test_create_generated_ui_without_name_auto_generated(generated_setup):
    async with _client() as client:
        response = await client.post(
            "/app/_generated/user=user123",
            json={
                "id": "dash1",
                "prompt": "Build a ui",
            },
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert response.status_code == 200
    assert generated_setup.last_create["name"] == "build-a-ui"


@pytest.mark.asyncio
async def test_create_generated_ui_without_name_fallback_slug(generated_setup):
    async with _client() as client:
        response = await client.post(
            "/app/_generated/user=user123",
            json={
                "id": "dash1",
                "prompt": "!!! ... ---",
            },
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert response.status_code == 200
    assert generated_setup.last_create["name"] == "generated-ui"


@pytest.mark.asyncio
async def test_create_generated_ui_without_name_collision_suffix(generated_setup):
    existing = {"build-a-ui", "build-a-ui-2"}

    def _exists(_scope, _ui_id, candidate):
        return candidate in existing

    generated_setup.storage.exists = _exists

    async with _client() as client:
        response = await client.post(
            "/app/_generated/user=user123",
            json={
                "id": "dash1",
                "prompt": "Build a ui",
            },
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert response.status_code == 200
    assert generated_setup.last_create["name"] == "build-a-ui-3"


@pytest.mark.asyncio
async def test_create_generated_ui_without_name_blank_value_auto_generated(
    generated_setup,
):
    async with _client() as client:
        response = await client.post(
            "/app/_generated/user=user123",
            json={
                "id": "dash1",
                "name": "  ",
                "prompt": "Build a ui",
            },
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert response.status_code == 200
    assert generated_setup.last_create["name"] == "build-a-ui"


@pytest.mark.asyncio
async def test_create_generated_ui_invalid_ui_id_returns_400(generated_setup):
    async with _client() as client:
        response = await client.post(
            "/app/_generated/user=user123",
            json={
                "id": "!bad",
                "prompt": "Build a ui",
            },
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_generated_ui_snippet(generated_setup):
    async with _client() as client:
        response = await client.get(
            "/app/_generated/user=user123/dash1/overview",
            params={"as": "snippet"},
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert response.status_code == 200
    assert response.text.strip() == "<div>Full Page</div>"


@pytest.mark.asyncio
async def test_update_generated_ui(generated_setup):
    async with _client() as client:
        response = await client.post(
            "/app/_generated/user=user123/dash1/overview",
            json={
                "prompt": "Refine the ui layout",
                "tools": ["insights_tool"],
            },
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert response.status_code == 200
    marker = "event: done\\ndata: "
    start = response.text.find(marker)
    assert start != -1
    start += len(marker)
    end = response.text.find("\\n\\n", start)
    assert end != -1
    body = json.loads(response.text[start:end])
    assert body["status"] == "updated"
    assert body["record"]["metadata"]["updated_at"] == "2024-01-02T00:00:00Z"
    assert generated_setup.last_update["tools"] == ["insights_tool"]


@pytest.mark.asyncio
async def test_create_generated_ui_streaming_already_exists(monkeypatch):
    class StreamStub:
        storage = type(
            "StorageStub",
            (),
            {"exists": lambda _self, _scope, _ui_id, _name: False},
        )()

        async def stream_generate_ui(self, **_kwargs):
            yield b'event: error\\ndata: {"error": "Ui already exists for this id and name"}\\n\\n'

    _setup_streaming_client(monkeypatch, StreamStub())

    async with _client() as client:
        response = await client.post(
            "/app/_generated/user=user123",
            json={"id": "dash1", "name": "overview", "prompt": "Build a ui"},
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert response.status_code == 200
    assert "Ui already exists for this id and name" in response.text


@pytest.mark.asyncio
async def test_create_generated_ui_streaming_non_owner_user(monkeypatch):
    class StreamStub:
        storage = type(
            "StorageStub",
            (),
            {"exists": lambda _self, _scope, _ui_id, _name: False},
        )()

        async def stream_generate_ui(self, **_kwargs):
            yield b'event: error\\ndata: {"error": "User uis may only be created by the owning user"}\\n\\n'

    non_owner = Actor(user_id="other_user", groups=["group123"])
    _setup_streaming_client(monkeypatch, StreamStub(), actor_override=non_owner)

    async with _client() as client:
        response = await client.post(
            "/app/_generated/user=user123",
            json={"id": "dash1", "name": "overview", "prompt": "Build a ui"},
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert response.status_code == 200
    assert "User uis may only be created by the owning user" in response.text


@pytest.mark.asyncio
async def test_create_generated_ui_streaming_group_non_member(monkeypatch):
    class StreamStub:
        storage = type(
            "StorageStub",
            (),
            {"exists": lambda _self, _scope, _ui_id, _name: False},
        )()

        async def stream_generate_ui(self, **_kwargs):
            yield b'event: error\\ndata: {"error": "Group uis may only be created by group members"}\\n\\n'

    not_member = Actor(user_id="someone", groups=["other_group"])
    _setup_streaming_client(monkeypatch, StreamStub(), actor_override=not_member)

    async with _client() as client:
        response = await client.post(
            "/app/_generated/group=groupX",
            json={"id": "dash1", "name": "overview", "prompt": "Build a ui"},
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert response.status_code == 200
    assert "Group uis may only be created by group members" in response.text


@pytest.mark.asyncio
async def test_create_generated_ui_streaming_success(monkeypatch):
    class StreamStub:
        storage = type(
            "StorageStub",
            (),
            {"exists": lambda _self, _scope, _ui_id, _name: False},
        )()

        async def stream_generate_ui(self, **_kwargs):
            yield b":\\n\\n"
            yield b'data: {"chunk": "part1"}\\n\\n'
            record = {
                "status": "created",
                "record": {
                    "metadata": {
                        "id": "dash1",
                        "name": "overview",
                        "scope": {"type": "user", "id": "user123"},
                    },
                    "current": {
                        "html": {"page": "<html/>", "snippet": "<div/>"},
                        "metadata": {"components": ["pfusch-card"]},
                    },
                },
            }
            payload = json.dumps(record, ensure_ascii=False)
            yield f"event: done\\ndata: {payload}\\n\\n".encode("utf-8")

    _setup_streaming_client(monkeypatch, StreamStub())

    async with _client() as client:
        response = await client.post(
            "/app/_generated/user=user123",
            json={"id": "dash1", "name": "overview", "prompt": "Build a ui"},
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert response.status_code == 200
    assert ":\\n\\n" in response.text
    assert '"chunk": "part1"' in response.text
    assert '"status": "created"' in response.text
