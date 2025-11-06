from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import pytest
from fastapi.testclient import TestClient
import json

from app.server import app
from app.app_facade.generated_service import Actor


class StubGeneratedService:
    def __init__(self):
        self.last_create: Optional[Dict[str, Any]] = None
        self.last_update: Optional[Dict[str, Any]] = None
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

    async def create_ui(
        self,
        *,
        session,
        scope,
        actor,
        ui_id,
        name,
        prompt,
        tools,
        access_token,
    ):
        self.last_create = {
            "session": session,
            "scope": scope,
            "actor": actor,
            "ui_id": ui_id,
            "name": name,
            "prompt": prompt,
            "tools": list(tools or []),
            "access_token": access_token,
        }
        return self.record

    async def update_ui(
        self,
        *,
        session,
        scope,
        actor,
        ui_id,
        name,
        prompt,
        tools,
        access_token,
    ):
        self.last_update = {
            "session": session,
            "scope": scope,
            "actor": actor,
            "ui_id": ui_id,
            "name": name,
            "prompt": prompt,
            "tools": list(tools or []),
            "access_token": access_token,
        }
        updated = dict(self.record)
        updated["metadata"] = dict(self.record["metadata"])
        updated["metadata"]["updated_at"] = "2024-01-02T00:00:00Z"
        return updated

    def get_ui(self, *, scope, actor, ui_id, name):
        return self.record


@pytest.fixture
def client(monkeypatch):
    stub_service = StubGeneratedService()

    monkeypatch.setenv("TGI_URL", "https://example.com/tgi")

    monkeypatch.setattr(
        "app.app_facade.route._get_generated_service", lambda: stub_service
    )

    monkeypatch.setattr(
        "app.app_facade.route._extract_actor",
        lambda access_token: Actor(user_id="user123", groups=["group123"]),
    )

    @asynccontextmanager
    async def dummy_session_context(*_args, **_kwargs):
        yield object()

    monkeypatch.setattr(
        "app.app_facade.route.mcp_session_context", dummy_session_context
    )

    client = TestClient(app)
    client.stub_service = stub_service  # type: ignore[attr-defined]
    return client


def test_create_generated_ui(client):
    response = client.post(
        "/app/_generated/user=user123",
        json={
            "id": "dash1",
            "name": "overview",
            "prompt": "Build a ui",
        },
        headers={"X-Auth-Request-Access-Token": "token"},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["id"] == "dash1"
    assert payload["name"] == "overview"
    assert payload["scope"] == {"type": "user", "id": "user123"}
    assert payload["metadata"]["components"] == ["pfusch-card"]
    stub = client.stub_service  # type: ignore[attr-defined]
    assert stub.last_create["prompt"] == "Build a ui"


def test_get_generated_ui_snippet(client):
    response = client.get(
        "/app/_generated/user=user123/dash1/overview",
        params={"as": "snippet"},
        headers={"X-Auth-Request-Access-Token": "token"},
    )
    assert response.status_code == 200
    assert response.text.strip() == "<div>Full Page</div>"


def test_update_generated_ui(client):
    response = client.post(
        "/app/_generated/user=user123/dash1/overview",
        json={
            "prompt": "Refine the ui layout",
            "tools": ["insights_tool"],
        },
        headers={"X-Auth-Request-Access-Token": "token"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["updated_at"] == "2024-01-02T00:00:00Z"
    stub = client.stub_service  # type: ignore[attr-defined]
    assert stub.last_update["tools"] == ["insights_tool"]


def _make_streaming_client(monkeypatch, stream_impl, actor_override=None):
    """Helper to create a TestClient with a given streaming service implementation.

    stream_impl: an object implementing async def stream_generate_ui(...)
    actor_override: optional Actor to return from _extract_actor
    """
    monkeypatch.setenv("TGI_URL", "https://example.com/tgi")

    monkeypatch.setattr(
        "app.app_facade.route._get_generated_service", lambda: stream_impl
    )

    if actor_override is not None:
        monkeypatch.setattr(
            "app.app_facade.route._extract_actor", lambda access_token: actor_override
        )
    else:
        monkeypatch.setattr(
            "app.app_facade.route._extract_actor",
            lambda access_token: Actor(user_id="user123", groups=["group123"]),
        )

    @asynccontextmanager
    async def dummy_session_context(*_args, **_kwargs):
        yield object()

    monkeypatch.setattr(
        "app.app_facade.route.mcp_session_context", dummy_session_context
    )

    client = TestClient(app)
    client.stub_service = stream_impl  # type: ignore[attr-defined]
    return client


def test_create_generated_ui_streaming_already_exists(monkeypatch):
    class StreamStub:
        async def stream_generate_ui(self, **_kwargs):
            # Simulate immediate error emitted as SSE
            yield b'event: error\ndata: {"error": "Ui already exists for this id and name"}\n\n'

    client = _make_streaming_client(monkeypatch, StreamStub())

    response = client.post(
        "/app/_generated/user=user123",
        json={"id": "dash1", "name": "overview", "prompt": "Build a ui"},
        headers={"X-Auth-Request-Access-Token": "token"},
    )

    assert response.status_code == 201
    assert b"Ui already exists for this id and name" in response.content


def test_create_generated_ui_streaming_non_owner_user(monkeypatch):
    class StreamStub:
        async def stream_generate_ui(self, **_kwargs):
            # Simulate permission error emitted as SSE
            yield b'event: error\ndata: {"error": "User uis may only be created by the owning user"}\n\n'

    # Actor not matching scope
    non_owner = Actor(user_id="other_user", groups=["group123"])
    client = _make_streaming_client(monkeypatch, StreamStub(), actor_override=non_owner)

    response = client.post(
        "/app/_generated/user=user123",
        json={"id": "dash1", "name": "overview", "prompt": "Build a ui"},
        headers={"X-Auth-Request-Access-Token": "token"},
    )

    assert response.status_code == 201
    assert b"User uis may only be created by the owning user" in response.content


def test_create_generated_ui_streaming_group_non_member(monkeypatch):
    class StreamStub:
        async def stream_generate_ui(self, **_kwargs):
            yield b'event: error\ndata: {"error": "Group uis may only be created by group members"}\n\n'

    # Actor who is not member of target group
    not_member = Actor(user_id="someone", groups=["other_group"])
    client = _make_streaming_client(
        monkeypatch, StreamStub(), actor_override=not_member
    )

    response = client.post(
        "/app/_generated/group=groupX",
        json={"id": "dash1", "name": "overview", "prompt": "Build a ui"},
        headers={"X-Auth-Request-Access-Token": "token"},
    )

    assert response.status_code == 201
    assert b"Group uis may only be created by group members" in response.content


def test_create_generated_ui_streaming_success(monkeypatch):
    class StreamStub:
        async def stream_generate_ui(self, **_kwargs):
            # Send a keepalive, then a chunk, then final done event
            yield b":\n\n"  # keepalive
            yield b'data: {"chunk": "part1"}\n\n'
            # simulate final created record payload
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
            yield f"event: done\ndata: {payload}\n\n".encode("utf-8")

    client = _make_streaming_client(monkeypatch, StreamStub())

    response = client.post(
        "/app/_generated/user=user123",
        json={"id": "dash1", "name": "overview", "prompt": "Build a ui"},
        headers={"X-Auth-Request-Access-Token": "token"},
    )

    assert response.status_code == 201
    # Ensure keepalive and chunk and done event are present
    assert b":\n\n" in response.content
    assert b'"chunk": "part1"' in response.content
    assert b'"status": "created"' in response.content
