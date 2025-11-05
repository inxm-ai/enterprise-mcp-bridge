from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import pytest
from fastapi.testclient import TestClient

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

    async def create_dashboard(
        self,
        *,
        session,
        scope,
        actor,
        dashboard_id,
        name,
        prompt,
        tools,
        access_token,
    ):
        self.last_create = {
            "session": session,
            "scope": scope,
            "actor": actor,
            "dashboard_id": dashboard_id,
            "name": name,
            "prompt": prompt,
            "tools": list(tools or []),
            "access_token": access_token,
        }
        return self.record

    async def update_dashboard(
        self,
        *,
        session,
        scope,
        actor,
        dashboard_id,
        name,
        prompt,
        tools,
        access_token,
    ):
        self.last_update = {
            "session": session,
            "scope": scope,
            "actor": actor,
            "dashboard_id": dashboard_id,
            "name": name,
            "prompt": prompt,
            "tools": list(tools or []),
            "access_token": access_token,
        }
        updated = dict(self.record)
        updated["metadata"] = dict(self.record["metadata"])
        updated["metadata"]["updated_at"] = "2024-01-02T00:00:00Z"
        return updated

    def get_dashboard(self, *, scope, actor, dashboard_id, name):
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


def test_create_generated_dashboard(client):
    response = client.post(
        "/app/_generated/user=user123",
        json={
            "id": "dash1",
            "name": "overview",
            "prompt": "Build a dashboard",
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
    assert stub.last_create["prompt"] == "Build a dashboard"


def test_get_generated_dashboard_snippet(client):
    response = client.get(
        "/app/_generated/user=user123/dash1/overview",
        params={"as": "snippet"},
        headers={"X-Auth-Request-Access-Token": "token"},
    )
    assert response.status_code == 200
    assert response.text.strip() == "<div>Full Page</div>"


def test_update_generated_dashboard(client):
    response = client.post(
        "/app/_generated/user=user123/dash1/overview",
        json={
            "prompt": "Refine the dashboard layout",
            "tools": ["insights_tool"],
        },
        headers={"X-Auth-Request-Access-Token": "token"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["updated_at"] == "2024-01-02T00:00:00Z"
    stub = client.stub_service  # type: ignore[attr-defined]
    assert stub.last_update["tools"] == ["insights_tool"]
