from contextlib import asynccontextmanager
import json

import httpx
import pytest

from app.server import app
from app.app_facade.generated_service import Actor


class StubConversationalService:
    def create_draft_session(self, **_kwargs):
        return {
            "session_id": "sess123",
            "base_version": 1,
            "draft_version": 1,
            "expires_at": "2099-01-01T00:00:00+00:00",
        }

    async def stream_chat_update(self, **_kwargs):
        yield b'event: assistant\ndata: {"delta":"ok"}\n\n'
        yield b'event: ui_updated\ndata: {"draft_version":2,"update_mode":"patch_applied"}\n\n'
        yield b'event: done\ndata: {"draft_version":2}\n\n'
        yield b"data: [DONE]\n\n"

    def get_draft_ui(self, **_kwargs):
        return {
            "metadata": {"id": "dash1", "name": "overview", "draft_version": 2},
            "current": {
                "html": {
                    "page": "<!DOCTYPE html><html><body><div>draft</div></body></html>",
                    "snippet": "<div>draft</div>",
                }
            },
        }

    def publish_draft_session(self, **_kwargs):
        return {
            "metadata": {"version": 2, "published_at": "2026-01-01T00:00:00+00:00"},
            "current": {"html": {"snippet": "<div>published</div>"}},
        }

    def discard_draft_session(self, **_kwargs):
        return True


@pytest.fixture
def conversational_setup(monkeypatch):
    monkeypatch.setenv("APP_CONVERSATIONAL_UI_ENABLED", "true")
    monkeypatch.setenv("TGI_URL", "https://example.com/tgi")
    stub = StubConversationalService()
    monkeypatch.setattr("app.app_facade.route._get_generated_service", lambda: stub)
    monkeypatch.setattr(
        "app.app_facade.route._extract_actor",
        lambda _token: Actor(user_id="user123", groups=["eng"]),
    )

    @asynccontextmanager
    async def dummy_session_context(*_args, **_kwargs):
        yield object()

    monkeypatch.setattr(
        "app.app_facade.route.mcp_session_context", dummy_session_context
    )
    return stub


def _client():
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_create_chat_session_endpoint(conversational_setup):
    async with _client() as client:
        response = await client.post(
            "/app/_generated/user=user123/dash1/overview/chat/sessions",
            json={},
            headers={"X-Auth-Request-Access-Token": "token"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "sess123"
    assert payload["base_version"] == 1


@pytest.mark.asyncio
async def test_chat_message_stream_endpoint(conversational_setup):
    async with _client() as client:
        response = await client.post(
            "/app/_generated/user=user123/dash1/overview/chat/sessions/sess123/messages",
            json={"message": "Update the title"},
            headers={"X-Auth-Request-Access-Token": "token"},
        )
    assert response.status_code == 200
    body = response.text
    assert "event: assistant" in body
    assert "event: ui_updated" in body
    assert "patch_applied" in body


@pytest.mark.asyncio
async def test_get_draft_card_and_publish(conversational_setup):
    async with _client() as client:
        draft_resp = await client.get(
            "/app/_generated/user=user123/dash1/overview/draft",
            params={"session_id": "sess123", "as": "card"},
            headers={"X-Auth-Request-Access-Token": "token"},
        )
        publish_resp = await client.post(
            "/app/_generated/user=user123/dash1/overview/chat/sessions/sess123/publish",
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert draft_resp.status_code == 200
    assert draft_resp.json()["metadata"]["draft_version"] == 2
    assert publish_resp.status_code == 200
    assert publish_resp.json()["status"] == "published"
    assert publish_resp.json()["version"] == 2


@pytest.mark.asyncio
async def test_container_endpoint_and_discard(conversational_setup):
    async with _client() as client:
        container_resp = await client.get(
            "/app/_generated/user=user123/dash1/overview/container"
        )
        discard_resp = await client.delete(
            "/app/_generated/user=user123/dash1/overview/chat/sessions/sess123",
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert container_resp.status_code == 200
    assert "Conversational UI Editor" in container_resp.text
    assert "/chat/sessions" in container_resp.text
    assert discard_resp.status_code == 200
    assert discard_resp.json()["session_deleted"] is True


@pytest.mark.asyncio
async def test_conversational_endpoints_disabled(monkeypatch):
    monkeypatch.setenv("APP_CONVERSATIONAL_UI_ENABLED", "false")
    async with _client() as client:
        response = await client.get(
            "/app/_generated/user=user123/dash1/overview/container"
        )

    assert response.status_code == 404
    assert "disabled" in json.dumps(response.json()).lower()
