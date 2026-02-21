from contextlib import asynccontextmanager
import json
from pathlib import Path

import httpx
import pytest

from app.server import app
from app.app_facade.generated_service import Actor


class StubConversationalService:
    def __init__(self):
        self.last_chat_kwargs = None

    def create_draft_session(self, **_kwargs):
        return {
            "session_id": "sess123",
            "base_version": 1,
            "draft_version": 1,
            "expires_at": "2099-01-01T00:00:00+00:00",
        }

    async def stream_chat_update(self, **kwargs):
        self.last_chat_kwargs = kwargs
        yield b'event: assistant\ndata: {"delta":"ok"}\n\n'
        yield b'event: ui_updated\ndata: {"draft_version":2,"update_mode":"patch_applied"}\n\n'
        yield b'event: tests_queued\ndata: {"run_id":"run-1","trigger":"post_update","draft_version":2}\n\n'
        yield b'event: done\ndata: {"draft_version":2}\n\n'
        yield b"data: [DONE]\n\n"

    async def queue_test_action(self, **_kwargs):
        return {"status": "queued", "run_id": "run-2", "trigger": "manual_run"}

    async def stream_test_events(self, **_kwargs):
        yield b'event: test_status\ndata: {"run_id":"run-2","state":"queued","trigger":"manual_run"}\n\n'
        yield b'event: test_result\ndata: {"status":"passed","passed":3,"failed":0}\n\n'
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
    draft_action = {
        "type": "runtime_service_exchanges",
        "cursor": 2,
        "entries": [{"tool": "list_items", "response_payload": {"items": []}}],
    }
    async with _client() as client:
        response = await client.post(
            "/app/_generated/user=user123/dash1/overview/chat/sessions/sess123/messages",
            json={"message": "Update the title", "draft_action": draft_action},
            headers={"X-Auth-Request-Access-Token": "token"},
        )
    assert response.status_code == 200
    body = response.text
    assert "event: assistant" in body
    assert "event: ui_updated" in body
    assert "event: tests_queued" in body
    assert "patch_applied" in body
    assert conversational_setup.last_chat_kwargs["draft_action"] == draft_action


@pytest.mark.asyncio
async def test_test_action_and_stream_endpoints(conversational_setup):
    async with _client() as client:
        action_resp = await client.post(
            "/app/_generated/user=user123/dash1/overview/chat/sessions/sess123/tests/actions",
            json={"action": "run"},
            headers={"X-Auth-Request-Access-Token": "token"},
        )
        stream_resp = await client.get(
            "/app/_generated/user=user123/dash1/overview/chat/sessions/sess123/tests/stream",
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert action_resp.status_code == 202
    assert action_resp.json()["status"] == "queued"
    assert stream_resp.status_code == 200
    assert "event: test_status" in stream_resp.text
    assert "event: test_result" in stream_resp.text


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
    assert "/tests/stream" in container_resp.text
    assert "Test stream is live per draft session." in container_resp.text
    assert "checking the feedback" in container_resp.text
    assert "running smoke tests and performing final fixes" in container_resp.text
    assert "loading..." in container_resp.text
    assert "generated_ui_container.html" not in container_resp.text
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


@pytest.mark.asyncio
async def test_start_endpoint_enabled(conversational_setup):
    async with _client() as client:
        response = await client.get(
            "/app/_generated/start",
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert response.status_code == 200
    assert "Start Generated UI" in response.text
    assert "tool_start" in response.text
    assert "test_result" in response.text
    assert "done" in response.text
    assert "const target = 'user=user123';" in response.text
    assert "const appPrefix = '/app';" in response.text
    assert "window.location.assign" in response.text
    assert "/container" in response.text


@pytest.mark.asyncio
async def test_start_endpoint_disabled(monkeypatch):
    monkeypatch.setenv("APP_CONVERSATIONAL_UI_ENABLED", "false")
    async with _client() as client:
        response = await client.get(
            "/app/_generated/start",
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert response.status_code == 404
    assert "disabled" in json.dumps(response.json()).lower()


@pytest.mark.asyncio
async def test_container_template_missing_returns_500(
    conversational_setup, monkeypatch
):
    monkeypatch.setattr(
        "app.app_facade.route._template_dir",
        lambda: Path("/definitely-missing-template-dir"),
    )
    async with _client() as client:
        response = await client.get(
            "/app/_generated/user=user123/dash1/overview/container"
        )

    assert response.status_code == 500
    assert "template not found" in json.dumps(response.json()).lower()


@pytest.mark.asyncio
async def test_start_template_missing_returns_500(conversational_setup, monkeypatch):
    monkeypatch.setattr(
        "app.app_facade.route._template_dir",
        lambda: Path("/definitely-missing-template-dir"),
    )
    async with _client() as client:
        response = await client.get(
            "/app/_generated/start",
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert response.status_code == 500
    assert "template not found" in json.dumps(response.json()).lower()


@pytest.mark.asyncio
async def test_start_template_read_error_returns_500(conversational_setup, monkeypatch):
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, encoding="utf-8": (_ for _ in ()).throw(OSError("read failed")),
    )
    async with _client() as client:
        response = await client.get(
            "/app/_generated/start",
            headers={"X-Auth-Request-Access-Token": "token"},
        )

    assert response.status_code == 500
    assert "failed to load template" in json.dumps(response.json()).lower()
