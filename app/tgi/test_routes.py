import os
from contextlib import asynccontextmanager
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.tgi.routes import router


def _create_test_client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# Initialize the FastAPI TestClient
client = _create_test_client()


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    os.environ["TGI_URL"] = "https://api.test-llm.com/v1"
    os.environ["TGI_TOKEN"] = "test-token-123"


@pytest.fixture(autouse=True)
def _mock_mcp_session_context(monkeypatch):
    @asynccontextmanager
    async def fake_mcp_context(
        sessions, x_inxm_mcp_session, access_token, group, incoming_headers=None
    ):
        class DummySession:
            pass

        yield DummySession()

    monkeypatch.setattr("app.tgi.routes.mcp_session_context", fake_mcp_context)


@pytest.mark.asyncio
async def test_chat_completion_forwards_request_headers(monkeypatch):
    captured = {}

    @asynccontextmanager
    async def fake_mcp_context(
        sessions, x_inxm_mcp_session, access_token, group, incoming_headers=None
    ):
        captured["incoming_headers"] = incoming_headers or {}

        class DummySession:
            pass

        yield DummySession()

    class MockService:
        async def chat_completion(self, *args, **kwargs):
            return {
                "choices": [
                    {"delta": {"content": "Hello Headers!"}, "index": 0},
                ]
            }

    monkeypatch.setattr("app.tgi.routes.mcp_session_context", fake_mcp_context)
    monkeypatch.setattr("app.tgi.routes.tgi_service", MockService())

    headers = {
        "X-Test-Header": "test-value",
        "X-Auth-Request-User": "demo@example.com",
    }
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Say hello"}],
        "stream": False,
    }

    response = client.post("/tgi/v1/chat/completions", json=payload, headers=headers)

    assert response.status_code == 200
    response_json = response.json()
    assert "Hello Headers!" in response_json["choices"][0]["delta"]["content"]
    forwarded = captured["incoming_headers"]
    assert forwarded.get("x-test-header") == "test-value"
    assert forwarded.get("x-auth-request-user") == "demo@example.com"
