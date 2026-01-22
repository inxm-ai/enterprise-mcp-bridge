import os
import json
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
    monkeypatch.setattr("app.tgi.routes.DEFAULT_MODEL", "test-model")


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
async def test_a2a_chat_completion_streaming(monkeypatch):
    # Simulate streaming response via patching the service used by the router
    chunks = []
    chunk1 = {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}
    chunk2 = {"choices": [{"delta": {"content": " World!"}, "index": 0}]}
    chunks.append("data: " + json.dumps(chunk1) + "\n\n")
    chunks.append("data: " + json.dumps(chunk2) + "\n\n")
    chunks.append("data: [DONE]\n\n")

    async def mock_chat_completion(*args, **kwargs):
        async def gen():
            for c in chunks:
                yield c

        return gen()

    class MockService:
        async def chat_completion(self, *args, **kwargs):
            return await mock_chat_completion()

    monkeypatch.setattr("app.tgi.routes.tgi_service", MockService())

    a2a_request = {
        "jsonrpc": "2.0",
        "method": "enterprise-mcp-bridge",
        "params": {"prompt": "Say hello"},
        "id": "1",
    }

    headers = {"accept": "text/event-stream"}
    response = client.post("/tgi/v1/a2a", json=a2a_request, headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    response_chunks = response.text.split("\n")
    assert any("Hello" in chunk for chunk in response_chunks)
    assert any("World!" in chunk for chunk in response_chunks)
    # A2A format doesn't have [DONE] marker - it just ends the stream
    # Check for JSON-RPC structure instead
    assert any(
        '"jsonrpc"' in chunk and '"result"' in chunk for chunk in response_chunks
    )


@pytest.mark.asyncio
async def test_a2a_chat_completion_non_streaming(monkeypatch):
    async def mock_chat_completion(*args, **kwargs):
        return [{"choices": [{"delta": {"content": "Hello World!"}, "index": 0}]}]

    class MockService:
        async def chat_completion(self, *args, **kwargs):
            return await mock_chat_completion()

    monkeypatch.setattr("app.tgi.routes.tgi_service", MockService())

    a2a_request = {
        "jsonrpc": "2.0",
        "method": "enterprise-mcp-bridge",
        "params": {"prompt": "Say hello"},
        "id": "1",
    }

    response = client.post("/tgi/v1/a2a", json=a2a_request)

    assert response.status_code == 200
    response_json = response.json()
    assert response_json["jsonrpc"] == "2.0"
    assert response_json["id"] == "1"
    assert response_json["error"] is None
    assert "Hello World!" in response_json["result"]["completion"]


@pytest.mark.asyncio
async def test_a2a_chat_completion_minimal_payload(monkeypatch):
    async def mock_chat_completion(*args, **kwargs):
        return [
            {
                "choices": [
                    {"delta": {"content": "Hello Minimal!"}, "index": 0},
                ]
            }
        ]

    class MockService:
        async def chat_completion(self, *args, **kwargs):
            return await mock_chat_completion()

    monkeypatch.setattr("app.tgi.routes.tgi_service", MockService())

    response = client.post("/tgi/v1/a2a", json={"prompt": "Say hello"})

    assert response.status_code == 200
    response_json = response.json()
    assert response_json["jsonrpc"] == "2.0"
    assert response_json["error"] is None
    assert response_json["id"] != "unknown"
    assert "Hello Minimal!" in response_json["result"]["completion"]


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
