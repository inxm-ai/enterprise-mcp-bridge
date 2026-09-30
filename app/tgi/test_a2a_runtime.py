import pytest
from app.tgi.a2a_runtime import _access_token_from_headers, build_a2a_app


def test_a2a_app_exposes_standard_jsonrpc_and_agent_card_routes():
    app = build_a2a_app()
    paths = {getattr(route, "path", None) for route in app.routes}

    assert "/tgi/v1/a2a" in paths
    assert "/.well-known/agent-card.json" in paths
    assert "/.well-known/agent.json" in paths


def test_a2a_auth_accepts_standard_bearer_token():
    assert (
        _access_token_from_headers({"authorization": "Bearer test-token"})
        == "test-token"
    )


def test_legacy_custom_prompt_payload_is_not_accepted_as_a2a():
    from fastapi.testclient import TestClient

    client = TestClient(build_a2a_app())
    response = client.post(
        "/tgi/v1/a2a",
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "enterprise-mcp-bridge",
            "params": {"prompt": "Say hello"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] == "1"
    assert payload["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_executor_forwards_request_context_and_completes(monkeypatch):
    from contextlib import asynccontextmanager

    from a2a.server.agent_execution import RequestContext
    from a2a.server.context import ServerCallContext
    from a2a.server.events import EventQueue
    from a2a.types import Message, Part, Role, SendMessageRequest, TaskArtifactUpdateEvent, TaskStatusUpdateEvent

    from app.tgi import a2a_runtime

    captured = {}

    class CaptureQueue(EventQueue):
        def __init__(self):
            self.events = []

        async def enqueue_event(self, event):
            self.events.append(event)

    @asynccontextmanager
    async def fake_mcp_context(
        sessions, session_key, access_token, group, incoming_headers=None
    ):
        captured["session_key"] = session_key
        captured["access_token"] = access_token
        captured["group"] = group
        captured["headers"] = incoming_headers

        class DummySession:
            pass

        yield DummySession()

    class MockService:
        async def chat_completion(
            self, session, request, user_token, access_token, prompt
        ):
            captured["request"] = request
            captured["user_token"] = user_token
            captured["prompt"] = prompt
            return {"choices": [{"message": {"content": "hello from agent"}}]}

    monkeypatch.setattr(a2a_runtime, "DEFAULT_MODEL", "test-model")
    monkeypatch.setattr(a2a_runtime, "mcp_session_context", fake_mcp_context)
    monkeypatch.setattr(a2a_runtime, "tgi_service", MockService())

    headers = {
        "authorization": "Bearer access-token",
        "x-inxm-mcp-session": "session-123",
        "x-forwarded-extra": "kept",
    }
    request = SendMessageRequest(
        message=Message(
            message_id="msg-1",
            role=Role.ROLE_USER,
            parts=[Part(text="hello")],
        ),
        metadata={
            "group": "engineering",
            "prompt": "system prompt",
            "use_workflow": True,
            "workflow_execution_id": "wf-1",
        },
    )
    context = RequestContext(
        call_context=ServerCallContext(state={"headers": headers}),
        request=request,
    )
    queue = CaptureQueue()

    await a2a_runtime.BridgeA2AExecutor().execute(context, queue)

    assert captured["access_token"] == "access-token"
    assert captured["group"] == "engineering"
    assert captured["headers"]["x-forwarded-extra"] == "kept"
    assert captured["prompt"] == "system prompt"
    assert captured["request"].use_workflow is True
    assert captured["request"].workflow_execution_id == "wf-1"
    assert captured["request"].messages[0].content == "hello"

    artifacts = [
        event for event in queue.events if isinstance(event, TaskArtifactUpdateEvent)
    ]
    statuses = [
        event for event in queue.events if isinstance(event, TaskStatusUpdateEvent)
    ]

    assert len(artifacts) == 1
    assert artifacts[0].artifact.parts[0].text == "hello from agent"
    assert statuses[-1].status.state.name == "TASK_STATE_COMPLETED"
