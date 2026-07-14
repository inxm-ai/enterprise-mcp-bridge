import asyncio

import jwt
import pytest
from starlette.requests import Request

from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.routes import _handle_chat_completion
from app.tgi.workflows.models import WorkflowExecutionState


@pytest.fixture(autouse=True)
def _set_tgi_url(monkeypatch):
    monkeypatch.setenv("TGI_URL", "https://api.test-llm.com/v1")
    monkeypatch.setenv("TGI_TOKEN", "test-token-123")


class _FakeSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class _FakeTracer:
    """Minimal stand-in for an opentelemetry Tracer, recording set_attribute calls."""

    def __init__(self):
        self.span = _FakeSpan()

    def start_as_current_span(self, name):
        return self

    def __enter__(self):
        return self.span

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def fake_tracer(monkeypatch):
    tracer = _FakeTracer()
    monkeypatch.setattr("app.tgi.routes.tracer", tracer)
    return tracer


def _make_request(headers: dict[str, str]) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/tgi/v1/chat/completions",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    return Request(scope)


class _StubStateStore:
    def __init__(self, state: WorkflowExecutionState):
        self._state = state

    def load_execution(self, execution_id: str):
        return self._state


class _StubEngine:
    def __init__(self, state: WorkflowExecutionState):
        self.state_store = _StubStateStore(state)

    def _enforce_workflow_owner(self, state: WorkflowExecutionState, user_token: str):
        pass


class _StubBackground:
    def __init__(self, queue_items: list[tuple[int, str, bool]]):
        self.queue_items = queue_items
        self.started = False

    async def get_or_start(self, execution_id, stream_factory, initial_event_count=0):
        self.started = True
        return None

    async def __aenter__(self):
        queue: asyncio.Queue = asyncio.Queue()
        for item in self.queue_items:
            await queue.put(item)
        await queue.put((0, None, False))
        return queue

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def subscribe(self, execution_id):
        return self

    def is_running(self, execution_id: str) -> bool:
        return False


def _looping_chat_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="New message")],
        stream=True,
        use_workflow="flow",
    )


def _setup_workflow_service(monkeypatch) -> _StubBackground:
    state = WorkflowExecutionState.new("exec-1", "flow")
    state.context["_workflow_loop"] = True
    engine = _StubEngine(state)
    background = _StubBackground([(0, "data: submitted\n\n", False)])

    class _StubService:
        workflow_engine = engine
        workflow_background = background

    monkeypatch.setattr("app.tgi.routes.tgi_service", _StubService())
    return background


@pytest.mark.asyncio
async def test_chat_span_records_user_id_and_execution_id(monkeypatch, fake_tracer):
    background = _setup_workflow_service(monkeypatch)
    token = jwt.encode(
        {"sub": "user-42"},
        key="test-secret-key-long-enough-for-hs256",
        algorithm="HS256",
    )
    request = _make_request(
        {
            "Accept": "text/event-stream",
            "x-inxm-workflow-background": "true",
            "X-Auth-Request-Access-Token": token,
        }
    )
    chat_request = _looping_chat_request()

    chunks = [
        chunk
        async for chunk in _handle_chat_completion(
            request, chat_request, None, None, None, None
        )
    ]

    assert background.started is True
    assert "[DONE]" in chunks[-1]

    attributes = fake_tracer.span.attributes
    assert attributes.get("user.id") == "user-42"
    assert chat_request.workflow_execution_id
    assert attributes.get("execution_id") == chat_request.workflow_execution_id


@pytest.mark.asyncio
async def test_chat_span_omits_user_id_when_token_undecodable(monkeypatch, fake_tracer):
    background = _setup_workflow_service(monkeypatch)
    request = _make_request(
        {
            "Accept": "text/event-stream",
            "x-inxm-workflow-background": "true",
            "X-Auth-Request-Access-Token": "not-a-valid-jwt",
        }
    )
    chat_request = _looping_chat_request()

    chunks = [
        chunk
        async for chunk in _handle_chat_completion(
            request, chat_request, None, None, None, None
        )
    ]

    assert background.started is True
    assert "[DONE]" in chunks[-1]

    attributes = fake_tracer.span.attributes
    assert "user.id" not in attributes
    assert attributes.get("execution_id") == chat_request.workflow_execution_id
