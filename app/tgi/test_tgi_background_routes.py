import asyncio
import pytest
from starlette.requests import Request

from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.routes import _handle_chat_completion
from app.tgi.routes import cancel_workflow
from app.tgi.workflows.models import WorkflowExecutionState


@pytest.fixture(autouse=True)
def _set_tgi_url(monkeypatch):
    monkeypatch.setenv("TGI_URL", "https://api.test-llm.com/v1")
    monkeypatch.setenv("TGI_TOKEN", "test-token-123")


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
    def __init__(self, state: WorkflowExecutionState, should_raise: bool = False):
        self.state_store = _StubStateStore(state)
        self.should_raise = should_raise
        self.enforced_tokens: list[str] = []
        self.cancelled: list[tuple[str, str | None]] = []

    def _enforce_workflow_owner(self, state: WorkflowExecutionState, user_token: str):
        self.enforced_tokens.append(user_token)
        if self.should_raise:
            raise PermissionError(
                "Workflow execution 'exec-1' belongs to a different user."
            )

    def cancel_execution(self, execution_id: str, reason: str | None = None) -> bool:
        self.cancelled.append((execution_id, reason))
        return True


class _StubBackground:
    def __init__(self, queue_items: list[tuple[int, str, bool]], running: bool = False):
        self.queue_items = queue_items
        self.running = running
        self.started = False

    async def get_or_start(self, execution_id, stream_factory, initial_event_count=0):
        self.started = True
        return None

    async def _queue_context(self):
        queue: asyncio.Queue = asyncio.Queue()
        for item in self.queue_items:
            await queue.put(item)
        await queue.put((0, None, False))
        return queue

    async def __aenter__(self):
        return await self._queue_context()

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def subscribe(self, execution_id):
        return self

    def is_running(self, execution_id: str) -> bool:
        return self.running


class _StubCancelManager:
    def __init__(self, cancel_result: bool = True):
        self.cancel_result = cancel_result
        self.cancelled: list[str] = []

    async def cancel(self, execution_id: str) -> bool:
        self.cancelled.append(execution_id)
        return self.cancel_result


@pytest.mark.asyncio
async def test_background_replays_history_and_dedupes(monkeypatch):
    state = WorkflowExecutionState.new("exec-1", "flow")
    state.events = ["data: event-1\n\n", "data: event-2\n\n"]

    engine = _StubEngine(state)
    background = _StubBackground(
        [
            (0, "data: submitted\n\n", False),
            (1, "data: event-1\n\n", True),
            (2, "data: event-2\n\n", True),
            (3, "data: event-3\n\n", True),
        ]
    )

    class _StubService:
        workflow_engine = engine
        workflow_background = background

    monkeypatch.setattr("app.tgi.routes.tgi_service", _StubService())

    request = _make_request(
        {
            "Accept": "text/event-stream",
            "x-inxm-workflow-background": "true",
            "X-Auth-Request-Access-Token": "token-ok",
        }
    )
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="[continue]")],
        stream=True,
        use_workflow="flow",
        workflow_execution_id="exec-1",
        return_full_state=True,
    )

    chunks = [
        chunk
        async for chunk in _handle_chat_completion(
            request, chat_request, None, None, None, None
        )
    ]

    assert background.started is True
    assert engine.enforced_tokens == ["token-ok"]
    assert "data: event-1" in chunks[0]
    assert "data: event-2" in chunks[1]
    assert "data: submitted" in chunks[2]
    assert "data: event-3" in chunks[3]
    assert "[DONE]" in chunks[-1]
    assert all("event-1" not in chunk for chunk in chunks[2:])
    assert all("event-2" not in chunk for chunk in chunks[2:])


@pytest.mark.asyncio
async def test_background_does_not_replay_history_without_request(monkeypatch):
    state = WorkflowExecutionState.new("exec-1", "flow")
    state.events = ["data: event-1\n\n", "data: event-2\n\n"]

    engine = _StubEngine(state)
    background = _StubBackground(
        [
            (0, "data: submitted\n\n", False),
            (3, "data: event-3\n\n", True),
        ]
    )

    class _StubService:
        workflow_engine = engine
        workflow_background = background

    monkeypatch.setattr("app.tgi.routes.tgi_service", _StubService())

    request = _make_request(
        {
            "Accept": "text/event-stream",
            "x-inxm-workflow-background": "true",
            "X-Auth-Request-Access-Token": "token-ok",
        }
    )
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="[continue]")],
        stream=True,
        use_workflow="flow",
        workflow_execution_id="exec-1",
    )

    chunks = [
        chunk
        async for chunk in _handle_chat_completion(
            request, chat_request, None, None, None, None
        )
    ]

    assert background.started is True
    assert "event-1" not in "".join(chunks)
    assert "event-2" not in "".join(chunks)
    assert "data: submitted" in chunks[0]
    assert "data: event-3" in chunks[1]
    assert "[DONE]" in chunks[-1]


@pytest.mark.asyncio
async def test_background_rejects_message_while_active(monkeypatch):
    state = WorkflowExecutionState.new("exec-1", "flow")
    state.awaiting_feedback = False
    state.completed = False

    engine = _StubEngine(state)
    background = _StubBackground([], running=True)

    class _StubService:
        workflow_engine = engine
        workflow_background = background

    monkeypatch.setattr("app.tgi.routes.tgi_service", _StubService())

    request = _make_request(
        {
            "Accept": "text/event-stream",
            "x-inxm-workflow-background": "true",
            "X-Auth-Request-Access-Token": "token-ok",
        }
    )
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="New message")],
        stream=True,
        use_workflow="flow",
        workflow_execution_id="exec-1",
    )

    chunks = [
        chunk
        async for chunk in _handle_chat_completion(
            request, chat_request, None, None, None, None
        )
    ]

    assert background.started is False
    assert "workflow_conflict" in chunks[0]
    assert "[DONE]" in chunks[-1]


@pytest.mark.asyncio
async def test_background_allows_feedback_while_active(monkeypatch):
    state = WorkflowExecutionState.new("exec-1", "flow")
    state.awaiting_feedback = True
    state.completed = False

    engine = _StubEngine(state)
    background = _StubBackground([(0, "data: submitted\n\n", False)], running=True)

    class _StubService:
        workflow_engine = engine
        workflow_background = background

    monkeypatch.setattr("app.tgi.routes.tgi_service", _StubService())

    request = _make_request(
        {
            "Accept": "text/event-stream",
            "x-inxm-workflow-background": "true",
            "X-Auth-Request-Access-Token": "token-ok",
        }
    )
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Proceed")],
        stream=True,
        use_workflow="flow",
        workflow_execution_id="exec-1",
    )

    chunks = [
        chunk
        async for chunk in _handle_chat_completion(
            request, chat_request, None, None, None, None
        )
    ]

    assert background.started is True
    assert "workflow_conflict" not in "".join(chunks)
    assert "[DONE]" in chunks[-1]


@pytest.mark.asyncio
async def test_background_denies_other_user(monkeypatch):
    state = WorkflowExecutionState.new("exec-1", "flow")
    state.events = ["data: event-1\n\n"]

    engine = _StubEngine(state, should_raise=True)
    background = _StubBackground([])

    class _StubService:
        workflow_engine = engine
        workflow_background = background

    monkeypatch.setattr("app.tgi.routes.tgi_service", _StubService())

    request = _make_request(
        {
            "Accept": "text/event-stream",
            "x-inxm-workflow-background": "true",
            "X-Auth-Request-Access-Token": "token-bad",
        }
    )
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Hi")],
        stream=True,
        use_workflow="flow",
        workflow_execution_id="exec-1",
    )

    chunks = [
        chunk
        async for chunk in _handle_chat_completion(
            request, chat_request, None, None, None, None
        )
    ]

    assert background.started is False
    assert "event-1" not in "".join(chunks)
    assert "access_denied" in chunks[0]
    assert "[DONE]" in chunks[-1]


@pytest.mark.asyncio
async def test_cancel_workflow_background(monkeypatch):
    state = WorkflowExecutionState.new("exec-1", "flow")
    engine = _StubEngine(state)
    manager = _StubCancelManager(cancel_result=True)

    class _StubService:
        workflow_engine = engine
        workflow_background = manager

    monkeypatch.setattr("app.tgi.routes.tgi_service", _StubService())

    request = _make_request(
        {
            "X-Auth-Request-Access-Token": "token-ok",
        }
    )

    response = await cancel_workflow("exec-1", request, access_token=None)
    assert response.status_code == 200
    assert manager.cancelled == ["exec-1"]
    assert engine.cancelled == [("exec-1", "Cancelled by request")]
