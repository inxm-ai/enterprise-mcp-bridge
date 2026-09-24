"""Idle-session expiry and sessionless mode (MCP_SESSION_IDLE_TIMEOUT_SECONDS, MCP_SESSIONLESS)."""

import types

import pytest
from fastapi.testclient import TestClient

import app.routes as routes
import app.session_manager.session_context as sc
from app import vars as app_vars
from app.server import app as fastapi_app
from app.session_manager.session_manager import InMemorySessionManager


class _FakeClock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class _StoppableTask:
    def __init__(self):
        self.stopped = False

    async def stop(self):
        self.stopped = True


def test_pop_idle_returns_only_sessions_unused_past_the_limit():
    clock = _FakeClock()
    sessions = InMemorySessionManager(clock=clock)
    active, idle = _StoppableTask(), _StoppableTask()
    sessions.set("active", active)
    sessions.set("idle", idle)

    clock.now += 50
    assert sessions.get("active") is active  # a use refreshes the timestamp
    clock.now += 60

    assert sessions.pop_idle(100) == [("idle", idle)]
    assert sessions.get("idle") is None
    assert sessions.get("active") is active


def test_pop_idle_forgets_closed_sessions():
    clock = _FakeClock()
    sessions = InMemorySessionManager(clock=clock)
    sessions.set("closed", _StoppableTask())
    sessions.pop("closed")
    clock.now += 1_000

    assert sessions.pop_idle(1) == []


@pytest.mark.asyncio
async def test_close_idle_sessions_stops_expired_tasks(monkeypatch):
    monkeypatch.setattr(app_vars, "MCP_SESSION_IDLE_TIMEOUT_SECONDS", 30.0)
    clock = _FakeClock()
    sessions = InMemorySessionManager(clock=clock)
    task = _StoppableTask()
    sessions.set("s1", task)
    clock.now += 31

    await sc.close_idle_sessions(sessions)

    assert task.stopped
    assert sessions.get("s1") is None


@pytest.mark.asyncio
async def test_close_idle_sessions_is_off_by_default(monkeypatch):
    monkeypatch.setattr(app_vars, "MCP_SESSION_IDLE_TIMEOUT_SECONDS", 0.0)
    clock = _FakeClock()
    sessions = InMemorySessionManager(clock=clock)
    task = _StoppableTask()
    sessions.set("s1", task)
    clock.now += 10_000

    await sc.close_idle_sessions(sessions)

    assert not task.stopped
    assert sessions.get("s1") is task


@pytest.mark.asyncio
async def test_sessionless_mode_ignores_a_session_id(monkeypatch):
    monkeypatch.setattr(app_vars, "MCP_SESSIONLESS", True)
    opened = []

    class _Transient:
        async def __aenter__(self):
            opened.append(True)
            return types.SimpleNamespace(
                list_tools=lambda: _async(types.SimpleNamespace(tools=[]))
            )

        async def __aexit__(self, *exc):
            return False

    async def _async(value):
        return value

    class _NoLookups:
        def get(self, session_id):
            raise AssertionError("sessionless mode must not look up sessions")

    monkeypatch.setattr(sc, "mcp_session", lambda *a, **k: _Transient())

    async with sc.mcp_session_context(
        sessions=_NoLookups(),
        x_inxm_mcp_session="stale-session-id",
        access_token=None,
        group=None,
    ) as delegate:
        assert await delegate.list_tools() == []

    assert opened == [True]


@pytest.fixture
def session_routes(monkeypatch):
    sessions = InMemorySessionManager()
    started = []

    class _Task:
        def __init__(self, strategy):
            self.strategy = strategy

        def start(self):
            started.append(self)

        async def stop(self):
            return None

    monkeypatch.setattr(routes, "sessions", sessions)
    monkeypatch.setattr(routes, "MCPLocalSessionTask", _Task)
    monkeypatch.setattr(routes, "build_mcp_client_strategy", lambda **kwargs: object())
    fastapi_app.dependency_overrides[routes.get_access_token] = lambda: None
    yield types.SimpleNamespace(client=TestClient(fastapi_app), sessions=sessions, started=started)
    fastapi_app.dependency_overrides.pop(routes.get_access_token, None)


def _session_url(path: str) -> str:
    return f"{app_vars.MCP_BASE_PATH}{path}"


def test_session_start_spawns_nothing_when_sessionless(monkeypatch, session_routes):
    monkeypatch.setattr(app_vars, "MCP_SESSIONLESS", True)

    started = session_routes.client.post(_session_url("/session/start"))
    session_value = started.json()[app_vars.SESSION_FIELD_NAME]
    closed = session_routes.client.post(
        _session_url("/session/close"),
        headers={app_vars.SESSION_FIELD_NAME: session_value},
    )

    assert started.status_code == 200
    assert session_routes.started == []
    assert closed.status_code == 200
    assert closed.json() == {"status": "closed"}


def test_session_start_still_spawns_a_task_by_default(monkeypatch, session_routes):
    monkeypatch.setattr(app_vars, "MCP_SESSIONLESS", False)

    started = session_routes.client.post(_session_url("/session/start"))

    assert started.status_code == 200
    assert len(session_routes.started) == 1
