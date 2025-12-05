import os
import types
import fcntl
import pytest
from unittest.mock import AsyncMock

import app.session_manager.session_context as sc


def dict_to_obj(d):
    filled = {
        k: d.get(k, None)
        for k in [
            "id",
            "name",
            "title",
            "description",
            "inputSchema",
            "outputSchema",
            "annotations",
            "meta",
        ]
    }
    filled.update(d)
    if not filled.get("id"):
        filled["id"] = d.get("id") or d.get("name")
    return types.SimpleNamespace(**filled)


@pytest.mark.asyncio
async def test_sessionless_call_tool_injects_headers(monkeypatch):
    # Prepare fake tools: toolA expects userId
    tools_list = [
        dict_to_obj(
            {
                "name": "toolA",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "userId": {"type": "string"},
                        "keep": {"type": "string"},
                    },
                },
            }
        )
    ]

    class _AsyncMCPContext:
        def __init__(self, tools):
            self._tools = types.SimpleNamespace(tools=tools)

        async def __aenter__(self):
            obj = types.SimpleNamespace()
            obj.list_tools = AsyncMock(return_value=self._tools)

            async def _call_tool(name, args):
                item = types.SimpleNamespace(
                    text="ok", structuredContent={"args": args}
                )
                return types.SimpleNamespace(
                    content=[item], structuredContent={"args": args}
                )

            obj.call_tool = AsyncMock(side_effect=_call_tool)
            return obj

        async def __aexit__(self, exc_type, exc, tb):
            return False

    # Patch mcp_session used inside session_context
    monkeypatch.setattr(sc, "mcp_session", lambda *a, **k: _AsyncMCPContext(tools_list))

    monkeypatch.setattr(
        sc,
        "decorate_args_with_oauth_token",
        AsyncMock(side_effect=lambda *a, **k: a[2]),
    )

    # Set mapping: userId comes from header x-auth-user-id
    monkeypatch.setattr(sc, "MCP_MAP_HEADER_TO_INPUT", {"userId": "x-auth-user-id"})

    incoming_headers = {"X-Auth-User-Id": "user-123"}

    async with sc.mcp_session_context(
        sessions=None,
        x_inxm_mcp_session=None,
        access_token=None,
        group=None,
        incoming_headers=incoming_headers,
    ) as delegate:
        tools = await delegate.list_tools()
        assert isinstance(tools, list)

        # call_tool should inject the header value for userId
        result = await delegate.call_tool("toolA", {"keep": "value"}, None)
        assert hasattr(result, "structuredContent")
        assert result.structuredContent["args"]["userId"] == "user-123"
        assert result.structuredContent["args"]["keep"] == "value"


@pytest.mark.asyncio
async def test_cached_mapped_tools_reads_and_writes(monkeypatch, tmp_path):
    cache_file = tmp_path / "tools.json"
    lock_file = tmp_path / "tools.json.lock"
    monkeypatch.setattr(sc, "TOOLS_CACHE_FILE", cache_file)
    monkeypatch.setattr(sc, "TOOLS_CACHE_LOCK_FILE", lock_file)
    monkeypatch.setattr(sc, "TOOLS_CACHE_ENABLED", True)

    tools_namespace = types.SimpleNamespace(
        tools=[dict_to_obj({"name": "toolA", "description": "first run"})]
    )
    call_count = {"count": 0}

    async def fetch():
        call_count["count"] += 1
        return tools_namespace

    first = await sc._cached_mapped_tools(fetch)
    assert cache_file.exists()
    assert call_count["count"] == 1
    assert first[0]["name"] == "toolA"

    tools_namespace.tools[0].description = "updated"
    second = await sc._cached_mapped_tools(fetch)
    assert call_count["count"] == 1  # served from cache, no re-fetch
    assert second == first  # cache contents should be returned


@pytest.mark.asyncio
async def test_cached_mapped_tools_skips_when_locked(monkeypatch, tmp_path):
    cache_file = tmp_path / "tools.json"
    lock_file = tmp_path / "tools.json.lock"
    monkeypatch.setattr(sc, "TOOLS_CACHE_FILE", cache_file)
    monkeypatch.setattr(sc, "TOOLS_CACHE_LOCK_FILE", lock_file)
    monkeypatch.setattr(sc, "TOOLS_CACHE_ENABLED", True)

    tools_namespace = types.SimpleNamespace(
        tools=[dict_to_obj({"name": "toolB", "description": "locked"})]
    )

    async def fetch():
        return tools_namespace

    lock_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        result = await sc._cached_mapped_tools(fetch)
        assert not cache_file.exists()
        assert result[0]["name"] == "toolB"
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    # Once the lock is free, the tools should be cached
    result_after_lock = await sc._cached_mapped_tools(fetch)
    assert cache_file.exists()
    assert result_after_lock[0]["name"] == "toolB"


@pytest.mark.asyncio
async def test_sessionless_progress_and_logs_forwarded(monkeypatch):
    progress_calls: list[tuple[float, float | None, str | None]] = []
    log_calls: list[tuple[str, str, str | None]] = []

    called = {"with_progress": False}

    class _AsyncMCPContext:
        async def __aenter__(self):
            class _Session:
                async def list_tools(self):
                    return types.SimpleNamespace(tools=[])

                async def call_tool_with_progress(
                    self,
                    name: str,
                    args,
                    progress_callback=None,
                    log_callback=None,
                ):
                    called["with_progress"] = True
                    if progress_callback:
                        await progress_callback(1.0, 2.0, "progress")
                    if log_callback:
                        await log_callback("info", "log message", "test_logger")
                    return types.SimpleNamespace(
                        content=[types.SimpleNamespace(text="ok")],
                        structuredContent={"called": True},
                    )

                async def call_tool(self, *_args, **_kwargs):
                    raise AssertionError(
                        "call_tool should not be used when progress API exists"
                    )

            return _Session()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(sc, "mcp_session", lambda *a, **k: _AsyncMCPContext())
    monkeypatch.setattr(
        sc,
        "decorate_args_with_oauth_token",
        AsyncMock(side_effect=lambda *a, **k: a[2]),
    )
    monkeypatch.setattr(
        sc, "inject_headers_into_args", lambda tools, tool, args, headers: args
    )

    async def _progress_cb(progress, total=None, message=None):
        progress_calls.append((progress, total, message))

    async def _log_cb(level, data, logger_name=None):
        log_calls.append((level, data, logger_name))

    async with sc.mcp_session_context(
        sessions=None,
        x_inxm_mcp_session=None,
        access_token=None,
        group=None,
        incoming_headers=None,
    ) as delegate:
        await delegate.call_tool_with_progress(
            "demo_tool",
            {"a": 1},
            None,
            progress_callback=_progress_cb,
            log_callback=_log_cb,
        )

    assert progress_calls == [(1.0, 2.0, "progress")]
    assert log_calls == [("info", "log message", "test_logger")]
    # Ensure call_tool_with_progress path was used
    assert called["with_progress"] is True


@pytest.mark.asyncio
async def test_sessionful_progress_and_logs_forwarded(monkeypatch):
    progress_calls: list[tuple[float, float | None, str | None]] = []
    log_calls: list[tuple[str, str, str | None]] = []

    class _FakeTask:
        def __init__(self):
            self.requests = []

        async def request(self, req):
            self.requests.append(req)
            if req == "list_tools":
                return []
            if isinstance(req, dict) and req.get("action") == "run_tool_with_progress":
                if req.get("progress_callback"):
                    await req["progress_callback"](0.5, None, "halfway")
                if req.get("log_callback"):
                    await req["log_callback"]("info", "log data", "test_logger")
                return types.SimpleNamespace(
                    content=[types.SimpleNamespace(text="ok")],
                    structuredContent={"called": True},
                )
            if isinstance(req, dict) and req.get("action") == "run_tool":
                return types.SimpleNamespace(
                    content=[types.SimpleNamespace(text="fallback")],
                    structuredContent={},
                )
            raise AssertionError(f"Unexpected request: {req}")

    class _FakeManager:
        def __init__(self, task):
            self._task = task

        def get(self, session_id):
            return self._task

        def set(self, session_id, session):
            self._task = session

        def pop(self, session_id, default=None):
            return default

    mcp_task = _FakeTask()
    sessions = _FakeManager(mcp_task)

    monkeypatch.setattr(
        sc,
        "decorate_args_with_oauth_token",
        AsyncMock(side_effect=lambda *a, **k: a[2]),
    )
    monkeypatch.setattr(
        sc, "inject_headers_into_args", lambda tools, tool, args, headers: args
    )

    async def _progress_cb(progress, total=None, message=None):
        progress_calls.append((progress, total, message))

    async def _log_cb(level, data, logger_name=None):
        log_calls.append((level, data, logger_name))

    async with sc.mcp_session_context(
        sessions=sessions,
        x_inxm_mcp_session="sess-1",
        access_token=None,
        group=None,
        incoming_headers=None,
    ) as delegate:
        await delegate.call_tool_with_progress(
            "demo_tool",
            {"b": 2},
            None,
            progress_callback=_progress_cb,
            log_callback=_log_cb,
        )

    assert any(
        isinstance(r, dict) and r.get("action") == "run_tool_with_progress"
        for r in mcp_task.requests
    )
    assert progress_calls == [(0.5, None, "halfway")]
    assert log_calls == [("info", "log data", "test_logger")]
