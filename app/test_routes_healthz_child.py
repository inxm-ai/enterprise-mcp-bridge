from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.routes import healthz_child


@pytest.mark.asyncio
async def test_healthz_child_reports_tool_count_when_child_works():
    async def list_tools():
        return SimpleNamespace(tools=[object(), object(), object()])

    @asynccontextmanager
    async def working_session(**kwargs):
        yield SimpleNamespace(list_tools=list_tools)

    with patch("app.routes.mcp_session", working_session):
        result = await healthz_child()

    assert result == {"status": "ok", "tools": 3}


@pytest.mark.asyncio
async def test_healthz_child_returns_503_when_child_fails_to_start():
    @asynccontextmanager
    async def broken_session(**kwargs):
        raise RuntimeError(
            "ImportError: cannot import name 'streamablehttp_client' (mcp==2.0.0)"
        )
        yield  # pragma: no cover

    with patch("app.routes.mcp_session", broken_session):
        with pytest.raises(HTTPException) as exc_info:
            await healthz_child()

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_healthz_child_returns_503_when_list_tools_fails():
    async def list_tools():
        raise RuntimeError("child session opened but tools are broken")

    @asynccontextmanager
    async def session_with_broken_tools(**kwargs):
        yield SimpleNamespace(list_tools=list_tools)

    with patch("app.routes.mcp_session", session_with_broken_tools):
        with pytest.raises(HTTPException) as exc_info:
            await healthz_child()

    assert exc_info.value.status_code == 503
