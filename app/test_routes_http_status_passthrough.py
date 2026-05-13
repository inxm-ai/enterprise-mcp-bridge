from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from fastapi import HTTPException

from app.routes import list_resources, list_tools, run_tool


def _upstream_401_error() -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://mcp.atlassian.com/v1/sse")
    response = httpx.Response(status_code=401, request=request)
    return httpx.HTTPStatusError(
        "Client error '401 Unauthorized' for url 'https://mcp.atlassian.com/v1/sse'",
        request=request,
        response=response,
    )


class _FailingMCPContext:
    async def __aenter__(self):
        raise _upstream_401_error()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _WrappedClientErrorMCPContext:
    def __init__(self, message: str):
        self._message = message

    async def __aenter__(self):
        raise RuntimeError(self._message)

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_list_resources_passes_through_upstream_http_status_code():
    with patch("app.routes.mcp_session_context", return_value=_FailingMCPContext()):
        with pytest.raises(HTTPException) as exc_info:
            await list_resources(
                request=SimpleNamespace(headers={}),
                access_token="token",
                x_inxm_mcp_session_header=None,
                x_inxm_mcp_session_cookie=None,
                group=None,
            )

    assert exc_info.value.status_code == 401
    assert "401 Unauthorized" in str(exc_info.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("upstream_message", "expected_status"),
    [
        (
            "future: <Task finished exception=Exception('429 Too Many Requests: rate limit exceeded')>",
            429,
        ),
        (
            "future: <Task finished exception=Exception('404 Not Found')>",
            404,
        ),
    ],
)
async def test_run_tool_passes_through_wrapped_upstream_client_error_status_code(
    upstream_message: str, expected_status: int
):
    with patch(
        "app.routes.mcp_session_context",
        return_value=_WrappedClientErrorMCPContext(upstream_message),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await run_tool(
                tool_name="search",
                request=SimpleNamespace(headers={}),
                x_inxm_mcp_session_header=None,
                x_inxm_mcp_session_cookie=None,
                x_inxm_dry_run=None,
                access_token="token",
                args={},
                group=None,
            )

    assert exc_info.value.status_code == expected_status
    assert str(expected_status) in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_list_tools_passes_through_upstream_http_status_code():
    with patch("app.routes.mcp_session_context", return_value=_FailingMCPContext()):
        with pytest.raises(HTTPException) as exc_info:
            await list_tools(
                request=SimpleNamespace(headers={}),
                access_token="token",
                x_inxm_mcp_session_header=None,
                x_inxm_mcp_session_cookie=None,
                group=None,
            )

    assert exc_info.value.status_code == 401
    assert "401 Unauthorized" in str(exc_info.value.detail)
