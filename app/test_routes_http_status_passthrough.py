from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from fastapi import HTTPException

from app.routes import list_resources, list_tools


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
