"""MCP-compliant SSE proxy endpoint.

Exposes the enterprise-mcp-bridge as a standard MCP SSE server.
Any MCP client can connect to ``GET /sse`` and communicate using the
standard MCP SSE transport protocol.  All requests are proxied to the
configured downstream MCP server with OAuth2 token exchange, tool
filtering, and header mapping applied.

Endpoints
---------
GET  {base}/sse              — Establish SSE connection
POST {base}/sse/messages     — JSON-RPC message channel
"""

import logging
from typing import Optional

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from mcp import types
from mcp.server import Server
from mcp.server.sse import SseServerTransport

from app.oauth.decorator import decorate_args_with_oauth_token
from app.session import mcp_session
from app.session_manager.session_context import (
    _to_tool_list,
    filter_tools,
    inject_headers_into_args,
    list_resources as _list_resources_helper,
)
from app.vars import (
    MCP_BASE_PATH,
    SERVICE_NAME,
    TOKEN_COOKIE_NAME,
    TOKEN_NAME,
    TOKEN_SOURCE,
)

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Transport setup
# ---------------------------------------------------------------------------


def _message_endpoint_path() -> str:
    """Full relative path for the SSE messages POST endpoint."""
    base = (MCP_BASE_PATH or "").rstrip("/")
    return f"{base}/sse/messages"


# Single transport instance – manages per-connection session IDs internally
sse_transport = SseServerTransport(_message_endpoint_path())


# ---------------------------------------------------------------------------
# Token / query helpers
# ---------------------------------------------------------------------------


def _extract_access_token(request: Request) -> Optional[str]:
    """Extract access token from request headers or cookies."""
    if TOKEN_SOURCE == "cookie":
        return request.cookies.get(TOKEN_COOKIE_NAME)
    return request.headers.get(TOKEN_NAME)


# ---------------------------------------------------------------------------
# Proxy MCP Server builder
# ---------------------------------------------------------------------------


def _build_proxy_server(
    downstream,
    access_token: Optional[str],
    incoming_headers: Optional[dict[str, str]],
) -> Server:
    """Create an MCP Server whose handlers proxy to *downstream*."""
    proxy = Server(SERVICE_NAME)

    @proxy.list_tools()
    async def _list_tools() -> list[types.Tool]:
        tools = await downstream.list_tools()
        return filter_tools(_to_tool_list(tools))

    @proxy.call_tool(validate_input=False)
    async def _call_tool(name: str, arguments: dict) -> types.CallToolResult:
        tools = await downstream.list_tools()
        args = await decorate_args_with_oauth_token(
            tools, name, arguments, access_token
        )
        args = inject_headers_into_args(tools, name, args, incoming_headers)
        return await downstream.call_tool(name, args)

    @proxy.list_prompts()
    async def _list_prompts() -> list[types.Prompt]:
        result = await downstream.list_prompts()
        return list(result.prompts) if hasattr(result, "prompts") else list(result)

    @proxy.get_prompt()
    async def _get_prompt(
        name: str, arguments: dict[str, str] | None
    ) -> types.GetPromptResult:
        return await downstream.get_prompt(name, arguments)

    @proxy.list_resources()
    async def _list_resources() -> list[types.Resource]:
        result = await _list_resources_helper(downstream.list_resources)
        return list(result.resources) if hasattr(result, "resources") else list(result)

    # read_resource – register handler directly for clean result pass-through
    async def _read_resource_handler(req: types.ReadResourceRequest):
        result = await downstream.read_resource(req.params.uri)
        return types.ServerResult(result)

    proxy.request_handlers[types.ReadResourceRequest] = _read_resource_handler

    return proxy


# ---------------------------------------------------------------------------
# ASGI apps  (callable classes so Starlette uses them as raw ASGI apps
# instead of wrapping them with request_response)
# ---------------------------------------------------------------------------


class _SSEConnectionApp:
    """``GET /sse`` – establish SSE connection and run proxy MCP server."""

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return

        request = Request(scope, receive, send)
        access_token = _extract_access_token(request)
        group = request.query_params.get("group")
        incoming_headers = dict(request.headers)

        # Validate group access *before* opening the SSE stream so we can
        # return a proper HTTP error response.
        if group and access_token:
            from app.oauth.user_info import get_data_access_manager

            data_manager = get_data_access_manager()
            try:
                data_manager.resolve_data_resource(access_token, group)
            except PermissionError as exc:
                logger.warning(f"[MCP-SSE] Group access denied: {exc}")
                resp = Response(f"Access denied: {exc}", status_code=403)
                await resp(scope, receive, send)
                return
            except AssertionError as exc:
                logger.error(f"[MCP-SSE] Group access assertion error: {exc}")
                resp = Response("Invalid group or token", status_code=400)
                await resp(scope, receive, send)
                return

        logger.info(f"[MCP-SSE] New SSE connection. Group: {group}")

        try:
            async with sse_transport.connect_sse(scope, receive, send) as (
                read_stream,
                write_stream,
            ):
                async with mcp_session(
                    access_token=access_token,
                    requested_group=group,
                    incoming_headers=incoming_headers,
                ) as downstream:
                    proxy = _build_proxy_server(
                        downstream, access_token, incoming_headers
                    )
                    await proxy.run(
                        read_stream,
                        write_stream,
                        proxy.create_initialization_options(),
                    )
        except Exception as exc:
            logger.error(f"[MCP-SSE] SSE session error: {exc}")


class _SSEMessagesApp:
    """``POST /sse/messages`` – JSON-RPC message channel."""

    async def __call__(self, scope, receive, send):
        await sse_transport.handle_post_message(scope, receive, send)


# Singleton instances
_sse_connection_app = _SSEConnectionApp()
_sse_messages_app = _SSEMessagesApp()


# ---------------------------------------------------------------------------
# Route factory
# ---------------------------------------------------------------------------


def get_sse_proxy_routes() -> list[Route]:
    """Return Starlette routes for the MCP SSE proxy.

    These must be added to the FastAPI app's route list directly
    (not via an APIRouter) because they use raw ASGI apps.
    """
    base = (MCP_BASE_PATH or "").rstrip("/")
    return [
        Route(f"{base}/sse", endpoint=_sse_connection_app),
        Route(f"{base}/sse/messages", endpoint=_sse_messages_app, methods=["POST"]),
    ]
