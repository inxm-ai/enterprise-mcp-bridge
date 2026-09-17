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
import time
from typing import Any, Awaitable, Callable, Optional

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from mcp import types
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.lowlevel.server import ServerRequestContext
from mcp.shared.exceptions import MCPError

from app.elicitation import ElicitationRequiredError, get_elicitation_coordinator
from app.oauth.decorator import decorate_args_with_oauth_token
from app.oauth.user_info import (
    CallerNotAuthorizedError,
    ensure_caller_in_required_groups,
)
from app.session import mcp_session
from app.session_manager.session_context import (
    ResponseTooLargeError,
    ToolPolicyDeniedError,
    _to_tool_list,
    enforce_response_ceiling,
    ensure_tool_allowed,
    filter_tools,
    inject_headers_into_args,
    list_resources as _list_resources_helper,
)
from app.utils import mcp_fields
from app.utils.mcp_operation import (
    MCP_METHOD_PROMPTS_GET,
    MCP_METHOD_RESOURCES_READ,
    MCP_METHOD_TOOLS_CALL,
    TRANSPORT_SSE,
    classify_error_result,
    downstream_call_kwargs,
    log_sanitized_exception,
    mcp_operation_span,
    safe_arg_keys,
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


def _request_telemetry_context(ctx: Optional[ServerRequestContext]) -> dict:
    """Best-effort request/client identifiers from the low-level MCP context.

    Missing context is omitted, never fabricated; nothing here reads request
    arguments or results.
    """
    info: dict = {
        "request_id": None,
        "protocol_version": None,
        "client_name": None,
        "client_version": None,
    }
    if ctx is None:
        return info
    request_id = getattr(ctx, "request_id", None)
    if request_id is not None:
        info["request_id"] = str(request_id)
    protocol_version = getattr(ctx, "protocol_version", None)
    client_params = getattr(getattr(ctx, "session", None), "client_params", None)
    if client_params is not None:
        protocol_version = protocol_version or mcp_fields.read(
            client_params, "protocol_version", "protocolVersion"
        )
        client_info = mcp_fields.read(client_params, "client_info", "clientInfo")
        if client_info is not None:
            info["client_name"] = getattr(client_info, "name", None)
            info["client_version"] = getattr(client_info, "version", None)
    if protocol_version:
        info["protocol_version"] = str(protocol_version)
    return info


def _build_proxy_handlers(
    downstream,
    access_token: Optional[str],
    incoming_headers: Optional[dict[str, str]],
    session_key: Optional[str],
) -> dict[str, Callable[..., Awaitable[Any]]]:
    """The proxy's request handlers, keyed by the ``Server`` keyword each one wires into.

    Handlers take the low-level server's ``(ctx, params)``; ``_build_proxy_server``
    passes the dict straight to ``Server``. Kept apart so a test can call one.
    """
    coordinator = get_elicitation_coordinator()

    async def list_tools(ctx, params) -> types.ListToolsResult:
        tools = await downstream.list_tools()
        return types.ListToolsResult(tools=filter_tools(_to_tool_list(tools)))

    async def call_tool(
        ctx, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        # The failures that mean "the tool ran, or was refused" leave as tool
        # errors, the same set the REST route maps to statuses. Anything else
        # is a bridge fault: it propagates, and the SDK answers with a generic
        # internal error rather than the exception's message.
        try:
            return await _call_tool(ctx, params)
        except (ToolPolicyDeniedError, ResponseTooLargeError, MCPError) as exc:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(exc))],
                is_error=True,
            )

    async def _call_tool(ctx, params: types.CallToolRequestParams):
        name = params.name
        arguments = params.arguments or {}
        with mcp_operation_span(
            method=MCP_METHOD_TOOLS_CALL,
            target=name,
            transport=TRANSPORT_SSE,
            session_value=session_key,
            access_token=access_token,
            arg_keys=safe_arg_keys(arguments),
            **_request_telemetry_context(ctx),
        ) as op:
            # Discovery filtering alone is bypassable by a direct call, so the
            # same policy check runs before any downstream contact.
            ensure_tool_allowed(name)
            tools = await downstream.list_tools()
            args = await decorate_args_with_oauth_token(
                tools, name, arguments, access_token
            )
            args = inject_headers_into_args(tools, name, args, incoming_headers)
            for _ in range(3):
                try:
                    result = await downstream.call_tool(
                        name,
                        args,
                        **downstream_call_kwargs(downstream.call_tool),
                    )
                    result = enforce_response_ceiling(result)
                    if mcp_fields.is_error(result):
                        op.record_error_result(classify_error_result(result), result)
                    else:
                        op.record_success(result)
                    return result
                except ElicitationRequiredError as exc:
                    if not session_key:
                        raise
                    session = getattr(ctx, "session", None)
                    if session is None:
                        raise
                    client_result = await session.elicit(
                        message=str(exc.payload.get("message") or ""),
                        requested_schema=exc.payload.get("requestedSchema") or {},
                    )
                    coordinator.submit_response(
                        session_key,
                        {
                            "action": client_result.action,
                            "content": client_result.content,
                        },
                    )
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text="Elicitation retry limit exceeded for proxied tool call",
                    )
                ],
                is_error=True,
            )

    async def list_prompts(ctx, params) -> types.ListPromptsResult:
        result = await downstream.list_prompts()
        prompts = list(result.prompts) if hasattr(result, "prompts") else list(result)
        return types.ListPromptsResult(prompts=prompts)

    async def get_prompt(
        ctx, params: types.GetPromptRequestParams
    ) -> types.GetPromptResult:
        with mcp_operation_span(
            method=MCP_METHOD_PROMPTS_GET,
            target=params.name,
            transport=TRANSPORT_SSE,
            session_value=session_key,
            access_token=access_token,
            arg_keys=safe_arg_keys(params.arguments),
            **_request_telemetry_context(ctx),
        ) as op:
            result = enforce_response_ceiling(
                await downstream.get_prompt(params.name, params.arguments)
            )
            op.record_success(result)
            return result

    async def list_resources(ctx, params) -> types.ListResourcesResult:
        result = await _list_resources_helper(downstream.list_resources)
        resources = (
            list(result.resources) if hasattr(result, "resources") else list(result)
        )
        return types.ListResourcesResult(resources=resources)

    async def read_resource(
        ctx, params: types.ReadResourceRequestParams
    ) -> types.ReadResourceResult:
        with mcp_operation_span(
            method=MCP_METHOD_RESOURCES_READ,
            transport=TRANSPORT_SSE,
            session_value=session_key,
            access_token=access_token,
            **_request_telemetry_context(ctx),
        ) as op:
            result = enforce_response_ceiling(
                await downstream.read_resource(params.uri)
            )
            op.record_success(result)
            return result

    return {
        "on_list_tools": list_tools,
        "on_call_tool": call_tool,
        "on_list_prompts": list_prompts,
        "on_get_prompt": get_prompt,
        "on_list_resources": list_resources,
        "on_read_resource": read_resource,
    }


def _build_proxy_server(
    downstream,
    access_token: Optional[str],
    incoming_headers: Optional[dict[str, str]],
    session_key: Optional[str],
) -> Server:
    """Create an MCP Server whose handlers proxy to *downstream*."""
    handlers = _build_proxy_handlers(
        downstream, access_token, incoming_headers, session_key
    )
    return Server(SERVICE_NAME, **handlers)


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

        # Bridge-level authorization boundary (BRIDGE_REQUIRED_GROUPS): reject
        # before the SSE stream opens so the client gets a proper HTTP status.
        try:
            ensure_caller_in_required_groups(access_token)
        except CallerNotAuthorizedError as exc:
            logger.warning(f"[MCP-SSE] Connection rejected: {exc.detail}")
            resp = Response(exc.detail, status_code=exc.status_code)
            await resp(scope, receive, send)
            return

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
            session_key = f"sse-proxy:{time.time_ns()}"
            async with sse_transport.connect_sse(scope, receive, send) as (
                read_stream,
                write_stream,
            ):
                async with mcp_session(
                    access_token=access_token,
                    requested_group=group,
                    incoming_headers=incoming_headers,
                    session_key=session_key,
                ) as downstream:
                    proxy = _build_proxy_server(
                        downstream, access_token, incoming_headers, session_key
                    )
                    await proxy.run(
                        read_stream,
                        write_stream,
                        proxy.create_initialization_options(),
                    )
        except Exception as exc:
            # Session errors can wrap downstream content; keep them sanitized.
            log_sanitized_exception(logger, "[MCP-SSE]", exc)


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
