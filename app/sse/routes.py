"""
SSE streaming routes for MCP tool calls with progress reporting.

This module provides REST endpoints that stream progress updates from MCP tool
executions using Server-Sent Events (SSE).
"""

import logging
import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, Cookie, Query, Request, HTTPException

from app.elicitation import get_elicitation_coordinator
from app.oauth.token_dependency import get_access_token
from app.oauth.token_exchange import UserLoggedOutException
from app.session import try_get_session_id, session_id
from app.session_manager import mcp_session_context, session_manager
from app.utils.exception_logging import (
    find_exception_in_exception_groups,
    log_exception_with_details,
)
from app.vars import SESSION_FIELD_NAME
from app.sse import stream_tool_call, create_sse_response
from opentelemetry import trace

router = APIRouter()
sessions = session_manager()

logger = logging.getLogger("uvicorn.error")
tracer = trace.get_tracer(__name__)
_USER_FEEDBACK_KEY_RE = re.compile(r"^_?user_feedback$", re.IGNORECASE)


def _extract_request_headers(request: Request) -> dict[str, str]:
    """Extract headers from the incoming request as a dictionary."""
    return dict(request.headers)


def _pop_user_feedback(args: Optional[dict]) -> tuple[Optional[dict], Optional[str]]:
    if not isinstance(args, dict):
        return args, None
    out = dict(args)
    feedback = None
    for key in list(out.keys()):
        if _USER_FEEDBACK_KEY_RE.match(str(key)):
            feedback = out.pop(key)
            break
    if isinstance(feedback, str):
        feedback = feedback.strip()
    if not feedback:
        feedback = None
    return out, feedback


@router.post("/tools/{tool_name}/stream")
async def run_tool_with_progress(
    tool_name: str,
    request: Request,
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
    access_token: Optional[str] = Depends(get_access_token),
    args: Optional[Dict] = None,
    group: Optional[str] = Query(
        None, description="Group name for sessionless group-specific data access"
    ),
):
    """
    Execute a tool and stream progress updates via Server-Sent Events (SSE).

    This endpoint is similar to POST /tools/{tool_name} but returns a stream
    of SSE events instead of waiting for the complete result. This is useful
    for long-running tools that report progress.

    Event Types:
        - progress: Reports progress updates with progress percentage and optional message
        - log: Reports log messages from the tool execution
        - result: Contains the final tool execution result
        - error: Reports any errors that occurred during execution

    Example with curl:
        curl -N -H "Accept: text/event-stream" \\
             -H "Content-Type: application/json" \\
             -d '{}' \\
             "http://localhost:8000/tools/report_progress_and_logs/stream"

    Example with JavaScript:
        const eventSource = new EventSource('/tools/my_tool/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ arg1: 'value1' })
        });
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(data);
        };

    Args:
        tool_name: Name of the tool to execute
        request: FastAPI request object
        x_inxm_mcp_session_header: Session ID from header
        x_inxm_mcp_session_cookie: Session ID from cookie
        access_token: OAuth access token
        args: Tool arguments
        group: Group name for group-specific data access

    Returns:
        StreamingResponse with SSE events
    """
    try:
        x_inxm_mcp_session = session_id(
            try_get_session_id(
                x_inxm_mcp_session_header,
                x_inxm_mcp_session_cookie,
                args.get("inxm-session", None) if args else None,
            ),
            access_token,
        )
        if args and "inxm-session" in args:
            args = dict(args)
            args.pop("inxm-session")
        args, user_feedback = _pop_user_feedback(args)
        if user_feedback:
            if not x_inxm_mcp_session:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "feedback_requires_session",
                        "detail": "User feedback resume requires a persistent MCP session id.",
                    },
                )
            coordinator = get_elicitation_coordinator()
            if not coordinator.submit_feedback(x_inxm_mcp_session, user_feedback):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "feedback_not_expected",
                        "detail": "No pending elicitation found for this session.",
                    },
                )

        incoming_headers = _extract_request_headers(request)

        # Create a generator that will stream the tool execution
        async def generate_events():
            try:
                async with mcp_session_context(
                    sessions,
                    x_inxm_mcp_session,
                    access_token,
                    group,
                    incoming_headers,
                ) as session:
                    # Create a streaming tool call function
                    async def call_tool_with_callbacks(
                        name: str,
                        tool_args: Dict[str, Any],
                        token: Optional[str],
                        progress_callback,
                        log_callback,
                    ):
                        return await session.call_tool_with_progress(
                            name,
                            tool_args,
                            token,
                            progress_callback=progress_callback,
                            log_callback=log_callback,
                        )

                    # Stream the tool call events
                    async for event in stream_tool_call(
                        call_tool_with_callbacks,
                        tool_name,
                        args,
                        access_token,
                    ):
                        yield event

            except HTTPException as e:
                from app.sse import SSEEvent

                yield SSEEvent.error_event(
                    e.detail if hasattr(e, "detail") else str(e)
                ).to_sse_string()
            except UserLoggedOutException as e:
                from app.sse import SSEEvent

                yield SSEEvent.error_event(e.message).to_sse_string()
            except Exception as e:
                log_exception_with_details(logger, "[SSE-Tool-Call]", e)
                from app.sse import SSEEvent

                yield SSEEvent.error_event(str(e)).to_sse_string()

        logger.info(
            f"[SSE-Tool-Call] Starting streaming tool call: {tool_name}, "
            f"Session: {x_inxm_mcp_session}, Group: {group}, Args: {args}"
        )

        return create_sse_response(generate_events())

    except HTTPException as e:
        raise e
    except UserLoggedOutException as e:
        logger.warning(f"[SSE-Tool-Call] Unauthorized access: {str(e)}")
        raise HTTPException(status_code=401, detail=e.message)
    except Exception as e:
        log_exception_with_details(logger, "[SSE-Tool-Call]", e)
        child_http_exception = find_exception_in_exception_groups(e, HTTPException)
        if child_http_exception:
            raise child_http_exception
        raise HTTPException(status_code=500, detail="Internal server error")
