import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Header, Cookie, Query, Request
from fastapi.responses import StreamingResponse
from opentelemetry import trace

from app.utils.traced_requests import traced_request
from app.session import try_get_session_id, session_id
from app.session_manager import mcp_session_context, session_manager
from app.oauth.token_exchange import UserLoggedOutException
from app.utils.exception_logging import (
    find_exception_in_exception_groups,
    log_exception_with_details,
)

from app.tgi.models import (
    ChatCompletionRequest,
)
from app.tgi.proxied_tgi_service import ProxiedTGIService

import os

# Configuration
TOKEN_NAME = os.environ.get("TOKEN_NAME", "X-Auth-Request-Access-Token")
SESSION_FIELD_NAME = os.environ.get("SESSION_FIELD_NAME", "x-inxm-mcp-session")

# Initialize components
router = APIRouter(prefix="/tgi/v1")
sessions = session_manager()
tgi_service = ProxiedTGIService()
tracer = trace.get_tracer(__name__)
logger = logging.getLogger("uvicorn.error")


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    chat_request: ChatCompletionRequest,
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
    prompt: Optional[str] = Query(None, description="Specific prompt name to use"),
    group: Optional[str] = Query(
        None, description="Group name for sessionless group-specific data access"
    ),
):
    """
    OpenAI-compatible chat completions endpoint with MCP integration.

    This endpoint:
    1. Finds and applies MCP prompts based on the 'prompt' query parameter or system role
    2. Filters/retrieves tools from the MCP server
    3. Executes tool calls through MCP
    4. Supports streaming responses
    5. Provides full OpenTelemetry tracing
    """

    if not os.environ.get("TGI_URL", None):
        logger.warning("[TGI] TGI_URL not set")
        raise HTTPException(
            status_code=400,
            detail="Environment variable TGI_URL not configured. This is a prerequisite for this endpoint to work.",
        )

    try:
        # Extract session information
        x_inxm_mcp_session = session_id(
            try_get_session_id(x_inxm_mcp_session_header, x_inxm_mcp_session_cookie),
            access_token,
        )

        # Check if streaming is requested
        accept_header = request.headers.get("accept", "")
        is_streaming = chat_request.stream or "text/event-stream" in accept_header

        with traced_request(
            tracer=tracer,
            operation="chat_completions",
            session_value=x_inxm_mcp_session,
            group=group,
            start_message=f"[TGI] Chat completion request. Stream: {is_streaming}, Messages: {len(chat_request.messages)}, Tools: {len(chat_request.tools) if chat_request.tools else 0}",
            extra_attrs={
                "chat.streaming": is_streaming,
                "chat.messages_count": len(chat_request.messages),
                "chat.tools_count": (
                    len(chat_request.tools) if chat_request.tools else 0
                ),
                "chat.model": chat_request.model,
                "chat.prompt_requested": prompt or "",
            },
        ):
            async with mcp_session_context(
                sessions, x_inxm_mcp_session, access_token, group
            ) as session:

                if is_streaming:
                    # Use the proxied service for streaming
                    return StreamingResponse(
                        tgi_service.chat_completion(
                            session, chat_request, access_token, prompt
                        ),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                        },
                    )
                else:
                    # Use the proxied service for non-streaming
                    return await tgi_service.chat_completion(
                        session, chat_request, access_token, prompt
                    )

    except HTTPException as e:
        raise e
    except UserLoggedOutException as e:
        logger.warning(f"[TGI] Unauthorized access: {str(e)}")
        raise HTTPException(status_code=401, detail=e.message)
    except Exception as e:
        log_exception_with_details(logger, "[TGI]", e)
        child_http_exception = find_exception_in_exception_groups(e, HTTPException)
        if child_http_exception:
            raise child_http_exception
        raise HTTPException(status_code=500, detail="Internal server error")
