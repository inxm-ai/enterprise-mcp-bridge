import logging
import os
from typing import Optional, Any, AsyncGenerator
from fastapi import APIRouter, HTTPException, Header, Cookie, Query, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, ValidationError
from opentelemetry import trace
import json
from uuid import uuid4

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
from app.tgi.services.proxied_tgi_service import ProxiedTGIService
from app.tgi.protocols.chunk_reader import chunk_reader, ChunkFormat, accumulate_content
from app.vars import DEFAULT_MODEL, TOKEN_NAME, SESSION_FIELD_NAME, SERVICE_NAME

# Initialize components
router = APIRouter(prefix="/tgi/v1")
sessions = session_manager()
tgi_service = ProxiedTGIService()
tracer = trace.get_tracer(__name__)
logger = logging.getLogger("uvicorn.error")


class A2AParams(BaseModel):
    prompt: str


class A2ARequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: A2AParams
    id: str


class A2AResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Any] = None
    id: str


# --- Helper Functions ---
def _create_a2a_response(result: Any, request_id: str) -> A2AResponse:
    """Creates a JSON-RPC 2.0 compliant success response."""
    return A2AResponse(result={"completion": result}, id=request_id)


def _create_a2a_error_response(code: int, message: str, request_id: str) -> A2AResponse:
    """Creates a JSON-RPC 2.0 compliant error response."""
    return A2AResponse(error={"code": code, "message": message}, id=request_id)


def _extract_request_id(raw_body: Any) -> str:
    """Extracts a best-effort request id from an incoming payload."""
    if isinstance(raw_body, dict):
        for key in ("id", "request_id", "requestId", "tool_call_id", "toolCallId"):
            value = raw_body.get(key)
            if value is not None:
                return str(value)
    return "unknown"


def _coerce_a2a_request(raw_body: Any) -> A2ARequest:
    """Coerces various payload shapes into an A2ARequest."""
    if not isinstance(raw_body, dict):
        raise ValueError("Request body must be a JSON object")

    if "jsonrpc" in raw_body or {"method", "params", "id"}.issubset(raw_body.keys()):
        return A2ARequest.model_validate(raw_body)

    # Some clients send params but omit jsonrpc/id. Respect their data when possible.
    if "params" in raw_body and isinstance(raw_body["params"], dict):
        params_dict = raw_body["params"]
        prompt = params_dict.get("prompt")
        if prompt:
            request_id = raw_body.get("id") or str(uuid4())
            method = raw_body.get("method") or SERVICE_NAME
            return A2ARequest(
                method=str(method),
                params=A2AParams(prompt=prompt),
                id=str(request_id),
                jsonrpc=str(raw_body.get("jsonrpc", "2.0")),
            )

    # Minimal payload (prompt at top level)
    prompt = raw_body.get("prompt")
    if prompt:
        request_id = raw_body.get("id") or raw_body.get("request_id") or str(uuid4())
        method = raw_body.get("method") or SERVICE_NAME
        return A2ARequest(
            method=str(method),
            params=A2AParams(prompt=prompt),
            id=str(request_id),
            jsonrpc=str(raw_body.get("jsonrpc", "2.0")),
        )

    raise ValueError("Invalid A2A request payload: missing 'prompt'")


# --- Core Logic Abstraction ---


async def _handle_chat_completion(
    request: Request,
    chat_request: ChatCompletionRequest,
    access_token: Optional[str],
    x_inxm_mcp_session: Optional[str],
    group: Optional[str],
    prompt: Optional[str],
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Handles the core logic for chat completions.
    Always returns an async generator.
    """
    if not os.environ.get("TGI_URL", None):
        logger.warning("[TGI] TGI_URL not set")
        raise HTTPException(
            status_code=400,
            detail="Environment variable TGI_URL not configured. This is a prerequisite for this endpoint to work.",
        )

    # Check if streaming is requested
    accept_header = request.headers.get("accept", "")
    chat_request.stream = chat_request.stream or "text/event-stream" in accept_header
    is_streaming = chat_request.stream

    with traced_request(
        tracer=tracer,
        operation="chat_completions",
        session_value=x_inxm_mcp_session,
        group=group,
        start_message=f"[TGI] Chat completion request. Stream: {is_streaming}, Messages: {len(chat_request.messages)}, Tools: {len(chat_request.tools) if chat_request.tools else 0}",
        extra_attrs={
            "chat.streaming": is_streaming,
            "chat.messages_count": len(chat_request.messages),
            "chat.tools_count": (len(chat_request.tools) if chat_request.tools else 0),
            "chat.tool_choice": chat_request.tool_choice or "",
            "chat.model": chat_request.model,
            "chat.prompt_requested": prompt or "",
        },
    ):
        async with mcp_session_context(
            sessions, x_inxm_mcp_session, access_token, group
        ) as session:
            result = await tgi_service.chat_completion(
                session, chat_request, access_token, prompt
            )

            # If the service returned an async-iterable (stream), forward chunks
            # while the mcp_session_context is still active. This ensures
            # any cancel scopes, ContextVars and session state created by the
            # session context remain valid for the lifetime of the stream.
            if hasattr(result, "__aiter__"):
                async for chunk in result:
                    yield chunk
            else:
                # Non-streaming dict result: yield once inside the context
                async def _single():
                    yield result

                async for chunk in _single():
                    yield chunk


def _is_async_iterable(obj: Any) -> bool:
    return hasattr(obj, "__aiter__")


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
    """
    try:
        # Validate required environment
        if not os.environ.get("TGI_URL", None):
            logger.warning("[TGI] TGI_URL not set")
            raise HTTPException(
                status_code=400,
                detail="Environment variable TGI_URL not configured. This is a prerequisite for this endpoint to work.",
            )

        x_inxm_mcp_session = session_id(
            try_get_session_id(x_inxm_mcp_session_header, x_inxm_mcp_session_cookie),
            access_token,
        )

        # Check if streaming is requested
        accept_header = request.headers.get("accept", "")
        chat_request.stream = (
            chat_request.stream or "text/event-stream" in accept_header
        )
        is_streaming = chat_request.stream

        if is_streaming:
            return StreamingResponse(
                _handle_chat_completion(
                    request,
                    chat_request,
                    access_token,
                    x_inxm_mcp_session,
                    group,
                    prompt,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Transfer-Encoding": "chunked",
                },
            )
        else:
            # Non-streaming: call the service and return JSON as-is when possible
            async with mcp_session_context(
                sessions, x_inxm_mcp_session, access_token, group
            ) as session:
                result = await tgi_service.chat_completion(
                    session, chat_request, access_token, prompt
                )

                if _is_async_iterable(result):
                    # Rare case: service streamed despite non-stream request.
                    # Use chunk_reader to accumulate content cleanly.
                    full_content = await accumulate_content(result)  # type: ignore[arg-type]

                    return JSONResponse(
                        content={"choices": [{"message": {"content": full_content}}]}
                    )
                else:
                    # Dict result path: passthrough
                    return JSONResponse(content=result)

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


@router.post("/a2a")
async def a2a_chat_completion(
    request: Request,
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
    prompt: Optional[str] = Query(None, description="Specific prompt name to use"),
    group: Optional[str] = Query(
        None, description="Group name for sessionless group-specific data access"
    ),
):
    """
    A2A-compliant endpoint that maps an A2A JSON-RPC request to an internal
    OpenAI-compatible chat completion call.
    """
    raw_body: Any = None
    parse_error: Optional[str] = None
    try:
        raw_body = await request.json()
    except json.JSONDecodeError as exc:
        parse_error = f"Invalid JSON payload: {exc.msg}"
    except Exception as exc:  # pragma: no cover - defensive, should be rare
        parse_error = f"Unable to read request body: {str(exc)}"

    request_id = _extract_request_id(raw_body)

    if parse_error:
        return JSONResponse(
            content=_create_a2a_error_response(
                code=-32600,
                message=f"Invalid Request: {parse_error}",
                request_id=request_id,
            ).model_dump()
        )

    try:
        a2a_request = _coerce_a2a_request(raw_body)
    except (ValidationError, ValueError) as exc:
        message = str(exc)
        return JSONResponse(
            content=_create_a2a_error_response(
                code=-32600,
                message=f"Invalid Request: {message}",
                request_id=request_id,
            ).model_dump()
        )

    stream_requested = False
    if isinstance(raw_body, dict):
        stream_requested = bool(
            raw_body.get("stream") or (raw_body.get("params") or {}).get("stream")
        )

    try:
        if a2a_request.method != SERVICE_NAME:
            return JSONResponse(
                content=_create_a2a_error_response(
                    code=-32601,
                    message=f"Method not found. Expected method '{SERVICE_NAME}'.",
                    request_id=a2a_request.id,
                ).model_dump()
            )

        model = DEFAULT_MODEL
        if not model:
            return JSONResponse(
                content=_create_a2a_error_response(
                    code=-32600,
                    message="No default model configured, cannot start agent.",
                    request_id=a2a_request.id,
                ).model_dump()
            )

        # Map A2A prompt to an OpenAI chat request
        chat_request = ChatCompletionRequest(
            messages=[{"role": "user", "content": a2a_request.params.prompt}],
            model=model,
            stream=stream_requested,
        )

        x_inxm_mcp_session = session_id(
            try_get_session_id(x_inxm_mcp_session_header, x_inxm_mcp_session_cookie),
            access_token,
        )

        accept_header = request.headers.get("accept", "")
        is_streaming = chat_request.stream or "text/event-stream" in accept_header

        if is_streaming:
            # We must set the streaming flag for the _handle_chat_completion function
            chat_request.stream = True

            async def a2a_streaming_response():
                stream_gen = _handle_chat_completion(
                    request,
                    chat_request,
                    access_token,
                    x_inxm_mcp_session,
                    group,
                    prompt,
                )
                # Iterate the ChunkReader without using its async context manager
                # so we can control when the underlying async generator is
                # closed and ensure it happens in this same task. Setting
                # _entered=True allows using the reader's async iterators.
                reader = chunk_reader(stream_gen)
                reader._entered = True
                try:
                    async for chunk in reader.as_json(ChunkFormat.A2A, request_id=a2a_request.id):
                        yield chunk
                finally:
                    # Close the underlying stream generator in this task and
                    # suppress any GeneratorExit/RuntimeError that can occur
                    # from cross-task cancel scopes in some test environments.
                    try:
                        if hasattr(stream_gen, "aclose"):
                            await stream_gen.aclose()
                    except BaseException as e:
                        # Catch BaseException (including ExceptionGroup) here because
                        # closing an async generator during context teardown can
                        # raise exception groups originating from other task/cancel
                        # scope interactions. These are expected in some test
                        # environments and safe to ignore for stream cleanup.
                        logger.debug(f"Ignoring base error closing stream_gen: {e}")

            return StreamingResponse(
                a2a_streaming_response(),
                media_type=None,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Transfer-Encoding": "chunked",
                },
            )
        else:
            chat_request.stream = False
            stream_gen = _handle_chat_completion(
                request,
                chat_request,
                access_token,
                x_inxm_mcp_session,
                group,
                prompt,
            )
            full_response = await accumulate_content(stream_gen)

            return JSONResponse(
                content=_create_a2a_response(
                    result=full_response, request_id=a2a_request.id
                ).model_dump()
            )

    except (HTTPException, ValidationError) as e:
        error_detail = e.detail if isinstance(e, HTTPException) else str(e)
        return JSONResponse(
            content=_create_a2a_error_response(
                code=-32600,
                message=f"Invalid Request: {error_detail}",
                request_id=a2a_request.id,
            ).model_dump()
        )
    except UserLoggedOutException as e:
        return JSONResponse(
            content=_create_a2a_error_response(
                code=-32000, message=e.message, request_id=a2a_request.id
            ).model_dump()
        )
    except Exception as e:
        log_exception_with_details(logger, "[A2A]", e)
        child_http_exception = find_exception_in_exception_groups(e, HTTPException)
        if child_http_exception:
            return JSONResponse(
                content=_create_a2a_error_response(
                    code=child_http_exception.status_code,
                    message=child_http_exception.detail,
                    request_id=a2a_request.id,
                ).model_dump()
            )
        return JSONResponse(
            content=_create_a2a_error_response(
                code=-32000, message="Internal server error", request_id=a2a_request.id
            ).model_dump()
        )
