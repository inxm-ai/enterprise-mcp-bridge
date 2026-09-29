import asyncio
import jwt
import logging
import os
from datetime import datetime, timezone
from typing import Optional, Any, AsyncGenerator
from fastapi import (
    APIRouter,
    HTTPException,
    Header,
    Cookie,
    Query,
    Request,
    Depends,
)
from fastapi.responses import StreamingResponse, JSONResponse
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
from app.elicitation import (
    ElicitationRequiredError,
    InvalidUserFeedbackError,
    UnsupportedElicitationSchemaError,
    get_elicitation_coordinator,
    parse_user_feedback_tag,
)

from app.tgi.models import ChatCompletionRequest, MessageRole
from app.tgi.workflows.models import WorkflowExecutionState
from app.tgi.services.proxied_tgi_service import ProxiedTGIService
from app.tgi.protocols.chunk_reader import (
    chunk_reader,
    accumulate_content,
)
from app.vars import SESSION_FIELD_NAME, TOKEN_NAME
from app.oauth.token_dependency import get_access_token

# Initialize components
router = APIRouter(prefix="/tgi/v1")
sessions = session_manager()
tgi_service = ProxiedTGIService()
tracer = trace.get_tracer(__name__)
logger = logging.getLogger("uvicorn.error")



# --- Helper Functions ---
def _resolve_user_token(
    incoming_headers: dict[str, str], access_token: Optional[str]
) -> Optional[str]:
    header_name = TOKEN_NAME.lower()
    return incoming_headers.get(header_name) or access_token


def _resolve_user_id_for_tracing(user_token: Optional[str]) -> Optional[str]:
    """Best-effort caller id for telemetry. Never breaks the chat request."""
    if not user_token:
        return None
    try:
        payload = jwt.decode(
            user_token,
            options={"verify_signature": False},
            algorithms=["RS256", "HS256"],
        )
        return payload.get("sub")
    except Exception as exc:
        logger.debug(f"[TGI] Could not resolve user.id for tracing: {exc}")
        return None


def _header_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _select_group_exception(
    exc_group: BaseExceptionGroup, expected_type: type[BaseException]
) -> BaseException:
    exceptions = getattr(exc_group, "exceptions", None) or []
    for exc in exceptions:
        if isinstance(exc, expected_type):
            return exc
    return exceptions[0] if exceptions else exc_group


def _permission_status_code(exc: PermissionError) -> int:
    message = str(exc).lower()
    if (
        "invalid access token" in message
        or "access token required" in message
        or "user identifier" in message
    ):
        return 401
    return 403


def _permission_error_payload(exc: PermissionError) -> dict[str, Any]:
    status_code = _permission_status_code(exc)
    return {
        "error": "unauthorized" if status_code == 401 else "access_denied",
        "detail": str(exc),
        "status": status_code,
    }


def _workflow_conflict_payload(detail: str, status: int = 409) -> dict[str, Any]:
    return {"error": "workflow_conflict", "detail": detail, "status": status}


def _extract_last_user_message(
    messages: Optional[list],
) -> Optional[str]:
    if not messages:
        return None
    for message in reversed(messages):
        if getattr(message, "role", None) == MessageRole.USER:
            return getattr(message, "content", None)
    return None


def _maybe_submit_pending_user_feedback(
    session_key: Optional[str], user_message: Optional[str]
) -> None:
    if not session_key or not user_message:
        return
    parsed = parse_user_feedback_tag(user_message)
    if not parsed:
        return
    coordinator = get_elicitation_coordinator()
    coordinator.submit_feedback(session_key, parsed)


def _is_continue_placeholder(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.strip().lower() == "[continue]"


def _is_looping_workflow_state(
    state: Optional[WorkflowExecutionState], engine: Optional[Any]
) -> bool:
    if not state:
        return False
    if isinstance(state.context, dict) and state.context.get("_workflow_loop"):
        return True
    if engine:
        repo = getattr(engine, "repository", None)
        if repo:
            try:
                return bool(repo.get(state.flow_id).loop)
            except Exception:
                return False
    return False


def _extract_completion_content(result: Any) -> str:
    if result is None:
        return ""

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        choices = result.get("choices") or []
        if choices:
            choice = choices[0] or {}
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            content = delta.get("content") or message.get("content")
            if content is not None:
                return str(content)

        if "result" in result:
            payload = result.get("result")
            if isinstance(payload, dict):
                completion = payload.get("completion")
                if completion is not None:
                    return str(completion)
            if isinstance(payload, str):
                return payload

        return json.dumps(result, ensure_ascii=False)

    if isinstance(result, list):
        parts = []
        for item in result:
            piece = _extract_completion_content(item)
            if piece:
                parts.append(piece)
        return "".join(parts)

    return str(result)


def _workflow_status(state: WorkflowExecutionState) -> str:
    return state.status()


def _serialize_workflow(state: WorkflowExecutionState) -> dict[str, Any]:
    description = None
    if isinstance(state.context, dict):
        description = state.context.get("_workflow_description") or state.context.get(
            "workflow_description"
        )
    return {
        "execution_id": state.execution_id,
        "workflow_id": state.flow_id,
        "status": _workflow_status(state),
        "awaiting_feedback": bool(state.awaiting_feedback),
        "current_agent": state.current_agent,
        "created_at": state.created_at,
        "last_change": state.last_change,
        "description": description,
    }


def _parse_timestamp(value: str) -> str:
    """
    Parse an ISO-8601 timestamp string and normalize it to UTC with a Z suffix.
    """
    if not value:
        raise ValueError("Timestamp is required")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (
        dt.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )



# --- Core Logic Abstraction ---


async def _handle_chat_completion(
    request: Request,
    chat_request: ChatCompletionRequest,
    access_token: Optional[str],
    x_inxm_mcp_session: Optional[str],
    group: Optional[str],
    prompt: Optional[str],
    incoming_headers: Optional[dict[str, str]] = None,
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

    incoming_headers = dict(request.headers)
    user_token = _resolve_user_token(incoming_headers, access_token)
    user_token = _resolve_user_token(incoming_headers, access_token)
    user_token = _resolve_user_token(incoming_headers, access_token)
    user_token = _resolve_user_token(incoming_headers, access_token)
    # Check if streaming is requested
    accept_header = request.headers.get("accept", "")
    chat_request.stream = chat_request.stream or "text/event-stream" in accept_header
    is_streaming = chat_request.stream
    background_requested = _header_truthy(
        request.headers.get("x-inxm-workflow-background")
    )
    _maybe_submit_pending_user_feedback(
        x_inxm_mcp_session, _extract_last_user_message(chat_request.messages)
    )

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
        user_id=_resolve_user_id_for_tracing(user_token),
    ) as span:
        done_sent = False
        try:
            if (
                background_requested
                and is_streaming
                and chat_request.use_workflow
                and tgi_service.workflow_background
            ):
                if not chat_request.workflow_execution_id:
                    chat_request.workflow_execution_id = str(uuid4())
                execution_id = chat_request.workflow_execution_id
                span.set_attribute("execution_id", execution_id)

                copier = getattr(chat_request, "model_copy", None)
                background_request = (
                    copier(deep=True)
                    if callable(copier)
                    else chat_request.copy(deep=True)
                )
                background_request.return_full_state = False
                background_request.stream = True

                engine = tgi_service.workflow_engine
                existing_state = (
                    engine.state_store.load_execution(execution_id)
                    if engine and execution_id
                    else None
                )
                if existing_state and engine:
                    engine._enforce_workflow_owner(existing_state, user_token or "")
                is_looping = _is_looping_workflow_state(existing_state, engine)
                user_message = _extract_last_user_message(chat_request.messages)
                has_new_message = bool(
                    user_message and not _is_continue_placeholder(user_message)
                )
                if execution_id and has_new_message:
                    if existing_state and existing_state.completed and not is_looping:
                        detail = (
                            f"Workflow execution '{execution_id}' has completed; "
                            "start a new workflow execution to continue."
                        )
                        error_payload = _workflow_conflict_payload(detail)
                        yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
                        return
                    if existing_state and existing_state.awaiting_feedback:
                        # Allow feedback even if a background task is still winding down.
                        pass
                    else:
                        is_running = (
                            tgi_service.workflow_background.is_running(execution_id)
                            if tgi_service.workflow_background
                            else False
                        )
                        if is_running:
                            detail = (
                                f"Workflow execution '{execution_id}' is still active; "
                                "wait for it to request feedback before sending a new message."
                            )
                            error_payload = _workflow_conflict_payload(detail)
                            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
                            return
                    if (
                        existing_state
                        and not existing_state.awaiting_feedback
                        and not is_looping
                    ):
                        detail = (
                            f"Workflow execution '{execution_id}' is not awaiting feedback; "
                            "wait for it to request feedback before sending a new message."
                        )
                        error_payload = _workflow_conflict_payload(detail)
                        yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
                        return
                if not (existing_state and existing_state.completed):
                    initial_count = len(existing_state.events) if existing_state else 0

                    async def _stream_factory():
                        async def _stream():
                            async with mcp_session_context(
                                sessions,
                                x_inxm_mcp_session,
                                access_token,
                                group,
                                incoming_headers,
                            ) as session:
                                result = await tgi_service.chat_completion(
                                    session,
                                    background_request,
                                    user_token,
                                    access_token,
                                    prompt,
                                )
                                if hasattr(result, "__aiter__"):
                                    async for chunk in result:
                                        yield chunk
                                else:
                                    if isinstance(result, str):
                                        yield result
                                    else:
                                        yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"

                        return _stream()

                    await tgi_service.workflow_background.get_or_start(
                        execution_id, _stream_factory, initial_count
                    )

                async with tgi_service.workflow_background.subscribe(
                    execution_id
                ) as queue:
                    state = (
                        engine.state_store.load_execution(execution_id)
                        if engine and execution_id
                        else existing_state
                    )
                    history_events = (
                        list(state.events)
                        if state and chat_request.return_full_state
                        else []
                    )
                    skip_index = len(history_events) if history_events else 0

                    for event in history_events:
                        if isinstance(event, str) and "[DONE]" in event:
                            done_sent = True
                        yield event

                    if queue is not None:
                        while True:
                            idx, chunk, recorded = await queue.get()
                            if chunk is None:
                                break
                            if recorded and skip_index and idx <= skip_index:
                                continue
                            if "[DONE]" in chunk:
                                done_sent = True
                            yield chunk
            else:
                async with mcp_session_context(
                    sessions,
                    x_inxm_mcp_session,
                    access_token,
                    group,
                    incoming_headers,
                ) as session:
                    result = await tgi_service.chat_completion(
                        session, chat_request, user_token, access_token, prompt
                    )

                    # If the service returned an async-iterable (stream), forward chunks
                    # while the mcp_session_context is still active. This ensures
                    # any cancel scopes, ContextVars and session state created by the
                    # session context remain valid for the lifetime of the stream.
                    if hasattr(result, "__aiter__"):
                        async for chunk in result:
                            if isinstance(chunk, str):
                                if "[DONE]" in chunk:
                                    done_sent = True
                                yield chunk
                            else:
                                yield chunk
                    else:
                        # Non-streaming dict result: yield once inside the context
                        async def _single():
                            yield result

                        async for chunk in _single():
                            yield chunk
        except* PermissionError as exc_group:
            permission_exc = _select_group_exception(exc_group, PermissionError)
            logger.warning(f"[TGI] Workflow access denied: {permission_exc}")
            if is_streaming:
                error_payload = _permission_error_payload(permission_exc)
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
            else:
                raise permission_exc
        except* asyncio.CancelledError:
            logger.info("[TGI] Stream cancelled (client disconnect or timeout)")
            done_sent = True
        except* ElicitationRequiredError as exc_group:
            elicitation_exc = _select_group_exception(
                exc_group, ElicitationRequiredError
            )
            if is_streaming:
                yield (
                    "data: "
                    + json.dumps(
                        elicitation_exc.to_client_payload(), ensure_ascii=False
                    )
                    + "\n\n"
                )
            else:
                raise elicitation_exc
        except* InvalidUserFeedbackError as exc_group:
            feedback_exc = _select_group_exception(exc_group, InvalidUserFeedbackError)
            if is_streaming:
                payload = {
                    "error": "invalid_feedback",
                    "detail": str(feedback_exc),
                    "elicitation": feedback_exc.payload,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            else:
                raise feedback_exc
        except* UnsupportedElicitationSchemaError as exc_group:
            schema_exc = _select_group_exception(
                exc_group, UnsupportedElicitationSchemaError
            )
            if is_streaming:
                payload = {
                    "error": "unsupported_elicitation_schema",
                    "detail": str(schema_exc),
                    "elicitation": schema_exc.payload,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            else:
                raise schema_exc
        except* Exception as exc_group:
            first_exc = _select_group_exception(exc_group, Exception)
            logger.error(f"[TGI] Streaming chat error: {first_exc}", exc_info=True)
            if is_streaming:
                error_payload = {"error": "internal_error", "detail": str(first_exc)}
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
            else:
                raise first_exc
        finally:
            if is_streaming and not done_sent:
                yield "data: [DONE]\n\n"


def _is_async_iterable(obj: Any) -> bool:
    return hasattr(obj, "__aiter__")


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    chat_request: ChatCompletionRequest,
    access_token: Optional[str] = Depends(get_access_token),
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
    incoming_headers = dict(request.headers)
    user_token = _resolve_user_token(incoming_headers, access_token)
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
                    incoming_headers,
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
                sessions,
                x_inxm_mcp_session,
                access_token,
                group,
                incoming_headers,
            ) as session:
                result = await tgi_service.chat_completion(
                    session, chat_request, user_token, access_token, prompt
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
    except ElicitationRequiredError as e:
        raise HTTPException(status_code=409, detail=e.to_client_payload())
    except InvalidUserFeedbackError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_feedback",
                "detail": str(e),
                "elicitation": e.payload,
            },
        )
    except UnsupportedElicitationSchemaError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_elicitation_schema",
                "detail": str(e),
                "elicitation": e.payload,
            },
        )
    except PermissionError as e:
        status_code = _permission_status_code(e)
        detail = str(e)
        logger.warning(f"[TGI] Access denied ({status_code}): {detail}")
        raise HTTPException(status_code=status_code, detail=detail)
    except UserLoggedOutException as e:
        logger.warning(f"[TGI] Unauthorized access: {str(e)}")
        raise HTTPException(status_code=401, detail=e.message)
    except Exception as e:
        log_exception_with_details(logger, "[TGI]", e)
        child_http_exception = find_exception_in_exception_groups(e, HTTPException)
        if child_http_exception:
            raise child_http_exception
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/workflows/{execution_id}")
async def cancel_workflow(
    execution_id: str,
    request: Request,
    access_token: Optional[str] = Depends(get_access_token),
):
    """
    Cancel a background workflow execution by id.
    """
    engine = tgi_service.workflow_engine
    manager = tgi_service.workflow_background
    if not engine or not manager:
        raise HTTPException(status_code=404, detail="Workflow engine not available")

    incoming_headers = dict(request.headers)
    user_token = _resolve_user_token(incoming_headers, access_token)
    state = engine.state_store.load_execution(execution_id)
    if not state:
        raise HTTPException(status_code=404, detail="Workflow execution not found")

    try:
        engine._enforce_workflow_owner(state, user_token or "")
    except PermissionError as exc:
        status_code = _permission_status_code(exc)
        raise HTTPException(status_code=status_code, detail=str(exc))

    cancelled = await manager.cancel(execution_id)
    if cancelled:
        engine.cancel_execution(execution_id, reason="Cancelled by request")
        return JSONResponse(
            content={"status": "cancelled", "execution_id": execution_id}
        )

    return JSONResponse(content={"status": "not_running", "execution_id": execution_id})


@router.get("/workflows")
async def list_workflows(
    request: Request,
    access_token: Optional[str] = Depends(get_access_token),
    limit: int = Query(20, ge=1, le=100),
    before: Optional[str] = Query(None),
    before_id: Optional[str] = Query(None),
    after: Optional[str] = Query(None),
    after_id: Optional[str] = Query(None),
):
    """
    List workflow executions for the current user, ordered by created_at desc.
    """
    engine = tgi_service.workflow_engine
    if not engine:
        raise HTTPException(status_code=404, detail="Workflow engine not available")

    incoming_headers = dict(request.headers)
    user_token = _resolve_user_token(incoming_headers, access_token)
    if not user_token:
        raise HTTPException(
            status_code=401, detail="Access token required to list workflows."
        )

    if before_id and not before:
        raise HTTPException(
            status_code=400,
            detail="before_id requires a before timestamp.",
        )
    if after_id and not after:
        raise HTTPException(
            status_code=400,
            detail="after_id requires an after timestamp.",
        )

    parsed_before = None
    if before:
        try:
            parsed_before = _parse_timestamp(before)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    parsed_after = None
    if after:
        try:
            parsed_after = _parse_timestamp(after)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        workflows = engine.list_workflows(
            user_token,
            limit=limit,
            before=parsed_before,
            before_id=before_id,
            after=parsed_after,
            after_id=after_id,
        )
    except PermissionError as exc:
        status_code = _permission_status_code(exc)
        raise HTTPException(status_code=status_code, detail=str(exc))

    return JSONResponse(
        content={"workflows": [_serialize_workflow(state) for state in workflows]}
    )
