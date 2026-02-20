import logging
import os
import re
from typing import Any, AsyncIterator, Dict, Optional
from urllib.parse import urljoin, urlparse
from http.cookies import SimpleCookie
import json

import httpx
from fastapi import APIRouter, Cookie, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from opentelemetry import trace

from app.vars import (
    MCP_BASE_PATH,
    SESSION_FIELD_NAME,
    TOKEN_NAME,
    APP_CONVERSATIONAL_UI_ENABLED,
)
from app.session import session_id, try_get_session_id
from app.session_manager import mcp_session_context, session_manager
from app.oauth.user_info import get_data_access_manager

from .generated_service import (
    Actor,
    GeneratedUIService,
    GeneratedUIStorage,
    Scope,
    validate_identifier,
)
from .schemas import (
    UiChatMessageRequest,
    UiChatSessionCreateRequest,
    UiCreateRequest,
    UiPublishResponse,
    UiUpdateRequest,
)

# Initialize components
router = APIRouter(prefix="/app")
tracer = trace.get_tracer(__name__)
logger = logging.getLogger("uvicorn.error")
sessions = session_manager()
_generated_service: Optional[GeneratedUIService] = None

# Configuration from environment
PROXY_PREFIX = os.environ.get("PROXY_PREFIX", MCP_BASE_PATH + "/app")
TARGET_SERVER_URL = os.environ.get("TARGET_SERVER_URL", "").rstrip("/")
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").rstrip(
    "/"
)  # Public-facing URL for rewrites
PROXY_TIMEOUT = int(os.environ.get("PROXY_TIMEOUT", "300"))  # 5 minutes default
REWRITE_HTML_URLS = os.environ.get("REWRITE_HTML_URLS", "true").lower() == "true"
REWRITE_JSON_URLS = os.environ.get("REWRITE_JSON_URLS", "true").lower() == "true"
REWRITE_CSS_URLS = os.environ.get("REWRITE_CSS_URLS", "true").lower() == "true"
REWRITE_JS_URLS = os.environ.get("REWRITE_JS_URLS", "true").lower() == "true"

# Hop-by-hop headers that should NOT be forwarded (RFC 2616)
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

# Headers that identify the original request for the backend
FORWARDED_HEADERS = {
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-forwarded-prefix",
    "x-real-ip",
    "x-scheme",
}


def _ensure_tgi_enabled() -> None:
    if not os.environ.get("TGI_URL"):
        raise HTTPException(
            status_code=503,
            detail="Text generation is not available because TGI_URL is not configured.",
        )


def _ensure_conversational_ui_enabled() -> None:
    enabled = (
        os.environ.get(
            "APP_CONVERSATIONAL_UI_ENABLED",
            "true" if APP_CONVERSATIONAL_UI_ENABLED else "false",
        ).lower()
        == "true"
    )
    if not enabled:
        raise HTTPException(
            status_code=404,
            detail="Conversational UI endpoints are disabled.",
        )


def _get_generated_service() -> GeneratedUIService:
    global _generated_service
    base_path = os.environ.get("GENERATED_WEB_PATH", "").strip()
    if not base_path:
        raise HTTPException(
            status_code=503,
            detail="GENERATED_WEB_PATH must be configured to use generated uis.",
        )
    base_path = os.path.abspath(base_path)
    existing_path = (
        getattr(getattr(_generated_service, "storage", None), "base_path", None)
        if _generated_service
        else None
    )
    if _generated_service and existing_path == base_path:
        return _generated_service
    _generated_service = GeneratedUIService(
        storage=GeneratedUIStorage(base_path),
    )
    return _generated_service


def _parse_scope(target: str) -> Scope:
    if not target:
        raise HTTPException(status_code=400, detail="Scope target is required")
    if target.startswith("group="):
        identifier = validate_identifier(target[len("group=") :], "group id")
        return Scope(kind="group", identifier=identifier)
    if target.startswith("user="):
        identifier = validate_identifier(target[len("user=") :], "user id")
        return Scope(kind="user", identifier=identifier)
    raise HTTPException(
        status_code=400,
        detail="Scope target must use 'group=<group_id>' or 'user=<user_id>' format",
    )


def _extract_actor(access_token: Optional[str]) -> Actor:
    if not access_token:
        raise HTTPException(status_code=401, detail="Access token is required")
    data_manager = get_data_access_manager()
    try:
        info = data_manager.user_extractor.extract_user_info(access_token)
    except AssertionError as exc:
        raise HTTPException(status_code=401, detail="Invalid access token") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=401, detail="Failed to decode access token"
        ) from exc

    user_id = info.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=401, detail="No user identifier found in access token"
        )
    groups = info.get("groups") or []
    groups_list = [str(group) for group in groups]
    return Actor(user_id=str(user_id), groups=groups_list)


def _format_ui_response(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = record.get("metadata", {})
    current = record.get("current", {})
    response = {
        "id": metadata.get("id"),
        "name": metadata.get("name"),
        "scope": metadata.get("scope"),
        "created_at": metadata.get("created_at"),
        "updated_at": metadata.get("updated_at"),
        "html": current.get("html"),
        "metadata": current.get("metadata"),
    }
    history = metadata.get("history")
    if history:
        response["history"] = history
    return response


@router.post("/_generated/{target}")
async def create_generated_ui(
    target: str,
    body: UiCreateRequest,
    request: Request,
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
):
    logger.info(
        f"[create_generated_ui] POST /_generated/{target} - ui_id={body.id}, name={body.name}"
    )
    _ensure_tgi_enabled()
    scope = _parse_scope(target)
    ui_id = validate_identifier(body.id, "ui id")
    name = validate_identifier(body.name, "ui name")
    prompt = (body.prompt or "").strip()
    if not prompt:
        logger.warning("[create_generated_ui] Empty prompt provided")
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    actor = _extract_actor(access_token)
    logger.info(
        f"[create_generated_ui] Actor: user_id={actor.user_id}, groups={len(actor.groups)}"
    )
    service = _get_generated_service()
    incoming_headers = dict(request.headers)
    session_key = session_id(
        try_get_session_id(
            x_inxm_mcp_session_header,
            x_inxm_mcp_session_cookie,
        ),
        access_token,
    )
    requested_group = scope.identifier if scope.kind == "group" else None

    async def event_stream() -> AsyncIterator[bytes]:
        logger.info("[create_generated_ui] Starting event stream generator")
        try:
            with tracer.start_as_current_span("generated_ui.create") as span:
                span.set_attribute("ui.scope", scope.kind)
                span.set_attribute("ui.scope_id", scope.identifier)
                span.set_attribute("ui.id", ui_id)
                span.set_attribute("ui.name", name)
                async with mcp_session_context(
                    sessions,
                    session_key,
                    access_token,
                    requested_group,
                    incoming_headers,
                ) as session_obj:
                    logger.info(
                        "[create_generated_ui] MCP session established, calling stream_generate_ui"
                    )
                    # Stream from the service; it's responsible for producing SSE-formatted bytes
                    chunk_count = 0
                    async for chunk in service.stream_generate_ui(
                        session=session_obj,
                        scope=scope,
                        actor=actor,
                        ui_id=ui_id,
                        name=name,
                        prompt=prompt,
                        tools=list(body.tools or []),
                        access_token=access_token,
                    ):  # pragma: no cover - streaming path
                        chunk_count += 1
                        logger.debug(
                            f"[create_generated_ui] Yielding chunk {chunk_count}, size={len(chunk)}"
                        )
                        yield chunk
                    logger.info(
                        f"[create_generated_ui] Finished streaming, total chunks={chunk_count}"
                    )
        except Exception as eg:
            # Some runtimes (py < 3.11) don't define ExceptionGroup/BaseExceptionGroup.
            # Try to extract nested HTTPExceptions if present, otherwise emit a generic error.
            if hasattr(eg, "exceptions"):
                for exc in getattr(eg, "exceptions"):
                    if isinstance(exc, HTTPException):
                        payload = {"error": exc.detail}
                        yield f"event: error\ndata: {json.dumps(payload)}\n\n".encode(
                            "utf-8"
                        )
                        return
            logger.error(
                f"[create_generated_ui] Exception (ExceptionGroup fallthrough): {str(eg)}",
                exc_info=eg,
            )
            yield f"event: error\ndata: {json.dumps({'error': 'Internal Server Error'})}\n\n".encode(
                "utf-8"
            )
            return
        except HTTPException as exc:
            logger.error(
                f"[create_generated_ui] HTTPException in event_stream: {exc.detail}",
                exc_info=exc,
            )
            payload = {"error": exc.detail}
            yield f"event: error\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
            return
        except Exception as exc:
            logger.error(
                f"[create_generated_ui] Unexpected exception in event_stream: {str(exc)}",
                exc_info=exc,
            )
            payload = {"error": "Internal Server Error"}
            yield f"event: error\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
            return

    # Return StreamingResponse for SSE
    logger.info("[create_generated_ui] Returning StreamingResponse")
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        },
        status_code=200,
    )


@router.get("/_generated/{target}/{ui_id}/{name}")
async def get_generated_ui(
    target: str,
    ui_id: str,
    name: str,
    request: Request,
    render_mode: str = Query("page", alias="as"),
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
):
    _ensure_tgi_enabled()
    scope = _parse_scope(target)
    ui_id = validate_identifier(ui_id, "ui id")
    name = validate_identifier(name, "ui name")
    actor = _extract_actor(access_token)
    service = _get_generated_service()

    mode = (render_mode or "card").lower()
    if mode not in {"card", "page", "snippet"}:
        raise HTTPException(status_code=400, detail="Invalid render mode requested")

    # Only expand payload when rendering HTML (not for card/JSON mode)
    expand_payload = mode in {"page", "snippet"}

    with tracer.start_as_current_span("generated_ui.fetch") as span:
        span.set_attribute("ui.scope", scope.kind)
        span.set_attribute("ui.scope_id", scope.identifier)
        span.set_attribute("ui.id", ui_id)
        span.set_attribute("ui.name", name)
        record = service.get_ui(
            scope=scope,
            actor=actor,
            ui_id=ui_id,
            name=name,
            expand=expand_payload,
        )

    if mode == "card":
        return JSONResponse(content=record)

    html_section = (record.get("current") or {}).get("html") or {}
    content = html_section.get(mode)
    if not content:
        raise HTTPException(
            status_code=500,
            detail=f"Stored ui does not contain HTML for mode '{mode}'",
        )
    return HTMLResponse(content=content, media_type="text/html")


@router.post("/_generated/{target}/{ui_id}/{name}")
async def update_generated_ui(
    target: str,
    ui_id: str,
    name: str,
    body: UiUpdateRequest,
    request: Request,
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
):
    _ensure_tgi_enabled()
    scope = _parse_scope(target)
    ui_id = validate_identifier(ui_id, "ui id")
    name = validate_identifier(name, "ui name")
    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    actor = _extract_actor(access_token)
    service = _get_generated_service()
    incoming_headers = dict(request.headers)
    session_key = session_id(
        try_get_session_id(
            x_inxm_mcp_session_header,
            x_inxm_mcp_session_cookie,
        ),
        access_token,
    )
    requested_group = scope.identifier if scope.kind == "group" else None

    async def event_stream() -> AsyncIterator[bytes]:
        logger.info("[update_generated_ui] Starting event stream generator")
        try:
            with tracer.start_as_current_span("generated_ui.update") as span:
                span.set_attribute("ui.scope", scope.kind)
                span.set_attribute("ui.scope_id", scope.identifier)
                span.set_attribute("ui.id", ui_id)
                span.set_attribute("ui.name", name)
                async with mcp_session_context(
                    sessions,
                    session_key,
                    access_token,
                    requested_group,
                    incoming_headers,
                ) as session_obj:
                    logger.info(
                        "[update_generated_ui] MCP session established, calling stream_update_ui"
                    )
                    chunk_count = 0
                    async for chunk in service.stream_update_ui(
                        session=session_obj,
                        scope=scope,
                        actor=actor,
                        ui_id=ui_id,
                        name=name,
                        prompt=prompt,
                        tools=list(body.tools or []),
                        access_token=access_token,
                    ):  # pragma: no cover - streaming path
                        chunk_count += 1
                        logger.debug(
                            f"[update_generated_ui] Yielding chunk {chunk_count}, size={len(chunk)}"
                        )
                        yield chunk
                    logger.info(
                        f"[update_generated_ui] Finished streaming, total chunks={chunk_count}"
                    )
        except Exception as eg:
            if hasattr(eg, "exceptions"):
                for exc in getattr(eg, "exceptions"):
                    if isinstance(exc, HTTPException):
                        payload = {"error": exc.detail}
                        yield f"event: error\ndata: {json.dumps(payload)}\n\n".encode(
                            "utf-8"
                        )
                        return
            logger.error(
                f"[update_generated_ui] Exception (ExceptionGroup fallthrough): {str(eg)}",
                exc_info=eg,
            )
            yield f"event: error\ndata: {json.dumps({'error': 'Internal Server Error'})}\n\n".encode(
                "utf-8"
            )
            return
        except HTTPException as exc:
            logger.error(
                f"[update_generated_ui] HTTPException in event_stream: {exc.detail}",
                exc_info=exc,
            )
            payload = {"error": exc.detail}
            yield f"event: error\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
            return
        except Exception as exc:
            logger.error(
                f"[update_generated_ui] Unexpected exception in event_stream: {str(exc)}",
                exc_info=exc,
            )
            payload = {"error": "Internal Server Error"}
            yield f"event: error\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
            return

    logger.info("[update_generated_ui] Returning StreamingResponse")
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        },
        status_code=200,
    )


@router.delete("/_generated/{target}/{ui_id}/{name}/reset")
async def reset_generated_ui(
    target: str,
    ui_id: str,
    name: str,
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
):
    """
    Reset the last change to a generated UI by removing the most recent history
    entry and restoring the previous state as current.

    This operation requires update permissions and will fail if there is only
    one history entry (the initial creation).
    """
    _ensure_tgi_enabled()
    scope = _parse_scope(target)
    ui_id = validate_identifier(ui_id, "ui id")
    name = validate_identifier(name, "ui name")
    actor = _extract_actor(access_token)
    service = _get_generated_service()

    with tracer.start_as_current_span("generated_ui.reset") as span:
        span.set_attribute("ui.scope", scope.kind)
        span.set_attribute("ui.scope_id", scope.identifier)
        span.set_attribute("ui.id", ui_id)
        span.set_attribute("ui.name", name)

        record = service.reset_last_change(
            scope=scope,
            actor=actor,
            ui_id=ui_id,
            name=name,
        )

    return JSONResponse(status_code=200, content=_format_ui_response(record))


@router.post("/_generated/{target}/{ui_id}/{name}/chat/sessions")
async def create_generated_ui_chat_session(
    target: str,
    ui_id: str,
    name: str,
    body: UiChatSessionCreateRequest,
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
):
    _ensure_conversational_ui_enabled()
    scope = _parse_scope(target)
    ui_id = validate_identifier(ui_id, "ui id")
    name = validate_identifier(name, "ui name")
    actor = _extract_actor(access_token)
    service = _get_generated_service()
    session_info = service.create_draft_session(
        scope=scope,
        actor=actor,
        ui_id=ui_id,
        name=name,
        tools=list(body.tools or []),
    )
    return JSONResponse(status_code=200, content=session_info)


@router.post(
    "/_generated/{target}/{ui_id}/{name}/chat/sessions/{chat_session_id}/messages"
)
async def chat_generated_ui(
    target: str,
    ui_id: str,
    name: str,
    chat_session_id: str,
    body: UiChatMessageRequest,
    request: Request,
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
):
    _ensure_conversational_ui_enabled()
    _ensure_tgi_enabled()
    scope = _parse_scope(target)
    ui_id = validate_identifier(ui_id, "ui id")
    name = validate_identifier(name, "ui name")
    chat_session_id = validate_identifier(chat_session_id, "session id")
    actor = _extract_actor(access_token)
    service = _get_generated_service()
    incoming_headers = dict(request.headers)
    session_key = session_id(
        try_get_session_id(
            x_inxm_mcp_session_header,
            x_inxm_mcp_session_cookie,
        ),
        access_token,
    )
    requested_group = scope.identifier if scope.kind == "group" else None

    async def event_stream() -> AsyncIterator[bytes]:
        try:
            async with mcp_session_context(
                sessions,
                session_key,
                access_token,
                requested_group,
                incoming_headers,
            ) as session_obj:
                async for chunk in service.stream_chat_update(
                    session=session_obj,
                    scope=scope,
                    actor=actor,
                    ui_id=ui_id,
                    name=name,
                    session_id=chat_session_id,
                    message=body.message,
                    tools=list(body.tools) if body.tools is not None else None,
                    tool_choice=body.tool_choice,
                    access_token=access_token,
                ):
                    yield chunk
        except HTTPException as exc:
            payload = {"error": exc.detail}
            yield f"event: error\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
        except Exception as exc:
            logger.error(
                "[chat_generated_ui] Unexpected exception: %s", str(exc), exc_info=exc
            )
            payload = {"error": "Internal Server Error"}
            yield f"event: error\ndata: {json.dumps(payload)}\n\n".encode("utf-8")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        },
        status_code=200,
    )


@router.get("/_generated/{target}/{ui_id}/{name}/draft")
async def get_generated_ui_draft(
    target: str,
    ui_id: str,
    name: str,
    session_id_query: str = Query(..., alias="session_id"),
    render_mode: str = Query("page", alias="as"),
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
):
    _ensure_conversational_ui_enabled()
    scope = _parse_scope(target)
    ui_id = validate_identifier(ui_id, "ui id")
    name = validate_identifier(name, "ui name")
    session_id_query = validate_identifier(session_id_query, "session id")
    actor = _extract_actor(access_token)
    service = _get_generated_service()

    mode = (render_mode or "card").lower()
    if mode not in {"card", "page", "snippet"}:
        raise HTTPException(status_code=400, detail="Invalid render mode requested")
    expand_payload = mode in {"page", "snippet"}
    record = service.get_draft_ui(
        scope=scope,
        actor=actor,
        ui_id=ui_id,
        name=name,
        session_id=session_id_query,
        expand=expand_payload,
    )

    if mode == "card":
        return JSONResponse(content=record)

    html_section = (record.get("current") or {}).get("html") or {}
    content = html_section.get(mode)
    if not content:
        raise HTTPException(
            status_code=500,
            detail=f"Draft payload does not contain HTML for mode '{mode}'",
        )
    return HTMLResponse(content=content, media_type="text/html")


@router.post(
    "/_generated/{target}/{ui_id}/{name}/chat/sessions/{chat_session_id}/publish"
)
async def publish_generated_ui_draft(
    target: str,
    ui_id: str,
    name: str,
    chat_session_id: str,
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
):
    _ensure_conversational_ui_enabled()
    scope = _parse_scope(target)
    ui_id = validate_identifier(ui_id, "ui id")
    name = validate_identifier(name, "ui name")
    chat_session_id = validate_identifier(chat_session_id, "session id")
    actor = _extract_actor(access_token)
    service = _get_generated_service()
    record = service.publish_draft_session(
        scope=scope,
        actor=actor,
        ui_id=ui_id,
        name=name,
        session_id=chat_session_id,
    )
    metadata = record.get("metadata", {})
    response = UiPublishResponse(
        status="published",
        version=int(metadata.get("version") or 1),
        published_at=str(metadata.get("published_at") or ""),
        record=record,
    )
    return JSONResponse(status_code=200, content=response.model_dump())


@router.delete("/_generated/{target}/{ui_id}/{name}/chat/sessions/{chat_session_id}")
async def discard_generated_ui_draft_session(
    target: str,
    ui_id: str,
    name: str,
    chat_session_id: str,
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
):
    _ensure_conversational_ui_enabled()
    scope = _parse_scope(target)
    ui_id = validate_identifier(ui_id, "ui id")
    name = validate_identifier(name, "ui name")
    chat_session_id = validate_identifier(chat_session_id, "session id")
    actor = _extract_actor(access_token)
    service = _get_generated_service()
    deleted = service.discard_draft_session(
        scope=scope,
        actor=actor,
        ui_id=ui_id,
        name=name,
        session_id=chat_session_id,
    )
    return JSONResponse(
        status_code=200, content={"status": "discarded", "session_deleted": deleted}
    )


@router.get("/_generated/{target}/{ui_id}/{name}/container")
async def get_generated_ui_container(
    target: str,
    ui_id: str,
    name: str,
):
    _ensure_conversational_ui_enabled()
    _parse_scope(target)
    validate_identifier(ui_id, "ui id")
    validate_identifier(name, "ui name")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Generated UI Container</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3f5f7;
      --panel: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --border: #d1d5db;
      --accent: #0f766e;
      --accent-strong: #115e59;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #eef4f4 0%, #f8fbfb 100%);
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(320px, 420px) 1fr;
      gap: 12px;
      min-height: 100vh;
      padding: 12px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
    }}
    .chat {{
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: 0;
    }}
    .header {{
      padding: 12px;
      border-bottom: 1px solid var(--border);
      font-size: 14px;
      font-weight: 600;
    }}
    .history {{
      padding: 12px;
      overflow-y: auto;
      min-height: 0;
      background: #fafbfc;
    }}
    .bubble {{
      margin-bottom: 10px;
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 13px;
      line-height: 1.35;
      white-space: pre-wrap;
    }}
    .bubble.user {{ background: #e5f4f2; border: 1px solid #c7e5e1; }}
    .bubble.assistant {{ background: #ffffff; border: 1px solid #dfe4ea; }}
    .toolbar {{
      display: flex;
      gap: 8px;
      padding: 10px 12px;
      border-top: 1px solid var(--border);
    }}
    .input {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      resize: vertical;
      min-height: 72px;
      font: inherit;
    }}
    .controls {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-top: 8px;
    }}
    .btn {{
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
      border-radius: 8px;
      padding: 7px 11px;
      cursor: pointer;
      font-size: 13px;
    }}
    .btn.primary {{
      background: var(--accent);
      border-color: var(--accent-strong);
      color: #fff;
    }}
    .btn:disabled {{ opacity: 0.6; cursor: not-allowed; }}
    iframe {{
      width: 100%;
      height: calc(100vh - 26px);
      border: 0;
      background: #fff;
    }}
    .status {{ color: var(--muted); font-size: 12px; }}
    @media (max-width: 960px) {{
      .layout {{ grid-template-columns: 1fr; }}
      iframe {{ height: 70vh; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <section class="panel chat">
      <div class="header">Conversational UI Editor</div>
      <div id="history" class="history"></div>
      <div class="toolbar">
        <div style="width:100%">
          <textarea id="message" class="input" placeholder="Describe the UI change..."></textarea>
          <div class="controls">
            <span id="status" class="status">Session initializing...</span>
            <div style="display:flex; gap:8px">
              <button id="discard" class="btn" type="button">Discard</button>
              <button id="publish" class="btn" type="button">Publish</button>
              <button id="send" class="btn primary" type="button">Send</button>
            </div>
          </div>
        </div>
      </div>
    </section>
    <section class="panel">
      <iframe id="preview" title="Generated UI Preview"></iframe>
    </section>
  </div>
  <script>
    const basePath = window.location.pathname.replace(/\\/container$/, '');
    const historyEl = document.getElementById('history');
    const messageEl = document.getElementById('message');
    const statusEl = document.getElementById('status');
    const previewEl = document.getElementById('preview');
    const sendBtn = document.getElementById('send');
    const publishBtn = document.getElementById('publish');
    const discardBtn = document.getElementById('discard');
    let sessionId = null;

    function addBubble(role, text) {{
      const div = document.createElement('div');
      div.className = `bubble ${{role}}`;
      div.textContent = text || '';
      historyEl.appendChild(div);
      historyEl.scrollTop = historyEl.scrollHeight;
      return div;
    }}

    function setStatus(text) {{ statusEl.textContent = text; }}

    function draftUrl() {{
      return `${{basePath}}/draft?session_id=${{encodeURIComponent(sessionId)}}&as=page&_ts=${{Date.now()}}`;
    }}

    async function initSession() {{
      const res = await fetch(`${{basePath}}/chat/sessions`, {{
        method: 'POST',
        credentials: 'include',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{}})
      }});
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      sessionId = data.session_id;
      previewEl.src = draftUrl();
      setStatus(`Draft v${{data.draft_version}}`);
    }}

    async function sendMessage() {{
      const message = messageEl.value.trim();
      if (!message || !sessionId) return;
      messageEl.value = '';
      sendBtn.disabled = true;
      publishBtn.disabled = true;
      discardBtn.disabled = true;

      addBubble('user', message);
      const assistantBubble = addBubble('assistant', '');
      setStatus('Updating draft...');

      const res = await fetch(`${{basePath}}/chat/sessions/${{encodeURIComponent(sessionId)}}/messages`, {{
        method: 'POST',
        credentials: 'include',
        headers: {{
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        }},
        body: JSON.stringify({{ message }})
      }});

      if (!res.ok || !res.body) {{
        assistantBubble.textContent = `Error: ${{await res.text()}}`;
        sendBtn.disabled = false;
        publishBtn.disabled = false;
        discardBtn.disabled = false;
        return;
      }}

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let currentEvent = null;

      while (true) {{
        const {{ done, value }} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {{ stream: true }});
        const lines = buffer.split('\\n');
        buffer = lines.pop() || '';
        for (const line of lines) {{
          if (line.startsWith('event: ')) {{
            currentEvent = line.slice(7).trim();
            continue;
          }}
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (raw === '[DONE]') continue;
          let data = null;
          try {{ data = JSON.parse(raw); }} catch (_err) {{ continue; }}
          if (currentEvent === 'assistant' && data.delta) {{
            assistantBubble.textContent += data.delta;
          }} else if (currentEvent === 'ui_updated') {{
            previewEl.src = draftUrl();
            setStatus(`Draft v${{data.draft_version}} (${{data.update_mode}})`);
          }} else if (currentEvent === 'error') {{
            assistantBubble.textContent += `\\nError: ${{typeof data.error === 'string' ? data.error : JSON.stringify(data.error)}}`;
            setStatus('Error');
          }}
        }}
      }}

      sendBtn.disabled = false;
      publishBtn.disabled = false;
      discardBtn.disabled = false;
    }}

    async function publishDraft() {{
      if (!sessionId) return;
      setStatus('Publishing...');
      const res = await fetch(`${{basePath}}/chat/sessions/${{encodeURIComponent(sessionId)}}/publish`, {{
        method: 'POST',
        credentials: 'include'
      }});
      const payload = await res.json().catch(() => ({{}}));
      if (!res.ok) {{
        setStatus('Publish failed');
        addBubble('assistant', `Publish failed: ${{JSON.stringify(payload)}}`);
        return;
      }}
      setStatus(`Published v${{payload.version}}`);
      addBubble('assistant', `Published successfully as version ${{payload.version}}.`);
    }}

    async function discardDraft() {{
      if (!sessionId) return;
      await fetch(`${{basePath}}/chat/sessions/${{encodeURIComponent(sessionId)}}`, {{
        method: 'DELETE',
        credentials: 'include'
      }});
      setStatus('Draft discarded, creating new session...');
      await initSession();
      historyEl.innerHTML = '';
    }}

    sendBtn.addEventListener('click', () => {{ void sendMessage(); }});
    publishBtn.addEventListener('click', () => {{ void publishDraft(); }});
    discardBtn.addEventListener('click', () => {{ void discardDraft(); }});
    void initSession().catch((err) => {{
      setStatus('Failed to initialize session');
      addBubble('assistant', `Initialization failed: ${{err.message}}`);
    }});
  </script>
</body>
</html>"""

    return HTMLResponse(content=html, media_type="text/html")


def get_target_url(request: Request) -> str:
    """Construct the target URL from the request path."""
    # Remove the proxy prefix from the path
    path = request.url.path
    if path.startswith(PROXY_PREFIX):
        path = path[len(PROXY_PREFIX) :]

    # Ensure path starts with /
    if not path.startswith("/"):
        path = "/" + path

    # Preserve query parameters
    query_string = str(request.url.query)
    if query_string:
        path = f"{path}?{query_string}"

    return urljoin(TARGET_SERVER_URL + "/", path.lstrip("/"))


def prepare_headers(request: Request) -> Dict[str, str]:
    """
    Prepare headers for forwarding to target server.
    Removes hop-by-hop headers and adds proxy headers.
    """
    headers = {}

    # Copy headers, excluding hop-by-hop headers
    for name, value in request.headers.items():
        name_lower = name.lower()
        if name_lower not in HOP_BY_HOP_HEADERS:
            headers[name] = value

    # Add/update forwarded headers
    # X-Forwarded-For: append client IP
    client_ip = request.client.host if request.client else "unknown"
    existing_xff = headers.get("x-forwarded-for", "")
    headers["x-forwarded-for"] = f"{existing_xff}, {client_ip}".strip(", ")

    # X-Forwarded-Host: original host
    headers["x-forwarded-host"] = request.headers.get("host", "")

    # X-Forwarded-Proto: original scheme
    headers["x-forwarded-proto"] = request.url.scheme

    # X-Forwarded-Prefix: proxy prefix for apps that need to know their base path
    headers["x-forwarded-prefix"] = PROXY_PREFIX

    # X-Real-IP: client IP (for single client identification)
    headers["x-real-ip"] = client_ip

    return headers


def rewrite_location_header(location: str, request: Request) -> str:
    """
    Rewrite Location header from target server to proxy URL.
    Handles both absolute and relative URLs.
    """
    if not location:
        return location

    # Parse the location URL
    parsed = urlparse(location)

    # If it's a relative URL, make it absolute with proxy prefix
    if not parsed.scheme and not parsed.netloc:
        # Relative URL - prepend proxy prefix
        if location.startswith("/"):
            return f"{PROXY_PREFIX}{location}"
        else:
            # Relative to current path
            current_path = request.url.path
            if current_path.startswith(PROXY_PREFIX):
                current_path = current_path[len(PROXY_PREFIX) :]
            base_path = "/".join(current_path.split("/")[:-1])
            return f"{PROXY_PREFIX}{base_path}/{location}"

    # If it's an absolute URL pointing to target server, rewrite to proxy
    if parsed.netloc == urlparse(TARGET_SERVER_URL).netloc:
        # Replace target server URL with proxy URL
        path = parsed.path
        query = f"?{parsed.query}" if parsed.query else ""
        fragment = f"#{parsed.fragment}" if parsed.fragment else ""

        # Use PUBLIC_URL if set, otherwise construct from request
        if PUBLIC_URL:
            base = PUBLIC_URL
        else:
            base = f"{request.url.scheme}://{request.headers.get('host', request.client.host)}"

        return f"{base}{PROXY_PREFIX}{path}{query}{fragment}"

    # Otherwise, return as-is (external redirect)
    return location


def rewrite_cookie_path(set_cookie: str) -> str:
    """
    Rewrite the Path attribute in Set-Cookie header to use proxy prefix.
    """
    cookie = SimpleCookie()
    try:
        cookie.load(set_cookie)
    except Exception as e:
        logger.warning(f"Failed to parse cookie: {set_cookie}, error: {e}")
        return set_cookie

    # Rewrite each cookie's path
    for morsel in cookie.values():
        path = morsel.get("path", "/")
        # If path is /, set it to proxy prefix
        if path == "/":
            morsel["path"] = PROXY_PREFIX or "/"
        # If path doesn't start with proxy prefix, prepend it
        elif not path.startswith(PROXY_PREFIX):
            morsel["path"] = f"{PROXY_PREFIX}{path}"

    # Return the rewritten Set-Cookie header
    result = []
    for morsel in cookie.values():
        result.append(morsel.OutputString())
    return "; ".join(result) if result else set_cookie


def rewrite_content_urls(content: bytes, content_type: str) -> bytes:
    """
    Rewrite URLs in HTML/CSS/JS/JSON content to use proxy prefix.
    This is best-effort and may not catch all cases.
    """
    # Only rewrite if configured and content type is appropriate
    if not content:
        return content

    try:
        text = content.decode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        # If we can't decode, return as-is
        logger.warning(
            f"Failed to decode content as UTF-8 for URL rewriting (content_type: {content_type})"
        )
        return content

    target_pattern = re.escape(TARGET_SERVER_URL)
    proxy_prefix_pattern = re.escape(PROXY_PREFIX) if PROXY_PREFIX else ""

    # HTML URL rewriting
    if REWRITE_HTML_URLS and "text/html" in content_type:
        # Rewrite href and src attributes pointing to target
        # Replace absolute URLs
        text = re.sub(
            f'(href|src)="({target_pattern})(/[^"]*)"', f'\\1="{PROXY_PREFIX}\\3"', text
        )
        # Replace absolute URLs with single quotes
        text = re.sub(
            f"(href|src)='({target_pattern})(/[^']*)'", f"\\1='{PROXY_PREFIX}\\3'", text
        )
        # Rewrite root-relative URLs (href="/path") but NOT if already prefixed
        # The negative lookahead ensures we don't rewrite URLs that already start with the prefix
        if proxy_prefix_pattern:
            # Protect against double-prefixing by checking if URL already starts with prefix
            text = re.sub(
                f'(href|src)="(?!{proxy_prefix_pattern})(/[^"]*)"',
                f'\\1="{PROXY_PREFIX}\\2"',
                text,
            )
            text = re.sub(
                f"(href|src)='(?!{proxy_prefix_pattern})(/[^']*)'",
                f"\\1='{PROXY_PREFIX}\\2'",
                text,
            )
        else:
            text = re.sub(r'(href|src)="(/[^"]*)"', f'\\1="{PROXY_PREFIX}\\2"', text)
            text = re.sub(r"(href|src)='(/[^']*)'", f"\\1='{PROXY_PREFIX}\\2'", text)

    # CSS URL rewriting
    if REWRITE_CSS_URLS and (
        "text/css" in content_type or "stylesheet" in content_type
    ):
        # Rewrite url() functions with absolute URLs
        text = re.sub(
            f"url\\([\"']?({target_pattern})(/[^)\"'\\ ]+)[\"']?\\)",
            f'url("{PROXY_PREFIX}\\2")',
            text,
        )
        # Rewrite url() functions with root-relative URLs (but not if already prefixed)
        if proxy_prefix_pattern:
            text = re.sub(
                f"url\\([\"']?(?!{proxy_prefix_pattern})(/[^)\"'\\ ]+)[\"']?\\)",
                f'url("{PROXY_PREFIX}\\1")',
                text,
            )
        else:
            text = re.sub(
                r'url\(["\']?(/[^)"\'\\ ]+)["\']?\)', f'url("{PROXY_PREFIX}\\1")', text
            )
        # Rewrite @import statements with absolute URLs
        text = re.sub(
            f"@import\\s+[\"']({target_pattern})(/[^\"']+)[\"']",
            f'@import "{PROXY_PREFIX}\\2"',
            text,
        )
        # Rewrite @import statements with root-relative URLs (but not if already prefixed)
        if proxy_prefix_pattern:
            text = re.sub(
                f"@import\\s+[\"']?(?!{proxy_prefix_pattern})(/[^\"']+)[\"']?",
                f'@import "{PROXY_PREFIX}\\1"',
                text,
            )
        else:
            text = re.sub(
                r'@import\s+["\'](/[^"\']+)["\']', f'@import "{PROXY_PREFIX}\\1"', text
            )

    # JavaScript URL rewriting
    if REWRITE_JS_URLS and (
        "javascript" in content_type or "application/json" in content_type
    ):
        # Rewrite webpack public path variables (critical for Next.js chunk loading)
        # Handles both full and minified webpack code: __webpack_require__.p or r.p or e.p
        # Patterns like: r.p = "/_next/" or __webpack_require__.p = "/"
        if proxy_prefix_pattern:
            # Rewrite any variable.p assignments (catches minified webpack: r.p, e.p, etc.)
            # Matches: someVar.p = "/" or someVar.p = "/_next/"
            text = re.sub(
                f"(\\b\\w+\\.p\\s*=\\s*[\"'])(?!{proxy_prefix_pattern})(/[^\"']*?)([\"'])",
                f"\\1{PROXY_PREFIX}\\2\\3",
                text,
            )
            # Rewrite webpack chunk URL functions like: return "/" + chunkId + ".js"
            # or: __webpack_require__.u = function(chunkId) { return "/_next/static/..." }
            text = re.sub(
                f"(return\\s+[\"'])(?!{proxy_prefix_pattern})(/[^\"']+?)([\"'])",
                f"\\1{PROXY_PREFIX}\\2\\3",
                text,
            )
        else:
            # Without proxy_prefix_pattern - match any .p assignments
            text = re.sub(
                r'(\b\w+\.p\s*=\s*["\'])(/[^"\']*?)(["\'"])',
                f"\\1{PROXY_PREFIX}\\2\\3",
                text,
            )
            # Rewrite webpack chunk URL return statements
            text = re.sub(
                r'(return\s+["\'])(/[^"\']+?)(["\'"])',
                f"\\1{PROXY_PREFIX}\\2\\3",
                text,
            )

        # Rewrite absolute URLs in strings (both single and double quotes)
        text = re.sub(
            f"([\"'])({target_pattern})(/[^\"']+)([\"'])",
            f"\\1{PROXY_PREFIX}\\3\\4",
            text,
        )
        # Rewrite root-relative URLs in strings that look like file paths
        # Only rewrite if it looks like a path (starts with /, contains . or /)
        # and is not already prefixed
        if proxy_prefix_pattern:
            # Match paths but not already prefixed ones
            # Enhanced pattern to better handle Next.js _next paths and webpack chunks
            text = re.sub(
                f"([\"'])(?!{proxy_prefix_pattern})(/(?:_next|static|api|v\\d+)/[^\"']+)([\"'])",
                f"\\1{PROXY_PREFIX}\\2\\3",
                text,
            )
            # Also match general file paths with extensions
            text = re.sub(
                f"([\"'])(?!{proxy_prefix_pattern})(/[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)*(?:\\.[a-zA-Z0-9]+)?)([\"'])",
                f"\\1{PROXY_PREFIX}\\2\\3",
                text,
            )
        else:
            # Without proxy_prefix_pattern, match Next.js paths and general file paths
            text = re.sub(
                r'(["\'])(/(?:_next|static|api|v\d+)/[^"\']+)(["\'])',
                f"\\1{PROXY_PREFIX}\\2\\3",
                text,
            )
            text = re.sub(
                r'(["\'](/(?:[a-zA-Z0-9_.-]+/)*[a-zA-Z0-9_.-]*\.[a-zA-Z0-9]+)["\'])',
                lambda m: (
                    f'"{PROXY_PREFIX}{m.group(2)}"'
                    if m.group(0)[0] == '"'
                    else f"'{PROXY_PREFIX}{m.group(2)}'"
                ),
                text,
            )

    # JSON URL rewriting (for API responses)
    if REWRITE_JSON_URLS and "application/json" in content_type:
        # Replace URLs in JSON strings with absolute target URLs
        text = re.sub(f'"{target_pattern}(/[^"]*)"', f'"{PROXY_PREFIX}\\1"', text)
        # Replace root-relative URLs that look like API paths (but not if already prefixed)
        if proxy_prefix_pattern:
            text = re.sub(
                f'"(?!{proxy_prefix_pattern})(/(?:api|v\\d+|_next|static)/[^"]+)"',
                f'"{PROXY_PREFIX}\\1"',
                text,
            )
        else:
            text = re.sub(
                r'"(/(?:api|v\d+|_next|static)/[^"]+)"', f'"{PROXY_PREFIX}\\1"', text
            )

    return text.encode("utf-8")


async def stream_response(
    response: httpx.Response, content_type: str
) -> AsyncIterator[bytes]:
    """
    Stream response from target server, optionally rewriting content.

    Note: httpx automatically decompresses responses, so we receive
    already-decompressed content even if content-encoding header is present.
    """
    # For text content that needs rewriting, we need to buffer
    should_rewrite = (
        (REWRITE_HTML_URLS and "text/html" in content_type)
        or (REWRITE_JSON_URLS and "application/json" in content_type)
        or (
            REWRITE_CSS_URLS
            and ("text/css" in content_type or "stylesheet" in content_type)
        )
        or (REWRITE_JS_URLS and "javascript" in content_type)
    )

    if should_rewrite:
        # Buffer entire response for rewriting
        # Note: httpx has already decompressed this for us
        content = b""
        async for chunk in response.aiter_bytes():
            content += chunk

        # Rewrite and yield
        rewritten = rewrite_content_urls(content, content_type)
        yield rewritten
    else:
        # Stream without modification
        async for chunk in response.aiter_bytes():
            yield chunk


async def forward_to_target(request: Request) -> Response:
    """
    Forward incoming requests to the target webserver.
    This function acts as a comprehensive reverse proxy with:
    - Proper header forwarding (X-Forwarded-*)
    - Cookie path rewriting
    - Location header rewriting for redirects
    - Optional URL rewriting in HTML/JSON content
    - Streaming support for large responses
    - Compression handling
    """
    if not TARGET_SERVER_URL:
        raise HTTPException(
            status_code=503,
            detail="TARGET_SERVER_URL is not configured. Proxy is unavailable.",
        )

    with tracer.start_as_current_span("proxy_request") as span:
        target_url = get_target_url(request)
        span.set_attribute("proxy.target_url", target_url)
        span.set_attribute("proxy.method", request.method)

        logger.debug(f"Proxying {request.method} {request.url.path} -> {target_url}")

        # Prepare headers
        headers = prepare_headers(request)

        # Read request body
        body = await request.body()

        try:
            # Create HTTP client with timeout
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(PROXY_TIMEOUT),
                follow_redirects=False,  # Handle redirects manually for rewriting
            ) as client:
                # Forward request to target
                response = await client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    content=body,
                )

                span.set_attribute("proxy.status_code", response.status_code)

                # Prepare response headers
                response_headers = {}

                for name, value in response.headers.items():
                    name_lower = name.lower()

                    # Skip hop-by-hop headers
                    if name_lower in HOP_BY_HOP_HEADERS:
                        continue

                    # Check if we need to rewrite content
                    content_type = response.headers.get("content-type", "")
                    should_rewrite = (
                        (REWRITE_HTML_URLS and "text/html" in content_type)
                        or (REWRITE_JSON_URLS and "application/json" in content_type)
                        or (
                            REWRITE_CSS_URLS
                            and (
                                "text/css" in content_type
                                or "stylesheet" in content_type
                            )
                        )
                        or (REWRITE_JS_URLS and "javascript" in content_type)
                    )

                    # Skip content-encoding header if we're rewriting content
                    # httpx has already decompressed the response for us, so we don't
                    # want to tell the client it's compressed when we're sending decompressed content
                    if name_lower == "content-encoding" and should_rewrite:
                        span.set_attribute("proxy.removed_encoding", value)
                        continue

                    # Rewrite Location header for redirects
                    if name_lower == "location":
                        value = rewrite_location_header(value, request)
                        span.set_attribute("proxy.rewritten_location", value)

                    # Rewrite Set-Cookie path
                    elif name_lower == "set-cookie":
                        value = rewrite_cookie_path(value)

                    response_headers[name] = value

                # Get content type for potential rewriting
                content_type = response.headers.get("content-type", "")

                # Return streaming response
                return StreamingResponse(
                    stream_response(response, content_type),
                    status_code=response.status_code,
                    headers=response_headers,
                    media_type=content_type or None,
                )

        except httpx.TimeoutException as e:
            logger.error(f"Proxy timeout for {target_url}: {e}")
            span.set_attribute("proxy.error", "timeout")
            raise HTTPException(status_code=504, detail="Gateway timeout")

        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to target {target_url}: {e}")
            span.set_attribute("proxy.error", "connection_failed")
            raise HTTPException(
                status_code=502, detail="Bad gateway - cannot connect to target"
            )

        except Exception as e:
            logger.error(f"Proxy error for {target_url}: {e}", exc_info=True)
            span.set_attribute("proxy.error", str(e))
            raise HTTPException(status_code=502, detail=f"Bad gateway: {str(e)}")


# Register catch-all route for proxying
@router.api_route(
    "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
)
async def proxy_all(request: Request, path: str):
    """Catch-all route that proxies all requests to the target server."""
    return await forward_to_target(request)
