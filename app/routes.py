import logging
import io
from app.vars import EFFECT_TOOLS, MCP_BASE_PATH, SESSION_FIELD_NAME
from fastapi import APIRouter, HTTPException, Header, Cookie, Query, Request, Depends
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
    HTMLResponse,
    Response,
)
from typing import Optional, Dict
import uuid
import os
import asyncio

from app.utils.traced_requests import traced_request
from app.session import (
    MCPLocalSessionTask,
    try_get_session_id,
    session_id,
    build_mcp_client_strategy,
)
from app.session_manager import mcp_session_context, session_manager
from .oauth.user_info import get_data_access_manager
from opentelemetry import trace
from .oauth.token_exchange import UserLoggedOutException
from .oauth.token_dependency import get_access_token
from .utils import mask_token
from .utils.exception_logging import (
    find_exception_in_exception_groups,
    log_exception_with_details,
)
from .tgi.routes import router as tgi_router
from .tgi.tool_dry_run.tool_response import get_tool_dry_run_response
from .app_facade.route import router as app_facade_router
from app.well_known.agent import router as agent_router

router = APIRouter()
sessions = session_manager()

logger = logging.getLogger("uvicorn.error")

tracer = trace.get_tracer(__name__)

if MCP_BASE_PATH:
    router.prefix = MCP_BASE_PATH
    logger.info(f"Using MCP_BASE_PATH: {MCP_BASE_PATH}")
else:
    logger.info("No MCP_BASE_PATH set, using root path")


def _extract_request_headers(request: Request) -> dict[str, str]:
    """Extract headers from the incoming request as a dictionary."""
    return dict(request.headers)


@router.get("/resources")
async def list_resources(
    request: Request,
    access_token: Optional[str] = Depends(get_access_token),
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
    group: Optional[str] = Query(
        None, description="Group name for sessionless group-specific data access"
    ),
):
    try:
        x_inxm_mcp_session = session_id(
            try_get_session_id(x_inxm_mcp_session_header, x_inxm_mcp_session_cookie),
            access_token,
        )
        incoming_headers = _extract_request_headers(request)
        with traced_request(
            tracer,
            operation="list_resources",
            session_value=x_inxm_mcp_session,
            group=group,
            start_message=f"[Resources] Listing resources. Session: {x_inxm_mcp_session}, Group: {group}",
        ):
            async with mcp_session_context(
                sessions, x_inxm_mcp_session, access_token, group, incoming_headers
            ) as session:
                result = await session.list_resources()
                logger.debug(
                    mask_token(
                        f"[Resources] Resources listed. Session: {x_inxm_mcp_session}",
                        x_inxm_mcp_session,
                    )
                )
                return result
    except HTTPException as e:
        raise e
    except UserLoggedOutException as e:
        logger.warning(f"[Resources] Unauthorized access: {str(e)}")
        raise HTTPException(status_code=401, detail=e.message)
    except Exception as e:
        log_exception_with_details(logger, "[Session]", e)
        child_http_exception = find_exception_in_exception_groups(e, HTTPException)
        if child_http_exception:
            raise child_http_exception
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/resources/{resource_name}")
async def get_resource_details(
    resource_name: str,
    request: Request,
    access_token: Optional[str] = Depends(get_access_token),
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
    group: Optional[str] = Query(
        None, description="Group name for sessionless group-specific data access"
    ),
):
    try:
        x_inxm_mcp_session = session_id(
            try_get_session_id(x_inxm_mcp_session_header, x_inxm_mcp_session_cookie),
            access_token,
        )
        incoming_headers = _extract_request_headers(request)
        with traced_request(
            tracer,
            operation="get_resource_details",
            session_value=x_inxm_mcp_session,
            group=group,
            start_message=f"[Resource-Details] Getting resource details. Session: {x_inxm_mcp_session}, Group: {group}",
        ):
            async with mcp_session_context(
                sessions, x_inxm_mcp_session, access_token, group, incoming_headers
            ) as session:
                result = await session.list_resources()
                # find the resource with the given name
                logger.debug(f"Looking for resource: {resource_name} in {result}")
                result = next(
                    (res for res in result.resources if res.name == resource_name), None
                )
                logger.debug(
                    mask_token(
                        f"[Resource-Details] Resource details retrieved. Session: {x_inxm_mcp_session}",
                        x_inxm_mcp_session,
                    )
                )
                if result is None:
                    raise HTTPException(status_code=404, detail="Resource not found")
                resource = await session.read_resource(result.uri)
                if resource is None:
                    raise HTTPException(status_code=404, detail="Resource not found")
                if resource.contents is None:
                    raise HTTPException(status_code=404, detail="Resource is empty")
                resource = resource.contents[0]
                logger.info(f"Resource retrieved: {resource}")
                mime_type = resource.mimeType or "text/plain"
                if hasattr(resource, "blob"):
                    blob = resource.blob
                else:
                    blob = None
                if hasattr(resource, "text"):
                    text = resource.text or ""
                else:
                    text = ""
                if blob:
                    # If blob is bytes, wrap in BytesIO; if file-like, use directly
                    if isinstance(blob, bytes):
                        stream = io.BytesIO(blob)
                    else:
                        stream = blob
                    return StreamingResponse(stream, media_type=mime_type)
                elif mime_type == "text/html":
                    return HTMLResponse(content=text, media_type=mime_type)
                elif text:
                    return Response(content=text, media_type=mime_type)
                else:
                    return Response(status_code=204)

    except HTTPException as e:
        raise e
    except UserLoggedOutException as e:
        logger.warning(f"[Resource-Details] Unauthorized access: {str(e)}")
        raise HTTPException(status_code=401, detail=e.message)
    except Exception as e:
        log_exception_with_details(logger, "[Resource-Details]", e)
        child_http_exception = find_exception_in_exception_groups(e, HTTPException)
        if child_http_exception:
            raise child_http_exception
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/prompts")
async def list_prompts(
    request: Request,
    access_token: Optional[str] = Depends(get_access_token),
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
    group: Optional[str] = Query(
        None, description="Group name for sessionless group-specific data access"
    ),
):
    try:
        x_inxm_mcp_session = session_id(
            try_get_session_id(x_inxm_mcp_session_header, x_inxm_mcp_session_cookie),
            access_token,
        )
        incoming_headers = _extract_request_headers(request)
        with traced_request(
            tracer,
            operation="list_prompts",
            session_value=x_inxm_mcp_session,
            group=group,
            start_message=f"[Prompts] Listing prompts. Session: {x_inxm_mcp_session}, Group: {group}",
        ):
            async with mcp_session_context(
                sessions, x_inxm_mcp_session, access_token, group, incoming_headers
            ) as session:
                result = await session.list_prompts()
                logger.debug(
                    mask_token(
                        f"[Prompts] Prompts listed. Session: {x_inxm_mcp_session}",
                        x_inxm_mcp_session,
                    )
                )
                return result
    except HTTPException as e:
        raise e
    except UserLoggedOutException as e:
        logger.warning(f"[Prompts] Unauthorized access: {str(e)}")
        raise HTTPException(status_code=401, detail=e.message)
    except Exception as e:
        log_exception_with_details(logger, "[Session]", e)
        child_http_exception = find_exception_in_exception_groups(e, HTTPException)
        if child_http_exception:
            raise child_http_exception
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/tools")
async def list_tools(
    request: Request,
    access_token: Optional[str] = Depends(get_access_token),
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
    group: Optional[str] = Query(
        None, description="Group name for sessionless group-specific data access"
    ),
):
    try:
        x_inxm_mcp_session = session_id(
            try_get_session_id(x_inxm_mcp_session_header, x_inxm_mcp_session_cookie),
            access_token,
        )
        incoming_headers = _extract_request_headers(request)
        with traced_request(
            tracer,
            operation="list_tools",
            session_value=x_inxm_mcp_session,
            group=group,
            start_message=f"[Tools] Listing tools. Session: {x_inxm_mcp_session}, Group: {group}",
        ):
            async with mcp_session_context(
                sessions, x_inxm_mcp_session, access_token, group, incoming_headers
            ) as session:
                result = await session.list_tools()
                logger.debug(
                    mask_token(
                        f"[Tools] Tools listed. Session: {x_inxm_mcp_session}",
                        x_inxm_mcp_session,
                    )
                )
                return result
    except UserLoggedOutException as e:
        logger.warning(f"[Tools] Unauthorized access: {str(e)}")
        raise HTTPException(status_code=401, detail=e.message)


@router.get("/tools/{tool_name}")
async def get_tool_details(
    tool_name: str,
    request: Request,
    access_token: Optional[str] = Depends(get_access_token),
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
    group: Optional[str] = Query(
        None, description="Group name for sessionless group-specific data access"
    ),
):
    try:
        x_inxm_mcp_session = session_id(
            try_get_session_id(x_inxm_mcp_session_header, x_inxm_mcp_session_cookie),
            access_token,
        )
        incoming_headers = _extract_request_headers(request)
        with traced_request(
            tracer,
            operation="get_tool_details",
            session_value=x_inxm_mcp_session,
            group=group,
            start_message=f"[Tool-Details] Getting tool details. Session: {x_inxm_mcp_session}, Group: {group}",
        ):
            async with mcp_session_context(
                sessions, x_inxm_mcp_session, access_token, group, incoming_headers
            ) as session:
                result = await session.list_tools()
                # find the tool with the given name
                result = next(
                    (tool for tool in result if tool.get("name") == tool_name), None
                )
                logger.debug(
                    mask_token(
                        f"[Tool-Details] Tool details retrieved. Session: {x_inxm_mcp_session}",
                        x_inxm_mcp_session,
                    )
                )
                if result is None:
                    raise HTTPException(status_code=404, detail="Tool not found")
                return result
    except HTTPException as e:
        raise e
    except UserLoggedOutException as e:
        logger.warning(f"[Tool-Details] Unauthorized access: {str(e)}")
        raise HTTPException(status_code=401, detail=e.message)
    except Exception as e:
        log_exception_with_details(logger, "[Tool-Details]", e)
        child_http_exception = find_exception_in_exception_groups(e, HTTPException)
        if child_http_exception:
            raise child_http_exception
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/tools/{tool_name}")
async def run_tool(
    tool_name: str,
    request: Request,
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
    x_inxm_dry_run: Optional[str] = Header(None, alias="X-Inxm-Dry-Run"),
    access_token: Optional[str] = Depends(get_access_token),
    args: Optional[Dict] = None,
    group: Optional[str] = Query(
        None, description="Group name for sessionless group-specific data access"
    ),
):
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
        incoming_headers = _extract_request_headers(request)
        with traced_request(
            tracer=tracer,
            operation="run_tool",
            session_value=x_inxm_mcp_session,
            group=group,
            start_message=f"[Tool-Call] Tool call: {tool_name}, Session: {x_inxm_mcp_session}, Group: {group}, Args: {args}",
            extra_attrs={"tool.name": tool_name},
        ):
            async with mcp_session_context(
                sessions, x_inxm_mcp_session, access_token, group, incoming_headers
            ) as session:
                if (
                    x_inxm_dry_run
                    and x_inxm_dry_run.lower() == "true"
                    and tool_name in EFFECT_TOOLS
                ):
                    tools = await session.list_tools()
                    tool = next(
                        (tool for tool in tools if tool.get("name") == tool_name), None
                    )
                    # get_tool_dry_run_response is async; but tests and other
                    # callsites may patch it with a sync function. Support both
                    # by detecting coroutine returns and awaiting when needed.
                    maybe_result = get_tool_dry_run_response(session, tool, args or {})
                    if asyncio.iscoroutine(maybe_result):
                        result = await maybe_result
                    else:
                        result = maybe_result
                else:
                    result = await session.call_tool(tool_name, args, access_token)

        logger.info(f"[Tool-Call] Tool {tool_name} called. Result: {result}")
        if result.isError:
            if "Unknown tool" in result.content[0].text:
                logger.info(f"[Tool-Call] Tool not found: {tool_name}")
                raise HTTPException(status_code=404, detail=str(result))
            if "validation error" in result.content[0].text:
                logger.info(
                    f"[Tool-Call] Tool called with invalid parameters: {tool_name}. Result: {result}"
                )
                raise HTTPException(status_code=400, detail=str(result))

            logger.error(f"[Tool-Call] Error in tool {tool_name}: {result}")
            raise HTTPException(status_code=500, detail=str(result))
        return result
    except HTTPException as e:
        raise e
    except UserLoggedOutException as e:
        logger.warning(f"[Tool-Call] Unauthorized access: {str(e)}")
        raise HTTPException(status_code=401, detail=e.message)
    except Exception as e:
        # Handle TaskGroup exceptions with multiple sub-exceptions
        log_exception_with_details(logger, "[Tool-Call]", e)
        child_http_exception = find_exception_in_exception_groups(e, HTTPException)
        if child_http_exception:
            raise child_http_exception
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/prompts/{prompt_name}")
async def run_prompt(
    prompt_name: str,
    request: Request,
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
    access_token: Optional[str] = Depends(get_access_token),
    args: Optional[Dict] = None,
    group: Optional[str] = Query(
        None, description="Group name for sessionless group-specific data access"
    ),
):
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
        incoming_headers = _extract_request_headers(request)
        with traced_request(
            tracer=tracer,
            operation="run_prompt",
            session_value=x_inxm_mcp_session,
            group=group,
            start_message=f"[Prompt-Call] Prompt call: {prompt_name}, Session: {x_inxm_mcp_session}, Group: {group}, Args: {args}",
            extra_attrs={"prompt.name": prompt_name},
        ):
            async with mcp_session_context(
                sessions, x_inxm_mcp_session, access_token, group, incoming_headers
            ) as session:
                result = await session.call_prompt(prompt_name, args)

        logger.info(f"[Prompt-Call] Prompt {prompt_name} called. Result: {result}")
        if result.isError:
            if "Unknown prompt" in result.content[0].text:
                logger.info(f"[Prompt-Call] Prompt not found: {prompt_name}")
                raise HTTPException(status_code=404, detail=str(result))
            if "validation error" in result.content[0].text:
                logger.info(
                    f"[Prompt-Call] Prompt called with invalid parameters: {prompt_name}. Result: {result}"
                )
                raise HTTPException(status_code=400, detail=str(result))

            logger.error(f"[Prompt-Call] Error in prompt {prompt_name}: {result}")
            raise HTTPException(status_code=500, detail=str(result))
        return result
    except HTTPException as e:
        raise e
    except UserLoggedOutException as e:
        logger.warning(f"[Prompt-Call] Unauthorized access: {str(e)}")
        raise HTTPException(status_code=401, detail=e.message)
    except Exception as e:
        # Handle TaskGroup exceptions with multiple sub-exceptions
        log_exception_with_details(logger, "[Prompt-Call]", e)
        child_http_exception = find_exception_in_exception_groups(e, HTTPException)
        if child_http_exception:
            raise child_http_exception
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/session/start")
async def start_session(
    access_token: Optional[str] = Depends(get_access_token),
    group: Optional[str] = Query(
        None, description="Group name for group-specific data access"
    ),
):
    try:
        with tracer.start_as_current_span("start_session") as span:
            x_inxm_mcp_session = session_id(str(uuid.uuid4()), access_token)
            span.set_attribute("session.id", x_inxm_mcp_session)
            if group:
                span.set_attribute("session.group", group)

            # Validate group access if specified
            if group:
                data_manager = get_data_access_manager()
                try:
                    # This will raise PermissionError if user doesn't have access
                    data_manager.resolve_data_resource(access_token, group)
                    logger.info(f"Group access validated for user, group: {group}")
                except AssertionError as e:
                    logger.error(f"Group access assertion error: {str(e)}")
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid group or token",
                    )
                except PermissionError as e:
                    logger.warning(f"Group access denied: {str(e)}")
                    raise HTTPException(
                        status_code=403,
                        detail=f"Access denied to group '{group}': {str(e)}",
                    )

            try:
                strategy = build_mcp_client_strategy(
                    access_token=access_token, requested_group=group
                )
            except ValueError as exc:
                logger.error(f"[Session] Invalid MCP configuration: {exc}")
                raise HTTPException(status_code=400, detail=str(exc))

            mcp_task = MCPLocalSessionTask(strategy)
            mcp_task.start()
            sessions.set(x_inxm_mcp_session, mcp_task)

            session_info = {SESSION_FIELD_NAME: x_inxm_mcp_session}
            if group:
                session_info["group"] = group

            logger.debug(
                mask_token(
                    f"[Session] New session started: {x_inxm_mcp_session} for group: {group}",
                    x_inxm_mcp_session,
                )
            )
            response = JSONResponse(content=session_info)
            response.set_cookie(
                key=SESSION_FIELD_NAME,
                value=x_inxm_mcp_session,
                httponly=True,
                samesite="lax",
                secure=(
                    True
                    if os.environ.get("HTTPS_ENABLED", "false").lower() == "true"
                    else False
                ),
            )
            return response
    except HTTPException as e:
        raise e
    except UserLoggedOutException as e:
        logger.warning(f"[Tool-Call] Unauthorized access: {str(e)}")
        raise HTTPException(status_code=401, detail=e.message)
    except Exception as e:
        # Handle TaskGroup exceptions with multiple sub-exceptions
        log_exception_with_details(logger, "[Session]", e)
        child_http_exception = find_exception_in_exception_groups(e, HTTPException)
        if child_http_exception:
            raise child_http_exception
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/session/close")
async def close_session(
    access_token: Optional[str] = Depends(get_access_token),
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
):
    try:
        with tracer.start_as_current_span("close_session") as span:
            x_inxm_mcp_session = session_id(
                try_get_session_id(
                    x_inxm_mcp_session_header, x_inxm_mcp_session_cookie
                ),
                access_token,
            )
            span.set_attribute("session.id", x_inxm_mcp_session)
            if x_inxm_mcp_session is None:
                logger.warning("[Session] Session header missing on close.")
                raise HTTPException(status_code=400, detail="Session header missing")
            mcp_task = sessions.pop(x_inxm_mcp_session, None)
            if not mcp_task:
                logger.warning(
                    mask_token(
                        f"[Session] Session not found on close: {x_inxm_mcp_session}",
                        x_inxm_mcp_session,
                    )
                )
                raise HTTPException(status_code=404, detail="Session not found")
            await mcp_task.stop()
            logger.debug(
                mask_token(
                    f"[Session] Session closed: {x_inxm_mcp_session}",
                    x_inxm_mcp_session,
                )
            )
            return {"status": "closed"}
    except HTTPException as e:
        raise e
    except UserLoggedOutException as e:
        logger.warning(f"[Tool-Call] Unauthorized access: {str(e)}")
        raise HTTPException(status_code=401, detail=e.message)
    except Exception as e:
        # Handle TaskGroup exceptions with multiple sub-exceptions
        log_exception_with_details(logger, "[Session]", e)
        raise HTTPException(status_code=500, detail="Internal server error")


# Include TGI router
router.include_router(tgi_router)

# Include well-known agent router
router.include_router(agent_router)

# Include app proxy router
router.include_router(app_facade_router)
