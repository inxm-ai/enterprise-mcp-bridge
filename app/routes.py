import logging
from fastapi import APIRouter, HTTPException, Header, Cookie, Query
from fastapi.responses import JSONResponse
from typing import Optional, Dict
import uuid
import os

from app.utils.traced_requests import traced_request
from app.session import MCPLocalSessionTask, try_get_session_id, session_id
from app.session_manager import mcp_session_context, session_manager
from .mcp_server import get_server_params
from .oauth.user_info import get_data_access_manager
from opentelemetry import trace
from .oauth.token_exchange import UserLoggedOutException
from .utils import mask_token
from .utils.exception_logging import (
    find_exception_in_exception_groups,
    log_exception_with_details,
)
from .tgi.routes import router as tgi_router

router = APIRouter()
sessions = session_manager()

logger = logging.getLogger("uvicorn.error")

TOKEN_NAME = os.environ.get("TOKEN_NAME", "X-Auth-Request-Access-Token")
SESSION_FIELD_NAME = os.environ.get("SESSION_FIELD_NAME", "x-inxm-mcp-session")
MCP_BASE_PATH = os.environ.get("MCP_BASE_PATH", "")
INCLUDE_TOOLS = [t for t in os.environ.get("INCLUDE_TOOLS", "").split(",") if t]
EXCLUDE_TOOLS = [t for t in os.environ.get("EXCLUDE_TOOLS", "").split(",") if t]

tracer = trace.get_tracer(__name__)

if MCP_BASE_PATH:
    router.prefix = MCP_BASE_PATH


@router.get("/prompts")
async def list_prompts(
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
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
        with traced_request(
            tracer,
            operation="list_prompts",
            session_value=x_inxm_mcp_session,
            group=group,
            start_message=f"[Prompts] Listing prompts. Session: {x_inxm_mcp_session}, Group: {group}",
        ):
            async with mcp_session_context(
                sessions, x_inxm_mcp_session, access_token, group
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
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
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
        with traced_request(
            tracer,
            operation="list_tools",
            session_value=x_inxm_mcp_session,
            group=group,
            start_message=f"[Tools] Listing tools. Session: {x_inxm_mcp_session}, Group: {group}",
        ):
            async with mcp_session_context(
                sessions, x_inxm_mcp_session, access_token, group
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


@router.post("/tools/{tool_name}")
async def run_tool(
    tool_name: str,
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
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
        with traced_request(
            tracer=tracer,
            operation="run_tool",
            session_value=x_inxm_mcp_session,
            group=group,
            start_message=f"[Tool-Call] Tool call: {tool_name}, Session: {x_inxm_mcp_session}, Group: {group}, Args: {args}",
            extra_attrs={"tool.name": tool_name},
        ):
            async with mcp_session_context(
                sessions, x_inxm_mcp_session, access_token, group
            ) as session:
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
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
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
        with traced_request(
            tracer=tracer,
            operation="run_prompt",
            session_value=x_inxm_mcp_session,
            group=group,
            start_message=f"[Prompt-Call] Prompt call: {prompt_name}, Session: {x_inxm_mcp_session}, Group: {group}, Args: {args}",
            extra_attrs={"prompt.name": prompt_name},
        ):
            async with mcp_session_context(
                sessions, x_inxm_mcp_session, access_token, group
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
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
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

            mcp_task = MCPLocalSessionTask(get_server_params(access_token, group))
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
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
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
