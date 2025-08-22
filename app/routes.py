import logging
from fastapi import APIRouter, HTTPException, Header, Cookie, Request
from fastapi.responses import JSONResponse
from typing import Optional, Dict
import uuid
import os
from .session import MCPLocalSessionTask, mcp_session, try_get_session_id, session_id
from .session_manager import session_manager
from .models import RunToolsResult
from .mcp_server import get_server_params
from .oauth.decorator import decorate_args_with_oauth_token
from fnmatch import fnmatch
from opentelemetry import trace
from .oauth.token_exchange import UserLoggedOutException
from .utils import mask_token

router = APIRouter()
sessions = session_manager()

logger = logging.getLogger("uvicorn.error")

TOKEN_NAME = os.environ.get("TOKEN_NAME", "X-Auth-Request-Access-Token")
SESSION_FIELD_NAME = os.environ.get("SESSION_FIELD_NAME", "x-inxm-mcp-session")
MCP_BASE_PATH = os.environ.get("MCP_BASE_PATH", "")
INCLUDE_TOOLS = [t for t in os.environ.get("INCLUDE_TOOLS", "").split(",") if t]
EXCLUDE_TOOLS = [t for t in os.environ.get("EXCLUDE_TOOLS", "").split(",") if t]

tracer = trace.get_tracer(__name__)


def matches_pattern(value, pattern):
    """Check if a value matches a glob-like pattern."""
    return fnmatch(value, pattern)


def map_tools(tools):
    """Map tools with INCLUDE_TOOLS and EXCLUDE_TOOLS filters applied."""

    filtered_tools = []
    for tool in tools.tools:
        include_match = any(
            matches_pattern(tool.name, pattern) for pattern in INCLUDE_TOOLS if pattern
        )
        exclude_match = any(
            matches_pattern(tool.name, pattern) for pattern in EXCLUDE_TOOLS if pattern
        )
        print(
            f"Tool: {tool.name}, Include Match: {include_match} - {INCLUDE_TOOLS}, Exclude Match: {exclude_match} - {EXCLUDE_TOOLS}"
        )
        if INCLUDE_TOOLS and any(INCLUDE_TOOLS) and not include_match:
            continue
        if EXCLUDE_TOOLS and any(EXCLUDE_TOOLS) and exclude_match:
            continue

        filtered_tools.append(
            {
                "name": tool.name,
                "title": tool.title,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
                "outputSchema": tool.outputSchema,
                "annotations": tool.annotations,
                "meta": tool.meta,
                "url": f"{MCP_BASE_PATH}/tools/{tool.name}",
            }
        )

    return filtered_tools


if MCP_BASE_PATH:
    router.prefix = MCP_BASE_PATH


@router.get("/tools")
async def list_tools(
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
):
    try:
        with tracer.start_as_current_span("list_tools") as span:
            x_inxm_mcp_session = session_id(
                try_get_session_id(
                    x_inxm_mcp_session_header, x_inxm_mcp_session_cookie
                ),
                access_token,
            )
            if x_inxm_mcp_session:
                span.set_attribute("session.id", x_inxm_mcp_session)
            logger.info(
                mask_token(
                    f"[Tools] Listing tools. Session: {x_inxm_mcp_session}",
                    x_inxm_mcp_session,
                )
            )
            if x_inxm_mcp_session is None:
                async with mcp_session(get_server_params(access_token)) as session:
                    result = await session.list_tools()
                    logger.debug("[Tools] Tools listed without session.")
                    return map_tools(result)

            mcp_task = sessions.get(x_inxm_mcp_session)
            if not mcp_task:
                logger.warning(
                    mask_token(
                        f"[Tools] Session not found: {x_inxm_mcp_session}",
                        x_inxm_mcp_session,
                    )
                )
                raise HTTPException(status_code=404, detail="Session not found")
            result = await mcp_task.request("list_tools")
            logger.debug(
                mask_token(
                    f"[Tools] Tools listed for session {x_inxm_mcp_session}.",
                    x_inxm_mcp_session,
                )
            )
            return map_tools(result)
    except UserLoggedOutException as e:
        logger.warning(f"[Tools] Unauthorized access: {str(e)}")
        raise HTTPException(status_code=401, detail=e.message)


@router.post("/tools/{tool_name}")
async def run_tool(
    tool_name: str,
    request: Request,
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias=SESSION_FIELD_NAME),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias=SESSION_FIELD_NAME),
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
    args: Optional[Dict] = None,
):
    try:
        with tracer.start_as_current_span(
            "run_tool", attributes={"tool.name": tool_name}
        ) as span:
            x_inxm_mcp_session = session_id(
                try_get_session_id(
                    x_inxm_mcp_session_header,
                    x_inxm_mcp_session_cookie,
                    args.get("inxm-session", None) if args else None,
                ),
                access_token,
            )
            if x_inxm_mcp_session:
                span.set_attribute("session.id", x_inxm_mcp_session)
            if args and "inxm-session" in args:
                args = dict(args)
                args.pop("inxm-session")
            logger.info(
                mask_token(
                    f"[Tool-Call] Tool call: {tool_name}, Session: {x_inxm_mcp_session}, Args: {args}",
                    x_inxm_mcp_session,
                )
            )
            if x_inxm_mcp_session is None:
                async with mcp_session(get_server_params(access_token)) as session:
                    tools = await session.list_tools()
                    decorated_args = await decorate_args_with_oauth_token(
                        tools, tool_name, args, access_token
                    )
                    result = await session.call_tool(tool_name, decorated_args)
                    result = RunToolsResult(result)
            else:
                mcp_task = sessions.get(x_inxm_mcp_session)
                if not mcp_task:
                    logger.warning(
                        mask_token(
                            f"[Tool-Call] Session not found: {x_inxm_mcp_session}",
                            x_inxm_mcp_session,
                        )
                    )
                    raise HTTPException(
                        status_code=404,
                        detail="Session not found. It might have expired, please start a new.",
                    )
                else:
                    tools = await mcp_task.request("list_tools")
                    decorated_args = await decorate_args_with_oauth_token(
                        tools, tool_name, args, access_token
                    )
                    result = RunToolsResult(
                        await mcp_task.request(
                            {
                                "action": "run_tool",
                                "tool_name": tool_name,
                                "args": decorated_args,
                            }
                        )
                    )

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
        logger.error(f"[Tool-Call] Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/session/start")
async def start_session(
    access_token: Optional[str] = Header(None, alias=TOKEN_NAME),
):
    try:
        with tracer.start_as_current_span("start_session") as span:
            x_inxm_mcp_session = session_id(str(uuid.uuid4()), access_token)
            span.set_attribute("session.id", x_inxm_mcp_session)
            mcp_task = MCPLocalSessionTask(get_server_params(access_token))
            mcp_task.start()
            sessions.set(x_inxm_mcp_session, mcp_task)
            logger.debug(
                mask_token(
                    f"[Session] New session started: {x_inxm_mcp_session}",
                    x_inxm_mcp_session,
                )
            )
            response = JSONResponse(content={SESSION_FIELD_NAME: x_inxm_mcp_session})
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
        logger.error(f"[Tool-Call] Unexpected error: {str(e)}")
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
        logger.error(f"[Tool-Call] Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
