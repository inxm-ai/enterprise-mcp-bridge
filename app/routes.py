
import logging
from fastapi import APIRouter, HTTPException, Header, Cookie, Request
from fastapi.responses import JSONResponse
from typing import Optional, Dict
import uuid
import os
from mcp import StdioServerParameters
from .session import MCPLocalSessionTask, mcp_session, try_get_session_id, session_id
from .session_manager import session_manager
from .models import RunToolRequest, RunToolsResult
from .mcp_server import get_server_params
from .oauth.decorator import decorate_args_with_oauth_token
import sys

router = APIRouter()
sessions = session_manager()

logger = logging.getLogger("uvicorn.error")

MCP_BASE_PATH = os.environ.get("MCP_BASE_PATH", "")

def map_tools(tools):
    logger.info(f"[map_tools] Mapping tools: {tools}")
    return [
        {
            "name": tool.name,
            "title": tool.title,
            "description": tool.description,
            "inputSchema": tool.inputSchema,
            "outputSchema": tool.outputSchema,
            "annotations": tool.annotations,
            "meta": tool.meta,
            "url": f"{MCP_BASE_PATH}/tools/{tool.name}"
        }
        for tool in tools.tools
    ]

@router.get("/tools")
async def list_tools(
    oauth_token: Optional[str] = Cookie(None, alias="_oauth2_proxy"),
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias="x-inxm-mcp-session"),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias="x-inxm-mcp-session")
):
    x_inxm_mcp_session = session_id(
        try_get_session_id(x_inxm_mcp_session_header, x_inxm_mcp_session_cookie),
        oauth_token
    )
    logger.info(f"[Tools] Listing tools. Session: {x_inxm_mcp_session}")
    if x_inxm_mcp_session is None:
        async with mcp_session(get_server_params(oauth_token)) as session:
            result = await session.list_tools()
            logger.debug("[Tools] Tools listed without session.")
            return map_tools(result)

    mcp_task = sessions.get(x_inxm_mcp_session)
    if not mcp_task:
        logger.warning(f"[Tools] Session not found: {x_inxm_mcp_session}")
        raise HTTPException(status_code=404, detail="Session not found")
    result = await mcp_task.request("list_tools")
    logger.debug(f"[Tools] Tools listed for session {x_inxm_mcp_session}.")
    return map_tools(result)

@router.post("/tools/{tool_name}")
async def run_tool(
    tool_name: str,
    request: Request,
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias="x-inxm-mcp-session"),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias="x-inxm-mcp-session"),
    oauth_token: Optional[str] = Cookie(None, alias="_oauth2_proxy"),
    args: Optional[Dict] = None, 
    ):
    x_inxm_mcp_session = session_id(
        try_get_session_id(x_inxm_mcp_session_header, x_inxm_mcp_session_cookie, args.get('inxm-session', None) if args else None),
        oauth_token
    )
    if not oauth_token:
        oauth_token = request.cookies.get("_oauth2_proxy", "")
    if args and 'inxm-session' in args:
        args = dict(args)
        args.pop('inxm-session')
    logger.info(f"[Tool-Call] Tool call: {tool_name}, Session: {x_inxm_mcp_session}, Args: {args}")
    if x_inxm_mcp_session is None:
        async with mcp_session(get_server_params(oauth_token)) as session:
            tools = await session.list_tools()
            decorated_args = await decorate_args_with_oauth_token(tools, tool_name, args, oauth_token)
            result = await session.call_tool(tool_name, decorated_args)
            result = RunToolsResult(result)
    else:
        mcp_task = sessions.get(x_inxm_mcp_session)
        if not mcp_task:
            logger.warning(f"[Tool-Call] Session not found: {x_inxm_mcp_session}")
            raise HTTPException(status_code=404, detail="Session not found. It might have expired, please start a new.")
        else:
            tools = await mcp_task.request("list_tools")
            decorated_args = await decorate_args_with_oauth_token(tools, tool_name, args, oauth_token)
            result = RunToolsResult(await mcp_task.request({"action": "run_tool", "tool_name": tool_name, "args": decorated_args}))

    logger.info(f"[Tool-Call] Tool {tool_name} called. Result: {result}")
    if result.isError:
        if "Unknown tool" in result.content[0].text:
            logger.info(f"[Tool-Call] Tool not found: {tool_name}")
            raise HTTPException(status_code=404, detail=str(result))
        if "validation error" in result.content[0].text:
            logger.info(f"[Tool-Call] Tool called with invalid parameters: {tool_name}. Result: {result}")
            raise HTTPException(status_code=400, detail=str(result))

        logger.error(f"[Tool-Call] Error in tool {tool_name}: {result}")
        raise HTTPException(status_code=500, detail=str(result))
    return result

@router.post("/session/start")
async def start_session(
    oauth_token: Optional[str] = Cookie(None, alias="_oauth2_proxy"),
):
    x_inxm_mcp_session = session_id(str(uuid.uuid4()), oauth_token)
    mcp_task = MCPLocalSessionTask(get_server_params(oauth_token))
    mcp_task.start()
    sessions.set(x_inxm_mcp_session, mcp_task)
    logger.debug(f"[Session] New session started: {x_inxm_mcp_session}")
    response = JSONResponse(content={"x-inxm-mcp-session": x_inxm_mcp_session})
    response.set_cookie(key="x-inxm-mcp-session", value=x_inxm_mcp_session, httponly=True, samesite="lax")
    return response

@router.post("/session/close")
async def close_session(
    oauth_token: Optional[str] = Cookie(None, alias="_oauth2_proxy"),
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias="x-inxm-mcp-session"),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias="x-inxm-mcp-session")
):
    x_inxm_mcp_session = session_id(
        try_get_session_id(x_inxm_mcp_session_header, x_inxm_mcp_session_cookie),
        oauth_token
    )
    if x_inxm_mcp_session is None:
        logger.warning("[Session] Session header missing on close.")
        raise HTTPException(status_code=400, detail="Session header missing")
    mcp_task = sessions.pop(x_inxm_mcp_session, None)
    if not mcp_task:
        logger.warning(f"[Session] Session not found on close: {x_inxm_mcp_session}")
        raise HTTPException(status_code=404, detail="Session not found")
    await mcp_task.stop()
    logger.debug(f"[Session] Session closed: {x_inxm_mcp_session}")
    return {"status": "closed"}
