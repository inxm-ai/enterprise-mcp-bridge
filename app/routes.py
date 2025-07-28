
import logging
from fastapi import APIRouter, HTTPException, Header, Cookie, Request
from fastapi.responses import JSONResponse
from typing import Optional, Dict
import uuid
import os
from mcp import StdioServerParameters
from .session import MCPLocalSessionTask, mcp_session
from .session_manager import session_manager
from .models import RunToolRequest, RunToolsResult
import sys

router = APIRouter()
sessions = session_manager()

logger = logging.getLogger("uvicorn.error")

MCP_BASE_PATH = os.environ.get("MCP_BASE_PATH", "")

# Enhanced: Support MCP_SERVER_COMMAND env variable (takes precedence), else sys.argv, else default
def get_server_params():
    env_command = os.environ.get("MCP_SERVER_COMMAND")
    env = os.environ.copy()
    if env_command:
        # Split the env variable into command and args (simple shell-like split)
        import shlex
        parts = shlex.split(env_command)
        command = parts[0]
        cmd_args = parts[1:]
        logger.info(f"Server-Params from MCP_SERVER_COMMAND: command={command}, args={cmd_args}")
        return StdioServerParameters(command=command, args=cmd_args, env=env)

    # Fallback: parse sys.argv for --
    args = {}
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        args["command"] = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else None
        args["args"] = sys.argv[idx + 2:] if len(sys.argv) > idx + 2 else []
        command = args["command"] or "python"
        cmd_args = args["args"] or [os.path.join(os.path.dirname(__file__), "..", "mcp", "server.py")]
        logger.info(f"Server-Params from sys.argv: command={command}, args={cmd_args}")
        return StdioServerParameters(command=command, args=cmd_args, env=env)

    # Default
    command = "python"
    cmd_args = [os.path.join(os.path.dirname(__file__), "..", "mcp", "server.py")]
    logger.info(f"Server-Params default: command={command}, args={cmd_args}")
    return StdioServerParameters(command=command, args=cmd_args, env=env)

server_params = get_server_params()

def try_get_session_id(
    x_inxm_mcp_session_header: Optional[str], 
    x_inxm_mcp_session_cookie: Optional[str],
    x_inxm_mcp_session_args: Optional[str] = None
) -> Optional[str]:
    if x_inxm_mcp_session_header:
        return x_inxm_mcp_session_header
    if x_inxm_mcp_session_cookie:
        return x_inxm_mcp_session_cookie
    if x_inxm_mcp_session_args:
        return x_inxm_mcp_session_args
    return None

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
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias="x-inxm-mcp-session"),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias="x-inxm-mcp-session")
):
    x_inxm_mcp_session = try_get_session_id(x_inxm_mcp_session_header, x_inxm_mcp_session_cookie)
    logger.info(f"[Tools] Listing tools. Session: {x_inxm_mcp_session}")
    if x_inxm_mcp_session is None:
        async with mcp_session(server_params) as session:
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

async def decorate_with_oauth_token(session, tool_name, args: Optional[Dict], oauth_token: Optional[str]) -> Dict:
    tools = await session.list_tools()
    tool_info = next((tool for tool in tools.tools if tool.name == tool_name), None)

    if args is None:
        args = {}
    if oauth_token and tool_info and "oauth_token" in tool_info.inputSchema:
        args['oauth_token'] = oauth_token
        logger.info(f"[Tool-Call] Tool {tool_name} will be called with oauth_token.")
    elif not oauth_token and tool_info and "oauth_token" in tool_info.inputSchema:
        logger.warning(f"[Tool-Call] Tool {tool_name} requires oauth_token but none provided.")
        raise HTTPException(status_code=401, detail="Tool requires oauth_token but none provided.")
    else:
        logger.info(f"[Tool-Call] Tool {tool_name} does not require oauth_token.")
    return args

@router.post("/tools/{tool_name}")
async def run_tool(
    tool_name: str,
    request: Request,
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias="x-inxm-mcp-session"),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias="x-inxm-mcp-session"),
    oauth_token: Optional[str] = Cookie(None, alias="_oauth2_proxy"),
    args: Optional[Dict] = None, 
    ):
    x_inxm_mcp_session = try_get_session_id(x_inxm_mcp_session_header, x_inxm_mcp_session_cookie, args.get('inxm-session', None) if args else None)
    if not oauth_token:
        oauth_token = request.cookies.get("_oauth2_proxy", "")
    if args and 'inxm-session' in args:
        args = dict(args)
        args.pop('inxm-session')
    logger.info(f"[Tool-Call] Tool call: {tool_name}, Session: {x_inxm_mcp_session}, Args: {args}")
    if x_inxm_mcp_session is None:
        async with mcp_session(server_params) as session:
            decorated_args = await decorate_with_oauth_token(session, tool_name, args, oauth_token)
            result = await session.call_tool(tool_name, decorated_args)
            result = RunToolsResult(result)
    else:
        session_id = x_inxm_mcp_session
        mcp_task = sessions.get(session_id)
        if not mcp_task:
            logger.warning(f"[Tool-Call] Session not found: {session_id}")
            raise HTTPException(status_code=404, detail="Session not found. It might have expired, please start a new.")
        else:
            decorated_args = await decorate_with_oauth_token(session, tool_name, args, oauth_token)
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
async def start_session():
    session_id = str(uuid.uuid4())
    mcp_task = MCPLocalSessionTask(server_params)
    mcp_task.start()
    sessions.set(session_id, mcp_task)
    logger.debug(f"[Session] New session started: {session_id}")
    response = JSONResponse(content={"x-inxm-mcp-session": session_id})
    response.set_cookie(key="x-inxm-mcp-session", value=session_id, httponly=True, samesite="lax")
    return response

@router.post("/session/close")
async def close_session(
    x_inxm_mcp_session_header: Optional[str] = Header(None, alias="x-inxm-mcp-session"),
    x_inxm_mcp_session_cookie: Optional[str] = Cookie(None, alias="x-inxm-mcp-session")
):
    x_inxm_mcp_session = try_get_session_id(x_inxm_mcp_session_header, x_inxm_mcp_session_cookie)
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
