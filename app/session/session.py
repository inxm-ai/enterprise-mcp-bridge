import asyncio
import datetime
import logging
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from abc import ABC, abstractmethod
from typing import Optional
from ..utils.exception_logging import log_exception_with_details


def try_get_session_id(
    x_inxm_mcp_session_header: Optional[str],
    x_inxm_mcp_session_cookie: Optional[str],
    x_inxm_mcp_session_args: Optional[str] = None,
) -> Optional[str]:
    if x_inxm_mcp_session_header:
        return x_inxm_mcp_session_header
    if x_inxm_mcp_session_cookie:
        return x_inxm_mcp_session_cookie
    if x_inxm_mcp_session_args:
        return x_inxm_mcp_session_args
    return None


def session_id(base_id: str, oauth_token: Optional[str] = None) -> str:
    if base_id and oauth_token:
        return f"{base_id}:{oauth_token}"
    if not base_id:
        return None
    return base_id


@asynccontextmanager
async def mcp_session(server_params: StdioServerParameters):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


logger = logging.getLogger("uvicorn.error")


class MCPSessionBase(ABC):
    def __init__(self, server_params):
        self.server_params = server_params
        self.request_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        self._task = None
        self._ping_task = None  # Reference to the ping task
        self._stop_event = asyncio.Event()  # Added stop event

    @abstractmethod
    async def run(self):
        pass  # pragma: no cover

    def start(self):
        logger.debug("[{}] Starting session task.".format(self.__class__.__name__))
        self._task = asyncio.create_task(self.run())

        async def ping_task():
            while self._stop_event.is_set():
                await asyncio.sleep(1)
                logger.debug(
                    "[{}] Sending periodic ping.".format(self.__class__.__name__)
                )
                await self.request_queue.put("ping")

        self._ping_task = asyncio.create_task(ping_task())  # Store the ping task

    async def stop(self):
        logger.debug("[{}] Stopping session task.".format(self.__class__.__name__))
        self._stop_event.set()
        await self.request_queue.put("close")
        if self._task:
            logger.debug(
                "[{}] Waiting for session task to finish.".format(
                    self.__class__.__name__
                )
            )
            await self._task

    async def request(self, req):
        logger.debug(f"[{self.__class__.__name__}] Received request: {req}")
        await self.request_queue.put(req)
        response = await self.response_queue.get()
        logger.debug(f"[{self.__class__.__name__}] Response: {response}")
        return response


class MCPLocalSessionTask(MCPSessionBase):
    async def run(self):
        logger.info("[MCPLocalSessionTask] Session task started.")
        last_trigger = datetime.datetime.now()
        try:
            async with mcp_session(self.server_params) as session:
                logger.info("[MCPLocalSessionTask] MCP session established.")
                while True:
                    req = await self.request_queue.get()
                    logger.debug(f"[MCPLocalSessionTask] Processing request: {req}")
                    if req == "ping":
                        if (
                            last_trigger
                            and (datetime.datetime.now() - last_trigger).total_seconds()
                            > 60
                        ):
                            logger.info(
                                "[MCPLocalSessionTask] Session inactive for too long, closing."
                            )
                            await self.response_queue.put("session_closed")
                            break
                        last_trigger = datetime.datetime.now()
                        await self.response_queue.put("pong")
                        continue
                    if req == "close":
                        logger.debug(
                            "[MCPLocalSessionTask] Received close request. Shutting down session."
                        )
                        break
                    if req == "list_tools":
                        logger.debug("[MCPLocalSessionTask] Listing available tools.")
                        try:
                            result = await session.list_tools()
                            await self.response_queue.put(result)
                        except Exception as e:
                            log_exception_with_details(
                                logger, "[MCPLocalSessionTask]", e
                            )
                            await self.response_queue.put({"error": str(e)})
                    elif isinstance(req, dict) and req.get("action") == "run_tool":
                        tool_name = req["tool_name"]
                        args = req.get("args", {})
                        logger.info(
                            f"[MCPLocalSessionTask] Running tool: {tool_name} with args: {args}"
                        )
                        try:
                            result = await session.call_tool(tool_name, **args)
                            await self.response_queue.put(result)
                        except Exception as e:
                            # Handle TaskGroup exceptions with multiple sub-exceptions
                            log_exception_with_details(
                                logger, "[MCPLocalSessionTask]", e
                            )
                            await self.response_queue.put({"error": str(e)})
                    else:
                        logger.warning(f"[MCPLocalSessionTask] Unknown request: {req}")
                        await self.response_queue.put({"error": "Unknown request"})
        except Exception as e:
            # Handle TaskGroup exceptions with multiple sub-exceptions
            log_exception_with_details(logger, "[MCPLocalSessionTask]", e)
        finally:
            logger.info("[MCPLocalSessionTask] Session task stopped.")
