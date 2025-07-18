import asyncio
import datetime
import logging
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from abc import ABC, abstractmethod

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

    @abstractmethod
    async def run(self):
        pass

    def start(self):
        logger.debug("[{}] Starting session task.".format(self.__class__.__name__))
        self._task = asyncio.create_task(self.run())
        async def ping_task():
            while True:
                await asyncio.sleep(10)
                logger.debug("[{}] Sending periodic ping.".format(self.__class__.__name__))
                await self.request_queue.put("ping")
        asyncio.create_task(ping_task())

    async def stop(self):
        logger.info("[{}] Stopping session task.".format(self.__class__.__name__))
        await self.request_queue.put("close")
        if self._task:
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
                        if last_trigger and (datetime.datetime.now() - last_trigger).total_seconds() > 60:
                            logger.info("[MCPLocalSessionTask] Session inactive for too long, closing.")
                            await self.response_queue.put("session_closed")
                            break
                        last_trigger = datetime.datetime.now()
                        await self.response_queue.put("pong")
                        continue
                    if req == "close":
                        logger.debug("[MCPLocalSessionTask] Received close request. Shutting down session.")
                        break
                    if req == "list_tools":
                        logger.debug("[MCPLocalSessionTask] Listing available tools.")
                        try:
                            result = await session.list_tools()
                            await self.response_queue.put(result)
                        except Exception as e:
                            logger.error(f"[MCPLocalSessionTask] Failed to list tools: {e}")
                            await self.response_queue.put({"error": str(e)})
                    elif isinstance(req, dict) and req.get("action") == "run_tool":
                        tool_name = req["tool_name"]
                        args = req.get("args", {})
                        logger.info(f"[MCPLocalSessionTask] Running tool: {tool_name} with args: {args}")
                        try:
                            result = await session.call_tool(tool_name, **args)
                            await self.response_queue.put(result)
                        except Exception as e:
                            logger.error(f"[MCPLocalSessionTask] Error running tool '{tool_name}': {e}")
                            await self.response_queue.put({"error": str(e)})
                    else:
                        logger.warning(f"[MCPLocalSessionTask] Unknown request: {req}")
                        await self.response_queue.put({"error": "Unknown request"})
        except Exception as e:
            logger.error(f"[MCPLocalSessionTask] Unhandled exception in session task: {e}", exc_info=True)
        finally:
            logger.info("[MCPLocalSessionTask] Session task stopped.")
