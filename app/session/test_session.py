import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.session.session import (
    try_get_session_id,
    session_id,
    MCPSessionBase,
    MCPLocalSessionTask,
)
import asyncio
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("test")


# Test for try_get_session_id
def test_try_get_session_id():
    assert try_get_session_id("header", None, None) == "header"
    assert try_get_session_id(None, "cookie", None) == "cookie"
    assert try_get_session_id(None, None, "args") == "args"
    assert try_get_session_id(None, None, None) is None


# Test for session_id
def test_session_id():
    assert session_id("base", "token") == "base:token"
    assert session_id("base", None) == "base"
    assert session_id("", "token") == None
    assert session_id("", None) == None
    assert session_id(None, None) == None


# Test for MCPSessionBase
@pytest.mark.asyncio
async def test_mcpsessionbase():
    logger.debug("[test_mcpsessionbase] Starting test.")

    class TestSession(MCPSessionBase):
        async def run(self):
            logger.debug("[TestSession] Run method started.")
            while True:
                req = await self.request_queue.get()
                logger.debug(f"[TestSession] Received request: {req}")
                if req == "close":
                    logger.debug("[TestSession] Stop request received. Exiting loop.")
                    break
                await self.response_queue.put(f"processed: {req}")

    logger.debug("[test_mcpsessionbase] Creating TestSession instance.")
    session = TestSession(server_params=None)
    session.start()

    logger.debug("[test_mcpsessionbase] Sending test_request.")
    response = await session.request("test_request")
    logger.debug(f"[test_mcpsessionbase] Received response: {response}")
    assert response == "processed: test_request"

    logger.debug("[test_mcpsessionbase] Sending stop request.")
    await session.stop()


# Test for MCPLocalSessionTask
@pytest.mark.asyncio
@patch("app.session.session.mcp_session")
async def test_mcplocalsessiontask(mock_mcp_session):
    mock_session = AsyncMock()
    mock_mcp_session.return_value.__aenter__.return_value = mock_session

    task = MCPLocalSessionTask(server_params=None)
    task.start()

    # Test ping
    response = await task.request("ping")
    assert response == "pong"

    # Test list_tools
    mock_session.list_tools.return_value = ["tool1", "tool2"]
    response = await task.request("list_tools")
    assert response == ["tool1", "tool2"]

    # Test run_tool
    mock_session.call_tool.return_value = {"result": "success"}
    response = await task.request(
        {"action": "run_tool", "tool_name": "tool1", "args": {"param": "value"}}
    )
    assert response == {"result": "success"}

    # test invalid tool
    mock_session.call_tool.side_effect = Exception("tool not found")
    response = await task.request(
        {"action": "run_tool", "tool_name": "invalid_tool", "args": {"param": "value"}}
    )
    assert response == {"error": "tool not found"}

    await task.stop()
