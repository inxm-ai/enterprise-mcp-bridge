import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from app.tgi.behaviours.well_planned_orchestrator import WellPlannedOrchestrator
from app.tgi.behaviours.todos.todo_manager import TodoItem, TodoManager
from app.tgi.models import ChatCompletionRequest, Message, MessageRole

@pytest.mark.asyncio
async def test_tool_selection_fallback_removed():
    """
    Verify that if a todo requests a tool that is not available,
    we proceed WITHOUT tools, rather than falling back to all tools.
    """
    # Setup
    llm_client = AsyncMock()
    prompt_service = AsyncMock()
    tool_service = AsyncMock()
    
    # Mock the chat callable
    stream_chat_mock = MagicMock()
    # It returns an async generator
    async def async_gen(*args, **kwargs):
        yield "data: [DONE]\n\n"
    stream_chat_mock.return_value = async_gen()

    orchestrator = WellPlannedOrchestrator(
        llm_client=llm_client,
        prompt_service=prompt_service,
        tool_service=tool_service,
        non_stream_chat_with_tools_callable=AsyncMock(),
        stream_chat_with_tools_callable=stream_chat_mock,
        tool_resolution=MagicMock(),
        logger_obj=MagicMock(),
    )

    # Define available tools
    tool_a = MagicMock()
    tool_a.function.name = "toolA"
    available_tools = [tool_a]

    # Define a todo that requests a NON-EXISTENT tool
    todo = TodoItem(
        id="t1",
        name="step1",
        goal="goal",
        needed_info=None,
        tools=["toolB"] # toolB is not in available_tools
    )
    
    todo_manager = TodoManager()
    todo_manager.add_todos([todo])

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="hi")],
        model="test",
        stream=True
    )

    # Run the streaming flow
    # We need to mock _stream_chat_with_tools to capture the tools passed to it
    
    # We can't easily mock the internal method call from outside without patching,
    # but we passed the callable in __init__.
    # However, _well_planned_streaming calls self._stream_chat_with_tools, which is the callable.
    
    # We need to iterate the generator to trigger execution
    gen = orchestrator._well_planned_streaming(
        todo_manager,
        None, # session
        request,
        available_tools,
        None, # access_token
        None, # span
    )
    
    async for _ in gen:
        pass

    # Verify what was passed to stream_chat_with_tools
    # args: session, focused_messages, filtered_tools, focused_request, access_token, span
    call_args = stream_chat_mock.call_args
    assert call_args is not None
    
    filtered_tools_arg = call_args[0][2]
    
    # CRITICAL CHECK: filtered_tools should be EMPTY
    # If the fallback was present, it would be equal to available_tools ([tool_a])
    assert filtered_tools_arg == [], f"Expected empty tools, got {filtered_tools_arg}"

@pytest.mark.asyncio
async def test_tool_selection_matching():
    """
    Verify that if a todo requests an AVAILABLE tool, it is passed.
    """
    # Setup
    llm_client = AsyncMock()
    prompt_service = AsyncMock()
    tool_service = AsyncMock()
    
    stream_chat_mock = MagicMock()
    async def async_gen(*args, **kwargs):
        yield "data: [DONE]\n\n"
    stream_chat_mock.return_value = async_gen()

    orchestrator = WellPlannedOrchestrator(
        llm_client=llm_client,
        prompt_service=prompt_service,
        tool_service=tool_service,
        non_stream_chat_with_tools_callable=AsyncMock(),
        stream_chat_with_tools_callable=stream_chat_mock,
        tool_resolution=MagicMock(),
        logger_obj=MagicMock(),
    )

    tool_a = MagicMock()
    tool_a.function.name = "toolA"
    available_tools = [tool_a]

    todo = TodoItem(
        id="t1",
        name="step1",
        goal="goal",
        needed_info=None,
        tools=["toolA"] # toolA IS available
    )
    
    todo_manager = TodoManager()
    todo_manager.add_todos([todo])

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="hi")],
        model="test",
        stream=True
    )

    gen = orchestrator._well_planned_streaming(
        todo_manager,
        None,
        request,
        available_tools,
        None,
        None,
    )
    
    async for _ in gen:
        pass

    call_args = stream_chat_mock.call_args
    filtered_tools_arg = call_args[0][2]
    
    assert len(filtered_tools_arg) == 1
    assert filtered_tools_arg[0].function.name == "toolA"
