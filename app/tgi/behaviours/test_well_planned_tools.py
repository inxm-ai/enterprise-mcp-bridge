import pytest
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
        tools=["toolB"],  # toolB is not in available_tools
    )

    todo_manager = TodoManager()
    todo_manager.add_todos([todo])

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="hi")],
        model="test",
        stream=True,
    )

    # Run the streaming flow
    # We need to mock _stream_chat_with_tools to capture the tools passed to it

    # We can't easily mock the internal method call from outside without patching,
    # but we passed the callable in __init__.
    # However, _well_planned_streaming calls self._stream_chat_with_tools, which is the callable.

    # We need to iterate the generator to trigger execution
    gen = orchestrator._well_planned_streaming(
        todo_manager,
        None,  # session
        request,
        available_tools,
        None,  # access_token
        None,  # span
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
        tools=["toolA"],  # toolA IS available
    )

    todo_manager = TodoManager()
    todo_manager.add_todos([todo])

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="hi")],
        model="test",
        stream=True,
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


@pytest.mark.asyncio
async def test_tool_call_indices_do_not_skip_the_remaining_todos():
    """
    A todo whose stream carries tool calls with a high index must not move
    the todo cursor: every todo runs and the final answer is streamed.
    """
    import json

    calls = []

    def stream_chat(session, messages, tools, request, access_token, span):
        step = len(calls)
        calls.append(step)

        async def gen():
            if step == 0:
                tool_call = {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 5,
                                        "id": "call-5",
                                        "function": {
                                            "name": "toolA",
                                            "arguments": "{}",
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                }
                yield f"data: {json.dumps(tool_call)}\n\n"
            content = f"result of step {step}"
            chunk = {"choices": [{"index": 0, "delta": {"content": content}}]}
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return gen()

    orchestrator = WellPlannedOrchestrator(
        llm_client=AsyncMock(),
        prompt_service=AsyncMock(),
        tool_service=AsyncMock(),
        non_stream_chat_with_tools_callable=AsyncMock(),
        stream_chat_with_tools_callable=stream_chat,
        tool_resolution=MagicMock(),
        logger_obj=MagicMock(),
    )

    tool_a = MagicMock()
    tool_a.function.name = "toolA"
    todo_manager = TodoManager()
    todo_manager.add_todos(
        [
            TodoItem(id="1", name="explore", goal="g1", tools=["toolA"]),
            TodoItem(id="2", name="inspect", goal="g2", tools=["toolA"]),
            TodoItem(id="3", name="final-answer", goal="g3", tools=[]),
        ]
    )
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="hi")],
        model="test",
        stream=True,
    )

    out = ""
    async for chunk in orchestrator._well_planned_streaming(
        todo_manager, None, request, [tool_a], None, None
    ):
        out += chunk

    assert calls == [0, 1, 2]
    assert "result of step 2" in out
    assert all(t.state.value == "DONE" for t in todo_manager.list_todos())
