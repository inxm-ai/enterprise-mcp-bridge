import json
import re
import time
from unittest.mock import AsyncMock

import pytest

from app.tgi.services.proxied_tgi_service import ProxiedTGIService
from app.tgi.behaviours.well_planned_orchestrator import WellPlannedOrchestrator
from app.tgi.behaviours.todos.todo_manager import TodoManager, TodoItem
from app.tgi.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    MessageRole,
    Usage,
)


def configure_plan_stream(service, todos, intent_payload=None):
    todos_payload = intent_payload or json.dumps(
        {"intent": "plan", "todos": todos}, ensure_ascii=False
    )
    chunk_payload = json.dumps(
        {"choices": [{"delta": {"content": todos_payload}, "index": 0}]}
    )
    plan_chunks = [
        f"data: {chunk_payload}" + "\n\n",
        "data: [DONE]" + "\n\n",
    ]

    async def plan_generator():
        for chunk in plan_chunks:
            yield chunk

    def fake_stream_completion(llm_request, access_token, span):
        return plan_generator()

    service.llm_client.stream_completion = fake_stream_completion
    service.llm_client.non_stream_completion = AsyncMock(
        side_effect=AssertionError(
            "non_stream_completion should not be used when streaming the todo plan"
        )
    )


def test_subtask_instructions_allow_multiple_tool_calls():
    class DummyLLM:
        def create_completion_id(self):
            return "dummy"

        def create_usage_stats(self):
            return {}

    orchestrator = WellPlannedOrchestrator(
        llm_client=DummyLLM(),
        prompt_service=None,
        tool_service=None,
        non_stream_chat_with_tools_callable=lambda *a, **k: None,
        stream_chat_with_tools_callable=lambda *a, **k: None,
        tool_resolution=None,
        logger_obj=None,
        model_name="test-model",
    )

    todo_manager = TodoManager()
    todo_manager.add_todos(
        [
            TodoItem(
                id="t1",
                name="step",
                goal="Do work",
                needed_info=None,
                tools=["tool-a", "tool-b"],
            )
        ]
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Do the task")],
        model="test-model",
        stream=False,
        tool_choice="auto",
    )

    focused_messages, _ = orchestrator._build_focused_request(
        todo_manager=todo_manager,
        todo=todo_manager.list_todos()[0],
        base_request=request,
        original_messages=request.messages,
        is_final_multistep_todo=False,
        is_final_step=False,
        root_goal="Do the task",
    )

    system_content = focused_messages[0].content or ""
    assert "Prefer smaller subtasks" in system_content
    assert "multiple tool calls are allowed" in system_content
    assert "At most one tool call" not in system_content


def test_focused_request_discourages_user_questions_when_tools_available():
    class DummyLLM:
        def create_completion_id(self):
            return "dummy"

        def create_usage_stats(self):
            return {}

    orchestrator = WellPlannedOrchestrator(
        llm_client=DummyLLM(),
        prompt_service=None,
        tool_service=None,
        non_stream_chat_with_tools_callable=lambda *a, **k: None,
        stream_chat_with_tools_callable=lambda *a, **k: None,
        tool_resolution=None,
        logger_obj=None,
        model_name="test-model",
    )

    todo_manager = TodoManager()
    todo_manager.add_todos(
        [
            TodoItem(
                id="t1",
                name="pick-team",
                goal="Identify the team",
                needed_info=None,
                tools=["select-from-tool-response"],
            )
        ]
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Do the task")],
        model="test-model",
        stream=False,
        tool_choice="auto",
    )

    focused_messages, _ = orchestrator._build_focused_request(
        todo_manager=todo_manager,
        todo=todo_manager.list_todos()[0],
        base_request=request,
        original_messages=request.messages,
        is_final_multistep_todo=False,
        is_final_step=False,
        root_goal="Do the task",
    )

    system_content = focused_messages[0].content or ""
    assert "Do not ask the user" in system_content
    assert "select-from-tool-response" in system_content


@pytest.mark.asyncio
async def test_well_planned_inserts_subtasks_from_step():
    service = ProxiedTGIService()

    todos = [
        {
            "id": "t1",
            "name": "parent",
            "goal": "Parent step",
            "needed_info": None,
            "tools": [],
        },
        {
            "id": "t2",
            "name": "final-answer",
            "goal": "Final answer",
            "needed_info": None,
            "tools": [],
        },
    ]

    configure_plan_stream(service, todos)

    call_goals = []
    captured_messages = []

    async def fake_non_stream_chat(session, messages, tools, req, access_token, span):
        captured_messages.append(messages)
        system = messages[0].content or ""
        match = re.search(r"Current Goal:\s*(.*)", system)
        goal = match.group(1).strip() if match else system
        call_goals.append(goal)

        if goal == "Parent step":
            return {
                "intent": "subtasks",
                "reason": "Need smaller steps",
                "subtasks": [
                    {
                        "id": "s1",
                        "name": "sub-1",
                        "goal": "Subtask one",
                        "needed_info": None,
                        "tools": [],
                    },
                    {
                        "id": "s2",
                        "name": "sub-2",
                        "goal": "Subtask two",
                        "needed_info": None,
                        "tools": [],
                    },
                ],
                "close_followups": False,
            }

        return ChatCompletionResponse(
            id="chatcmpl-test",
            created=int(time.time()),
            model=req.model or "test-model",
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=f"Result for {goal}",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    service._non_stream_chat_with_tools = fake_non_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Do the thing")],
        model="test-model",
        stream=False,
        tool_choice="auto",
    )

    await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    assert call_goals[:3] == ["Parent step", "Subtask one", "Subtask two"]
    assert call_goals[3].startswith(
        "Use only the existing information to answer the user's request:"
    )

    # Subtask calls should include the parent summary in their context.
    subtask_messages = captured_messages[1]
    assert any(
        "Created 2 subtasks" in (m.content or "")
        for m in subtask_messages
        if m.role == MessageRole.ASSISTANT
    )


@pytest.mark.asyncio
async def test_well_planned_expands_placeholder_tools_for_subtasks():
    service = ProxiedTGIService()

    todos = [
        {
            "id": "t1",
            "name": "parent",
            "goal": "Parent step",
            "needed_info": None,
            "tools": [],
        },
        {
            "id": "t2",
            "name": "final-answer",
            "goal": "Final answer",
            "needed_info": None,
            "tools": [],
        },
    ]

    configure_plan_stream(service, todos)

    available_tools = [
        {
            "type": "function",
            "function": {"name": "list-joined-teams", "description": "..."},
        },
        {
            "type": "function",
            "function": {"name": "list-team-channels", "description": "..."},
        },
    ]

    captured_tools = {}

    async def fake_non_stream_chat(session, messages, tools, req, access_token, span):
        system = messages[0].content or ""
        match = re.search(r"Current Goal:\s*(.*)", system)
        goal = match.group(1).strip() if match else system

        if goal == "Parent step":
            return {
                "intent": "subtasks",
                "reason": "Need tool calls",
                "subtasks": [
                    {
                        "id": "s1",
                        "name": "sub-1",
                        "goal": "Subtask one",
                        "needed_info": None,
                        "tools": ["call_tool"],
                    },
                    {
                        "id": "s2",
                        "name": "sub-2",
                        "goal": "Subtask two",
                        "needed_info": None,
                        "tools": ["call_tool"],
                    },
                ],
                "close_followups": False,
            }

        if goal == "Subtask one":
            captured_tools["sub-1"] = tools
        if goal == "Subtask two":
            captured_tools["sub-2"] = tools

        return ChatCompletionResponse(
            id="chatcmpl-test",
            created=int(time.time()),
            model=req.model or "test-model",
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=f"Result for {goal}",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    service._non_stream_chat_with_tools = fake_non_stream_chat

    class DummySession:
        async def list_tools(self):
            return available_tools

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Do the thing")],
        model="test-model",
        stream=False,
        tool_choice="auto",
    )

    await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    expected_names = [t["function"]["name"] for t in available_tools]
    if "select-from-tool-response" not in expected_names:
        expected_names.append("select-from-tool-response")
    if "describe_tool" not in expected_names:
        expected_names.append("describe_tool")
    for key in ("sub-1", "sub-2"):
        tool_names = [
            (t.get("function") or {}).get("name") for t in (captured_tools[key] or [])
        ]
        assert tool_names == expected_names


@pytest.mark.asyncio
async def test_well_planned_parses_subtasks_from_code_block():
    service = ProxiedTGIService()

    todos = [
        {
            "id": "t1",
            "name": "parent",
            "goal": "Parent step",
            "needed_info": None,
            "tools": [],
        },
        {
            "id": "t2",
            "name": "final-answer",
            "goal": "Final answer",
            "needed_info": None,
            "tools": [],
        },
    ]

    configure_plan_stream(service, todos)

    call_goals = []

    async def fake_non_stream_chat(session, messages, tools, req, access_token, span):
        system = messages[0].content or ""
        match = re.search(r"Current Goal:\s*(.*)", system)
        goal = match.group(1).strip() if match else system
        call_goals.append(goal)

        if goal == "Parent step":
            subtask_payload = {
                "intent": "subtasks",
                "reason": "Need smaller steps",
                "subtasks": [
                    {
                        "id": "s1",
                        "name": "sub-1",
                        "goal": "Subtask one",
                        "needed_info": None,
                        "tools": [],
                    },
                    {
                        "id": "s2",
                        "name": "sub-2",
                        "goal": "Subtask two",
                        "needed_info": None,
                        "tools": [],
                    },
                ],
                "close_followups": False,
            }
            content = (
                "Here are the subtasks:\n```json\n"
                + json.dumps(subtask_payload, ensure_ascii=False)
                + "\n```\n"
            )
            return ChatCompletionResponse(
                id="chatcmpl-test",
                created=int(time.time()),
                model=req.model or "test-model",
                choices=[
                    Choice(
                        index=0,
                        message=Message(
                            role=MessageRole.ASSISTANT,
                            content=content,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )

        return ChatCompletionResponse(
            id="chatcmpl-test",
            created=int(time.time()),
            model=req.model or "test-model",
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=f"Result for {goal}",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    service._non_stream_chat_with_tools = fake_non_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Do the thing")],
        model="test-model",
        stream=False,
        tool_choice="auto",
    )

    await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    assert call_goals[:3] == ["Parent step", "Subtask one", "Subtask two"]


@pytest.mark.asyncio
async def test_well_planned_streaming_inserts_subtasks_with_tool_results():
    service = ProxiedTGIService()

    todos = [
        {
            "id": "t1",
            "name": "parent",
            "goal": "Parent step",
            "needed_info": None,
            "tools": [],
        },
        {
            "id": "t2",
            "name": "final-answer",
            "goal": "Final answer",
            "needed_info": None,
            "tools": [],
        },
    ]

    configure_plan_stream(service, todos)

    call_goals = []

    async def fake_stream_chat(session, messages, tools, req, access_token, span):
        system = messages[0].content or ""
        match = re.search(r"Current Goal:\s*(.*)", system)
        goal = match.group(1).strip() if match else system
        call_goals.append(goal)

        if goal == "Parent step":
            subtask_payload = {
                "intent": "subtasks",
                "reason": "Need smaller steps",
                "subtasks": [
                    {
                        "id": "s1",
                        "name": "sub-1",
                        "goal": "Subtask one",
                        "needed_info": None,
                        "tools": [],
                    },
                    {
                        "id": "s2",
                        "name": "sub-2",
                        "goal": "Subtask two",
                        "needed_info": None,
                        "tools": [],
                    },
                ],
                "close_followups": False,
            }
            content = (
                "Perfect! I found the integration.\n```json\n"
                + json.dumps(subtask_payload, ensure_ascii=False)
                + "\n```\n"
            )
            tool_result_event = {
                "choices": [
                    {
                        "delta": {
                            "tool_result": {
                                "name": "list-chats",
                                "content": "[]",
                            }
                        },
                        "index": 0,
                    }
                ]
            }
            yield "data: " + json.dumps(tool_result_event) + "\n\n"
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"content": content}, "index": 0}]}
            ) + "\n\n"
            yield "data: [DONE]\n\n"
            return

        yield "data: " + json.dumps(
            {"choices": [{"delta": {"content": f"Result for {goal}"}, "index": 0}]}
        ) + "\n\n"
        yield "data: [DONE]\n\n"

    service._stream_chat_with_tools = fake_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Do the thing")],
        model="test-model",
        stream=True,
        tool_choice="auto",
    )

    gen = await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    async for _chunk in gen:
        pass

    assert call_goals[:3] == ["Parent step", "Subtask one", "Subtask two"]
    assert call_goals[3].startswith(
        "Use only the existing information to answer the user's request:"
    )


@pytest.mark.asyncio
async def test_well_planned_relaxes_subtask_tools_from_select_only():
    service = ProxiedTGIService()

    todos = [
        {
            "id": "t1",
            "name": "parent",
            "goal": "Parent step",
            "needed_info": None,
            "tools": ["select-from-tool-response"],
        },
        {
            "id": "t2",
            "name": "final-answer",
            "goal": "Final answer",
            "needed_info": None,
            "tools": [],
        },
    ]

    configure_plan_stream(service, todos)

    available_tools = [
        {
            "type": "function",
            "function": {"name": "call_tool", "description": "..."},
        },
        {
            "type": "function",
            "function": {"name": "select-from-tool-response", "description": "..."},
        },
        {
            "type": "function",
            "function": {"name": "list-team-channels", "description": "..."},
        },
    ]

    captured_tools = {}

    async def fake_non_stream_chat(session, messages, tools, req, access_token, span):
        system = messages[0].content or ""
        match = re.search(r"Current Goal:\s*(.*)", system)
        goal = match.group(1).strip() if match else system

        if goal == "Parent step":
            return {
                "intent": "subtasks",
                "reason": "Need a tool call",
                "subtasks": [
                    {
                        "id": "s1",
                        "name": "list-channels",
                        "goal": "List channels for the team",
                        "needed_info": None,
                        "tools": ["select-from-tool-response"],
                    }
                ],
                "close_followups": False,
            }

        if goal == "List channels for the team":
            captured_tools["list-channels"] = tools

        return ChatCompletionResponse(
            id="chatcmpl-test",
            created=int(time.time()),
            model=req.model or "test-model",
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=f"Result for {goal}",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    service._non_stream_chat_with_tools = fake_non_stream_chat

    class DummySession:
        async def list_tools(self):
            return available_tools

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Do the thing")],
        model="test-model",
        stream=False,
        tool_choice="auto",
    )

    await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    tool_names = [
        (t.get("function") or {}).get("name")
        for t in (captured_tools.get("list-channels") or [])
    ]
    assert "call_tool" in tool_names
