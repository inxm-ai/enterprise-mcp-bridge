import json
from types import SimpleNamespace
from typing import Any, AsyncGenerator

import pytest

from app.tgi.models import ChatCompletionRequest, MessageRole
from app.tgi.workflows.contextual_llm_helper import run_context_aware_llm_helper
from app.tgi.workflows.state import WorkflowExecutionState


class StubStateStore:
    def __init__(self, state: WorkflowExecutionState):
        self.state = state

    def get_or_create(self, execution_id: str, flow_id: str):
        return self.state


class OneTurnLLM:
    def stream_completion(
        self, request: ChatCompletionRequest, access_token: str, span: Any
    ) -> AsyncGenerator[str, None]:
        async def _gen():
            payload = {
                "choices": [{"delta": {"content": "Rendered question."}, "index": 0}]
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return _gen()


class ToolCallingLLM:
    def __init__(self):
        self.tool_call_id_seen = None

    def stream_completion(
        self, request: ChatCompletionRequest, access_token: str, span: Any
    ) -> AsyncGenerator[str, None]:
        has_tool_message = any(msg.role == MessageRole.TOOL for msg in request.messages)
        if not has_tool_message:

            async def _gen_first():
                payload = {
                    "choices": [
                        {
                            "delta": {
                                "content": "Need context.",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {
                                            "name": "get_workflow_context",
                                            "arguments": json.dumps(
                                                {
                                                    "operation": "get_value",
                                                    "path": "agents.select_tools.selected_tools",
                                                }
                                            ),
                                        },
                                    }
                                ],
                            },
                            "index": 0,
                        }
                    ]
                }
                yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

            return _gen_first()

        for message in request.messages:
            if message.role == MessageRole.TOOL:
                self.tool_call_id_seen = message.tool_call_id

        async def _gen_second():
            payload = {
                "choices": [
                    {
                        "delta": {
                            "content": "I found integrations: list_commits and search_repositories."
                        },
                        "index": 0,
                    }
                ]
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return _gen_second()


class EndlessToolLLM:
    def stream_completion(
        self, request: ChatCompletionRequest, access_token: str, span: Any
    ) -> AsyncGenerator[str, None]:
        async def _gen():
            payload = {
                "choices": [
                    {
                        "delta": {
                            "content": "Still need more context.",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_loop",
                                    "function": {
                                        "name": "get_workflow_context",
                                        "arguments": '{"operation":"summary"}',
                                    },
                                }
                            ],
                        },
                        "index": 0,
                    }
                ]
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return _gen()


@pytest.mark.asyncio
async def test_contextual_helper_one_turn_without_tools():
    llm = OneTurnLLM()
    request = ChatCompletionRequest(model="test-model", messages=[])

    result = await run_context_aware_llm_helper(
        llm_client=llm,
        base_request=request,
        access_token=None,
        span=None,
        system_prompt="SYSTEM",
        user_payload="USER",
        state_store=None,
        execution_id=None,
        max_turns=2,
    )

    assert result.text == "Rendered question."
    assert result.used_tools is False
    assert result.turns == 1
    assert result.stopped_by_max_turns is False


@pytest.mark.asyncio
async def test_contextual_helper_lazy_context_tool_two_turns_with_tool_call_id():
    llm = ToolCallingLLM()
    state = WorkflowExecutionState(
        execution_id="exec-1",
        flow_id="flow",
        context={
            "agents": {
                "select_tools": {
                    "selected_tools": [
                        {"tool_name": "list_commits"},
                        {"tool_name": "search_repositories"},
                    ]
                }
            }
        },
    )
    store = StubStateStore(state)
    request = ChatCompletionRequest(model="test-model", messages=[])

    result = await run_context_aware_llm_helper(
        llm_client=llm,
        base_request=request,
        access_token=None,
        span=None,
        system_prompt="SYSTEM",
        user_payload="USER",
        state_store=store,
        execution_id="exec-1",
        max_turns=2,
    )

    assert "list_commits" in result.text
    assert result.used_tools is True
    assert result.turns == 2
    assert result.stopped_by_max_turns is False
    assert llm.tool_call_id_seen == "call_1"


@pytest.mark.asyncio
async def test_contextual_helper_max_turn_protection():
    llm = EndlessToolLLM()
    state = WorkflowExecutionState(execution_id="exec-2", flow_id="flow", context={})
    store = StubStateStore(state)
    request = ChatCompletionRequest(model="test-model", messages=[])

    result = await run_context_aware_llm_helper(
        llm_client=llm,
        base_request=request,
        access_token=None,
        span=None,
        system_prompt="SYSTEM",
        user_payload="USER",
        state_store=store,
        execution_id="exec-2",
        max_turns=2,
    )

    assert result.turns == 2
    assert result.stopped_by_max_turns is True
