import json
from typing import Any, AsyncGenerator

import pytest

from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.workflows.feedback import FeedbackService
from app.tgi.workflows.state import WorkflowExecutionState


class StubStateStore:
    def __init__(self, state: WorkflowExecutionState):
        self.state = state

    def get_or_create(self, execution_id: str, flow_id: str):
        return self.state


class BudgetAwareLLM:
    def __init__(self):
        self.ask_calls = []
        self.stream_calls = []

    async def ask(
        self,
        base_prompt: str,
        base_request: ChatCompletionRequest,
        question: str = None,
        access_token: str = None,
        outer_span=None,
        assistant_statement: str = None,
    ) -> str:
        self.ask_calls.append(base_prompt)
        return "ASK_PATH"

    def stream_completion(
        self, request: ChatCompletionRequest, access_token: str, span: Any
    ) -> AsyncGenerator[str, None]:
        system_prompt = request.messages[0].content or ""
        self.stream_calls.append(system_prompt)
        has_tool = any(msg.role == MessageRole.TOOL for msg in request.messages)

        if not has_tool:

            async def _gen_first():
                payload = {
                    "choices": [
                        {
                            "delta": {
                                "content": "Need context.",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_fb_1",
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

            return _gen_first()

        if "USER_FEEDBACK_QUESTION" in system_prompt:
            content = "Question from helper"
        elif "USER_QUERY_SUMMARY" in system_prompt:
            content = "Rewritten request from helper"
        else:
            content = "RERUN"

        async def _gen_second():
            payload = {"choices": [{"delta": {"content": content}, "index": 0}]}
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return _gen_second()


@pytest.mark.asyncio
async def test_feedback_service_uses_context_helper_for_all_llm_methods_when_budget_exceeded():
    llm = BudgetAwareLLM()
    state = WorkflowExecutionState(
        execution_id="exec-feedback-svc",
        flow_id="flow",
        context={"agents": {"a": {"content": "x"}}},
    )
    store = StubStateStore(state)
    service = FeedbackService(llm_client=llm, state_store=store)
    service.SIZE_THRESHOLD = 1

    request = ChatCompletionRequest(model="test-model", messages=[])

    rendered = await service.render_feedback_question(
        ask_config={"question": "Render question", "expected_responses": []},
        agent_context={"result": {"foo": "bar"}},
        shared_context={"agents": {"a": {"content": "x"}}},
        request=request,
        access_token=None,
        span=None,
        execution_id="exec-feedback-svc",
    )
    assert rendered == "Question from helper"

    summarized = await service.summarize_user_query(
        base_query="original",
        feedback="yes",
        feedback_prompt="confirm?",
        request=request,
        access_token=None,
        span=None,
        execution_id="exec-feedback-svc",
    )
    assert summarized == "Rewritten request from helper"

    rerun = await service.should_rerun_feedback_agent(
        base_query="original",
        feedback="yes",
        feedback_prompt="confirm?",
        agent_name="agent",
        agent_context={"large": "x" * 2000},
        request=request,
        access_token=None,
        span=None,
        execution_id="exec-feedback-svc",
    )
    assert rerun is True

    assert llm.ask_calls == []
    assert any("USER_FEEDBACK_QUESTION" in call for call in llm.stream_calls)
    assert any("USER_QUERY_SUMMARY" in call for call in llm.stream_calls)
    assert any("FEEDBACK_RERUN_DECISION" in call for call in llm.stream_calls)


@pytest.mark.asyncio
async def test_feedback_service_keeps_fast_ask_path_for_small_context():
    llm = BudgetAwareLLM()
    service = FeedbackService(llm_client=llm, state_store=None)

    request = ChatCompletionRequest(model="test-model", messages=[])

    rendered = await service.render_feedback_question(
        ask_config={"question": "Render question", "expected_responses": []},
        agent_context={"content": "ok"},
        shared_context={"agents": {}},
        request=request,
        access_token=None,
        span=None,
        execution_id=None,
    )

    assert rendered == "ASK_PATH"
    assert llm.ask_calls
