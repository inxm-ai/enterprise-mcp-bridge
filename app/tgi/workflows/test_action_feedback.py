"""Machine action round trips over the feedback carrier and exposed returns.

Covers the three additive protocol pieces used by the planning-agent spikes:
``payload_from`` (agent → client action carrier), ``ask.kind == "action"``
correlation/consumption on the way back, and ``expose_returns`` chunks.
"""

import json

import pytest

from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.workflows import agent_result, stream_processor
from app.tgi.workflows.arg_injector import ToolResultCapture
from app.tgi.workflows.dict_utils import collect_exposed_returns
from app.tgi.workflows.feedback import (
    ACCEPTED_ACTION_RESULTS_KEY,
    FeedbackService,
    awaiting_action_result,
    build_feedback_payload,
    classify_action_feedback,
)
from app.tgi.workflows.models import WorkflowAgentDef, WorkflowDefinition
from app.tgi.workflows.state import WorkflowExecutionState

AGENT = "coordinator"
ACTION = {"op": "start_live_job", "args": {"user_query": "Summarize stale PRs"}}
CHOICES = [
    {
        "id": "action_result",
        "to": AGENT,
        "with": [],
        "payload_from": "action",
        "input": {"result_json": {"type": "string"}},
    }
]


def action_payload(request_id="req-1"):
    return build_feedback_payload(
        "Action requested.",
        CHOICES,
        {"action": json.dumps(ACTION)},
        {},
        kind="action",
        request_id=request_id,
    )


def result_message(request_id="req-1", selection="action_result", result="{}"):
    body = {
        "action": "accept",
        "request_id": request_id,
        "content": {"selection": selection, "result_json": result},
    }
    return f"<user_feedback>{json.dumps(body)}</user_feedback>"


def paused_state(request_id="req-1"):
    state = WorkflowExecutionState.new("exec-1", "flow")
    state.current_agent = AGENT
    state.awaiting_feedback = True
    state.context["user_query"] = "Summarize stale PRs"
    state.context["agents"][AGENT] = {
        "content": "",
        "awaiting_feedback": True,
        "elicitation_spec": action_payload(request_id),
        "elicitation_choices": CHOICES,
    }
    return state


class NoLLM:
    async def ask(self, **_kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("structured action feedback must not call the model")


def test_payload_from_carries_decoded_agent_data_and_correlation_meta():
    payload = action_payload()
    entry = payload["meta"]["expected_responses"][0]
    assert entry["payload"] == ACTION
    assert "payload_from" not in entry
    assert payload["meta"]["kind"] == "action"
    assert payload["meta"]["request_id"] == "req-1"
    assert payload["requestedSchema"]["additionalProperties"] is False


def test_human_pauses_are_untouched():
    plain = build_feedback_payload("Proceed?", CHOICES, {"action": "{}"}, {})
    assert "kind" not in plain["meta"]
    entry = {"elicitation_spec": plain}
    assert classify_action_feedback(entry, {}, "yes please") is None


@pytest.mark.parametrize(
    "feedback, code",
    [
        ("exclude drafts too", "untagged_text"),
        ("<user_feedback>not json</user_feedback>", "malformed"),
        ("<user_feedback>[1]</user_feedback>", "malformed"),
        (
            '<user_feedback>{"action":"accept","content":{"selection":"action_result"}}</user_feedback>',
            "missing_request_id",
        ),
        (result_message(request_id="req-stale"), "request_mismatch"),
        (result_message(selection="something_else"), "unknown_selection"),
    ],
)
def test_action_pause_rejects_untagged_malformed_stale_and_mismatched(feedback, code):
    entry = paused_state().context["agents"][AGENT]
    decision = classify_action_feedback(entry, {}, feedback)
    assert decision["verdict"] == "reject"
    assert decision["code"] == code
    assert decision["pending_request_id"] == "req-1"


def test_accepted_result_replays_and_altered_duplicate_conflicts():
    entry = paused_state().context["agents"][AGENT]
    accepted = classify_action_feedback(entry, {}, result_message())
    assert accepted["verdict"] == "accept"
    ledger = {ACCEPTED_ACTION_RESULTS_KEY: {"req-1": accepted["digest"]}}
    replay = classify_action_feedback(entry, ledger, result_message())
    assert replay["verdict"] == "replay"
    assert replay["code"] == "duplicate_replay"
    altered = classify_action_feedback(
        entry, ledger, result_message(result='{"status":"failed"}')
    )
    assert altered["verdict"] == "reject"
    assert altered["code"] == "duplicate_conflict"


@pytest.mark.asyncio
async def test_merge_feedback_rejection_leaves_pause_and_state_untouched():
    state = paused_state()
    saves = []
    service = FeedbackService(NoLLM())
    decision = await service.merge_feedback(
        state,
        "exclude drafts too",
        ChatCompletionRequest(messages=[Message(role=MessageRole.USER, content="x")]),
        None,
        None,
        save_fn=saves.append,
    )
    assert decision["verdict"] == "reject"
    assert state.awaiting_feedback is True
    assert state.context["agents"][AGENT]["awaiting_feedback"] is True
    assert saves == []
    assert "feedback" not in state.context


@pytest.mark.asyncio
async def test_merge_feedback_accepts_result_once_and_keeps_goal_text():
    state = paused_state()
    saves = []
    service = FeedbackService(NoLLM())
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content=result_message())]
    )
    decision = await service.merge_feedback(
        state, result_message(), request, None, None, save_fn=saves.append
    )
    assert decision is None
    assert state.awaiting_feedback is False
    entry = state.context["agents"][AGENT]
    assert entry["pending_user_reroute"]["target"] == AGENT
    assert entry["pending_user_reroute"]["assignments"]["result_json"] == "{}"
    assert state.context["_resume_agent"] == AGENT
    assert state.context["user_query"] == "Summarize stale PRs"
    assert "req-1" in state.context[ACCEPTED_ACTION_RESULTS_KEY]
    assert len(saves) == 1

    replay = await service.merge_feedback(
        state, result_message(), request, None, None, save_fn=saves.append
    )
    assert replay is None or replay["verdict"] == "replay"


def test_awaiting_action_result_only_for_machine_pauses():
    state = paused_state()
    assert awaiting_action_result(state) is True
    state.awaiting_feedback = False
    assert awaiting_action_result(state) is False
    human = paused_state()
    human.context["agents"][AGENT]["elicitation_spec"] = build_feedback_payload(
        "Proceed?", CHOICES, {"action": "{}"}, {}
    )
    assert awaiting_action_result(human) is False


def test_collect_exposed_returns_decodes_json_text_only_for_listed_names():
    context = {"outcome": '{"status": "completed"}', "job": {"id": "j1"}, "secret": "x"}
    exposed = collect_exposed_returns(["outcome", "job", "missing"], context)
    assert exposed == {"outcome": {"status": "completed"}, "job": {"id": "j1"}}


def test_tool_result_capture_reports_changed_exposed_returns():
    capture = ToolResultCapture(AGENT, ["job_id"])
    agent_context: dict = {}
    state_context = {"agents": {AGENT: agent_context}}
    result = stream_processor.StreamResult()
    first = stream_processor._process_tool_result(
        {"name": "start_live_job", "content": json.dumps({"job_id": "job-1"})},
        agent_context,
        state_context,
        result,
        capture,
        expose_returns=["job_id"],
    )
    assert first == {"job_id": "job-1"}
    unchanged = stream_processor._process_tool_result(
        {"name": "inspect_job", "content": json.dumps({"job_id": "job-1"})},
        agent_context,
        state_context,
        result,
        capture,
        expose_returns=["job_id"],
    )
    assert unchanged == {}


@pytest.mark.asyncio
async def test_finalize_emits_agent_returns_before_done():
    agent_def = WorkflowAgentDef(
        agent=AGENT, description="", pass_through=True, expose_returns=["outcome"]
    )
    workflow_def = WorkflowDefinition(
        flow_id="flow", root_intent="", agents=[agent_def]
    )
    state = WorkflowExecutionState.new("exec-1", "flow")
    agent_context = state.context["agents"].setdefault(AGENT, {"content": ""})
    recorded = []

    def record_event(state, text, status="in_progress", **kwargs):
        recorded.append({"text": text, "status": status, **kwargs})
        return f"event:{status}"

    events = []
    async for event in agent_result.finalize_agent_result(
        agent_def=agent_def,
        workflow_def=workflow_def,
        state=state,
        agent_context=agent_context,
        content_text='Done. <return name="outcome">{"status":"awaiting_user","job_id":"job-1"}</return>',
        tool_errors=[],
        tool_outcomes=[],
        passthrough_history=[],
        persist_inner_thinking=False,
        was_awaiting_feedback=False,
        had_feedback=False,
        result_capture=None,
        no_reroute=True,
        request=None,
        access_token=None,
        span=None,
        record_event_fn=record_event,
        save_fn=lambda _state: None,
        render_feedback_question_fn=None,
        routing_decide_fn=None,
        max_return_retries=1,
    ):
        events.append(event)

    assert events[0] == "event:agent_returns"
    assert recorded[0]["metadata"] == {
        "agent": AGENT,
        "returns": {"outcome": {"status": "awaiting_user", "job_id": "job-1"}},
    }
    assert events[-1]["status"] == "done"
    assert "<return" not in events[-1]["content"]
