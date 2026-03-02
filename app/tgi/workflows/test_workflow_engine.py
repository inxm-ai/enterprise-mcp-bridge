import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncGenerator

import jwt
import pytest

from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.services.tool_service import ToolService
from app.tgi.workflows.engine import WorkflowEngine
from app.tgi.workflows.models import WorkflowExecutionState
from app.tgi.workflows.repository import WorkflowRepository
from app.tgi.workflows.state import WorkflowStateStore


class StubSession:
    def __init__(
        self, prompts: list[Any] | None = None, tools: list[Any] | None = None
    ):
        self.prompts = prompts or []
        self.tools = tools or []

    async def list_prompts(self):
        return SimpleNamespace(prompts=self.prompts)

    async def call_prompt(self, name: str, args: dict[str, Any]):
        # Return the prompt text in the same shape PromptService expects
        text = next(
            (p.content for p in self.prompts if getattr(p, "name", "") == name), ""
        )
        message = SimpleNamespace(content=SimpleNamespace(text=text))
        return SimpleNamespace(isError=False, messages=[message])

    async def list_tools(self):
        return self.tools


class StubPrompt:
    def __init__(self, name: str, content: str):
        self.name = name
        self.description = f"Prompt for {name}"
        self.content = content


class StubLLMClient:
    """
    Minimal LLM stub that returns predetermined responses keyed by agent name.
    """

    def __init__(
        self,
        responses: dict[str, str | list[str]],
        ask_responses: dict[str, str] | None = None,
    ):
        self.responses = responses
        self.calls: list[str] = []
        self.request_tools: list[list[str] | None] = []
        self.system_prompts: list[str] = []
        self.response_counters: dict[str, int] = {}
        self.user_messages: list[str] = []
        self.ask_responses = ask_responses or {}
        self.ask_calls: list[dict[str, str | None]] = []

    def stream_completion(
        self, request: ChatCompletionRequest, access_token: str, span: Any
    ) -> AsyncGenerator[str, None]:
        if request.messages and request.messages[0].role == MessageRole.SYSTEM:
            self.system_prompts.append(request.messages[0].content or "")

        last_message = request.messages[-1].content or ""
        self.user_messages.append(last_message)
        agent_marker = ""
        if "<agent:" in last_message:
            agent_marker = last_message.split("<agent:", 1)[1].split(">", 1)[0]
        elif "ROUTING_INTENT_CHECK" in last_message:
            agent_marker = "routing_intent"
        elif "ROUTING_WHEN_CHECK" in last_message:
            agent_marker = f"routing_when:{self._extract_value(last_message, 'agent')}"
        elif "ROUTING_REROUTE_DECISION" in last_message:
            agent_marker = (
                f"routing_reroute:{self._extract_value(last_message, 'agent')}"
            )

        response_source = self.responses.get(agent_marker, "")
        if isinstance(response_source, list):
            idx = self.response_counters.get(agent_marker, 0)
            if response_source:
                response = (
                    response_source[idx]
                    if idx < len(response_source)
                    else response_source[-1]
                )
            else:
                response = ""
            self.response_counters[agent_marker] = idx + 1
        else:
            response = response_source
        self.calls.append(agent_marker)
        tool_names = None
        if request.tools is None:
            tool_names = None
        elif request.tools == []:
            tool_names = []
        else:
            collected: list[str] = []
            for t in request.tools:
                if isinstance(t, dict):
                    func = t.get("function", {})
                    name = func.get("name")
                else:
                    func = getattr(t, "function", None)
                    name = getattr(func, "name", None) if func else None
                if name:
                    collected.append(name)
            tool_names = collected if collected else []
        self.request_tools.append(tool_names)

        async def _gen():
            yield f"data: {response}\n\n"
            yield "data: [DONE]\n\n"

        return _gen()

    def _extract_value(self, text: str, key: str) -> str:
        try:
            return text.split(f"{key}=", 1)[1].split("\n", 1)[0].strip()
        except Exception:
            return ""

    async def ask(
        self,
        base_prompt: str,
        base_request: ChatCompletionRequest,
        outer_span,
        question: str = None,
        assistant_statement: str = None,
        access_token: str = None,
    ) -> str:
        self.ask_calls.append(
            {
                "base_prompt": base_prompt,
                "question": question,
                "assistant": assistant_statement,
            }
        )
        if "USER_QUERY_SUMMARY" in (base_prompt or ""):
            response = self.ask_responses.get("user_query_summary")
            if response is not None:
                return response
            return ""
        if "FEEDBACK_RERUN_DECISION" in (base_prompt or ""):
            response = self.ask_responses.get("feedback_rerun_decision")
            if response is not None:
                return response
            return "RERUN"
        if "USER_FEEDBACK_QUESTION" in (base_prompt or ""):
            response = self.ask_responses.get("feedback_question")
            if response is not None:
                return response
            return ""
        return self.ask_responses.get("default", "")


def _write_workflow(tmpdir: Path, name: str, payload: dict[str, Any]) -> None:
    (tmpdir / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def _make_access_token(user_id: str) -> str:
    token = jwt.encode(
        {"sub": user_id, "email": f"{user_id}@example.com"},
        "secret",
        algorithm="HS256",
    )
    if isinstance(token, bytes):
        return token.decode("utf-8")
    return token


def test_repository_loads_and_matches(monkeypatch, tmp_path):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow_today = {
        "flow_id": "what_can_i_do_today",
        "root_intent": "SUGGEST_TODAY_ACTIVITIES",
        "agents": [{"agent": "get_location", "description": "Find city"}],
    }
    flow_other = {
        "flow_id": "book_trip",
        "root_intent": "BOOK_TRIP",
        "agents": [{"agent": "plan_flight", "description": "Plan flight"}],
    }
    _write_workflow(workflows_dir, "today", flow_today)
    _write_workflow(workflows_dir, "trip", flow_other)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    repo = WorkflowRepository()

    assert repo.get("what_can_i_do_today").flow_id == "what_can_i_do_today"
    matched = repo.match_workflow("what can I do today around town?")
    assert matched is not None
    assert matched.flow_id == "what_can_i_do_today"

    with pytest.raises(ValueError):
        WorkflowRepository(workflows_path=str(workflows_dir / "missing"))


@pytest.mark.asyncio
async def test_workflow_resume_requires_same_user(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "locked_flow",
        "root_intent": "LOCKED",
        "agents": [{"agent": "only_agent", "description": "Run once."}],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    repo = WorkflowRepository()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    llm = StubLLMClient({"only_agent": "ok"})
    engine = WorkflowEngine(repo, store, llm)

    token_owner = _make_access_token("alice")
    token_other = _make_access_token("bob")

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Start")],
        use_workflow="locked_flow",
        workflow_execution_id="exec-locked",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, token_owner, None, None
    )
    assert stream is not None
    await stream.aclose()

    with pytest.raises(PermissionError, match="different user"):
        await engine.start_or_resume_workflow(
            StubSession(), request, token_other, None, None
        )

    with pytest.raises(PermissionError, match="Access token required"):
        await engine.start_or_resume_workflow(StubSession(), request, None, None, None)


@pytest.mark.asyncio
async def test_end_message_completes_workflow_and_streams_done_only(
    tmp_path, monkeypatch
):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "simple_flow",
        "root_intent": "SIMPLE_TASK",
        "agents": [{"agent": "only_agent", "description": "Do the work"}],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    repo = WorkflowRepository()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    llm = StubLLMClient({"only_agent": "ok"})
    engine = WorkflowEngine(repo, store, llm)

    exec_id = "exec-end"
    store.save_state(WorkflowExecutionState.new(exec_id, flow["flow_id"]))

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="[end]")],
        use_workflow="simple_flow",
        workflow_execution_id=exec_id,
        stream=True,
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    assert stream is not None
    chunks = [chunk async for chunk in stream]

    assert chunks == ["data: [DONE]\n\n"]
    state = store.load_execution(exec_id)
    assert state.completed is True
    assert state.awaiting_feedback is False
    assert state.context.get("_workflow_outcome") == "success"
    assert llm.calls == []


@pytest.mark.asyncio
async def test_engine_runs_agents_and_pass_through(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "what_can_i_do_today",
        "root_intent": "SUGGEST_TODAY_ACTIVITIES",
        "agents": [
            {
                "agent": "get_location",
                "description": "Ask or infer the user's current city or coordinates.",
                "pass_through": True,
            },
            {
                "agent": "get_weather",
                "description": "Fetch the weather forecast for a given location.",
                "depends_on": ["get_location"],
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    repo = WorkflowRepository()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    llm = StubLLMClient({"get_location": "San Francisco", "get_weather": "Sunny"})
    engine = WorkflowEngine(repo, store, llm)

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="What can I do?")],
        model="test-model",
        stream=True,
        use_workflow="what_can_i_do_today",
        workflow_execution_id="exec-pass",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    assert stream is not None
    chunks = [chunk async for chunk in stream]
    payload = "\n".join(chunks)

    expected_tag = '<workflow_execution_id for="what_can_i_do_today">exec-pass</workflow_execution_id>'
    assert expected_tag in payload or "exec-pass" in payload
    assert "Fetching your location" in payload
    assert "San Francisco" in payload  # pass-through shown
    assert "Workflow complete" in payload

    state = store.load_execution("exec-pass")
    assert state.completed is True
    assert state.context.get("_persist_inner_thinking") is False
    assert state.context["agents"]["get_location"]["content"] == "San Francisco"
    assert state.context["agents"]["get_weather"]["content"] == ""
    assert llm.calls == ["routing_intent", "get_location", "get_weather"]


@pytest.mark.asyncio
async def test_routing_agent_blocks_intent_mismatch(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "what_can_i_do_today",
        "root_intent": "SUGGEST_TODAY_ACTIVITIES",
        "agents": [{"agent": "get_location", "description": "Ask location"}],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient({"routing_intent": "<reroute>INTENT_MISMATCH</reroute>"})
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Tell me a joke")],
        model="test-model",
        stream=True,
        use_workflow="what_can_i_do_today",
        workflow_execution_id="exec-intent",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    output = "".join([chunk async for chunk in stream])
    assert "<reroute>INTENT_MISMATCH</reroute>" in output
    state = engine.state_store.load_execution("exec-intent")
    assert state.completed is True


@pytest.mark.asyncio
async def test_agent_context_can_be_disabled(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "context_toggle",
        "root_intent": "TEST_CONTEXT",
        "agents": [
            {
                "agent": "first",
                "description": "First agent",
                "pass_through": True,
            },
            {
                "agent": "second",
                "description": "Second agent",
                "context": False,
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient({"first": "First out", "second": "Second out"})
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="start")],
        model="test-model",
        stream=True,
        use_workflow="context_toggle",
        workflow_execution_id="exec-context-toggle",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    second_idx = llm.calls.index("second")
    second_prompt = llm.user_messages[second_idx]

    # Context is disabled (empty), so we get an empty dict
    assert "Context summary: {}" in second_prompt
    assert "First out" not in second_prompt


@pytest.mark.asyncio
async def test_agent_context_can_be_scoped(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "context_scoped",
        "root_intent": "TEST_CONTEXT_SCOPED",
        "agents": [
            {
                "agent": "first",
                "description": "First agent",
                "pass_through": True,
            },
            {
                "agent": "second",
                "description": "Second agent",
                "context": ["first.content"],
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient({"first": "First out", "second": "Second out"})
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="start")],
        model="test-model",
        stream=True,
        use_workflow="context_scoped",
        workflow_execution_id="exec-context-scoped",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    second_idx = llm.calls.index("second")
    second_prompt = llm.user_messages[second_idx]

    assert "First out" in second_prompt
    assert "task_id" not in second_prompt


@pytest.mark.asyncio
async def test_agent_context_can_use_original_user_prompt(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "context_user_prompt",
        "root_intent": "TEST_CONTEXT_USER_PROMPT",
        "agents": [
            {
                "agent": "first",
                "description": "First agent",
                "pass_through": True,
            },
            {
                "agent": "second",
                "description": "Second agent",
                "context": "user_prompt",
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient({"first": "First out", "second": "Second out"})
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="original question")],
        model="test-model",
        stream=True,
        use_workflow="context_user_prompt",
        workflow_execution_id="exec-context-user-prompt",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    second_idx = llm.calls.index("second")
    second_prompt = llm.user_messages[second_idx]

    # Context is now provided as a summary
    assert "Context summary:" in second_prompt
    assert '"user_prompt": "original question"' in second_prompt
    assert "First out" not in second_prompt
    assert "task_id" not in second_prompt


@pytest.mark.asyncio
async def test_when_condition_via_routing_agent(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "conditional_flow",
        "root_intent": "CHECK_WEATHER",
        "agents": [
            {
                "agent": "get_weather",
                "description": "Fetch weather",
                "when": "context.should_check_weather",
            },
            {"agent": "finalize", "description": "Done"},
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient({"routing_when:get_weather": "<run>false</run>"})
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    exec_id = "exec-when"
    request = ChatCompletionRequest(
        messages=[
            Message(role=MessageRole.USER, content="Decide if we need weather check")
        ],
        model="test-model",
        stream=True,
        use_workflow="conditional_flow",
        workflow_execution_id=exec_id,
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]
    state = engine.state_store.load_execution(exec_id)
    # Agent skipped due to routing decision
    assert state.context["agents"]["get_weather"]["skipped"] is True
    assert state.context["agents"]["get_weather"]["reason"] == "condition_not_met"


@pytest.mark.asyncio
async def test_when_eval_beats_routing_when_for_mutually_exclusive_agents(
    tmp_path, monkeypatch
):
    """
    Deterministic when conditions should not be overridden by routing_when.

    If detect_existing_plans reroutes to execution, schedule_plan should be
    skipped even if routing_when returns <run>true>.
    """
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "mutual_exclusive_exec",
        "root_intent": "EXEC_OR_SCHEDULE",
        "agents": [
            {
                "agent": "detect_existing_plans",
                "description": "Detect",
                "reroute": [
                    {"on": ["FOUND_EXECUTE"], "to": "execute_plan"},
                    {"on": ["FOUND_SCHEDULE"], "to": "schedule_plan"},
                ],
            },
            {
                "agent": "execute_plan",
                "description": "Execute",
                "depends_on": ["detect_existing_plans"],
                "when": "context.get('agents', {}).get('detect_existing_plans', {}).get('reroute_reason') == 'FOUND_EXECUTE'",
            },
            {
                "agent": "schedule_plan",
                "description": "Schedule",
                "depends_on": ["detect_existing_plans"],
                "when": "context.get('agents', {}).get('detect_existing_plans', {}).get('reroute_reason') == 'FOUND_SCHEDULE'",
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "detect_existing_plans": "<reroute>FOUND_EXECUTE</reroute>",
            "execute_plan": "executed",
            # Routing agent tries to force schedule_plan to run, but the eval guard should skip it
            "routing_when:schedule_plan": "<run>true</run>",
        }
    )
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    exec_id = "exec-mutual-exclusive"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="run now")],
        model="test-model",
        stream=True,
        use_workflow="mutual_exclusive_exec",
        workflow_execution_id=exec_id,
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]
    state = engine.state_store.load_execution(exec_id)

    # Execute plan ran
    assert state.context["agents"]["execute_plan"]["completed"] is True
    # Schedule plan was skipped by condition, despite routing_when <run>true>
    assert state.context["agents"]["schedule_plan"]["skipped"] is True
    assert state.context["agents"]["schedule_plan"]["reason"] == "condition_not_met"
    assert "schedule_plan" not in llm.calls


@pytest.mark.asyncio
async def test_missing_arg_mapping_fails_workflow(tmp_path, monkeypatch):
    """
    If a tool arg mapping cannot be resolved, the workflow should fail loudly
    with details about the missing context.
    """
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "arg_inject_fail",
        "root_intent": "TEST",
        "agents": [
            {
                "agent": "runner",
                "description": "Calls a tool with mapped args",
                "tools": [
                    {
                        "run_tool": {
                            "args": {"location": "generate_inputs.result"},
                        }
                    }
                ],
            }
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_tool",
                "description": "Run",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    class ArgFailLLM(StubLLMClient):
        def stream_completion(self, request, access_token, span):
            # Immediately emit a tool call for run_tool with no args; ArgInjector will fail
            async def _gen():
                yield (
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                    '"function":{"name":"run_tool","arguments":"{}"}}]},"index":0}]}\n\n'
                )
                yield "data: [DONE]\n\n"

            return _gen()

    llm = ArgFailLLM({})
    tool_service = ToolService()

    async def mock_execute_tool_calls(*args, **kwargs):
        return ([], True, [])

    tool_service.execute_tool_calls = mock_execute_tool_calls

    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
        tool_service=tool_service,
    )

    exec_id = "exec-arg-fail"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="go")],
        model="test-model",
        stream=True,
        use_workflow="arg_inject_fail",
        workflow_execution_id=exec_id,
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(tools=tools), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]
    payload = "\n".join(chunks)

    assert "configuration error" in payload.lower()
    assert "generate_inputs" in payload or "available context" in payload.lower()
    state = engine.state_store.load_execution(exec_id)
    assert state.completed is True


@pytest.mark.asyncio
async def test_no_reroute_override(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "reroute_flow",
        "root_intent": "ACTIVITY",
        "agents": [
            {
                "agent": "get_outdoor",
                "description": "Suggest outdoor",
                "reroute": {"on": ["NO_GOOD_OUTDOOR_OPTIONS"], "to": "get_indoor"},
            },
            {"agent": "get_indoor", "description": "Indoor ideas"},
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient({"get_outdoor": "<reroute>NO_GOOD_OUTDOOR_OPTIONS</reroute>"})
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    exec_id = "exec-noreroute"
    request = ChatCompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content="Plan my day <no_reroute>",
            )
        ],
        model="test-model",
        stream=True,
        use_workflow="reroute_flow",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]
    state = engine.state_store.load_execution(exec_id)
    # Reroute was suppressed; current agent content recorded, but next agent still executed in order
    assert (
        state.context["agents"]["get_outdoor"]["reroute_reason"]
        == "NO_GOOD_OUTDOOR_OPTIONS"
    )
    assert "get_indoor" in state.context["agents"]


@pytest.mark.asyncio
async def test_dynamic_reroute_decision(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "dynamic_reroute",
        "root_intent": "ACTIVITY",
        "agents": [
            {"agent": "first", "description": "First agent"},
            {"agent": "second", "description": "Second agent"},
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "first": "<reroute>GO_TO_SECOND</reroute>",
            "routing_reroute:first": "<next_agent>second</next_agent>",
        }
    )
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    exec_id = "exec-dynamic"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="start")],
        model="test-model",
        stream=True,
        use_workflow="dynamic_reroute",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]
    state = engine.state_store.load_execution(exec_id)
    assert state.context["agents"]["first"]["reroute_reason"] == "GO_TO_SECOND"
    assert "second" in state.context["agents"]


@pytest.mark.asyncio
async def test_missing_reroute_target_emits_error_and_completes(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "missing_reroute",
        "root_intent": "TEST",
        "agents": [{"agent": "first", "description": "First agent"}],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "first": "<reroute>FAILURE</reroute>",
            "routing_reroute:first": "<next_agent>ghost_agent</next_agent>",
        }
    )
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    exec_id = "exec-missing-reroute"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="start")],
        model="test-model",
        stream=True,
        use_workflow="missing_reroute",
        workflow_execution_id=exec_id,
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]
    payload = "\n".join(chunks)

    assert "ghost_agent" in payload
    assert "not defined" in payload or "not runnable" in payload
    assert '"status":"error"' in payload or '"status": "error"' in payload
    assert chunks[-1].strip() == "data: [DONE]"

    state = engine.state_store.load_execution(exec_id)
    assert state.completed is True
    assert state.context["agents"]["first"]["reroute_reason"] == "FAILURE"

    initial_call_count = len(llm.calls)
    resume_stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    resume_chunks = [chunk async for chunk in resume_stream]
    assert resume_chunks[-1].strip() == "data: [DONE]"
    # No new LLM invocations on resume because the workflow was marked complete
    assert len(llm.calls) == initial_call_count


@pytest.mark.asyncio
async def test_workflow_reroute_to_new_flow_with_start_with(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    first_flow = {
        "flow_id": "first_flow",
        "root_intent": "FIRST",
        "agents": [{"agent": "first_agent", "description": "First"}],
    }
    second_flow = {
        "flow_id": "second_flow",
        "root_intent": "SECOND",
        "agents": [{"agent": "second_agent", "description": "Second"}],
    }
    _write_workflow(workflows_dir, "first", first_flow)
    _write_workflow(workflows_dir, "second", second_flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "first_agent": '<reroute start_with=\'{"args":{"prefill":"set"}}\'>workflows[second_flow]</reroute>',
            "second_agent": "Second workflow complete",
        }
    )
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="start handoff")],
        model="test-model",
        stream=True,
        use_workflow="first_flow",
        workflow_execution_id="exec-handoff",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]
    payload = "\n".join(chunks)

    assert "Rerouting to workflow 'second_flow'" in payload
    assert "Second workflow complete" in payload
    assert "first_agent" in llm.calls
    assert "second_agent" in llm.calls
    assert llm.calls.index("first_agent") < llm.calls.index("second_agent")

    state = engine.state_store.load_execution("exec-handoff")
    assert state.flow_id == "second_flow"
    assert state.context.get("prefill") == "set"
    assert "first_agent" not in (state.context.get("agents") or {})
    with engine.state_store._connect() as conn:
        rows = conn.execute("SELECT execution_id FROM workflow_executions").fetchall()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_workflow_reroute_resets_agent_context_for_overlap(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    first_flow = {
        "flow_id": "first_flow",
        "root_intent": "FIRST",
        "agents": [{"agent": "shared_agent", "description": "First"}],
    }
    second_flow = {
        "flow_id": "second_flow",
        "root_intent": "SECOND",
        "agents": [{"agent": "shared_agent", "description": "Second"}],
    }
    _write_workflow(workflows_dir, "first", first_flow)
    _write_workflow(workflows_dir, "second", second_flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "shared_agent": [
                "<reroute>workflows[second_flow]</reroute>",
                "Second workflow complete",
            ]
        }
    )
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="start handoff")],
        model="test-model",
        stream=True,
        use_workflow="first_flow",
        workflow_execution_id="exec-overlap",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    payload = "\n".join([chunk async for chunk in stream])

    assert llm.calls.count("shared_agent") == 2
    assert "Second workflow complete" in payload
    state = engine.state_store.load_execution("exec-overlap")
    assert state.flow_id == "second_flow"


@pytest.mark.asyncio
async def test_workflow_reroute_with_fields_prefills_new_context(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    first_flow = {
        "flow_id": "select_run_mode_flow",
        "root_intent": "SELECT_RUN_MODE",
        "agents": [
            {
                "agent": "select_run_mode",
                "description": "Choose next action.",
                "reroute": [
                    {
                        "on": ["workflows[plan_run]"],
                        "to": "workflows[plan_run]",
                        "with": ["plan_id"],
                    }
                ],
            }
        ],
    }
    second_flow = {
        "flow_id": "plan_run",
        "root_intent": "PLAN_RUN",
        "agents": [
            {
                "agent": "get_plan",
                "description": "Fetch the plan.",
                "tools": [{"get_plan": {"args": {"plan_id": "plan_id"}}}],
            }
        ],
    }
    _write_workflow(workflows_dir, "first", first_flow)
    _write_workflow(workflows_dir, "second", second_flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_plan",
                "description": "Fetch plan",
                "parameters": {
                    "type": "object",
                    "properties": {"plan_id": {"type": "string"}},
                    "required": ["plan_id"],
                },
            },
        }
    ]

    class RerouteWorkflowLLM(StubLLMClient):
        def __init__(self):
            super().__init__(
                {
                    "select_run_mode": '<return name="plan_id">plan-xyz</return><reroute>workflows[plan_run]</reroute>',
                    "get_plan": "<passthrough>Done</passthrough>",
                }
            )
            self._emitted_tool = False

        def stream_completion(self, request, access_token, span):
            last_message = request.messages[-1].content or ""
            agent_marker = ""
            if "<agent:" in last_message:
                agent_marker = last_message.split("<agent:", 1)[1].split(">", 1)[0]

            if agent_marker == "get_plan" and not self._emitted_tool:
                self._emitted_tool = True

                async def _gen_tool():
                    payload = {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "function": {
                                                "name": "get_plan",
                                                "arguments": "{}",
                                            },
                                        }
                                    ]
                                },
                                "index": 0,
                            }
                        ]
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    yield "data: [DONE]\n\n"

                return _gen_tool()

            return super().stream_completion(request, access_token, span)

    class CaptureSession(StubSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tool_calls: list[dict[str, Any]] = []

        async def call_tool(self, name: str, args: dict, access_token: str):
            self.tool_calls.append({"name": name, "args": args})
            return SimpleNamespace(
                isError=False,
                content=[
                    SimpleNamespace(
                        text=json.dumps({"plan": {"id": args.get("plan_id")}})
                    )
                ],
            )

    llm = RerouteWorkflowLLM()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="run")],
        model="test-model",
        stream=True,
        use_workflow="select_run_mode_flow",
        workflow_execution_id="exec-handoff-fields",
    )

    session = CaptureSession(tools=tools)
    stream = await engine.start_or_resume_workflow(session, request, None, None, None)
    _ = [chunk async for chunk in stream]

    state = store.load_execution("exec-handoff-fields")
    assert state.flow_id == "plan_run"
    assert state.context.get("plan_id") == "plan-xyz"
    assert "select_run_mode" not in (state.context.get("agents") or {})
    assert session.tool_calls
    assert session.tool_calls[0]["args"].get("plan_id") == "plan-xyz"


@pytest.mark.asyncio
async def test_workflow_reroute_with_shared_context_value(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    first_flow = {
        "flow_id": "select_plan_flow",
        "root_intent": "SELECT_PLAN",
        "agents": [
            {
                "agent": "select_plan",
                "description": "Choose plan.",
                "reroute": [
                    {
                        "on": ["PLAN_SELECTED"],
                        "to": "select_run_mode",
                        "with": ["plan_id"],
                    }
                ],
            },
            {
                "agent": "select_run_mode",
                "description": "Choose next action.",
                "depends_on": ["select_plan"],
                "reroute": [
                    {
                        "on": ["workflows[plan_run]"],
                        "to": "workflows[plan_run]",
                        "with": ["plan_id"],
                    }
                ],
            },
        ],
    }
    second_flow = {
        "flow_id": "plan_run",
        "root_intent": "PLAN_RUN",
        "agents": [
            {
                "agent": "get_plan",
                "description": "Fetch the plan.",
                "tools": [{"get_plan": {"args": {"plan_id": "plan_id"}}}],
            }
        ],
    }
    _write_workflow(workflows_dir, "first", first_flow)
    _write_workflow(workflows_dir, "second", second_flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_plan",
                "description": "Fetch plan",
                "parameters": {
                    "type": "object",
                    "properties": {"plan_id": {"type": "string"}},
                    "required": ["plan_id"],
                },
            },
        }
    ]

    class SharedContextWorkflowLLM(StubLLMClient):
        def __init__(self):
            super().__init__(
                {
                    "select_plan": '<return name="plan_id">plan-xyz</return><reroute>PLAN_SELECTED</reroute>',
                    "select_run_mode": "<reroute>workflows[plan_run]</reroute>",
                    "get_plan": "<passthrough>Done</passthrough>",
                }
            )
            self._emitted_tool = False

        def stream_completion(self, request, access_token, span):
            last_message = request.messages[-1].content or ""
            agent_marker = ""
            if "<agent:" in last_message:
                agent_marker = last_message.split("<agent:", 1)[1].split(">", 1)[0]

            if agent_marker == "get_plan" and not self._emitted_tool:
                self._emitted_tool = True

                async def _gen_tool():
                    payload = {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "function": {
                                                "name": "get_plan",
                                                "arguments": "{}",
                                            },
                                        }
                                    ]
                                },
                                "index": 0,
                            }
                        ]
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    yield "data: [DONE]\n\n"

                return _gen_tool()

            return super().stream_completion(request, access_token, span)

    class CaptureSession(StubSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tool_calls: list[dict[str, Any]] = []

        async def call_tool(self, name: str, args: dict, access_token: str):
            self.tool_calls.append({"name": name, "args": args})
            return SimpleNamespace(
                isError=False,
                content=[
                    SimpleNamespace(
                        text=json.dumps({"plan": {"id": args.get("plan_id")}})
                    )
                ],
            )

    llm = SharedContextWorkflowLLM()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="run")],
        model="test-model",
        stream=True,
        use_workflow="select_plan_flow",
        workflow_execution_id="exec-handoff-shared",
    )

    session = CaptureSession(tools=tools)
    stream = await engine.start_or_resume_workflow(session, request, None, None, None)
    _ = [chunk async for chunk in stream]

    state = store.load_execution("exec-handoff-shared")
    assert state.flow_id == "plan_run"
    assert state.context.get("plan_id") == "plan-xyz"
    assert "select_plan" not in (state.context.get("agents") or {})
    assert session.tool_calls
    assert session.tool_calls[0]["args"].get("plan_id") == "plan-xyz"


@pytest.mark.asyncio
async def test_workflow_reroute_loop_detection(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    first_flow = {
        "flow_id": "loop_a",
        "root_intent": "A",
        "agents": [{"agent": "first", "description": "First"}],
    }
    second_flow = {
        "flow_id": "loop_b",
        "root_intent": "B",
        "agents": [{"agent": "second", "description": "Second"}],
    }
    _write_workflow(workflows_dir, "a", first_flow)
    _write_workflow(workflows_dir, "b", second_flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "first": "<reroute>workflows[loop_b]</reroute>",
            "second": "<reroute>workflows[loop_a]</reroute>",
        }
    )
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="loop")],
        model="test-model",
        stream=True,
        use_workflow="loop_a",
        workflow_execution_id="exec-loop",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    payload = "\n".join([chunk async for chunk in stream])

    assert "Workflow reroute loop detected" in payload
    assert llm.calls.count("first") == 1
    assert llm.calls.count("second") == 1

    state = engine.state_store.load_execution("exec-loop")
    assert state.flow_id == "loop_b"


@pytest.mark.asyncio
async def test_workflow_reroute_to_missing_flow(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    first_flow = {
        "flow_id": "primary_flow",
        "root_intent": "PRIMARY",
        "agents": [{"agent": "first_agent", "description": "First"}],
    }
    _write_workflow(workflows_dir, "primary", first_flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient({"first_agent": "<reroute>workflows[ghost_flow]</reroute>"})
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="missing")],
        model="test-model",
        stream=True,
        use_workflow="primary_flow",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    payload = "\n".join([chunk async for chunk in stream])

    assert "Workflow 'ghost_flow' is not defined." in payload
    assert llm.calls.count("first_agent") == 1

    with engine.state_store._connect() as conn:
        rows = conn.execute("SELECT flow_id FROM workflow_executions").fetchall()
    assert {row[0] for row in rows} == {"primary_flow"}


@pytest.mark.asyncio
async def test_reroute_to_completed_agent_fails_even_with_dependencies_met(
    tmp_path, monkeypatch
):
    """
    If an agent reroutes back to a previously completed agent, the engine cannot
    re-run it even when its dependencies are satisfied (save_plan -> test_plan -> heal_plan -> test_plan).
    """
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "healing_flow",
        "root_intent": "TEST",
        "agents": [
            {"agent": "save_plan", "description": "Save the plan"},
            {
                "agent": "test_plan",
                "description": "Test the plan",
                "depends_on": ["save_plan"],
                "reroute": {"on": ["TEST_FAILED"], "to": "heal_plan"},
            },
            {
                "agent": "heal_plan",
                "description": "Heal failed tests",
                "depends_on": ["test_plan"],
                "reroute": {"on": ["HEALED"], "to": "test_plan"},
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "save_plan": "Plan saved.",
            "test_plan": "<reroute>TEST_FAILED</reroute>",
            "heal_plan": "<reroute>HEALED</reroute>",
        }
    )
    repo = WorkflowRepository()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(repo, store, llm)

    exec_id = "exec-healing-loop"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="run tests")],
        model="test-model",
        stream=True,
        use_workflow="healing_flow",
        workflow_execution_id=exec_id,
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]
    payload = "\n".join(chunks)

    assert "Reroute target 'test_plan' was already completed" in payload

    state = store.load_execution(exec_id)
    assert state.completed is True
    # Dependencies were satisfied, but the reroute still failed because test_plan was already completed
    assert state.context["agents"]["save_plan"]["completed"] is True
    assert state.context["agents"]["test_plan"]["reroute_reason"] == "TEST_FAILED"
    assert state.context["agents"]["heal_plan"]["reroute_reason"] == "HEALED"


@pytest.mark.asyncio
async def test_no_runnable_agents_produces_terminal_error(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "deadlock_flow",
        "root_intent": "TEST",
        "agents": [
            {"agent": "alpha", "description": "Alpha", "depends_on": ["beta"]},
            {"agent": "beta", "description": "Beta", "depends_on": ["alpha"]},
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient({"routing_intent": ""})
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    exec_id = "exec-deadlock"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="start")],
        model="test-model",
        stream=True,
        use_workflow="deadlock_flow",
        workflow_execution_id=exec_id,
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]
    payload = "\n".join(chunks)

    assert "No runnable agents" in payload
    assert chunks[-1].strip() == "data: [DONE]"

    state = engine.state_store.load_execution(exec_id)
    assert state.completed is True
    # Only the routing probe should have been executed
    assert llm.calls and llm.calls[0] == "routing_intent"


@pytest.mark.asyncio
async def test_list_reroute_config(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "list_reroute_flow",
        "root_intent": "ACTIVITY",
        "agents": [
            {
                "agent": "detect",
                "description": "Detect intent",
                "reroute": [
                    {"on": ["GO_TO_NEXT"], "to": "runner"},
                    {"on": ["IGNORE"], "to": "noop"},
                ],
            },
            {"agent": "runner", "description": "Run task"},
            {
                "agent": "noop",
                "description": "Should not run",
                "when": "context.get('agents', {}).get('detect', {}).get('reroute_reason') == 'IGNORE'",
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient({"detect": "<reroute>GO_TO_NEXT</reroute>"})
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    exec_id = "exec-list-reroute"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="run")],
        model="test-model",
        stream=True,
        use_workflow="list_reroute_flow",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]
    state = engine.state_store.load_execution(exec_id)
    # reroute reason captured and the matching target agent executed; unmatched agent skipped
    assert state.context["agents"]["detect"]["reroute_reason"] == "GO_TO_NEXT"
    assert "runner" in state.context["agents"]
    assert state.context["agents"]["noop"]["skipped"] is True
    assert state.context["agents"]["noop"]["reason"] == "condition_not_met"


@pytest.mark.asyncio
async def test_agent_tools_scoping(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "tools_flow",
        "root_intent": "TOOLS_TEST",
        "agents": [
            {"agent": "get_location", "description": "Ask location"},
            {
                "agent": "get_weather",
                "description": "Fetch weather",
                "tools": ["get_weather"],
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_location",
                "description": "Get location",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    llm = StubLLMClient({"get_location": "loc", "get_weather": "sunny"})
    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
    )

    exec_id = "exec-tools"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="run")],
        model="test-model",
        stream=True,
        use_workflow="tools_flow",
        workflow_execution_id=exec_id,
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(tools=tools), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    # First routing call has no tools, agent calls include scoped tools
    assert llm.request_tools[1] and set(llm.request_tools[1]) >= {"get_location"}
    assert llm.request_tools[2] == ["get_weather"]


@pytest.mark.asyncio
async def test_agent_tools_fallback_to_all_when_none_match(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "fallback_tools_flow",
        "root_intent": "TOOLS_TEST",
        "agents": [
            {
                "agent": "detect",
                "description": "Should use tools even if names do not match",
                "tools": ["missing_tool"],
            }
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "available_tool",
                "description": "Some tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    session = StubSession(tools=tools)
    llm = StubLLMClient({"detect": ""})
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    exec_id = "exec-fallback-tools"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="run")],
        model="test-model",
        stream=True,
        use_workflow="fallback_tools_flow",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(session, request, None, None, None)
    _ = [chunk async for chunk in stream]

    # Even though the agent requested a missing tool, engine should pass all tools
    # (plus the lazy context tool added by routing)
    assert set(llm.request_tools[0]) == {
        "available_tool",
        "get_workflow_context",
        "select-from-tool-response",
    }


@pytest.mark.asyncio
async def test_agent_executes_tool_calls(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "tool_call_flow",
        "root_intent": "TOOLS_TEST",
        "agents": [
            {
                "agent": "detect",
                "description": "Detect intent",
                "tools": ["search_plan"],
            }
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_plan",
                "description": "Search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    class ToolCallingLLM(StubLLMClient):
        def __init__(self):
            super().__init__({})
            self.call_count = 0

        def stream_completion(self, request, access_token, span):
            self.call_count += 1

            async def _gen_first():
                yield (
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"search_plan","arguments":"{}"}}]},"index":0}]}\n\n'
                )
                yield "data: [DONE]\n\n"

            async def _gen_second():
                yield 'data: {"choices":[{"delta":{"content":"done"},"index":0}]}\n\n'
                yield "data: [DONE]\n\n"

            return _gen_first() if self.call_count == 1 else _gen_second()

    llm = ToolCallingLLM()
    tool_service = ToolService()

    call_tracker = {"called": False}

    async def mock_execute_tool_calls(*args, **kwargs):
        call_tracker["called"] = True
        if kwargs.get("return_raw_results"):
            return ([], True, [])
        return ([], True)

    tool_service.execute_tool_calls = mock_execute_tool_calls

    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
        tool_service=tool_service,
    )

    exec_id = "exec-tool-call"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="run")],
        model="test-model",
        stream=True,
        use_workflow="tool_call_flow",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(tools=tools), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    assert call_tracker["called"], "execute_tool_calls should have been called"
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_agent_prompt_includes_feedback_guidelines(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "guidelines_flow",
        "root_intent": "TEST",
        "agents": [{"agent": "get_location", "description": "Ask location"}],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient({"get_location": "loc"})
    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
    )
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="where am I?")],
        model="test-model",
        stream=True,
        use_workflow="guidelines_flow",
        workflow_execution_id="exec-guidelines",
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]
    # First system prompt is routing_agent; second is agent prompt with guidelines
    assert any("user_feedback_needed" in prompt for prompt in llm.system_prompts)


@pytest.mark.asyncio
async def test_custom_routing_prompt_by_flow_id(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "guidelines_flow",
        "root_intent": "TEST",
        "agents": [{"agent": "get_location", "description": "Ask location"}],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    routing_prompt = StubPrompt("guidelines_flow", "Custom routing instructions")
    llm = StubLLMClient({"routing_intent": "<reroute>STOP</reroute>"})
    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
    )
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="where am I?")],
        model="test-model",
        stream=True,
        use_workflow="guidelines_flow",
        workflow_execution_id="exec-routing-prompt",
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(prompts=[routing_prompt]), request, None, None, None
    )
    _ = [chunk async for chunk in stream]
    # First prompt sent to routing agent should include custom and default text
    assert llm.system_prompts
    assert "Custom routing instructions" in llm.system_prompts[0]
    assert "You are the routing_agent" in llm.system_prompts[0]


@pytest.mark.asyncio
async def test_agent_tools_empty_list_disables_tools(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "tools_none_flow",
        "root_intent": "TOOLS_TEST",
        "agents": [
            {"agent": "get_location", "description": "Ask location", "tools": []},
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_location",
                "description": "Get location",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    llm = StubLLMClient({"get_location": "loc"})
    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
    )

    exec_id = "exec-tools-none"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="run")],
        model="test-model",
        stream=True,
        use_workflow="tools_none_flow",
        workflow_execution_id=exec_id,
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(tools=tools), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    # No tools should be passed to LLM for the agent
    assert llm.request_tools[-1] == []


@pytest.mark.asyncio
async def test_engine_resumes_after_user_feedback(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "preference_flow",
        "root_intent": "ASK_PREFERENCE",
        "agents": [
            {"agent": "ask_preference", "description": "Ask for preference."},
            {
                "agent": "finalize",
                "description": "Summarize the plan.",
                "depends_on": ["ask_preference"],
            },
        ],
    }
    _write_workflow(workflows_dir, "pref", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    repo = WorkflowRepository()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    llm = StubLLMClient(
        {
            "ask_preference": [
                "<user_feedback_needed>Please tell me if you prefer indoors</user_feedback_needed>",
                "Thanks, I will plan for indoors now.",
            ],
            "finalize": "Great, here is your indoor plan.",
        }
    )
    engine = WorkflowEngine(repo, store, llm)

    initial_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Plan my day")],
        model="test-model",
        stream=True,
        use_workflow="preference_flow",
        workflow_execution_id="exec-feedback",
    )

    first_stream = await engine.start_or_resume_workflow(
        StubSession(), initial_request, None, None, None
    )
    assert first_stream is not None
    first_chunks = [chunk async for chunk in first_stream]
    first_payload = "\n".join(first_chunks)
    assert "feedback" in first_payload.lower()
    state = store.load_execution("exec-feedback")
    assert state.awaiting_feedback is True
    assert state.completed is False
    first_task_id = state.context.get("task_id")

    resume_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="I prefer indoors")],
        model="test-model",
        stream=True,
        use_workflow="preference_flow",
        workflow_execution_id="exec-feedback",
    )
    resume_stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    assert resume_stream is not None
    resume_chunks = [chunk async for chunk in resume_stream]
    resume_payload = "\n".join(resume_chunks)
    # execution id should be present in the resumed payload (XML tag or id)
    expected_tag = '<workflow_execution_id for="preference_flow">exec-feedback</workflow_execution_id>'
    assert expected_tag in resume_payload or "exec-feedback" in resume_payload
    assert "routing workflow preference_flow" not in resume_payload.lower()
    assert "indoor plan" in resume_payload.lower()

    final_state = store.load_execution("exec-feedback")
    assert final_state.completed is True
    assert final_state.awaiting_feedback is False
    assert (
        final_state.context["agents"]["ask_preference"]["content"]
        == "Thanks, I will plan for indoors now."
    )
    assert final_state.context.get("task_id") != first_task_id
    agent_calls = [call for call in llm.calls if call in {"ask_preference", "finalize"}]
    assert agent_calls == ["ask_preference", "ask_preference", "finalize"]


@pytest.mark.asyncio
async def test_resume_without_use_workflow_uses_state_flow(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "preference_flow",
        "root_intent": "ASK_PREFERENCE",
        "agents": [
            {"agent": "ask_preference", "description": "Ask for preference."},
            {
                "agent": "finalize",
                "description": "Summarize the plan.",
                "depends_on": ["ask_preference"],
            },
        ],
    }
    _write_workflow(workflows_dir, "pref", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    repo = WorkflowRepository()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    llm = StubLLMClient(
        {
            "ask_preference": [
                "<user_feedback_needed>Please tell me if you prefer indoors</user_feedback_needed>",
                "Thanks, I will plan for indoors now.",
            ],
            "finalize": "Great, here is your indoor plan.",
        }
    )
    engine = WorkflowEngine(repo, store, llm)

    initial_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Plan my day")],
        model="test-model",
        stream=True,
        use_workflow="preference_flow",
        workflow_execution_id="exec-feedback-no-use",
    )
    first_stream = await engine.start_or_resume_workflow(
        StubSession(), initial_request, None, None, None
    )
    assert first_stream is not None
    _ = [chunk async for chunk in first_stream]
    state = store.load_execution("exec-feedback-no-use")
    assert state.awaiting_feedback is True

    resume_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="I prefer indoors")],
        model="test-model",
        stream=True,
        workflow_execution_id="exec-feedback-no-use",
    )
    resume_stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    assert resume_stream is not None
    resume_chunks = [chunk async for chunk in resume_stream]
    resume_payload = "\n".join(resume_chunks)
    expected_tag = '<workflow_execution_id for="preference_flow">exec-feedback-no-use</workflow_execution_id>'
    assert expected_tag in resume_payload or "exec-feedback-no-use" in resume_payload
    assert "indoor plan" in resume_payload.lower()


@pytest.mark.asyncio
async def test_resume_continue_placeholder_does_not_append_user_message(
    tmp_path, monkeypatch
):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "simple_flow",
        "root_intent": "SIMPLE",
        "agents": [{"agent": "finalize", "description": "Finalize."}],
    }
    _write_workflow(workflows_dir, "simple", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    repo = WorkflowRepository()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    llm = StubLLMClient({"finalize": "All set."})
    engine = WorkflowEngine(repo, store, llm)

    initial_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Start the flow")],
        model="test-model",
        stream=True,
        use_workflow="simple_flow",
        workflow_execution_id="exec-continue",
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), initial_request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    state = store.load_execution("exec-continue")
    assert state.context.get("user_messages") == ["Start the flow"]
    assert state.context.get("user_query") == "Start the flow"

    resume_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="[continue]")],
        model="test-model",
        stream=True,
        use_workflow="simple_flow",
        workflow_execution_id="exec-continue",
    )
    resume_stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    resume_chunks = [chunk async for chunk in resume_stream]
    resume_payload = "\n".join(resume_chunks)
    assert "<workflow_execution_id" not in resume_payload

    resumed_state = store.load_execution("exec-continue")
    assert resumed_state.context.get("user_messages") == ["Start the flow"]
    assert resumed_state.context.get("user_query") == "Start the flow"


@pytest.mark.asyncio
async def test_feedback_pause_notice_emits_once(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "pause_flow",
        "root_intent": "ASK_PREFERENCE",
        "agents": [{"agent": "ask_preference", "description": "Ask for preference."}],
    }
    _write_workflow(workflows_dir, "pause", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    repo = WorkflowRepository()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    llm = StubLLMClient(
        {
            "ask_preference": [
                "<user_feedback_needed>Please tell me if you prefer indoors</user_feedback_needed>"
            ]
        }
    )
    engine = WorkflowEngine(repo, store, llm)

    initial_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Plan my day")],
        model="test-model",
        stream=True,
        use_workflow="pause_flow",
        workflow_execution_id="exec-pause",
    )
    first_stream = await engine.start_or_resume_workflow(
        StubSession(), initial_request, None, None, None
    )
    _ = [chunk async for chunk in first_stream]

    paused_state = store.load_execution("exec-pause")
    assert paused_state.awaiting_feedback is True
    initial_event_count = len(paused_state.events)

    resume_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="[continue]")],
        model="test-model",
        stream=True,
        use_workflow="pause_flow",
        workflow_execution_id="exec-pause",
    )
    resume_stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    resume_payload = "\n".join([chunk async for chunk in resume_stream])
    assert "Workflow paused awaiting user feedback." in resume_payload

    mid_state = store.load_execution("exec-pause")
    assert len(mid_state.events) == initial_event_count

    second_stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    second_payload = "\n".join([chunk async for chunk in second_stream])
    assert "Workflow paused awaiting user feedback." not in second_payload


@pytest.mark.asyncio
async def test_looping_workflow_allows_multiple_turns(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "loop_flow",
        "root_intent": "CHAT",
        "loop": True,
        "agents": [
            {
                "agent": "chat",
                "description": "General conversation agent.",
                "pass_through": True,
            }
        ],
    }
    _write_workflow(workflows_dir, "loop", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    repo = WorkflowRepository()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    llm = StubLLMClient({"chat": ["Hello!", "Sure."]})
    engine = WorkflowEngine(repo, store, llm)

    exec_id = "exec-loop-chat"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Hi")],
        model="test-model",
        stream=True,
        use_workflow="loop_flow",
        workflow_execution_id=exec_id,
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]
    assert chunks and "[DONE]" in chunks[-1]

    state = store.load_execution(exec_id)
    assert state.completed is False
    assert state.awaiting_feedback is False
    assert state.context.get("user_messages") == ["Hi"]
    assert state.context.get("assistant_messages") == ["Hello!"]

    resume_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Tell me more")],
        model="test-model",
        stream=True,
        use_workflow="loop_flow",
        workflow_execution_id=exec_id,
    )
    resume_stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    _ = [chunk async for chunk in resume_stream]

    final_state = store.load_execution(exec_id)
    assert final_state.context.get("user_messages") == ["Hi", "Tell me more"]
    assert final_state.context.get("assistant_messages") == ["Hello!", "Sure."]
    assert llm.calls.count("chat") == 2


@pytest.mark.asyncio
async def test_resume_can_return_full_state_on_request(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "preference_flow",
        "root_intent": "ASK_PREFERENCE",
        "agents": [
            {"agent": "ask_preference", "description": "Ask for preference."},
            {
                "agent": "finalize",
                "description": "Summarize the plan.",
                "depends_on": ["ask_preference"],
            },
        ],
    }
    _write_workflow(workflows_dir, "pref", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    repo = WorkflowRepository()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    llm = StubLLMClient(
        {
            "ask_preference": [
                "<user_feedback_needed>Please tell me if you prefer indoors</user_feedback_needed>",
                "Thanks, I will plan for indoors now.",
            ],
            "finalize": "Great, here is your indoor plan.",
        }
    )
    engine = WorkflowEngine(repo, store, llm)

    initial_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Plan my day")],
        model="test-model",
        stream=True,
        use_workflow="preference_flow",
        workflow_execution_id="exec-feedback-full",
    )

    first_stream = await engine.start_or_resume_workflow(
        StubSession(), initial_request, None, None, None
    )
    assert first_stream is not None
    _ = [chunk async for chunk in first_stream]

    state = store.load_execution("exec-feedback-full")
    assert state.awaiting_feedback is True

    resume_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="I prefer indoors")],
        model="test-model",
        stream=True,
        use_workflow="preference_flow",
        workflow_execution_id="exec-feedback-full",
        return_full_state=True,
    )
    resume_stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    assert resume_stream is not None
    resume_chunks = [chunk async for chunk in resume_stream]
    resume_payload = "\n".join(resume_chunks)

    assert "routing workflow preference_flow" in resume_payload.lower()
    assert "indoor plan" in resume_payload.lower()
    final_state = store.load_execution("exec-feedback-full")
    assert final_state.completed is True
    assert final_state.awaiting_feedback is False


@pytest.mark.asyncio
async def test_feedback_resume_without_client_history_uses_state(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "notification_flow",
        "root_intent": "TEST_NOTIFICATIONS",
        "agents": [
            {
                "agent": "ask_channel",
                "description": "Ask which Teams channel to notify.",
            },
            {
                "agent": "finalize",
                "description": "Confirm notification target.",
                "depends_on": ["ask_channel"],
            },
        ],
    }
    _write_workflow(workflows_dir, "notify", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "ask_channel": [
                "<user_feedback_needed>Which Teams channel should I use?</user_feedback_needed>",
                "Great, I'll use that Teams channel.",
            ],
            "finalize": "Notifications configured.",
        }
    )
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    exec_id = "exec-notify"
    initial_request = ChatCompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content="Set up Teams notifications for my monitoring plan.",
            )
        ],
        model="test-model",
        stream=True,
        use_workflow="notification_flow",
        workflow_execution_id=exec_id,
    )

    first_stream = await engine.start_or_resume_workflow(
        StubSession(), initial_request, None, None, None
    )
    assert first_stream is not None
    _ = [chunk async for chunk in first_stream]

    state = engine.state_store.load_execution(exec_id)
    assert state.awaiting_feedback is True
    assert state.context.get("user_messages") == [
        "Set up Teams notifications for my monitoring plan."
    ]

    resume_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Use chat ID 12345")],
        model="test-model",
        stream=True,
        use_workflow="notification_flow",
        workflow_execution_id=exec_id,
    )
    resume_stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    assert resume_stream is not None
    resume_chunks = [chunk async for chunk in resume_stream]
    resume_payload = "\n".join(resume_chunks)
    assert "notifications configured" in resume_payload.lower()

    final_state = engine.state_store.load_execution(exec_id)
    assert final_state.completed is True
    assert final_state.awaiting_feedback is False
    assert final_state.context.get("user_messages") == [
        "Set up Teams notifications for my monitoring plan.",
        "Use chat ID 12345",
    ]

    ask_calls = [idx for idx, call in enumerate(llm.calls) if call == "ask_channel"]
    assert len(ask_calls) == 2
    resumed_user_content = llm.user_messages[ask_calls[1]]
    assert "Set up Teams notifications for my monitoring plan." in resumed_user_content
    assert "Use chat ID 12345" in resumed_user_content
    assert llm.calls[-1] == "finalize"


@pytest.mark.asyncio
async def test_feedback_updates_user_query_with_selection(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "newsletter_flow",
        "root_intent": "RUN_NEWSLETTER",
        "agents": [
            {"agent": "select_newsletter", "description": "Select newsletter."},
            {
                "agent": "finalize",
                "description": "Finalize newsletter run.",
                "depends_on": ["select_newsletter"],
            },
        ],
    }
    _write_workflow(workflows_dir, "newsletter", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "select_newsletter": [
                "<user_feedback_needed>I found two, Weekly Security Newsletter and Monthly Security newsletter. Which one?</user_feedback_needed>",
                "Running it now.",
            ],
            "finalize": "Done.",
        },
        ask_responses={"user_query_summary": "Run the Weekly Security Newsletter"},
    )
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    exec_id = "exec-newsletter"
    initial_request = ChatCompletionRequest(
        messages=[
            Message(role=MessageRole.USER, content="Run Security Newsletter now")
        ],
        model="test-model",
        stream=True,
        use_workflow="newsletter_flow",
        workflow_execution_id=exec_id,
    )

    first_stream = await engine.start_or_resume_workflow(
        StubSession(), initial_request, None, None, None
    )
    assert first_stream is not None
    _ = [chunk async for chunk in first_stream]

    state = engine.state_store.load_execution(exec_id)
    assert state.awaiting_feedback is True

    resume_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="The first one")],
        model="test-model",
        stream=True,
        use_workflow="newsletter_flow",
        workflow_execution_id=exec_id,
    )
    resume_stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    assert resume_stream is not None
    _ = [chunk async for chunk in resume_stream]

    updated_state = engine.state_store.load_execution(exec_id)
    user_query = updated_state.context.get("user_query", "")
    assert user_query == "Run the Weekly Security Newsletter"
    assert llm.ask_calls


@pytest.mark.asyncio
async def test_feedback_confirmation_skips_agent_rerun(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "confirm_flow",
        "root_intent": "CONFIRM_ACTION",
        "agents": [
            {"agent": "perform_action", "description": "Perform action."},
            {
                "agent": "finalize",
                "description": "Finalize the workflow.",
                "depends_on": ["perform_action"],
            },
        ],
    }
    _write_workflow(workflows_dir, "confirm", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "perform_action": [
                "<user_feedback_needed>I synced project Alpha. Does that look right?</user_feedback_needed>"
            ],
            "finalize": "Done.",
        },
        ask_responses={
            "user_query_summary": "Confirm the sync for project Alpha",
            "feedback_rerun_decision": "USE_PREVIOUS",
        },
    )
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    exec_id = "exec-confirm"
    initial_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Sync project Alpha")],
        model="test-model",
        stream=True,
        use_workflow="confirm_flow",
        workflow_execution_id=exec_id,
    )

    first_stream = await engine.start_or_resume_workflow(
        StubSession(), initial_request, None, None, None
    )
    assert first_stream is not None
    _ = [chunk async for chunk in first_stream]

    state = engine.state_store.load_execution(exec_id)
    assert state.awaiting_feedback is True

    resume_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Yes, that's correct.")],
        model="test-model",
        stream=True,
        use_workflow="confirm_flow",
        workflow_execution_id=exec_id,
    )
    resume_stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    assert resume_stream is not None
    _ = [chunk async for chunk in resume_stream]

    assert llm.calls.count("perform_action") == 1
    assert llm.calls[-1] == "finalize"


@pytest.mark.asyncio
async def test_on_tool_error_reroutes_to_failure_handler(tmp_path, monkeypatch):
    """Test that on_tool_error triggers automatic reroute when tool fails."""
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "save_plan_flow",
        "root_intent": "SAVE_PLAN",
        "agents": [
            {
                "agent": "save_plan",
                "description": "Save the plan using the save_plan tool.",
                "tools": ["save_plan"],
                "on_tool_error": "summarize_failure",
            },
            {
                "agent": "summarize_failure",
                "description": "Summarize that save failed and provide options.",
                "tools": [],
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "save_plan",
                "description": "Save a plan",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    class ToolErrorLLM(StubLLMClient):
        def __init__(self):
            super().__init__(
                {"summarize_failure": "Sorry, the save failed. Please try again."}
            )
            self.call_count = 0

        def stream_completion(self, request, access_token, span):
            self.call_count += 1
            last_message = request.messages[-1].content or ""

            # Routing intent check
            if "ROUTING_INTENT_CHECK" in last_message:

                async def _gen():
                    yield 'data: {"choices":[{"delta":{"content":""},"index":0}]}\n\n'
                    yield "data: [DONE]\n\n"

                return _gen()

            # Agent save_plan - make a tool call that will fail
            if "<agent:save_plan>" in last_message:

                async def _gen_tool_call():
                    yield (
                        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                        '"function":{"name":"save_plan","arguments":"{}"}}]},"index":0}]}\n\n'
                    )
                    yield "data: [DONE]\n\n"

                return _gen_tool_call()

            # After tool error - LLM just responds normally (doesn't emit reroute)
            if self.call_count > 2 and "save_plan" in last_message.lower():

                async def _gen_after_error():
                    yield 'data: {"choices":[{"delta":{"content":"I tried to save."},"index":0}]}\n\n'
                    yield "data: [DONE]\n\n"

                return _gen_after_error()

            # Summarize failure agent
            if "<agent:summarize_failure>" in last_message:

                async def _gen_summarize():
                    yield 'data: {"choices":[{"delta":{"content":"Sorry, the save failed."},"index":0}]}\n\n'
                    yield "data: [DONE]\n\n"

                return _gen_summarize()

            return super().stream_completion(request, access_token, span)

    from app.tgi.services.tool_service import ToolService

    tool_service = ToolService()

    async def mock_execute_tool_calls(*args, **kwargs):
        # Simulate a tool error - return raw_results with error content
        error_result = {
            "name": "save_plan",
            "tool_call_id": "call_1",
            "content": '{"error": "Client error \'400 Bad Request\'"}',
        }
        from app.tgi.models import Message, MessageRole

        tool_message = Message(
            role=MessageRole.TOOL,
            content='{"error": "Client error \'400 Bad Request\'"}',
            name="save_plan",
            tool_call_id="call_1",
        )
        if kwargs.get("return_raw_results"):
            return ([tool_message], False, [error_result])
        return ([tool_message], False)

    tool_service.execute_tool_calls = mock_execute_tool_calls

    llm = ToolErrorLLM()
    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
        tool_service=tool_service,
    )

    exec_id = "exec-tool-error"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="save my plan")],
        model="test-model",
        stream=True,
        use_workflow="save_plan_flow",
        workflow_execution_id=exec_id,
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(tools=tools), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]
    payload = "\n".join(chunks)

    state = engine.state_store.load_execution(exec_id)

    # Verify tool error was detected and stored
    assert "tool_errors" in state.context["agents"]["save_plan"]
    assert len(state.context["agents"]["save_plan"]["tool_errors"]) == 1
    assert state.context["agents"]["save_plan"]["tool_errors"][0]["name"] == "save_plan"

    # Verify reroute happened
    assert state.context["agents"]["save_plan"]["reroute_reason"] == "TOOL_ERROR"
    assert "summarize_failure" in state.context["agents"]

    # Verify reroute message in output
    assert "rerouting" in payload.lower() or "summarize_failure" in payload.lower()


@pytest.mark.asyncio
async def test_tool_success_reroutes_via_reroute_config(tmp_path, monkeypatch):
    """Tool success triggers reroute config without explicit <reroute> output."""
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "tool_reroute_success",
        "root_intent": "GET_PLAN",
        "agents": [
            {
                "agent": "get_plan",
                "description": "Fetch the plan using tools.",
                "tools": ["get_plan"],
                "reroute": [
                    {"on": ["tool:get_plan:success"], "to": "analyse_plan"},
                    {"on": ["tool:get_plan:error"], "to": "get_plan_failed"},
                ],
            },
            {
                "agent": "analyse_plan",
                "description": "Analyse the plan.",
                "tools": [],
                "when": "context.get('agents', {}).get('get_plan', {}).get('reroute_reason') == 'tool:get_plan:success'",
            },
            {
                "agent": "get_plan_failed",
                "description": "Handle tool failure.",
                "tools": [],
                "when": "context.get('agents', {}).get('get_plan', {}).get('reroute_reason') == 'tool:get_plan:error'",
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_plan",
                "description": "Fetch a plan",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    class ToolSuccessLLM(StubLLMClient):
        def __init__(self):
            super().__init__(
                {
                    "analyse_plan": "Analysed plan.",
                    "get_plan_failed": "Plan fetch failed.",
                }
            )
            self.tool_payload_seen = False

        def stream_completion(self, request, access_token, span):
            last_message = request.messages[-1].content or ""
            if any(m.role == MessageRole.TOOL for m in request.messages):
                self.tool_payload_seen = True

            if "ROUTING_INTENT_CHECK" in last_message:

                async def _gen():
                    yield 'data: {"choices":[{"delta":{"content":""},"index":0}]}\n\n'
                    yield "data: [DONE]\n\n"

                return _gen()

            if any(m.role == MessageRole.TOOL for m in request.messages):

                async def _gen_after_tool():
                    yield 'data: {"choices":[{"delta":{"content":"Plan retrieved."},"index":0}]}\n\n'
                    yield "data: [DONE]\n\n"

                return _gen_after_tool()

            if "<agent:get_plan>" in last_message:

                async def _gen_tool_call():
                    yield (
                        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                        '"function":{"name":"get_plan","arguments":"{}"}}]},"index":0}]}\n\n'
                    )
                    yield "data: [DONE]\n\n"

                return _gen_tool_call()

            if "<agent:analyse_plan>" in last_message:

                async def _gen_analyse():
                    yield 'data: {"choices":[{"delta":{"content":"Analysis complete."},"index":0}]}\n\n'
                    yield "data: [DONE]\n\n"

                return _gen_analyse()

            if "<agent:get_plan_failed>" in last_message:

                async def _gen_failed():
                    yield 'data: {"choices":[{"delta":{"content":"Failed to get plan."},"index":0}]}\n\n'
                    yield "data: [DONE]\n\n"

                return _gen_failed()

            return super().stream_completion(request, access_token, span)

    tool_service = ToolService()

    async def mock_execute_tool_calls(*_args, **kwargs):
        success_content = '{"success": true, "plan": {"id": "plan-1"}}'
        tool_message = Message(
            role=MessageRole.TOOL,
            content=success_content,
            name="get_plan",
            tool_call_id="call_1",
        )
        raw_result = {
            "name": "get_plan",
            "tool_call_id": "call_1",
            "content": success_content,
        }
        if kwargs.get("return_raw_results"):
            return ([tool_message], True, [raw_result])
        return ([tool_message], True)

    tool_service.execute_tool_calls = mock_execute_tool_calls

    llm = ToolSuccessLLM()
    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
        tool_service=tool_service,
    )

    exec_id = "exec-tool-success"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="get my plan")],
        model="test-model",
        stream=True,
        use_workflow="tool_reroute_success",
        workflow_execution_id=exec_id,
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(tools=tools), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]
    payload = "\n".join(chunks)

    state = engine.state_store.load_execution(exec_id)

    assert (
        state.context["agents"]["get_plan"]["reroute_reason"] == "tool:get_plan:success"
    )
    assert state.context["agents"]["analyse_plan"]["completed"] is True
    assert "rerouting to analyse_plan" in payload.lower()
    assert llm.tool_payload_seen is False


@pytest.mark.asyncio
async def test_tool_error_reroute_config_precedes_on_tool_error(tmp_path, monkeypatch):
    """Tool error triggers reroute config before on_tool_error fallback."""
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "tool_reroute_error",
        "root_intent": "GET_PLAN",
        "agents": [
            {
                "agent": "get_plan",
                "description": "Fetch the plan using tools.",
                "tools": ["get_plan"],
                "reroute": [{"on": ["tool:get_plan:error"], "to": "get_plan_failed"}],
                "on_tool_error": "fallback_failure",
            },
            {
                "agent": "get_plan_failed",
                "description": "Handle tool failure.",
                "tools": [],
                "when": "context.get('agents', {}).get('get_plan', {}).get('reroute_reason') == 'tool:get_plan:error'",
            },
            {
                "agent": "fallback_failure",
                "description": "Fallback error handler.",
                "tools": [],
                "when": "context.get('agents', {}).get('get_plan', {}).get('reroute_reason') == 'TOOL_ERROR'",
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_plan",
                "description": "Fetch a plan",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    class ToolErrorRerouteLLM(StubLLMClient):
        def __init__(self):
            super().__init__(
                {
                    "get_plan_failed": "Plan failed.",
                    "fallback_failure": "Fallback failed.",
                }
            )
            self.tool_payload_seen = False

        def stream_completion(self, request, access_token, span):
            last_message = request.messages[-1].content or ""
            if any(m.role == MessageRole.TOOL for m in request.messages):
                self.tool_payload_seen = True

            if "ROUTING_INTENT_CHECK" in last_message:

                async def _gen():
                    yield 'data: {"choices":[{"delta":{"content":""},"index":0}]}\n\n'
                    yield "data: [DONE]\n\n"

                return _gen()

            if any(m.role == MessageRole.TOOL for m in request.messages):

                async def _gen_after_tool():
                    yield 'data: {"choices":[{"delta":{"content":"Tried tool."},"index":0}]}\n\n'
                    yield "data: [DONE]\n\n"

                return _gen_after_tool()

            if "<agent:get_plan>" in last_message:

                async def _gen_tool_call():
                    yield (
                        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                        '"function":{"name":"get_plan","arguments":"{}"}}]},"index":0}]}\n\n'
                    )
                    yield "data: [DONE]\n\n"

                return _gen_tool_call()

            if "<agent:get_plan_failed>" in last_message:

                async def _gen_failed():
                    yield 'data: {"choices":[{"delta":{"content":"Handled error."},"index":0}]}\n\n'
                    yield "data: [DONE]\n\n"

                return _gen_failed()

            if "<agent:fallback_failure>" in last_message:

                async def _gen_fallback():
                    yield 'data: {"choices":[{"delta":{"content":"Fallback handler."},"index":0}]}\n\n'
                    yield "data: [DONE]\n\n"

                return _gen_fallback()

            return super().stream_completion(request, access_token, span)

    tool_service = ToolService()

    async def mock_execute_tool_calls(*_args, **kwargs):
        error_content = '{"error": "Boom"}'
        tool_message = Message(
            role=MessageRole.TOOL,
            content=error_content,
            name="get_plan",
            tool_call_id="call_1",
        )
        raw_result = {
            "name": "get_plan",
            "tool_call_id": "call_1",
            "content": error_content,
        }
        if kwargs.get("return_raw_results"):
            return ([tool_message], False, [raw_result])
        return ([tool_message], False)

    tool_service.execute_tool_calls = mock_execute_tool_calls

    llm = ToolErrorRerouteLLM()
    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
        tool_service=tool_service,
    )

    exec_id = "exec-tool-error-config"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="get my plan")],
        model="test-model",
        stream=True,
        use_workflow="tool_reroute_error",
        workflow_execution_id=exec_id,
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(tools=tools), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]
    payload = "\n".join(chunks)

    state = engine.state_store.load_execution(exec_id)

    assert (
        state.context["agents"]["get_plan"]["reroute_reason"] == "tool:get_plan:error"
    )
    assert state.context["agents"]["get_plan_failed"]["completed"] is True
    assert state.context["agents"]["fallback_failure"].get("skipped") is True
    assert "rerouting to get_plan_failed" in payload.lower()
    assert llm.tool_payload_seen is False


@pytest.mark.asyncio
async def test_missing_returns_retry_then_abort(tmp_path, monkeypatch):
    """Agents with required returns should retry up to 3 times before aborting."""
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "missing_returns_flow",
        "root_intent": "PLAN",
        "agents": [
            {
                "agent": "collect_plan",
                "description": "Collect plan output",
                "returns": [{"field": "payload.result", "from": "plan", "as": "plan"}],
            }
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient({"collect_plan": ["no data", "still nothing", "empty"]})
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    exec_id = "exec-missing-returns"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="gather plan")],
        model="test-model",
        stream=True,
        use_workflow="missing_returns_flow",
        workflow_execution_id=exec_id,
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]
    payload = "\n".join(chunks)

    # Retry events surfaced and final abort after 3 attempts
    assert "Retrying (attempt 1/3" in payload
    assert "after 3 attempts" in payload
    state = engine.state_store.load_execution(exec_id)
    assert state.completed is True
    assert state.context["agents"]["collect_plan"]["return_attempts"] == 3
    assert llm.calls.count("collect_plan") == 3


@pytest.mark.asyncio
async def test_missing_returns_with_error_aborts_immediately(tmp_path, monkeypatch):
    """If missing returns include a fatal error, abort without retrying."""
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "missing_returns_error",
        "root_intent": "PLAN",
        "agents": [
            {
                "agent": "collect_plan",
                "description": "Collect plan output",
                "returns": [{"field": "payload.result", "from": "plan", "as": "plan"}],
            }
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient({"collect_plan": "Internal Server Error while planning"})
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    exec_id = "exec-missing-returns-error"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="gather plan")],
        model="test-model",
        stream=True,
        use_workflow="missing_returns_error",
        workflow_execution_id=exec_id,
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]
    payload = "\n".join(chunks)

    assert "non-recoverable error" in payload.lower()
    assert "needs to be fixed" in payload.lower()
    assert "internal server error" in payload.lower()
    assert "retrying" not in payload.lower()
    state = engine.state_store.load_execution(exec_id)
    assert state.completed is True
    assert state.context["agents"]["collect_plan"]["return_attempts"] == 1


@pytest.mark.asyncio
async def test_streaming_tool_returns_are_captured_without_retry(tmp_path, monkeypatch):
    """Streaming tool results should satisfy returns without triggering retries."""
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "stream_returns",
        "root_intent": "PLAN",
        "agents": [
            {
                "agent": "create_plan",
                "description": "Create plan",
                "tools": [
                    {
                        "plan": {
                            "settings": {"streaming": True},
                        }
                    }
                ],
                "returns": [{"field": "payload.result", "from": "plan", "as": "plan"}],
            }
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "plan",
                "description": "Create plan",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    class StreamingToolLLM(StubLLMClient):
        def __init__(self):
            super().__init__({})
            self.call_count = 0

        def stream_completion(self, request, access_token, span):
            self.call_count += 1

            async def _gen_first():
                yield (
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"plan","arguments":"{}"}}]},"index":0}]}\n\n'
                )
                yield "data: [DONE]\n\n"

            async def _gen_second():
                yield 'data: {"choices":[{"delta":{"content":"done"},"index":0}]}\n\n'
                yield "data: [DONE]\n\n"

            return _gen_first() if self.call_count == 1 else _gen_second()

    class StreamingSession(StubSession):
        async def call_tool_streaming(
            self, name: str, args: dict[str, Any], access_token: str
        ):
            async def stream():
                yield {"type": "progress", "progress": 0.5, "message": "halfway"}
                yield {
                    "type": "result",
                    "data": {"payload": {"result": {"id": "plan-42"}}},
                }

            return stream()

    llm = StreamingToolLLM()
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    exec_id = "exec-stream-returns"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="create a plan")],
        model="test-model",
        stream=True,
        use_workflow="stream_returns",
        workflow_execution_id=exec_id,
    )

    session = StreamingSession(tools=tools)
    stream = await engine.start_or_resume_workflow(session, request, None, None, None)
    chunks = [chunk async for chunk in stream]
    payload = "\n".join(chunks)

    assert "retry" not in payload.lower()
    state = engine.state_store.load_execution(exec_id)
    assert state.completed is True
    agent_ctx = state.context["agents"]["create_plan"]
    assert agent_ctx.get("plan") == {"id": "plan-42"}
    assert "return_attempts" not in agent_ctx
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_missing_returns_recoverable_error_retries_with_summary(
    tmp_path, monkeypatch
):
    """Recoverable errors should retry and include the error summary."""
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "missing_returns_recoverable",
        "root_intent": "PLAN",
        "agents": [
            {
                "agent": "collect_plan",
                "description": "Collect plan output",
                "returns": [{"field": "payload.result", "from": "plan", "as": "plan"}],
            }
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "collect_plan": [
                "temporary error: timeout",
                "temporary error: timeout",
                "temporary error: timeout",
            ]
        }
    )
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    exec_id = "exec-missing-returns-recoverable"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="gather plan")],
        model="test-model",
        stream=True,
        use_workflow="missing_returns_recoverable",
        workflow_execution_id=exec_id,
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]
    payload = "\n".join(chunks)

    assert "retrying (attempt 1/3" in payload.lower()
    assert "temporary error" in payload.lower()
    assert "needs to be fixed" in payload.lower()
    state = engine.state_store.load_execution(exec_id)
    assert state.completed is True
    assert state.context["agents"]["collect_plan"]["return_attempts"] == 3
    assert llm.calls.count("collect_plan") == 3


@pytest.mark.asyncio
async def test_missing_returns_with_tools_without_results_aborts(tmp_path, monkeypatch):
    """Agents with returns and tools still fail when no tool output is captured."""
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "returns_with_tools",
        "root_intent": "PLAN",
        "agents": [
            {
                "agent": "create_plan",
                "description": "Create plan",
                "tools": ["plan"],
                "returns": [{"field": "payload.result", "from": "plan", "as": "plan"}],
            }
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "plan",
                "description": "Plan",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    llm = StubLLMClient({"create_plan": "no tool call"})
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    exec_id = "exec-returns-tools-missing"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="make a plan")],
        model="test-model",
        stream=True,
        use_workflow="returns_with_tools",
        workflow_execution_id=exec_id,
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(tools=tools), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]
    payload = "\n".join(chunks)

    assert "needs to be fixed" in payload.lower()
    assert "after 3 attempts" in payload.lower()
    state = engine.state_store.load_execution(exec_id)
    assert state.completed is True
    assert state.context["agents"]["create_plan"]["return_attempts"] == 3
    assert llm.calls.count("create_plan") == 3


@pytest.mark.asyncio
async def test_arg_mapping_works_when_context_disabled(tmp_path, monkeypatch):
    """
    Even when an agent hides context from the LLM (context: false),
    arg mappings should still resolve from the full workflow state.
    """
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "save_plan_flow",
        "root_intent": "PLAN",
        "agents": [
            {
                "agent": "select_tools",
                "description": "Select tools",
                "tools": ["select_tools"],
                "returns": ["selected_tools"],
            },
            {
                "agent": "create_plan",
                "description": "Create plan",
                "tools": [
                    {
                        "plan": {
                            "args": {"selected_tools": "select_tools.selected_tools"},
                        }
                    }
                ],
                "depends_on": ["select_tools"],
                "returns": [{"field": "payload.result", "from": "plan", "as": "plan"}],
            },
            {
                "agent": "save_plan",
                "description": "Save plan",
                "context": False,
                "tools": [
                    {
                        "save_plan": {
                            "args": {"plan": "create_plan.plan"},
                        }
                    }
                ],
                "depends_on": ["create_plan"],
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "select_tools",
                "description": "Select tools",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "plan",
                "description": "Plan",
                "parameters": {
                    "type": "object",
                    "properties": {"selected_tools": {"type": "array"}},
                    "required": ["selected_tools"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "save_plan",
                "description": "Save plan",
                "parameters": {"type": "object", "properties": {"plan": {}}},
            },
        },
    ]

    class MappingLLM(StubLLMClient):
        def __init__(self):
            super().__init__(
                responses={
                    "select_tools": "<passthrough>Selecting</passthrough>",
                    "create_plan": "<passthrough>Created</passthrough>",
                    "save_plan": "<passthrough>Saved</passthrough>",
                }
            )
            self._call_sequence = {}

        def stream_completion(self, request, access_token, span):
            last_message = request.messages[-1].content or ""
            agent_marker = ""
            if "<agent:" in last_message:
                agent_marker = last_message.split("<agent:", 1)[1].split(">", 1)[0]

            # Emit a tool call for each agent once
            if agent_marker and agent_marker not in self._call_sequence:
                self._call_sequence[agent_marker] = 1

                tool_name = agent_marker
                if agent_marker == "create_plan":
                    tool_name = "plan"
                if agent_marker == "select_tools":
                    tool_name = "select_tools"
                if agent_marker == "save_plan":
                    tool_name = "save_plan"

                async def _gen_tool_call():
                    import json

                    payload = {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "function": {
                                                "name": tool_name,
                                                "arguments": "{}",
                                            },
                                        }
                                    ]
                                },
                                "index": 0,
                            }
                        ]
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    yield "data: [DONE]\n\n"

                return _gen_tool_call()

            return super().stream_completion(request, access_token, span)

    llm = MappingLLM()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="run")],
        model="test-model",
        stream=True,
        use_workflow="save_plan_flow",
        workflow_execution_id="exec-ctx-false-args",
    )

    class CaptureSession(StubSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tool_calls = []

        async def call_tool(self, name: str, args: dict, access_token: str):
            self.tool_calls.append({"name": name, "args": args})
            result_payload = {"result": f"{name}_result"}
            if name == "select_tools":
                result_payload = {"selected_tools": ["tool_a", "tool_b"]}
            if name == "plan":
                result_payload = {"payload": {"result": {"id": "plan-123"}}}
            return SimpleNamespace(
                isError=False,
                content=[
                    SimpleNamespace(text=json.dumps(result_payload, ensure_ascii=False))
                ],
            )

    # Pre-seed select_tools output so arg injection has data even though context is hidden
    seeded_state = store.get_or_create("exec-ctx-false-args", "save_plan_flow")
    seeded_state.context["agents"]["select_tools"] = {
        "selected_tools": ["tool_a", "tool_b"],
        "completed": True,
        "content": "",
    }
    seeded_state.context["agents"]["create_plan"] = {
        "plan": {"id": "plan-123"},
        "completed": True,
        "content": "",
    }
    store.save_state(seeded_state)

    session = CaptureSession(tools=tools)
    stream = await engine.start_or_resume_workflow(session, request, None, None, None)
    _ = [chunk async for chunk in stream]

    # The save_plan tool should receive injected plan arg despite context False
    save_calls = [c for c in session.tool_calls if c["name"] == "save_plan"]
    assert save_calls, "save_plan tool was not called"
    assert save_calls[0]["args"].get("plan") is not None


@pytest.mark.asyncio
async def test_tool_error_detection_various_formats(tmp_path, monkeypatch):
    """Test that _tool_result_has_error detects various error formats."""
    from app.tgi.workflows.error_analysis import tool_result_has_error

    # JSON with error key
    assert tool_result_has_error('{"error": "something went wrong"}') is True

    # JSON with isError flag
    assert tool_result_has_error('{"isError": true}') is True

    # JSON with success false
    assert tool_result_has_error('{"success": false}') is True

    # List with error item
    assert tool_result_has_error('[{"error": "fail"}]') is True
    # MCP-style content[] text wrapping JSON error
    assert (
        tool_result_has_error('[{"text":"{\\"error\\":\\"wrapped fail\\"}"}]') is True
    )

    # Plain text with HTTP error
    assert tool_result_has_error("Client error '400 Bad Request'") is True
    assert tool_result_has_error("500 Internal Server Error") is True
    assert tool_result_has_error("401 unauthorized access") is True

    # Normal successful responses
    assert tool_result_has_error('{"result": "success"}') is False
    assert (
        tool_result_has_error(
            '[{"text":"{\\"payload\\":{\\"result\\":{\\"id\\":\\"p1\\"}}"}]'
        )
        is False
    )
    assert tool_result_has_error("Plan saved successfully") is False
    assert tool_result_has_error('{"data": [1, 2, 3]}') is False
    assert tool_result_has_error("") is False


@pytest.mark.asyncio
async def test_explicit_reroute_takes_precedence_over_on_tool_error(
    tmp_path, monkeypatch
):
    """Test that LLM's explicit reroute tag takes precedence over on_tool_error."""
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    # The workflow design is important:
    # - first agent has on_tool_error pointing to error_handler
    # - first agent also has explicit reroute config for CUSTOM_FAILURE -> custom_handler
    # - When tool fails AND LLM emits <reroute>CUSTOM_FAILURE</reroute>, explicit reroute should win
    # - Both error_handler and custom_handler depend on first, so only the routed one runs
    flow = {
        "flow_id": "reroute_precedence_flow",
        "root_intent": "TEST",
        "agents": [
            {
                "agent": "first",
                "description": "First agent",
                "tools": ["some_tool"],
                "on_tool_error": "error_handler",
                "reroute": {"on": ["CUSTOM_FAILURE"], "to": "custom_handler"},
            },
            {
                "agent": "error_handler",
                "description": "Generic error handler",
                "tools": [],
                "depends_on": ["first"],
                "when": "context.get('agents', {}).get('first', {}).get('reroute_reason') == 'TOOL_ERROR'",
            },
            {
                "agent": "custom_handler",
                "description": "Custom failure handler",
                "tools": [],
                "depends_on": ["first"],
                "when": "context.get('agents', {}).get('first', {}).get('reroute_reason') == 'CUSTOM_FAILURE'",
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "some_tool",
                "description": "Some tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    class ExplicitRerouteLLM(StubLLMClient):
        def __init__(self):
            super().__init__({})
            self.call_count = 0
            self.agent_call_counts = {}

        def stream_completion(self, request, access_token, span):
            self.call_count += 1

            # Check all messages to find agent context
            all_content = " ".join(m.content or "" for m in request.messages)
            last_message = request.messages[-1].content or ""

            if "ROUTING_INTENT_CHECK" in last_message:

                async def _gen():
                    yield 'data: {"choices":[{"delta":{"content":""},"index":0}]}\n\n'
                    yield "data: [DONE]\n\n"

                return _gen()

            # For "first" agent, check if we're in the tool loop (tool result in history)
            if "<agent:first>" in all_content:
                # Check if there's a tool result in the messages (indicating we're in retry)
                has_tool_result = any(
                    m.role.value == "tool"
                    for m in request.messages
                    if hasattr(m.role, "value")
                ) or any(
                    getattr(m, "name", None) == "some_tool" for m in request.messages
                )

                if not has_tool_result:
                    # First call: return tool call
                    async def _gen_tool_call():
                        yield (
                            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                            '"function":{"name":"some_tool","arguments":"{}"}}]},"index":0}]}\n\n'
                        )
                        yield "data: [DONE]\n\n"

                    return _gen_tool_call()
                else:
                    # After tool error: LLM emits explicit reroute
                    async def _gen_reroute():
                        yield 'data: {"choices":[{"delta":{"content":"<reroute>CUSTOM_FAILURE</reroute>"},"index":0}]}\n\n'
                        yield "data: [DONE]\n\n"

                    return _gen_reroute()

            # custom_handler and error_handler just return simple responses
            if "<agent:custom_handler>" in all_content:

                async def _gen():
                    yield 'data: {"choices":[{"delta":{"content":"Handling custom failure gracefully."},"index":0}]}\n\n'
                    yield "data: [DONE]\n\n"

                return _gen()

            if "<agent:error_handler>" in all_content:

                async def _gen():
                    yield 'data: {"choices":[{"delta":{"content":"Generic error handling."},"index":0}]}\n\n'
                    yield "data: [DONE]\n\n"

                return _gen()

            async def _gen():
                yield 'data: {"choices":[{"delta":{"content":"fallback"},"index":0}]}\n\n'
                yield "data: [DONE]\n\n"

            return _gen()

    from app.tgi.services.tool_service import ToolService

    tool_service = ToolService()

    async def mock_execute_tool_calls(*args, **kwargs):
        error_result = {
            "name": "some_tool",
            "tool_call_id": "call_1",
            "content": '{"error": "failed"}',
        }
        from app.tgi.models import Message, MessageRole

        tool_message = Message(
            role=MessageRole.TOOL,
            content='{"error": "failed"}',
            name="some_tool",
            tool_call_id="call_1",
        )
        if kwargs.get("return_raw_results"):
            return ([tool_message], False, [error_result])
        return ([tool_message], False)

    tool_service.execute_tool_calls = mock_execute_tool_calls

    llm = ExplicitRerouteLLM()
    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
        tool_service=tool_service,
    )

    exec_id = "exec-reroute-precedence"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="test")],
        model="test-model",
        stream=True,
        use_workflow="reroute_precedence_flow",
        workflow_execution_id=exec_id,
        persist_inner_thinking=True,
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(tools=tools), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    state = engine.state_store.load_execution(exec_id)

    assert state.context.get("_persist_inner_thinking") is True
    # Explicit reroute should take precedence - tool errors were detected but LLM's reroute wins
    assert state.context["agents"]["first"]["reroute_reason"] == "CUSTOM_FAILURE"
    assert (
        "tool_errors" in state.context["agents"]["first"]
    )  # Tool errors were still recorded

    # Custom handler should have been executed (routed to by explicit reroute)
    assert "custom_handler" in state.context["agents"]
    assert (
        state.context["agents"]["custom_handler"]["content"]
        == "Handling custom failure gracefully."
    )

    # error_handler should be skipped (condition not met since reroute_reason is CUSTOM_FAILURE)
    assert "error_handler" in state.context["agents"]
    assert state.context["agents"]["error_handler"].get("skipped") is True


@pytest.mark.asyncio
async def test_arg_resolution_failure_stops_workflow(tmp_path):
    """If argument resolution fails, the workflow should terminate immediately."""
    from app.tgi.workflows.engine import WorkflowEngine
    from app.tgi.workflows.models import (
        WorkflowAgentDef,
        WorkflowDefinition,
        WorkflowExecutionState,
    )
    from app.tgi.workflows.repository import WorkflowRepository
    from app.tgi.workflows.state import WorkflowStateStore
    from app.tgi.workflows.arg_injector import ArgResolutionError

    class FailingRunner:
        def stream_chat_with_tools(
            self,
            session,
            messages,
            available_tools,
            chat_request,
            access_token,
            parent_span,
            emit_think_messages=True,
            arg_injector=None,
            tools_for_validation=None,
            streaming_tools=None,
            stop_after_tool_results=None,
        ):
            async def _gen():
                # Simulate an argument mapping failure before any output is streamed
                raise ArgResolutionError(
                    tool_name="needs_args",
                    arg_name="properties",
                    source_path="generate_inputs.result",
                    available_context={"content": "only content present"},
                )

                yield  # pragma: no cover

            return _gen()

    class Session:
        async def list_tools(self):
            return []

    class DummyLLM:
        def stream_completion(self, *args, **kwargs):
            async def _gen():
                yield "data: [DONE]\n\n"

            return _gen()

    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    os.environ["WORKFLOWS_PATH"] = str(workflows_dir)

    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        DummyLLM(),
        tool_chat_runner=FailingRunner(),
    )

    # Two agents: the first will fail arg resolution, the second should never run
    workflow_def = WorkflowDefinition(
        flow_id="arg_resolution_flow",
        root_intent="ARG_FAIL",
        agents=[
            WorkflowAgentDef(
                agent="generate_inputs",
                description="Generate inputs",
                tools=[
                    {"needs_args": {"args": {"properties": "generate_inputs.result"}}}
                ],
            ),
            WorkflowAgentDef(agent="should_not_run", description="Should not run"),
        ],
    )

    state = WorkflowExecutionState.new("exec-arg-resolve", "arg_resolution_flow")
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="hi")],
        model="test-model",
        stream=True,
    )

    stream = engine._run_agents(
        workflow_def,
        state,
        Session(),
        request,
        None,
        access_token=None,
        span=None,
        no_reroute=False,
    )
    _ = [chunk async for chunk in stream]

    assert state.completed is True
    # First agent context exists with error content; second agent never ran
    assert "generate_inputs" in state.context.get("agents", {})
    assert "should_not_run" not in state.context.get("agents", {})


@pytest.mark.asyncio
async def test_stop_point_halts_workflow_execution(tmp_path, monkeypatch):
    """Test that a stop_point agent halts execution and prevents subsequent agents from running."""
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "workflow_with_stop",
        "root_intent": "TEST_STOP_POINT",
        "agents": [
            {
                "agent": "summarize_result",
                "description": "Summarize result",
                "stop_point": True,
            },
            {
                "agent": "should_not_run",
                "description": "This should not run",
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "summarize_result": "Summary complete",
            "should_not_run": "This should not be called",
        }
    )
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="test")],
        model="test-model",
        stream=True,
        use_workflow="workflow_with_stop",
        workflow_execution_id="exec-stop",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    output = "".join([chunk async for chunk in stream])

    # Stop point message should appear
    assert "Stop point reached" in output
    assert "summarize_result" in output

    # Verify that only the first agent ran
    state = engine.state_store.load_execution("exec-stop")
    assert state.completed is True
    assert "summarize_result" in state.context.get("agents", {})
    assert "should_not_run" not in state.context.get("agents", {})

    # Verify LLM was only called once (for summarize_result, not should_not_run)
    assert len(llm.calls) == 2  # routing_intent + summarize_result
    assert "should_not_run" not in llm.calls


@pytest.mark.asyncio
async def test_stop_point_with_reroute_flow(tmp_path, monkeypatch):
    """Test that stop_point works correctly with a reroute flow (e.g., error summary)."""
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "workflow_with_error_stop",
        "root_intent": "TEST_ERROR_STOP",
        "agents": [
            {
                "agent": "process_data",
                "description": "Process data",
                "reroute": [
                    {
                        "on": ["ERROR"],
                        "to": "summarize_error",
                    }
                ],
            },
            {
                "agent": "summarize_error",
                "description": "Summarize error",
                "stop_point": True,
            },
            {
                "agent": "cleanup",
                "description": "Cleanup resources",
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "process_data": "<reroute>ERROR</reroute>",
            "summarize_error": "Error summary",
            "cleanup": "Cleanup done",
        }
    )
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="test")],
        model="test-model",
        stream=True,
        use_workflow="workflow_with_error_stop",
        workflow_execution_id="exec-error-stop",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    output = "".join([chunk async for chunk in stream])

    # Should have rerouted and then stopped
    assert "Rerouting to summarize_error" in output
    assert "Stop point reached" in output

    # Verify the state
    state = engine.state_store.load_execution("exec-error-stop")
    assert state.completed is True
    assert "process_data" in state.context.get("agents", {})
    assert "summarize_error" in state.context.get("agents", {})
    assert "cleanup" not in state.context.get("agents", {})

    # Verify LLM calls: routing_intent, process_data, summarize_error (but NOT cleanup)
    assert "cleanup" not in llm.calls


@pytest.mark.asyncio
async def test_stop_point_without_stop_allows_continuation(tmp_path, monkeypatch):
    """Test that workflows without stop_point continue to the next agent."""
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "workflow_without_stop",
        "root_intent": "TEST_NO_STOP",
        "agents": [
            {
                "agent": "first",
                "description": "First agent",
            },
            {
                "agent": "second",
                "description": "Second agent",
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "first": "First output",
            "second": "Second output",
        }
    )
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="test")],
        model="test-model",
        stream=True,
        use_workflow="workflow_without_stop",
        workflow_execution_id="exec-no-stop",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    output = "".join([chunk async for chunk in stream])

    # Should NOT have a stop point message
    assert "Stop point reached" not in output

    # Both agents should have run
    state = engine.state_store.load_execution("exec-no-stop")
    assert state.completed is True
    assert "first" in state.context.get("agents", {})
    assert "second" in state.context.get("agents", {})

    # Both agents should have been called
    assert "first" in llm.calls
    assert "second" in llm.calls


@pytest.mark.asyncio
async def test_return_tags_capture_into_agent_context(tmp_path, monkeypatch):
    """Inline <return> tags should populate agent context without tools."""
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "return_capture_flow",
        "root_intent": "TEST_RETURN_CAPTURE",
        "agents": [
            {
                "agent": "collect",
                "description": "Collect plan options.",
            }
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "collect": "<return name=\"plan_list\">[1,2]</return><return name='meta.author'>alice</return>Here you go"
        }
    )
    engine = WorkflowEngine(
        WorkflowRepository(), WorkflowStateStore(db_path=tmp_path / "state.db"), llm
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="run")],
        model="test-model",
        stream=True,
        use_workflow="return_capture_flow",
        workflow_execution_id="exec-return-capture",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    state = engine.state_store.load_execution("exec-return-capture")
    agent_ctx = state.context["agents"]["collect"]
    assert agent_ctx["plan_list"] == "[1,2]"
    assert agent_ctx["meta"]["author"] == "alice"
    assert agent_ctx["content"] == "Here you go"
    # Ensure the shared context was not polluted automatically
    assert "plan_list" not in state.context


@pytest.mark.asyncio
async def test_reroute_with_shared_plan_id(tmp_path, monkeypatch):
    """
    Reroute config with 'with' should copy return-tag values to shared context
    so downstream tools can consume them via top-level arg mappings.
    """
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "return_with_flow",
        "root_intent": "TEST_RETURN_WITH",
        "agents": [
            {
                "agent": "find_plan",
                "description": "Locate a matching plan.",
                "reroute": [
                    {
                        "on": ["FOUND_PERFECT_MATCH"],
                        "to": "get_plan",
                        "with": ["plan_id"],
                    }
                ],
            },
            {
                "agent": "get_plan",
                "description": "Fetch the plan details.",
                "depends_on": ["find_plan"],
                "tools": [{"get_plan": {"args": {"plan_id": "plan_id"}}}],
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_plan",
                "description": "Fetch plan",
                "parameters": {
                    "type": "object",
                    "properties": {"plan_id": {"type": "string"}},
                    "required": ["plan_id"],
                },
            },
        }
    ]

    class RerouteLLM(StubLLMClient):
        def __init__(self):
            super().__init__(
                {
                    "find_plan": '<return name="plan_id">plan-abc</return><reroute>FOUND_PERFECT_MATCH</reroute>',
                    "get_plan": "<passthrough>Done</passthrough>",
                }
            )
            self._emitted_tool = False

        def stream_completion(self, request, access_token, span):
            last_message = request.messages[-1].content or ""
            agent_marker = ""
            if "<agent:" in last_message:
                agent_marker = last_message.split("<agent:", 1)[1].split(">", 1)[0]

            if agent_marker == "get_plan" and not self._emitted_tool:
                self._emitted_tool = True

                async def _gen_tool():
                    payload = {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "function": {
                                                "name": "get_plan",
                                                "arguments": "{}",
                                            },
                                        }
                                    ]
                                },
                                "index": 0,
                            }
                        ]
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    yield "data: [DONE]\n\n"

                return _gen_tool()

            return super().stream_completion(request, access_token, span)

    class CaptureSession(StubSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tool_calls: list[dict[str, Any]] = []

        async def call_tool(self, name: str, args: dict, access_token: str):
            self.tool_calls.append({"name": name, "args": args})
            return SimpleNamespace(
                isError=False,
                content=[
                    SimpleNamespace(
                        text=json.dumps({"plan": {"id": args.get("plan_id")}})
                    )
                ],
            )

    llm = RerouteLLM()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="run")],
        model="test-model",
        stream=True,
        use_workflow="return_with_flow",
        workflow_execution_id="exec-return-with",
    )

    session = CaptureSession(tools=tools)
    stream = await engine.start_or_resume_workflow(session, request, None, None, None)
    _ = [chunk async for chunk in stream]

    # plan_id should be copied to shared context and used for arg injection
    state = store.load_execution("exec-return-with")
    assert state.context.get("plan_id") == "plan-abc"
    assert state.context["agents"]["find_plan"]["plan_id"] == "plan-abc"
    assert session.tool_calls
    assert session.tool_calls[0]["args"].get("plan_id") == "plan-abc"


@pytest.mark.asyncio
async def test_reroute_ask_single_match_routes_on_user_feedback(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "ask_flow_single",
        "root_intent": "ASK_SINGLE",
        "agents": [
            {
                "agent": "find_plan",
                "description": "Locate a matching plan.",
                "reroute": [
                    {
                        "on": ["FOUND_PERFECT_MATCH"],
                        "ask": {
                            "question": "Present the plan and ask whether to proceed or create a new plan.",
                            "expected_responses": [
                                {
                                    "proceed": {
                                        "to": "select_run_mode",
                                        "with": ["plan_id"],
                                    },
                                    "create_new": {"to": "workflows[plan_create]"},
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "agent": "select_run_mode",
                "description": "Select a run mode.",
                "depends_on": ["find_plan"],
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "find_plan": '<return name="plan_id">plan-1</return><reroute>FOUND_PERFECT_MATCH</reroute>',
            "select_run_mode": "<passthrough>ready</passthrough>",
        },
        ask_responses={
            "feedback_question": "Use the matching plan or create a new one?"
        },
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    exec_id = "exec-ask-single"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Run a plan")],
        model="test-model",
        stream=True,
        use_workflow="ask_flow_single",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]
    output = "\n".join(chunks)
    assert "<user_feedback_needed>" in output
    state = store.load_execution(exec_id)
    assert state.awaiting_feedback is True
    assert llm.calls.count("find_plan") == 1

    resume_request = ChatCompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content="<user_feedback>select_run_mode</user_feedback>",
            )
        ],
        model="test-model",
        stream=True,
        use_workflow="ask_flow_single",
        workflow_execution_id=exec_id,
    )
    resume_stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    _ = [chunk async for chunk in resume_stream]

    final_state = store.load_execution(exec_id)
    assert final_state.context.get("plan_id") == "plan-1"
    assert "select_run_mode" in llm.calls
    assert llm.calls.count("find_plan") == 1


@pytest.mark.asyncio
async def test_reroute_ask_single_match_routes_to_workflow_with_paren_feedback(
    tmp_path, monkeypatch
):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "ask_flow_single",
        "root_intent": "ASK_SINGLE",
        "agents": [
            {
                "agent": "find_plan",
                "description": "Locate a matching plan.",
                "reroute": [
                    {
                        "on": ["FOUND_PERFECT_MATCH"],
                        "ask": {
                            "question": "Present the plan and ask whether to proceed or create a new plan.",
                            "expected_responses": [
                                {
                                    "proceed": {
                                        "to": "select_run_mode",
                                        "with": ["plan_id"],
                                    },
                                    "create_new": {"to": "workflows[plan_create]"},
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "agent": "select_run_mode",
                "description": "Select a run mode.",
                "depends_on": ["find_plan"],
            },
        ],
    }
    plan_create_flow = {
        "flow_id": "plan_create",
        "root_intent": "CREATE_PLAN",
        "agents": [{"agent": "select_tools", "description": "Select tools."}],
    }
    _write_workflow(workflows_dir, "flow", flow)
    _write_workflow(workflows_dir, "plan_create", plan_create_flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "find_plan": '<return name="plan_id">plan-1</return><reroute>FOUND_PERFECT_MATCH</reroute>',
            "select_run_mode": "<passthrough>ready</passthrough>",
            "select_tools": "<passthrough>ok</passthrough>",
        },
        ask_responses={
            "feedback_question": "Use the matching plan or create a new one?"
        },
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    exec_id = "exec-ask-single-workflow"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Run a plan")],
        model="test-model",
        stream=True,
        use_workflow="ask_flow_single",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]
    state = store.load_execution(exec_id)
    assert state.awaiting_feedback is True

    resume_request = ChatCompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content="<user_feedback>workflows[plan_create]()</user_feedback>",
            )
        ],
        model="test-model",
        stream=True,
        use_workflow="ask_flow_single",
        workflow_execution_id=exec_id,
    )
    resume_stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    _ = [chunk async for chunk in resume_stream]

    final_state = store.load_execution(exec_id)
    assert final_state.flow_id == "plan_create"


@pytest.mark.asyncio
async def test_reroute_ask_multiple_match_assigns_selection(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "ask_flow_multi",
        "root_intent": "ASK_MULTI",
        "agents": [
            {
                "agent": "find_plan",
                "description": "Locate matching plans.",
                "reroute": [
                    {
                        "on": ["FOUND_MULTIPLE_MATCHES"],
                        "ask": {
                            "question": "Present matching plans and ask which to run.",
                            "expected_responses": [
                                {
                                    "select_plan": {
                                        "to": "select_run_mode",
                                        "each": "plans",
                                    },
                                    "create_new": {"to": "workflows[plan_create]"},
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "agent": "select_run_mode",
                "description": "Select a run mode.",
                "depends_on": ["find_plan"],
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    plans = [
        {"key": "plan-1", "value": "Plan one"},
        {"key": "plan-2", "value": "Plan two"},
    ]
    llm = StubLLMClient(
        {
            "find_plan": (
                f'<return name="plans">{json.dumps(plans)}</return>'
                "<reroute>FOUND_MULTIPLE_MATCHES</reroute>"
            ),
            "select_run_mode": "<passthrough>ready</passthrough>",
        },
        ask_responses={"feedback_question": "Which plan should I run?"},
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    exec_id = "exec-ask-multi"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Run a plan")],
        model="test-model",
        stream=True,
        use_workflow="ask_flow_multi",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]
    state = store.load_execution(exec_id)
    assert state.awaiting_feedback is True
    feedback_spec = state.context["agents"]["find_plan"].get("feedback_spec") or {}
    expected = feedback_spec.get("expected_responses") or []
    select_entry = next(
        (entry for entry in expected if entry.get("id") == "select_plan"), None
    )
    assert select_entry is not None
    assert select_entry.get("options")
    assert select_entry.get("with") in (None, [])
    assert select_entry.get("each") == "plans"

    resume_request = ChatCompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content='<user_feedback>select_run_mode("plan-2")</user_feedback>',
            )
        ],
        model="test-model",
        stream=True,
        use_workflow="ask_flow_multi",
        workflow_execution_id=exec_id,
    )
    resume_stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    _ = [chunk async for chunk in resume_stream]

    final_state = store.load_execution(exec_id)
    assert final_state.context.get("plan_id") is None
    assert "select_run_mode" in llm.calls


@pytest.mark.asyncio
async def test_reroute_ask_multiple_match_with_explicit_with_sets_plan_id(
    tmp_path, monkeypatch
):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "ask_flow_multi_with",
        "root_intent": "ASK_MULTI_WITH",
        "agents": [
            {
                "agent": "find_plan",
                "description": "Locate matching plans.",
                "reroute": [
                    {
                        "on": ["FOUND_MULTIPLE_MATCHES"],
                        "ask": {
                            "question": "Present matching plans and ask which to run.",
                            "expected_responses": [
                                {
                                    "select_plan": {
                                        "to": "select_run_mode",
                                        "each": "plans",
                                        "with": ["plan_id"],
                                    },
                                    "create_new": {"to": "workflows[plan_create]"},
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "agent": "select_run_mode",
                "description": "Select a run mode.",
                "depends_on": ["find_plan"],
                "context": ["plan_id"],
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    plans = [
        {"key": "plan-1", "value": "Plan one"},
        {"key": "plan-2", "value": "Plan two"},
    ]
    llm = StubLLMClient(
        {
            "find_plan": (
                f'<return name="plans">{json.dumps(plans)}</return>'
                "<reroute>FOUND_MULTIPLE_MATCHES</reroute>"
            ),
            "select_run_mode": "<passthrough>ready</passthrough>",
        },
        ask_responses={"feedback_question": "Which plan should I run?"},
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    exec_id = "exec-ask-multi-with"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Run a plan")],
        model="test-model",
        stream=True,
        use_workflow="ask_flow_multi_with",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    resume_request = ChatCompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content='<user_feedback>select_run_mode("plan-2")</user_feedback>',
            )
        ],
        model="test-model",
        stream=True,
        use_workflow="ask_flow_multi_with",
        workflow_execution_id=exec_id,
    )
    resume_stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    _ = [chunk async for chunk in resume_stream]

    final_state = store.load_execution(exec_id)
    assert final_state.context.get("plan_id") == "plan-2"
    assert "select_run_mode" in llm.calls
    select_calls = [
        idx for idx, call in enumerate(llm.calls) if call == "select_run_mode"
    ]
    assert select_calls
    select_prompt = llm.user_messages[select_calls[-1]]
    assert '"plan_id": "plan-2"' in select_prompt


@pytest.mark.asyncio
async def test_scoped_context_root_reference_resolves_top_level_key(
    tmp_path, monkeypatch
):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "root_context_flow",
        "root_intent": "ROOT_CONTEXT",
        "agents": [
            {
                "agent": "first",
                "description": "Set a plan id and reroute.",
                "reroute": [{"on": ["GO"], "to": "second", "with": ["plan_id"]}],
            },
            {
                "agent": "second",
                "description": "Consumes shared plan id.",
                "depends_on": ["first"],
                "context": ["plan_id"],
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "first": '<return name="plan_id">plan-123</return><reroute>GO</reroute>',
            "second": "<passthrough>ok</passthrough>",
        }
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    exec_id = "exec-root-context"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Run")],
        model="test-model",
        stream=True,
        use_workflow="root_context_flow",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    second_calls = [idx for idx, call in enumerate(llm.calls) if call == "second"]
    assert second_calls
    second_prompt = llm.user_messages[second_calls[-1]]
    assert '"plan_id": "plan-123"' in second_prompt


@pytest.mark.asyncio
async def test_feedback_choice_passthrough_fields(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "choice_passthrough",
        "root_intent": "CHOICE_PASSTHROUGH",
        "agents": [
            {
                "agent": "select_run_mode",
                "description": "Ask how to run the plan.",
                "reroute": [
                    {
                        "on": ["ASK_USER"],
                        "ask": {
                            "question": "Pick a mode.",
                            "expected_responses": [
                                {
                                    "run_now": {
                                        "to": "workflows[plan_run]",
                                        "with": ["plan_id"],
                                        "value": "Run the plan now",
                                    },
                                    "abort": {"to": "end", "value": "Abort"},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "select_run_mode": "<reroute>ASK_USER</reroute>",
        },
        ask_responses={"feedback_question": "Pick a mode."},
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    exec_id = "exec-choice-pass"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Run")],
        model="test-model",
        stream=True,
        use_workflow="choice_passthrough",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    state = store.load_execution(exec_id)
    feedback_spec = (
        state.context["agents"]["select_run_mode"].get("feedback_spec") or {}
    )
    expected = feedback_spec.get("expected_responses") or []
    run_now = next((entry for entry in expected if entry.get("id") == "run_now"), None)
    abort = next((entry for entry in expected if entry.get("id") == "abort"), None)
    assert run_now is not None
    assert abort is not None
    assert run_now.get("value") == "Run the plan now"
    assert abort.get("value") == "Abort"


@pytest.mark.asyncio
async def test_feedback_payload_external_with_values_from_agent_context(
    tmp_path, monkeypatch
):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "external_choice",
        "root_intent": "EXTERNAL_CHOICE",
        "agents": [
            {
                "agent": "select_run_mode",
                "description": "Ask how to run the plan.",
                "reroute": [
                    {
                        "on": ["ASK_USER"],
                        "ask": {
                            "question": "Pick a mode.",
                            "expected_responses": [
                                {
                                    "run_now": {
                                        "to": "external[trigger]",
                                        "with": ["plan_id"],
                                        "value": "Run the plan now",
                                    },
                                    "schedule": {
                                        "to": "workflows[plan_schedule]",
                                        "with": ["plan_id"],
                                        "value": "Schedule the plan for later",
                                    },
                                    "abort": {"to": "end", "value": "Abort"},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "select_run_mode": '<return name="plan_id">plan-123</return><reroute>ASK_USER</reroute>',
        },
        ask_responses={"feedback_question": "Pick a mode."},
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    exec_id = "exec-external-choice"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Run")],
        model="test-model",
        stream=True,
        use_workflow="external_choice",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    state = store.load_execution(exec_id)
    feedback_spec = (
        state.context["agents"]["select_run_mode"].get("feedback_spec") or {}
    )
    expected = feedback_spec.get("expected_responses") or []
    run_now = next((entry for entry in expected if entry.get("id") == "run_now"), None)
    schedule = next(
        (entry for entry in expected if entry.get("id") == "schedule"), None
    )
    assert run_now is not None
    assert schedule is not None
    assert run_now.get("to") == "external[trigger]"
    assert run_now.get("with") == ["plan-123"]
    assert schedule.get("with") == ["plan_id"]


@pytest.mark.asyncio
async def test_feedback_payload_external_with_values_from_shared_context(
    tmp_path, monkeypatch
):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "external_choice_shared",
        "root_intent": "EXTERNAL_CHOICE_SHARED",
        "agents": [
            {
                "agent": "select_run_mode",
                "description": "Ask how to run the plan.",
                "reroute": [
                    {
                        "on": ["ASK_USER"],
                        "ask": {
                            "question": "Pick a mode.",
                            "expected_responses": [
                                {
                                    "run_now": {
                                        "to": "external[trigger]",
                                        "with": ["plan_id"],
                                        "value": "Run the plan now",
                                    }
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "select_run_mode": "<reroute>ASK_USER</reroute>",
        },
        ask_responses={"feedback_question": "Pick a mode."},
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    exec_id = "exec-external-choice-shared"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Run")],
        model="test-model",
        stream=True,
        use_workflow="external_choice_shared",
        workflow_execution_id=exec_id,
        start_with={"args": {"plan_id": "plan-999"}},
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    state = store.load_execution(exec_id)
    feedback_spec = (
        state.context["agents"]["select_run_mode"].get("feedback_spec") or {}
    )
    expected = feedback_spec.get("expected_responses") or []
    run_now = next((entry for entry in expected if entry.get("id") == "run_now"), None)
    assert run_now is not None
    assert run_now.get("with") == ["plan-999"]


@pytest.mark.asyncio
async def test_feedback_payload_external_missing_context_uses_null(
    tmp_path, monkeypatch
):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "external_choice_missing",
        "root_intent": "EXTERNAL_CHOICE_MISSING",
        "agents": [
            {
                "agent": "select_run_mode",
                "description": "Ask how to run the plan.",
                "reroute": [
                    {
                        "on": ["ASK_USER"],
                        "ask": {
                            "question": "Pick a mode.",
                            "expected_responses": [
                                {
                                    "run_now": {
                                        "to": "external[trigger]",
                                        "with": ["plan_id"],
                                        "value": "Run the plan now",
                                    }
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "select_run_mode": "<reroute>ASK_USER</reroute>",
        },
        ask_responses={"feedback_question": "Pick a mode."},
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    exec_id = "exec-external-choice-missing"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Run")],
        model="test-model",
        stream=True,
        use_workflow="external_choice_missing",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    state = store.load_execution(exec_id)
    feedback_spec = (
        state.context["agents"]["select_run_mode"].get("feedback_spec") or {}
    )
    expected = feedback_spec.get("expected_responses") or []
    run_now = next((entry for entry in expected if entry.get("id") == "run_now"), None)
    assert run_now is not None
    assert run_now.get("with") == [None]


@pytest.mark.asyncio
async def test_feedback_payload_external_from_llm_tag_resolves_context(
    tmp_path, monkeypatch
):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "llm_feedback_external",
        "root_intent": "LLM_FEEDBACK_EXTERNAL",
        "agents": [
            {
                "agent": "select_run_mode",
                "description": "Ask how to run the plan.",
            }
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    payload = {
        "question": "Pick a mode.",
        "expected_responses": [
            {
                "id": "run_now",
                "to": "external[trigger]",
                "with": ["plan_id"],
                "value": "Run the plan now",
            },
            {
                "id": "schedule",
                "to": "workflows[plan_schedule]",
                "with": ["plan_id"],
                "value": "Schedule the plan",
            },
        ],
    }
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    llm = StubLLMClient(
        {
            "select_run_mode": (
                '<return name="plan_id">plan-123</return>'
                f"<user_feedback_needed>{payload_json}</user_feedback_needed>"
            ),
        }
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    exec_id = "exec-llm-feedback-external"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Run")],
        model="test-model",
        stream=True,
        use_workflow="llm_feedback_external",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]
    output = "".join(chunks)

    assert "<user_feedback_needed>" in output
    assert "plan-123" in output
    state = store.load_execution(exec_id)
    feedback_spec = (
        state.context["agents"]["select_run_mode"].get("feedback_spec") or {}
    )
    expected = feedback_spec.get("expected_responses") or []
    run_now = next((entry for entry in expected if entry.get("id") == "run_now"), None)
    schedule = next(
        (entry for entry in expected if entry.get("id") == "schedule"), None
    )
    assert run_now is not None
    assert schedule is not None
    assert run_now.get("with") == ["plan-123"]
    assert schedule.get("with") == ["plan_id"]


@pytest.mark.asyncio
async def test_feedback_payload_external_from_llm_tag_prefers_shared_context(
    tmp_path, monkeypatch
):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "llm_feedback_external_shared",
        "root_intent": "LLM_FEEDBACK_EXTERNAL_SHARED",
        "agents": [
            {
                "agent": "select_run_mode",
                "description": "Ask how to run the plan.",
            }
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    payload = {
        "question": "Pick a mode.",
        "expected_responses": [
            {
                "id": "run_now",
                "to": "external[trigger]",
                "with": ["plan_id"],
                "value": "Run the plan now",
            },
            {
                "id": "schedule",
                "to": "workflows[plan_schedule]",
                "with": ["plan_id"],
                "value": "Schedule the plan",
            },
        ],
    }
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    llm = StubLLMClient(
        {
            "select_run_mode": (
                '<return name="plan_id">plan_id</return>'
                f"<user_feedback_needed>{payload_json}</user_feedback_needed>"
            ),
        }
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    exec_id = "exec-llm-feedback-external-shared"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Run")],
        model="test-model",
        stream=True,
        use_workflow="llm_feedback_external_shared",
        workflow_execution_id=exec_id,
        start_with={"args": {"plan_id": "plan-999"}},
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    state = store.load_execution(exec_id)
    feedback_spec = (
        state.context["agents"]["select_run_mode"].get("feedback_spec") or {}
    )
    expected = feedback_spec.get("expected_responses") or []
    run_now = next((entry for entry in expected if entry.get("id") == "run_now"), None)
    schedule = next(
        (entry for entry in expected if entry.get("id") == "schedule"), None
    )
    assert run_now is not None
    assert schedule is not None
    assert run_now.get("with") == ["plan-999"]
    assert schedule.get("with") == ["plan_id"]


@pytest.mark.asyncio
async def test_feedback_resume_preserves_reroute_with_context(tmp_path, monkeypatch):
    """
    After resuming from user feedback, reroute `with` values should still be
    copied into shared context so downstream agents can inject tool args.
    """
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "plan_run",
        "root_intent": "CREATE_OR_RUN_PLAN",
        "agents": [
            {
                "agent": "find_plan",
                "description": "Locate a matching plan.",
                "reroute": [
                    {
                        "on": ["FOUND_MULTIPLE_MATCHES"],
                        "to": "ask_user_to_select_plan",
                        "with": ["plan_ids"],
                    }
                ],
            },
            {
                "agent": "ask_user_to_select_plan",
                "description": "Ask which plan to run.",
                "depends_on": ["find_plan"],
                "pass_through": True,
                "reroute": [
                    {
                        "on": ["PLAN_SELECTED"],
                        "to": "get_plan",
                        "with": ["plan_id"],
                    }
                ],
            },
            {
                "agent": "get_plan",
                "description": "Fetch the selected plan.",
                "depends_on": ["find_plan"],
                "context": ["plan_id"],
                "tools": [{"get_plan": {"args": {"plan_id": "plan_id"}}}],
            },
        ],
    }
    _write_workflow(workflows_dir, "plan_run", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_plan",
                "description": "Fetch plan",
                "parameters": {
                    "type": "object",
                    "properties": {"plan_id": {"type": "string"}},
                    "required": ["plan_id"],
                },
            },
        }
    ]

    class FeedbackRerouteLLM(StubLLMClient):
        def __init__(self):
            super().__init__(
                {
                    "find_plan": '<return name="plan_ids">["plan-1","plan-2"]</return><reroute>FOUND_MULTIPLE_MATCHES</reroute>',
                    "ask_user_to_select_plan": [
                        "<user_feedback_needed>Select a plan to run.</user_feedback_needed>",
                        '<return name="plan_id">plan-1</return><reroute>PLAN_SELECTED</reroute>',
                    ],
                    "get_plan": "<passthrough>Plan loaded</passthrough>",
                }
            )
            self._emitted_tool = False

        def stream_completion(self, request, access_token, span):
            last_message = request.messages[-1].content or ""
            agent_marker = ""
            if "<agent:" in last_message:
                agent_marker = last_message.split("<agent:", 1)[1].split(">", 1)[0]
            if agent_marker == "get_plan" and not self._emitted_tool:
                self._emitted_tool = True
                self.calls.append(agent_marker)

                async def _gen_tool():
                    payload = {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "function": {
                                                "name": "get_plan",
                                                "arguments": "{}",
                                            },
                                        }
                                    ]
                                },
                                "index": 0,
                            }
                        ]
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    yield "data: [DONE]\n\n"

                return _gen_tool()
            return super().stream_completion(request, access_token, span)

    class CaptureSession(StubSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tool_calls: list[dict[str, Any]] = []

        async def call_tool(self, name: str, args: dict, access_token: str):
            self.tool_calls.append({"name": name, "args": args})
            return SimpleNamespace(
                isError=False,
                content=[
                    SimpleNamespace(
                        text=json.dumps({"plan": {"id": args.get("plan_id")}})
                    )
                ],
            )

    llm = FeedbackRerouteLLM()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)
    session = CaptureSession(tools=tools)

    exec_id = "plan-run-feedback"
    first_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Run a plan")],
        model="test-model",
        stream=True,
        use_workflow="plan_run",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        session, first_request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    paused_state = store.load_execution(exec_id)
    assert paused_state.awaiting_feedback is True
    # Simulate legacy state without explicit resume marker to exercise fallback
    paused_state.context.pop("_resume_agent", None)
    store.save_state(paused_state)

    resume_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Use plan-1")],
        model="test-model",
        stream=True,
        use_workflow="plan_run",
        workflow_execution_id=exec_id,
    )
    resume_stream = await engine.start_or_resume_workflow(
        session, resume_request, None, None, None
    )
    _ = [chunk async for chunk in resume_stream]

    final_state = store.load_execution(exec_id)
    assert final_state.context.get("plan_id") == "plan-1"
    assert session.tool_calls
    assert session.tool_calls[0]["args"].get("plan_id") == "plan-1"
    # The agent that asked for feedback should resume before moving on
    assert llm.calls.count("ask_user_to_select_plan") == 2


@pytest.mark.asyncio
async def test_start_with_prefills_context_and_forces_agent(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "prefill_flow",
        "root_intent": "RUN_PLAN",
        "agents": [
            {"agent": "find_plan", "description": "Find a plan."},
            {
                "agent": "get_plan",
                "description": "Get plan by id.",
                "depends_on": ["find_plan"],
                "tools": [{"get_plan": {"args": {"plan_id": "plan_id"}}}],
            },
        ],
    }
    _write_workflow(workflows_dir, "prefill_flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_plan",
                "description": "Fetch plan",
                "parameters": {
                    "type": "object",
                    "properties": {"plan_id": {"type": "string"}},
                    "required": ["plan_id"],
                },
            },
        }
    ]

    class StartWithLLM(StubLLMClient):
        def __init__(self):
            super().__init__({"get_plan": "<passthrough>loaded</passthrough>"})
            self._emitted_tool = False

        def stream_completion(self, request, access_token, span):
            last_message = request.messages[-1].content or ""
            agent_marker = ""
            if "<agent:" in last_message:
                agent_marker = last_message.split("<agent:", 1)[1].split(">", 1)[0]
            if agent_marker == "get_plan" and not self._emitted_tool:
                self._emitted_tool = True
                self.calls.append(agent_marker)

                async def _gen_tool():
                    payload = {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "function": {
                                                "name": "get_plan",
                                                "arguments": "{}",
                                            },
                                        }
                                    ]
                                },
                                "index": 0,
                            }
                        ]
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    yield "data: [DONE]\n\n"

                return _gen_tool()
            return super().stream_completion(request, access_token, span)

    class CaptureSession(StubSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tool_calls: list[dict[str, Any]] = []

        async def call_tool(self, name: str, args: dict, access_token: str):
            self.tool_calls.append({"name": name, "args": args})
            return SimpleNamespace(
                isError=False,
                content=[
                    SimpleNamespace(
                        text=json.dumps({"plan": {"id": args.get("plan_id")}})
                    )
                ],
            )

    llm = StartWithLLM()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)
    session = CaptureSession(tools=tools)

    exec_id = "start-with-exec"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Run the plan")],
        model="test-model",
        stream=True,
        use_workflow="prefill_flow",
        workflow_execution_id=exec_id,
        start_with={"args": {"plan_id": "plan-123"}, "agent": "get_plan"},
    )

    stream = await engine.start_or_resume_workflow(session, request, None, None, None)
    _ = [chunk async for chunk in stream]

    state = store.load_execution(exec_id)
    assert state.context.get("plan_id") == "plan-123"
    assert state.context["agents"]["find_plan"]["completed"] is True
    assert state.context["agents"]["find_plan"]["reason"] == "start_with_prefill"
    assert session.tool_calls
    assert session.tool_calls[0]["args"].get("plan_id") == "plan-123"
    assert "find_plan" not in llm.calls


@pytest.mark.asyncio
async def test_routing_agent_tool_result_includes_tool_call_id(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "routing_flow",
        "root_intent": "ROUTING_TEST",
        "agents": [{"agent": "only_agent", "description": "Run once."}],
    }
    _write_workflow(workflows_dir, "routing", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    class RoutingToolLLM(StubLLMClient):
        def __init__(self):
            super().__init__({"only_agent": "ok"})
            self.tool_call_id_seen = None
            self._routing_called = False

        def stream_completion(self, request, access_token, span):
            last_message = request.messages[-1].content or ""
            if "ROUTING_INTENT_CHECK" in last_message and not self._routing_called:
                self._routing_called = True

                async def _gen_tool():
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

                return _gen_tool()

            for message in request.messages:
                if message.role == MessageRole.TOOL:
                    self.tool_call_id_seen = message.tool_call_id
                    break

            return super().stream_completion(request, access_token, span)

    llm = RoutingToolLLM()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="run")],
        model="test-model",
        stream=True,
        use_workflow="routing_flow",
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    assert llm.tool_call_id_seen == "call_1"


@pytest.mark.asyncio
async def test_scoped_context_summary_includes_large_return_values(
    tmp_path, monkeypatch
):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "plan_flow",
        "root_intent": "PLAN_FLOW",
        "agents": [
            {
                "agent": "find_plan",
                "description": "Find matching plans.",
                "reroute": [
                    {
                        "on": ["FOUND_MULTIPLE_MATCHES"],
                        "to": "ask_user_to_select_plan",
                    }
                ],
            },
            {
                "agent": "ask_user_to_select_plan",
                "description": "Ask which plan to run.",
                "depends_on": ["find_plan"],
                "pass_through": True,
                "context": ["find_plan.plans"],
            },
        ],
    }
    _write_workflow(workflows_dir, "plan_flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    plans = [
        {
            "id": f"plan-{idx}",
            "description": "Summary " + ("x" * 120),
        }
        for idx in range(30)
    ]
    llm = StubLLMClient(
        {
            "find_plan": (
                f'<return name="plans">{json.dumps(plans)}</return>'
                "<reroute>FOUND_MULTIPLE_MATCHES</reroute>"
            ),
            "ask_user_to_select_plan": (
                "<user_feedback_needed>Select a plan.</user_feedback_needed>"
            ),
        }
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Find a plan")],
        model="test-model",
        stream=True,
        use_workflow="plan_flow",
        workflow_execution_id="plan-flow-exec",
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    prompt = next(
        message
        for message in llm.user_messages
        if "<agent:ask_user_to_select_plan>" in message
    )
    assert "plan-29" in prompt


@pytest.mark.asyncio
async def test_workflow_handoff_preserves_user_query_and_messages(
    tmp_path, monkeypatch
):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    welcome_flow = {
        "flow_id": "welcome_flow",
        "root_intent": "WELCOME",
        "agents": [
            {"agent": "find_plan", "description": "Find matching plans."},
            {
                "agent": "ask_user_to_select_plan",
                "description": "Ask which plan to run.",
                "depends_on": ["find_plan"],
                "pass_through": True,
            },
        ],
    }
    plan_create_flow = {
        "flow_id": "plan_create",
        "root_intent": "CREATE_PLAN",
        "agents": [{"agent": "select_tools", "description": "Select tools."}],
    }
    _write_workflow(workflows_dir, "welcome", welcome_flow)
    _write_workflow(workflows_dir, "plan_create", plan_create_flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    class HandoffLLM(StubLLMClient):
        def __init__(self):
            super().__init__(
                {
                    "find_plan": (
                        '<return name="plans">[]</return>'
                        "<reroute>FOUND_MULTIPLE_MATCHES</reroute>"
                    ),
                    "ask_user_to_select_plan": [
                        "<user_feedback_needed>Select a plan.</user_feedback_needed>",
                        "<reroute>workflows[plan_create]</reroute>",
                    ],
                    "select_tools": "<passthrough>selecting</passthrough>",
                },
                ask_responses={
                    "user_query_summary": "Create a new plan to summarize the last 24 hours of news.",
                    "feedback_rerun_decision": "RERUN",
                },
            )

    llm = HandoffLLM()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    exec_id = "welcome-exec"
    first_request = ChatCompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content="Generate a plan to create a news summary for the last 24 hours",
            )
        ],
        model="test-model",
        stream=True,
        use_workflow="welcome_flow",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), first_request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    paused_state = store.load_execution(exec_id)
    assert paused_state.awaiting_feedback is True

    resume_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Create a new one please")],
        model="test-model",
        stream=True,
        use_workflow="welcome_flow",
        workflow_execution_id=exec_id,
    )
    resume_stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    _ = [chunk async for chunk in resume_stream]

    state = store.load_execution(exec_id)
    assert state.flow_id == "plan_create"
    assert (
        state.context.get("user_query")
        == "Create a new plan to summarize the last 24 hours of news."
    )
    assert state.context.get("user_messages") == [
        "Generate a plan to create a news summary for the last 24 hours",
        "Create a new one please",
    ]


@pytest.mark.asyncio
async def test_run_agents_parallelizes_safe_independent_agents(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "parallel_safe_flow",
        "root_intent": "PARALLEL_SAFE",
        "agents": [
            {"agent": "alpha", "description": "Alpha", "tools": []},
            {"agent": "beta", "description": "Beta", "tools": []},
            {
                "agent": "finalize",
                "description": "Finalize",
                "depends_on": ["alpha", "beta"],
                "tools": [],
            },
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    class TrackingEngine(WorkflowEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.running = 0
            self.max_running = 0
            self.started: list[str] = []

        async def _execute_agent(
            self,
            workflow_def,
            agent_def,
            state,
            session,
            request,
            access_token,
            span,
            persist_inner_thinking=False,
            no_reroute=False,
        ):
            self.started.append(agent_def.agent)
            self.running += 1
            self.max_running = max(self.max_running, self.running)
            try:
                await asyncio.sleep(0.03)
            finally:
                self.running -= 1
            yield {
                "status": "done",
                "content": f"{agent_def.agent} complete",
                "pass_through": False,
            }

    repo = WorkflowRepository()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = TrackingEngine(repo, store, StubLLMClient({}))
    engine.max_parallel_agents = 4

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="run")],
        use_workflow="parallel_safe_flow",
        workflow_execution_id="exec-parallel-safe",
        stream=True,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    assert engine.max_running >= 2
    assert "alpha" in engine.started and "beta" in engine.started
    assert engine.started.index("finalize") > max(
        engine.started.index("alpha"), engine.started.index("beta")
    )


@pytest.mark.asyncio
async def test_run_agents_keeps_unsafe_agents_serial(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "parallel_guard_flow",
        "root_intent": "PARALLEL_GUARD",
        "agents": [
            {
                "agent": "unsafe_first",
                "description": "Unsafe",
                "tools": [],
                "reroute": {"on": ["X"], "to": "safe_second"},
            },
            {"agent": "safe_second", "description": "Safe", "tools": []},
        ],
    }
    _write_workflow(workflows_dir, "flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    class TrackingEngine(WorkflowEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.running = 0
            self.max_running = 0

        async def _execute_agent(
            self,
            workflow_def,
            agent_def,
            state,
            session,
            request,
            access_token,
            span,
            persist_inner_thinking=False,
            no_reroute=False,
        ):
            self.running += 1
            self.max_running = max(self.max_running, self.running)
            try:
                await asyncio.sleep(0.02)
            finally:
                self.running -= 1
            yield {
                "status": "done",
                "content": f"{agent_def.agent} complete",
                "pass_through": False,
            }

    repo = WorkflowRepository()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = TrackingEngine(repo, store, StubLLMClient({}))
    engine.max_parallel_agents = 4

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="run")],
        use_workflow="parallel_guard_flow",
        workflow_execution_id="exec-parallel-guard",
        stream=True,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    assert engine.max_running == 1


# ---------------------------------------------------------------------------
# Per-choice input field tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feedback_payload_includes_per_choice_input_fields(tmp_path, monkeypatch):
    """
    When a reroute ask config defines per-choice `input` fields, the generated
    requestedSchema should include those fields as optional properties, and
    meta.input_fields should map each field to its owning selection id.
    """
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "input_field_flow",
        "root_intent": "INPUT_FIELD_TEST",
        "agents": [
            {
                "agent": "select_tools",
                "description": "Select tools.",
                "reroute": [
                    {
                        "on": ["ASK_USER"],
                        "ask": {
                            "question": "Proceed or adjust?",
                            "expected_responses": [
                                {
                                    "proceed": {
                                        "to": "create_plan",
                                        "with": ["selected_tools"],
                                    },
                                    "adjust": {
                                        "to": "select_tools",
                                        "value": "Adjust the integration selection",
                                        "input": {
                                            "feedback": {
                                                "type": "string",
                                                "description": "What would you like to adjust?",
                                            }
                                        },
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "agent": "create_plan",
                "description": "Create the plan.",
                "depends_on": ["select_tools"],
            },
        ],
    }
    _write_workflow(workflows_dir, "input_field_flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {"select_tools": "<reroute>ASK_USER</reroute>"},
        ask_responses={"feedback_question": "Proceed or adjust?"},
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    exec_id = "exec-input-field"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Run")],
        model="test-model",
        stream=True,
        use_workflow="input_field_flow",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )

    [chunk async for chunk in stream]

    state = store.load_execution(exec_id)
    assert state.awaiting_feedback is True

    elic_spec = state.context["agents"]["select_tools"].get("elicitation_spec") or {}
    schema = elic_spec.get("requestedSchema") or {}

    # selection enum should contain both choices
    props = schema.get("properties", {})
    assert "selection" in props
    assert set(props["selection"].get("enum", [])) == {"proceed", "adjust"}

    # per-choice input field should be present in schema
    assert "feedback" in props
    assert props["feedback"]["type"] == "string"
    assert props["feedback"]["description"] == "What would you like to adjust?"

    # feedback should NOT be in required (only selection is required)
    assert schema.get("required") == ["selection"]

    # meta.input_fields should map the field to its owning choice
    meta = elic_spec.get("meta", {})
    assert "input_fields" in meta
    assert meta["input_fields"]["feedback"]["for_selection"] == "adjust"


@pytest.mark.asyncio
async def test_feedback_question_uses_context_helper_for_large_lossy_context(
    tmp_path, monkeypatch
):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "feedback_helper_flow",
        "root_intent": "FEEDBACK_HELPER",
        "agents": [
            {
                "agent": "select_tools",
                "description": "Select tools.",
                "reroute": [
                    {
                        "on": ["ASK_USER"],
                        "ask": {
                            "question": "Present selected integrations and ask whether to proceed.",
                            "expected_responses": [
                                {
                                    "proceed": {
                                        "to": "create_plan",
                                        "with": ["selected_tools"],
                                    }
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }
    _write_workflow(workflows_dir, "feedback_helper_flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    class HelperFeedbackLLM(StubLLMClient):
        def __init__(self):
            super().__init__({"select_tools": "<reroute>ASK_USER</reroute>"})
            self.helper_calls = 0

        def stream_completion(self, request, access_token, span):
            system_prompt = (
                request.messages[0].content
                if request.messages and request.messages[0].content
                else ""
            )
            if "USER_FEEDBACK_QUESTION" in system_prompt:
                self.helper_calls += 1
                has_tool_message = any(
                    msg.role == MessageRole.TOOL for msg in request.messages
                )
                if not has_tool_message:

                    async def _gen_tool():
                        payload = {
                            "choices": [
                                {
                                    "delta": {
                                        "content": "Need exact integration details.",
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "call_feedback_1",
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

                    return _gen_tool()

                tool_payload = ""
                for message in request.messages:
                    if message.role == MessageRole.TOOL:
                        tool_payload = message.content or ""
                        break
                integration_name = "integration"
                try:
                    parsed = json.loads(tool_payload)
                    data = parsed.get("data")
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        integration_name = (
                            data[0].get("tool_name")
                            or data[0].get("name")
                            or integration_name
                        )
                except Exception:
                    pass

                async def _gen_final():
                    payload = {
                        "choices": [
                            {
                                "delta": {
                                    "content": (
                                        f"I found these integrations, including {integration_name}. "
                                        "Proceed with them?"
                                    )
                                },
                                "index": 0,
                            }
                        ]
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    yield "data: [DONE]\n\n"

                return _gen_final()

            return super().stream_completion(request, access_token, span)

    llm = HelperFeedbackLLM()
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    exec_id = "exec-feedback-helper"
    seeded_state = store.get_or_create(exec_id, "feedback_helper_flow")
    seeded_state.context["large_blob"] = "x" * 13000
    seeded_state.context["agents"]["select_tools"] = {
        "selected_tools": [
            {
                "tool_name": "list_commits",
                "description": "Get repository commits.",
            },
            {
                "tool_name": "search_repositories",
                "description": "Find repositories.",
            },
        ],
        "content": "",
        "completed": False,
    }
    store.save_state(seeded_state)

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Create workflow")],
        model="test-model",
        stream=True,
        use_workflow="feedback_helper_flow",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]
    _ = "".join(chunks)

    final_state = store.load_execution(exec_id)
    feedback_message = (
        final_state.context["agents"]["select_tools"]
        .get("elicitation_spec", {})
        .get("message", "")
    )
    assert "I found these integrations" in feedback_message
    assert llm.helper_calls >= 2
    assert not any(
        "USER_FEEDBACK_QUESTION" in (call.get("base_prompt") or "")
        for call in llm.ask_calls
    )


@pytest.mark.asyncio
async def test_feedback_question_uses_fast_ask_for_small_context(tmp_path, monkeypatch):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "feedback_small_flow",
        "root_intent": "FEEDBACK_SMALL",
        "agents": [
            {
                "agent": "select_tools",
                "description": "Select tools.",
                "reroute": [
                    {
                        "on": ["ASK_USER"],
                        "ask": {
                            "question": "Proceed or adjust?",
                            "expected_responses": [
                                {
                                    "proceed": {
                                        "to": "create_plan",
                                        "with": ["selected_tools"],
                                    }
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }
    _write_workflow(workflows_dir, "feedback_small_flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {"select_tools": "<reroute>ASK_USER</reroute>"},
        ask_responses={"feedback_question": "Proceed with these integrations?"},
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Create workflow")],
        model="test-model",
        stream=True,
        use_workflow="feedback_small_flow",
        workflow_execution_id="exec-feedback-small",
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    assert any(
        "USER_FEEDBACK_QUESTION" in (call.get("base_prompt") or "")
        for call in llm.ask_calls
    )


@pytest.mark.asyncio
async def test_feedback_input_field_value_appended_as_user_message(
    tmp_path, monkeypatch
):
    """
    When a user submits feedback with a per-choice input field value, that value
    should be appended as a user message so the target agent's LLM sees it.
    """
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "input_resume_flow",
        "root_intent": "INPUT_RESUME_TEST",
        "agents": [
            {
                "agent": "select_tools",
                "description": "Select tools.",
                "reroute": [
                    {
                        "on": ["ASK_USER"],
                        "ask": {
                            "question": "Proceed or adjust?",
                            "expected_responses": [
                                {
                                    "proceed": {
                                        "to": "create_plan",
                                        "with": ["selected_tools"],
                                    },
                                    "adjust": {
                                        "to": "select_tools",
                                        "value": "Adjust",
                                        "input": {
                                            "feedback": {
                                                "type": "string",
                                                "description": "What to adjust?",
                                            }
                                        },
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "agent": "create_plan",
                "description": "Create the plan.",
                "depends_on": ["select_tools"],
            },
        ],
    }
    _write_workflow(workflows_dir, "input_resume_flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    # First call: agent reroutes to ASK_USER, pauses for feedback
    # Second call (after feedback): agent runs again with the user's input
    llm = StubLLMClient(
        {
            "select_tools": [
                "<reroute>ASK_USER</reroute>",
                "Adjusted the selection.",
            ],
            "create_plan": "Plan created.",
        },
        ask_responses={"feedback_question": "Proceed or adjust?"},
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    exec_id = "exec-input-resume"
    original_query = "Generate a plan that every morning summarizes my commits from github in the inxm-ai org, and gives some hints what to do today"
    adjustment = (
        "Can you also add github issues, and microsoft teams so we can post the result?"
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content=original_query)],
        model="test-model",
        stream=True,
        use_workflow="input_resume_flow",
        workflow_execution_id=exec_id,
    )

    # First pass — pauses at feedback
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]
    state = store.load_execution(exec_id)
    assert state.awaiting_feedback is True

    # Resume with structured feedback selecting "adjust" + input field
    resume_request = ChatCompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content=f'<user_feedback>{{"action":"accept","content":{{"selection":"adjust","feedback":"{adjustment}"}}}}</user_feedback>',
            )
        ],
        model="test-model",
        stream=True,
        use_workflow="input_resume_flow",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    resume_chunks = [chunk async for chunk in stream]
    resume_payload = "\n".join(resume_chunks)

    state = store.load_execution(exec_id)
    assert "Reroute target 'select_tools' was already completed" not in resume_payload
    assert state.completed is True
    assert state.awaiting_feedback is False
    assert "create_plan" in llm.calls
    assert llm.calls.count("select_tools") == 2
    # The input text should have been appended to user_messages
    user_messages = state.context.get("user_messages", [])
    assert any(
        adjustment in msg for msg in user_messages
    ), f"Expected input text in user_messages, got: {user_messages}"
    # user_query should preserve original intent and include adjustment details
    user_query = state.context.get("user_query", "")
    assert original_query in user_query
    assert f"Adjustment: {adjustment}" in user_query
    assert user_query != adjustment

    # The select_tools agent should have been rerouted back to itself
    # and its feedback value stored in the agent context
    agent_ctx = state.context["agents"]["select_tools"]
    assert agent_ctx.get("feedback") == adjustment

    select_calls = [idx for idx, call in enumerate(llm.calls) if call == "select_tools"]
    assert len(select_calls) == 2
    rerun_prompt = llm.user_messages[select_calls[1]]
    assert original_query in rerun_prompt
    assert f"Adjustment: {adjustment}" in rerun_prompt


@pytest.mark.asyncio
async def test_feedback_adjust_self_reroute_preserves_original_intent(
    tmp_path, monkeypatch
):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "adjust_self_flow",
        "root_intent": "ADJUST_SELF_TEST",
        "agents": [
            {
                "agent": "select_tools",
                "description": "Select tools.",
                "reroute": [
                    {
                        "on": ["ASK_USER"],
                        "ask": {
                            "question": "Proceed or adjust?",
                            "expected_responses": [
                                {
                                    "proceed": {
                                        "to": "create_plan",
                                        "with": ["selected_tools"],
                                    },
                                    "adjust": {
                                        "to": "select_tools",
                                        "value": "Adjust",
                                        "input": {
                                            "feedback": {
                                                "type": "string",
                                                "description": "What to adjust?",
                                            }
                                        },
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "agent": "create_plan",
                "description": "Create plan.",
                "depends_on": ["select_tools"],
            },
        ],
    }
    _write_workflow(workflows_dir, "adjust_self_flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "select_tools": ["<reroute>ASK_USER</reroute>", "Adjusted selection."],
            "create_plan": "Plan created.",
        },
        ask_responses={"feedback_question": "Proceed or adjust?"},
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    original_query = "Generate a plan that every morning summarizes my commits from github in the inxm-ai org, and gives some hints what to do today"
    adjustment = (
        "Can you also add github issues, and microsoft teams so we can post the result?"
    )

    exec_id = "exec-adjust-self"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content=original_query)],
        model="test-model",
        stream=True,
        use_workflow="adjust_self_flow",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    resume_request = ChatCompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content=f'<user_feedback>{{"action":"accept","content":{{"selection":"adjust","feedback":"{adjustment}"}}}}</user_feedback>',
            )
        ],
        model="test-model",
        stream=True,
        use_workflow="adjust_self_flow",
        workflow_execution_id=exec_id,
    )
    resume_stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    _ = [chunk async for chunk in resume_stream]

    state = store.load_execution(exec_id)
    assert state.completed is True
    user_query = state.context.get("user_query", "")
    assert original_query in user_query
    assert f"Adjustment: {adjustment}" in user_query
    assert llm.calls.count("select_tools") == 2

    select_calls = [idx for idx, call in enumerate(llm.calls) if call == "select_tools"]
    rerun_prompt = llm.user_messages[select_calls[1]]
    assert original_query in rerun_prompt
    assert f"Adjustment: {adjustment}" in rerun_prompt


@pytest.mark.asyncio
async def test_feedback_input_field_non_self_reroute_keeps_original_user_query(
    tmp_path, monkeypatch
):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "adjust_other_flow",
        "root_intent": "ADJUST_OTHER_TEST",
        "agents": [
            {
                "agent": "select_tools",
                "description": "Select tools.",
                "reroute": [
                    {
                        "on": ["ASK_USER"],
                        "ask": {
                            "question": "Proceed or adjust?",
                            "expected_responses": [
                                {
                                    "proceed": {"to": "finalize"},
                                    "adjust": {
                                        "to": "review_tools",
                                        "value": "Adjust",
                                        "input": {
                                            "feedback": {
                                                "type": "string",
                                                "description": "What to adjust?",
                                            }
                                        },
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "agent": "review_tools",
                "description": "Review selected tools.",
                "depends_on": ["select_tools"],
            },
            {
                "agent": "finalize",
                "description": "Finalize the plan.",
                "depends_on": ["review_tools"],
            },
        ],
    }
    _write_workflow(workflows_dir, "adjust_other_flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "select_tools": "<reroute>ASK_USER</reroute>",
            "review_tools": "Reviewed.",
            "finalize": "Done.",
        },
        ask_responses={"feedback_question": "Proceed or adjust?"},
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    original_query = "Generate a plan that every morning summarizes my commits."
    adjustment = "Also include github issues and microsoft teams posting."

    exec_id = "exec-adjust-other"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content=original_query)],
        model="test-model",
        stream=True,
        use_workflow="adjust_other_flow",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    resume_request = ChatCompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content=f'<user_feedback>{{"action":"accept","content":{{"selection":"adjust","feedback":"{adjustment}"}}}}</user_feedback>',
            )
        ],
        model="test-model",
        stream=True,
        use_workflow="adjust_other_flow",
        workflow_execution_id=exec_id,
    )
    resume_stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    _ = [chunk async for chunk in resume_stream]

    state = store.load_execution(exec_id)
    assert state.completed is True
    assert llm.calls.count("select_tools") == 1
    assert original_query == state.context.get("user_query")

    user_messages = state.context.get("user_messages", [])
    assert any(adjustment in msg for msg in user_messages)


@pytest.mark.asyncio
async def test_feedback_self_reroute_function_syntax_reruns_agent(
    tmp_path, monkeypatch
):
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "self_reroute_feedback_flow",
        "root_intent": "SELF_REROUTE_FEEDBACK_TEST",
        "agents": [
            {
                "agent": "select_tools",
                "description": "Select tools.",
                "reroute": [
                    {
                        "on": ["ASK_USER"],
                        "ask": {
                            "question": "Proceed or adjust?",
                            "expected_responses": [
                                {
                                    "proceed": {
                                        "to": "create_plan",
                                        "with": ["selected_tools"],
                                    },
                                    "adjust": {
                                        "to": "select_tools",
                                        "value": "Adjust",
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "agent": "create_plan",
                "description": "Create the plan.",
                "depends_on": ["select_tools"],
            },
        ],
    }
    _write_workflow(workflows_dir, "self_reroute_feedback_flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {
            "select_tools": [
                "<reroute>ASK_USER</reroute>",
                "Adjusted the selection.",
            ],
            "create_plan": "Plan created.",
        },
        ask_responses={"feedback_question": "Proceed or adjust?"},
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    exec_id = "exec-self-reroute-fn"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Run")],
        model="test-model",
        stream=True,
        use_workflow="self_reroute_feedback_flow",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    state = store.load_execution(exec_id)
    assert state.awaiting_feedback is True

    resume_request = ChatCompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content="<user_feedback>select_tools()</user_feedback>",
            )
        ],
        model="test-model",
        stream=True,
        use_workflow="self_reroute_feedback_flow",
        workflow_execution_id=exec_id,
    )
    resume_stream = await engine.start_or_resume_workflow(
        StubSession(), resume_request, None, None, None
    )
    resume_chunks = [chunk async for chunk in resume_stream]
    resume_payload = "\n".join(resume_chunks)

    final_state = store.load_execution(exec_id)
    assert "Reroute target 'select_tools' was already completed" not in resume_payload
    assert final_state.completed is True
    assert final_state.awaiting_feedback is False
    assert llm.calls.count("select_tools") == 2
    assert "create_plan" in llm.calls


@pytest.mark.asyncio
async def test_feedback_without_input_field_unchanged(tmp_path, monkeypatch):
    """
    When no per-choice input fields are defined, the requestedSchema should
    remain the same as before (backward compatibility).
    """
    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "no_input_flow",
        "root_intent": "NO_INPUT_TEST",
        "agents": [
            {
                "agent": "select_tools",
                "description": "Select tools.",
                "reroute": [
                    {
                        "on": ["ASK_USER"],
                        "ask": {
                            "question": "Proceed or adjust?",
                            "expected_responses": [
                                {
                                    "proceed": {
                                        "to": "create_plan",
                                        "with": ["selected_tools"],
                                    },
                                    "adjust": {
                                        "to": "select_tools",
                                        "value": "Adjust",
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "agent": "create_plan",
                "description": "Create the plan.",
                "depends_on": ["select_tools"],
            },
        ],
    }
    _write_workflow(workflows_dir, "no_input_flow", flow)
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(
        {"select_tools": "<reroute>ASK_USER</reroute>"},
        ask_responses={"feedback_question": "Proceed or adjust?"},
    )
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    engine = WorkflowEngine(WorkflowRepository(), store, llm)

    exec_id = "exec-no-input"
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Run")],
        model="test-model",
        stream=True,
        use_workflow="no_input_flow",
        workflow_execution_id=exec_id,
    )
    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    state = store.load_execution(exec_id)
    elic_spec = state.context["agents"]["select_tools"].get("elicitation_spec") or {}
    schema = elic_spec.get("requestedSchema") or {}
    props = schema.get("properties", {})

    # Should only have selection and value — no extra input fields
    assert set(props.keys()) == {"selection", "value"}
    assert schema.get("required") == ["selection"]

    # meta should NOT have input_fields
    meta = elic_spec.get("meta", {})
    assert "input_fields" not in meta
