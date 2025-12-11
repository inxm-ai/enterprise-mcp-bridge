import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncGenerator

import pytest

from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.services.tool_service import ToolService
from app.tgi.workflows.engine import WorkflowEngine
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

    def __init__(self, responses: dict[str, str | list[str]]):
        self.responses = responses
        self.calls: list[str] = []
        self.request_tools: list[list[str] | None] = []
        self.system_prompts: list[str] = []
        self.response_counters: dict[str, int] = {}
        self.user_messages: list[str] = []

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


def _write_workflow(tmpdir: Path, name: str, payload: dict[str, Any]) -> None:
    (tmpdir / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


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

    stream = await engine.start_or_resume_workflow(StubSession(), request, None, None)
    assert stream is not None
    chunks = [chunk async for chunk in stream]
    payload = "\n".join(chunks)

    assert "<workflow_execution_id>exec-pass</workflow_execution_id>" in payload
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

    stream = await engine.start_or_resume_workflow(StubSession(), request, None, None)
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

    stream = await engine.start_or_resume_workflow(StubSession(), request, None, None)
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

    stream = await engine.start_or_resume_workflow(StubSession(), request, None, None)
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

    stream = await engine.start_or_resume_workflow(StubSession(), request, None, None)
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

    stream = await engine.start_or_resume_workflow(StubSession(), request, None, None)
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

    stream = await engine.start_or_resume_workflow(StubSession(), request, None, None)
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
        StubSession(tools=tools), request, None, None
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
    stream = await engine.start_or_resume_workflow(StubSession(), request, None, None)
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
    stream = await engine.start_or_resume_workflow(StubSession(), request, None, None)
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

    stream = await engine.start_or_resume_workflow(StubSession(), request, None, None)
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
        StubSession(), request, None, None
    )
    resume_chunks = [chunk async for chunk in resume_stream]
    assert resume_chunks[-1].strip() == "data: [DONE]"
    # No new LLM invocations on resume because the workflow was marked complete
    assert len(llm.calls) == initial_call_count


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

    stream = await engine.start_or_resume_workflow(StubSession(), request, None, None)
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

    stream = await engine.start_or_resume_workflow(StubSession(), request, None, None)
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
    stream = await engine.start_or_resume_workflow(StubSession(), request, None, None)
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
        StubSession(tools=tools), request, None, None
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
    stream = await engine.start_or_resume_workflow(session, request, None, None)
    _ = [chunk async for chunk in stream]

    # Even though the agent requested a missing tool, engine should pass all tools
    # (plus the lazy context tool added by routing)
    assert set(llm.request_tools[0]) == {"available_tool", "get_workflow_context"}


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
        StubSession(tools=tools), request, None, None
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
    stream = await engine.start_or_resume_workflow(StubSession(), request, None, None)
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
        StubSession(prompts=[routing_prompt]), request, None, None
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
        StubSession(tools=tools), request, None, None
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
        StubSession(), initial_request, None, None
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
        StubSession(), resume_request, None, None
    )
    assert resume_stream is not None
    resume_chunks = [chunk async for chunk in resume_stream]
    resume_payload = "\n".join(resume_chunks)
    # history replayed and execution id delivered first
    assert (
        "<workflow_execution_id>exec-feedback</workflow_execution_id>"
        in resume_chunks[0]
    )
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
        StubSession(), initial_request, None, None
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
        StubSession(), resume_request, None, None
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
        StubSession(tools=tools), request, None, None
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

    stream = await engine.start_or_resume_workflow(StubSession(), request, None, None)
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

    stream = await engine.start_or_resume_workflow(StubSession(), request, None, None)
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
    stream = await engine.start_or_resume_workflow(session, request, None, None)
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

    stream = await engine.start_or_resume_workflow(StubSession(), request, None, None)
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
        StubSession(tools=tools), request, None, None
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
    stream = await engine.start_or_resume_workflow(session, request, None, None)
    _ = [chunk async for chunk in stream]

    # The save_plan tool should receive injected plan arg despite context False
    save_calls = [c for c in session.tool_calls if c["name"] == "save_plan"]
    assert save_calls, "save_plan tool was not called"
    assert save_calls[0]["args"].get("plan") is not None


@pytest.mark.asyncio
async def test_tool_error_detection_various_formats(tmp_path, monkeypatch):
    """Test that _tool_result_has_error detects various error formats."""
    from app.tgi.workflows.engine import WorkflowEngine
    from app.tgi.workflows.repository import WorkflowRepository
    from app.tgi.workflows.state import WorkflowStateStore

    engine = WorkflowEngine(
        WorkflowRepository(workflows_path=str(tmp_path)),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        StubLLMClient({}),
    )

    # JSON with error key
    assert engine._tool_result_has_error('{"error": "something went wrong"}') is True

    # JSON with isError flag
    assert engine._tool_result_has_error('{"isError": true}') is True

    # JSON with success false
    assert engine._tool_result_has_error('{"success": false}') is True

    # List with error item
    assert engine._tool_result_has_error('[{"error": "fail"}]') is True

    # Plain text with HTTP error
    assert engine._tool_result_has_error("Client error '400 Bad Request'") is True
    assert engine._tool_result_has_error("500 Internal Server Error") is True
    assert engine._tool_result_has_error("401 unauthorized access") is True

    # Normal successful responses
    assert engine._tool_result_has_error('{"result": "success"}') is False
    assert engine._tool_result_has_error("Plan saved successfully") is False
    assert engine._tool_result_has_error('{"data": [1, 2, 3]}') is False
    assert engine._tool_result_has_error("") is False


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
        StubSession(tools=tools), request, None, None
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
        access_token=None,
        span=None,
        no_reroute=False,
    )
    _ = [chunk async for chunk in stream]

    assert state.completed is True
    # First agent context exists with error content; second agent never ran
    assert "generate_inputs" in state.context.get("agents", {})
    assert "should_not_run" not in state.context.get("agents", {})
