import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncGenerator

import pytest

from app.tgi.models import ChatCompletionRequest, Message, MessageRole
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

    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.calls: list[str] = []
        self.request_tools: list[list[str] | None] = []
        self.system_prompts: list[str] = []

    def stream_completion(
        self, request: ChatCompletionRequest, access_token: str, span: Any
    ) -> AsyncGenerator[str, None]:
        if request.messages and request.messages[0].role == MessageRole.SYSTEM:
            self.system_prompts.append(request.messages[0].content or "")

        last_message = request.messages[-1].content or ""
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

        response = self.responses.get(agent_marker, "")
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

    assert "Fetching your location" in payload
    assert "San Francisco" in payload  # pass-through shown
    assert "Workflow complete" in payload

    state = store.load_execution("exec-pass")
    assert state.completed is True
    assert state.context["agents"]["get_location"]["content"] == "San Francisco"
    assert state.context["agents"]["get_weather"]["content"] == "Sunny"
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
            "ask_preference": "<user_feedback_needed>Please tell me if you prefer indoors</user_feedback_needed>",
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
    # history replayed
    assert first_chunks[0] in resume_chunks[0]
    assert "indoor plan" in resume_payload.lower()

    final_state = store.load_execution("exec-feedback")
    assert final_state.completed is True
    assert final_state.awaiting_feedback is False
    assert final_state.context["agents"]["ask_preference"]["content"].endswith(
        "prefer indoors"
    )
