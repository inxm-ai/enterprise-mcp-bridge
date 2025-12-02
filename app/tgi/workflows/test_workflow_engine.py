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
    def __init__(self, prompts: list[Any] | None = None):
        self.prompts = prompts or []

    async def list_prompts(self):
        return SimpleNamespace(prompts=self.prompts)

    async def call_prompt(self, name: str, args: dict[str, Any]):
        # Return the prompt text in the same shape PromptService expects
        text = next(
            (p.content for p in self.prompts if getattr(p, "name", "") == name), ""
        )
        message = SimpleNamespace(content=SimpleNamespace(text=text))
        return SimpleNamespace(isError=False, messages=[message])


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

    def stream_completion(
        self, request: ChatCompletionRequest, access_token: str, span: Any
    ) -> AsyncGenerator[str, None]:
        last_message = request.messages[-1].content or ""
        # Agent marker is appended by the workflow engine
        agent_marker = ""
        if "<agent:" in last_message:
            agent_marker = last_message.split("<agent:", 1)[1].split(">", 1)[0]
        response = self.responses.get(agent_marker, "")
        self.calls.append(agent_marker)

        async def _gen():
            yield f"data: {response}\n\n"
            yield "data: [DONE]\n\n"

        return _gen()


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
    assert llm.calls == ["get_location", "get_weather"]


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
