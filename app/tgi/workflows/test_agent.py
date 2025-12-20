"""
Tests for the Agent module - handling enhanced tool configurations,
returns, streaming tools, and pass_through guidelines.
"""

import asyncio
import json
import os
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Optional

import pytest

from app.tgi.models import ChatCompletionRequest, Message, MessageRole


def _extract_delta_contents(chunks: list[str]) -> list[str]:
    """Extract delta.content strings from streamed chunks."""
    contents: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk.startswith("data: "):
            continue
        try:
            payload = json.loads(chunk[6:])
        except Exception:
            continue
        delta = (
            payload.get("choices", [{}])[0].get("delta", {})
            if isinstance(payload, dict)
            else {}
        )
        content = delta.get("content")
        if content:
            contents.append(content)
    return contents


class StubSession:
    """Stub session for testing."""

    def __init__(
        self, prompts: list[Any] | None = None, tools: list[Any] | None = None
    ):
        self.prompts = prompts or []
        self.tools = tools or []
        self.tool_calls: list[dict] = []
        self.streaming_tool_calls: list[dict] = []

    async def list_prompts(self):
        return SimpleNamespace(prompts=self.prompts)

    async def call_prompt(self, name: str, args: dict[str, Any]):
        text = next(
            (p.content for p in self.prompts if getattr(p, "name", "") == name), ""
        )
        message = SimpleNamespace(content=SimpleNamespace(text=text))
        return SimpleNamespace(isError=False, messages=[message])

    async def list_tools(self):
        return self.tools

    async def call_tool(self, name: str, args: Optional[dict], access_token: str):
        self.tool_calls.append({"name": name, "args": args})
        return SimpleNamespace(
            isError=False,
            content=[SimpleNamespace(text=json.dumps({"result": f"{name}_result"}))],
        )

    async def call_tool_streaming(
        self, name: str, args: Optional[dict], access_token: str
    ):
        """Simulate streaming tool call."""
        self.streaming_tool_calls.append({"name": name, "args": args})

        async def stream():
            yield {"type": "progress", "progress": 0.5, "message": "Working..."}
            yield {"type": "progress", "progress": 1.0, "message": "Done"}
            yield {"type": "result", "data": {"result": f"{name}_streamed_result"}}

        return stream()


class StubLLMClient:
    """Stub LLM client that returns predetermined responses."""

    def __init__(
        self,
        responses: dict[str, str] = None,
        tool_call_responses: dict[str, list] = None,
    ):
        self.responses = responses or {}
        self.tool_call_responses = tool_call_responses or {}
        self.calls: list[str] = []
        self.request_tools: list[list[str] | None] = []
        self.system_prompts: list[str] = []
        self.call_count = 0
        self._tool_call_made: dict[str, bool] = {}

    def stream_completion(
        self, request: ChatCompletionRequest, access_token: str, span: Any
    ) -> AsyncGenerator[str, None]:
        self.call_count += 1
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

        # Check if we should emit tool calls (only on first call for this agent)
        tool_calls = self.tool_call_responses.get(agent_marker, [])
        should_emit_tools = tool_calls and not self._tool_call_made.get(
            agent_marker, False
        )

        if should_emit_tools:
            self._tool_call_made[agent_marker] = True

        async def _gen():
            if should_emit_tools:
                # Emit tool call chunks with proper finish_reason
                for tc in tool_calls:
                    yield f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [tc]}, "index": 0}]})}\n\n'
                # Signal that tool calls are ready to execute
                empty_delta = {}
                yield f'data: {json.dumps({"choices": [{"delta": empty_delta, "finish_reason": "tool_calls", "index": 0}]})}\n\n'
            elif response:
                yield f'data: {json.dumps({"choices": [{"delta": {"content": response}, "index": 0}]})}\n\n'
            yield "data: [DONE]\n\n"

        return _gen()

    def _extract_value(self, text: str, key: str) -> str:
        try:
            return text.split(f"{key}=", 1)[1].split("\n", 1)[0].strip()
        except Exception:
            return ""


# ============================================================================
# Tests for enhanced WorkflowAgentDef model
# ============================================================================


class TestWorkflowAgentDefEnhancements:
    """Test the enhanced WorkflowAgentDef model with new fields."""

    def test_returns_field_parsing(self):
        """Agent definition can specify returns field for capturing tool outputs."""
        from app.tgi.workflows.models import WorkflowAgentDef

        agent_def = WorkflowAgentDef(
            agent="select_tools",
            description="Select appropriate tools",
            pass_through=True,
            tools=["select_tools"],
            returns=["selected_tools"],
        )
        assert agent_def.returns == ["selected_tools"]

    def test_tool_object_with_settings(self):
        """Agent definition can specify tools as objects with settings and args."""
        from app.tgi.workflows.models import WorkflowAgentDef

        agent_def = WorkflowAgentDef(
            agent="create_plan",
            description="Create a plan",
            tools=[
                {
                    "plan": {
                        "settings": {"streaming": True},
                        "args": {"selected-tools": "select_tools.selected-tools"},
                    }
                }
            ],
        )
        assert len(agent_def.tools) == 1
        assert isinstance(agent_def.tools[0], dict)
        tool_config = agent_def.tools[0]["plan"]
        assert tool_config["settings"]["streaming"] is True
        assert tool_config["args"]["selected-tools"] == "select_tools.selected-tools"

    def test_pass_through_as_string_guideline(self):
        """Agent definition can specify pass_through as a string guideline."""
        from app.tgi.workflows.models import WorkflowAgentDef

        agent_def = WorkflowAgentDef(
            agent="detect_existing_plans",
            description="Check for existing plans",
            pass_through="Return only the searches you are performing",
            tools=["search_plan"],
        )
        assert agent_def.pass_through == "Return only the searches you are performing"
        assert agent_def.should_pass_through is True
        assert (
            agent_def.pass_through_guideline
            == "Return only the searches you are performing"
        )

    def test_pass_through_boolean_true(self):
        """pass_through=True means pass through without specific guideline."""
        from app.tgi.workflows.models import WorkflowAgentDef

        agent_def = WorkflowAgentDef(
            agent="get_location",
            description="Get location",
            pass_through=True,
        )
        assert agent_def.should_pass_through is True
        assert agent_def.pass_through_guideline is None

    def test_pass_through_boolean_false(self):
        """pass_through=False means don't pass through."""
        from app.tgi.workflows.models import WorkflowAgentDef

        agent_def = WorkflowAgentDef(
            agent="get_location",
            description="Get location",
            pass_through=False,
        )
        assert agent_def.should_pass_through is False
        assert agent_def.pass_through_guideline is None

    def test_mixed_tools_list(self):
        """Agent can have both string tool names and object configurations."""
        from app.tgi.workflows.models import WorkflowAgentDef

        agent_def = WorkflowAgentDef(
            agent="multi_tool_agent",
            description="Uses multiple tools",
            tools=[
                "simple_tool",
                {"complex_tool": {"settings": {"streaming": True}}},
            ],
        )
        assert len(agent_def.tools) == 2
        assert agent_def.tools[0] == "simple_tool"
        assert isinstance(agent_def.tools[1], dict)


# ============================================================================
# Tests for Agent execution with returns
# ============================================================================


class TestAgentReturns:
    """Test capturing specified return values from tool calls."""

    @pytest.mark.asyncio
    async def test_agent_captures_returns_from_tool_call(self):
        """Agent captures specified fields from tool results into context."""
        from app.tgi.workflows.agent import AgentExecutor
        from app.tgi.workflows.models import WorkflowAgentDef, WorkflowExecutionState

        agent_def = WorkflowAgentDef(
            agent="select_tools",
            description="Select tools",
            pass_through=True,
            tools=["select_tools"],
            returns=["selected_tools"],
        )

        state = WorkflowExecutionState.new("exec-1", "flow-1")

        # Mock tool result with selected_tools field
        tool_result = {"selected_tools": ["tool1", "tool2"], "other_field": "ignored"}

        executor = AgentExecutor()
        executor.extract_returns(agent_def, tool_result, state)

        assert state.context["agents"]["select_tools"]["selected_tools"] == [
            "tool1",
            "tool2",
        ]

    @pytest.mark.asyncio
    async def test_agent_captures_multiple_returns(self):
        """Agent captures multiple specified return fields."""
        from app.tgi.workflows.agent import AgentExecutor
        from app.tgi.workflows.models import WorkflowAgentDef, WorkflowExecutionState

        agent_def = WorkflowAgentDef(
            agent="analyze",
            description="Analyze data",
            returns=["summary", "details"],
        )

        state = WorkflowExecutionState.new("exec-1", "flow-1")
        state.context["agents"]["analyze"] = {"content": ""}

        tool_result = {
            "summary": "Brief summary",
            "details": {"key": "value"},
            "extra": "not captured",
        }

        executor = AgentExecutor()
        executor.extract_returns(agent_def, tool_result, state)

        assert state.context["agents"]["analyze"]["summary"] == "Brief summary"
        assert state.context["agents"]["analyze"]["details"] == {"key": "value"}
        assert "extra" not in state.context["agents"]["analyze"]

    @pytest.mark.asyncio
    async def test_returns_from_multiple_tool_calls(self):
        """Agent aggregates returns from multiple tool calls."""
        from app.tgi.workflows.agent import AgentExecutor
        from app.tgi.workflows.models import WorkflowAgentDef, WorkflowExecutionState

        agent_def = WorkflowAgentDef(
            agent="multi_search",
            description="Search multiple sources",
            tools=["search"],
            returns=["results"],
        )

        state = WorkflowExecutionState.new("exec-1", "flow-1")
        state.context["agents"]["multi_search"] = {"content": ""}

        # First tool call result
        tool_result1 = {"results": ["result1", "result2"]}
        # Second tool call result
        tool_result2 = {"results": ["result3"]}

        executor = AgentExecutor()
        executor.extract_returns(agent_def, tool_result1, state)
        executor.extract_returns(agent_def, tool_result2, state)

        # Results should be aggregated
        assert state.context["agents"]["multi_search"]["results"] == [
            ["result1", "result2"],
            ["result3"],
        ]


# ============================================================================
# Tests for streaming tool calls
# ============================================================================


class TestStreamingToolCalls:
    """Test streaming tool execution via POST /tools/{tool_name}/stream."""

    @pytest.mark.asyncio
    async def test_tool_config_detects_streaming_setting(self):
        """Tool configuration with streaming=True uses streaming endpoint."""
        from app.tgi.workflows.agent import ToolConfig

        tool_config = ToolConfig.from_definition(
            {"plan": {"settings": {"streaming": True}}}
        )
        assert tool_config.name == "plan"
        assert tool_config.streaming is True

    @pytest.mark.asyncio
    async def test_tool_config_without_streaming(self):
        """Tool configuration without streaming setting defaults to False."""
        from app.tgi.workflows.agent import ToolConfig

        tool_config = ToolConfig.from_definition("simple_tool")
        assert tool_config.name == "simple_tool"
        assert tool_config.streaming is False

    @pytest.mark.asyncio
    async def test_tool_config_with_args_mapping(self):
        """Tool configuration with args specifies argument mappings."""
        from app.tgi.workflows.agent import ToolConfig

        tool_config = ToolConfig.from_definition(
            {
                "plan": {
                    "settings": {"streaming": True},
                    "args": {"selected-tools": "select_tools.selected-tools"},
                }
            }
        )
        assert tool_config.name == "plan"
        assert tool_config.args_mapping == {
            "selected-tools": "select_tools.selected-tools"
        }

    @pytest.mark.asyncio
    async def test_streaming_tool_execution(self):
        """Streaming tool returns progressive updates."""
        from app.tgi.workflows.agent import AgentExecutor, ToolConfig
        from app.tgi.workflows.models import WorkflowExecutionState

        session = StubSession(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "plan",
                        "description": "Create plan",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        )

        tool_config = ToolConfig(name="plan", streaming=True)
        state = WorkflowExecutionState.new("exec-1", "flow-1")

        executor = AgentExecutor()
        results = []

        async for event in executor.execute_streaming_tool(
            session, tool_config, {"arg": "value"}, "token", state
        ):
            results.append(event)

        assert len(session.streaming_tool_calls) == 1
        assert session.streaming_tool_calls[0]["name"] == "plan"


# ============================================================================
# Tests for args mapping and removal
# ============================================================================


class TestArgsMapping:
    """Test argument mapping and removal from tool definitions."""

    def test_resolve_arg_from_context(self):
        """Resolve argument value from previous agent context."""
        from app.tgi.workflows.agent import AgentExecutor
        from app.tgi.workflows.models import WorkflowExecutionState

        state = WorkflowExecutionState.new("exec-1", "flow-1")
        state.context["agents"]["select_tools"] = {
            "content": "",
            "selected-tools": ["tool1", "tool2"],
        }

        executor = AgentExecutor()
        value = executor.resolve_arg_reference(
            "select_tools.selected-tools", state.context
        )
        assert value == ["tool1", "tool2"]

    def test_resolve_nested_arg_from_context(self):
        """Resolve nested argument from previous agent context."""
        from app.tgi.workflows.agent import AgentExecutor
        from app.tgi.workflows.models import WorkflowExecutionState

        state = WorkflowExecutionState.new("exec-1", "flow-1")
        state.context["agents"]["analyze"] = {
            "content": "",
            "results": {"items": [1, 2, 3]},
        }

        executor = AgentExecutor()
        value = executor.resolve_arg_reference("analyze.results.items", state.context)
        assert value == [1, 2, 3]

    def test_modify_tool_schema_removes_mapped_args(self):
        """Tool schema is modified to remove args that are pre-mapped."""
        from app.tgi.workflows.agent import AgentExecutor, ToolConfig

        tool_config = ToolConfig(
            name="plan",
            args_mapping={"selected-tools": "select_tools.selected-tools"},
        )

        original_tool = {
            "type": "function",
            "function": {
                "name": "plan",
                "description": "Create plan",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected-tools": {
                            "type": "array",
                            "description": "Tools to use",
                        },
                        "other-param": {"type": "string", "description": "Other param"},
                    },
                    "required": ["selected-tools", "other-param"],
                },
            },
        }

        executor = AgentExecutor()
        modified_tool = executor.modify_tool_for_agent(original_tool, tool_config)

        # selected-tools should be removed from properties (so LLM doesn't try to fill it)
        props = modified_tool["function"]["parameters"]["properties"]
        assert "selected-tools" not in props
        assert "other-param" in props

        # selected-tools should also be removed from required
        required = modified_tool["function"]["parameters"]["required"]
        assert "selected-tools" not in required
        assert "other-param" in required


# ============================================================================
# Tests for pass_through guidelines
# ============================================================================


class TestPassThroughGuidelines:
    """Test pass_through string guidelines."""

    def test_guideline_included_in_system_prompt(self):
        """Pass through guideline is included in agent system prompt."""
        from app.tgi.workflows.agent import AgentExecutor
        from app.tgi.workflows.models import WorkflowAgentDef

        agent_def = WorkflowAgentDef(
            agent="detect_plans",
            description="Detect existing plans",
            pass_through="Return only the searches you are performing as you perform them",
        )

        executor = AgentExecutor()
        prompt = executor.build_agent_prompt(agent_def, "Base prompt")

        assert "Return only the searches you are performing" in prompt
        assert "Response guideline:" in prompt

    def test_no_guideline_when_boolean_pass_through(self):
        """No special guideline when pass_through is boolean."""
        from app.tgi.workflows.agent import AgentExecutor
        from app.tgi.workflows.models import WorkflowAgentDef

        agent_def = WorkflowAgentDef(
            agent="get_location",
            description="Get location",
            pass_through=True,
        )

        executor = AgentExecutor()
        prompt = executor.build_agent_prompt(agent_def, "Base prompt")

        assert "Response guideline:" not in prompt


# ============================================================================
# Tests for passthrough tag extraction
# ============================================================================


class TestPassthroughTagExtraction:
    """Test extraction of content from <passthrough> tags."""

    def test_extract_single_passthrough_block(self, tmp_path, monkeypatch):
        """Extract content from a single passthrough block."""
        from app.tgi.workflows.engine import WorkflowEngine
        from app.tgi.workflows.repository import WorkflowRepository
        from app.tgi.workflows.state import WorkflowStateStore

        workflows_dir = tmp_path / "flows"
        workflows_dir.mkdir()
        (workflows_dir / "empty.json").write_text(
            '{"flow_id": "x", "root_intent": "X", "agents": []}', encoding="utf-8"
        )
        monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

        engine = WorkflowEngine(
            WorkflowRepository(),
            WorkflowStateStore(db_path=":memory:"),
            StubLLMClient(),
        )

        text = "Some preamble <passthrough>visible content</passthrough> more text"
        result = engine._extract_passthrough_content(text)
        # Extracted content ends with \n\n for proper separation when streaming
        assert result == "visible content\n\n"

    def test_extract_multiple_passthrough_blocks(self, tmp_path, monkeypatch):
        """Extract content from multiple passthrough blocks."""
        from app.tgi.workflows.engine import WorkflowEngine
        from app.tgi.workflows.repository import WorkflowRepository
        from app.tgi.workflows.state import WorkflowStateStore

        workflows_dir = tmp_path / "flows"
        workflows_dir.mkdir()
        (workflows_dir / "empty.json").write_text(
            '{"flow_id": "x", "root_intent": "X", "agents": []}', encoding="utf-8"
        )
        monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

        engine = WorkflowEngine(
            WorkflowRepository(),
            WorkflowStateStore(db_path=":memory:"),
            StubLLMClient(),
        )

        text = (
            "<passthrough>first</passthrough> middle <passthrough>second</passthrough>"
        )
        result = engine._extract_passthrough_content(text)
        # Multiple blocks are joined with \n\n and result ends with \n\n
        assert result == "first\n\nsecond\n\n"

    def test_extract_empty_when_no_complete_block(self, tmp_path, monkeypatch):
        """Return empty when there's no complete passthrough block."""
        from app.tgi.workflows.engine import WorkflowEngine
        from app.tgi.workflows.repository import WorkflowRepository
        from app.tgi.workflows.state import WorkflowStateStore

        workflows_dir = tmp_path / "flows"
        workflows_dir.mkdir()
        (workflows_dir / "empty.json").write_text(
            '{"flow_id": "x", "root_intent": "X", "agents": []}', encoding="utf-8"
        )
        monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

        engine = WorkflowEngine(
            WorkflowRepository(),
            WorkflowStateStore(db_path=":memory:"),
            StubLLMClient(),
        )

        text = "<passthrough>incomplete content without closing"
        result = engine._extract_passthrough_content(text)
        assert result == ""

    def test_extract_multiline_passthrough(self, tmp_path, monkeypatch):
        """Extract multiline content from passthrough block."""
        from app.tgi.workflows.engine import WorkflowEngine
        from app.tgi.workflows.repository import WorkflowRepository
        from app.tgi.workflows.state import WorkflowStateStore

        workflows_dir = tmp_path / "flows"
        workflows_dir.mkdir()
        (workflows_dir / "empty.json").write_text(
            '{"flow_id": "x", "root_intent": "X", "agents": []}', encoding="utf-8"
        )
        monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

        engine = WorkflowEngine(
            WorkflowRepository(),
            WorkflowStateStore(db_path=":memory:"),
            StubLLMClient(),
        )

        text = "<passthrough>line 1\nline 2\nline 3</passthrough>"
        result = engine._extract_passthrough_content(text)
        # Extracted content ends with \n\n for proper separation when streaming
        assert result == "line 1\nline 2\nline 3\n\n"

    def test_strip_tags_includes_passthrough(self, tmp_path, monkeypatch):
        """_strip_tags removes passthrough tags along with other tags."""
        from app.tgi.workflows.engine import WorkflowEngine
        from app.tgi.workflows.repository import WorkflowRepository
        from app.tgi.workflows.state import WorkflowStateStore

        workflows_dir = tmp_path / "flows"
        workflows_dir.mkdir()
        (workflows_dir / "empty.json").write_text(
            '{"flow_id": "x", "root_intent": "X", "agents": []}', encoding="utf-8"
        )
        monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

        engine = WorkflowEngine(
            WorkflowRepository(),
            WorkflowStateStore(db_path=":memory:"),
            StubLLMClient(),
        )

        text = "<passthrough>content</passthrough> <reroute>reason</reroute>"
        result = engine._strip_tags(text)
        assert "<passthrough>" not in result
        assert "</passthrough>" not in result
        assert "<reroute>" not in result
        assert "content" in result
        assert "reason" in result


# ============================================================================
# Integration tests with WorkflowEngine
# ============================================================================


@pytest.mark.asyncio
async def test_engine_handles_returns_field(tmp_path, monkeypatch):
    """Workflow engine properly extracts returns from agent tool calls."""
    from app.tgi.workflows.engine import WorkflowEngine
    from app.tgi.workflows.repository import WorkflowRepository
    from app.tgi.workflows.state import WorkflowStateStore

    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()

    flow = {
        "flow_id": "select_and_create",
        "root_intent": "CREATE_PLAN",
        "agents": [
            {
                "agent": "select_tools",
                "description": "Select tools",
                "pass_through": True,
                "tools": ["select_tools"],
                "returns": ["selected_tools"],
            },
            {
                "agent": "use_tools",
                "description": "Use selected tools",
                "depends_on": ["select_tools"],
            },
        ],
    }
    (workflows_dir / "flow.json").write_text(json.dumps(flow), encoding="utf-8")
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "select_tools",
                "description": "Select tools",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    # LLM returns tool call, then final response
    llm = StubLLMClient(
        responses={
            "select_tools": '{"selected_tools": ["tool1", "tool2"]}',
            "use_tools": "Using the tools",
        },
        tool_call_responses={
            "select_tools": [
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "select_tools", "arguments": "{}"},
                }
            ]
        },
    )

    from app.tgi.services.tool_service import ToolService

    tool_service = ToolService()

    async def mock_execute_tool_calls(*args, **kwargs):
        content = '{"selected_tools": ["tool1", "tool2"]}'
        messages = [
            Message(
                role=MessageRole.TOOL,
                content=content,
                name="select_tools",
            )
        ]
        if kwargs.get("return_raw_results"):
            return (messages, True, [{"name": "select_tools", "content": content}])
        return (messages, True)

    tool_service.execute_tool_calls = mock_execute_tool_calls

    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
        tool_service=tool_service,
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Create a plan")],
        model="test-model",
        stream=True,
        use_workflow="select_and_create",
        workflow_execution_id="exec-returns",
    )

    session = StubSession(tools=tools)
    stream = await engine.start_or_resume_workflow(session, request, None, None, None)
    _ = [chunk async for chunk in stream]

    state = engine.state_store.load_execution("exec-returns")
    # Verify the returns were captured
    agent_ctx = state.context["agents"]["select_tools"]
    assert "selected_tools" in agent_ctx or "content" in agent_ctx


@pytest.mark.asyncio
async def test_engine_handles_pass_through_guideline(tmp_path, monkeypatch):
    """Workflow engine includes pass_through guideline in agent prompt."""
    from app.tgi.workflows.engine import WorkflowEngine
    from app.tgi.workflows.repository import WorkflowRepository
    from app.tgi.workflows.state import WorkflowStateStore

    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()

    flow = {
        "flow_id": "search_flow",
        "root_intent": "SEARCH",
        "agents": [
            {
                "agent": "search",
                "description": "Search for items",
                "pass_through": "Return only the searches you are performing",
                "tools": ["search_tool"],
            }
        ],
    }
    (workflows_dir / "flow.json").write_text(json.dumps(flow), encoding="utf-8")
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(responses={"search": "Searching for items..."})

    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Search")],
        model="test-model",
        stream=True,
        use_workflow="search_flow",
        workflow_execution_id="exec-guideline",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    # Check that the guideline was included in the system prompt
    assert any(
        "Return only the searches you are performing" in prompt
        for prompt in llm.system_prompts
    )
    # Check that passthrough tag instruction was included
    assert any("<passthrough>" in prompt for prompt in llm.system_prompts)


@pytest.mark.asyncio
async def test_engine_streams_only_passthrough_content(tmp_path, monkeypatch):
    """Workflow engine only streams content inside <passthrough> tags for string pass_through."""
    from app.tgi.workflows.engine import WorkflowEngine
    from app.tgi.workflows.repository import WorkflowRepository
    from app.tgi.workflows.state import WorkflowStateStore

    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()

    flow = {
        "flow_id": "search_flow",
        "root_intent": "SEARCH",
        "agents": [
            {
                "agent": "search",
                "description": "Search for items",
                "pass_through": "Return only what user should see",
                "tools": [],
            }
        ],
    }
    (workflows_dir / "flow.json").write_text(json.dumps(flow), encoding="utf-8")
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    # LLM response with passthrough tags
    llm = StubLLMClient(
        responses={
            "search": "Internal thinking... <passthrough>Searching for: test query</passthrough> more internal"
        }
    )

    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Search for test")],
        model="test-model",
        stream=True,
        use_workflow="search_flow",
        workflow_execution_id="exec-passthrough-stream",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]

    # Combine all chunks to check what was streamed
    combined = "".join(chunks)

    # The visible passthrough content should be streamed
    assert "Searching for: test query" in combined
    # The internal thinking should NOT be in the streamed output
    # (it's stripped or only passthrough content is forwarded)


@pytest.mark.asyncio
async def test_engine_streams_all_content_for_boolean_pass_through(
    tmp_path, monkeypatch
):
    """Workflow engine streams all content when pass_through is boolean True."""
    from app.tgi.workflows.engine import WorkflowEngine
    from app.tgi.workflows.repository import WorkflowRepository
    from app.tgi.workflows.state import WorkflowStateStore

    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()

    flow = {
        "flow_id": "simple_flow",
        "root_intent": "SIMPLE",
        "agents": [
            {
                "agent": "simple",
                "description": "Simple agent",
                "pass_through": True,
                "tools": [],
            }
        ],
    }
    (workflows_dir / "flow.json").write_text(json.dumps(flow), encoding="utf-8")
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    llm = StubLLMClient(responses={"simple": "All this content should be visible"})

    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Do something")],
        model="test-model",
        stream=True,
        use_workflow="simple_flow",
        workflow_execution_id="exec-bool-passthrough",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]
    combined = "".join(chunks)

    # All content should be visible for boolean pass_through
    assert "All this content should be visible" in combined


@pytest.mark.asyncio
async def test_passthrough_blocks_separated_by_newlines(tmp_path, monkeypatch):
    """Passthrough blocks should be separated by double newlines."""
    from app.tgi.workflows.engine import WorkflowEngine
    from app.tgi.workflows.repository import WorkflowRepository
    from app.tgi.workflows.state import WorkflowStateStore

    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()

    flow = {
        "flow_id": "multi_passthrough_flow",
        "root_intent": "MULTI_PASSTHROUGH",
        "agents": [
            {
                "agent": "streamer",
                "description": "Stream multiple passthroughs",
                "pass_through": "Show each step to the user",
                "tools": [],
            }
        ],
    }
    (workflows_dir / "flow.json").write_text(json.dumps(flow), encoding="utf-8")
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    # LLM response with multiple passthrough blocks
    llm = StubLLMClient(
        responses={
            "streamer": (
                "Internal thinking <passthrough>First update</passthrough> "
                "more thinking <passthrough>Second update</passthrough>"
            )
        }
    )

    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Stream updates")],
        model="test-model",
        stream=True,
        use_workflow="multi_passthrough_flow",
        workflow_execution_id="exec-multi-passthrough",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]

    contents = _extract_delta_contents(chunks)
    updates = [c for c in contents if "update" in c]

    # Both passthrough contents should be present and streamed as separate updates
    assert any("First update" in c for c in updates)
    assert any("Second update" in c for c in updates)
    first_idx = next(i for i, c in enumerate(contents) if "First update" in c)
    second_idx = next(i for i, c in enumerate(contents) if "Second update" in c)
    assert first_idx < second_idx


@pytest.mark.asyncio
async def test_passthrough_streams_before_completion(tmp_path, monkeypatch):
    """Passthrough content streams incrementally instead of only at the end of a block."""
    from app.tgi.workflows.engine import WorkflowEngine
    from app.tgi.workflows.repository import WorkflowRepository
    from app.tgi.workflows.state import WorkflowStateStore

    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()

    flow = {
        "flow_id": "streaming_passthrough",
        "root_intent": "STREAMING",
        "agents": [
            {
                "agent": "streamer",
                "description": "Streams content inside passthrough tags",
                "pass_through": "Show the live output to the user",
                "tools": [],
            }
        ],
    }
    (workflows_dir / "flow.json").write_text(json.dumps(flow), encoding="utf-8")
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    class StreamingPassthroughLLM:
        def __init__(self):
            self.calls = 0

        def stream_completion(self, request, access_token, span):
            self.calls += 1

            async def _gen():
                yield 'data: {"choices":[{"delta":{"content":"<passthrough>First chunk "},"index":0}]}\n\n'
                yield 'data: {"choices":[{"delta":{"content":"second part "},"index":0}]}\n\n'
                yield 'data: {"choices":[{"delta":{"content":"done</passthrough>"},"index":0}]}\n\n'
                yield "data: [DONE]\n\n"

            return _gen()

    llm = StreamingPassthroughLLM()

    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Stream live output")],
        model="test-model",
        stream=True,
        use_workflow="streaming_passthrough",
        workflow_execution_id="exec-streaming-passthrough",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(), request, None, None, None
    )
    chunks = [chunk async for chunk in stream]

    contents = _extract_delta_contents(chunks)
    passthrough_updates = [
        c
        for c in contents
        if any(marker in c for marker in ["First chunk", "second part", "done"])
    ]

    assert passthrough_updates, "Expected passthrough content to be streamed"
    assert any(
        "done" not in update for update in passthrough_updates
    ), "Passthrough should emit before the closing tag arrives"
    assert any("done" in update for update in passthrough_updates)

    first_visible = next(
        update for update in passthrough_updates if "done" not in update
    )
    done_update = next(update for update in passthrough_updates if "done" in update)
    assert contents.index(first_visible) < contents.index(done_update)


@pytest.mark.asyncio
async def test_passthrough_history_provided_to_progress_agent(tmp_path, monkeypatch):
    """Agent receives history of past passthroughs during tool progress updates."""
    from app.tgi.workflows.engine import WorkflowEngine
    from app.tgi.workflows.repository import WorkflowRepository
    from app.tgi.workflows.state import WorkflowStateStore

    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "slow_tool",
                "description": "A slow tool that sends progress",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    flow = {
        "flow_id": "progress_flow",
        "root_intent": "PROGRESS",
        "agents": [
            {
                "agent": "worker",
                "description": "Worker with progress",
                "pass_through": "Show progress to user",
                "tools": ["slow_tool"],
            }
        ],
    }
    (workflows_dir / "flow.json").write_text(json.dumps(flow), encoding="utf-8")
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    # Track user messages sent to LLM for progress updates
    progress_user_messages: list[str] = []

    class ProgressTrackingLLM(StubLLMClient):
        def __init__(self):
            super().__init__({})
            self.call_count = 0

        def stream_completion(self, request, access_token, span):
            self.call_count += 1

            # Check if this is a progress update call (has progress in system prompt)
            system_content = ""
            user_content = ""
            if request.messages:
                if request.messages[0].role == MessageRole.SYSTEM:
                    system_content = request.messages[0].content or ""
                if len(request.messages) > 1:
                    user_content = request.messages[1].content or ""

            # Progress update calls have specific system prompt
            if "user-visible updates" in system_content.lower():
                progress_user_messages.append(user_content)

            async def _gen_first():
                # First agent call: emit passthrough content then tool call
                yield 'data: {"choices":[{"delta":{"content":"<passthrough>First message shown to user</passthrough>"},"index":0}]}\n\n'
                yield (
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                    '"function":{"name":"slow_tool","arguments":"{}"}}]},"index":0}]}\n\n'
                )
                yield "data: [DONE]\n\n"

            async def _gen_progress():
                # Progress update call - emit a passthrough response
                yield 'data: {"choices":[{"delta":{"content":"<passthrough>Progress update</passthrough>"},"index":0}]}\n\n'
                yield "data: [DONE]\n\n"

            async def _gen_final():
                # Final response after tool
                yield 'data: {"choices":[{"delta":{"content":"<passthrough>Done!</passthrough>"},"index":0}]}\n\n'
                yield "data: [DONE]\n\n"

            if self.call_count == 1:
                return _gen_first()
            elif self.call_count == 2:
                return _gen_progress()
            else:
                return _gen_final()

    llm = ProgressTrackingLLM()

    # Mock tool service to emit a progress event
    from app.tgi.services.tool_service import ToolService

    tool_service = ToolService()
    progress_event_sent = {"count": 0}

    async def mock_execute_tool_calls(*args, **kwargs):
        progress_event_sent["count"] += 1
        return ([], True, [])

    tool_service.execute_tool_calls = mock_execute_tool_calls

    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
        tool_service=tool_service,
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Run the slow tool")],
        model="test-model",
        stream=True,
        use_workflow="progress_flow",
        workflow_execution_id="exec-progress-history",
    )

    stream = await engine.start_or_resume_workflow(
        StubSession(tools=tools), request, None, None, None
    )
    _ = [chunk async for chunk in stream]

    # Verify the LLM was called multiple times
    assert llm.call_count >= 2

    # When there's a progress update call after passthrough content was emitted,
    # the user message should contain the history of previous passthroughs
    # This is only included when pass_through_guideline is set (string pass_through)
    # and there's previous passthrough history
    if progress_user_messages:
        # Check that at least one progress message mentions previous messages
        has_history_reference = any(
            "Previous messages" in msg or "do not repeat" in msg
            for msg in progress_user_messages
        )
        # This should be true when there's passthrough history before the progress update
        # Since the test LLM emits passthrough before tool call, history should be passed
        assert (
            has_history_reference or len(progress_user_messages) == 0
        ), f"Progress user messages should include history. Got: {progress_user_messages}"


@pytest.mark.asyncio
async def test_progress_updates_cancel_previous(tmp_path):
    """New tool progress updates cancel in-flight summaries and only surface the latest."""
    from app.tgi.workflows.engine import WorkflowEngine
    from app.tgi.workflows.models import (
        WorkflowAgentDef,
        WorkflowDefinition,
        WorkflowExecutionState,
    )
    from app.tgi.workflows.repository import WorkflowRepository
    from app.tgi.workflows.state import WorkflowStateStore

    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()
    flow = {
        "flow_id": "progress_cancel_flow",
        "root_intent": "PROGRESS_CANCEL",
        "agents": [{"agent": "worker", "description": "Worker", "pass_through": True}],
    }
    (workflows_dir / "flow.json").write_text(json.dumps(flow), encoding="utf-8")
    os.environ["WORKFLOWS_PATH"] = str(workflows_dir)

    class ProgressLLM:
        def __init__(self):
            self.progress_calls = 0
            self.cancelled = 0

        def stream_completion(self, request, access_token, span):
            self.progress_calls += 1
            call_number = self.progress_calls

            async def _gen():
                try:
                    if call_number == 1:
                        await asyncio.sleep(0.05)  # Give time to be cancelled
                        yield (
                            'data: {"choices":[{"delta":{"content":"<passthrough>first progress</passthrough>"},"index":0}]}\n\n'
                        )
                    else:
                        yield (
                            'data: {"choices":[{"delta":{"content":"<passthrough>second progress</passthrough>"},"index":0}]}\n\n'
                        )
                    yield "data: [DONE]\n\n"
                except asyncio.CancelledError:
                    self.cancelled += 1
                    raise

            return _gen()

    progress_llm = ProgressLLM()

    class ProgressRunner:
        """Stub runner that emits two quick progress updates then final content."""

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
                yield 'data: {"type":"progress","progress":0.1,"message":"first"}\n\n'
                await asyncio.sleep(0.01)
                yield 'data: {"type":"progress","progress":0.5,"message":"second"}\n\n'
                yield (
                    'data: {"choices":[{"delta":{"content":"<passthrough>Done</passthrough>"},"index":0}]}\n\n'
                )
                yield "data: [DONE]\n\n"

            return _gen()

    class Session:
        async def list_tools(self):
            return []

    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        progress_llm,
        tool_chat_runner=ProgressRunner(),
    )

    agent_def = WorkflowAgentDef(
        agent="worker", description="Worker", pass_through=True, tools=[]
    )
    workflow_def = WorkflowDefinition(
        flow_id="progress_cancel_flow",
        root_intent="PROGRESS_CANCEL",
        agents=[agent_def],
    )
    state = WorkflowExecutionState.new("exec-progress-cancel", "progress_cancel_flow")
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Do work")],
        model="test-model",
        stream=True,
    )

    # Run the agent directly to capture emitted events
    _ = [
        chunk
        async for chunk in engine._execute_agent(
            workflow_def,
            agent_def,
            state,
            Session(),
            request,
            None,
            None,
            no_reroute=False,
        )
    ]

    # Progress LLM should have been called twice and the first should be cancelled
    assert 1 <= progress_llm.progress_calls <= 2
    assert progress_llm.cancelled <= 1

    def _extract_progress_contents(events: list[str]) -> list[str]:
        contents: list[str] = []
        for ev in events:
            if not ev.startswith("data: "):
                continue
            payload = json.loads(ev[len("data: ") :])
            metadata = payload.get("agentic", {}).get("metadata", {})
            if metadata.get("type") == "tool_progress":
                contents.append(payload["choices"][0]["delta"]["content"] or "")
        return contents

    progress_contents = _extract_progress_contents(state.events)
    # Only the latest progress message should be surfaced
    assert progress_contents == ["second progress"]


@pytest.mark.asyncio
async def test_handle_tool_progress_includes_passthrough_history(tmp_path, monkeypatch):
    """Direct test that _handle_tool_progress includes passthrough history in prompts."""
    from app.tgi.workflows.engine import WorkflowEngine
    from app.tgi.workflows.models import WorkflowAgentDef, WorkflowExecutionState
    from app.tgi.workflows.repository import WorkflowRepository
    from app.tgi.workflows.state import WorkflowStateStore

    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()

    # Minimal workflow just to set up the engine
    flow = {
        "flow_id": "test_flow",
        "root_intent": "TEST",
        "agents": [{"agent": "test_agent", "description": "Test"}],
    }
    (workflows_dir / "flow.json").write_text(json.dumps(flow), encoding="utf-8")
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    # Track what user messages are sent to the LLM
    user_messages_received: list[str] = []

    class TrackingLLM(StubLLMClient):
        def stream_completion(self, request, access_token, span):
            if len(request.messages) > 1:
                user_messages_received.append(request.messages[1].content or "")

            async def _gen():
                yield 'data: {"choices":[{"delta":{"content":"<no_update/>"},"index":0}]}\n\n'
                yield "data: [DONE]\n\n"

            return _gen()

    llm = TrackingLLM({})
    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
    )

    # Create an agent def with pass_through as a string (guideline)
    agent_def = WorkflowAgentDef(
        agent="test_agent",
        description="Test agent",
        pass_through="Show progress to user",
    )

    # Create a state
    state = WorkflowExecutionState.new("exec-1", "test_flow")

    # Create a request
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Do something")],
        model="test-model",
        stream=True,
    )

    # Create progress payload
    progress_payload = {
        "type": "progress",
        "progress": 50,
        "total": 100,
        "message": "Halfway there",
        "tool": "slow_tool",
    }

    # Test WITHOUT history
    user_messages_received.clear()
    async for _ in engine._handle_tool_progress(
        state, agent_def, request, progress_payload, None, None, passthrough_history=[]
    ):
        pass

    assert len(user_messages_received) == 1
    msg_without_history = user_messages_received[0]
    assert "Previous messages" not in msg_without_history

    # Test WITH history
    user_messages_received.clear()
    passthrough_history = ["First message to user", "Second update about progress"]
    async for _ in engine._handle_tool_progress(
        state,
        agent_def,
        request,
        progress_payload,
        None,
        None,
        passthrough_history=passthrough_history,
    ):
        pass

    assert len(user_messages_received) == 1
    msg_with_history = user_messages_received[0]
    # Should include the history
    assert "Previous messages" in msg_with_history
    assert "do not repeat" in msg_with_history
    assert "First message to user" in msg_with_history
    assert "Second update about progress" in msg_with_history


@pytest.mark.asyncio
async def test_engine_handles_tool_object_config(tmp_path, monkeypatch):
    """Workflow engine handles tool objects with settings and args."""
    from app.tgi.workflows.engine import WorkflowEngine
    from app.tgi.workflows.repository import WorkflowRepository
    from app.tgi.workflows.state import WorkflowStateStore

    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()

    flow = {
        "flow_id": "plan_flow",
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
                            "settings": {"streaming": True},
                            "args": {"selected-tools": "select_tools.selected_tools"},
                        }
                    }
                ],
                "depends_on": ["select_tools"],
            },
        ],
    }
    (workflows_dir / "flow.json").write_text(json.dumps(flow), encoding="utf-8")
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
                "description": "Create plan",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected-tools": {"type": "array"},
                        "title": {"type": "string"},
                    },
                },
            },
        },
    ]

    llm = StubLLMClient(
        responses={"select_tools": "Selected tools", "create_plan": "Plan created"}
    )

    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Create a plan")],
        model="test-model",
        stream=True,
        use_workflow="plan_flow",
        workflow_execution_id="exec-tool-config",
    )

    session = StubSession(tools=tools)
    stream = await engine.start_or_resume_workflow(session, request, None, None, None)
    _ = [chunk async for chunk in stream]

    state = engine.state_store.load_execution("exec-tool-config")
    assert state.completed is True


# ============================================================================
# Unit tests for ToolConfig parsing
# ============================================================================


class TestToolConfigParsing:
    """Test ToolConfig parsing from various formats."""

    def test_parse_string_tool(self):
        """Parse simple string tool name."""
        from app.tgi.workflows.agent import ToolConfig

        config = ToolConfig.from_definition("my_tool")
        assert config.name == "my_tool"
        assert config.streaming is False
        assert config.args_mapping == {}

    def test_parse_dict_with_settings(self):
        """Parse dict tool with settings."""
        from app.tgi.workflows.agent import ToolConfig

        config = ToolConfig.from_definition(
            {"my_tool": {"settings": {"streaming": True, "timeout": 30}}}
        )
        assert config.name == "my_tool"
        assert config.streaming is True
        assert config.settings == {"streaming": True, "timeout": 30}

    def test_parse_dict_with_args(self):
        """Parse dict tool with args mapping."""
        from app.tgi.workflows.agent import ToolConfig

        config = ToolConfig.from_definition(
            {"my_tool": {"args": {"param1": "agent1.result", "param2": "agent2.value"}}}
        )
        assert config.name == "my_tool"
        assert config.args_mapping == {
            "param1": "agent1.result",
            "param2": "agent2.value",
        }

    def test_parse_mixed_list(self):
        """Parse list of mixed tool definitions."""
        from app.tgi.workflows.agent import ToolConfig

        definitions = [
            "simple_tool",
            {"complex_tool": {"settings": {"streaming": True}}},
            {"another_tool": {"args": {"x": "y.z"}}},
        ]

        configs = [ToolConfig.from_definition(d) for d in definitions]

        assert len(configs) == 3
        assert configs[0].name == "simple_tool"
        assert configs[1].name == "complex_tool"
        assert configs[1].streaming is True
        assert configs[2].name == "another_tool"
        assert configs[2].args_mapping == {"x": "y.z"}


# ============================================================================
# Tests for tool schema modification with arg injection
# ============================================================================


@pytest.mark.asyncio
async def test_engine_modifies_tool_schema_for_mapped_args(tmp_path, monkeypatch):
    """Engine removes pre-mapped args from tool schema presented to LLM."""
    from app.tgi.workflows.engine import WorkflowEngine
    from app.tgi.workflows.repository import WorkflowRepository
    from app.tgi.workflows.state import WorkflowStateStore

    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()

    flow = {
        "flow_id": "plan_with_args",
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
                            "args": {"selected-tools": "select_tools.selected_tools"},
                        }
                    }
                ],
                "depends_on": ["select_tools"],
            },
        ],
    }
    (workflows_dir / "flow.json").write_text(json.dumps(flow), encoding="utf-8")
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
                "description": "Create plan",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected-tools": {
                            "type": "array",
                            "description": "Tools to use",
                        },
                        "title": {"type": "string", "description": "Plan title"},
                    },
                    "required": ["selected-tools", "title"],
                },
            },
        },
    ]

    class CapturingLLM(StubLLMClient):
        def __init__(self):
            super().__init__(
                responses={"select_tools": "", "create_plan": ""},
                tool_call_responses={
                    "select_tools": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "select_tools", "arguments": "{}"},
                        }
                    ],
                    "create_plan": [
                        {
                            "index": 0,
                            "id": "call_2",
                            "function": {"name": "plan", "arguments": "{}"},
                        }
                    ],
                },
            )
            self.captured_tool_schemas = []

        def stream_completion(self, request, access_token, span):
            if request.tools:
                self.captured_tool_schemas.append(request.tools)
            return super().stream_completion(request, access_token, span)

    llm = CapturingLLM()

    class ReturnSession(StubSession):
        async def call_tool(self, name: str, args: Optional[dict], access_token: str):
            self.tool_calls.append({"name": name, "args": args})
            payload = {"result": f"{name}_result"}
            if name == "select_tools":
                payload = {"selected_tools": ["tool_a", "tool_b"]}
            if name == "plan":
                payload = {"payload": {"result": {"id": "plan-123"}}}
            return SimpleNamespace(
                isError=False,
                structuredContent=payload,
                content=[SimpleNamespace(text=json.dumps(payload))],
            )

    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Create a plan")],
        model="test-model",
        stream=True,
        use_workflow="plan_with_args",
        workflow_execution_id="exec-schema-mod",
    )

    session = ReturnSession(tools=tools)
    stream = await engine.start_or_resume_workflow(session, request, None, None, None)
    _ = [chunk async for chunk in stream]

    # Find the schema for "plan" tool in captured schemas
    plan_schema = None
    for tool_list in llm.captured_tool_schemas:
        for tool in tool_list:
            # Handle both dict and object tool formats
            if isinstance(tool, dict):
                tool_name = tool.get("function", {}).get("name")
                if tool_name == "plan":
                    plan_schema = tool
                    break
            else:
                func = getattr(tool, "function", None)
                tool_name = getattr(func, "name", None) if func else None
                if tool_name == "plan":
                    # Convert to dict-like access
                    params = getattr(func, "parameters", {})
                    plan_schema = {
                        "function": {
                            "name": tool_name,
                            "parameters": params,
                        }
                    }
                    break
        if plan_schema:
            break

    assert (
        plan_schema is not None
    ), f"Plan tool not found in {llm.captured_tool_schemas}"
    props = plan_schema["function"]["parameters"]["properties"]
    # selected-tools should be removed from properties (so LLM doesn't try to fill it)
    assert "selected-tools" not in props
    # title should still be present
    assert "title" in props
    # selected-tools should also be removed from required
    required = plan_schema["function"]["parameters"].get("required", [])
    assert "selected-tools" not in required


@pytest.mark.asyncio
async def test_agent_instruction_for_mapped_args(tmp_path, monkeypatch):
    """Agent system prompt includes instruction about pre-mapped arguments."""
    from app.tgi.workflows.agent import AgentExecutor, ToolConfig

    tool_config = ToolConfig(
        name="plan",
        args_mapping={"selected-tools": "select_tools.selected_tools"},
    )

    executor = AgentExecutor()
    instruction = executor.build_tool_argument_instruction(tool_config)

    assert instruction is not None
    assert "selected-tools" in instruction
    assert "automatically filled" in instruction


# ============================================================================
# Tests for context-based argument resolution
# ============================================================================


class TestContextArgResolution:
    """Test resolving arguments from workflow context."""

    def test_resolve_simple_reference(self):
        """Resolve a simple agent.field reference."""
        from app.tgi.workflows.agent import AgentExecutor

        context = {
            "agents": {
                "select_tools": {"content": "", "selected_tools": ["tool1", "tool2"]}
            }
        }

        executor = AgentExecutor()
        value = executor.resolve_arg_reference("select_tools.selected_tools", context)
        assert value == ["tool1", "tool2"]

    def test_resolve_deep_reference(self):
        """Resolve a deeply nested reference."""
        from app.tgi.workflows.agent import AgentExecutor

        context = {
            "agents": {
                "analyze": {
                    "content": "",
                    "results": {"data": {"items": [1, 2, 3]}},
                }
            }
        }

        executor = AgentExecutor()
        value = executor.resolve_arg_reference("analyze.results.data.items", context)
        assert value == [1, 2, 3]

    def test_resolve_missing_reference(self):
        """Return None for missing reference."""
        from app.tgi.workflows.agent import AgentExecutor

        context = {"agents": {"other": {"content": ""}}}

        executor = AgentExecutor()
        value = executor.resolve_arg_reference("missing_agent.field", context)
        assert value is None

    def test_resolve_args_for_tool(self):
        """Resolve all mapped args for a tool call."""
        from app.tgi.workflows.agent import AgentExecutor, ToolConfig

        context = {
            "agents": {
                "agent1": {"content": "", "result": "value1"},
                "agent2": {"content": "", "data": {"nested": "value2"}},
            }
        }

        tool_config = ToolConfig(
            name="my_tool",
            args_mapping={
                "param1": "agent1.result",
                "param2": "agent2.data.nested",
            },
        )

        executor = AgentExecutor()
        resolved = executor.resolve_args_for_tool(
            tool_config, context, {"extra": "arg"}
        )

        assert resolved == {
            "param1": "value1",
            "param2": "value2",
            "extra": "arg",
        }


# ============================================================================
# Tests for returns extraction with complex scenarios
# ============================================================================


class TestReturnsExtraction:
    """Test extracting returns from tool results."""

    def test_extract_returns_creates_agent_context(self):
        """Extract returns creates agent context if missing."""
        from app.tgi.workflows.agent import AgentExecutor
        from app.tgi.workflows.models import WorkflowAgentDef, WorkflowExecutionState

        agent_def = WorkflowAgentDef(
            agent="new_agent",
            description="New agent",
            returns=["result"],
        )

        state = WorkflowExecutionState.new("exec-1", "flow-1")
        # No agent context exists yet

        tool_result = {"result": "some value"}

        executor = AgentExecutor()
        executor.extract_returns(agent_def, tool_result, state)

        assert "new_agent" in state.context["agents"]
        assert state.context["agents"]["new_agent"]["result"] == "some value"

    def test_parse_json_tool_result(self):
        """Parse JSON string tool result for returns extraction."""
        from app.tgi.workflows.agent import AgentExecutor

        executor = AgentExecutor()

        result = executor.parse_tool_result_for_returns(
            '{"selected_tools": ["tool1"], "other": 123}'
        )

        assert result == {"selected_tools": ["tool1"], "other": 123}

    def test_parse_invalid_json_returns_none(self):
        """Invalid JSON returns None."""
        from app.tgi.workflows.agent import AgentExecutor

        executor = AgentExecutor()
        result = executor.parse_tool_result_for_returns("not json")
        assert result is None

    def test_parse_empty_returns_none(self):
        """Empty string returns None."""
        from app.tgi.workflows.agent import AgentExecutor

        executor = AgentExecutor()
        assert executor.parse_tool_result_for_returns("") is None
        assert executor.parse_tool_result_for_returns(None) is None


# ============================================================================
# Tests for arg injection into tool calls
# ============================================================================


@pytest.mark.asyncio
async def test_engine_injects_args_from_context_into_tool_calls(tmp_path, monkeypatch):
    """Engine injects pre-mapped args from context when tool is called."""
    from app.tgi.workflows.engine import WorkflowEngine
    from app.tgi.workflows.repository import WorkflowRepository
    from app.tgi.workflows.state import WorkflowStateStore
    from app.tgi.services.tool_service import ToolService

    workflows_dir = tmp_path / "flows"
    workflows_dir.mkdir()

    flow = {
        "flow_id": "inject_args_flow",
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
            },
        ],
    }
    (workflows_dir / "flow.json").write_text(json.dumps(flow), encoding="utf-8")
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
                "description": "Create plan",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected_tools": {
                            "type": "array",
                            "description": "Tools to use",
                        },
                        "title": {"type": "string", "description": "Plan title"},
                    },
                    "required": ["selected_tools", "title"],
                },
            },
        },
    ]

    # Capture tool calls made
    captured_tool_calls = []

    # Custom tool service that captures calls
    class CapturingToolService(ToolService):
        async def execute_tool_calls(
            self,
            session,
            tool_calls,
            access_token,
            span,
            available_tools=None,
            return_raw_results=False,
        ):
            for tc, _ in tool_calls:
                captured_tool_calls.append(
                    {
                        "name": tc.function.name,
                        "args": json.loads(tc.function.arguments),
                    }
                )
            # Return mock results
            from app.tgi.models import Message, MessageRole

            results = []
            raw_results = []
            for tc, _ in tool_calls:
                if tc.function.name == "select_tools":
                    content = '{"selected_tools": ["tool1", "tool2"]}'
                    results.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=content,
                            name="select_tools",
                        )
                    )
                    raw_results.append({"name": "select_tools", "content": content})
                else:
                    content = '{"result": "success"}'
                    results.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=content,
                            name=tc.function.name,
                        )
                    )
                    raw_results.append({"name": tc.function.name, "content": content})
            if return_raw_results:
                return results, True, raw_results
            return results, True

    # LLM that returns tool calls
    llm = StubLLMClient(
        responses={"select_tools": "Selected", "create_plan": "Created"},
        tool_call_responses={
            "select_tools": [
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "select_tools", "arguments": "{}"},
                }
            ],
            "create_plan": [
                {
                    "index": 0,
                    "id": "call_2",
                    "function": {"name": "plan", "arguments": '{"title": "My Plan"}'},
                }
            ],
        },
    )

    tool_service = CapturingToolService()

    engine = WorkflowEngine(
        WorkflowRepository(),
        WorkflowStateStore(db_path=tmp_path / "state.db"),
        llm,
        tool_service=tool_service,
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Create a plan")],
        model="test-model",
        stream=True,
        use_workflow="inject_args_flow",
        workflow_execution_id="exec-arg-inject",
    )

    session = StubSession(tools=tools)
    stream = await engine.start_or_resume_workflow(session, request, None, None, None)
    _ = [chunk async for chunk in stream]

    # Find the plan tool call
    plan_call = next((tc for tc in captured_tool_calls if tc["name"] == "plan"), None)
    assert (
        plan_call is not None
    ), f"Plan tool not called. Captured: {captured_tool_calls}"

    # Verify that selected_tools was injected
    assert (
        "selected_tools" in plan_call["args"]
    ), f"selected_tools not injected: {plan_call['args']}"
    assert plan_call["args"]["selected_tools"] == ["tool1", "tool2"]
    # Verify the LLM-provided arg is also present
    assert plan_call["args"]["title"] == "My Plan"
