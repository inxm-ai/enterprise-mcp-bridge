import json
from typing import Any, AsyncGenerator

import pytest

from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.workflows.feedback import (
    FeedbackService,
    _infer_primitive_schema,
    _flatten_schema_properties,
    build_feedback_payload,
    flatten_object_to_input_fields,
    resolve_choice_input_fields,
)
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


# ---------------------------------------------------------------------------
# Pure-function unit tests for context-driven form generation
# ---------------------------------------------------------------------------


class TestInferPrimitiveSchema:
    def test_string(self):
        assert _infer_primitive_schema("hello") == {"type": "string"}

    def test_bool(self):
        assert _infer_primitive_schema(True) == {"type": "boolean"}

    def test_int(self):
        assert _infer_primitive_schema(42) == {"type": "integer"}

    def test_float(self):
        assert _infer_primitive_schema(3.14) == {"type": "number"}

    def test_none(self):
        assert _infer_primitive_schema(None) == {"type": "null"}

    def test_list_falls_back_to_string(self):
        assert _infer_primitive_schema([1, 2]) == {"type": "string"}

    def test_bool_before_int(self):
        # bool is a subclass of int; must be detected as boolean
        assert _infer_primitive_schema(False) == {"type": "boolean"}


class TestFlattenObjectToInputFields:
    def test_flat_dict(self):
        obj = {"host": "localhost", "port": 5432}
        result = flatten_object_to_input_fields(obj)
        assert set(result.keys()) == {"host", "port"}
        assert result["host"]["type"] == "string"
        assert result["port"]["type"] == "integer"
        assert result["host"]["_nested_path"] == ["host"]
        assert result["port"]["_nested_path"] == ["port"]

    def test_nested_dict_flattened(self):
        obj = {"db": {"host": "localhost", "port": 5432}}
        result = flatten_object_to_input_fields(obj)
        assert set(result.keys()) == {"db.host", "db.port"}
        assert result["db.host"]["_nested_path"] == ["db", "host"]
        assert result["db.port"]["_nested_path"] == ["db", "port"]

    def test_deeply_nested(self):
        obj = {"a": {"b": {"c": True}}}
        result = flatten_object_to_input_fields(obj)
        assert "a.b.c" in result
        assert result["a.b.c"]["type"] == "boolean"
        assert result["a.b.c"]["_nested_path"] == ["a", "b", "c"]

    def test_scalar_input_uses_value_key(self):
        result = flatten_object_to_input_fields("hello")
        assert "value" in result
        assert result["value"]["type"] == "string"
        assert result["value"]["_nested_path"] == ["value"]

    def test_scalar_with_prefix(self):
        result = flatten_object_to_input_fields("hello", prefix="my.key")
        assert "my.key" in result
        assert result["my.key"]["_nested_path"] == ["my", "key"]

    def test_empty_dict(self):
        result = flatten_object_to_input_fields({})
        assert result == {}

    def test_mixed_depth(self):
        obj = {"name": "Alice", "address": {"city": "Berlin"}}
        result = flatten_object_to_input_fields(obj)
        assert "name" in result
        assert "address.city" in result
        assert result["name"]["type"] == "string"
        assert result["address.city"]["type"] == "string"


class TestFlattenSchemaProperties:
    def test_flat_properties(self):
        props = {"host": {"type": "string"}, "port": {"type": "integer"}}
        result = _flatten_schema_properties(props)
        assert set(result.keys()) == {"host", "port"}
        assert result["host"]["_nested_path"] == ["host"]

    def test_nested_object_property(self):
        props = {
            "db": {
                "type": "object",
                "properties": {
                    "host": {"type": "string"},
                    "port": {"type": "integer"},
                },
            }
        }
        result = _flatten_schema_properties(props)
        assert set(result.keys()) == {"db.host", "db.port"}
        assert result["db.host"]["_nested_path"] == ["db", "host"]

    def test_missing_type_defaults_to_string(self):
        props = {"name": {}}
        result = _flatten_schema_properties(props)
        assert result["name"]["type"] == "string"


class TestResolveChoiceInputFields:
    def test_static_input_flat(self):
        choice = {
            "id": "adjust",
            "input": {"feedback": {"type": "string", "description": "What to change?"}},
        }
        schemas, paths = resolve_choice_input_fields(choice, {}, {})
        assert "feedback" in schemas
        assert schemas["feedback"]["type"] == "string"
        assert "feedback" not in paths  # not dotted, no nested_path

    def test_static_input_nested_object_flattened(self):
        choice = {
            "id": "configure",
            "input": {
                "config": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer"},
                    },
                }
            },
        }
        schemas, paths = resolve_choice_input_fields(choice, {}, {})
        assert set(schemas.keys()) == {"config.host", "config.port"}
        assert paths["config.host"] == ["config", "host"]
        assert paths["config.port"] == ["config", "port"]

    def test_input_from_resolves_context(self):
        choice = {"id": "adjust", "input_from": "planner.config"}
        agent_context = {"planner": {"config": {"host": "localhost", "port": 5432}}}
        schemas, paths = resolve_choice_input_fields(choice, agent_context, {})
        assert set(schemas.keys()) == {"host", "port"}
        assert schemas["host"]["type"] == "string"
        assert schemas["port"]["type"] == "integer"
        assert "_nested_path" not in schemas["host"]
        assert paths["host"] == ["host"]
        assert paths["port"] == ["port"]

    def test_input_from_nested_object(self):
        choice = {"id": "adjust", "input_from": "agent.settings"}
        agent_context = {
            "agent": {"settings": {"db": {"host": "localhost", "port": 5432}}}
        }
        schemas, paths = resolve_choice_input_fields(choice, agent_context, {})
        assert "db.host" in schemas
        assert "db.port" in schemas
        assert paths["db.host"] == ["db", "host"]

    def test_input_from_falls_back_to_shared_context(self):
        choice = {"id": "adjust", "input_from": "global_cfg"}
        shared_context = {"global_cfg": {"retries": 3}}
        schemas, paths = resolve_choice_input_fields(choice, {}, shared_context)
        assert "retries" in schemas
        assert schemas["retries"]["type"] == "integer"

    def test_input_from_missing_returns_empty(self):
        choice = {"id": "adjust", "input_from": "nonexistent.path"}
        schemas, paths = resolve_choice_input_fields(choice, {}, {})
        assert schemas == {}
        assert paths == {}

    def test_no_input_returns_empty(self):
        choice = {"id": "proceed", "to": "next_agent"}
        schemas, paths = resolve_choice_input_fields(choice, {}, {})
        assert schemas == {}
        assert paths == {}


class TestBuildFeedbackPayloadContextDrivenInput:
    def test_input_from_fields_in_schema(self):
        choices = [
            {
                "id": "proceed",
                "to": "next_agent",
            },
            {
                "id": "configure",
                "to": "configure_agent",
                "input_from": "planner.settings",
            },
        ]
        agent_context = {"planner": {"settings": {"timeout": 30, "retries": 3}}}
        payload = build_feedback_payload("Configure?", choices, agent_context, {})

        props = payload["requestedSchema"]["properties"]
        assert "timeout" in props
        assert "retries" in props
        assert props["timeout"]["type"] == "integer"
        assert props["retries"]["type"] == "integer"

        meta = payload["meta"]
        assert "input_fields" in meta
        assert meta["input_fields"]["timeout"]["for_selection"] == "configure"
        assert meta["input_fields"]["timeout"]["nested_path"] == ["timeout"]
        assert meta["input_fields"]["retries"]["nested_path"] == ["retries"]

    def test_nested_input_from_flattened_with_path_meta(self):
        choices = [
            {
                "id": "setup",
                "to": "setup_agent",
                "input_from": "builder.db_config",
            },
        ]
        agent_context = {
            "builder": {
                "db_config": {
                    "connection": {"host": "db.example.com", "port": 5432},
                    "pool_size": 10,
                }
            }
        }
        payload = build_feedback_payload("Setup DB?", choices, agent_context, {})
        props = payload["requestedSchema"]["properties"]

        assert "connection.host" in props
        assert "connection.port" in props
        assert "pool_size" in props

        meta = payload["meta"]["input_fields"]
        assert meta["connection.host"]["nested_path"] == ["connection", "host"]
        assert meta["connection.port"]["nested_path"] == ["connection", "port"]
        assert meta["pool_size"]["nested_path"] == ["pool_size"]

    def test_static_input_nested_object_flattened_in_payload(self):
        choices = [
            {
                "id": "adjust",
                "to": "adjust_agent",
                "input": {
                    "db": {
                        "type": "object",
                        "properties": {
                            "host": {"type": "string"},
                            "port": {"type": "integer"},
                        },
                    }
                },
            },
        ]
        payload = build_feedback_payload("Adjust DB?", choices, {}, {})
        props = payload["requestedSchema"]["properties"]

        assert "db.host" in props
        assert "db.port" in props
        meta = payload["meta"]["input_fields"]
        assert meta["db.host"]["nested_path"] == ["db", "host"]
        assert meta["db.port"]["nested_path"] == ["db", "port"]

    def test_static_flat_input_no_nested_path_in_meta(self):
        choices = [
            {
                "id": "adjust",
                "to": "adjust_agent",
                "input": {"feedback": {"type": "string"}},
            },
        ]
        payload = build_feedback_payload("Adjust?", choices, {}, {})
        meta = payload["meta"]["input_fields"]
        assert "feedback" in meta
        assert "nested_path" not in meta["feedback"]
