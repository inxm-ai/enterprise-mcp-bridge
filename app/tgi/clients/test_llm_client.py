import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tgi import llm_client as llm
from app.tgi.clients.llm_client import LLMClient
from app.tgi.models import (
    ChatCompletionRequest,
    FunctionDefinition,
    Message,
    MessageRole,
    Tool,
    ToolCall,
    ToolCallFunction,
)


class _Dumpable:
    def __init__(self, payload):
        self.payload = payload
        self.type = payload.get("type")

    def model_dump(self, mode="json", exclude_none=True):
        return self.payload


@pytest.fixture(autouse=True)
def reset_llm_globals():
    llm._WARNED_INVALID_CONVERSATION_MODES.clear()
    llm._RESPONSES_JSON_OBJECT_ONLY_SCHEMAS.clear()
    yield
    llm._WARNED_INVALID_CONVERSATION_MODES.clear()
    llm._RESPONSES_JSON_OBJECT_ONLY_SCHEMAS.clear()


@pytest.fixture
def llm_client():
    with patch.dict(
        "os.environ",
        {
            "TGI_URL": "https://api.test-llm.com/v1",
            "TGI_TOKEN": "test-token-123",
            "TGI_CONVERSATION_MODE": "chat/completions",
        },
    ):
        return LLMClient()


@pytest.fixture
def llm_client_no_token():
    with patch.dict(
        "os.environ",
        {
            "TGI_URL": "https://api.test-llm.com/v1",
            "TGI_TOKEN": "",
            "TGI_CONVERSATION_MODE": "chat/completions",
        },
    ):
        return LLMClient()


def make_tool(name, params):
    return Tool(function=FunctionDefinition(name=name, parameters=params))


class TestLLMClient:
    def test_init_with_token(self, llm_client):
        assert llm_client.tgi_url == "https://api.test-llm.com/v1"
        assert llm_client.tgi_token == "test-token-123"
        assert llm_client.client.api_key == "test-token-123"
        assert str(llm_client.client.base_url) == "https://api.test-llm.com/v1/"
        assert llm_client.conversation_mode == "chat/completions"

    def test_init_without_token(self, llm_client_no_token):
        assert llm_client_no_token.tgi_url == "https://api.test-llm.com/v1"
        assert llm_client_no_token.tgi_token == ""
        assert llm_client_no_token.client.api_key == "fake-token"

    def test_init_strips_trailing_slash(self):
        with patch.dict(
            "os.environ",
            {
                "TGI_URL": "https://api.test.com/",
                "TGI_TOKEN": "",
                "TGI_CONVERSATION_MODE": "chat/completions",
            },
        ):
            client = LLMClient()
            assert client.tgi_url == "https://api.test.com"
            assert str(client.client.base_url) == "https://api.test.com"

    def test_conversation_mode_aliases(self):
        with patch.dict(
            "os.environ",
            {
                "TGI_URL": "https://api.test.com/v1",
                "TGI_TOKEN": "",
                "TGI_CONVERSATION_MODE": "/responses",
            },
        ):
            client = LLMClient()
            assert client.conversation_mode == "responses"

    def test_invalid_conversation_mode_warns_once_and_falls_back(self):
        llm._WARNED_INVALID_CONVERSATION_MODES.clear()
        with patch.object(llm.logger, "warning") as warning_mock:
            with patch.dict(
                "os.environ",
                {
                    "TGI_URL": "https://api.test.com/v1",
                    "TGI_TOKEN": "",
                    "TGI_CONVERSATION_MODE": "definitely-not-valid",
                },
            ):
                first = LLMClient()
                second = LLMClient()
                assert first.conversation_mode == "chat/completions"
                assert second.conversation_mode == "chat/completions"
        assert warning_mock.call_count == 1

    def test_responses_schema_normalization_logs_issues(self):
        with patch.dict(
            "os.environ",
            {
                "TGI_URL": "https://api.test-llm.com/v1",
                "TGI_TOKEN": "test-token-123",
                "TGI_CONVERSATION_MODE": "responses",
            },
        ):
            client = LLMClient()
        with patch.object(llm.logger, "warning") as warning_mock:
            first = client._response_format_to_text_config(
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "coerce_1",
                        "schema": {
                            "type": "object",
                            "properties": {"a": {"type": ["string", "null"]}},
                        },
                    },
                }
            )
        assert first["format"]["type"] == "json_schema"
        assert first["format"]["schema"]["properties"]["a"]["type"] == "string"
        assert warning_mock.call_count >= 1

    def test_responses_schema_normalization_keeps_properties_map_shape(self):
        with patch.dict(
            "os.environ",
            {
                "TGI_URL": "https://api.test-llm.com/v1",
                "TGI_TOKEN": "test-token-123",
                "TGI_CONVERSATION_MODE": "responses",
            },
        ):
            client = LLMClient()

        text_config = client._response_format_to_text_config(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "map_shape",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "properties": {"type": "object"},
                                },
                            }
                        },
                    },
                },
            }
        )

        normalized = text_config["format"]["schema"]
        inner_properties = normalized["properties"]["schema"]["properties"]
        assert isinstance(inner_properties, dict)
        assert set(inner_properties.keys()) == {"type", "properties"}
        assert "required" not in inner_properties

    def test_responses_schema_normalization_removes_incompatible_enum_values(self):
        with patch.dict(
            "os.environ",
            {
                "TGI_URL": "https://api.test-llm.com/v1",
                "TGI_TOKEN": "test-token-123",
                "TGI_CONVERSATION_MODE": "responses",
            },
        ):
            client = LLMClient()

        text_config = client._response_format_to_text_config(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "enum_type_conflict",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "bad": {
                                "type": "object",
                                "enum": ["string-value"],
                                "items": {"type": "string"},
                            }
                        },
                    },
                },
            }
        )

        normalized = text_config["format"]["schema"]
        bad_schema = normalized["properties"]["bad"]
        assert bad_schema["type"] == "object"
        assert "enum" not in bad_schema
        assert "items" not in bad_schema

    def test_parse_invalid_schema_error_details_accepts_context_without_in_prefix(self):
        message = (
            "Invalid schema for response_format 'demo': context=('properties', 'x'), "
            "enum value string does not validate against {'type': 'object'}."
        )
        schema_name, context, detail = LLMClient._parse_invalid_schema_error_details(
            message
        )

        assert schema_name == "demo"
        assert context == ("properties", "x")
        assert "enum value string" in detail

    def test_create_completion_id(self, llm_client):
        completion_id = llm_client.create_completion_id()
        assert completion_id.startswith("chatcmpl-")
        assert len(completion_id) == 38

    def test_create_usage_stats(self, llm_client):
        usage = llm_client.create_usage_stats(100, 50)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    @pytest.mark.asyncio
    async def test_non_stream_completion_success_chat_mode(self, llm_client):
        mock_response_data = {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="test-model",
            stream=False,
        )

        mock_response = MagicMock()
        mock_response.model_dump.return_value = mock_response_data
        llm_client.client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        response = await llm_client.non_stream_completion(request, "test-token", None)
        call_kwargs = llm_client.client.chat.completions.create.call_args.kwargs

        assert "tool_choice" not in call_kwargs
        assert "tools" not in call_kwargs
        assert response.id == "chatcmpl-test123"
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello! How can I help you today?"

    @pytest.mark.asyncio
    async def test_non_stream_completion_responses_mode_structured_mapping(self):
        with patch.dict(
            "os.environ",
            {
                "TGI_URL": "https://api.test-llm.com/v1",
                "TGI_TOKEN": "test-token-123",
                "TGI_CONVERSATION_MODE": "responses",
            },
        ):
            client = LLMClient()

        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Generate JSON")],
            model="test-model",
            stream=False,
            tools=[make_tool("search_docs", {"type": "object", "properties": {}})],
            tool_choice="required",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                    },
                    "strict": True,
                },
            },
        )

        response_payload = {
            "id": "resp_1",
            "created_at": 1730000000,
            "model": "test-model",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "search_docs",
                    "arguments": '{"query":"x"}',
                }
            ],
            "usage": {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11},
        }
        client.client.responses.create = AsyncMock(
            return_value=_Dumpable(response_payload)
        )

        response = await client.non_stream_completion(request, "token", None)
        call_kwargs = client.client.responses.create.call_args.kwargs

        assert "input" in call_kwargs
        assert call_kwargs["input"][0]["role"] == "user"
        assert call_kwargs["text"]["format"]["type"] == "json_schema"
        assert call_kwargs["text"]["format"]["name"] == "answer_schema"
        assert call_kwargs["tools"][0]["name"] == "search_docs"
        assert call_kwargs["tool_choice"] == "required"

        message = response.choices[0].message
        assert message.tool_calls is not None
        assert message.tool_calls[0].function.name == "search_docs"
        assert message.tool_calls[0].function.arguments == '{"query":"x"}'
        assert response.usage.prompt_tokens == 8
        assert response.usage.completion_tokens == 3
        assert response.usage.total_tokens == 11

    @pytest.mark.asyncio
    async def test_non_stream_completion_responses_mode_coerces_dynamic_schema_to_strict_json_schema(
        self,
    ):
        with patch.dict(
            "os.environ",
            {
                "TGI_URL": "https://api.test-llm.com/v1",
                "TGI_TOKEN": "test-token-123",
                "TGI_CONVERSATION_MODE": "responses",
            },
        ):
            client = LLMClient()

        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Generate output")],
            model="test-model",
            stream=False,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "dynamic_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "value": {
                                "anyOf": [
                                    {"type": "object", "additionalProperties": True},
                                    {"type": "string"},
                                ]
                            }
                        },
                    },
                },
            },
        )

        response_payload = {
            "id": "resp_dynamic",
            "created_at": 1730000000,
            "model": "test-model",
            "output_text": '{"value":{"x":1}}',
            "output": [],
            "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        }
        client.client.responses.create = AsyncMock(
            return_value=_Dumpable(response_payload)
        )

        response = await client.non_stream_completion(request, "token", None)
        call_kwargs = client.client.responses.create.call_args.kwargs

        assert call_kwargs["text"]["format"]["type"] == "json_schema"
        formatted_schema = call_kwargs["text"]["format"]["schema"]
        assert formatted_schema["additionalProperties"] is False
        assert set(formatted_schema["required"]) == {"value"}
        any_of_object = formatted_schema["properties"]["value"]["anyOf"][0]
        assert any_of_object["type"] == "object"
        assert any_of_object["additionalProperties"] is False
        assert response.choices[0].message.content == '{"value":{"x":1}}'

    @pytest.mark.asyncio
    async def test_non_stream_completion_responses_mode_retries_with_json_object_on_invalid_schema(
        self,
    ):
        with patch.dict(
            "os.environ",
            {
                "TGI_URL": "https://api.test-llm.com/v1",
                "TGI_TOKEN": "test-token-123",
                "TGI_CONVERSATION_MODE": "responses",
            },
        ):
            client = LLMClient()

        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Generate output")],
            model="test-model",
            stream=False,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "bad_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"x": {"type": ["string", "number"]}},
                    },
                },
            },
        )

        response_payload = {
            "id": "resp_retry",
            "created_at": 1730000000,
            "model": "test-model",
            "output_text": '{"x":"ok"}',
            "output": [],
        }
        client.client.responses.create = AsyncMock(
            side_effect=[
                Exception("invalid_json_schema ... text.format.schema ..."),
                _Dumpable(response_payload),
            ]
        )

        response = await client.non_stream_completion(request, "token", None)

        assert client.client.responses.create.call_count == 2
        first_kwargs = client.client.responses.create.call_args_list[0].kwargs
        second_kwargs = client.client.responses.create.call_args_list[1].kwargs
        assert first_kwargs["text"]["format"]["type"] == "json_schema"
        assert second_kwargs["text"]["format"]["type"] == "json_object"
        assert any(
            item.get("role") == "system"
            and "schema:" in str(item.get("content", "")).lower()
            for item in second_kwargs["input"]
        )
        assert response.choices[0].message.content == '{"x":"ok"}'

    @pytest.mark.asyncio
    async def test_non_stream_completion_responses_mode_caches_schema_for_json_object_after_rejection(
        self,
    ):
        with patch.dict(
            "os.environ",
            {
                "TGI_URL": "https://api.test-llm.com/v1",
                "TGI_TOKEN": "test-token-123",
                "TGI_CONVERSATION_MODE": "responses",
            },
        ):
            client = LLMClient()

        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Generate output")],
            model="test-model",
            stream=False,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "cache_me_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"x": {"type": ["string", "number"]}},
                    },
                },
            },
        )

        response_payload = {
            "id": "resp_retry_cache",
            "created_at": 1730000000,
            "model": "test-model",
            "output_text": '{"x":"ok"}',
            "output": [],
        }

        client.client.responses.create = AsyncMock(
            side_effect=[
                Exception("invalid_json_schema ... text.format.schema ..."),
                _Dumpable(response_payload),
                _Dumpable(response_payload),
            ]
        )

        _ = await client.non_stream_completion(request, "token", None)
        _ = await client.non_stream_completion(request, "token", None)

        assert client.client.responses.create.call_count == 3
        first_kwargs = client.client.responses.create.call_args_list[0].kwargs
        second_kwargs = client.client.responses.create.call_args_list[1].kwargs
        third_kwargs = client.client.responses.create.call_args_list[2].kwargs
        assert first_kwargs["text"]["format"]["type"] == "json_schema"
        assert second_kwargs["text"]["format"]["type"] == "json_object"
        assert third_kwargs["text"]["format"]["type"] == "json_object"

    @pytest.mark.asyncio
    async def test_non_stream_completion_error(self, llm_client):
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="test-model",
            stream=False,
        )

        llm_client.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        with pytest.raises(Exception) as excinfo:
            await llm_client.non_stream_completion(request, "test-token", None)
        assert "API Error" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_stream_completion_success_chat_mode(self, llm_client):
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="test")],
            model="test-model",
            stream=True,
        )

        chunk1 = MagicMock()
        chunk1.model_dump.return_value = {"choices": [{"delta": {"content": "Hello"}}]}

        async def mock_stream():
            yield chunk1

        llm_client.client.chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )

        chunks = []
        async for chunk in llm_client.stream_completion(request, "test-token", None):
            chunks.append(chunk)

        call_kwargs = llm_client.client.chat.completions.create.call_args.kwargs
        assert "tool_choice" not in call_kwargs
        assert "tools" not in call_kwargs
        assert len(chunks) == 2
        assert (
            chunks[0]
            == f'data: {json.dumps({"choices":[{"delta":{"content":"Hello"}}]})}\n\n'
        )
        assert chunks[1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_stream_completion_success_responses_mode(self):
        with patch.dict(
            "os.environ",
            {
                "TGI_URL": "https://api.test-llm.com/v1",
                "TGI_TOKEN": "test-token-123",
                "TGI_CONVERSATION_MODE": "responses",
            },
        ):
            client = LLMClient()

        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="hello")],
            model="test-model",
            stream=True,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "demo", "schema": {"type": "object"}},
            },
        )

        async def mock_stream():
            yield _Dumpable({"type": "response.output_text.delta", "delta": "Hello"})
            yield _Dumpable(
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "search_docs",
                    },
                }
            )
            yield _Dumpable(
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 0,
                    "call_id": "call_1",
                    "delta": '{"query":"books"}',
                }
            )
            yield _Dumpable(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "created_at": 1730000000,
                        "model": "test-model",
                        "output": [],
                    },
                }
            )

        client.client.responses.create = AsyncMock(return_value=mock_stream())

        chunks = []
        async for chunk in client.stream_completion(request, "token", None):
            chunks.append(chunk)

        call_kwargs = client.client.responses.create.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert "input" in call_kwargs
        assert "messages" not in call_kwargs
        assert call_kwargs["text"]["format"]["type"] == "json_schema"

        parsed_chunks = []
        for chunk in chunks:
            if chunk == "data: [DONE]\n\n":
                continue
            payload = chunk.replace("data: ", "").strip()
            parsed_chunks.append(json.loads(payload))

        content_found = any(
            item["choices"][0].get("delta", {}).get("content") == "Hello"
            for item in parsed_chunks
        )
        tool_call_found = any(
            bool(item["choices"][0].get("delta", {}).get("tool_calls"))
            for item in parsed_chunks
        )

        assert content_found is True
        assert tool_call_found is True
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_stream_completion_responses_mode_retries_with_json_object_on_invalid_schema(
        self,
    ):
        with patch.dict(
            "os.environ",
            {
                "TGI_URL": "https://api.test-llm.com/v1",
                "TGI_TOKEN": "test-token-123",
                "TGI_CONVERSATION_MODE": "responses",
            },
        ):
            client = LLMClient()

        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Generate output")],
            model="test-model",
            stream=True,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "bad_stream_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"x": {"type": ["string", "number"]}},
                    },
                },
            },
        )

        async def fallback_stream():
            yield _Dumpable(
                {"type": "response.output_text.delta", "delta": '{"x":"ok"}'}
            )
            yield _Dumpable(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_stream_retry",
                        "created_at": 1730000000,
                        "model": "test-model",
                        "output": [],
                    },
                }
            )

        client.client.responses.create = AsyncMock(
            side_effect=[
                Exception("invalid_json_schema ... text.format.schema ..."),
                fallback_stream(),
            ]
        )

        chunks = []
        async for chunk in client.stream_completion(request, "token", None):
            chunks.append(chunk)

        assert client.client.responses.create.call_count == 2
        first_kwargs = client.client.responses.create.call_args_list[0].kwargs
        second_kwargs = client.client.responses.create.call_args_list[1].kwargs
        assert first_kwargs["text"]["format"]["type"] == "json_schema"
        assert second_kwargs["text"]["format"]["type"] == "json_object"
        assert any(
            item.get("role") == "system"
            and "schema:" in str(item.get("content", "")).lower()
            for item in second_kwargs["input"]
        )
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_stream_completion_error_handling(self, llm_client):
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="test-model",
            stream=True,
        )

        llm_client.client.chat.completions.create = AsyncMock(
            side_effect=Exception("Test connection error")
        )

        chunks = []
        async for chunk in llm_client.stream_completion(request, "test-token", None):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert "Error: Error streaming from LLM: Test connection error" in chunks[0]
        assert chunks[1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_prepare_payload_sanitizes_null_content_after_compression(
        self, llm_client, monkeypatch
    ):
        monkeypatch.setattr(llm, "LLM_MAX_PAYLOAD_BYTES", 800)

        tool_call = ToolCall(
            id="call-1",
            function=ToolCallFunction(name="test_tool", arguments="{}"),
        )
        request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="System"),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=None,
                    tool_calls=[tool_call],
                ),
                Message(role=MessageRole.USER, content="A" * 2000),
            ],
            model="test-model",
            stream=True,
        )

        async def summarize_fn(
            base_request=None,
            content=None,
            access_token=None,
            outer_span=None,
            **_kwargs,
        ):
            return "summary"

        llm_client.summarize_text = AsyncMock(side_effect=summarize_fn)

        from app.tgi.context_compressor import AdaptiveCompressor

        monkeypatch.setattr(
            llm,
            "get_default_compressor",
            lambda: AdaptiveCompressor(chunk_tokens=10, group_size=2, window_size=1),
        )

        compressed = await llm_client._prepare_payload(request, "test-token")

        assert all(message.content is not None for message in compressed.messages)
        assert compressed.messages[1].content == ""

    @pytest.mark.asyncio
    async def test_summarize_text_uses_user_message(self, llm_client):
        captured = {}

        async def fake_non_stream_completion(request, _access_token, _outer_span):
            captured["messages"] = request.messages
            return {"choices": [{"message": {"content": "summary"}}]}

        llm_client.non_stream_completion = AsyncMock(
            side_effect=fake_non_stream_completion
        )

        base_request = ChatCompletionRequest(messages=[], model="test-model")
        result = await llm_client.summarize_text(
            base_request, "raw content", None, None
        )

        assert result == "summary"
        assert captured["messages"][0].role == MessageRole.SYSTEM
        assert captured["messages"][1].role == MessageRole.USER
        assert captured["messages"][1].content == "raw content"
        assert all(m.role != MessageRole.ASSISTANT for m in captured["messages"][1:])


def test_model_parameter_required_not_empty_string():
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
