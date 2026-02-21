import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.app_facade.generated_dummy_data import DummyDataGenerator


class MockTGIService:
    def __init__(self):
        self.llm_client = MagicMock()
        self.llm_client.client = MagicMock()
        self.llm_client.client.chat.completions.create = AsyncMock()
        self.llm_client._build_request_params = MagicMock(
            side_effect=lambda x: {
                "messages": x.messages,
                "response_format": x.response_format,
            }
        )


@pytest.fixture
def mock_tgi_service():
    return MockTGIService()


@pytest.fixture
def dummy_data_generator(mock_tgi_service):
    return DummyDataGenerator(mock_tgi_service)


def test_build_schema_simple(dummy_data_generator):
    tool_specs = [
        {
            "name": "tool1",
            "outputSchema": {
                "type": "object",
                "properties": {"foo": {"type": "string"}},
            },
        }
    ]

    schema = dummy_data_generator._build_dummy_data_schema(tool_specs)

    assert schema["type"] == "object"
    assert "tool1" in schema["properties"]
    prop = schema["properties"]["tool1"]
    assert prop["type"] == "object"
    assert prop["properties"]["foo"]["type"] == "string"


def test_build_schema_multiple_tools(dummy_data_generator):
    tool_specs = [
        {"name": "t1", "outputSchema": {"type": "string"}},
        {"name": "t2", "outputSchema": {"type": "number"}},
    ]

    schema = dummy_data_generator._build_dummy_data_schema(tool_specs)

    assert set(schema["properties"].keys()) == {"t1", "t2"}
    assert schema["properties"]["t1"]["type"] == "string"
    assert schema["properties"]["t2"]["type"] == "number"


def test_build_schema_missing_output_schema_is_permissive(dummy_data_generator):
    tool_specs = [{"name": "unknown_tool", "outputSchema": None}]
    schema = dummy_data_generator._build_dummy_data_schema(tool_specs)
    fallback = schema["properties"]["unknown_tool"]
    assert "anyOf" in fallback
    assert {entry.get("type") for entry in fallback["anyOf"]} == {
        "array",
        "boolean",
        "null",
        "number",
        "object",
        "string",
    }


def test_convert_to_js_module(dummy_data_generator):
    data = {"tool1": {"foo": "bar"}}
    hints = {
        "tool1": {
            "schema_status": "missing_output_schema",
            "next_action": "ask_for_schema_then_regenerate_dummy_data",
            "fallback_mode": "llm_generated_any_schema",
            "tool": "tool1",
        }
    }
    js = dummy_data_generator._convert_to_js_module(data, hints)

    assert "export const dummyData =" in js
    assert "export const dummyDataSchemaHints =" in js
    assert '"tool1":' in js
    assert '"foo": "bar"' in js
    assert '"missing_output_schema"' in js


@pytest.mark.asyncio
async def test_generate_dummy_data_success(dummy_data_generator, mock_tgi_service):
    tool_specs = [{"name": "tool1", "outputSchema": {"type": "string"}}]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({"tool1": "hello"})

    mock_tgi_service.llm_client.client.chat.completions.create.return_value = (
        mock_response
    )

    result = await dummy_data_generator.generate_dummy_data(
        prompt="test prompt", tool_specs=tool_specs
    )

    assert "export const dummyData =" in result
    assert "export const dummyDataSchemaHints =" in result
    assert '"tool1"' in result
    assert '"hello"' in result

    mock_tgi_service.llm_client._build_request_params.assert_called_once()
    call_args = mock_tgi_service.llm_client._build_request_params.call_args[0][0]

    assert (
        call_args.response_format["json_schema"]["schema"]["properties"]["tool1"][
            "type"
        ]
        == "string"
    )


@pytest.mark.asyncio
async def test_generate_dummy_data_no_tools(dummy_data_generator):
    result = await dummy_data_generator.generate_dummy_data(
        prompt="test", tool_specs=[]
    )
    assert "export const dummyData = {};" in result
    assert "export const dummyDataSchemaHints = {};" in result


@pytest.mark.asyncio
async def test_generate_dummy_data_json_error(dummy_data_generator, mock_tgi_service):
    tool_specs = [{"name": "tool1", "outputSchema": {"type": "string"}}]
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "invalid json"
    mock_tgi_service.llm_client.client.chat.completions.create.return_value = (
        mock_response
    )

    result = await dummy_data_generator.generate_dummy_data(
        prompt="t", tool_specs=tool_specs
    )
    assert "export const dummyData = {};" in result
    assert "export const dummyDataSchemaHints = {};" in result


@pytest.mark.asyncio
async def test_generate_dummy_data_derives_schema_and_uses_observed_sample(
    dummy_data_generator, mock_tgi_service
):
    tool_specs = [
        {
            "name": "weather",
            "outputSchema": None,
            "sampleStructuredContent": {"city": "Berlin", "temperature_c": 8.2},
        }
    ]

    schema_response = MagicMock()
    schema_response.choices = [MagicMock()]
    schema_response.choices[0].message.content = json.dumps(
        {
            "schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "temperature_c": {"type": "number"},
                },
                "required": ["city", "temperature_c"],
                "additionalProperties": True,
            }
        }
    )

    dummy_response = MagicMock()
    dummy_response.choices = [MagicMock()]
    dummy_response.choices[0].message.content = json.dumps(
        {"weather": {"structuredContent": {"city": "Wrong"}}}
    )

    mock_tgi_service.llm_client.client.chat.completions.create.side_effect = [
        schema_response,
        dummy_response,
    ]

    result = await dummy_data_generator.generate_dummy_data(
        prompt="Weather UI", tool_specs=tool_specs
    )

    assert '"city": "Berlin"' in result
    assert '"temperature_c": 8.2' in result

    generation_request = (
        mock_tgi_service.llm_client._build_request_params.call_args_list[1][0][0]
    )
    schema = generation_request.response_format["json_schema"]["schema"]
    assert schema["properties"]["weather"]["properties"]["city"]["type"] == "string"


@pytest.mark.asyncio
async def test_generate_dummy_data_normalizes_legacy_structured_content_payload(
    dummy_data_generator, mock_tgi_service
):
    tool_specs = [
        {
            "name": "weather",
            "outputSchema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(
        {"weather": {"structuredContent": {"city": "Berlin"}}}
    )
    mock_tgi_service.llm_client.client.chat.completions.create.return_value = (
        mock_response
    )

    result = await dummy_data_generator.generate_dummy_data(
        prompt="Weather UI", tool_specs=tool_specs
    )

    assert '"weather": {' in result
    assert '"city": "Berlin"' in result
    assert '"structuredContent"' not in result


@pytest.mark.asyncio
async def test_generate_dummy_data_adds_schema_hints_for_missing_output_schema(
    dummy_data_generator, mock_tgi_service
):
    tool_specs = [{"name": "tool_without_schema", "outputSchema": None}]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(
        {"tool_without_schema": {"value": 1}}
    )
    mock_tgi_service.llm_client.client.chat.completions.create.return_value = (
        mock_response
    )

    result = await dummy_data_generator.generate_dummy_data(
        prompt="UI", tool_specs=tool_specs
    )

    assert '"tool_without_schema": {' in result
    assert '"value": 1' in result
    assert "dummyDataSchemaHints" in result
    assert '"schema_status": "missing_output_schema"' in result
    assert '"next_action": "ask_for_schema_then_regenerate_dummy_data"' in result
