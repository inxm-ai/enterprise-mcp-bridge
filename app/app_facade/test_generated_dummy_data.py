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
    assert "structuredContent" in prop["properties"]
    assert (
        prop["properties"]["structuredContent"]["properties"]["foo"]["type"] == "string"
    )


def test_build_schema_multiple_tools(dummy_data_generator):
    tool_specs = [
        {"name": "t1", "outputSchema": {"type": "string"}},
        {"name": "t2", "outputSchema": {"type": "number"}},
    ]

    schema = dummy_data_generator._build_dummy_data_schema(tool_specs)

    assert set(schema["properties"].keys()) == {"t1", "t2"}
    assert (
        schema["properties"]["t1"]["properties"]["structuredContent"]["type"]
        == "string"
    )
    assert (
        schema["properties"]["t2"]["properties"]["structuredContent"]["type"]
        == "number"
    )


def test_convert_to_js_module(dummy_data_generator):
    data = {"tool1": {"structuredContent": {"foo": "bar"}}}
    js = dummy_data_generator._convert_to_js_module(data)

    assert "export const dummyData =" in js
    assert '"tool1":' in js
    assert '"foo": "bar"' in js


@pytest.mark.asyncio
async def test_generate_dummy_data_success(dummy_data_generator, mock_tgi_service):
    tool_specs = [{"name": "tool1", "outputSchema": {"type": "string"}}]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(
        {"tool1": {"structuredContent": "hello"}}
    )

    mock_tgi_service.llm_client.client.chat.completions.create.return_value = (
        mock_response
    )

    result = await dummy_data_generator.generate_dummy_data(
        prompt="test prompt", tool_specs=tool_specs
    )

    assert "export const dummyData =" in result
    assert '"tool1"' in result

    mock_tgi_service.llm_client._build_request_params.assert_called_once()
    call_args = mock_tgi_service.llm_client._build_request_params.call_args[0][0]

    assert (
        call_args.response_format["json_schema"]["schema"]["properties"]["tool1"][
            "properties"
        ]["structuredContent"]["type"]
        == "string"
    )


@pytest.mark.asyncio
async def test_generate_dummy_data_no_tools(dummy_data_generator):
    result = await dummy_data_generator.generate_dummy_data(
        prompt="test", tool_specs=[]
    )
    assert result == "export const dummyData = {};\n"


@pytest.mark.asyncio
async def test_generate_dummy_data_json_error(dummy_data_generator, mock_tgi_service):
    tool_specs = [{"name": "tool1", "outputSchema": {}}]
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "invalid json"
    mock_tgi_service.llm_client.client.chat.completions.create.return_value = (
        mock_response
    )

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await dummy_data_generator.generate_dummy_data(
            prompt="t", tool_specs=tool_specs
        )
    assert exc.value.status_code == 502
