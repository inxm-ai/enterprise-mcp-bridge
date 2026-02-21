import json
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from app.app_facade.generated_phase1 import run_phase1_attempt
from app.tgi.models import Message, MessageRole


@pytest.mark.asyncio
async def test_phase1_recovers_components_script_from_template_parts():
    class MockChunk:
        def __init__(self, content=None, is_done=False):
            self.content = content
            self.is_done = is_done
            self.tool_calls = None
            self.accumulated_tool_calls = None
            self.finish_reason = None

    async def mock_stream(*_args, **_kwargs):
        payload = {
            "template_parts": {"script": "console.log('from-template');"},
            "test_script": "import { describe, it } from 'node:test';",
        }
        yield MockChunk(content=json.dumps(payload), is_done=False)
        yield MockChunk(content=None, is_done=True)

    tgi_service = MagicMock()
    tgi_service.llm_client = MagicMock()
    tgi_service.llm_client.stream_completion = mock_stream

    run_tests_calls = []

    def run_tests(service_script, components_script, test_script, dummy_data):
        run_tests_calls.append(
            {
                "service_script": service_script,
                "components_script": components_script,
                "test_script": test_script,
                "dummy_data": dummy_data,
            }
        )
        return True, "ok"

    async def iterative_test_fix(**_kwargs):
        return False, "", "", "", None, []

    @asynccontextmanager
    async def mock_chunk_reader(stream_source):
        class Reader:
            def as_parsed(self):
                return stream_source

        yield Reader()

    messages = [
        Message(role=MessageRole.SYSTEM, content="sys"),
        Message(role=MessageRole.USER, content="user"),
    ]

    result_payload = None
    async for item in run_phase1_attempt(
        attempt=1,
        max_attempts=3,
        messages=messages,
        allowed_tools=[],
        dummy_data=None,
        access_token=None,
        tgi_service=tgi_service,
        parse_json=lambda content: json.loads(content),
        run_tests=run_tests,
        iterative_test_fix=iterative_test_fix,
        chunk_reader=mock_chunk_reader,
        ui_model_headers={},
    ):
        if isinstance(item, dict) and item.get("type") == "result":
            result_payload = item

    assert result_payload is not None
    assert result_payload["success"] is True
    assert (
        result_payload["payload"]["components_script"]
        == "console.log('from-template');"
    )
    assert len(run_tests_calls) == 1
    assert run_tests_calls[0]["components_script"] == "console.log('from-template');"


@pytest.mark.asyncio
async def test_phase1_rejects_timeout_risk_test_patterns():
    class MockChunk:
        def __init__(self, content=None, is_done=False):
            self.content = content
            self.is_done = is_done
            self.tool_calls = None
            self.accumulated_tool_calls = None
            self.finish_reason = None

    async def mock_stream(*_args, **_kwargs):
        payload = {
            "components_script": "console.log('ok');",
            "test_script": (
                "import { it } from 'node:test';\n"
                "it('hangs', async () => { await new Promise(() => {}); });"
            ),
        }
        yield MockChunk(content=json.dumps(payload), is_done=False)
        yield MockChunk(content=None, is_done=True)

    tgi_service = MagicMock()
    tgi_service.llm_client = MagicMock()
    tgi_service.llm_client.stream_completion = mock_stream

    run_tests_called = False

    def run_tests(_service_script, _components_script, _test_script, _dummy_data):
        nonlocal run_tests_called
        run_tests_called = True
        return True, "ok"

    async def iterative_test_fix(**_kwargs):
        return False, "", "", "", None, []

    @asynccontextmanager
    async def mock_chunk_reader(stream_source):
        class Reader:
            def as_parsed(self):
                return stream_source

        yield Reader()

    messages = [
        Message(role=MessageRole.SYSTEM, content="sys"),
        Message(role=MessageRole.USER, content="user"),
    ]

    result_payload = None
    async for item in run_phase1_attempt(
        attempt=1,
        max_attempts=3,
        messages=messages,
        allowed_tools=[],
        dummy_data=None,
        access_token=None,
        tgi_service=tgi_service,
        parse_json=lambda content: json.loads(content),
        run_tests=run_tests,
        iterative_test_fix=iterative_test_fix,
        chunk_reader=mock_chunk_reader,
        ui_model_headers={},
    ):
        if isinstance(item, dict) and item.get("type") == "result":
            result_payload = item

    assert result_payload is not None
    assert result_payload["success"] is False
    assert "timeout_risk_patterns_detected" in result_payload["reason"]
    assert run_tests_called is False


@pytest.mark.asyncio
async def test_phase1_rejects_component_state_signature_risk():
    class MockChunk:
        def __init__(self, content=None, is_done=False):
            self.content = content
            self.is_done = is_done
            self.tool_calls = None
            self.accumulated_tool_calls = None
            self.finish_reason = None

    async def mock_stream(*_args, **_kwargs):
        payload = {
            "components_script": "const renderCard = ({ state }) => state.loading ? 'x' : 'y';",
            "test_script": "import { it } from 'node:test'; it('ok', () => {});",
        }
        yield MockChunk(content=json.dumps(payload), is_done=False)
        yield MockChunk(content=None, is_done=True)

    tgi_service = MagicMock()
    tgi_service.llm_client = MagicMock()
    tgi_service.llm_client.stream_completion = mock_stream

    run_tests_called = False

    def run_tests(_service_script, _components_script, _test_script, _dummy_data):
        nonlocal run_tests_called
        run_tests_called = True
        return True, "ok"

    async def iterative_test_fix(**_kwargs):
        return False, "", "", "", None, []

    @asynccontextmanager
    async def mock_chunk_reader(stream_source):
        class Reader:
            def as_parsed(self):
                return stream_source

        yield Reader()

    messages = [
        Message(role=MessageRole.SYSTEM, content="sys"),
        Message(role=MessageRole.USER, content="user"),
    ]

    result_payload = None
    async for item in run_phase1_attempt(
        attempt=1,
        max_attempts=3,
        messages=messages,
        allowed_tools=[],
        dummy_data=None,
        access_token=None,
        tgi_service=tgi_service,
        parse_json=lambda content: json.loads(content),
        run_tests=run_tests,
        iterative_test_fix=iterative_test_fix,
        chunk_reader=mock_chunk_reader,
        ui_model_headers={},
    ):
        if isinstance(item, dict) and item.get("type") == "result":
            result_payload = item

    assert result_payload is not None
    assert result_payload["success"] is False
    assert "quality_risk_patterns_detected" in result_payload["reason"]
    assert "destructured_state_callback_signature" in result_payload["reason"]
    assert run_tests_called is False


@pytest.mark.asyncio
async def test_phase1_rejects_live_network_test_patterns():
    class MockChunk:
        def __init__(self, content=None, is_done=False):
            self.content = content
            self.is_done = is_done
            self.tool_calls = None
            self.accumulated_tool_calls = None
            self.finish_reason = None

    async def mock_stream(*_args, **_kwargs):
        payload = {
            "components_script": "console.log('ok');",
            "test_script": (
                "import { it } from 'node:test';\n"
                "it('live', async () => {\n"
                "  const response = await fetch('/api/mcp-weather-server/tools/get_current_weather');\n"
                "  await response.json();\n"
                "});"
            ),
        }
        yield MockChunk(content=json.dumps(payload), is_done=False)
        yield MockChunk(content=None, is_done=True)

    tgi_service = MagicMock()
    tgi_service.llm_client = MagicMock()
    tgi_service.llm_client.stream_completion = mock_stream

    run_tests_called = False

    def run_tests(_service_script, _components_script, _test_script, _dummy_data):
        nonlocal run_tests_called
        run_tests_called = True
        return True, "ok"

    async def iterative_test_fix(**_kwargs):
        return False, "", "", "", None, []

    @asynccontextmanager
    async def mock_chunk_reader(stream_source):
        class Reader:
            def as_parsed(self):
                return stream_source

        yield Reader()

    messages = [
        Message(role=MessageRole.SYSTEM, content="sys"),
        Message(role=MessageRole.USER, content="user"),
    ]

    result_payload = None
    async for item in run_phase1_attempt(
        attempt=1,
        max_attempts=3,
        messages=messages,
        allowed_tools=[],
        dummy_data=None,
        access_token=None,
        tgi_service=tgi_service,
        parse_json=lambda content: json.loads(content),
        run_tests=run_tests,
        iterative_test_fix=iterative_test_fix,
        chunk_reader=mock_chunk_reader,
        ui_model_headers={},
    ):
        if isinstance(item, dict) and item.get("type") == "result":
            result_payload = item

    assert result_payload is not None
    assert result_payload["success"] is False
    assert "quality_risk_patterns_detected" in result_payload["reason"]
    assert "direct_network_without_fixtures" in result_payload["reason"]
    assert run_tests_called is False


@pytest.mark.asyncio
async def test_phase1_rejects_runtime_dummy_data_imports():
    class MockChunk:
        def __init__(self, content=None, is_done=False):
            self.content = content
            self.is_done = is_done
            self.tool_calls = None
            self.accumulated_tool_calls = None
            self.finish_reason = None

    async def mock_stream(*_args, **_kwargs):
        payload = {
            "components_script": (
                "import { dummyData } from './dummy_data.js';\n"
                "console.log(dummyData);"
            ),
            "test_script": "import { it } from 'node:test'; it('ok', () => {});",
        }
        yield MockChunk(content=json.dumps(payload), is_done=False)
        yield MockChunk(content=None, is_done=True)

    tgi_service = MagicMock()
    tgi_service.llm_client = MagicMock()
    tgi_service.llm_client.stream_completion = mock_stream

    run_tests_called = False

    def run_tests(_service_script, _components_script, _test_script, _dummy_data):
        nonlocal run_tests_called
        run_tests_called = True
        return True, "ok"

    async def iterative_test_fix(**_kwargs):
        return False, "", "", "", None, []

    @asynccontextmanager
    async def mock_chunk_reader(stream_source):
        class Reader:
            def as_parsed(self):
                return stream_source

        yield Reader()

    messages = [
        Message(role=MessageRole.SYSTEM, content="sys"),
        Message(role=MessageRole.USER, content="user"),
    ]

    result_payload = None
    async for item in run_phase1_attempt(
        attempt=1,
        max_attempts=3,
        messages=messages,
        allowed_tools=[],
        dummy_data=None,
        access_token=None,
        tgi_service=tgi_service,
        parse_json=lambda content: json.loads(content),
        run_tests=run_tests,
        iterative_test_fix=iterative_test_fix,
        chunk_reader=mock_chunk_reader,
        ui_model_headers={},
    ):
        if isinstance(item, dict) and item.get("type") == "result":
            result_payload = item

    assert result_payload is not None
    assert result_payload["success"] is False
    assert "quality_risk_patterns_detected" in result_payload["reason"]
    assert "runtime_script:dummy_data_import" in result_payload["reason"]
    assert run_tests_called is False


@pytest.mark.asyncio
async def test_phase1_allows_service_level_mocks_without_fetch_routes():
    class MockChunk:
        def __init__(self, content=None, is_done=False):
            self.content = content
            self.is_done = is_done
            self.tool_calls = None
            self.accumulated_tool_calls = None
            self.finish_reason = None

    async def mock_stream(*_args, **_kwargs):
        payload = {
            "components_script": "console.log('ok');",
            "test_script": (
                "import { it } from 'node:test';\n"
                "import './app.js';\n"
                "it('uses service mock only', async () => {\n"
                "  const svc = globalThis.service || new globalThis.McpService();\n"
                "  svc.test.addResponse('list_items', { structuredContent: { result: [] } });\n"
                "  await svc.call('list_items', {}, { resultKey: 'result' });\n"
                "});"
            ),
        }
        yield MockChunk(content=json.dumps(payload), is_done=False)
        yield MockChunk(content=None, is_done=True)

    tgi_service = MagicMock()
    tgi_service.llm_client = MagicMock()
    tgi_service.llm_client.stream_completion = mock_stream

    run_tests_called = False

    def run_tests(_service_script, _components_script, _test_script, _dummy_data):
        nonlocal run_tests_called
        run_tests_called = True
        return True, "ok"

    async def iterative_test_fix(**_kwargs):
        return False, "", "", "", None, []

    @asynccontextmanager
    async def mock_chunk_reader(stream_source):
        class Reader:
            def as_parsed(self):
                return stream_source

        yield Reader()

    messages = [
        Message(role=MessageRole.SYSTEM, content="sys"),
        Message(role=MessageRole.USER, content="user"),
    ]

    result_payload = None
    async for item in run_phase1_attempt(
        attempt=1,
        max_attempts=3,
        messages=messages,
        allowed_tools=[],
        dummy_data=None,
        access_token=None,
        tgi_service=tgi_service,
        parse_json=lambda content: json.loads(content),
        run_tests=run_tests,
        iterative_test_fix=iterative_test_fix,
        chunk_reader=mock_chunk_reader,
        ui_model_headers={},
    ):
        if isinstance(item, dict) and item.get("type") == "result":
            result_payload = item

    assert result_payload is not None
    assert result_payload["success"] is True
    assert run_tests_called is True


@pytest.mark.asyncio
async def test_phase1_rejects_weather_schema_mismatch_and_weak_assertions():
    class MockChunk:
        def __init__(self, content=None, is_done=False):
            self.content = content
            self.is_done = is_done
            self.tool_calls = None
            self.accumulated_tool_calls = None
            self.finish_reason = None

    async def mock_stream(*_args, **_kwargs):
        payload = {
            "components_script": (
                "pfusch('current-weather', { city: 'Berlin', data: null, loading: true, error: null }, (state) => [\n"
                "  html.div(`${Math.round(state.data.current?.temperature ?? state.data.temperature ?? 0)}°C`)\n"
                "]);"
            ),
            "test_script": (
                "import { it } from 'node:test';\n"
                "import assert from 'node:assert/strict';\n"
                "import { dummyData } from './dummy_data.js';\n"
                "it('weak weather assertion', async () => {\n"
                "  const svc = globalThis.service || new globalThis.McpService();\n"
                "  const weather = dummyData.get_weather_details ?? { current: { temperature: 22 } };\n"
                "  svc.test.addResolved('get_weather_details', weather);\n"
                "  const text = '0°C';\n"
                "  assert.ok(text.includes('22') || text.includes('°C'));\n"
                "});"
            ),
        }
        yield MockChunk(content=json.dumps(payload), is_done=False)
        yield MockChunk(content=None, is_done=True)

    tgi_service = MagicMock()
    tgi_service.llm_client = MagicMock()
    tgi_service.llm_client.stream_completion = mock_stream

    run_tests_called = False
    iterative_fix_called = False

    def run_tests(_service_script, _components_script, _test_script, _dummy_data):
        nonlocal run_tests_called
        run_tests_called = True
        return True, "ok"

    async def iterative_test_fix(**_kwargs):
        nonlocal iterative_fix_called
        iterative_fix_called = True
        return False, "", "", "", None, []

    @asynccontextmanager
    async def mock_chunk_reader(stream_source):
        class Reader:
            def as_parsed(self):
                return stream_source

        yield Reader()

    messages = [
        Message(role=MessageRole.SYSTEM, content="sys"),
        Message(role=MessageRole.USER, content="user"),
    ]

    dummy_data = (
        "export const dummyData = {\n"
        '  "get_weather_details": {\n'
        '    "city": "Berlin",\n'
        '    "temperature_c": 7.7,\n'
        '    "relative_humidity_percent": 94,\n'
        '    "wind_speed_kmh": 10.9\n'
        "  }\n"
        "};\n"
        "export const dummyDataSchemaHints = {};"
    )

    result_payload = None
    async for item in run_phase1_attempt(
        attempt=1,
        max_attempts=3,
        messages=messages,
        allowed_tools=[],
        dummy_data=dummy_data,
        access_token=None,
        tgi_service=tgi_service,
        parse_json=lambda content: json.loads(content),
        run_tests=run_tests,
        iterative_test_fix=iterative_test_fix,
        chunk_reader=mock_chunk_reader,
        ui_model_headers={},
    ):
        if isinstance(item, dict) and item.get("type") == "result":
            result_payload = item

    assert result_payload is not None
    assert result_payload["success"] is False
    assert "tool_fix_failed_after_initial_test_failure" in result_payload["reason"]
    assert "schema_contract_risk_detected" in result_payload["reason"]
    assert run_tests_called is True
    assert iterative_fix_called is True


@pytest.mark.asyncio
async def test_phase1_allows_weather_schema_exact_keys_and_strong_assertions():
    class MockChunk:
        def __init__(self, content=None, is_done=False):
            self.content = content
            self.is_done = is_done
            self.tool_calls = None
            self.accumulated_tool_calls = None
            self.finish_reason = None

    async def mock_stream(*_args, **_kwargs):
        payload = {
            "components_script": (
                "pfusch('current-weather', { city: 'Berlin', data: null, loading: true, error: null }, (state) => [\n"
                "  html.div(`${Math.round(state.data.temperature_c ?? 0)}°C`)\n"
                "]);"
            ),
            "test_script": (
                "import { it } from 'node:test';\n"
                "import assert from 'node:assert/strict';\n"
                "import { dummyData } from './dummy_data.js';\n"
                "it('schema exact assertion', async () => {\n"
                "  const svc = globalThis.service || new globalThis.McpService();\n"
                "  svc.test.addResolved('get_weather_details', dummyData.get_weather_details);\n"
                "  const expected = String(Math.round(dummyData.get_weather_details.temperature_c));\n"
                "  const text = `${expected}°C`;\n"
                "  assert.ok(text.includes(expected));\n"
                "});"
            ),
        }
        yield MockChunk(content=json.dumps(payload), is_done=False)
        yield MockChunk(content=None, is_done=True)

    tgi_service = MagicMock()
    tgi_service.llm_client = MagicMock()
    tgi_service.llm_client.stream_completion = mock_stream

    run_tests_called = False

    def run_tests(_service_script, _components_script, _test_script, _dummy_data):
        nonlocal run_tests_called
        run_tests_called = True
        return True, "ok"

    async def iterative_test_fix(**_kwargs):
        return False, "", "", "", None, []

    @asynccontextmanager
    async def mock_chunk_reader(stream_source):
        class Reader:
            def as_parsed(self):
                return stream_source

        yield Reader()

    messages = [
        Message(role=MessageRole.SYSTEM, content="sys"),
        Message(role=MessageRole.USER, content="user"),
    ]

    dummy_data = (
        "export const dummyData = {\n"
        '  "get_weather_details": {\n'
        '    "city": "Berlin",\n'
        '    "temperature_c": 7.7,\n'
        '    "relative_humidity_percent": 94,\n'
        '    "wind_speed_kmh": 10.9\n'
        "  }\n"
        "};\n"
        "export const dummyDataSchemaHints = {};"
    )

    result_payload = None
    async for item in run_phase1_attempt(
        attempt=1,
        max_attempts=3,
        messages=messages,
        allowed_tools=[],
        dummy_data=dummy_data,
        access_token=None,
        tgi_service=tgi_service,
        parse_json=lambda content: json.loads(content),
        run_tests=run_tests,
        iterative_test_fix=iterative_test_fix,
        chunk_reader=mock_chunk_reader,
        ui_model_headers={},
    ):
        if isinstance(item, dict) and item.get("type") == "result":
            result_payload = item

    assert result_payload is not None
    assert result_payload["success"] is True
    assert run_tests_called is True


@pytest.mark.asyncio
async def test_phase1_rejects_iterative_fix_result_when_schema_risks_remain():
    class MockChunk:
        def __init__(self, content=None, is_done=False):
            self.content = content
            self.is_done = is_done
            self.tool_calls = None
            self.accumulated_tool_calls = None
            self.finish_reason = None

    async def mock_stream(*_args, **_kwargs):
        payload = {
            "components_script": (
                "pfusch('current-weather', { city: 'Berlin', data: null, loading: true, error: null }, (state) => [\n"
                "  html.div(`${Math.round(state.data.current?.temperature ?? state.data.temperature ?? 0)}°C`)\n"
                "]);"
            ),
            "test_script": (
                "import { it } from 'node:test';\n" "it('placeholder', () => {});"
            ),
        }
        yield MockChunk(content=json.dumps(payload), is_done=False)
        yield MockChunk(content=None, is_done=True)

    tgi_service = MagicMock()
    tgi_service.llm_client = MagicMock()
    tgi_service.llm_client.stream_completion = mock_stream

    run_tests_calls = 0

    def run_tests(_service_script, _components_script, _test_script, _dummy_data):
        nonlocal run_tests_calls
        run_tests_calls += 1
        return True, "ok"

    async def iterative_test_fix(**_kwargs):
        # Pretend fixer succeeded but returned scripts that still violate schema mapping.
        return (
            True,
            "",
            "pfusch('current-weather', { city: 'Berlin', data: null, loading: true, error: null }, (state) => [html.div(`${Math.round(state.data.temperature ?? 0)}°C`)]);",
            "import { it } from 'node:test'; it('ok', () => {});",
            _kwargs.get("dummy_data"),
            _kwargs.get("messages") or [],
        )

    @asynccontextmanager
    async def mock_chunk_reader(stream_source):
        class Reader:
            def as_parsed(self):
                return stream_source

        yield Reader()

    messages = [
        Message(role=MessageRole.SYSTEM, content="sys"),
        Message(role=MessageRole.USER, content="user"),
    ]

    dummy_data = (
        "export const dummyData = {\n"
        '  "get_weather_details": {\n'
        '    "city": "Berlin",\n'
        '    "temperature_c": 7.7\n'
        "  }\n"
        "};\n"
        "export const dummyDataSchemaHints = {};"
    )

    result_payload = None
    async for item in run_phase1_attempt(
        attempt=1,
        max_attempts=3,
        messages=messages,
        allowed_tools=[],
        dummy_data=dummy_data,
        access_token=None,
        tgi_service=tgi_service,
        parse_json=lambda content: json.loads(content),
        run_tests=run_tests,
        iterative_test_fix=iterative_test_fix,
        chunk_reader=mock_chunk_reader,
        ui_model_headers={},
    ):
        if isinstance(item, dict) and item.get("type") == "result":
            result_payload = item

    assert result_payload is not None
    assert result_payload["success"] is False
    assert "schema_contract_risk_after_fix" in result_payload["reason"]
    assert run_tests_calls >= 1
