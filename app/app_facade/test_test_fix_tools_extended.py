import asyncio
import pytest
from app.app_facade.test_fix_tools import (
    IterativeTestFixer,
    ToolResult,
    _parse_tap_output,
    run_tool_driven_test_fix,
)
from app.tgi.models import Message, MessageRole
import json


def test_parse_tap_output_summary():
    """Test parsing TAP output with summary lines."""
    # Test with standard TAP output with summary
    tap_output = """
TAP version 13
# Subtest: Test Suite
    # Subtest: should pass
    ok 1 - should pass
    1..1
ok 1 - Test Suite
1..1
# tests 10
# suites 3
# pass 5
# fail 5
# cancelled 0
# skipped 0
# todo 0
# duration_ms 279.372845
"""
    passed, failed, failed_tests = _parse_tap_output(tap_output)
    assert passed == 5
    assert failed == 5
    assert (
        len(failed_tests) == 0
    )  # Summary doesn't provide failed test names directly usually, but logic should handle it if present

    # Test with standard TAP output without summary (should rely on counting lines)
    tap_output_no_summary = """
TAP version 13
ok 1 - Test 1
not ok 2 - Test 2
ok 3 - Test 3
"""
    passed, failed, failed_tests = _parse_tap_output(tap_output_no_summary)
    assert passed == 2
    assert failed == 1
    assert "Test 2" in failed_tests

    # Test with only summary (e.g. if output was truncated/weird)
    tap_output_only_summary = """
# pass 10
# fail 2
"""
    passed, failed, failed_tests = _parse_tap_output(tap_output_only_summary)
    assert passed == 10
    assert failed == 2


@pytest.mark.asyncio
async def test_tool_driven_cycle_read_only_tools_dont_count():
    """Test that read-only tools do not count towards the attempt limit."""
    service_script = "export class McpService {}"
    components_script = "// components"
    # Failing test initially
    test_script = """
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

describe('Tool-driven cycle', () => {
    it('fails initially', () => {
        assert.ok(false);
    });
});
"""

    class MockLLMClient:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                # First call: Failing test context provided. Helper uses search_files (read-only)
                # Should NOT increment attempt count effectively (attempt stays 0 or logic adjusts)
                return type(
                    "Response",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "message": type(
                                        "Message",
                                        (),
                                        {
                                            "content": None,
                                            "tool_calls": [
                                                type(
                                                    "ToolCall",
                                                    (),
                                                    {
                                                        "id": "call_1",
                                                        "type": "function",
                                                        "function": type(
                                                            "Function",
                                                            (),
                                                            {
                                                                "name": "search_files",
                                                                "arguments": '{"regex": "fails initially"}',
                                                            },
                                                        ),
                                                    },
                                                )
                                            ],
                                        },
                                    )
                                },
                            )
                        ]
                    },
                )
            elif self.calls == 2:
                # Second call: Still failing context (since no changes made). Helper uses get_script_lines (read-only)
                return type(
                    "Response",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "message": type(
                                        "Message",
                                        (),
                                        {
                                            "content": None,
                                            "tool_calls": [
                                                type(
                                                    "ToolCall",
                                                    (),
                                                    {
                                                        "id": "call_2",
                                                        "type": "function",
                                                        "function": type(
                                                            "Function",
                                                            (),
                                                            {
                                                                "name": "get_script_lines",
                                                                "arguments": '{"script_type": "test"}',
                                                            },
                                                        ),
                                                    },
                                                )
                                            ],
                                        },
                                    )
                                },
                            )
                        ]
                    },
                )
            elif self.calls == 3:
                # Third call: Helper fixes the test
                fixed_test = (
                    "import { describe, it } from 'node:test'; "
                    "import assert from 'node:assert/strict'; "
                    "describe('Tool-driven cycle', () => { "
                    "it('passes', () => { assert.ok(true); }); "
                    "});"
                )
                return type(
                    "Response",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "message": type(
                                        "Message",
                                        (),
                                        {
                                            "content": None,
                                            "tool_calls": [
                                                type(
                                                    "ToolCall",
                                                    (),
                                                    {
                                                        "id": "call_3",
                                                        "type": "function",
                                                        "function": type(
                                                            "Function",
                                                            (),
                                                            {
                                                                "name": "update_test_script",
                                                                "arguments": json.dumps(
                                                                    {
                                                                        "new_script": fixed_test
                                                                    }
                                                                ),
                                                            },
                                                        ),
                                                    },
                                                )
                                            ],
                                        },
                                    )
                                },
                            )
                        ]
                    },
                )
            else:
                # Subsequent calls: Helper verifies tests passed.
                return type(
                    "Response",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "message": type(
                                        "Message",
                                        (),
                                        {
                                            "content": None,
                                            "tool_calls": [
                                                type(
                                                    "ToolCall",
                                                    (),
                                                    {
                                                        "id": "call_4",
                                                        "type": "function",
                                                        "function": type(
                                                            "Function",
                                                            (),
                                                            {
                                                                "name": "run_tests",
                                                                "arguments": "{}",
                                                            },
                                                        ),
                                                    },
                                                )
                                            ],
                                        },
                                    )
                                },
                            )
                        ]
                    },
                )

        def _build_request_params(self, request):
            return {}

    class MockTGIService:
        def __init__(self):
            # Create a class for the client to support method binding
            class Client:
                def __init__(self):
                    self.client = type(
                        "Chat",
                        (),
                        {
                            "chat": type(
                                "Completions", (), {"completions": MockLLMClient()}
                            )
                        },
                    )()
                    self.model_format = None

                def _build_request_params(self, request):
                    return {}

            self.llm_client = Client()

    messages = [Message(role=MessageRole.SYSTEM, content="system")]

    # We set max_attempts to 2.
    # If read-only tools counted, we'd fail after call 2 (attempt 2).
    # Since we have 2 read-only calls and then a fix, we need the read-only calls to NOT consume attempts.
    # The logic in run_tool_driven_test_fix is:
    # while attempt < max_attempts:
    #   attempt += 1
    #   ...
    #   if read_only: attempt -= 1
    # So effectively, unlimited read-only calls as long as they don't loop forever or hit other limits (which aren't there explicitly besides eventually giving up or providing a fix).
    # Actually wait, `no_tool_call_attempts` handles empty responses.
    # The read-only logic decrements `attempt`.
    # So calls 1 (search) -> attempt becomes 1 -> read-only -> attempt becomes 0
    # Call 2 (read) -> attempt becomes 1 -> read-only -> attempt becomes 0
    # Call 3 (fix) -> attempt becomes 1 -> modification -> attempt stays 1
    # Call 4 (done) -> attempt becomes 2 -> done

    success, service, components, test, dummy, updated = await asyncio.wait_for(
        run_tool_driven_test_fix(
            tgi_service=MockTGIService(),
            service_script=service_script,
            components_script=components_script,
            test_script=test_script,
            dummy_data=None,
            messages=messages,
            allowed_tools=None,
            access_token=None,
            max_attempts=2,  # Low limit to prove read-only tools don't consume it
            event_queue=None,
            extra_headers=None,
        ),
        timeout=10,
    )

    assert success
    assert "assert.ok(true)" in test


def _make_tool_call_response(tool_name: str, arguments: str = "{}"):
    return type(
        "Response",
        (),
        {
            "choices": [
                type(
                    "Choice",
                    (),
                    {
                        "message": type(
                            "Message",
                            (),
                            {
                                "content": None,
                                "tool_calls": [
                                    type(
                                        "ToolCall",
                                        (),
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": type(
                                                "Function",
                                                (),
                                                {
                                                    "name": tool_name,
                                                    "arguments": arguments,
                                                },
                                            ),
                                        },
                                    )
                                ],
                            },
                        )
                    },
                )
            ]
        },
    )


def _tool_name(tool):
    if isinstance(tool, dict):
        return (tool.get("function") or {}).get("name")
    function = getattr(tool, "function", None)
    if isinstance(function, dict):
        return function.get("name")
    return getattr(function, "name", None)


@pytest.mark.asyncio
async def test_strategy_adjust_test_filters_service_mutations(monkeypatch):
    observed_tool_names = []
    run_calls = {"count": 0}

    def fake_run_tests(self, test_name=None):
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            return ToolResult(success=False, content="not ok 1 - failing")
        return ToolResult(
            success=True,
            content="TAP version 13\n# pass 1\n# fail 0\n",
            metadata={"passed": 1, "failed": 0},
        )

    monkeypatch.setattr(IterativeTestFixer, "run_tests", fake_run_tests)

    class MockCompletions:
        async def create(self, **kwargs):
            request = kwargs["chat_request"]
            observed_tool_names.extend(
                [_tool_name(tool) for tool in (request.tools or [])]
            )
            return _make_tool_call_response("run_tests")

    class MockClient:
        def __init__(self):
            self.client = type(
                "Chat",
                (),
                {
                    "chat": type(
                        "CompletionsWrap", (), {"completions": MockCompletions()}
                    )()
                },
            )()
            self.model_format = None

        def _build_request_params(self, request):
            return {"chat_request": request}

    class MockTGIService:
        def __init__(self):
            self.llm_client = MockClient()

    success, *_ = await run_tool_driven_test_fix(
        tgi_service=MockTGIService(),
        service_script="export class McpService {}",
        components_script="export const init = () => {};",
        test_script="import { test } from 'node:test'; test('x', () => {});",
        dummy_data=None,
        messages=[Message(role=MessageRole.SYSTEM, content="system")],
        allowed_tools=None,
        access_token=None,
        max_attempts=2,
        event_queue=None,
        extra_headers=None,
        strategy_mode="adjust_test",
    )

    assert success
    assert "update_test_script" in observed_tool_names
    assert "update_dummy_data" in observed_tool_names
    assert "update_service_script" not in observed_tool_names
    assert "update_components_script" not in observed_tool_names


@pytest.mark.asyncio
async def test_strategy_fix_code_filters_test_mutations(monkeypatch):
    observed_tool_names = []
    run_calls = {"count": 0}

    def fake_run_tests(self, test_name=None):
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            return ToolResult(success=False, content="not ok 1 - failing")
        return ToolResult(
            success=True,
            content="TAP version 13\n# pass 1\n# fail 0\n",
            metadata={"passed": 1, "failed": 0},
        )

    monkeypatch.setattr(IterativeTestFixer, "run_tests", fake_run_tests)

    class MockCompletions:
        async def create(self, **kwargs):
            request = kwargs["chat_request"]
            observed_tool_names.extend(
                [_tool_name(tool) for tool in (request.tools or [])]
            )
            return _make_tool_call_response("run_tests")

    class MockClient:
        def __init__(self):
            self.client = type(
                "Chat",
                (),
                {
                    "chat": type(
                        "CompletionsWrap", (), {"completions": MockCompletions()}
                    )()
                },
            )()
            self.model_format = None

        def _build_request_params(self, request):
            return {"chat_request": request}

    class MockTGIService:
        def __init__(self):
            self.llm_client = MockClient()

    success, *_ = await run_tool_driven_test_fix(
        tgi_service=MockTGIService(),
        service_script="export class McpService {}",
        components_script="export const init = () => {};",
        test_script="import { test } from 'node:test'; test('x', () => {});",
        dummy_data=None,
        messages=[Message(role=MessageRole.SYSTEM, content="system")],
        allowed_tools=None,
        access_token=None,
        max_attempts=2,
        event_queue=None,
        extra_headers=None,
        strategy_mode="fix_code",
    )

    assert success
    assert "update_service_script" in observed_tool_names
    assert "update_components_script" in observed_tool_names
    assert "update_test_script" not in observed_tool_names
    assert "update_dummy_data" not in observed_tool_names


def _make_multi_tool_call_response(tool_calls):
    calls = []
    for idx, (tool_name, arguments) in enumerate(tool_calls, start=1):
        calls.append(
            type(
                "ToolCall",
                (),
                {
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": type(
                        "Function",
                        (),
                        {
                            "name": tool_name,
                            "arguments": arguments,
                        },
                    ),
                },
            )
        )

    return type(
        "Response",
        (),
        {
            "choices": [
                type(
                    "Choice",
                    (),
                    {
                        "message": type(
                            "Message",
                            (),
                            {
                                "content": None,
                                "tool_calls": calls,
                            },
                        )
                    },
                )
            ]
        },
    )


@pytest.mark.asyncio
async def test_tool_fix_rolls_back_on_sharp_regression(monkeypatch):
    run_calls = {"count": 0}

    def fake_run_tests(self, test_name=None):
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            return ToolResult(success=False, content="# pass 3\n# fail 2\n")
        return ToolResult(success=False, content="# pass 0\n# fail 5\n")

    monkeypatch.setattr(IterativeTestFixer, "run_tests", fake_run_tests)

    class MockCompletions:
        async def create(self, **kwargs):
            return _make_multi_tool_call_response(
                [
                    (
                        "update_service_script",
                        json.dumps({"new_script": "service-bad"}),
                    ),
                    ("run_tests", "{}"),
                ]
            )

    class MockClient:
        def __init__(self):
            self.client = type(
                "Chat",
                (),
                {
                    "chat": type(
                        "CompletionsWrap", (), {"completions": MockCompletions()}
                    )()
                },
            )()
            self.model_format = None

        def _build_request_params(self, request):
            return {"chat_request": request}

    class MockTGIService:
        def __init__(self):
            self.llm_client = MockClient()

    original_service = "service-original"
    success, service, *_ = await run_tool_driven_test_fix(
        tgi_service=MockTGIService(),
        service_script=original_service,
        components_script="components-original",
        test_script="test-original",
        dummy_data=None,
        messages=[Message(role=MessageRole.SYSTEM, content="system")],
        allowed_tools=None,
        access_token=None,
        max_attempts=1,
        event_queue=None,
        extra_headers=None,
    )

    assert success is False
    assert service == original_service


@pytest.mark.asyncio
async def test_tool_fix_forces_run_tests_after_mutation(monkeypatch):
    run_calls = {"count": 0}

    def fake_run_tests(self, test_name=None):
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            return ToolResult(success=False, content="# pass 0\n# fail 1\n")
        return ToolResult(success=True, content="# pass 1\n# fail 0\n")

    monkeypatch.setattr(IterativeTestFixer, "run_tests", fake_run_tests)

    class MockCompletions:
        async def create(self, **kwargs):
            return _make_tool_call_response(
                "update_test_script",
                json.dumps(
                    {
                        "new_script": "import { test } from 'node:test'; test('x', () => {});"
                    }
                ),
            )

    class MockClient:
        def __init__(self):
            self.client = type(
                "Chat",
                (),
                {
                    "chat": type(
                        "CompletionsWrap", (), {"completions": MockCompletions()}
                    )()
                },
            )()
            self.model_format = None

        def _build_request_params(self, request):
            return {"chat_request": request}

    class MockTGIService:
        def __init__(self):
            self.llm_client = MockClient()

    success, *_ = await run_tool_driven_test_fix(
        tgi_service=MockTGIService(),
        service_script="export class McpService {}",
        components_script="export const init = () => {};",
        test_script="import { test } from 'node:test'; test('x', () => { throw new Error('fail'); });",
        dummy_data=None,
        messages=[Message(role=MessageRole.SYSTEM, content="system")],
        allowed_tools=None,
        access_token=None,
        max_attempts=2,
        event_queue=None,
        extra_headers=None,
    )

    assert success is True
    assert run_calls["count"] >= 2


@pytest.mark.asyncio
async def test_tool_fix_read_only_streak_forces_run_tests(monkeypatch):
    monkeypatch.setattr("app.app_facade.test_fix_tools.READ_ONLY_STREAK_LIMIT", 2)

    run_calls = {"count": 0}

    def fake_run_tests(self, test_name=None):
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            return ToolResult(success=False, content="# pass 0\n# fail 1\n")
        return ToolResult(success=True, content="# pass 1\n# fail 0\n")

    monkeypatch.setattr(IterativeTestFixer, "run_tests", fake_run_tests)

    class MockCompletions:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            return _make_tool_call_response(
                "search_files",
                json.dumps({"regex": "x"}),
            )

    class MockClient:
        def __init__(self):
            self.completions = MockCompletions()
            self.client = type(
                "Chat",
                (),
                {
                    "chat": type(
                        "CompletionsWrap",
                        (),
                        {"completions": self.completions},
                    )()
                },
            )()
            self.model_format = None

        def _build_request_params(self, request):
            return {"chat_request": request}

    class MockTGIService:
        def __init__(self):
            self.llm_client = MockClient()

    success, *_ = await run_tool_driven_test_fix(
        tgi_service=MockTGIService(),
        service_script="export class McpService {}",
        components_script="export const init = () => {};",
        test_script="import { test } from 'node:test'; test('x', () => { throw new Error('fail'); });",
        dummy_data=None,
        messages=[Message(role=MessageRole.SYSTEM, content="system")],
        allowed_tools=None,
        access_token=None,
        max_attempts=3,
        event_queue=None,
        extra_headers=None,
    )

    assert success is True
    assert run_calls["count"] >= 2


@pytest.mark.asyncio
async def test_fix_code_bails_early_on_repeated_assertion_signature(monkeypatch):
    monkeypatch.setattr("app.app_facade.test_fix_tools.READ_ONLY_STREAK_LIMIT", 1)

    run_calls = {"count": 0}

    def fake_run_tests(self, test_name=None):
        run_calls["count"] += 1
        return ToolResult(
            success=False,
            content=(
                "TAP version 13\n"
                "not ok 1 - Weather Dashboard Tests\n"
                "AssertionError [ERR_ASSERTION]: expected true\n"
                "at TestContext.<anonymous> (file:///tmp/tmpx/user_test.js:27:10)\n"
                "# pass 1\n"
                "# fail 2\n"
            ),
        )

    monkeypatch.setattr(IterativeTestFixer, "run_tests", fake_run_tests)

    class MockCompletions:
        async def create(self, **kwargs):
            return _make_tool_call_response(
                "search_files",
                json.dumps({"regex": "aqData\\.pm2_5", "script_type": "test"}),
            )

    class MockClient:
        def __init__(self):
            self.client = type(
                "Chat",
                (),
                {
                    "chat": type(
                        "CompletionsWrap", (), {"completions": MockCompletions()}
                    )()
                },
            )()
            self.model_format = None

        def _build_request_params(self, request):
            return {"chat_request": request}

    class MockTGIService:
        def __init__(self):
            self.llm_client = MockClient()

    success, *_ = await run_tool_driven_test_fix(
        tgi_service=MockTGIService(),
        service_script="export class McpService {}",
        components_script="export const init = () => {};",
        test_script="import { test } from 'node:test'; test('x', () => { throw new Error('fail'); });",
        dummy_data=None,
        messages=[Message(role=MessageRole.SYSTEM, content="system")],
        allowed_tools=None,
        access_token=None,
        max_attempts=12,
        event_queue=None,
        extra_headers=None,
        strategy_mode="fix_code",
    )

    assert success is False
    assert run_calls["count"] <= 3


@pytest.mark.asyncio
async def test_tool_start_event_includes_fix_explanation(monkeypatch):
    run_calls = {"count": 0}

    def fake_run_tests(self, test_name=None):
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            return ToolResult(success=False, content="# pass 0\n# fail 1\n")
        return ToolResult(success=True, content="# pass 1\n# fail 0\n")

    monkeypatch.setattr(IterativeTestFixer, "run_tests", fake_run_tests)

    class MockCompletions:
        async def create(self, **kwargs):
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {
                                        "content": (
                                            "FIX_EXPLANATION: The test expectation is inverted; "
                                            "update the assertion to match intended behavior."
                                        ),
                                        "tool_calls": [
                                            type(
                                                "ToolCall",
                                                (),
                                                {
                                                    "id": "call_1",
                                                    "type": "function",
                                                    "function": type(
                                                        "Function",
                                                        (),
                                                        {
                                                            "name": "update_test_script",
                                                            "arguments": json.dumps(
                                                                {
                                                                    "new_script": (
                                                                        "import { test } from 'node:test'; "
                                                                        "test('x', () => {});"
                                                                    )
                                                                }
                                                            ),
                                                        },
                                                    ),
                                                },
                                            )
                                        ],
                                    },
                                )
                            },
                        )
                    ]
                },
            )

    class MockClient:
        def __init__(self):
            self.client = type(
                "Chat",
                (),
                {
                    "chat": type(
                        "CompletionsWrap", (), {"completions": MockCompletions()}
                    )()
                },
            )()
            self.model_format = None

        def _build_request_params(self, request):
            return {"chat_request": request}

    class MockTGIService:
        def __init__(self):
            self.llm_client = MockClient()

    event_queue: asyncio.Queue = asyncio.Queue()
    success, *_ = await run_tool_driven_test_fix(
        tgi_service=MockTGIService(),
        service_script="export class McpService {}",
        components_script="export const init = () => {};",
        test_script="import { test } from 'node:test'; test('x', () => { throw new Error('fail'); });",
        dummy_data=None,
        messages=[Message(role=MessageRole.SYSTEM, content="system")],
        allowed_tools=None,
        access_token=None,
        max_attempts=2,
        event_queue=event_queue,
        extra_headers=None,
    )

    tool_start_events = []
    while not event_queue.empty():
        event = event_queue.get_nowait()
        if event.get("event") == "tool_start":
            tool_start_events.append(event)

    assert success is True
    assert tool_start_events
    assert tool_start_events[0]["tool"] == "update_test_script"
    assert "inverted" in str(tool_start_events[0].get("fix_explanation", ""))
    assert "inverted" in str(tool_start_events[0].get("why", ""))


@pytest.mark.asyncio
async def test_tool_start_event_infers_fix_explanation_when_missing(monkeypatch):
    run_calls = {"count": 0}

    def fake_run_tests(self, test_name=None):
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            return ToolResult(success=False, content="# pass 0\n# fail 1\n")
        return ToolResult(success=True, content="# pass 1\n# fail 0\n")

    monkeypatch.setattr(IterativeTestFixer, "run_tests", fake_run_tests)

    class MockCompletions:
        async def create(self, **kwargs):
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {
                                        "content": None,
                                        "tool_calls": [
                                            type(
                                                "ToolCall",
                                                (),
                                                {
                                                    "id": "call_1",
                                                    "type": "function",
                                                    "function": type(
                                                        "Function",
                                                        (),
                                                        {
                                                            "name": "search_files",
                                                            "arguments": json.dumps(
                                                                {"regex": "forecast"}
                                                            ),
                                                        },
                                                    ),
                                                },
                                            )
                                        ],
                                    },
                                )
                            },
                        )
                    ]
                },
            )

    class MockClient:
        def __init__(self):
            self.client = type(
                "Chat",
                (),
                {
                    "chat": type(
                        "CompletionsWrap", (), {"completions": MockCompletions()}
                    )()
                },
            )()
            self.model_format = None

        def _build_request_params(self, request):
            return {"chat_request": request}

    class MockTGIService:
        def __init__(self):
            self.llm_client = MockClient()

    event_queue: asyncio.Queue = asyncio.Queue()
    success, *_ = await run_tool_driven_test_fix(
        tgi_service=MockTGIService(),
        service_script="export class McpService {}",
        components_script="export const init = () => {};",
        test_script="import { test } from 'node:test'; test('x', () => { throw new Error('fail'); });",
        dummy_data=None,
        messages=[Message(role=MessageRole.SYSTEM, content="system")],
        allowed_tools=None,
        access_token=None,
        max_attempts=2,
        event_queue=event_queue,
        extra_headers=None,
    )

    tool_start_events = []
    while not event_queue.empty():
        event = event_queue.get_nowait()
        if event.get("event") == "tool_start":
            tool_start_events.append(event)

    assert success is True
    assert tool_start_events
    assert tool_start_events[0]["tool"] == "search_files"
    assert str(tool_start_events[0].get("fix_explanation", "")).strip()
    assert str(tool_start_events[0].get("why", "")).strip()


@pytest.mark.asyncio
async def test_post_success_validator_retries_instead_of_returning_success(monkeypatch):
    run_calls = {"count": 0}

    def fake_run_tests(self, test_name=None):
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            return ToolResult(success=False, content="# pass 0\n# fail 1\n")
        return ToolResult(
            success=True,
            content="# pass 1\n# fail 0\n",
            metadata={"passed": 1, "failed": 0},
        )

    monkeypatch.setattr(IterativeTestFixer, "run_tests", fake_run_tests)

    class MockCompletions:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            return _make_tool_call_response("run_tests", "{}")

    class MockClient:
        def __init__(self):
            self.completions = MockCompletions()
            self.client = type(
                "Chat",
                (),
                {
                    "chat": type(
                        "CompletionsWrap",
                        (),
                        {"completions": self.completions},
                    )()
                },
            )()
            self.model_format = None

        def _build_request_params(self, request):
            return {"chat_request": request}

    class MockTGIService:
        def __init__(self):
            self.llm_client = MockClient()

    validator_calls = {"count": 0}

    def always_fail_validator(_svc, _comps, _tests, _fixtures):
        validator_calls["count"] += 1
        return False, "schema_contract_rejected"

    success, _svc, _comps, _tests, _dummy, updated = await run_tool_driven_test_fix(
        tgi_service=MockTGIService(),
        service_script="export class McpService {}",
        components_script="export const mode = 'bad';",
        test_script="import { test } from 'node:test'; test('x', () => { throw new Error('fail'); });",
        dummy_data=None,
        messages=[Message(role=MessageRole.SYSTEM, content="system")],
        allowed_tools=None,
        access_token=None,
        max_attempts=2,
        event_queue=None,
        extra_headers=None,
        post_success_validator=always_fail_validator,
    )

    assert success is False
    assert validator_calls["count"] >= 2
    assert any(
        (
            msg.role == MessageRole.USER
            and "contract validation failed" in (msg.content or "")
        )
        for msg in updated
    )


@pytest.mark.asyncio
async def test_post_success_validator_allows_success_after_mutation(monkeypatch):
    run_calls = {"count": 0}

    def fake_run_tests(self, test_name=None):
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            return ToolResult(success=False, content="# pass 0\n# fail 1\n")
        return ToolResult(
            success=True,
            content="# pass 1\n# fail 0\n",
            metadata={"passed": 1, "failed": 0},
        )

    monkeypatch.setattr(IterativeTestFixer, "run_tests", fake_run_tests)

    class MockCompletions:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return _make_tool_call_response("run_tests", "{}")
            return _make_tool_call_response(
                "update_components_script",
                json.dumps({"new_script": "export const mode = 'good';"}),
            )

    class MockClient:
        def __init__(self):
            self.completions = MockCompletions()
            self.client = type(
                "Chat",
                (),
                {
                    "chat": type(
                        "CompletionsWrap",
                        (),
                        {"completions": self.completions},
                    )()
                },
            )()
            self.model_format = None

        def _build_request_params(self, request):
            return {"chat_request": request}

    class MockTGIService:
        def __init__(self):
            self.llm_client = MockClient()

    validator_calls = {"count": 0}

    def mode_validator(_svc, comps, _tests, _fixtures):
        validator_calls["count"] += 1
        if "mode = 'good'" in comps:
            return True, None
        return False, "mode_not_fixed"

    success, _svc, comps, _tests, _dummy, _updated = await run_tool_driven_test_fix(
        tgi_service=MockTGIService(),
        service_script="export class McpService {}",
        components_script="export const mode = 'bad';",
        test_script="import { test } from 'node:test'; test('x', () => { throw new Error('fail'); });",
        dummy_data=None,
        messages=[Message(role=MessageRole.SYSTEM, content="system")],
        allowed_tools=None,
        access_token=None,
        max_attempts=3,
        event_queue=None,
        extra_headers=None,
        post_success_validator=mode_validator,
    )

    assert success is True
    assert "mode = 'good'" in comps
    assert validator_calls["count"] >= 2
