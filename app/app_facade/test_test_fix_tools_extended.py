import asyncio
import pytest
from app.app_facade.test_fix_tools import (
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
