"""Tests for test_fix_tools module."""

import asyncio
import os

import pytest
import json

from app.app_facade.test_fix_tools import (
    IterativeTestFixer,
    run_tool_driven_test_fix,
    _parse_tap_output,
    _summarize_test_output,
)
from app.tgi.models import Message, MessageRole


def test_toolkit_initialization():
    """Test that toolkit initializes correctly."""
    helpers_dir = os.path.join(os.path.dirname(__file__), "node_test_helpers")
    toolkit = IterativeTestFixer(helpers_dir)
    assert toolkit.helpers_dir == helpers_dir
    assert toolkit.current_service_script is None
    assert toolkit.current_components_script is None
    assert toolkit.current_test_script is None


def test_setup_and_cleanup():
    """Test setup and cleanup of test environment."""
    helpers_dir = os.path.join(os.path.dirname(__file__), "node_test_helpers")
    toolkit = IterativeTestFixer(helpers_dir)

    service = "export class McpService {}"
    components = "import { pfusch } from './pfusch.js';"
    test = "import { describe, it } from 'node:test';"

    toolkit.setup_test_environment(service, components, test)

    assert toolkit.current_service_script == service
    assert toolkit.current_components_script == components
    assert toolkit.current_test_script == test
    assert toolkit.tmpdir is not None
    assert os.path.exists(toolkit.tmpdir)

    # Cleanup
    tmpdir = toolkit.tmpdir
    toolkit.cleanup()
    assert toolkit.tmpdir is None
    assert not os.path.exists(tmpdir)


def test_update_scripts():
    """Test updating individual scripts."""
    helpers_dir = os.path.join(os.path.dirname(__file__), "node_test_helpers")
    toolkit = IterativeTestFixer(helpers_dir)

    toolkit.setup_test_environment("s1", "c1", "t1")

    # Update test script
    result = toolkit.update_test_script("new test")
    assert result.success
    assert toolkit.current_test_script == "new test"

    # Update service script
    result = toolkit.update_service_script("new service")
    assert result.success
    assert toolkit.current_service_script == "new service"

    # Update components script
    result = toolkit.update_components_script("new components")
    assert result.success
    assert toolkit.current_components_script == "new components"

    toolkit.cleanup()


def test_get_script_lines():
    """Test getting specific lines from scripts."""
    helpers_dir = os.path.join(os.path.dirname(__file__), "node_test_helpers")
    toolkit = IterativeTestFixer(helpers_dir)

    test_script = "line 1\nline 2\nline 3\nline 4\nline 5"
    toolkit.setup_test_environment("service", "components", test_script)

    # Get lines 2-3
    result = toolkit.get_script_lines("test", 2, 3)
    assert result.success
    assert result.content == "line 2\nline 3"

    # Get full file when line numbers are omitted
    result = toolkit.get_script_lines("test")
    assert result.success
    assert result.content == test_script

    # Invalid range
    result = toolkit.get_script_lines("test", 1, 100)
    assert not result.success

    # Invalid script type
    result = toolkit.get_script_lines("invalid", 1, 2)
    assert not result.success

    toolkit.cleanup()


def test_get_current_scripts():
    """Test getting script metadata."""
    helpers_dir = os.path.join(os.path.dirname(__file__), "node_test_helpers")
    toolkit = IterativeTestFixer(helpers_dir)

    toolkit.setup_test_environment("line 1\nline 2", "line 1\nline 2\nline 3", "line 1")

    result = toolkit.get_current_scripts()
    assert result.success
    assert result.metadata["service_lines"] == 2
    assert result.metadata["components_lines"] == 3
    assert result.metadata["test_lines"] == 1

    toolkit.cleanup()


def test_run_tests_injects_mcp_service_helper_for_components_fallback():
    """Iterative fixer should provide globalThis.McpService when service script is empty."""
    helpers_dir = os.path.join(os.path.dirname(__file__), "node_test_helpers")
    toolkit = IterativeTestFixer(helpers_dir)
    components = (
        "const mcp = globalThis.service || new globalThis.McpService();\n"
        "if (!mcp || typeof mcp.call !== 'function') {\n"
        "  throw new Error('mcp unavailable');\n"
        "}\n"
    )
    test_script = (
        "import { describe, it } from 'node:test';\n"
        "import assert from 'node:assert/strict';\n"
        "import './app.js';\n"
        "describe('mcp helper prelude', () => {\n"
        "  it('bootstraps global mcp service', () => {\n"
        "    assert.equal(typeof globalThis.McpService, 'function');\n"
        "    assert.equal(typeof globalThis.service?.call, 'function');\n"
        "  });\n"
        "});\n"
    )

    toolkit.setup_test_environment("", components, test_script)
    result = toolkit.run_tests()
    assert result.success, result.content
    toolkit.cleanup()


def test_tool_definitions():
    """Test that tool definitions are valid."""
    helpers_dir = os.path.join(os.path.dirname(__file__), "node_test_helpers")
    toolkit = IterativeTestFixer(helpers_dir)

    tools = toolkit.get_tool_definitions()

    assert len(tools) == 9
    tool_names = [t["function"]["name"] for t in tools]

    assert "run_tests" in tool_names
    assert "run_debug_code" in tool_names
    assert "update_test_script" in tool_names
    assert "update_service_script" in tool_names
    assert "update_components_script" in tool_names
    assert "update_dummy_data" in tool_names
    assert "get_script_lines" in tool_names
    assert "get_current_scripts" in tool_names
    assert "search_files" in tool_names

    # Check structure
    for tool in tools:
        assert tool["type"] == "function"
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]


def test_execute_tool():
    """Test tool execution."""
    helpers_dir = os.path.join(os.path.dirname(__file__), "node_test_helpers")
    toolkit = IterativeTestFixer(helpers_dir)

    toolkit.setup_test_environment("service", "components", "test")

    # Execute update_test_script
    result = toolkit.execute_tool("update_test_script", {"new_script": "updated"})
    assert result.success
    assert toolkit.current_test_script == "updated"

    # Execute get_current_scripts
    result = toolkit.execute_tool("get_current_scripts", {})
    assert result.success

    # Execute unknown tool
    result = toolkit.execute_tool("unknown_tool", {})
    assert not result.success

    toolkit.cleanup()


def test_run_simple_passing_test():
    """Test running a simple passing test."""
    helpers_dir = os.path.join(os.path.dirname(__file__), "node_test_helpers")
    toolkit = IterativeTestFixer(helpers_dir)

    service = """
export class McpService {
    constructor() {
        this.baseUrl = '/api/test/tools';
    }
    
    async testMethod() {
        return { success: true };
    }
}
"""

    components = """
// Simple component for testing
"""

    test = """
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

describe('Test Suite', () => {
    it('should pass', () => {
        assert.ok(true);
    });
});
"""

    toolkit.setup_test_environment(service, components, test)
    result = toolkit.run_tests()

    assert result.success
    assert "pass 1" in result.content.lower() or "ok 1" in result.content.lower()

    toolkit.cleanup()


def test_run_simple_failing_test():
    """Test running a simple failing test."""
    helpers_dir = os.path.join(os.path.dirname(__file__), "node_test_helpers")
    toolkit = IterativeTestFixer(helpers_dir)

    service = "export class McpService {}"
    components = ""
    test = """
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

describe('Test Suite', () => {
    it('should fail', () => {
        assert.equal(1, 2);
    });
});
"""

    toolkit.setup_test_environment(service, components, test)
    result = toolkit.run_tests()

    assert not result.success
    assert "fail" in result.content.lower() or "not ok" in result.content.lower()

    toolkit.cleanup()


@pytest.mark.asyncio
async def test_tool_driven_cycle_passes_without_llm():
    service_script = "export class McpService {}"
    components_script = "// components"
    test_script = """
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

describe('Tool-driven cycle', () => {
    it('passes immediately', () => {
        assert.ok(true);
    });
});
"""

    class DummyTGI:
        llm_client = None

    messages = [Message(role=MessageRole.SYSTEM, content="system")]

    success, service, components, test, dummy, updated = await run_tool_driven_test_fix(
        tgi_service=DummyTGI(),
        service_script=service_script,
        components_script=components_script,
        test_script=test_script,
        dummy_data=None,
        messages=messages,
        allowed_tools=None,
        access_token=None,
        max_attempts=1,
        event_queue=None,
        extra_headers=None,
    )

    assert success
    assert service == service_script
    assert components == components_script
    assert test == test_script
    assert dummy is None
    assert updated == messages


@pytest.mark.asyncio
async def test_tool_driven_fix_no_tool_calls_does_not_loop_forever():
    service_script = "export class McpService {}"
    components_script = "// components"
    test_script = """
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

describe('fails first', () => {
    it('fails', () => {
        assert.equal(1, 2);
    });
});
"""

    class MockLLMClient:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
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
                                        "content": "x" * 400,
                                        "tool_calls": [],
                                    },
                                )
                            },
                        )
                    ]
                },
            )

    class MockTGIService:
        def __init__(self):
            self._mock = MockLLMClient()

            class Client:
                def __init__(self, mock):
                    self.client = type(
                        "Chat",
                        (),
                        {"chat": type("Completions", (), {"completions": mock})},
                    )()
                    self.model_format = None

                def _build_request_params(self, request):
                    return {}

            self.llm_client = Client(self._mock)

    messages = [Message(role=MessageRole.SYSTEM, content="system")]
    tgi = MockTGIService()

    result = await asyncio.wait_for(
        run_tool_driven_test_fix(
            tgi_service=tgi,
            service_script=service_script,
            components_script=components_script,
            test_script=test_script,
            dummy_data=None,
            messages=messages,
            allowed_tools=None,
            access_token=None,
            max_attempts=3,
            event_queue=None,
            extra_headers=None,
        ),
        timeout=8,
    )

    success, _service, _components, _test, _dummy, updated = result
    assert success is False
    assert tgi._mock.calls >= 3
    assert any(
        (
            msg.role == MessageRole.USER
            and "No tool calls were returned" in (msg.content or "")
        )
        for msg in updated
    )


def test_run_debug_code():
    """Test running debug snippets without the full test suite."""
    helpers_dir = os.path.join(os.path.dirname(__file__), "node_test_helpers")
    toolkit = IterativeTestFixer(helpers_dir)

    service = """
export class McpService {
    ping() {
        return 'pong';
    }
}
"""
    components = ""
    test = "import { describe, it } from 'node:test';"
    dummy_data = "export const dummyData = { value: 123 };"

    toolkit.setup_test_environment(service, components, test, dummy_data)

    result = toolkit.run_debug_code(
        "const svc = new McpService();\n"
        "return { ping: svc.ping(), dummy: dummyData?.value ?? null };"
    )

    assert result.success
    assert "pong" in result.content
    assert "dummy" in result.content

    toolkit.cleanup()


def test_run_debug_code_failure():
    """Test failure scenarios for run_debug_code."""
    helpers_dir = os.path.join(os.path.dirname(__file__), "node_test_helpers")
    toolkit = IterativeTestFixer(helpers_dir)

    service = "export class McpService {}"
    toolkit.setup_test_environment(service, "", "")

    # invalid syntax
    result = toolkit.run_debug_code("this is not valid javascript")
    assert not result.success
    # The error message content is what we are now logging, verifying it exists
    assert "SyntaxError" in result.content

    # runtime error
    result = toolkit.run_debug_code("throw new Error('Custom Error');")
    assert not result.success
    assert "Custom Error" in result.content

    toolkit.cleanup()


def test_search_files():
    """Test searching files with regex."""
    helpers_dir = os.path.join(os.path.dirname(__file__), "node_test_helpers")
    toolkit = IterativeTestFixer(helpers_dir)

    service = "class MyService {\n  method() { return 'value'; }\n}"
    components = "class MyComponent extends HTMLElement {}"
    test = "describe('MyTest', () => {\n  it('checks value', () => {\n    // ...\n  });\n});"

    toolkit.setup_test_environment(service, components, test)

    # Search for "value" in all files
    result = toolkit.search_files("value")
    assert result.success
    # Should find matches in service and test
    assert "service match" in result.content
    assert "test match" in result.content
    assert "MyService" in result.content  # context

    # Search for specific file
    result = toolkit.search_files("class", "components")
    assert result.success
    assert "components match" in result.content
    # Should not find matches in service because we scoped to components
    assert "service match" not in result.content

    # Search with no matches
    result = toolkit.search_files("nonexistent")
    assert result.success
    assert "No matches found" in result.content

    # Invalid regex
    result = toolkit.search_files("[")
    assert not result.success
    assert "Invalid regex" in result.content

    toolkit.cleanup()


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


def test_summarize_test_output_keeps_failure_context():
    failure_line = "not ok 19 - weather-forecast: renders forecast data after fetch"
    long_prefix = "\n".join([f"ok {i} - pass-{i}" for i in range(1, 250)])
    tap_output = (
        "TAP version 13\n"
        f"{long_prefix}\n"
        f"{failure_line}\n"
        "# tests 250\n"
        "# suites 1\n"
        "# pass 249\n"
        "# fail 1\n"
    )
    summarized = _summarize_test_output(tap_output, limit=500)
    assert "...(trimmed" in summarized
    assert failure_line in summarized
    assert "# fail 1" in summarized

    passed, failed, failed_tests = _parse_tap_output(summarized)
    assert passed == 249
    assert failed == 1
    assert any("weather-forecast" in entry for entry in failed_tests)


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
