"""
Lightweight tool framework for iterative test fixing.

This module provides a simple tool execution framework for helping LLMs
fix generated test code through iterative refinement.
"""

import asyncio
import copy
import json
import logging
import os
import re
import contextlib
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from app.app_facade.generated_output_factory import MCP_SERVICE_TEST_HELPER_SCRIPT
from app.vars import GENERATED_UI_READ_ONLY_STREAK_LIMIT, LLM_MAX_PAYLOAD_BYTES
from app.tgi.models import (
    ChatCompletionRequest,
    Message,
    MessageRole,
    ToolCall,
    ToolCallFunction,
)
from app.tgi.services.tools.tool_resolution import ToolResolutionStrategy

logger = logging.getLogger("uvicorn.error")

_SCRIPT_TYPE_TO_FILENAME = {
    "test": "user_test.js",
    "service": "app.js",
    "components": "components.js",
    "dummy_data": "dummy_data.js",
    "dummy-data": "dummy_data.js",
    "domstubs": "domstubs.js",
    "pfusch": "pfusch.js",
}

TOOL_DESCRIPTIONS = {
    "run_tests": lambda args: "running tests",
    "run_debug_code": lambda args: "running debug snippet",
    "update_test_script": lambda args: "updating test code",
    "update_service_script": lambda args: "updating service code",
    "update_components_script": lambda args: "updating component code",
    "update_dummy_data": lambda args: "updating test fixtures",
    "apply_script_patch": lambda args: (
        f"patching {_SCRIPT_TYPE_TO_FILENAME.get(args.get('script_type'), 'script')}"
    ),
    "get_script_lines": lambda args: (
        f"reading source code from {_SCRIPT_TYPE_TO_FILENAME.get(args.get('script_type'), 'unknown')}"
        + (
            f": {args.get('start_line', '?')}:{args.get('end_line', '?')}"
            if args.get("start_line")
            else ""
        )
    ),
    "get_current_scripts": lambda args: "reading source code",
    "get_last_test_output": lambda args: "reading full output from last test run",
    "search_files": lambda args: (
        f"searching for '{args.get('regex', '')}'"
        + (
            f" [in file(s): {args.get('script_type')}]"
            if args.get("script_type")
            else ""
        )
    ),
}


def _positive_int_env(name: str, default: int) -> int:
    try:
        parsed = int(os.environ.get(name, str(default)))
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


NODE_TEST_TIMEOUT_MS = _positive_int_env("GENERATED_UI_NODE_TEST_TIMEOUT_MS", 8000)
READ_ONLY_STREAK_LIMIT = max(1, int(GENERATED_UI_READ_ONLY_STREAK_LIMIT))
FIX_CODE_ASSERTION_BAILOUT_LIMIT = _positive_int_env(
    "GENERATED_UI_FIX_CODE_ASSERTION_BAILOUT_LIMIT", 3
)
ITERATIVE_FIX_MAX_MESSAGES = _positive_int_env("GENERATED_UI_FIX_MAX_MESSAGES", 80)
ITERATIVE_FIX_MAX_MESSAGE_BYTES = _positive_int_env(
    "GENERATED_UI_FIX_MAX_MESSAGE_BYTES", 12000
)
ITERATIVE_FIX_MAX_TOOL_MESSAGE_BYTES = _positive_int_env(
    "GENERATED_UI_FIX_MAX_TOOL_MESSAGE_BYTES", 6000
)
ITERATIVE_FIX_MAX_PAYLOAD_BYTES = _positive_int_env(
    "GENERATED_UI_FIX_MAX_PAYLOAD_BYTES",
    max(10000, int(LLM_MAX_PAYLOAD_BYTES * 0.9)),
)
FOCUS_SINGLE_FAILURE_ENABLED = (
    os.getenv("GENERATED_UI_FOCUS_SINGLE_FAILING_TEST", "true").lower() == "true"
)


def _bytes_len(value: Any) -> int:
    with contextlib.suppress(Exception):
        return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
    return len(str(value).encode("utf-8"))


def _truncate_text_for_bytes(text: str, max_bytes: int) -> str:
    if not isinstance(text, str):
        text = str(text)
    if max_bytes <= 0:
        return ""
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    suffix = "...[truncated]"
    suffix_bytes = len(suffix.encode("utf-8"))
    budget = max(0, max_bytes - suffix_bytes)
    low = 0
    high = len(text)
    while low < high:
        mid = (low + high + 1) // 2
        candidate = text[:mid]
        if len(candidate.encode("utf-8")) <= budget:
            low = mid
        else:
            high = mid - 1
    return text[:low] + suffix


def _message_role_value(message: Message) -> str:
    role = getattr(message, "role", "")
    if hasattr(role, "value"):
        return str(role.value)
    return str(role)


def _message_content_text(message: Message) -> str:
    content = getattr(message, "content", "")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


def _copy_message_with_content(message: Message, content: str) -> Message:
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"content": content})
    cloned = copy.deepcopy(message)
    cloned.content = content
    return cloned


def _messages_payload_bytes(messages: List[Message]) -> int:
    return sum(_bytes_len(_message_content_text(message)) for message in messages)


def _collect_payload_diagnostics(
    messages: List[Message], tools: List[Dict[str, Any]]
) -> Dict[str, Any]:
    top_messages: List[Dict[str, Any]] = []
    for index, message in enumerate(messages):
        text = _message_content_text(message)
        top_messages.append(
            {
                "index": index,
                "role": _message_role_value(message),
                "bytes": _bytes_len(text),
            }
        )
    top_messages.sort(key=lambda item: item["bytes"], reverse=True)

    section_sizes: List[Dict[str, Any]] = []
    for index, message in enumerate(messages):
        text = _message_content_text(message).strip()
        if not text.startswith("{"):
            continue
        with contextlib.suppress(Exception):
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    section_sizes.append(
                        {
                            "index": index,
                            "role": _message_role_value(message),
                            "section": key,
                            "bytes": _bytes_len(value),
                        }
                    )
    section_sizes.sort(key=lambda item: item["bytes"], reverse=True)

    return {
        "message_count": len(messages),
        "messages_bytes": _messages_payload_bytes(messages),
        "tools_count": len(tools or []),
        "tools_bytes": _bytes_len(tools or []),
        "top_messages": top_messages[:6],
        "top_sections": section_sizes[:8],
    }


def _cap_fix_messages_for_request(
    messages: List[Message],
    *,
    max_messages: int,
    max_total_bytes: int,
    max_message_bytes: int,
    max_tool_message_bytes: int,
) -> Tuple[List[Message], Dict[str, Any]]:
    capped: List[Message] = []
    trimmed_messages = 0
    dropped_messages = 0
    original_bytes = _messages_payload_bytes(messages)

    for message in messages:
        content = _message_content_text(message)
        role = _message_role_value(message)
        per_message_limit = (
            max_tool_message_bytes
            if role == MessageRole.TOOL.value
            else max_message_bytes
        )
        if _bytes_len(content) > per_message_limit:
            content = _truncate_text_for_bytes(content, per_message_limit)
            trimmed_messages += 1
        capped.append(_copy_message_with_content(message, content))

    has_system_prefix = (
        bool(capped) and _message_role_value(capped[0]) == MessageRole.SYSTEM.value
    )
    while len(capped) > max_messages:
        drop_index = 1 if has_system_prefix and len(capped) > 1 else 0
        capped.pop(drop_index)
        dropped_messages += 1
        has_system_prefix = (
            bool(capped) and _message_role_value(capped[0]) == MessageRole.SYSTEM.value
        )

    while len(capped) > 1 and _messages_payload_bytes(capped) > max_total_bytes:
        has_system_prefix = (
            bool(capped) and _message_role_value(capped[0]) == MessageRole.SYSTEM.value
        )
        drop_index = None
        start = 1 if has_system_prefix else 0
        for index in range(start, len(capped)):
            if _message_role_value(capped[index]) == MessageRole.TOOL.value:
                drop_index = index
                break
        if drop_index is None:
            drop_index = 1 if has_system_prefix and len(capped) > 1 else 0
        capped.pop(drop_index)
        dropped_messages += 1

    stats = {
        "original_messages": len(messages),
        "final_messages": len(capped),
        "original_bytes": original_bytes,
        "final_bytes": _messages_payload_bytes(capped),
        "trimmed_messages": trimmed_messages,
        "dropped_messages": dropped_messages,
    }
    return capped, stats


def _tool_description(
    tool_name: str, arguments: Optional[Dict[str, Any]] = None
) -> str:
    handler = TOOL_DESCRIPTIONS.get(tool_name)
    if callable(handler):
        try:
            return handler(arguments or {})
        except Exception:
            return tool_name
    return tool_name


def _extract_fix_explanation(
    content: Optional[str], max_len: int = 600
) -> Optional[str]:
    """
    Extract a concise fix explanation from assistant text.

    Expected format in assistant content:
      FIX_EXPLANATION: <why this change should fix tests>
    """
    text = (content or "").strip()
    if not text:
        return None

    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-* ").strip()
        if not line:
            continue
        if line.lower().startswith("fix_explanation:"):
            explanation = line.split(":", 1)[1].strip()
            if not explanation:
                return None
            if len(explanation) > max_len:
                return f"{explanation[:max_len]}...(trimmed {len(explanation) - max_len} chars)"
            return explanation
    return None


def _extract_general_explanation(
    content: Optional[str], max_len: int = 600
) -> Optional[str]:
    """
    Best-effort extraction for models that do not emit FIX_EXPLANATION.
    """
    text = (content or "").strip()
    if not text:
        return None

    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-* ").strip()
        if not line:
            continue
        lower = line.lower()
        if lower in {"tests_passing", "test_passing"}:
            continue
        if lower.startswith("fix_explanation:"):
            continue
        if lower.startswith("```") or lower.startswith("{") or lower.startswith("["):
            continue
        if len(line) > max_len:
            return f"{line[:max_len]}...(trimmed {len(line) - max_len} chars)"
        return line
    return None


def _inferred_tool_explanation(tool_names: List[str]) -> Optional[str]:
    """
    Fallback explanation when assistant content has no explicit rationale.
    """
    if not tool_names:
        return None

    unique_names = []
    for name in tool_names:
        if name and name not in unique_names:
            unique_names.append(name)

    if not unique_names:
        return None

    if all(
        name in {"get_script_lines", "get_current_scripts", "search_files"}
        for name in unique_names
    ):
        return "Inspecting source context before applying a change."

    mapping = {
        "run_tests": "Re-running tests to validate current scripts.",
        "run_debug_code": "Running a debug snippet to inspect runtime behavior.",
        "update_test_script": "Updating test logic to match intended behavior and remove failing assumptions.",
        "update_service_script": "Updating service behavior to satisfy test expectations.",
        "update_components_script": "Updating component rendering/state flow to satisfy test expectations.",
        "update_dummy_data": "Updating fixtures to match expected payload shapes.",
        "get_script_lines": "Reading specific source lines for diagnosis.",
        "get_current_scripts": "Inspecting script metadata before making edits.",
        "search_files": "Searching source for relevant patterns related to the failure.",
    }

    reasons = []
    for name in unique_names[:2]:
        reason = mapping.get(name)
        if reason:
            reasons.append(reason)
    if not reasons:
        return (
            f"Applying tool-driven diagnosis/fix steps: {', '.join(unique_names[:2])}."
        )
    return " ".join(reasons)


def _extract_bail_override(content: Optional[str], max_len: int = 600) -> Optional[str]:
    """
    Extract explicit bailout override from assistant text.

    Expected format in assistant content:
      BAIL_OVERRIDE: <runtime hypothesis>
    """
    text = (content or "").strip()
    if not text:
        return None

    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-* ").strip()
        if not line:
            continue
        if line.lower().startswith("bail_override:"):
            override = line.split(":", 1)[1].strip()
            if not override:
                return None
            if len(override) > max_len:
                return (
                    f"{override[:max_len]}...(trimmed {len(override) - max_len} chars)"
                )
            return override
    return None


def _parse_tap_output(tap_output: str) -> Tuple[int, int, List[str]]:
    passed = 0
    failed = 0
    failed_tests = []

    for line in tap_output.split("\n"):
        line = line.strip()
        if line.startswith("ok "):
            passed += 1
        elif line.startswith("not ok "):
            failed += 1
            match = re.search(r"not ok \d+\s*(?:-\s*)?(.+?)(?:\s*#|$)", line)
            if match:
                failed_tests.append(match.group(1).strip())

    # Check for summary lines (e.g. from node --test) and prefer them if present
    summary_pass = re.search(r"^# pass\s+(\d+)", tap_output, re.MULTILINE)
    if summary_pass:
        passed = int(summary_pass.group(1))

    summary_fail = re.search(r"^# fail\s+(\d+)", tap_output, re.MULTILINE)
    if summary_fail:
        failed = int(summary_fail.group(1))

    return passed, failed, failed_tests


def _summarize_test_output(output: str, limit: int = 4000) -> str:
    """
    Trim oversized test output while preserving failing test names and summary lines.
    """
    text = output or ""
    if len(text) <= limit:
        return text

    failure_lines: List[str] = []
    seen_failures = set()
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("not ok ") and line not in seen_failures:
            seen_failures.add(line)
            failure_lines.append(line)

    summary_lines: List[str] = []
    for pattern in (
        r"^# tests\s+\d+",
        r"^# suites\s+\d+",
        r"^# pass\s+\d+",
        r"^# fail\s+\d+",
    ):
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            summary_lines.append(match.group(0))

    prefix_parts = [f"...(trimmed {len(text) - limit} chars)"]
    if failure_lines:
        preview = failure_lines[:8]
        if len(failure_lines) > 8:
            preview.append(f"... ({len(failure_lines) - 8} more failing entries)")
        prefix_parts.append(
            "Failed tests:\n" + "\n".join(f"- {line}" for line in preview)
        )
    if summary_lines:
        prefix_parts.append("Summary:\n" + "\n".join(summary_lines))

    prefix = "\n".join(prefix_parts).strip() + "\n\n"
    remaining = max(800, limit - len(prefix))
    head_budget = max(200, remaining // 3)
    tail_budget = max(400, remaining - head_budget)
    head = text[:head_budget]
    tail = text[-tail_budget:]
    return f"{prefix}" f"--- OUTPUT HEAD ---\n{head}\n" f"--- OUTPUT TAIL ---\n{tail}"


def _log_test_run_output(context: str, output: str) -> None:
    output = output or ""
    if not output.strip():
        logger.info("%s Test run produced no output", context)
        return
    logger.info("%s Test run output:\n%s", context, _summarize_test_output(output))


@dataclass
class ToolDefinition:
    """Definition of a tool that can be invoked by the LLM."""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable


@dataclass
class ToolResult:
    """Result from a tool execution."""

    success: bool
    content: str
    metadata: Optional[Dict[str, Any]] = None


class IterativeTestFixer:
    """Toolkit for fixing generated test code iteratively."""

    def __init__(self, helpers_dir: str):
        """
        Initialize the toolkit.

        Args:
            helpers_dir: Path to node_test_helpers directory
        """
        self.helpers_dir = helpers_dir
        self.current_service_script: Optional[str] = None
        self.current_components_script: Optional[str] = None
        self.current_test_script: Optional[str] = None
        self.current_dummy_data: Optional[str] = None
        self.tmpdir: Optional[str] = None
        self.focus_test_name: Optional[str] = None
        self.last_test_output_raw: Optional[str] = None
        self.last_test_output_summarized: Optional[str] = None
        self.last_test_metadata: Optional[Dict[str, Any]] = None

    def set_focus_test_name(self, test_name: Optional[str]) -> None:
        """Set or clear the focused failing test name used for targeted prechecks."""
        normalized = (test_name or "").strip()
        self.focus_test_name = normalized or None

    def setup_test_environment(
        self,
        service_script: str,
        components_script: str,
        test_script: str,
        dummy_data: Optional[str] = None,
    ) -> None:
        """Setup the test environment with current scripts."""
        self.current_service_script = service_script
        self.current_components_script = components_script
        self.current_test_script = test_script
        self.current_dummy_data = dummy_data

        # Create temporary directory for testing
        if self.tmpdir:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
        self.tmpdir = tempfile.mkdtemp()

        # Copy helpers
        for filename in ["domstubs.js", "pfusch.js"]:
            src = os.path.join(self.helpers_dir, filename)
            dst = os.path.join(self.tmpdir, filename)
            if os.path.exists(src):
                shutil.copy(src, dst)

        # Write package.json
        with open(
            os.path.join(self.tmpdir, "package.json"), "w", encoding="utf-8"
        ) as f:
            f.write('{"type":"module"}\n')

    def cleanup(self) -> None:
        """Clean up temporary directory."""
        if self.tmpdir:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            self.tmpdir = None

    def _normalize_script_type(self, script_type: Optional[str]) -> Optional[str]:
        if not script_type:
            return script_type
        return script_type.strip().replace("-", "_")

    def _line_count(self, script: Optional[Any]) -> int:
        if script is None:
            return 0
        if not isinstance(script, str):
            script = str(script)
        return len(script.split("\n"))

    def _write_current_files(self) -> None:
        """Write current scripts to temporary directory."""
        if not self.tmpdir:
            raise RuntimeError("Test environment not set up")

        def _normalize_pfusch_imports(source: Optional[str]) -> str:
            text = source or ""
            return (
                text.replace(
                    "https://matthiaskainer.github.io/pfusch/pfusch.min.js",
                    "./pfusch.js",
                )
                .replace(
                    "https://matthiaskainer.github.io/pfusch/pfusch.js",
                    "./pfusch.js",
                )
                .replace(
                    "https://matthiaskainer.github.io/pfusch/pfusch.min.mjs",
                    "./pfusch.js",
                )
            )

        # Write combined service + components script to mirror production bundling.
        mocked_service = _normalize_pfusch_imports(self.current_service_script)
        mocked_components = _normalize_pfusch_imports(self.current_components_script)
        combined_script = (
            f"{MCP_SERVICE_TEST_HELPER_SCRIPT}\n\n"
            + mocked_service
            + "\n\n"
            + mocked_components
        )
        with open(os.path.join(self.tmpdir, "app.js"), "w", encoding="utf-8") as f:
            f.write(combined_script)
        with open(
            os.path.join(self.tmpdir, "components.js"), "w", encoding="utf-8"
        ) as f:
            f.write("import './app.js';\n")

        # Write user test
        with open(
            os.path.join(self.tmpdir, "user_test.js"), "w", encoding="utf-8"
        ) as f:
            f.write(_normalize_pfusch_imports(self.current_test_script))

        # Write dummy data module if provided
        if self.current_dummy_data is not None:
            with open(
                os.path.join(self.tmpdir, "dummy_data.js"), "w", encoding="utf-8"
            ) as f:
                f.write(self.current_dummy_data or "")

        # Write test wrapper
        test_wrapper = (
            "import { setupDomStubs, pfuschTest } from './domstubs.js';\n"
            "if (typeof globalThis.HTMLElement === 'undefined') {\n"
            "  setupDomStubs();\n"
            "}\n"
            "await import('./user_test.js');\n"
        )
        with open(os.path.join(self.tmpdir, "test.js"), "w", encoding="utf-8") as f:
            f.write(test_wrapper)

    def run_tests(self, test_name: Optional[str] = None) -> ToolResult:
        """
        Run the test suite or a single test.

        Args:
            test_name: Optional name of a single test to run

        Returns:
            ToolResult with test output
        """
        try:
            self._write_current_files()

            def _run_node_tests_once(pattern: Optional[str]) -> ToolResult:
                cmd = [
                    "node",
                    "--test",
                    "--test-force-exit",
                    "--test-timeout",
                    str(NODE_TEST_TIMEOUT_MS),
                    "test.js",
                ]
                if pattern:
                    cmd.extend(["--test-name-pattern", pattern])

                env = os.environ.copy()
                env["NODE_PATH"] = self.tmpdir

                logger.info(
                    "[iterative_test_fix] Running node tests%s",
                    f" (pattern: {pattern})" if pattern else "",
                )
                start = time.monotonic()
                result = subprocess.run(
                    cmd,
                    cwd=self.tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=env,
                )
                duration = time.monotonic() - start
                logger.info(
                    "[iterative_test_fix] Node tests completed with code %s in %.2fs",
                    result.returncode,
                    duration,
                )

                output = result.stdout + "\n" + result.stderr
                success = result.returncode == 0
                metadata = self._parse_tap_output(output)
                metadata["test_name_pattern"] = pattern
                self.last_test_output_raw = output
                self.last_test_metadata = metadata

                if not success:
                    output = _summarize_test_output(output)
                self.last_test_output_summarized = output

                return ToolResult(
                    success=success,
                    content=output,
                    metadata=metadata,
                )

            if test_name:
                return _run_node_tests_once(test_name)

            focused_name = self.focus_test_name
            if focused_name:
                focused_result = _run_node_tests_once(focused_name)
                focused_metadata = focused_result.metadata or {}
                focused_metadata["focus_precheck"] = True
                focused_result.metadata = focused_metadata
                if not focused_result.success:
                    return focused_result

                logger.info(
                    "[iterative_test_fix] Focused precheck passed for '%s'; running full suite",
                    focused_name,
                )
                full_result = _run_node_tests_once(None)
                full_metadata = full_result.metadata or {}
                full_metadata["focus_precheck"] = True
                full_metadata["focus_precheck_pattern"] = focused_name
                full_metadata["focus_precheck_passed"] = True
                full_result.metadata = full_metadata
                return full_result

            return _run_node_tests_once(None)

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                content=(
                    "Tests timed out after 30 seconds. "
                    "Check for unresolved async waits (for example: await new Promise(...), "
                    "while(true), or setInterval without cleanup)."
                ),
            )
        except Exception as e:
            return ToolResult(success=False, content=f"Error running tests: {str(e)}")

    def run_debug_code(
        self, code: str, timeout_seconds: Optional[int] = None
    ) -> ToolResult:
        """
        Run a debug snippet in the current test environment.

        Args:
            code: JavaScript snippet to run (no import/export statements)
            timeout_seconds: Optional timeout in seconds (default: 10)

        Returns:
            ToolResult with debug output
        """
        try:
            if code is None:
                return ToolResult(success=False, content="Debug code is required.")
            if not isinstance(code, str):
                code = str(code)

            try:
                timeout = 10 if timeout_seconds is None else int(timeout_seconds)
            except (TypeError, ValueError):
                return ToolResult(
                    success=False,
                    content=(
                        f"Invalid timeout: {timeout_seconds} "
                        "(must be a positive integer)"
                    ),
                )
            if timeout <= 0:
                return ToolResult(
                    success=False,
                    content="Timeout must be a positive integer.",
                )

            self._write_current_files()

            debug_wrapper = "\n".join(
                [
                    "import { setupDomStubs, pfuschTest } from './domstubs.js';",
                    "if (typeof globalThis.HTMLElement === 'undefined') {",
                    "  setupDomStubs();",
                    "}",
                    "const { McpService } = await import('./app.js');",
                    "await import('./components.js');",
                    "let dummyData = null;",
                    "try {",
                    "  const mod = await import('./dummy_data.js');",
                    "  dummyData = mod.dummyData ?? mod.default ?? mod;",
                    "} catch (err) {",
                    "  dummyData = null;",
                    "}",
                    "// Debug code starts here",
                    "const __result = await (async () => {",
                    code,
                    "})();",
                    "if (typeof __result !== 'undefined') {",
                    "  console.log(JSON.stringify({ result: __result }, null, 2));",
                    "}",
                    "",
                ]
            )

            debug_path = os.path.join(self.tmpdir, "debug.js")
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(debug_wrapper)

            env = os.environ.copy()
            env["NODE_PATH"] = self.tmpdir

            result = subprocess.run(
                ["node", "debug.js"],
                cwd=self.tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

            output = result.stdout + "\n" + result.stderr
            success = result.returncode == 0

            if len(output) > 4000:
                output = f"...(trimmed {len(output) - 4000} chars)\n{output[-4000:]}"

            return ToolResult(
                success=success,
                content=output,
                metadata={
                    "exit_code": result.returncode,
                    "timeout_seconds": timeout,
                },
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                content=f"Debug run timed out after {timeout} seconds.",
            )
        except Exception as e:
            return ToolResult(
                success=False, content=f"Error running debug code: {str(e)}"
            )

    def _parse_tap_output(self, output: str) -> Dict[str, Any]:
        """Parse TAP output to extract test statistics."""
        metadata = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "suites": 0,
        }

        # Look for summary lines like "# tests 10", "# pass 7", "# fail 3"
        for match in re.finditer(r"# (\w+) (\d+)", output):
            key = match.group(1)
            value = int(match.group(2))
            if key == "tests":
                metadata["total"] = value
            elif key == "pass":
                metadata["passed"] = value
            elif key == "fail":
                metadata["failed"] = value
            elif key == "suites":
                metadata["suites"] = value

        return metadata

    def get_last_test_output(self, max_bytes: Optional[int] = None) -> ToolResult:
        """
        Return full output from the most recent run_tests invocation.

        Args:
            max_bytes: Optional byte cap for returned text

        Returns:
            ToolResult with the last captured output
        """
        if self.last_test_output_raw is None:
            return ToolResult(
                success=False,
                content="No test output captured yet. Run run_tests first.",
            )

        content = self.last_test_output_raw
        if max_bytes is not None:
            try:
                cap = int(max_bytes)
            except (TypeError, ValueError):
                return ToolResult(
                    success=False,
                    content=(
                        f"Invalid max_bytes value: {max_bytes} "
                        "(must be a positive integer)"
                    ),
                )
            if cap <= 0:
                return ToolResult(
                    success=False,
                    content="max_bytes must be a positive integer.",
                )
            content = _truncate_text_for_bytes(content, cap)

        return ToolResult(
            success=True,
            content=content,
            metadata=self.last_test_metadata or {},
        )

    def apply_script_patch(
        self,
        script_type: str,
        search: str,
        replace: str,
        replace_all: bool = False,
        use_regex: bool = False,
    ) -> ToolResult:
        """
        Apply a targeted patch to a mutable script using literal or regex replacement.
        """
        script_type = self._normalize_script_type(script_type)
        if script_type in {"domstubs", "pfusch"}:
            return ToolResult(
                success=False,
                content=f"{script_type} is read-only and cannot be patched.",
            )

        script_map: Dict[str, Optional[str]] = {
            "test": self.current_test_script,
            "service": self.current_service_script,
            "components": self.current_components_script,
            "dummy_data": self.current_dummy_data,
        }
        current = script_map.get(script_type)
        if current is None:
            return ToolResult(
                success=False,
                content=f"Unknown or unavailable mutable script type: {script_type}",
            )

        if not isinstance(current, str):
            current = str(current)

        if use_regex:
            try:
                pattern = re.compile(search, re.MULTILINE)
            except re.error as exc:
                return ToolResult(
                    success=False,
                    content=f"Invalid regex pattern: {exc}",
                )
            if replace_all:
                updated, count = pattern.subn(replace, current)
            else:
                updated, count = pattern.subn(replace, current, count=1)
        else:
            count = current.count(search)
            if count == 0:
                updated = current
            elif replace_all:
                updated = current.replace(search, replace)
            else:
                updated = current.replace(search, replace, 1)
                count = 1

        if count == 0:
            return ToolResult(
                success=False,
                content="No matches found for patch search pattern.",
            )

        if script_type == "test":
            self.current_test_script = updated
        elif script_type == "service":
            self.current_service_script = updated
        elif script_type == "components":
            self.current_components_script = updated
        elif script_type == "dummy_data":
            self.current_dummy_data = updated

        return ToolResult(
            success=True,
            content=(
                f"Patched {script_type} successfully. Replacements applied: {count}."
            ),
            metadata={"replacements": count, "script_type": script_type},
        )

    def update_test_script(self, new_script: str) -> ToolResult:
        """
        Update the test script.

        Args:
            new_script: New test script content

        Returns:
            ToolResult confirming the update
        """
        try:
            self.current_test_script = new_script
            return ToolResult(
                success=True,
                content="Test script updated successfully.",
                metadata={"length": len(new_script)},
            )
        except Exception as e:
            return ToolResult(success=False, content=f"Error updating test: {str(e)}")

    def update_service_script(self, new_script: str) -> ToolResult:
        """
        Update the service script.

        Args:
            new_script: New service script content

        Returns:
            ToolResult confirming the update
        """
        try:
            self.current_service_script = new_script
            return ToolResult(
                success=True,
                content="Service script updated successfully.",
                metadata={"length": len(new_script)},
            )
        except Exception as e:
            return ToolResult(
                success=False, content=f"Error updating service: {str(e)}"
            )

    def update_components_script(self, new_script: str) -> ToolResult:
        """
        Update the components script.

        Args:
            new_script: New components script content

        Returns:
            ToolResult confirming the update
        """
        try:
            self.current_components_script = new_script
            return ToolResult(
                success=True,
                content="Components script updated successfully.",
                metadata={"length": len(new_script)},
            )
        except Exception as e:
            return ToolResult(
                success=False, content=f"Error updating components: {str(e)}"
            )

    def _get_helper_content(self, filename: str) -> Optional[str]:
        """Read content of a helper file."""
        path = os.path.join(self.helpers_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def get_script_lines(
        self,
        script_type: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> ToolResult:
        """
        Get specific lines from a script.

        Args:
            script_type: Type of script ('test', 'service', 'components', 'dummy_data', 'domstubs', 'pfusch')
            start_line: Optional starting line number (1-indexed)
            end_line: Optional ending line number (inclusive)

        Returns:
            ToolResult with the requested lines
        """
        try:
            script_type = self._normalize_script_type(script_type)
            script_map = {
                "test": self.current_test_script,
                "service": self.current_service_script,
                "components": self.current_components_script,
                "dummy_data": self.current_dummy_data,
                "domstubs": self._get_helper_content("domstubs.js"),
                "pfusch": self._get_helper_content("pfusch.js"),
            }

            script = script_map.get(script_type)
            if script is None:
                return ToolResult(
                    success=False,
                    content=f"Unknown script type or content not available: {script_type}",
                )

            if not isinstance(script, str):
                script = str(script)
            lines = script.split("\n")
            total_lines = len(lines)

            try:
                start_line = 1 if start_line is None else int(start_line)
                end_line = total_lines if end_line is None else int(end_line)
            except (TypeError, ValueError):
                return ToolResult(
                    success=False,
                    content=(
                        f"Invalid line range: {start_line}-{end_line} "
                        "(line numbers must be integers)"
                    ),
                )
            if start_line < 1 or end_line > len(lines) or start_line > end_line:
                return ToolResult(
                    success=False,
                    content=(
                        f"Invalid line range: {start_line}-{end_line} "
                        f"(script has {total_lines} lines)"
                    ),
                )

            # Convert to 0-indexed
            selected_lines = lines[start_line - 1 : end_line]
            content = "\n".join(selected_lines)

            # Map logical script types to filenames used by the test runner
            filename_map = {
                "test": "user_test.js",
                "service": "app.js",
                "components": "components.js",
                "dummy_data": "dummy_data.js",
                "domstubs": os.path.join(self.helpers_dir, "domstubs.js"),
                "pfusch": os.path.join(self.helpers_dir, "pfusch.js"),
            }
            filename = filename_map.get(script_type, f"{script_type}.js")

            return ToolResult(
                success=True,
                content=content,
                metadata={
                    "filename": filename,
                    "start_line": start_line,
                    "end_line": end_line,
                    "lines": [start_line, end_line],
                    "total_lines": total_lines,
                },
            )

        except Exception as e:
            return ToolResult(success=False, content=f"Error reading lines: {str(e)}")

    def get_current_scripts(self) -> ToolResult:
        """
        Get metadata about current scripts.

        Returns:
            ToolResult with script metadata
        """
        try:
            domstubs = self._get_helper_content("domstubs.js") or ""
            pfusch = self._get_helper_content("pfusch.js") or ""

            metadata = {
                "service_lines": self._line_count(self.current_service_script),
                "components_lines": self._line_count(self.current_components_script),
                "test_lines": self._line_count(self.current_test_script),
                "dummy_data_lines": self._line_count(self.current_dummy_data),
                "domstubs_lines": self._line_count(domstubs),
                "pfusch_lines": self._line_count(pfusch),
                "read_only": ["domstubs", "pfusch"],
            }

            content = json.dumps(metadata, indent=2)
            return ToolResult(success=True, content=content, metadata=metadata)

        except Exception as e:
            return ToolResult(
                success=False, content=f"Error getting metadata: {str(e)}"
            )

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI-compatible tool definitions for use in chat completions.

        Returns:
            List of tool definitions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "apply_script_patch",
                    "description": (
                        "Apply a targeted edit to one mutable script using literal "
                        "or regex replacement. Prefer this for small surgical changes "
                        "instead of rewriting whole files."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "script_type": {
                                "type": "string",
                                "enum": [
                                    "test",
                                    "service",
                                    "components",
                                    "dummy_data",
                                    "dummy-data",
                                ],
                                "description": "Which mutable script to patch",
                            },
                            "search": {
                                "type": "string",
                                "description": "Search pattern (literal by default)",
                            },
                            "replace": {
                                "type": "string",
                                "description": "Replacement text",
                            },
                            "replace_all": {
                                "type": "boolean",
                                "description": "Whether to replace all matches (default false)",
                            },
                            "use_regex": {
                                "type": "boolean",
                                "description": "Interpret search as regex (default false)",
                            },
                        },
                        "required": ["script_type", "search", "replace"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_last_test_output",
                    "description": (
                        "Return full output from the most recent run_tests call. "
                        "Useful when summarized output hides important stack/context lines."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "max_bytes": {
                                "type": "integer",
                                "description": "Optional byte cap for returned output",
                            }
                        },
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_tests",
                    "description": (
                        "Run the complete test suite or a single test by name. "
                        "Returns TAP output with test results including pass/fail counts."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "test_name": {
                                "type": "string",
                                "description": (
                                    "Optional. Name pattern of a specific test to run. "
                                    "If omitted, runs all tests."
                                ),
                            }
                        },
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_debug_code",
                    "description": (
                        "Run a short JavaScript snippet in the current test environment "
                        "without executing the full test suite. The snippet runs inside "
                        "an async function after DOM stubs are set up and the service/"
                        "components are loaded. Available bindings: McpService, pfuschTest, "
                        "dummyData (if provided). Return a value to print JSON output, "
                        "and console.log output is captured. Prefer to run the tests "
                        "directly using 'run_tests' when possible, only use if stuck."
                        "You're inside a closure, so don't do top level things but rather "
                        "ie `const { dummyData } = await import('./dummy_data.js');`"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": (
                                    "JavaScript snippet to run. Do not include "
                                    "import/export statements."
                                ),
                            },
                            "timeout_seconds": {
                                "type": "integer",
                                "description": (
                                    "Optional. Timeout in seconds (default 10)."
                                ),
                            },
                        },
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_test_script",
                    "description": (
                        "Replace the entire test script with new content. "
                        "Use this to fix failing tests."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "new_script": {
                                "type": "string",
                                "description": "Complete new test script content",
                            }
                        },
                        "required": ["new_script"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_service_script",
                    "description": (
                        "Replace the entire service script with new content. "
                        "Use this if the service implementation needs fixes."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "new_script": {
                                "type": "string",
                                "description": "Complete new service script content",
                            }
                        },
                        "required": ["new_script"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_components_script",
                    "description": (
                        "Replace the entire components script with new content. "
                        "Use this if component code needs fixes."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "new_script": {
                                "type": "string",
                                "description": "Complete new components script content",
                            }
                        },
                        "required": ["new_script"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_dummy_data",
                    "description": (
                        "Replace the entire dummy data module with new content. "
                        "Use this when test fixtures need updates."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "new_script": {
                                "type": "string",
                                "description": "Complete new dummy data module content",
                            }
                        },
                        "required": ["new_script"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_script_lines",
                    "description": (
                        "Read specific lines from a script file for inspection. "
                        "Useful for examining code before making changes. "
                        "Can also read read-only helper scripts 'domstubs' and 'pfusch'. "
                        "If line numbers are omitted, returns the entire file."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "script_type": {
                                "type": "string",
                                "enum": [
                                    "test",
                                    "service",
                                    "components",
                                    "dummy_data",
                                    "dummy-data",
                                    "domstubs",
                                    "pfusch",
                                ],
                                "description": "Which script to read from",
                            },
                            "start_line": {
                                "type": "integer",
                                "description": "Optional. Starting line number (1-indexed). Defaults to the first line.",
                            },
                            "end_line": {
                                "type": "integer",
                                "description": "Optional. Ending line number (inclusive). Defaults to the last line.",
                            },
                        },
                        "required": ["script_type"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": (
                        "Search for a regex pattern across one or all files. "
                        "Returns matching lines with context (2 lines before/after)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "regex": {
                                "type": "string",
                                "description": "Regular expression to search for",
                            },
                            "script_type": {
                                "type": "string",
                                "enum": [
                                    "test",
                                    "service",
                                    "components",
                                    "dummy_data",
                                    "dummy-data",
                                    "domstubs",
                                    "pfusch",
                                ],
                                "description": "Optional. Specific script to search. If omitted, searches all.",
                            },
                        },
                        "required": ["regex"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_scripts",
                    "description": (
                        "Get metadata about current scripts including line counts. "
                        "Useful for understanding script size before reading."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name with given arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments for the tool

        Returns:
            ToolResult from the tool execution
        """
        tool_map = {
            "apply_script_patch": lambda: self.apply_script_patch(
                arguments["script_type"],
                arguments["search"],
                arguments["replace"],
                bool(arguments.get("replace_all", False)),
                bool(arguments.get("use_regex", False)),
            ),
            "run_tests": lambda: self.run_tests(arguments.get("test_name")),
            "get_last_test_output": lambda: self.get_last_test_output(
                arguments.get("max_bytes")
            ),
            "run_debug_code": lambda: self.run_debug_code(
                arguments.get("code"), arguments.get("timeout_seconds")
            ),
            "update_test_script": lambda: self.update_test_script(
                arguments["new_script"]
            ),
            "update_service_script": lambda: self.update_service_script(
                arguments["new_script"]
            ),
            "update_components_script": lambda: self.update_components_script(
                arguments["new_script"]
            ),
            "update_dummy_data": lambda: self.update_dummy_data(
                arguments["new_script"]
            ),
            "get_script_lines": lambda: self.get_script_lines(
                arguments["script_type"],
                arguments.get("start_line"),
                arguments.get("end_line"),
            ),
            "get_current_scripts": lambda: self.get_current_scripts(),
            "search_files": lambda: self.search_files(
                arguments["regex"], arguments.get("script_type")
            ),
        }

        handler = tool_map.get(tool_name)
        if not handler:
            return ToolResult(success=False, content=f"Unknown tool: {tool_name}")

        try:
            return handler()
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}")
            return ToolResult(success=False, content=f"Tool execution error: {str(e)}")

    def update_dummy_data(self, new_script: str) -> ToolResult:
        """
        Update the dummy data module.

        Args:
            new_script: New dummy data module content

        Returns:
            ToolResult confirming the update
        """
        try:
            self.current_dummy_data = new_script
            return ToolResult(
                success=True,
                content="Dummy data module updated successfully.",
                metadata={"length": len(new_script)},
            )
        except Exception as e:
            return ToolResult(
                success=False, content=f"Error updating dummy data: {str(e)}"
            )

    def search_files(self, regex: str, script_type: Optional[str] = None) -> ToolResult:
        """
        Search for a regex pattern in files.

        Args:
            regex: Regular expression pattern
            script_type: Optional script type to restrict search

        Returns:
            ToolResult with matches
        """
        try:
            try:
                pattern = re.compile(regex, re.MULTILINE)
            except re.error as e:
                return ToolResult(success=False, content=f"Invalid regex: {str(e)}")

            script_type = self._normalize_script_type(script_type)
            script_map = {
                "test": self.current_test_script,
                "service": self.current_service_script,
                "components": self.current_components_script,
                "dummy_data": self.current_dummy_data,
                "domstubs": self._get_helper_content("domstubs.js"),
                "pfusch": self._get_helper_content("pfusch.js"),
            }

            results = []
            files_to_search = (
                [script_type] if script_type in script_map else script_map.keys()
            )

            for name in files_to_search:
                content = script_map.get(name)
                if not content:
                    continue

                if not isinstance(content, str):
                    content = str(content)
                lines = content.split("\n")

                # Find all matches
                for i, line in enumerate(lines):
                    if pattern.search(line):
                        # Context: 2 lines before and after
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)

                        context_lines = []
                        for j in range(start, end):
                            prefix = "> " if j == i else "  "
                            context_lines.append(f"{j+1:4d} | {prefix}{lines[j]}")

                        results.append(f"--- {name} match at line {i+1} ---")
                        results.append("\n".join(context_lines))
                        results.append("")

            if not results:
                return ToolResult(
                    success=True,
                    content=f"No matches found for pattern '{regex}'",
                    metadata={"count": 0},
                )

            return ToolResult(
                success=True,
                content="\n".join(results),
                metadata={"count": len(results) // 3},  # Approximate count
            )

        except Exception as e:
            return ToolResult(success=False, content=f"Search error: {str(e)}")


async def run_tool_driven_test_fix(
    *,
    tgi_service: Any,
    service_script: str,
    components_script: str,
    test_script: str,
    dummy_data: Optional[str],
    messages: List[Message],
    allowed_tools: Optional[List[Dict[str, Any]]],
    access_token: Optional[str],
    max_attempts: int = 25,
    event_queue: Optional[asyncio.Queue] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    strategy_mode: Literal["default", "fix_code", "adjust_test"] = "default",
    post_success_validator: Optional[
        Callable[[str, str, str, Optional[str]], Tuple[bool, Optional[str]]]
    ] = None,
) -> Tuple[bool, str, str, str, Optional[str], List[Message]]:
    """
    Use LLM with specialized tools to iteratively fix failing tests.

    Returns:
        Tuple of (success, service_script, components_script, test_script, dummy_data, updated_messages)
    """
    helpers_dir = os.path.join(os.path.dirname(__file__), "node_test_helpers")
    toolkit = IterativeTestFixer(helpers_dir)
    tool_resolution = ToolResolutionStrategy()
    if getattr(tgi_service, "llm_client", None) and getattr(
        tgi_service.llm_client, "model_format", None
    ):
        tool_resolution = (
            tgi_service.llm_client.model_format.create_tool_resolution_strategy()
        )
    require_tool_calls = False
    no_tool_call_attempts = 0

    try:
        toolkit.setup_test_environment(
            service_script,
            components_script,
            test_script,
            dummy_data,
        )

        fix_tools = toolkit.get_tool_definitions()
        strategy_allowed_tools: Optional[set[str]] = None
        strategy_instruction = ""
        objective_instruction = "Use the provided tools to fix the failing tests. "
        code_fix_instruction = (
            "If the service or components code is incorrect, fix them accordingly. "
        )
        test_fix_instruction = "If tests are wrong, ie have incorrect assumptions, or test things that are unrealistic or not matching the dummy data, correct them with targeted edits that preserve intent. "
        missing_tests_instruction = "If there are tests missing that should be present, add them to the test script. "
        fixture_instruction = (
            "If test fixtures need updates, add data to the dummy data module. "
            "Do not delete from dummy data. "
        )
        if strategy_mode == "fix_code":
            strategy_allowed_tools = {
                "run_tests",
                "get_last_test_output",
                "run_debug_code",
                "apply_script_patch",
                "update_service_script",
                "update_components_script",
                "get_script_lines",
                "get_current_scripts",
                "search_files",
            }
            strategy_instruction = (
                "Important: prioritize runtime fixes in service_script/components_script. "
                "Do NOT modify test_script or dummy_data in this mode. "
                "If failures appear to be test-only issues, note that briefly and continue with runtime-focused fixes. "
                "Never import './dummy_data.js' or reference dummyData in runtime code. "
                "If the harness warns about repeated assertion-signature failures and you have a strong runtime hypothesis, "
                "respond with 'BAIL_OVERRIDE: <runtime hypothesis>' before tool calls to continue one focused runtime attempt."
            )
            objective_instruction = "Use the provided tools to fix runtime issues that cause test failures. "
            test_fix_instruction = ""
            missing_tests_instruction = ""
            fixture_instruction = (
                "Do not change test fixtures in this mode; fix runtime code only. "
            )
        elif strategy_mode == "adjust_test":
            strategy_allowed_tools = {
                "run_tests",
                "get_last_test_output",
                "run_debug_code",
                "apply_script_patch",
                "update_test_script",
                "update_dummy_data",
                "get_script_lines",
                "get_current_scripts",
                "search_files",
            }
            strategy_instruction = (
                "Important: you may and should modify test_script/dummy_data when tests are broken. "
                "Do NOT modify service_script or components_script in this mode. "
                "Prefer minimal, targeted edits that keep test intent and coverage strong."
            )
            objective_instruction = (
                "Use the provided tools to repair broken tests and fixtures. "
            )
            code_fix_instruction = ""
            test_fix_instruction = "If tests are wrong, correct them (for example invalid assumptions about wrappers/events/timing/data shape). "
        if strategy_allowed_tools:
            fix_tools = [
                tool
                for tool in fix_tools
                if tool.get("function", {}).get("name") in strategy_allowed_tools
            ]

        fix_messages = copy.deepcopy(messages)
        test_result = toolkit.run_tests()
        _log_test_run_output(
            "[iterative_test_fix] Initial test run",
            test_result.content,
        )
        initial_passed, initial_failed, _ = _parse_tap_output(test_result.content)
        last_passed = initial_passed
        focused_failure_name: Optional[str] = None

        def _sync_single_failure_focus(
            failed: int,
            failed_tests: List[str],
            *,
            context: str,
        ) -> None:
            nonlocal focused_failure_name
            if not FOCUS_SINGLE_FAILURE_ENABLED:
                return

            next_focus: Optional[str] = None
            if failed == 1 and failed_tests:
                candidate = (failed_tests[0] or "").strip()
                if candidate:
                    next_focus = candidate

            if next_focus == focused_failure_name:
                return

            previous_focus = focused_failure_name
            focused_failure_name = next_focus
            toolkit.set_focus_test_name(next_focus)

            if next_focus:
                logger.info(
                    "[iterative_test_fix] Single failing test focus (%s): %s",
                    context,
                    next_focus,
                )
                fix_messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=(
                            "Focus mode: exactly one failing test is active. "
                            f"Prioritize fixing '{next_focus}' with minimal, localized changes, "
                            "then run the full suite to guard against regressions."
                        ),
                    )
                )
            elif previous_focus:
                logger.info(
                    "[iterative_test_fix] Clearing single-test focus (%s)",
                    context,
                )

        _sync_single_failure_focus(
            initial_failed,
            _parse_tap_output(test_result.content)[2],
            context="initial run",
        )

        fix_messages.append(
            Message(
                role=MessageRole.USER,
                content=(
                    "You are an expert JavaScript QA developer tasked with fixing the "
                    "failing tests and/or contract validation issues with the following output:\n\n"
                    f"```\n{test_result.content}\n```\n\n"
                    f"{objective_instruction}"
                    f"{missing_tests_instruction}"
                    f"{code_fix_instruction}"
                    f"{fixture_instruction}"
                    f"{test_fix_instruction}"
                    "Call run_tests to verify your changes. "
                    "Harness note: pfuschTest returns a PfuschNodeCollection wrapper around the host component; "
                    "component internals are on comp.host (for example comp.host.state / comp.host.shadowRoot). "
                    "Mock-shape note: svc.test.addResolved(tool, payload) should generally provide the post-resultKey shape; "
                    "for calls using resultKey='value', provide an array (not wrapped { value: [...] }) unless runtime explicitly expects wrappers. "
                    "Hyphenated custom element names (for example 'air-quality') are valid; do not rename tags just to avoid hyphens. "
                    "Slot note: avoid slot-only refactors unless tests explicitly provide slotted Light DOM content. "
                    "Do not use `html.slot() || fallback` patterns and do not depend on `slot.assignedNodes()` / `slot.assignedElements()`. "
                    "Treat helpers.children(...) as initial Light DOM capture, not dynamic future-child discovery. "
                    "Selector note: do not assume global button indexes when asserting clicks; use scoped selectors/text. "
                    "Before any tool calls, add one line in your assistant response as "
                    "'FIX_EXPLANATION: <brief root cause and why the planned change should help>'. "
                    f"{strategy_instruction} "
                    "When all tests pass, respond with 'TESTS_PASSING' and nothing else."
                ),
            )
        )

        attempt = 0
        read_only_streak = 0
        fix_code_assertion_failures = 0
        fix_code_pending_bail_decision = False
        fix_code_bail_override_used = False
        fix_code_bail_override_grace_failures = 0
        best_passed = initial_passed
        best_failed = initial_failed
        best_snapshot = {
            "service_script": service_script,
            "components_script": components_script,
            "test_script": test_script,
            "dummy_data": dummy_data,
            "messages": copy.deepcopy(fix_messages),
        }

        def _validate_post_success(
            *,
            context: str,
            current_service_script: str,
            current_components_script: str,
            current_test_script: str,
            current_dummy_data: Optional[str],
        ) -> Tuple[bool, Optional[str]]:
            if not post_success_validator:
                return True, None
            try:
                is_valid, reason = post_success_validator(
                    current_service_script,
                    current_components_script,
                    current_test_script,
                    current_dummy_data,
                )
            except Exception as exc:
                logger.error(
                    "[iterative_test_fix] Post-success validator errored during %s: %s",
                    context,
                    exc,
                    exc_info=exc,
                )
                return False, f"post_success_validator_exception: {exc}"
            if is_valid:
                return True, None
            normalized_reason = (reason or "post_success_validation_failed").strip()
            logger.warning(
                "[iterative_test_fix] Post-success validation failed during %s: %s",
                context,
                normalized_reason,
            )
            return False, normalized_reason

        async def _handle_post_success_validation_failure(
            *,
            context: str,
            reason: Optional[str],
            passed: int,
            failed: int,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> None:
            normalized_reason = (reason or "post_success_validation_failed").strip()
            fix_messages.append(
                Message(
                    role=MessageRole.USER,
                    content=(
                        "Tests are passing but contract validation failed "
                        f"({context}): {normalized_reason}. "
                        "Continue fixing scripts until BOTH tests and contract checks pass."
                    ),
                )
            )
            best_snapshot["messages"] = copy.deepcopy(fix_messages)
            if event_queue:
                await event_queue.put(
                    {
                        "event": "test_result",
                        "status": "contract_failed",
                        "message": "Tests passed but contract validation failed.",
                        "reason": normalized_reason,
                        "passed": passed,
                        "failed": failed,
                        "metadata": metadata,
                    }
                )

        def _contract_failure_requires_adjust_test(reason: Optional[str]) -> bool:
            if strategy_mode != "fix_code":
                return False
            normalized_reason = (reason or "").lower()
            return any(
                marker in normalized_reason
                for marker in (
                    "schema_field_drift:test_mock:",
                    "schema_field_drift:test_ref:",
                    "weak_assertion:",
                )
            )

        async def _return_for_adjust_test_handoff(
            reason: Optional[str],
            *,
            reason_code: str,
            use_best_snapshot: bool = False,
            handoff_message: Optional[str] = None,
            passed: Optional[int] = None,
            failed: Optional[int] = None,
        ) -> Tuple[bool, str, str, str, Optional[str], List[Message]]:
            normalized_reason = (
                reason or "contract_validation_requires_adjust_test"
            ).strip()
            handoff_note = handoff_message or (
                "Contract validation issue appears to require test/fixture edits "
                "(not allowed in fix_code mode). Handing off to adjust_test stage. "
                f"Reason: {normalized_reason}"
            )
            logger.info("[iterative_test_fix] %s", handoff_note)
            fix_messages.append(
                Message(
                    role=MessageRole.USER,
                    content=handoff_note,
                )
            )
            best_snapshot["messages"] = copy.deepcopy(fix_messages)
            if event_queue:
                await event_queue.put(
                    {
                        "event": "test_result",
                        "status": "handoff",
                        "message": "Handing off to adjust_test stage.",
                        "reason_code": reason_code,
                        "reason": normalized_reason,
                        "suggested_action": "adjust_test",
                        "passed": int(best_passed if passed is None else passed),
                        "failed": int(best_failed if failed is None else failed),
                    }
                )
            if use_best_snapshot:
                return (
                    False,
                    best_snapshot["service_script"],
                    best_snapshot["components_script"],
                    best_snapshot["test_script"],
                    best_snapshot["dummy_data"],
                    best_snapshot["messages"],
                )
            return (
                False,
                toolkit.current_service_script,
                toolkit.current_components_script,
                toolkit.current_test_script,
                toolkit.current_dummy_data,
                fix_messages,
            )

        initial_validation_ok = False
        initial_validation_reason: Optional[str] = None
        if test_result.success:
            initial_validation_ok, initial_validation_reason = _validate_post_success(
                context="initial run",
                current_service_script=service_script,
                current_components_script=components_script,
                current_test_script=test_script,
                current_dummy_data=dummy_data,
            )
            if initial_validation_ok:
                logger.info("[iterative_test_fix] Tests passed on first try!")
                return (
                    True,
                    service_script,
                    components_script,
                    test_script,
                    dummy_data,
                    messages,
                )
            initial_failed = max(1, initial_failed)
            await _handle_post_success_validation_failure(
                context="initial run",
                reason=initial_validation_reason,
                passed=initial_passed,
                failed=initial_failed,
                metadata=test_result.metadata,
            )
            if _contract_failure_requires_adjust_test(initial_validation_reason):
                return await _return_for_adjust_test_handoff(
                    initial_validation_reason,
                    reason_code="contract_validation_requires_adjust_test",
                )

        logger.info(
            "[iterative_test_fix] Initial checks failed or contract validation rejected candidate, starting fix loop (max %s attempts)",
            max_attempts,
        )

        def _reset_attempts_on_progress(passed: int, context: str) -> None:
            nonlocal last_passed, attempt
            if passed > last_passed:
                logger.info(
                    "[iterative_test_fix] Test progress %s: %s passed (was %s); resetting attempts",
                    context,
                    passed,
                    last_passed,
                )
                attempt = 0
            last_passed = passed

        def _capture_snapshot() -> Dict[str, Any]:
            return {
                "service_script": toolkit.current_service_script,
                "components_script": toolkit.current_components_script,
                "test_script": toolkit.current_test_script,
                "dummy_data": toolkit.current_dummy_data,
                "messages": copy.deepcopy(fix_messages),
            }

        def _update_best_snapshot(passed: int, failed: int) -> None:
            nonlocal best_passed, best_failed, best_snapshot
            if passed > best_passed or (passed == best_passed and failed < best_failed):
                best_passed = passed
                best_failed = failed
                best_snapshot = _capture_snapshot()

        def _restore_best_snapshot(reason: str) -> None:
            toolkit.current_service_script = best_snapshot["service_script"]
            toolkit.current_components_script = best_snapshot["components_script"]
            toolkit.current_test_script = best_snapshot["test_script"]
            toolkit.current_dummy_data = best_snapshot["dummy_data"]
            logger.warning("[iterative_test_fix] %s Restored best snapshot.", reason)
            fix_messages.append(
                Message(
                    role=MessageRole.USER,
                    content=(
                        f"{reason} A previous better script snapshot was restored "
                        "automatically. Continue from the restored scripts."
                    ),
                )
            )

        def _looks_like_test_assertion_shape_failure(output: str) -> bool:
            text = output or ""
            lowered = text.lower()
            has_assertion_signal = (
                "assertionerror" in lowered
                or "err_assertion" in lowered
                or "cannot read properties of undefined" in lowered
            )
            has_test_reference = (
                "user_test.js" in text
                or "testcontext.<anonymous>" in lowered
                or "failuretype: 'subtestsfailed'" in lowered
            )
            return has_assertion_signal and has_test_reference

        def _reset_fix_code_bail_tracking() -> None:
            nonlocal fix_code_assertion_failures
            nonlocal fix_code_pending_bail_decision
            nonlocal fix_code_bail_override_used
            nonlocal fix_code_bail_override_grace_failures
            fix_code_assertion_failures = 0
            fix_code_pending_bail_decision = False
            fix_code_bail_override_used = False
            fix_code_bail_override_grace_failures = 0

        def _register_fix_code_assertion_failure(output: str, failed: int) -> bool:
            nonlocal fix_code_assertion_failures
            nonlocal fix_code_pending_bail_decision
            if strategy_mode != "fix_code":
                _reset_fix_code_bail_tracking()
                return False
            if failed <= 0:
                _reset_fix_code_bail_tracking()
                return False
            if _looks_like_test_assertion_shape_failure(output):
                fix_code_assertion_failures += 1
                logger.warning(
                    "[iterative_test_fix] fix_code assertion-style failure signal (%s/%s)",
                    fix_code_assertion_failures,
                    FIX_CODE_ASSERTION_BAILOUT_LIMIT,
                )
            else:
                _reset_fix_code_bail_tracking()
                return False

            if fix_code_assertion_failures < FIX_CODE_ASSERTION_BAILOUT_LIMIT:
                return False

            if not fix_code_pending_bail_decision:
                fix_code_pending_bail_decision = True
                decision_note = (
                    "Repeated assertion-signature failures in fix_code mode suggest possible test/fixture-shape mismatch. "
                    "Before handing off to adjust_test, decide explicitly: "
                    "If runtime fix is still likely, respond with "
                    "'BAIL_OVERRIDE: <runtime hypothesis>' and continue with focused runtime-only edits; "
                    "otherwise proceed without override and handoff will occur on the next repeated signal."
                )
                logger.warning("[iterative_test_fix] %s", decision_note)
                fix_messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=decision_note,
                    )
                )
                best_snapshot["messages"] = copy.deepcopy(fix_messages)
                return False

            return True

        def _apply_fix_code_bail_override(content: str) -> None:
            nonlocal fix_code_pending_bail_decision
            nonlocal fix_code_bail_override_used
            nonlocal fix_code_bail_override_grace_failures
            override = _extract_bail_override(content)
            if not override:
                return
            if strategy_mode != "fix_code":
                return
            if not fix_code_pending_bail_decision:
                logger.info(
                    "[iterative_test_fix] Ignoring BAIL_OVERRIDE because bail-decision mode is not active"
                )
                return
            fix_code_pending_bail_decision = False
            fix_code_bail_override_used = True
            fix_code_bail_override_grace_failures = 1
            logger.info(
                "[iterative_test_fix] BAIL_OVERRIDE accepted; continuing fix_code with focused runtime attempt: %s",
                override,
            )
            fix_messages.append(
                Message(
                    role=MessageRole.USER,
                    content=(
                        "BAIL_OVERRIDE accepted. Continue with one focused runtime-only attempt "
                        "(service/components) and run_tests. "
                        f"Hypothesis: {override}"
                    ),
                )
            )
            best_snapshot["messages"] = copy.deepcopy(fix_messages)

        async def _should_handoff_after_assertion_failure(
            *,
            output: str,
            passed: int,
            failed: int,
        ) -> Optional[Tuple[bool, str, str, str, Optional[str], List[Message]]]:
            nonlocal fix_code_bail_override_grace_failures
            should_handoff = _register_fix_code_assertion_failure(output, failed)
            if not should_handoff:
                return None
            if fix_code_bail_override_grace_failures > 0:
                fix_code_bail_override_grace_failures -= 1
                logger.warning(
                    "[iterative_test_fix] assertion signature persists after BAIL_OVERRIDE; grace remaining=%s",
                    fix_code_bail_override_grace_failures,
                )
                return None
            if fix_code_pending_bail_decision:
                reason = (
                    "Repeated assertion-signature failures persisted without BAIL_OVERRIDE. "
                    "Handing off to adjust_test so assertions/fixtures can be repaired."
                )
            elif fix_code_bail_override_used:
                reason = (
                    "Repeated assertion-signature failures persisted after BAIL_OVERRIDE focused runtime attempt. "
                    "Handing off to adjust_test so assertions/fixtures can be repaired."
                )
            else:
                reason = (
                    "Repeated assertion-signature failures in fix_code mode suggest test/fixture-shape mismatch. "
                    "Handing off to adjust_test so assertions/fixtures can be repaired."
                )
            return await _return_for_adjust_test_handoff(
                reason,
                reason_code="assertion_signature_repeat",
                use_best_snapshot=True,
                handoff_message=reason,
                passed=passed,
                failed=failed,
            )

        while attempt < max_attempts:
            attempt += 1
            logger.info(
                "[iterative_test_fix] Fix iteration %s/%s", attempt, max_attempts
            )

            before_diag = _collect_payload_diagnostics(fix_messages, fix_tools)
            logger.info(
                "[iterative_test_fix] Payload diagnostics before cap (iteration %s): %s",
                attempt,
                before_diag,
            )

            capped_messages, cap_stats = _cap_fix_messages_for_request(
                fix_messages,
                max_messages=ITERATIVE_FIX_MAX_MESSAGES,
                max_total_bytes=ITERATIVE_FIX_MAX_PAYLOAD_BYTES,
                max_message_bytes=ITERATIVE_FIX_MAX_MESSAGE_BYTES,
                max_tool_message_bytes=ITERATIVE_FIX_MAX_TOOL_MESSAGE_BYTES,
            )
            if cap_stats["trimmed_messages"] or cap_stats["dropped_messages"]:
                logger.warning(
                    "[iterative_test_fix] Message history capped before request (iteration %s): %s",
                    attempt,
                    cap_stats,
                )
            fix_messages = capped_messages

            after_diag = _collect_payload_diagnostics(fix_messages, fix_tools)
            logger.info(
                "[iterative_test_fix] Payload diagnostics after cap (iteration %s): %s",
                attempt,
                after_diag,
            )

            chat_request = ChatCompletionRequest(
                messages=fix_messages,
                tools=fix_tools,
                tool_choice="required" if require_tool_calls else "auto",
                stream=False,
                extra_headers=extra_headers,
            )

            try:
                logger.info(
                    "[iterative_test_fix] Requesting LLM response (iteration %s)",
                    attempt,
                )
                llm_client = getattr(tgi_service, "llm_client", None)
                if llm_client is None:
                    raise RuntimeError("LLM client unavailable for iterative_test_fix")

                if hasattr(llm_client, "_prepare_payload"):
                    chat_request = await llm_client._prepare_payload(
                        chat_request,
                        access_token or "",
                    )
                if hasattr(llm_client, "_payload_size"):
                    logger.info(
                        "[iterative_test_fix] Prepared chat request payload bytes (iteration %s): %s",
                        attempt,
                        llm_client._payload_size(chat_request),
                    )

                non_stream = getattr(llm_client, "non_stream_completion", None)
                if callable(non_stream):
                    response = await non_stream(chat_request, access_token or "", None)
                else:
                    response = await llm_client.client.chat.completions.create(
                        **llm_client._build_request_params(chat_request)
                    )
                logger.info(
                    "[iterative_test_fix] LLM response received (iteration %s)",
                    attempt,
                )
            except Exception as e:
                logger.error(f"[iterative_test_fix] LLM call failed: {e}")
                break

            choice = response.choices[0]
            message = choice.message
            content = message.content or ""

            assistant_msg = Message(
                role=MessageRole.ASSISTANT,
                content=content,
            )
            tool_calls = []
            if message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        type=tc.type,
                        function=ToolCallFunction(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        ),
                    )
                    for tc in message.tool_calls
                ]
            if not tool_calls and content:
                parsed_tool_calls, _ = tool_resolution.resolve_tool_calls(content)
                if parsed_tool_calls:
                    tool_calls = [
                        ToolCall(
                            id=tc.id,
                            type="function",
                            function=ToolCallFunction(
                                name=tc.name,
                                arguments=json.dumps(tc.arguments),
                            ),
                        )
                        for tc in parsed_tool_calls
                    ]
                    logger.warning(
                        "[iterative_test_fix] Parsed %s tool call(s) from content fallback",
                        len(tool_calls),
                    )
            if tool_calls:
                assistant_msg.tool_calls = tool_calls
            tool_call_names = [
                tc.function.name
                for tc in tool_calls
                if tc.function and tc.function.name
            ]
            fix_explanation = _extract_fix_explanation(content)
            if not fix_explanation:
                fix_explanation = _extract_general_explanation(content)
            if not fix_explanation:
                fix_explanation = _inferred_tool_explanation(tool_call_names)
            if fix_explanation:
                logger.info(
                    "[iterative_test_fix] Fix explanation (iteration %s): %s",
                    attempt,
                    fix_explanation,
                )
                logger.info(
                    "[iterative_test_fix] Why (iteration %s): %s",
                    attempt,
                    fix_explanation,
                )
            fix_messages.append(assistant_msg)
            _apply_fix_code_bail_override(content)

            if not tool_calls:
                no_tool_call_attempts += 1
                require_tool_calls = True
                content_preview = content.strip() or "<empty>"
                if len(content_preview) > 300:
                    content_preview = f"{content_preview[:300]}...[truncated]"
                logger.warning(
                    "[iterative_test_fix] No tool calls returned on iteration %s (consecutive %s): %s",
                    attempt,
                    no_tool_call_attempts,
                    content_preview,
                )
                if "TESTS_PASSING" in content:
                    logger.info("[iterative_test_fix] LLM indicates tests are passing")
                    final_result = toolkit.run_tests()
                    _log_test_run_output(
                        "[iterative_test_fix] Verification test run",
                        final_result.content,
                    )
                    passed, failed, _ = _parse_tap_output(final_result.content)
                    _sync_single_failure_focus(
                        failed,
                        _parse_tap_output(final_result.content)[2],
                        context="verification run",
                    )
                    _reset_attempts_on_progress(passed, "verification run")
                    _update_best_snapshot(passed, failed)
                    if final_result.success:
                        validation_ok, validation_reason = _validate_post_success(
                            context="verification run",
                            current_service_script=toolkit.current_service_script,
                            current_components_script=toolkit.current_components_script,
                            current_test_script=toolkit.current_test_script,
                            current_dummy_data=toolkit.current_dummy_data,
                        )
                        if validation_ok:
                            return (
                                True,
                                toolkit.current_service_script,
                                toolkit.current_components_script,
                                toolkit.current_test_script,
                                toolkit.current_dummy_data,
                                fix_messages,
                            )
                        await _handle_post_success_validation_failure(
                            context="verification run",
                            reason=validation_reason,
                            passed=passed,
                            failed=max(failed, 1),
                            metadata=final_result.metadata,
                        )
                        if _contract_failure_requires_adjust_test(validation_reason):
                            return await _return_for_adjust_test_handoff(
                                validation_reason,
                                reason_code="contract_validation_requires_adjust_test",
                            )
                    else:
                        fix_messages.append(
                            Message(
                                role=MessageRole.USER,
                                content=(
                                    "Tests are still failing:\n"
                                    f"```\n{final_result.content}\n```\nContinue fixing."
                                ),
                            ),
                        )
                else:
                    fix_messages.append(
                        Message(
                            role=MessageRole.USER,
                            content=(
                                "No tool calls were returned. Use the provided tools to "
                                "inspect or modify scripts, then run_tests again."
                            ),
                        )
                    )
                # Keep the best snapshot messages current even when no tests were run yet,
                # so callers receive the latest retry guidance on failure.
                best_snapshot["messages"] = copy.deepcopy(fix_messages)
                if no_tool_call_attempts <= 2:
                    attempt -= 1
                else:
                    logger.warning(
                        "[iterative_test_fix] Consecutive no-tool-call iterations=%s; counting attempts to avoid infinite retry loop",
                        no_tool_call_attempts,
                    )
                continue
            no_tool_call_attempts = 0

            tool_results = []
            all_tests_passed = False
            has_modification_tools = False
            mutation_since_last_test = False

            # Don't count attempts that only read/search code
            read_only_tools = {
                "get_script_lines",
                "get_current_scripts",
                "search_files",
            }
            read_only_only = bool(tool_calls) and all(
                tc.function.name in read_only_tools for tc in tool_calls
            )
            if read_only_only:
                logger.info(
                    "[iterative_test_fix] Only read/search tools used - not counting as attempt"
                )
                read_only_streak += 1
            else:
                read_only_streak = 0

            should_force_run_tests_for_streak = (
                read_only_only and read_only_streak >= READ_ONLY_STREAK_LIMIT
            )
            if should_force_run_tests_for_streak:
                logger.info(
                    "[iterative_test_fix] Read-only streak limit reached (%s). Forcing run_tests.",
                    READ_ONLY_STREAK_LIMIT,
                )
                read_only_streak = 0
            elif read_only_only:
                attempt -= 1

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                if strategy_allowed_tools and tool_name not in strategy_allowed_tools:
                    disallowed = (
                        f"Tool '{tool_name}' is not allowed in strategy '{strategy_mode}'. "
                        "Use an allowed tool and continue."
                    )
                    logger.warning("[iterative_test_fix] %s", disallowed)
                    tool_results.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": disallowed,
                        }
                    )
                    continue
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arg_preview = tool_call.function.arguments or ""
                    if len(arg_preview) > 200:
                        arg_preview = f"{arg_preview[:200]}...[truncated]"
                    logger.warning(
                        "[iterative_test_fix] Invalid JSON arguments for %s: %s",
                        tool_name,
                        arg_preview,
                    )
                    arguments = {}

                if isinstance(arguments, dict) and isinstance(
                    arguments.get("input"), dict
                ):
                    arguments = arguments["input"]

                if not isinstance(arguments, dict):
                    logger.warning(
                        "[iterative_test_fix] Non-object arguments for %s: %r",
                        tool_name,
                        arguments,
                    )
                    arguments = {}

                modification_tools = {
                    "update_test_script",
                    "update_service_script",
                    "update_components_script",
                    "update_dummy_data",
                }
                if tool_name in modification_tools:
                    has_modification_tools = True
                    mutation_since_last_test = True

                tool_desc = _tool_description(tool_name, arguments)
                logger.info(f"[iterative_test_fix] {tool_desc}")

                if event_queue:
                    tool_start_payload = {
                        "event": "tool_start",
                        "tool": tool_name,
                        "description": tool_desc,
                    }
                    if fix_explanation:
                        tool_start_payload["fix_explanation"] = fix_explanation
                        tool_start_payload["why"] = fix_explanation
                    await event_queue.put(tool_start_payload)

                result = toolkit.execute_tool(tool_name, arguments)
                if not result.success and tool_name != "run_tests":
                    logger.warning(
                        "[iterative_test_fix] Failed to %s:\n%s",
                        tool_desc,
                        result.content,
                    )

                tool_results.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_name,
                        "content": result.content,
                    }
                )

                if tool_name == "run_tests":
                    mutation_since_last_test = False
                    _log_test_run_output(
                        f"[iterative_test_fix] Test run (iteration {attempt})",
                        result.content,
                    )
                    passed, failed, failed_tests = _parse_tap_output(result.content)
                    _sync_single_failure_focus(
                        failed,
                        failed_tests,
                        context=f"iteration {attempt} run_tests",
                    )
                    _reset_attempts_on_progress(passed, f"iteration {attempt}")
                    _update_best_snapshot(passed, failed)
                    if best_passed > 0 and (
                        passed == 0 or passed <= max(0, best_passed - 3)
                    ):
                        _restore_best_snapshot(
                            (
                                "Sharp test regression detected "
                                f"(passed={passed}, best_passed={best_passed})."
                            )
                        )
                        if event_queue:
                            await event_queue.put(
                                {
                                    "event": "test_result",
                                    "status": "regression_rollback",
                                    "passed": passed,
                                    "failed": failed,
                                    "best_passed": best_passed,
                                    "best_failed": best_failed,
                                }
                            )
                        continue
                    if result.success:
                        validation_ok, validation_reason = _validate_post_success(
                            context=f"iteration {attempt} run_tests",
                            current_service_script=toolkit.current_service_script,
                            current_components_script=toolkit.current_components_script,
                            current_test_script=toolkit.current_test_script,
                            current_dummy_data=toolkit.current_dummy_data,
                        )
                        if validation_ok:
                            all_tests_passed = True
                            logger.info("[iterative_test_fix] All tests passed!")

                            if event_queue:
                                await event_queue.put(
                                    {
                                        "event": "test_result",
                                        "status": "passed",
                                        "message": "All tests passed!",
                                        "passed": passed,
                                        "failed": failed,
                                        "metadata": result.metadata,
                                    }
                                )
                        else:
                            await _handle_post_success_validation_failure(
                                context=f"iteration {attempt} run_tests",
                                reason=validation_reason,
                                passed=passed,
                                failed=max(failed, 1),
                                metadata=result.metadata,
                            )
                            if _contract_failure_requires_adjust_test(
                                validation_reason
                            ):
                                return await _return_for_adjust_test_handoff(
                                    validation_reason,
                                    reason_code="contract_validation_requires_adjust_test",
                                )
                    else:
                        if failed > 0:
                            test_list = ", ".join(failed_tests[:3])
                            if len(failed_tests) > 3:
                                test_list += f", ... ({len(failed_tests) - 3} more)"
                            logger.info(
                                "[iterative_test_fix] Tests failed: %s passed, %s failed | %s",
                                passed,
                                failed,
                                test_list,
                            )
                            if event_queue:
                                await event_queue.put(
                                    {
                                        "event": "test_result",
                                        "status": "failed",
                                        "passed": passed,
                                        "failed": failed,
                                        "failed_tests": failed_tests,
                                        "metadata": result.metadata,
                                    }
                                )
                            handoff_result = (
                                await _should_handoff_after_assertion_failure(
                                    output=result.content,
                                    passed=passed,
                                    failed=failed,
                                )
                            )
                            if handoff_result is not None:
                                return handoff_result
                        else:
                            logger.warning(
                                "[iterative_test_fix] Test run failed with unexpected output"
                            )
                            if event_queue:
                                await event_queue.put(
                                    {
                                        "event": "test_result",
                                        "status": "error",
                                        "message": "Test run failed with unexpected output",
                                    }
                                )

            should_force_run_tests_for_mutation = mutation_since_last_test
            if should_force_run_tests_for_mutation or should_force_run_tests_for_streak:
                forced_reason = (
                    "mutation_without_run_tests"
                    if should_force_run_tests_for_mutation
                    else "read_only_streak_limit"
                )
                logger.info(
                    "[iterative_test_fix] Forcing run_tests (%s) before next LLM round",
                    forced_reason,
                )
                forced_result = toolkit.run_tests()
                _log_test_run_output(
                    f"[iterative_test_fix] Forced test run (iteration {attempt})",
                    forced_result.content,
                )
                forced_passed, forced_failed, forced_failed_tests = _parse_tap_output(
                    forced_result.content
                )
                _sync_single_failure_focus(
                    forced_failed,
                    forced_failed_tests,
                    context=f"forced iteration {attempt}",
                )
                _reset_attempts_on_progress(
                    forced_passed, f"forced iteration {attempt}"
                )
                _update_best_snapshot(forced_passed, forced_failed)

                fix_messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=(
                            f"[forced run_tests result (iteration {attempt})]\n"
                            f"{forced_result.content}"
                        ),
                    )
                )

                if best_passed > 0 and (
                    forced_passed == 0 or forced_passed <= max(0, best_passed - 3)
                ):
                    _restore_best_snapshot(
                        (
                            "Sharp test regression detected after forced run "
                            f"(passed={forced_passed}, best_passed={best_passed})."
                        )
                    )
                    if event_queue:
                        await event_queue.put(
                            {
                                "event": "test_result",
                                "status": "regression_rollback",
                                "passed": forced_passed,
                                "failed": forced_failed,
                                "best_passed": best_passed,
                                "best_failed": best_failed,
                            }
                        )
                elif forced_result.success:
                    validation_ok, validation_reason = _validate_post_success(
                        context=f"forced iteration {attempt}",
                        current_service_script=toolkit.current_service_script,
                        current_components_script=toolkit.current_components_script,
                        current_test_script=toolkit.current_test_script,
                        current_dummy_data=toolkit.current_dummy_data,
                    )
                    if validation_ok:
                        all_tests_passed = True
                        if event_queue:
                            await event_queue.put(
                                {
                                    "event": "test_result",
                                    "status": "passed",
                                    "message": "All tests passed!",
                                    "passed": forced_passed,
                                    "failed": forced_failed,
                                    "metadata": forced_result.metadata,
                                }
                            )
                    else:
                        await _handle_post_success_validation_failure(
                            context=f"forced iteration {attempt}",
                            reason=validation_reason,
                            passed=forced_passed,
                            failed=max(forced_failed, 1),
                            metadata=forced_result.metadata,
                        )
                        if _contract_failure_requires_adjust_test(validation_reason):
                            return await _return_for_adjust_test_handoff(
                                validation_reason,
                                reason_code="contract_validation_requires_adjust_test",
                            )
                elif event_queue:
                    await event_queue.put(
                        {
                            "event": "test_result",
                            "status": "failed",
                            "passed": forced_passed,
                            "failed": forced_failed,
                            "failed_tests": forced_failed_tests,
                            "metadata": forced_result.metadata,
                        }
                    )

                handoff_result = await _should_handoff_after_assertion_failure(
                    output=forced_result.content,
                    passed=forced_passed,
                    failed=forced_failed,
                )
                if handoff_result is not None:
                    return handoff_result

            for tool_result in tool_results:
                fix_messages.append(
                    Message(
                        role=MessageRole.TOOL,
                        content=tool_result["content"],
                        tool_call_id=tool_result["tool_call_id"],
                        name=tool_result["name"],
                    )
                )

            if "TESTS_PASSING" in content and not all_tests_passed:
                logger.info("[iterative_test_fix] LLM indicates tests are passing")
                final_result = toolkit.run_tests()
                _log_test_run_output(
                    "[iterative_test_fix] Verification test run",
                    final_result.content,
                )
                passed, failed, _ = _parse_tap_output(final_result.content)
                _sync_single_failure_focus(
                    failed,
                    _parse_tap_output(final_result.content)[2],
                    context="verification run",
                )
                _reset_attempts_on_progress(passed, "verification run")
                _update_best_snapshot(passed, failed)
                if final_result.success:
                    validation_ok, validation_reason = _validate_post_success(
                        context="verification run",
                        current_service_script=toolkit.current_service_script,
                        current_components_script=toolkit.current_components_script,
                        current_test_script=toolkit.current_test_script,
                        current_dummy_data=toolkit.current_dummy_data,
                    )
                    if validation_ok:
                        return (
                            True,
                            toolkit.current_service_script,
                            toolkit.current_components_script,
                            toolkit.current_test_script,
                            toolkit.current_dummy_data,
                            fix_messages,
                        )
                    await _handle_post_success_validation_failure(
                        context="verification run",
                        reason=validation_reason,
                        passed=passed,
                        failed=max(failed, 1),
                        metadata=final_result.metadata,
                    )
                    if _contract_failure_requires_adjust_test(validation_reason):
                        return await _return_for_adjust_test_handoff(
                            validation_reason,
                            reason_code="contract_validation_requires_adjust_test",
                        )
                else:
                    fix_messages.append(
                        Message(
                            role=MessageRole.USER,
                            content=(
                                "Tests are still failing:\n"
                                f"```\n{final_result.content}\n```\nContinue fixing."
                            ),
                        )
                    )
                if not has_modification_tools:
                    attempt -= 1
                continue

            if all_tests_passed:
                return (
                    True,
                    toolkit.current_service_script,
                    toolkit.current_components_script,
                    toolkit.current_test_script,
                    toolkit.current_dummy_data,
                    fix_messages,
                )

        logger.warning(
            "[iterative_test_fix] Failed to fix tests after %s attempts. Returning best snapshot (passed=%s, failed=%s)",
            max_attempts,
            best_passed,
            best_failed,
        )
        return (
            False,
            best_snapshot["service_script"],
            best_snapshot["components_script"],
            best_snapshot["test_script"],
            best_snapshot["dummy_data"],
            best_snapshot["messages"],
        )

    finally:
        toolkit.cleanup()
