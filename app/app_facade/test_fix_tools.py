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
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    "get_script_lines": lambda args: (
        f"reading source code from {_SCRIPT_TYPE_TO_FILENAME.get(args.get('script_type'), 'unknown')}"
        + (
            f": {args.get('start_line', '?')}:{args.get('end_line', '?')}"
            if args.get("start_line")
            else ""
        )
    ),
    "get_current_scripts": lambda args: "reading source code",
    "search_files": lambda args: (
        f"searching for '{args.get('regex', '')}'"
        + (
            f" [in file(s): {args.get('script_type')}]"
            if args.get("script_type")
            else ""
        )
    ),
}


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


def _log_test_run_output(context: str, output: str) -> None:
    output = output or ""
    if not output.strip():
        logger.info("%s Test run produced no output", context)
        return
    output_tail = output[-4000:] if len(output) > 4000 else output
    if len(output) > 4000:
        output_tail = f"...(trimmed {len(output) - 4000} chars)\n{output_tail}"
    logger.info("%s Test run output:\n%s", context, output_tail)


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

        # Write combined service + components script to mirror production bundling.
        mocked_components = (
            (self.current_components_script or "")
            .replace(
                "https://matthiaskainer.github.io/pfusch/pfusch.min.js", "./pfusch.js"
            )
            .replace("https://matthiaskainer.github.io/pfusch/pfusch.js", "./pfusch.js")
        )
        combined_script = (
            (self.current_service_script or "") + "\n\n" + mocked_components
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
            f.write(self.current_test_script or "")

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

            cmd = ["node", "--test", "test.js"]
            if test_name:
                cmd.extend(["--test-name-pattern", test_name])

            env = os.environ.copy()
            env["NODE_PATH"] = self.tmpdir

            logger.info(
                "[iterative_test_fix] Running node tests%s",
                f" (pattern: {test_name})" if test_name else "",
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

            # Extract summary from TAP output
            metadata = self._parse_tap_output(output)

            if not success:
                # Truncate output if too long
                if len(output) > 4000:
                    output = (
                        f"...(trimmed {len(output) - 4000} chars)\n{output[-4000:]}"
                    )

            return ToolResult(
                success=success,
                content=output,
                metadata=metadata,
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False, content="Tests timed out after 30 seconds."
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
            "run_tests": lambda: self.run_tests(arguments.get("test_name")),
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

        test_result = toolkit.run_tests()
        _log_test_run_output(
            "[iterative_test_fix] Initial test run",
            test_result.content,
        )
        initial_passed, _, _ = _parse_tap_output(test_result.content)
        last_passed = initial_passed
        if test_result.success:
            logger.info("[iterative_test_fix] Tests passed on first try!")
            return (
                True,
                service_script,
                components_script,
                test_script,
                dummy_data,
                messages,
            )

        logger.info(
            "[iterative_test_fix] Initial tests failed, starting fix loop (max %s attempts)",
            max_attempts,
        )

        fix_messages = copy.deepcopy(messages)
        fix_messages.append(
            Message(
                role=MessageRole.USER,
                content=(
                    "You are an expert JavaScript QA developer tasked with fixing the "
                    "failing tests with the following output:\n\n"
                    f"```\n{test_result.content}\n```\n\n"
                    "Use the provided tools to fix the failing tests. "
                    "If there are tests missing that should be present, add them to the test script. "
                    "If the service or components code is incorrect, fix them accordingly. "
                    "If test fixtures need updates, add data to the dummy data module. Do not delete from dummy data. "
                    "If the test are wrong, correct them. "
                    "Call run_tests to verify your changes. "
                    "When all tests pass, respond with 'TESTS_PASSING' and nothing else."
                ),
            )
        )

        attempt = 0

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

        while attempt < max_attempts:
            attempt += 1
            logger.info(
                "[iterative_test_fix] Fix iteration %s/%s", attempt, max_attempts
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
                response = await tgi_service.llm_client.client.chat.completions.create(
                    **tgi_service.llm_client._build_request_params(chat_request)
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
            fix_messages.append(assistant_msg)

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
                        logger.info(
                            "[iterative_test_fix] LLM indicates tests are passing"
                        )
                        final_result = toolkit.run_tests()
                        _log_test_run_output(
                            "[iterative_test_fix] Verification test run",
                            final_result.content,
                        )
                        passed, _, _ = _parse_tap_output(final_result.content)
                        _reset_attempts_on_progress(passed, "verification run")
                        if final_result.success:
                            return (
                                True,
                                toolkit.current_service_script,
                                toolkit.current_components_script,
                                toolkit.current_test_script,
                                toolkit.current_dummy_data,
                                fix_messages,
                            )
                    fix_messages.append(
                        Message(
                            role=MessageRole.USER,
                            content=(
                                "Tests are still failing:\n"
                                f"```\n{final_result.content}\n```\nContinue fixing."
                            ),
                        )
                    )
                else:
                    fix_messages.append(
                        Message(
                            role=MessageRole.USER,
                            content=(
                                "Please use the tools to fix the tests or respond "
                                "with 'TESTS_PASSING' if done."
                            ),
                        )
                    )
                attempt -= 1
                continue
            no_tool_call_attempts = 0

            tool_results = []
            all_tests_passed = False
            has_modification_tools = False

            # Don't count attempts that only read/search code
            read_only_tools = {
                "get_script_lines",
                "get_current_scripts",
                "search_files",
            }
            if tool_calls and all(
                tc.function.name in read_only_tools for tc in tool_calls
            ):
                logger.info(
                    "[iterative_test_fix] Only read/search tools used - not counting as attempt"
                )
                attempt -= 1

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
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

                tool_desc = _tool_description(tool_name, arguments)
                logger.info(f"[iterative_test_fix] {tool_desc}")

                if event_queue:
                    await event_queue.put(
                        {
                            "event": "tool_start",
                            "tool": tool_name,
                            "description": tool_desc,
                        }
                    )

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
                    _log_test_run_output(
                        f"[iterative_test_fix] Test run (iteration {attempt})",
                        result.content,
                    )
                    passed, failed, failed_tests = _parse_tap_output(result.content)
                    _reset_attempts_on_progress(passed, f"iteration {attempt}")
                    if result.success:
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
                passed, _, _ = _parse_tap_output(final_result.content)
                _reset_attempts_on_progress(passed, "verification run")
                if final_result.success:
                    return (
                        True,
                        toolkit.current_service_script,
                        toolkit.current_components_script,
                        toolkit.current_test_script,
                        toolkit.current_dummy_data,
                        fix_messages,
                    )
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
            "[iterative_test_fix] Failed to fix tests after %s attempts", max_attempts
        )
        return (
            False,
            toolkit.current_service_script,
            toolkit.current_components_script,
            toolkit.current_test_script,
            toolkit.current_dummy_data,
            fix_messages,
        )

    finally:
        toolkit.cleanup()
