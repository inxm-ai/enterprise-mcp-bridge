"""
Tool resolution module for handling different tool call formats from various AI models.

This module provides a unified way to detect, parse, and combine tool calls from different
formats including:
- OpenAI-style tool calls in JSON format
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("uvicorn.error")


class ToolCallFormat(Enum):
    """Enum representing different tool call formats."""

    OPENAI_JSON = "openai_json"
    UNKNOWN = "unknown"


@dataclass
class ParsedToolCall:
    """Standardized tool call representation."""

    id: str
    index: int
    name: str
    arguments: Dict[str, Any]
    format: ToolCallFormat
    raw_content: str


class ToolResolutionStrategy:
    """Main class for resolving tool calls from various formats."""

    def __init__(self):
        self.logger = logger

    def detect_format(
        self, content: str, tool_call_chunks: Optional[Dict] = None
    ) -> ToolCallFormat:
        """
        Detect the format of tool calls in the content.

        Args:
            content: The content string to analyze
            tool_call_chunks: Optional existing tool call chunks from streaming

        Returns:
            ToolCallFormat indicating the detected format
        """
        has_openai_format = bool(tool_call_chunks) or self._has_openai_json_format(
            content
        )

        if has_openai_format:
            return ToolCallFormat.OPENAI_JSON
        else:
            logger.warning(
                "[ToolResolution] No known tool call format detected. Chunks: %s - Content: %s",
                tool_call_chunks,
                content,
            )
            return ToolCallFormat.UNKNOWN

    def resolve_tool_calls(
        self, content: str, tool_call_chunks: Optional[Dict] = None
    ) -> Tuple[List[ParsedToolCall], bool]:
        """
        Main method to resolve tool calls from content and chunks.

        Args:
            content: The content string containing potential tool calls
            tool_call_chunks: Optional streaming tool call chunks

        Returns:
            Tuple of (list of parsed tool calls, success flag)
        """
        try:
            format_type = self.detect_format(content, tool_call_chunks)
            self.logger.debug(f"[ToolResolution] Detected format: {format_type}")

            tool_calls = []

            if format_type == ToolCallFormat.OPENAI_JSON:
                tool_calls = self._resolve_openai_format(content, tool_call_chunks)

            success = len(tool_calls) > 0
            self.logger.debug(f"[ToolResolution] Resolved {len(tool_calls)} tool calls")

            return tool_calls, success

        except Exception as e:
            self.logger.error(f"[ToolResolution] Error resolving tool calls: {e}")
            return [], False

    def _has_openai_json_format(self, content: str) -> bool:
        """Check if content contains OpenAI-style JSON tool calls."""
        # Look for JSON objects with tool call structure
        json_objects = self._extract_json_objects(content)
        for json_obj in json_objects:
            if self._is_valid_openai_tool_call(json_obj):
                return True
        return False

    def _resolve_openai_format(
        self, content: str, tool_call_chunks: Optional[Dict] = None
    ) -> List[ParsedToolCall]:
        """Resolve OpenAI-style tool calls."""
        tool_calls = []

        # First, handle streaming chunks if available
        if tool_call_chunks:
            for index, chunk_data in tool_call_chunks.items():
                try:
                    parsed_call = ParsedToolCall(
                        id=chunk_data.get("id", f"chunk_{index}"),
                        index=int(index),
                        name=chunk_data.get("name", ""),
                        arguments=self._parse_arguments(
                            chunk_data.get("arguments", "{}")
                        ),
                        format=ToolCallFormat.OPENAI_JSON,
                        raw_content=str(chunk_data),
                    )
                    tool_calls.append(parsed_call)
                except Exception as e:
                    self.logger.warning(
                        f"[ToolResolution] Error parsing chunk {index}: {e}"
                    )

        # Then, look for complete JSON tool calls in content
        json_objects = self._extract_json_objects(content)
        for json_obj in json_objects:
            if self._is_valid_openai_tool_call(json_obj):
                try:
                    function_data = json_obj.get("function", {})
                    parsed_call = ParsedToolCall(
                        id=json_obj.get("id", f"content_{len(tool_calls)}"),
                        index=json_obj.get("index", len(tool_calls)),
                        name=function_data.get("name", ""),
                        arguments=self._parse_arguments(
                            function_data.get("arguments", "{}")
                        ),
                        format=ToolCallFormat.OPENAI_JSON,
                        raw_content=json.dumps(json_obj),
                    )
                    tool_calls.append(parsed_call)
                except Exception as e:
                    self.logger.warning(
                        f"[ToolResolution] Error parsing JSON tool call: {e}"
                    )

        return tool_calls

    def _extract_json_objects(self, content: str) -> List[Dict]:
        """Extract all valid JSON objects from content, respecting quoted strings."""
        json_objects = []

        # Find potential JSON by balancing braces while skipping quoted text
        start = None
        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(content):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True if in_string else False
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                # Ignore braces inside quoted strings
                continue

            if char == "{":
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start is not None:
                    try:
                        json_str = content[start : i + 1]
                        json_obj = json.loads(json_str)
                        json_objects.append(json_obj)
                    except json.JSONDecodeError:
                        pass
                    start = None

        return json_objects

    def _is_valid_openai_tool_call(self, json_obj: Dict) -> bool:
        """Check if a JSON object is a valid OpenAI tool call."""
        required_keys = ["id", "type", "function"]
        if not all(key in json_obj for key in required_keys):
            return False

        function_obj = json_obj.get("function", {})
        return "name" in function_obj and "arguments" in function_obj

    def _parse_arguments(self, arguments_str: str) -> Dict[str, Any]:
        """Parse arguments string into a dictionary."""
        if not arguments_str:
            return {}

        try:
            if isinstance(arguments_str, str):
                parsed = json.loads(arguments_str)
            elif isinstance(arguments_str, dict):
                parsed = arguments_str
            else:
                parsed = arguments_str
        except (json.JSONDecodeError, TypeError):
            return {"raw": str(arguments_str)}

        if parsed is None:
            return {}

        return parsed
