"""
Tool resolution module for handling different tool call formats from various AI models.

This module provides a unified way to detect, parse, and combine tool calls from different
formats including:
- OpenAI-style tool calls in JSON format
- Claude-style XML tags (<function_name>, <create_entities>, etc.)
- Mixed formats and edge cases
"""

import json
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("uvicorn.error")


class ToolCallFormat(Enum):
    """Enum representing different tool call formats."""

    OPENAI_JSON = "openai_json"
    CLAUDE_XML = "claude_xml"
    MIXED = "mixed"
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
        self._xml_tag_patterns = [
            # Generic function call patterns
            # Allow tag names with non-alpha characters (e.g. create-entities, create+entity)
            r"<([^>]+)>(.*?)</\1>",
            # Function call with explicit arguments
            r'<function_calls?>\s*<invoke\s+name="([^"]+)"[^>]*>(.*?)</invoke>\s*</function_calls?>',
        ]

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
        has_claude_format = self._has_claude_xml_format(content)

        if has_openai_format and has_claude_format:
            return ToolCallFormat.MIXED
        elif has_openai_format:
            return ToolCallFormat.OPENAI_JSON
        elif has_claude_format:
            return ToolCallFormat.CLAUDE_XML
        else:
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
            elif format_type == ToolCallFormat.CLAUDE_XML:
                tool_calls = self._resolve_claude_xml_format(content)
            elif format_type == ToolCallFormat.MIXED:
                # Combine both formats
                openai_calls = self._resolve_openai_format(content, tool_call_chunks)
                claude_calls = self._resolve_claude_xml_format(content)
                tool_calls = self._merge_tool_calls(openai_calls, claude_calls)

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

    def _has_claude_xml_format(self, content: str) -> bool:
        """Check if content contains Claude-style XML tags."""
        for pattern in self._xml_tag_patterns:
            if re.search(pattern, content, re.DOTALL | re.IGNORECASE):
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

    def _resolve_claude_xml_format(self, content: str) -> List[ParsedToolCall]:
        """Resolve Claude-style XML tool calls."""
        tool_calls = []

        # First, handle Anthropic-style function invocations (highest priority)
        invoke_matches = re.findall(
            self._xml_tag_patterns[1], content, re.DOTALL | re.IGNORECASE
        )

        # Keep track of function_calls content that was processed by invoke pattern
        processed_invoke_content = set()

        for function_name, function_content in invoke_matches:
            try:
                arguments = self._parse_xml_content(function_content.strip())

                parsed_call = ParsedToolCall(
                    id=f"claude_invoke_{function_name}_{len(tool_calls)}",
                    index=len(tool_calls),
                    name=function_name,
                    arguments=arguments,
                    format=ToolCallFormat.CLAUDE_XML,
                    raw_content=f'<invoke name="{function_name}">{function_content}</invoke>',
                )
                tool_calls.append(parsed_call)

                # Mark this content as processed to avoid double-processing
                processed_invoke_content.add(function_content.strip())
            except Exception as e:
                self.logger.warning(
                    f"[ToolResolution] Error parsing invoke call {function_name}: {e}"
                )

        generic_matches = re.findall(
            self._xml_tag_patterns[0], content, re.DOTALL | re.IGNORECASE
        )

        for tag_name, tag_content in generic_matches:
            # Skip if this is a function_calls tag that was handled by invoke pattern
            if tag_name.lower() == "function_calls" and invoke_matches:
                continue

            # Skip common HTML/XML tags that aren't function calls
            if tag_name.lower() in [
                "think",
                "reasoning",
                "analysis",
                "p",
                "div",
                "span",
                "invoke",
            ]:
                continue

            try:
                arguments = self._parse_xml_content(tag_content.strip())

                parsed_call = ParsedToolCall(
                    id=f"claude_{tag_name}_{len(tool_calls)}",
                    index=len(tool_calls),
                    name=tag_name,
                    arguments=arguments,
                    format=ToolCallFormat.CLAUDE_XML,
                    raw_content=f"<{tag_name}>{tag_content}</{tag_name}>",
                )
                tool_calls.append(parsed_call)
            except Exception as e:
                self.logger.warning(
                    f"[ToolResolution] Error parsing generic XML call {tag_name}: {e}"
                )

        return tool_calls

    def _merge_tool_calls(
        self, openai_calls: List[ParsedToolCall], claude_calls: List[ParsedToolCall]
    ) -> List[ParsedToolCall]:
        """Merge tool calls from different formats, removing duplicates."""
        all_calls = openai_calls + claude_calls

        # Simple deduplication based on name and arguments similarity
        unique_calls = []
        seen_signatures = set()

        for call in all_calls:
            # Create a signature based on name and arguments
            signature = (
                f"{call.name}:{hash(json.dumps(call.arguments, sort_keys=True))}"
            )

            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_calls.append(call)
            else:
                self.logger.debug(
                    f"[ToolResolution] Skipping duplicate tool call: {call.name}"
                )

        # Re-index the calls
        for i, call in enumerate(unique_calls):
            call.index = i

        return unique_calls

    def _extract_json_objects(self, content: str) -> List[Dict]:
        """Extract all valid JSON objects from content."""
        json_objects = []

        # Find potential JSON by balancing braces
        start = None
        brace_count = 0

        for i, char in enumerate(content):
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

    def _parse_xml_content(self, content: str) -> Dict[str, Any]:
        """Parse XML tag content into arguments dictionary."""
        content = content.strip()

        if not content:
            return {}

        # Try parsing as JSON first
        try:
            if content.startswith("{") and content.endswith("}"):
                return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try parsing as key-value pairs
        try:
            # Look for parameter tags
            param_pattern = r'<parameter\s+name="([^"]+)">(.*?)</parameter>'
            param_matches = re.findall(
                param_pattern, content, re.DOTALL | re.IGNORECASE
            )

            if param_matches:
                result = {}
                for param_name, param_value in param_matches:
                    param_value = param_value.strip()
                    # Only parse obvious JSON structures (objects, arrays, quoted strings), leave other values as strings
                    try:
                        if param_value.startswith(("{", "[", '"')):
                            result[param_name] = json.loads(param_value)
                        else:
                            # Keep as string to match test expectations
                            result[param_name] = param_value
                    except json.JSONDecodeError:
                        result[param_name] = param_value
                return result
        except Exception:
            pass

        # Try parsing as comma-separated key=value pairs
        try:
            if "=" in content and ("," in content or content.count("=") == 1):
                pairs = content.split(",")
                result = {}
                for pair in pairs:
                    if "=" in pair:
                        key, value = pair.split("=", 1)
                        result[key.strip()] = value.strip()
                return result
        except Exception:
            pass

        # Fallback to raw content
        return {"raw": content}
