"""
ChunkReader: A pythonic helper for reading streaming LLM responses.

Supports multiple formats:
- OpenAI (SSE with "data: " prefix)
- A2A (JSON-RPC 2.0 format)
- TGI (Text Generation Inference format)

Usage:
    # As JSON with format conversion
    async with chunk_reader(stream_generator) as reader:
        async for chunk in reader.as_json(ChunkFormat.A2A):
            yield chunk

    # As string content accumulation
    async with chunk_reader(stream_generator) as reader:
        async for content_piece in reader.as_str():
            full_content += content_piece

    # As raw chunks (passthrough)
    async with chunk_reader(stream_generator) as reader:
        async for raw_chunk in reader.as_raw():
            yield raw_chunk
"""

import json
import logging
from datetime import datetime, timezone
from enum import Enum
import time
from typing import AsyncGenerator, Optional, Any, Dict, Set
from app.vars import TGI_MODEL_NAME
from opentelemetry import trace

# NOTE: OpenTelemetry context detachment warnings may appear in logs when async generators
# exit early (e.g., streaming completes or client disconnects). These are cosmetic and do not
# affect functionality. They occur because OpenTelemetry tries to clean up span contexts that
# were created in a different async context. This is normal behavior for async generators.

logger = logging.getLogger("uvicorn.error")
tracer = trace.get_tracer(__name__)


class ChunkFormat(Enum):
    """Supported chunk formats."""

    OPENAI = "openai"
    A2A = "a2a"
    TGI = "tgi"
    RAW = "raw"


class ParsedChunk:
    """
    A parsed chunk with unified access to common fields.
    """

    def __init__(
        self,
        raw: str,
        parsed: Optional[dict] = None,
        content: Optional[str] = None,
        tool_calls: Optional[list] = None,
        finish_reason: Optional[str] = None,
        is_done: bool = False,
        accumulated_tool_calls: Optional[Dict[int, dict]] = None,
        tool_result: Optional[dict] = None,
    ):
        self.raw = raw
        self.parsed = parsed or {}
        self.content = content
        self.tool_calls = tool_calls
        self.finish_reason = finish_reason
        self.is_done = is_done
        # Accumulated tool calls: {index: {id, name, arguments}}
        self.accumulated_tool_calls = accumulated_tool_calls or {}
        # Tool execution result: {name: str, content: str}
        self.tool_result = tool_result

    def __repr__(self):
        return f"ParsedChunk(content={self.content!r}, is_done={self.is_done}, tool_calls={len(self.tool_calls or [])}, tool_result={self.tool_result is not None})"


class ChunkReader:
    """
    A context manager for reading streaming LLM responses.

    Provides pythonic interfaces for different consumption patterns:
    - as_json(format): Convert chunks to specific format
    - as_str(): Extract content strings only
    - as_raw(): Pass through raw chunks
    - as_parsed(): Get unified ParsedChunk objects
    """

    def __init__(self, source: AsyncGenerator[Any, None], enable_tracing: bool = True):
        """
        Initialize the chunk reader.

        Args:
            source: An async generator yielding raw chunks (str, bytes, or dict)
            enable_tracing: Whether to enable OpenTelemetry tracing (default: True)
        """
        self.source = source
        self._entered = False
        self.enable_tracing = enable_tracing
        # Tool call accumulator: {index: {id, name, arguments}}
        self._tool_call_chunks: Dict[int, dict] = {}
        self._tool_call_ready: Set[int] = set()

    async def __aenter__(self):
        """Enter the async context."""
        self._entered = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context."""
        self._entered = False  # Mark as exited
        # Clean up if needed, but suppress GeneratorExit
        if hasattr(self.source, "aclose"):
            try:
                await self.source.aclose()
            except GeneratorExit:
                # This is expected when the generator is closed early
                pass
            except Exception as e:
                logger.debug(f"[ChunkReader] Error closing source: {e}")
        return False

    def _accumulate_tool_calls(self, tool_calls: Optional[list]) -> None:
        """
        Accumulate tool call chunks across multiple parsed chunks.

        Args:
            tool_calls: List of tool call delta chunks from the current parsed chunk
        """
        if not tool_calls:
            return

        for tc in tool_calls:
            tc_index = tc.get("index")
            if tc_index is None:
                continue

            if tc_index not in self._tool_call_chunks:
                self._tool_call_chunks[tc_index] = {
                    "index": tc_index,
                    "name": "",
                    "arguments": "",
                }

            # Accumulate ID (only set once, not concatenated)
            if "id" in tc and tc["id"]:
                self._tool_call_chunks[tc_index]["id"] = tc["id"]

            # Accumulate function name and arguments (concatenate for streaming)
            tc_func = tc.get("function", {})
            if "name" in tc_func and tc_func["name"]:
                self._tool_call_chunks[tc_index]["name"] += tc_func["name"]
            if "arguments" in tc_func and tc_func["arguments"]:
                self._tool_call_chunks[tc_index]["arguments"] += tc_func["arguments"]

            # Mark as ready if we have both name and arguments
            if (
                self._tool_call_chunks[tc_index]["name"]
                and self._tool_call_chunks[tc_index]["arguments"]
                and tc_index not in self._tool_call_ready
            ):
                self._tool_call_ready.add(tc_index)

    def get_accumulated_tool_calls(self) -> Dict[int, dict]:
        """
        Get the currently accumulated tool calls.

        Returns:
            Dictionary mapping tool call index to accumulated tool call data
        """
        return self._tool_call_chunks.copy()

    def get_ready_tool_calls(self) -> Set[int]:
        """
        Get the set of tool call indices that are ready (have both name and arguments).

        Returns:
            Set of ready tool call indices
        """
        return self._tool_call_ready.copy()

    def clear_tool_calls(self) -> None:
        """Clear accumulated tool calls (useful when starting a new iteration)."""
        self._tool_call_chunks.clear()
        self._tool_call_ready.clear()

    def _normalize_chunk(self, raw_chunk: Any) -> str:
        """
        Normalize a chunk to a string.

        Args:
            raw_chunk: Raw chunk (str, bytes, bytearray, or dict)

        Returns:
            Normalized string representation
        """
        if isinstance(raw_chunk, (bytes, bytearray)):
            return raw_chunk.decode("utf-8")
        elif isinstance(raw_chunk, dict):
            return json.dumps(raw_chunk)
        else:
            return str(raw_chunk)

    def _parse_sse_chunk(self, chunk_str: str) -> Optional[ParsedChunk]:
        """
        Parse a Server-Sent Events (SSE) chunk.

        Args:
            chunk_str: Raw chunk string

        Returns:
            ParsedChunk or None if not parseable
        """
        chunk_str = chunk_str.strip()

        # Check for [DONE] marker
        if chunk_str == "data: [DONE]" or chunk_str == "[DONE]":
            return ParsedChunk(raw=chunk_str, is_done=True)

        # Extract JSON from SSE format
        if chunk_str.startswith("data: "):
            json_str = chunk_str[6:].strip()
        else:
            json_str = chunk_str

        # Try to parse as JSON
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            # Not valid JSON, treat as raw content (use json_str without "data: " prefix)
            return ParsedChunk(raw=chunk_str, content=json_str)

        # Extract common fields
        content = None
        tool_calls = None
        finish_reason = None
        tool_result = None

        # OpenAI format
        choices = parsed.get("choices", [])
        if choices:
            choice = choices[0]
            delta = choice.get("delta", {})
            message = choice.get("message", {})
            if not message:
                message = {}

            # Try delta.content first (streaming)
            content = delta.get("content") or message.get("content")

            # Try tool_calls
            tool_calls = delta.get("tool_calls") or message.get("tool_calls")

            # Try tool_result (emitted by tool_chat_runner after tool execution)
            tool_result = delta.get("tool_result")

            # Finish reason
            finish_reason = choice.get("finish_reason")

        # A2A format
        if "result" in parsed:
            result = parsed.get("result", {})
            if isinstance(result, dict):
                content = result.get("completion")
            elif isinstance(result, str):
                content = result

        return ParsedChunk(
            raw=chunk_str,
            parsed=parsed,
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            is_done=False,
            tool_result=tool_result,
        )

    async def as_parsed(self) -> AsyncGenerator[ParsedChunk, None]:
        """
        Yield parsed chunks with unified access to common fields.
        Automatically accumulates tool calls across chunks.

        Yields:
            ParsedChunk objects with accumulated tool call state
        """
        if not self._entered:
            raise RuntimeError("ChunkReader must be used as async context manager")

        span_name = "chunk_reader.as_parsed"
        if self.enable_tracing:
            # Manually manage span to avoid context detachment issues in async generators
            span = tracer.start_span(span_name)
            chunk_count = 0
            chunks_with_content = 0
            chunks_with_tools = 0
            try:
                async for raw_chunk in self.source:
                    chunk_count += 1
                    chunk_str = self._normalize_chunk(raw_chunk)
                    parsed = self._parse_sse_chunk(chunk_str)
                    if parsed:
                        # Accumulate tool calls
                        if parsed.tool_calls:
                            self._accumulate_tool_calls(parsed.tool_calls)
                            chunks_with_tools += 1

                        # Add accumulated state to parsed chunk
                        parsed.accumulated_tool_calls = (
                            self.get_accumulated_tool_calls()
                        )

                        if parsed.content:
                            chunks_with_content += 1

                        yield parsed
                    else:
                        logger.debug(f"[ChunkReader] Unparseable chunk: {chunk_str!r}")
                        yield {"raw": chunk_str, "content": chunk_str, "is_done": False}

                span.set_attribute("chunk_reader.total_chunks", chunk_count)
                span.set_attribute(
                    "chunk_reader.chunks_with_content", chunks_with_content
                )
                span.set_attribute("chunk_reader.chunks_with_tools", chunks_with_tools)
                span.set_attribute(
                    "chunk_reader.tool_calls_accumulated", len(self._tool_call_chunks)
                )
            except GeneratorExit:
                # Generator closed early (normal for streaming)
                span.set_attribute("chunk_reader.early_exit", True)
                raise
            finally:
                span.end()
        else:
            # No tracing version
            async for raw_chunk in self.source:
                chunk_str = self._normalize_chunk(raw_chunk)
                parsed = self._parse_sse_chunk(chunk_str)
                if parsed:
                    # Accumulate tool calls
                    if parsed.tool_calls:
                        self._accumulate_tool_calls(parsed.tool_calls)

                    # Add accumulated state to parsed chunk
                    parsed.accumulated_tool_calls = self.get_accumulated_tool_calls()

                    yield parsed

    async def as_str(self) -> AsyncGenerator[str, None]:
        """
        Yield only content strings from chunks.

        Yields:
            Content strings (empty strings are filtered out)
        """
        span_name = "chunk_reader.as_str"
        if self.enable_tracing:
            span = tracer.start_span(span_name)
            content_chunks = 0
            try:
                async for parsed in self.as_parsed():
                    if parsed.is_done:
                        break
                    if parsed.content:
                        content_chunks += 1
                        yield parsed.content

                span.set_attribute("chunk_reader.content_chunks", content_chunks)
            except GeneratorExit:
                span.set_attribute("chunk_reader.early_exit", True)
                raise
            finally:
                span.end()
        else:
            async for parsed in self.as_parsed():
                if parsed.is_done:
                    break
                if parsed.content:
                    yield parsed.content

    async def as_raw(self) -> AsyncGenerator[str, None]:
        """
        Yield raw chunk strings (passthrough mode).

        Yields:
            Raw chunk strings
        """
        if not self._entered:
            raise RuntimeError("ChunkReader must be used as async context manager")

        span_name = "chunk_reader.as_raw"
        if self.enable_tracing:
            span = tracer.start_span(span_name)
            raw_chunk_count = 0
            try:
                async for raw_chunk in self.source:
                    raw_chunk_count += 1
                    yield self._normalize_chunk(raw_chunk)

                span.set_attribute("chunk_reader.raw_chunks", raw_chunk_count)
            except GeneratorExit:
                span.set_attribute("chunk_reader.early_exit", True)
                raise
            finally:
                span.end()
        else:
            async for raw_chunk in self.source:
                yield self._normalize_chunk(raw_chunk)

    async def as_json(
        self, output_format: ChunkFormat, request_id: str = "unknown"
    ) -> AsyncGenerator[str, None]:
        """
        Convert chunks to a specific JSON format.

        Args:
            output_format: Target format (OPENAI, A2A, TGI)
            request_id: Request ID for A2A format

        Yields:
            Formatted JSON strings
        """
        span_name = "chunk_reader.as_json"
        if self.enable_tracing:
            span = tracer.start_span(span_name)
            span.set_attribute("chunk_reader.output_format", output_format.value)
            span.set_attribute("chunk_reader.request_id", request_id)
            converted_chunks = 0

            try:
                async for parsed in self.as_parsed():
                    if parsed.is_done:
                        if (
                            output_format == ChunkFormat.OPENAI
                            or output_format == ChunkFormat.TGI
                        ):
                            yield "data: [DONE]\n\n"
                        elif output_format == ChunkFormat.A2A:
                            # A2A doesn't have a [DONE] marker in streaming
                            pass
                        break

                    converted_chunks += 1
                    if output_format == ChunkFormat.OPENAI:
                        yield self._to_openai_format(parsed)
                    elif output_format == ChunkFormat.A2A:
                        yield self._to_a2a_format(parsed, request_id)
                    elif output_format == ChunkFormat.TGI:
                        yield self._to_tgi_format(parsed)
                    elif output_format == ChunkFormat.RAW:
                        yield parsed.raw

                span.set_attribute("chunk_reader.converted_chunks", converted_chunks)
            except GeneratorExit:
                span.set_attribute("chunk_reader.early_exit", True)
                raise
            finally:
                span.end()
        else:
            async for parsed in self.as_parsed():
                if parsed.is_done:
                    if (
                        output_format == ChunkFormat.OPENAI
                        or output_format == ChunkFormat.TGI
                    ):
                        yield "data: [DONE]\n\n"
                    elif output_format == ChunkFormat.A2A:
                        # A2A doesn't have a [DONE] marker in streaming
                        pass
                    break

                if output_format == ChunkFormat.OPENAI:
                    yield self._to_openai_format(parsed)
                elif output_format == ChunkFormat.A2A:
                    yield self._to_a2a_format(parsed, request_id)
                elif output_format == ChunkFormat.TGI:
                    yield self._to_tgi_format(parsed)
                elif output_format == ChunkFormat.RAW:
                    yield parsed.raw

    def _to_openai_format(self, parsed: ParsedChunk) -> str:
        """
        Convert parsed chunk to OpenAI SSE format.

        Args:
            parsed: ParsedChunk to convert

        Returns:
            SSE formatted string
        """
        # If already in OpenAI format, return as-is
        if parsed.raw.startswith("data: "):
            return parsed.raw if parsed.raw.endswith("\n\n") else f"{parsed.raw}\n\n"

        # Build OpenAI format chunk
        chunk_dict = {
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": parsed.finish_reason,
                }
            ]
        }

        if parsed.content:
            chunk_dict["choices"][0]["delta"]["content"] = parsed.content

        if parsed.tool_calls:
            chunk_dict["choices"][0]["delta"]["tool_calls"] = parsed.tool_calls

        return f"data: {json.dumps(chunk_dict)}\n\n"

    def _to_a2a_format(self, parsed: ParsedChunk, request_id: str) -> str:
        """
        Convert parsed chunk to A2A JSON-RPC format.

        Args:
            parsed: ParsedChunk to convert
            request_id: Request ID for the response

        Returns:
            A2A formatted JSON string with newline
        """
        parsed_payload = parsed.parsed if isinstance(parsed.parsed, dict) else {}
        agentic = (
            parsed_payload.get("agentic") if isinstance(parsed_payload, dict) else {}
        )
        content = parsed.content or ""

        if agentic:
            context_id = (
                agentic.get("context_id")
                or agentic.get("workflow_id")
                or parsed_payload.get("id")
            )
            task_id = agentic.get("task_id") or parsed_payload.get("id") or request_id
            history_parts = agentic.get("parts") or [{"kind": "text", "text": content}]
            status_state = agentic.get("status") or "in_progress"
            result = {
                "id": task_id,
                "contextId": context_id,
                "status": {
                    "state": status_state,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "history": [
                    {
                        "role": agentic.get("role") or "assistant",
                        "parts": history_parts,
                        "messageId": task_id,
                        "taskId": task_id,
                        "contextId": context_id,
                    }
                ],
                "kind": agentic.get("kind") or "task",
                "metadata": agentic.get("metadata") or {},
            }
            if agentic.get("error"):
                result["error"] = agentic["error"]
        else:
            result = {"completion": content}

        a2a_dict = {"jsonrpc": "2.0", "result": result, "id": request_id}

        return f"data: {json.dumps(a2a_dict)}\n\n"

    def _to_tgi_format(self, parsed: ParsedChunk) -> str:
        """
        Convert parsed chunk to TGI format.

        Args:
            parsed: ParsedChunk to convert

        Returns:
            TGI formatted string
        """
        # TGI format is similar to OpenAI SSE
        return self._to_openai_format(parsed)


def chunk_reader(
    source: AsyncGenerator[Any, None], enable_tracing: bool = True
) -> ChunkReader:
    """
    Create a ChunkReader context manager.

    Args:
        source: An async generator yielding raw chunks
        enable_tracing: Whether to enable OpenTelemetry tracing (default: True)

    Returns:
        ChunkReader instance

    Example:
        async with chunk_reader(stream_generator) as reader:
            async for content in reader.as_str():
                print(content)
    """
    return ChunkReader(source, enable_tracing=enable_tracing)


async def accumulate_content(source: AsyncGenerator[Any, None]) -> str:
    """
    Accumulate all content from a streaming source into a single string.

    Args:
        source: An async generator yielding raw chunks

    Returns:
        Accumulated content string

    Example:
        content = await accumulate_content(stream_generator)
    """
    result = ""
    async with chunk_reader(source) as reader:
        async for content_piece in reader.as_str():
            result += content_piece
    return result


async def collect_parsed_chunks(source: AsyncGenerator[Any, None]) -> list[ParsedChunk]:
    """
    Collect all parsed chunks into a list.

    Args:
        source: An async generator yielding raw chunks

    Returns:
        List of ParsedChunk objects

    Example:
        chunks = await collect_parsed_chunks(stream_generator)
    """
    result = []
    async with chunk_reader(source) as reader:
        async for parsed in reader.as_parsed():
            if not parsed.is_done:
                result.append(parsed)
    return result


def create_response_chunk(id: str, content: str) -> str:
    if content == "[DONE]":
        return "data: [DONE]\n\n"
    chunk_dict = {
        "id": id,
        "model": TGI_MODEL_NAME,
        "created": int(time.time()),
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"
