import json
import pytest
from typing import AsyncGenerator, List

from app.tgi.well_planned_orchestrator import read_todo_stream, create_response_chunk
from app.tgi.chunk_reader import ParsedChunk


async def sse_json_stream() -> AsyncGenerator[str, None]:
    # Two JSON content chunks then DONE
    yield create_response_chunk("mcp-1", "Hello")
    yield create_response_chunk("mcp-1", " world")
    yield create_response_chunk("mcp-1", "[DONE]")


async def raw_text_stream() -> AsyncGenerator[str, None]:
    # Non-JSON raw text should be passed through as raw content
    yield "data: some plain text chunk\n\n"
    yield "data: [DONE]\n\n"


async def mixed_stream() -> AsyncGenerator[str, None]:
    # JSON chunk, malformed chunk, then done
    yield create_response_chunk("mcp-2", "PartA")
    yield "data: {this is not json}\n\n"
    yield "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_read_todo_stream_parses_json_chunks():
    parts: List[ParsedChunk] = []
    async for parsed in read_todo_stream(sse_json_stream()):
        parts.append(parsed)

    # Expect three parsed chunks (two content + one done)
    assert len(parts) == 3
    assert parts[0].content == "Hello"
    assert parts[1].content == " world"
    # The final chunk may be marked done
    assert parts[-1].is_done is True or parts[-1].content == "[DONE]"


@pytest.mark.asyncio
async def test_read_todo_stream_handles_raw_text():
    parts: List[ParsedChunk] = []
    async for parsed in read_todo_stream(raw_text_stream()):
        parts.append(parsed)

    assert len(parts) == 2
    # Raw chunk should have content equal to the raw string without leading 'data: '
    assert "plain text chunk" in parts[0].content
    assert parts[1].is_done is True


@pytest.mark.asyncio
async def test_read_todo_stream_mixed_malformed():
    parts: List[ParsedChunk] = []
    async for parsed in read_todo_stream(mixed_stream()):
        parts.append(parsed)

    assert len(parts) == 3
    assert parts[0].content == "PartA"
    # Malformed JSON should appear as raw content
    assert "this is not json" in parts[1].content
    assert parts[2].is_done is True
