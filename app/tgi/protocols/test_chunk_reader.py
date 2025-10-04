"""
Unit tests for chunk_reader module.

Tests cover:
- Different chunk formats (OpenAI, A2A, TGI)
- Content extraction
- Format conversion
- Error handling
- Edge cases
"""

import json
import pytest
from typing import AsyncGenerator, List

from app.tgi.protocols.chunk_reader import (
    chunk_reader,
    ChunkFormat,
    ParsedChunk,
    accumulate_content,
    collect_parsed_chunks,
)


# Test fixtures - async generators for different formats


async def openai_stream_generator() -> AsyncGenerator[str, None]:
    """Generate OpenAI-style SSE chunks."""
    yield 'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'
    yield 'data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n'
    yield 'data: {"choices":[{"delta":{"content":"!"},"index":0,"finish_reason":"stop"}]}\n\n'
    yield "data: [DONE]\n\n"


async def openai_tool_call_stream() -> AsyncGenerator[str, None]:
    """Generate OpenAI-style chunks with tool calls."""
    yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather"}}]},"index":0}]}\n\n'
    yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\""}}]},"index":0}]}\n\n'
    yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\\"NYC\\"}"}}]},"index":0}]}\n\n'
    yield 'data: {"choices":[{"delta":{},"index":0,"finish_reason":"tool_calls"}]}\n\n'
    yield "data: [DONE]\n\n"


async def a2a_stream_generator() -> AsyncGenerator[str, None]:
    """Generate A2A JSON-RPC chunks."""
    yield '{"jsonrpc":"2.0","result":{"completion":"Hello"},"id":"req-1"}\n'
    yield '{"jsonrpc":"2.0","result":{"completion":" world"},"id":"req-1"}\n'
    yield '{"jsonrpc":"2.0","result":{"completion":"!"},"id":"req-1"}\n'


async def mixed_format_stream() -> AsyncGenerator[str, None]:
    """Generate mixed format chunks (bytes and strings)."""
    yield b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'
    yield 'data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n'
    yield b"data: [DONE]\n\n"


async def dict_stream_generator() -> AsyncGenerator[dict, None]:
    """Generate dict chunks directly."""
    yield {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}
    yield {"choices": [{"delta": {"content": " world"}, "index": 0}]}
    yield {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}


async def malformed_json_stream() -> AsyncGenerator[str, None]:
    """Generate stream with malformed JSON."""
    yield 'data: {"choices":[{"delta":{"content":"Good"},"index":0}]}\n\n'
    yield "data: {this is not valid json}\n\n"
    yield 'data: {"choices":[{"delta":{"content":" chunk"},"index":0}]}\n\n'
    yield "data: [DONE]\n\n"


async def empty_stream() -> AsyncGenerator[str, None]:
    """Generate empty stream."""
    if False:
        yield ""


async def single_chunk_stream() -> AsyncGenerator[str, None]:
    """Generate stream with single chunk."""
    yield 'data: {"choices":[{"delta":{"content":"Single"},"index":0}]}\n\n'
    yield "data: [DONE]\n\n"


# Tests for as_parsed()


@pytest.mark.asyncio
async def test_as_parsed_openai_format():
    """Test parsing OpenAI format chunks."""
    chunks: List[ParsedChunk] = []
    async with chunk_reader(openai_stream_generator()) as reader:
        async for parsed in reader.as_parsed():
            chunks.append(parsed)

    # Should have 4 chunks (3 content + 1 DONE)
    assert len(chunks) == 4
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " world"
    assert chunks[2].content == "!"
    assert chunks[2].finish_reason == "stop"
    assert chunks[3].is_done is True


@pytest.mark.asyncio
async def test_as_parsed_tool_calls():
    """Test parsing tool call chunks."""
    chunks: List[ParsedChunk] = []
    async with chunk_reader(openai_tool_call_stream()) as reader:
        async for parsed in reader.as_parsed():
            chunks.append(parsed)

    # Should have tool_calls in parsed chunks
    assert len(chunks) == 5
    assert chunks[0].tool_calls is not None
    assert chunks[0].tool_calls[0]["id"] == "call_1"
    assert chunks[0].tool_calls[0]["function"]["name"] == "get_weather"
    assert chunks[3].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_as_parsed_a2a_format():
    """Test parsing A2A format chunks."""
    chunks: List[ParsedChunk] = []
    async with chunk_reader(a2a_stream_generator()) as reader:
        async for parsed in reader.as_parsed():
            chunks.append(parsed)

    assert len(chunks) == 3
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " world"
    assert chunks[2].content == "!"
    # Check parsed dict has A2A structure
    assert chunks[0].parsed["jsonrpc"] == "2.0"
    assert chunks[0].parsed["id"] == "req-1"


@pytest.mark.asyncio
async def test_as_parsed_dict_input():
    """Test parsing when input is dict objects."""
    chunks: List[ParsedChunk] = []
    async with chunk_reader(dict_stream_generator()) as reader:
        async for parsed in reader.as_parsed():
            chunks.append(parsed)

    assert len(chunks) == 3
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " world"
    assert chunks[2].finish_reason == "stop"


@pytest.mark.asyncio
async def test_as_parsed_mixed_bytes_strings():
    """Test parsing mixed bytes and string chunks."""
    chunks: List[ParsedChunk] = []
    async with chunk_reader(mixed_format_stream()) as reader:
        async for parsed in reader.as_parsed():
            chunks.append(parsed)

    assert len(chunks) == 3
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " world"
    assert chunks[2].is_done is True


@pytest.mark.asyncio
async def test_as_parsed_malformed_json():
    """Test parsing with malformed JSON (should handle gracefully)."""
    chunks: List[ParsedChunk] = []
    async with chunk_reader(malformed_json_stream()) as reader:
        async for parsed in reader.as_parsed():
            chunks.append(parsed)

    # Should have 4 chunks: good, malformed (as raw), good, done
    assert len(chunks) == 4
    assert chunks[0].content == "Good"
    # Malformed JSON becomes raw content
    assert chunks[1].content == "{this is not valid json}"
    assert chunks[2].content == " chunk"
    assert chunks[3].is_done is True


# Tests for as_str()


@pytest.mark.asyncio
async def test_as_str_content_extraction():
    """Test extracting content strings only."""
    content_parts: List[str] = []
    async with chunk_reader(openai_stream_generator()) as reader:
        async for content in reader.as_str():
            content_parts.append(content)

    assert content_parts == ["Hello", " world", "!"]


@pytest.mark.asyncio
async def test_as_str_stops_at_done():
    """Test that as_str stops at [DONE] marker."""
    content_parts: List[str] = []
    async with chunk_reader(openai_stream_generator()) as reader:
        async for content in reader.as_str():
            content_parts.append(content)

    # Should not include [DONE] in content
    assert len(content_parts) == 3
    assert all(part != "[DONE]" for part in content_parts)


@pytest.mark.asyncio
async def test_as_str_empty_stream():
    """Test as_str with empty stream."""
    content_parts: List[str] = []
    async with chunk_reader(empty_stream()) as reader:
        async for content in reader.as_str():
            content_parts.append(content)

    assert len(content_parts) == 0


@pytest.mark.asyncio
async def test_as_str_filters_empty_content():
    """Test that as_str filters out chunks with no content."""

    async def stream_with_empty_content():
        yield 'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'
        yield 'data: {"choices":[{"delta":{},"index":0}]}\n\n'  # No content
        yield 'data: {"choices":[{"delta":{"content":"world"},"index":0}]}\n\n'

    content_parts: List[str] = []
    async with chunk_reader(stream_with_empty_content()) as reader:
        async for content in reader.as_str():
            content_parts.append(content)

    assert content_parts == ["Hello", "world"]


# Tests for as_raw()


@pytest.mark.asyncio
async def test_as_raw_passthrough():
    """Test raw passthrough mode."""
    raw_chunks: List[str] = []
    async with chunk_reader(openai_stream_generator()) as reader:
        async for raw in reader.as_raw():
            raw_chunks.append(raw)

    assert len(raw_chunks) == 4
    assert raw_chunks[0].startswith('data: {"choices"')
    assert raw_chunks[-1].strip() == "data: [DONE]"


@pytest.mark.asyncio
async def test_as_raw_normalizes_bytes():
    """Test that as_raw normalizes bytes to strings."""
    raw_chunks: List[str] = []
    async with chunk_reader(mixed_format_stream()) as reader:
        async for raw in reader.as_raw():
            raw_chunks.append(raw)

    # All should be strings
    assert all(isinstance(chunk, str) for chunk in raw_chunks)


# Tests for as_json()


@pytest.mark.asyncio
async def test_as_json_to_openai():
    """Test converting to OpenAI format."""
    chunks: List[str] = []
    async with chunk_reader(a2a_stream_generator()) as reader:
        async for chunk in reader.as_json(ChunkFormat.OPENAI):
            chunks.append(chunk)

    # A2A streams don't have [DONE] marker, so we only get content chunks
    assert len(chunks) == 3
    assert chunks[0].startswith("data: ")

    # Verify structure
    chunk_data = json.loads(chunks[0][6:])
    assert "choices" in chunk_data
    assert "delta" in chunk_data["choices"][0]
    assert chunk_data["choices"][0]["delta"]["content"] == "Hello"


@pytest.mark.asyncio
async def test_as_json_to_a2a():
    """Test converting to A2A format."""
    chunks: List[str] = []
    async with chunk_reader(openai_stream_generator()) as reader:
        async for chunk in reader.as_json(ChunkFormat.A2A, request_id="test-123"):
            chunks.append(chunk)

    # Should have 3 content chunks (no DONE in A2A streaming)
    assert len(chunks) == 3

    # Verify structure
    chunk_data = json.loads(chunks[0])
    assert chunk_data["jsonrpc"] == "2.0"
    assert chunk_data["id"] == "test-123"
    assert chunk_data["result"]["completion"] == "Hello"


@pytest.mark.asyncio
async def test_as_json_openai_passthrough():
    """Test that OpenAI format chunks pass through correctly."""
    original_chunks: List[str] = []
    converted_chunks: List[str] = []

    async with chunk_reader(openai_stream_generator()) as reader:
        async for chunk in reader.as_json(ChunkFormat.OPENAI):
            converted_chunks.append(chunk)

    # Collect original for comparison
    async for chunk in openai_stream_generator():
        original_chunks.append(chunk)

    # Should be nearly identical (might have minor formatting differences)
    assert len(converted_chunks) == len(original_chunks)


@pytest.mark.asyncio
async def test_as_json_preserves_tool_calls():
    """Test that tool calls are preserved in format conversion."""
    chunks: List[str] = []
    async with chunk_reader(openai_tool_call_stream()) as reader:
        async for chunk in reader.as_json(ChunkFormat.OPENAI):
            chunks.append(chunk)

    # Parse first chunk with tool call
    chunk_data = json.loads(chunks[0][6:])
    assert "tool_calls" in chunk_data["choices"][0]["delta"]
    assert chunk_data["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_1"


# Tests for helper functions


@pytest.mark.asyncio
async def test_accumulate_content():
    """Test accumulate_content helper."""
    content = await accumulate_content(openai_stream_generator())
    assert content == "Hello world!"


@pytest.mark.asyncio
async def test_accumulate_content_empty():
    """Test accumulate_content with empty stream."""
    content = await accumulate_content(empty_stream())
    assert content == ""


@pytest.mark.asyncio
async def test_collect_parsed_chunks():
    """Test collect_parsed_chunks helper."""
    chunks = await collect_parsed_chunks(openai_stream_generator())
    assert len(chunks) == 3  # Excludes [DONE]
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " world"
    assert chunks[2].content == "!"


@pytest.mark.asyncio
async def test_collect_parsed_chunks_preserves_metadata():
    """Test that collect_parsed_chunks preserves all metadata."""
    chunks = await collect_parsed_chunks(openai_tool_call_stream())
    assert len(chunks) == 4  # Excludes [DONE]
    assert chunks[0].tool_calls is not None
    assert chunks[3].finish_reason == "tool_calls"


# Error handling tests


@pytest.mark.asyncio
async def test_context_manager_required_for_as_str():
    """Test that using as_str without context manager raises error."""
    reader = chunk_reader(openai_stream_generator())

    with pytest.raises(RuntimeError, match="must be used as async context manager"):
        async for _ in reader.as_str():
            pass


@pytest.mark.asyncio
async def test_context_manager_required_for_as_raw():
    """Test that using as_raw without context manager raises error."""
    reader = chunk_reader(openai_stream_generator())

    with pytest.raises(RuntimeError, match="must be used as async context manager"):
        async for _ in reader.as_raw():
            pass


@pytest.mark.asyncio
async def test_handles_exception_in_source():
    """Test that exceptions from source are propagated."""

    async def failing_stream():
        yield 'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'
        raise ValueError("Stream error")

    with pytest.raises(ValueError, match="Stream error"):
        async with chunk_reader(failing_stream()) as reader:
            async for _ in reader.as_str():
                pass


# Edge cases


@pytest.mark.asyncio
async def test_single_chunk():
    """Test handling single chunk stream."""
    chunks: List[str] = []
    async with chunk_reader(single_chunk_stream()) as reader:
        async for content in reader.as_str():
            chunks.append(content)

    assert chunks == ["Single"]


@pytest.mark.asyncio
async def test_whitespace_handling():
    """Test that whitespace is properly handled."""

    async def whitespace_stream():
        yield 'data: {"choices":[{"delta":{"content":"  Hello  "},"index":0}]}\n\n'
        yield 'data: {"choices":[{"delta":{"content":"  world  "},"index":0}]}\n\n'

    content = await accumulate_content(whitespace_stream())
    assert content == "  Hello    world  "  # Preserves whitespace


@pytest.mark.asyncio
async def test_unicode_content():
    """Test handling unicode content."""

    async def unicode_stream():
        yield 'data: {"choices":[{"delta":{"content":"Hello 世界"},"index":0}]}\n\n'
        yield 'data: {"choices":[{"delta":{"content":" 🌍"},"index":0}]}\n\n'

    content = await accumulate_content(unicode_stream())
    assert content == "Hello 世界 🌍"


@pytest.mark.asyncio
async def test_large_content_chunk():
    """Test handling large content chunks."""
    large_content = "A" * 10000

    async def large_chunk_stream():
        yield f'data: {{"choices":[{{"delta":{{"content":"{large_content}"}},"index":0}}]}}\n\n'

    content = await accumulate_content(large_chunk_stream())
    assert len(content) == 10000
    assert content == large_content


@pytest.mark.asyncio
async def test_multiple_choices_uses_first():
    """Test that when multiple choices exist, first is used."""

    async def multi_choice_stream():
        yield 'data: {"choices":[{"delta":{"content":"First"},"index":0},{"delta":{"content":"Second"},"index":1}]}\n\n'

    chunks: List[ParsedChunk] = []
    async with chunk_reader(multi_choice_stream()) as reader:
        async for parsed in reader.as_parsed():
            chunks.append(parsed)

    # Should extract content from first choice
    assert chunks[0].content == "First"


@pytest.mark.asyncio
async def test_finish_reason_propagation():
    """Test that finish_reason is properly propagated."""

    async def stream_with_reasons():
        yield 'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'
        yield 'data: {"choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    chunks: List[ParsedChunk] = []
    async with chunk_reader(stream_with_reasons()) as reader:
        async for parsed in reader.as_parsed():
            chunks.append(parsed)

    assert chunks[0].finish_reason is None
    assert chunks[1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_mixed_content_and_tool_calls():
    """Test handling chunks with both content and tool calls."""

    async def mixed_stream():
        yield 'data: {"choices":[{"delta":{"content":"Calling tool...","tool_calls":[{"index":0,"id":"call_1"}]},"index":0}]}\n\n'

    chunks: List[ParsedChunk] = []
    async with chunk_reader(mixed_stream()) as reader:
        async for parsed in reader.as_parsed():
            chunks.append(parsed)

    # Should have both content and tool_calls
    assert chunks[0].content == "Calling tool..."
    assert chunks[0].tool_calls is not None
    assert chunks[0].tool_calls[0]["id"] == "call_1"


@pytest.mark.asyncio
async def test_done_variations():
    """Test different variations of DONE marker."""

    async def done_variations_stream():
        yield 'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'
        yield "data: [DONE]\n\n"

    async def done_without_data_prefix():
        yield 'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'
        yield "[DONE]\n\n"

    # Test with "data: [DONE]"
    chunks1: List[ParsedChunk] = []
    async with chunk_reader(done_variations_stream()) as reader:
        async for parsed in reader.as_parsed():
            chunks1.append(parsed)
    assert chunks1[-1].is_done is True

    # Test with just "[DONE]"
    chunks2: List[ParsedChunk] = []
    async with chunk_reader(done_without_data_prefix()) as reader:
        async for parsed in reader.as_parsed():
            chunks2.append(parsed)
    assert chunks2[-1].is_done is True


@pytest.mark.asyncio
async def test_context_manager_cleanup():
    """Test that context manager properly cleans up."""
    call_count = 0

    async def counting_stream():
        nonlocal call_count
        try:
            yield 'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'
            call_count += 1
            yield 'data: {"choices":[{"delta":{"content":"world"},"index":0}]}\n\n'
            call_count += 1
        finally:
            # Simulate cleanup
            pass

    async with chunk_reader(counting_stream()) as reader:
        async for _ in reader.as_str():
            pass

    assert call_count == 2


@pytest.mark.asyncio
async def test_reuse_reader_not_allowed():
    """Test that reader cannot be reused after context exit."""
    async with chunk_reader(openai_stream_generator()) as reader:
        async for _ in reader.as_str():
            pass

    # After exiting context, reader should not be usable
    # (This is enforced by the _entered flag)
    with pytest.raises(RuntimeError):
        async for _ in reader.as_str():
            pass


# Tests for tool call accumulation


async def tool_call_stream_with_accumulation() -> AsyncGenerator[str, None]:
    """Generate OpenAI-style chunks with tool calls spread across multiple chunks."""
    # First chunk: tool call ID and start of function name
    yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc123","function":{"name":"get"}}]},"index":0}]}\n\n'
    # Second chunk: rest of function name
    yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"_weather"}}]},"index":0}]}\n\n'
    # Third chunk: start of arguments
    yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"location\\":"}}]},"index":0}]}\n\n'
    # Fourth chunk: rest of arguments
    yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"NYC\\"}"}}]},"index":0}]}\n\n'
    # Fifth chunk: another tool call
    yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_def456","function":{"name":"get_time","arguments":"{\\"tz\\":\\"UTC\\"}"}}]},"index":0}]}\n\n'
    yield "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_tool_call_accumulation():
    """Test that tool calls are properly accumulated across chunks."""
    async with chunk_reader(tool_call_stream_with_accumulation()) as reader:
        chunks_with_tool_calls = 0
        final_accumulated = {}
        
        async for parsed in reader.as_parsed():
            if parsed.is_done:
                break
            if parsed.tool_calls:
                chunks_with_tool_calls += 1
            # Track the final accumulated state
            final_accumulated = parsed.accumulated_tool_calls

    # Should have seen 5 chunks with tool calls
    assert chunks_with_tool_calls == 5
    
    # Check accumulated tool calls
    assert len(final_accumulated) == 2
    
    # First tool call should be complete
    assert final_accumulated[0]["id"] == "call_abc123"
    assert final_accumulated[0]["name"] == "get_weather"
    assert final_accumulated[0]["arguments"] == '{"location":"NYC"}'
    
    # Second tool call should be complete
    assert final_accumulated[1]["id"] == "call_def456"
    assert final_accumulated[1]["name"] == "get_time"
    assert final_accumulated[1]["arguments"] == '{"tz":"UTC"}'


@pytest.mark.asyncio
async def test_get_accumulated_tool_calls():
    """Test the get_accumulated_tool_calls method."""
    async with chunk_reader(tool_call_stream_with_accumulation()) as reader:
        # Process some chunks
        chunk_count = 0
        async for parsed in reader.as_parsed():
            if parsed.is_done:
                break
            chunk_count += 1
            if chunk_count == 3:
                # After 3 chunks, we should have partial accumulation
                accumulated = reader.get_accumulated_tool_calls()
                assert 0 in accumulated
                assert accumulated[0]["id"] == "call_abc123"
                # Name might be partial at this point
                assert "get" in accumulated[0]["name"]


@pytest.mark.asyncio
async def test_clear_tool_calls():
    """Test that clear_tool_calls resets the accumulator."""
    async with chunk_reader(tool_call_stream_with_accumulation()) as reader:
        # Process a few chunks
        chunk_count = 0
        async for parsed in reader.as_parsed():
            if parsed.is_done:
                break
            chunk_count += 1
            if chunk_count == 3:
                # Clear accumulated tool calls
                reader.clear_tool_calls()
                accumulated = reader.get_accumulated_tool_calls()
                assert len(accumulated) == 0
                ready = reader.get_ready_tool_calls()
                assert len(ready) == 0
                break


@pytest.mark.asyncio
async def test_tool_call_ready_detection():
    """Test that get_ready_tool_calls identifies complete tool calls."""
    async with chunk_reader(tool_call_stream_with_accumulation()) as reader:
        ready_at_each_step = []
        
        async for parsed in reader.as_parsed():
            if parsed.is_done:
                break
            ready = reader.get_ready_tool_calls()
            ready_at_each_step.append(len(ready))
        
        # Should have 0, 0, 0, 1 (first complete), 2 (both complete)
        assert ready_at_each_step[-1] == 2  # Both tool calls ready at the end
        assert 1 in ready_at_each_step  # First tool call became ready at some point


@pytest.mark.asyncio
async def test_tracing_disabled():
    """Test that tracing can be disabled."""
    chunks: List[str] = []
    
    # Create reader with tracing disabled
    async with chunk_reader(openai_stream_generator(), enable_tracing=False) as reader:
        async for content in reader.as_str():
            chunks.append(content)
    
    # Should still work the same way
    assert chunks == ["Hello", " world", "!"]


@pytest.mark.asyncio
async def test_mixed_content_and_tool_calls_accumulation():
    """Test handling chunks with both content and tool calls."""
    
    async def mixed_stream():
        yield 'data: {"choices":[{"delta":{"content":"Calling tool...","tool_calls":[{"index":0,"id":"call_1","function":{"name":"test"}}]},"index":0}]}\n\n'
        yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"a\\":1}"}}]},"index":0}]}\n\n'
        yield 'data: {"choices":[{"delta":{"content":" Done!"},"index":0}]}\n\n'

    content_parts = []
    final_accumulated = {}
    
    async with chunk_reader(mixed_stream()) as reader:
        async for parsed in reader.as_parsed():
            if parsed.content:
                content_parts.append(parsed.content)
            final_accumulated = parsed.accumulated_tool_calls
    
    # Should have both content and accumulated tool calls
    assert "".join(content_parts) == "Calling tool... Done!"
    assert len(final_accumulated) == 1
    assert final_accumulated[0]["name"] == "test"


@pytest.mark.asyncio
async def test_early_generator_exit_with_tracing():
    """Test that early generator exit doesn't cause OpenTelemetry errors."""
    
    async def long_stream():
        """Generate many chunks to simulate a long stream."""
        for i in range(100):
            yield f'data: {{"choices":[{{"delta":{{"content":"chunk{i}"}},"index":0}}]}}\n\n'
        yield "data: [DONE]\n\n"
    
    # Exit after reading only 3 chunks (simulates client disconnect or break)
    chunks_read = 0
    async with chunk_reader(long_stream(), enable_tracing=True) as reader:
        async for parsed in reader.as_parsed():
            if parsed.content:
                chunks_read += 1
                if chunks_read == 3:
                    break  # Early exit - this triggers GeneratorExit
    
    # Should have read exactly 3 chunks without errors
    assert chunks_read == 3
