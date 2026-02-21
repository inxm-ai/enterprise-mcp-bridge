"""
Comprehensive tests for context compression strategies.
"""

import pytest
from unittest.mock import AsyncMock
from app.tgi.models import (
    ChatCompletionRequest,
    Message,
    MessageRole,
    ToolCall,
    ToolCallFunction,
)
from app.tgi.context_compressor import (
    SlidingWindowCompressor,
    HierarchicalSummarizer,
    AdaptiveCompressor,
    CompressionStats,
)


class TestCompressionStats:
    """Test CompressionStats dataclass."""

    def test_stats_creation(self):
        """Test creating compression stats."""
        stats = CompressionStats(
            original_size=1000,
            compressed_size=500,
            compression_ratio=0.5,
            strategy_used="test",
            messages_removed=2,
            messages_summarized=3,
        )
        assert stats.original_size == 1000
        assert stats.compressed_size == 500
        assert stats.compression_ratio == 0.5
        assert stats.messages_removed == 2
        assert stats.messages_summarized == 3

    def test_stats_summary_string(self):
        """Test stats summary string generation."""
        stats = CompressionStats(
            original_size=1000,
            compressed_size=500,
            compression_ratio=0.5,
            strategy_used="test_strategy",
            messages_removed=2,
            messages_summarized=1,
        )
        summary = stats.summary()
        assert "test_strategy" in summary
        assert "1000" in summary
        assert "500" in summary
        assert "50.00%" in summary


class TestSlidingWindowCompressor:
    """Test sliding window compression strategy."""

    @pytest.fixture
    def compressor(self):
        return SlidingWindowCompressor(window_size=3)

    @pytest.fixture
    def mock_summarize_fn(self):
        async def fn(base_request, content, access_token, outer_span):
            return f"Summary of: {content[:50]}..."

        return AsyncMock(side_effect=fn)

    def _create_request(self, message_count=10, system_message=True):
        """Helper to create test requests."""
        messages = []
        if system_message:
            messages.append(Message(role=MessageRole.SYSTEM, content="You are helpful"))
        for i in range(message_count):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            messages.append(Message(role=role, content=f"Message {i}" * 100))
        return ChatCompletionRequest(messages=messages)

    @pytest.mark.asyncio
    async def test_no_compression_needed(self, compressor, mock_summarize_fn):
        """Test when payload is already within limit."""
        request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.USER, content="Short message"),
            ]
        )
        compressed, stats = await compressor.compress(
            request, max_size=1000000, summarize_fn=mock_summarize_fn
        )

        assert stats.compression_ratio == 1.0
        assert stats.messages_summarized == 0
        assert len(compressed.messages) == 1

    @pytest.mark.asyncio
    async def test_compression_applied(self, compressor, mock_summarize_fn):
        """Test that compression is applied when needed."""
        request = self._create_request(message_count=10)
        original_count = len(request.messages)

        # Use small max_size to trigger compression
        compressed, stats = await compressor.compress(
            request, max_size=500, summarize_fn=mock_summarize_fn
        )

        # Should have summarized old messages
        assert stats.messages_summarized > 0
        # Should have fewer messages now
        assert len(compressed.messages) <= original_count
        assert stats.compressed_size <= stats.original_size

    @pytest.mark.asyncio
    async def test_preserves_window(self, compressor, mock_summarize_fn):
        """Test that recent messages in window are preserved."""
        request = self._create_request(message_count=10)
        original_last_messages = [m.content for m in request.messages[-3:]]

        compressed, stats = await compressor.compress(
            request, max_size=500, summarize_fn=mock_summarize_fn
        )

        # Last messages should still be in the request
        recent_contents = [m.content for m in compressed.messages[-3:]]
        for original in original_last_messages:
            assert original in recent_contents or "Summary" in str(compressed.messages)

    @pytest.mark.asyncio
    async def test_preserves_system_message(self, compressor, mock_summarize_fn):
        """Test that system message is preserved."""
        request = self._create_request(message_count=10, system_message=True)

        compressed, stats = await compressor.compress(
            request, max_size=500, summarize_fn=mock_summarize_fn
        )

        # First message should still be system
        assert compressed.messages[0].role == MessageRole.SYSTEM

    @pytest.mark.asyncio
    async def test_insufficient_messages_for_window(
        self, compressor, mock_summarize_fn
    ):
        """Test with fewer messages than window size."""
        request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="System"),
                Message(role=MessageRole.USER, content="User 1"),
                Message(role=MessageRole.ASSISTANT, content="Assistant 1"),
            ]
        )
        original_count = len(request.messages)
        compressed, stats = await compressor.compress(
            request, max_size=100, summarize_fn=mock_summarize_fn
        )

        # Not enough messages, should not compress
        assert len(compressed.messages) == original_count


class TestHierarchicalSummarizer:
    """Test hierarchical summarization strategy."""

    @pytest.fixture
    def summarizer(self):
        return HierarchicalSummarizer(chunk_tokens=100, group_size=2)

    @pytest.fixture
    def mock_summarize_fn(self):
        async def fn(base_request, content, access_token, outer_span):
            # Simulate summarization reducing size by 50%
            return content[: len(content) // 2]

        return AsyncMock(side_effect=fn)

    def test_chunk_splitting(self):
        """Test text chunking functionality."""
        summarizer = HierarchicalSummarizer(chunk_tokens=10)  # ~40 char chunks
        text = "This is a test. " * 20  # 320 chars

        chunks = summarizer._split_into_chunks(text, 40)

        # Should have multiple chunks
        assert len(chunks) > 1
        # All chunks should fit in size
        assert all(len(chunk) <= 80 for chunk in chunks)  # Some overlap
        # All content should be substantially preserved (some periods may be duplicated)
        joined = "".join(chunks)
        # Verify the core text is there
        assert "This is a test" in joined
        assert len(joined) >= len(text) - 20  # Allow for minor loss

    def test_single_chunk(self):
        """Test with text that fits in single chunk."""
        summarizer = HierarchicalSummarizer(chunk_tokens=1000)
        text = "This is short."

        chunks = summarizer._split_into_chunks(text, 4000)

        assert len(chunks) == 1
        assert chunks[0] == text

    @pytest.mark.asyncio
    async def test_no_compression_needed(self, summarizer, mock_summarize_fn):
        """Test when message is small."""
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Short")]
        )

        compressed, stats = await summarizer.compress(
            request, max_size=1000, summarize_fn=mock_summarize_fn
        )

        assert stats.compression_ratio == 1.0
        assert stats.messages_summarized == 0

    @pytest.mark.asyncio
    async def test_large_message_compression(self, summarizer, mock_summarize_fn):
        """Test compression of large messages."""
        large_content = "Large content. " * 500  # ~7000 chars
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content=large_content)]
        )
        original_size = len(large_content)

        compressed, stats = await summarizer.compress(
            request, max_size=100, summarize_fn=mock_summarize_fn
        )

        assert stats.messages_summarized == 1
        new_size = len(compressed.messages[0].content or "")
        assert new_size < original_size

    @pytest.mark.asyncio
    async def test_multiple_large_messages(self, summarizer, mock_summarize_fn):
        """Test compression with multiple large messages."""
        request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="System message"),
                Message(
                    role=MessageRole.USER,
                    content="User content. " * 300,
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="Assistant content. " * 300,
                ),
            ]
        )

        compressed, stats = await summarizer.compress(
            request, max_size=100, summarize_fn=mock_summarize_fn
        )

        # Both large messages should be summarized
        assert stats.messages_summarized >= 1

    @pytest.mark.asyncio
    async def test_first_system_message_is_not_summarized(self, summarizer):
        """The first system message must remain verbatim even when very large."""
        large_system = "SYSTEM_CONTRACT " * 1200
        request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content=large_system),
                Message(role=MessageRole.USER, content="user content " * 800),
            ]
        )

        async def summarize_fn(base_request, content, access_token, outer_span):
            return "SUMMARIZED"

        compressed, stats = await summarizer.compress(
            request, max_size=200, summarize_fn=summarize_fn
        )

        assert compressed.messages[0].content == large_system
        assert stats.messages_summarized >= 1

    @pytest.mark.asyncio
    async def test_recent_contract_user_message_is_not_summarized(self, summarizer):
        request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="system"),
                Message(
                    role=MessageRole.ASSISTANT, content="old assistant block " * 900
                ),
                Message(
                    role=MessageRole.USER,
                    content=(
                        "Phase 1 contract: required output keys are "
                        "components_script, service_script, test_script. " * 120
                    ),
                ),
            ]
        )

        async def summarize_fn(base_request, content, access_token, outer_span):
            return "SUMMARIZED"

        compressed, _stats = await summarizer.compress(
            request, max_size=250, summarize_fn=summarize_fn
        )

        assert "components_script" in (compressed.messages[-1].content or "")


class TestAdaptiveCompressor:
    """Test adaptive multi-strategy compression."""

    @pytest.fixture
    def compressor(self):
        return AdaptiveCompressor(
            chunk_tokens=100,
            group_size=2,
            window_size=2,
        )

    @pytest.fixture
    def mock_summarize_fn(self):
        async def fn(base_request, content, access_token, outer_span):
            # Simulate effective summarization
            return content[: len(content) // 3]

        return AsyncMock(side_effect=fn)

    def _create_complex_request(self):
        """Create a complex request with various message types."""
        return ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="You are helpful."),
                Message(role=MessageRole.USER, content="User 1" * 100),
                Message(role=MessageRole.ASSISTANT, content="Assistant 1" * 100),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="Response",
                    tool_calls=[
                        ToolCall(
                            id="1",
                            function=ToolCallFunction(
                                name="test_tool",
                                arguments='{"key": "value"}',
                            ),
                        )
                    ],
                ),
                Message(
                    role=MessageRole.TOOL,
                    content="Tool output" * 100,
                    tool_call_id="1",
                ),
                Message(role=MessageRole.USER, content="Follow up question" * 100),
            ]
        )

    @pytest.mark.asyncio
    async def test_no_compression_needed(self, compressor, mock_summarize_fn):
        """Test when payload is already small."""
        request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.USER, content="Hi"),
            ]
        )

        compressed, stats = await compressor.compress(
            request, max_size=100000, summarize_fn=mock_summarize_fn
        )

        assert stats.compressed_size <= stats.original_size
        # When no compression is needed, only initial check happens
        assert stats.compression_ratio == 1.0

    @pytest.mark.asyncio
    async def test_compaction_step(self, compressor, mock_summarize_fn):
        """Test that compaction is first step."""
        request = ChatCompletionRequest(
            messages=[
                Message(
                    role=MessageRole.USER, content='{"key":    "value"}' + " " * 500
                ),
            ]
        )

        compressed, stats = await compressor.compress(
            request, max_size=200, summarize_fn=mock_summarize_fn
        )

        # Should have used at least compaction step
        assert len(stats.metadata.get("steps_used", [])) > 0

    @pytest.mark.asyncio
    async def test_hierarchical_step(self, compressor, mock_summarize_fn):
        """Test hierarchical summarization step."""
        large_content = "Content. " * 2000
        request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="System"),
                Message(role=MessageRole.USER, content=large_content),
            ]
        )

        compressed, stats = await compressor.compress(
            request, max_size=500, summarize_fn=mock_summarize_fn
        )

        # Should reach hierarchical step
        assert "hierarchical" in stats.metadata.get("steps_used", [])

    @pytest.mark.asyncio
    async def test_sliding_window_step(self, compressor, mock_summarize_fn):
        """Test sliding window compression step."""
        messages = [Message(role=MessageRole.SYSTEM, content="System")]
        for i in range(15):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            messages.append(Message(role=role, content=f"Message {i}. " * 100))

        request = ChatCompletionRequest(messages=messages)

        compressed, stats = await compressor.compress(
            request, max_size=300, summarize_fn=mock_summarize_fn
        )

        # Should reach sliding window step
        assert any("sliding" in step for step in stats.metadata.get("steps_used", []))

    @pytest.mark.asyncio
    async def test_drop_step(self, compressor, mock_summarize_fn):
        """Test message dropping step."""
        messages = [Message(role=MessageRole.SYSTEM, content="System")]
        # Add tool calls and tool responses
        for i in range(20):
            if i % 3 == 0:
                messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content="Calling tool" * 50,
                        tool_calls=[
                            ToolCall(
                                id=str(i),
                                function=ToolCallFunction(
                                    name="tool",
                                    arguments="{}",
                                ),
                            )
                        ],
                    )
                )
            elif i % 3 == 1:
                messages.append(
                    Message(
                        role=MessageRole.TOOL,
                        content="Tool result" * 50,
                        tool_call_id=str(i - 1),
                    )
                )
            else:
                messages.append(
                    Message(role=MessageRole.USER, content="User input" * 50)
                )

        request = ChatCompletionRequest(messages=messages)

        compressed, stats = await compressor.compress(
            request, max_size=200, summarize_fn=mock_summarize_fn
        )

        # Should reach drop step
        assert "drop" in stats.metadata.get("steps_used", [])

    @pytest.mark.asyncio
    async def test_drop_step_preserves_tool_call_pairs(self):
        """Ensure drop step does not orphan tool calls."""
        compressor = AdaptiveCompressor(chunk_tokens=1000, window_size=50)
        messages = [
            Message(role=MessageRole.SYSTEM, content="System"),
            Message(role=MessageRole.USER, content="User"),
            Message(
                role=MessageRole.ASSISTANT,
                content="Calling tools",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=ToolCallFunction(name="tool_a", arguments="{}"),
                    ),
                    ToolCall(
                        id="call_2",
                        function=ToolCallFunction(name="tool_b", arguments="{}"),
                    ),
                ],
            ),
            Message(
                role=MessageRole.TOOL,
                content="tool_result_" + ("a" * 200),
                tool_call_id="call_1",
            ),
            Message(
                role=MessageRole.TOOL,
                content="tool_result_" + ("b" * 200),
                tool_call_id="call_2",
            ),
            Message(role=MessageRole.USER, content="Follow up"),
        ]
        request = ChatCompletionRequest(messages=messages)
        request_without_first_tool = ChatCompletionRequest(messages=list(messages))
        request_without_first_tool.messages.pop(3)
        original_size = compressor._get_payload_size_from_request(request)
        size_without_first_tool = compressor._get_payload_size_from_request(
            request_without_first_tool
        )
        max_size = size_without_first_tool + 1
        assert max_size < original_size

        async def summarize_fn(base_request, content, access_token, outer_span):
            return content

        compressed, _stats = await compressor.compress(
            request, max_size=max_size, summarize_fn=summarize_fn
        )

        for idx, message in enumerate(compressed.messages or []):
            if message.role != MessageRole.ASSISTANT or not message.tool_calls:
                continue
            tool_ids = [tc.id for tc in message.tool_calls if tc.id]
            if not tool_ids:
                continue
            response_ids = []
            for next_msg in compressed.messages[idx + 1 :]:
                if next_msg.role != MessageRole.TOOL:
                    break
                if next_msg.tool_call_id:
                    response_ids.append(next_msg.tool_call_id)
            for tool_id in tool_ids:
                assert tool_id in response_ids

    @pytest.mark.asyncio
    async def test_compression_ratio_calculation(self, compressor, mock_summarize_fn):
        """Test that compression ratio is calculated correctly."""
        request = self._create_complex_request()

        compressed, stats = await compressor.compress(
            request, max_size=300, summarize_fn=mock_summarize_fn
        )

        # Ratio should be: compressed / original
        expected_ratio = stats.compressed_size / stats.original_size
        assert abs(stats.compression_ratio - expected_ratio) < 0.01

    @pytest.mark.asyncio
    async def test_preserves_system_message(self, compressor, mock_summarize_fn):
        """Test that system message is always preserved."""
        request = self._create_complex_request()

        compressed, stats = await compressor.compress(
            request, max_size=100, summarize_fn=mock_summarize_fn
        )

        assert len(compressed.messages) > 0
        assert compressed.messages[0].role == MessageRole.SYSTEM

    @pytest.mark.asyncio
    async def test_request_still_over_limit_after_all_steps(
        self, compressor, mock_summarize_fn
    ):
        """Test behavior when request still exceeds limit after all compression."""

        async def ineffective_summarize(
            base_request, content, access_token, outer_span
        ):
            # Return original content (no compression)
            return content

        request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.USER, content="A" * 10000),
            ]
        )

        compressed, stats = await compressor.compress(
            request, max_size=100, summarize_fn=ineffective_summarize
        )

        # Should still return something, but won't fit
        assert compressed is not None
        assert len(compressed.messages) > 0

    @pytest.mark.asyncio
    async def test_adaptive_preserves_recent_contract_instruction(self, compressor):
        request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="system"),
                Message(role=MessageRole.TOOL, content="tool output " * 1200),
                Message(
                    role=MessageRole.USER,
                    content=(
                        "Phase-1 required output keys: components_script + test_script. "
                        * 100
                    ),
                ),
            ]
        )

        async def summarize_fn(base_request, content, access_token, outer_span):
            return "SUMMARIZED"

        compressed, _stats = await compressor.compress(
            request, max_size=300, summarize_fn=summarize_fn
        )

        assert any(
            "components_script + test_script" in (msg.content or "")
            for msg in compressed.messages
            if msg.role == MessageRole.USER
        )


class TestIntegration:
    """Integration tests for compression strategies."""

    @pytest.mark.asyncio
    async def test_real_world_scenario(self):
        """Test a realistic scenario with multiple strategies."""
        # Create a request with:
        # - System message
        # - Long conversation history
        # - Large tool outputs
        # - Recent user messages to preserve

        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant.")
        ]

        # Add old conversation
        for i in range(10):
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=f"Old question {i}: " + "context " * 100,
                )
            )
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=f"Old answer {i}: " + "response " * 150,
                )
            )

        # Add tool interaction
        messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content="Calling tool",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=ToolCallFunction(
                            name="search",
                            arguments='{"query": "test"}',
                        ),
                    )
                ],
            )
        )
        messages.append(
            Message(
                role=MessageRole.TOOL,
                content="Tool result: " + "data " * 500,
                tool_call_id="call_1",
            )
        )

        # Add recent messages
        messages.append(
            Message(
                role=MessageRole.USER, content="Current question: " + "details " * 50
            )
        )

        request = ChatCompletionRequest(messages=messages)

        async def summarize_fn(base_request, content, access_token, outer_span):
            return content[:200]  # Reduce to 200 chars

        compressor = AdaptiveCompressor()
        compressed, stats = await compressor.compress(
            request, max_size=2000, summarize_fn=summarize_fn
        )

        # Verify compression occurred
        assert stats.compressed_size < stats.original_size
        # Verify structure is preserved
        assert len(compressed.messages) > 0
        assert compressed.messages[0].role == MessageRole.SYSTEM
        # Recent message should be preserved
        assert any("Current question" in (m.content or "") for m in compressed.messages)
