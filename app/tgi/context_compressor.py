"""
Context compression module for managing large conversation histories.

Implements multiple strategies for handling payload size limits:
- Sliding window: Keep recent messages, compress older ones
- Hierarchical summarization: Build a summary tree for massive messages
- Adaptive compression: Balance between quality and size constraints
"""

import json
import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from app.tgi.models import (
    ChatCompletionRequest,
    Message,
    MessageRole,
)

logger = logging.getLogger("uvicorn.error")


@dataclass
class CompressionStats:
    """Statistics about compression operations."""

    original_size: int
    compressed_size: int
    compression_ratio: float
    strategy_used: str
    messages_removed: int = 0
    messages_summarized: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Get a human-readable summary of compression stats."""
        return (
            f"[{self.strategy_used}] "
            f"Size: {self.original_size} → {self.compressed_size} bytes "
            f"(ratio: {self.compression_ratio:.2%}), "
            f"Removed: {self.messages_removed}, "
            f"Summarized: {self.messages_summarized}"
        )


class CompressionStrategy(ABC):
    """Base class for compression strategies."""

    def __init__(self, logger_instance=None):
        self.logger = logger_instance or logger

    @abstractmethod
    async def compress(
        self,
        request: ChatCompletionRequest,
        max_size: int,
        summarize_fn,
    ) -> Tuple[ChatCompletionRequest, CompressionStats]:
        """
        Compress the request to fit within max_size.

        Args:
            request: ChatCompletionRequest to compress
            max_size: Maximum allowed size in bytes
            summarize_fn: Async function(base_request, content, access_token, span) -> str

        Returns:
            Tuple of (compressed_request, stats)
        """
        pass

    def _serialize_payload(self, payload: dict) -> str:
        """Serialize payload to JSON."""
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _payload_size(self, serialized_payload: str) -> int:
        """Calculate payload size in bytes."""
        return len(serialized_payload.encode("utf-8"))

    def _get_payload_size_from_request(self, request: ChatCompletionRequest) -> int:
        """Get current payload size from request."""
        payload = request.model_dump(exclude_none=True)
        payload.pop("persist_inner_thinking", None)
        serialized = self._serialize_payload(payload)
        return self._payload_size(serialized)


class SlidingWindowCompressor(CompressionStrategy):
    """
    Sliding window strategy: Keep recent messages, compress older ones into summaries.

    Preserves the most recent messages (typically user/assistant turns) while
    summarizing older messages to maintain context awareness.
    """

    def __init__(
        self,
        window_size: int = 5,
        logger_instance=None,
    ):
        """
        Initialize sliding window compressor.

        Args:
            window_size: Number of recent messages to keep uncompressed
            logger_instance: Optional logger instance
        """
        super().__init__(logger_instance)
        self.window_size = window_size

    async def compress(
        self,
        request: ChatCompletionRequest,
        max_size: int,
        summarize_fn,
    ) -> Tuple[ChatCompletionRequest, CompressionStats]:
        """Apply sliding window compression."""
        original_size = self._get_payload_size_from_request(request)
        stats = CompressionStats(
            original_size=original_size,
            compressed_size=original_size,
            compression_ratio=1.0,
            strategy_used="sliding_window",
        )

        if original_size <= max_size:
            return request, stats

        # Identify messages to compress: all except last window_size messages
        messages = request.messages or []
        if len(messages) <= self.window_size + 1:
            # Not enough messages to apply sliding window, skip
            self.logger.debug(
                "[SlidingWindowCompressor] Not enough messages for sliding window "
                f"(have {len(messages)}, window={self.window_size})"
            )
            return request, stats

        # Keep system message and last window_size messages
        system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]
        other_messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        to_summarize = other_messages[: -self.window_size]
        to_keep = other_messages[-self.window_size :]

        # Summarize the old messages together
        if to_summarize:
            summary = await self._summarize_conversation(
                to_summarize, request, summarize_fn
            )
            # Create a summary message
            summary_msg = Message(
                role=MessageRole.ASSISTANT,
                content=f"[CONTEXT SUMMARY]\n{summary}",
            )
            request.messages = system_messages + [summary_msg] + to_keep
            stats.messages_summarized = len(to_summarize)

        # Check new size
        new_size = self._get_payload_size_from_request(request)
        stats.compressed_size = new_size
        stats.compression_ratio = new_size / original_size if original_size > 0 else 1.0

        self.logger.info(f"[SlidingWindowCompressor] {stats.summary()}")

        return request, stats

    async def _summarize_conversation(
        self, messages: List[Message], base_request: ChatCompletionRequest, summarize_fn
    ) -> str:
        """Summarize a list of messages into a single summary."""
        conversation_text = "\n".join(
            [
                f"## {message.role.value.capitalize()}\n{message.content}"
                for message in messages
                if message.content and message.role != MessageRole.SYSTEM
            ]
        )

        summary_prompt = (
            "Provide a concise summary of this conversation excerpt, "
            "preserving key information, decisions, and context:\n\n"
            f"{conversation_text}"
        )

        return await summarize_fn(
            base_request=base_request,
            content=summary_prompt,
            access_token="",
            outer_span=None,
        )


class HierarchicalSummarizer(CompressionStrategy):
    """
    Hierarchical summarization for single massive messages.

    Breaks a message into chunks, summarizes each chunk, then recursively
    summarizes the summaries, building a tree structure. Handles extremely
    large documents (logs, code dumps, etc.) with minimal information loss.
    """

    def __init__(
        self,
        chunk_tokens: int = 2000,
        group_size: int = 5,
        logger_instance=None,
    ):
        """
        Initialize hierarchical summarizer.

        Args:
            chunk_tokens: Approximate tokens per chunk (rough estimate: 4 chars ≈ 1 token)
            group_size: Number of summaries to group before summarizing again
            logger_instance: Optional logger instance
        """
        super().__init__(logger_instance)
        self.chunk_tokens = chunk_tokens
        self.chunk_size_chars = chunk_tokens * 4  # Rough estimate: 4 chars per token
        self.group_size = group_size

    async def compress(
        self,
        request: ChatCompletionRequest,
        max_size: int,
        summarize_fn,
    ) -> Tuple[ChatCompletionRequest, CompressionStats]:
        """Apply hierarchical summarization to large messages."""
        original_size = self._get_payload_size_from_request(request)
        stats = CompressionStats(
            original_size=original_size,
            compressed_size=original_size,
            compression_ratio=1.0,
            strategy_used="hierarchical",
        )

        if original_size <= max_size:
            return request, stats

        # Find and compress large messages
        for message in request.messages or []:
            if (
                message.content
                and isinstance(message.content, str)
                and len(message.content) > self.chunk_size_chars
            ):
                self.logger.info(
                    f"[HierarchicalSummarizer] Found large message "
                    f"(role={message.role}, size={len(message.content)}), "
                    f"applying hierarchical summarization"
                )

                message.content = await self._hierarchical_summarize(
                    message.content, request, summarize_fn
                )
                stats.messages_summarized += 1

                # Check if we're under the limit
                new_size = self._get_payload_size_from_request(request)
                if new_size <= max_size:
                    stats.compressed_size = new_size
                    stats.compression_ratio = (
                        new_size / original_size if original_size > 0 else 1.0
                    )
                    self.logger.info(f"[HierarchicalSummarizer] {stats.summary()}")
                    return request, stats

        # Update stats
        new_size = self._get_payload_size_from_request(request)
        stats.compressed_size = new_size
        stats.compression_ratio = new_size / original_size if original_size > 0 else 1.0
        self.logger.info(f"[HierarchicalSummarizer] {stats.summary()}")

        return request, stats

    async def _hierarchical_summarize(
        self, content: str, base_request: ChatCompletionRequest, summarize_fn
    ) -> str:
        """
        Recursively summarize content in a hierarchical manner.

        Process:
        1. Split content into chunks
        2. Summarize each chunk
        3. Group summaries and summarize groups (map-reduce pattern)
        4. Return final summary
        """
        chunks = self._split_into_chunks(content, self.chunk_size_chars)

        if len(chunks) == 1:
            # Single chunk, no hierarchical processing needed
            self.logger.debug(
                f"[HierarchicalSummarizer] Single chunk ({len(chunks[0])} chars), "
                f"applying simple summarization"
            )
            return await summarize_fn(
                base_request=base_request,
                content=content,
                access_token="",
                outer_span=None,
            )

        # Step 1: Summarize each chunk
        self.logger.debug(
            f"[HierarchicalSummarizer] Summarizing {len(chunks)} chunks at level 0"
        )
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            summary = await summarize_fn(
                base_request=base_request,
                content=f"Summarize this section:\n\n{chunk}",
                access_token="",
                outer_span=None,
            )
            chunk_summaries.append(summary)
            if (i + 1) % 5 == 0:
                self.logger.debug(
                    f"[HierarchicalSummarizer] Summarized {i + 1}/{len(chunks)} chunks"
                )

        # Step 2-N: Recursively summarize summaries (map-reduce)
        level = 1
        while len(chunk_summaries) > 1:
            if len(chunk_summaries) <= self.group_size:
                # Final level: combine all summaries
                self.logger.debug(
                    f"[HierarchicalSummarizer] Final level: combining "
                    f"{len(chunk_summaries)} summaries"
                )
                combined = "\n\n".join(
                    [f"[Section {i + 1}]\n{s}" for i, s in enumerate(chunk_summaries)]
                )
                final_summary = await summarize_fn(
                    base_request=base_request,
                    content=(
                        "Create a comprehensive summary by combining these sections:\n\n"
                        f"{combined}"
                    ),
                    access_token="",
                    outer_span=None,
                )
                return final_summary

            # Group summaries and summarize each group
            self.logger.debug(
                f"[HierarchicalSummarizer] Level {level}: "
                f"grouping {len(chunk_summaries)} summaries "
                f"(group_size={self.group_size})"
            )
            grouped_summaries = []
            for i in range(0, len(chunk_summaries), self.group_size):
                group = chunk_summaries[i : i + self.group_size]
                group_text = "\n\n".join(
                    [f"[Part {j + 1}]\n{s}" for j, s in enumerate(group)]
                )
                group_summary = await summarize_fn(
                    base_request=base_request,
                    content=(
                        "Summarize these related sections into a coherent overview:\n\n"
                        f"{group_text}"
                    ),
                    access_token="",
                    outer_span=None,
                )
                grouped_summaries.append(group_summary)

            chunk_summaries = grouped_summaries
            level += 1

        return chunk_summaries[0] if chunk_summaries else content

    @staticmethod
    def _split_into_chunks(content: str, chunk_size: int) -> List[str]:
        """
        Split content into roughly equal-sized chunks.

        Tries to split at sentence boundaries to maintain readability.
        """
        if len(content) <= chunk_size:
            return [content]

        chunks = []
        current_chunk = ""

        sentences = content.split(". ")
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 2 <= chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.rstrip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.rstrip())

        # Handle case where sentences are very long
        if not chunks:
            chunks = [
                content[i : i + chunk_size] for i in range(0, len(content), chunk_size)
            ]

        return chunks


class AdaptiveCompressor(CompressionStrategy):
    """
    Adaptive compression that combines multiple strategies intelligently.

    Strategy selection:
    1. First pass: Compact JSON and whitespace
    2. Second pass: Apply hierarchical summarization to large messages
    3. Third pass: Apply sliding window compression to conversation history
    4. Final pass: Drop oldest tool-related messages if still oversized
    """

    def __init__(
        self,
        chunk_tokens: int = 2000,
        group_size: int = 5,
        window_size: int = 5,
        logger_instance=None,
    ):
        """
        Initialize adaptive compressor.

        Args:
            chunk_tokens: Token size for hierarchical chunks
            group_size: Group size for hierarchical summarization
            window_size: Window size for sliding window compression
            logger_instance: Optional logger instance
        """
        super().__init__(logger_instance)
        self.hierarchical = HierarchicalSummarizer(
            chunk_tokens=chunk_tokens,
            group_size=group_size,
            logger_instance=self.logger,
        )
        self.sliding_window = SlidingWindowCompressor(
            window_size=window_size,
            logger_instance=self.logger,
        )

    async def compress(
        self,
        request: ChatCompletionRequest,
        max_size: int,
        summarize_fn,
    ) -> Tuple[ChatCompletionRequest, CompressionStats]:
        """Apply adaptive multi-strategy compression."""
        original_size = self._get_payload_size_from_request(request)
        stats = CompressionStats(
            original_size=original_size,
            compressed_size=original_size,
            compression_ratio=1.0,
            strategy_used="adaptive",
        )

        if original_size <= max_size:
            return request, stats

        # Step 1: Compact JSON and whitespace
        self.logger.info("[AdaptiveCompressor] Step 1: Compacting JSON and whitespace")
        request = self._compact_request(request)
        current_size = self._get_payload_size_from_request(request)

        if current_size <= max_size:
            stats.compressed_size = current_size
            stats.compression_ratio = current_size / original_size
            stats.metadata["steps_used"] = ["compact"]
            self.logger.info(f"[AdaptiveCompressor] {stats.summary()}")
            return request, stats

        # Step 2: Apply hierarchical summarization to large messages
        self.logger.info("[AdaptiveCompressor] Step 2: Hierarchical summarization")
        request, hier_stats = await self.hierarchical.compress(
            request, max_size, summarize_fn
        )
        current_size = self._get_payload_size_from_request(request)

        if current_size <= max_size:
            stats.compressed_size = current_size
            stats.compression_ratio = current_size / original_size
            stats.messages_summarized = hier_stats.messages_summarized
            stats.metadata["steps_used"] = ["compact", "hierarchical"]
            self.logger.info(f"[AdaptiveCompressor] {stats.summary()}")
            return request, stats

        # Step 3: Apply sliding window compression
        self.logger.info("[AdaptiveCompressor] Step 3: Sliding window compression")
        request, window_stats = await self.sliding_window.compress(
            request, max_size, summarize_fn
        )
        current_size = self._get_payload_size_from_request(request)

        if current_size <= max_size:
            stats.compressed_size = current_size
            stats.compression_ratio = current_size / original_size
            stats.messages_summarized = (
                hier_stats.messages_summarized + window_stats.messages_summarized
            )
            stats.metadata["steps_used"] = ["compact", "hierarchical", "sliding_window"]
            self.logger.info(f"[AdaptiveCompressor] {stats.summary()}")
            return request, stats

        # Step 4: Drop oldest tool-related messages
        self.logger.info("[AdaptiveCompressor] Step 4: Dropping old messages")
        request = self._drop_oldest_messages(request, max_size)
        current_size = self._get_payload_size_from_request(request)

        stats.compressed_size = current_size
        stats.compression_ratio = current_size / original_size
        stats.metadata["steps_used"] = [
            "compact",
            "hierarchical",
            "sliding_window",
            "drop",
        ]
        self.logger.info(f"[AdaptiveCompressor] {stats.summary()}")

        return request, stats

    @staticmethod
    def _compact_text(text: str) -> str:
        """Trim whitespace and minify JSON-like payloads."""
        if not isinstance(text, str):
            return text
        compact = text.strip()
        if not compact:
            return compact
        if compact[0] in ("{", "["):
            try:
                compact = json.dumps(
                    json.loads(compact),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return compact

    def _compact_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Compact all text in request."""
        for message in request.messages or []:
            if message.content:
                message.content = self._compact_text(message.content)
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if getattr(tool_call, "function", None) and getattr(
                        tool_call.function, "arguments", None
                    ):
                        tool_call.function.arguments = self._compact_text(
                            tool_call.function.arguments
                        )
        return request

    def _find_oldest_droppable_message_span(
        self, request: ChatCompletionRequest
    ) -> Optional[tuple[int, int]]:
        """Find the oldest message span that can be safely dropped."""
        messages = request.messages or []
        for idx, message in enumerate(messages):
            if idx == 0:
                continue  # keep first message (likely system)
            if message.role == MessageRole.ASSISTANT and bool(message.tool_calls):
                end = idx + 1
                while end < len(messages) and messages[end].role == MessageRole.TOOL:
                    end += 1
                return idx, end
            if message.role == MessageRole.TOOL:
                return idx, idx + 1
            if getattr(message, "name", None) == "mcp_tool_retry_hint":
                return idx, idx + 1
        return None

    def _drop_oldest_messages(
        self, request: ChatCompletionRequest, max_size: int
    ) -> ChatCompletionRequest:
        """Drop oldest messages until request fits."""
        current_size = self._get_payload_size_from_request(request)

        while current_size > max_size and len(request.messages or []) > 1:
            drop_span = self._find_oldest_droppable_message_span(request)
            if drop_span is None:
                break
            start, end = drop_span
            dropped = request.messages[start:end]
            del request.messages[start:end]
            dropped_desc = ", ".join(
                f"role={msg.role}, name={getattr(msg, 'name', 'N/A')}"
                for msg in dropped
            )
            self.logger.debug(f"[AdaptiveCompressor] Dropped messages: {dropped_desc}")
            current_size = self._get_payload_size_from_request(request)

        return request


# Default compressor instance
_default_compressor: Optional[AdaptiveCompressor] = None


def get_default_compressor() -> AdaptiveCompressor:
    """Get or create the default adaptive compressor."""
    global _default_compressor
    if _default_compressor is None:
        _default_compressor = AdaptiveCompressor(logger_instance=logger)
    return _default_compressor
