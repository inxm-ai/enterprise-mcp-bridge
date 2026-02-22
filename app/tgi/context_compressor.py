"""
Context compression module for managing large conversation histories.

Implements multiple strategies for handling payload size limits:
- Sliding window: Keep recent messages, compress older ones
- Hierarchical summarization: Build a summary tree for massive messages
- Adaptive compression: Balance between quality and size constraints
"""

import json
import logging
import asyncio
import re
import hashlib
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from app.tgi.models import (
    ChatCompletionRequest,
    Message,
    MessageRole,
)

logger = logging.getLogger("uvicorn.error")

_CONTRACT_HINTS = (
    "components_script",
    "service_script",
    "test_script",
    "template_parts",
    "response_format",
    "json_schema",
    "phase 1",
    "phase-1",
    "required output",
)
_RECENT_CONTRACT_WINDOW = 6
_HUGE_BLOCK_CHUNK_RATIO = 0.75
_HUGE_BLOCK_OVERLAP_RATIO = 0.15


def _is_contract_bearing_user_message(
    message: Message, index: int, total_messages: int
) -> bool:
    if message.role != MessageRole.USER:
        return False
    if not isinstance(message.content, str):
        return False
    if index < max(0, total_messages - _RECENT_CONTRACT_WINDOW):
        return False

    text = message.content.lower()
    if any(hint in text for hint in _CONTRACT_HINTS):
        return True
    if "```json" in text and ("required" in text or "schema" in text):
        return True
    return False


def _is_protected_message(message: Message, index: int, total_messages: int) -> bool:
    if index == 0 and message.role == MessageRole.SYSTEM:
        return True
    return _is_contract_bearing_user_message(message, index, total_messages)


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
        # while protecting contract-bearing messages.
        messages = request.messages or []
        if len(messages) <= self.window_size + 1:
            # Not enough messages to apply sliding window, skip
            self.logger.debug(
                "[SlidingWindowCompressor] Not enough messages for sliding window "
                f"(have {len(messages)}, window={self.window_size})"
            )
            return request, stats

        protected_indexes = {
            idx
            for idx, message in enumerate(messages)
            if _is_protected_message(message, idx, len(messages))
        }

        # Keep all system messages, plus protected non-system messages,
        # plus the most recent `window_size` unprotected non-system messages.
        system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]
        non_system_indexed = [
            (idx, m) for idx, m in enumerate(messages) if m.role != MessageRole.SYSTEM
        ]
        keep_indexes = {
            idx for idx, _ in non_system_indexed if idx in protected_indexes
        }
        unprotected_indexes = [
            idx for idx, _ in non_system_indexed if idx not in keep_indexes
        ]
        keep_indexes.update(unprotected_indexes[-self.window_size :])

        to_summarize = [m for idx, m in non_system_indexed if idx not in keep_indexes]
        to_keep = [m for idx, m in non_system_indexed if idx in keep_indexes]

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

        # Find and compress large messages while preserving protected contract instructions.
        messages = request.messages or []
        role_priority = {
            MessageRole.TOOL: 0,
            MessageRole.ASSISTANT: 1,
            MessageRole.USER: 2,
            MessageRole.SYSTEM: 3,
        }
        candidates = []
        for idx, message in enumerate(messages):
            is_protected = _is_protected_message(message, idx, len(messages))
            if idx == 0 and message.role == MessageRole.SYSTEM:
                continue
            if not (message.content and isinstance(message.content, str)):
                continue
            message_size = len(message.content.encode("utf-8"))
            if message_size <= self.chunk_size_chars:
                continue
            if is_protected and message_size <= max_size:
                continue
            candidates.append(
                (
                    role_priority.get(message.role, 4),
                    idx,
                    message,
                    is_protected,
                )
            )

        candidates.sort(key=lambda item: (item[0], item[1]))

        for _, _, message, is_protected in candidates:
            source = self._classify_message_source(message)
            self.logger.info(
                f"[HierarchicalSummarizer] Found large message "
                f"(role={message.role}, size={len((message.content or '').encode('utf-8'))}, "
                f"source={source}, protected={is_protected}), "
                f"applying hierarchical summarization"
            )

            message.content = await self._summarize_message_content(
                content=message.content,
                base_request=request,
                summarize_fn=summarize_fn,
                max_size=max_size,
                preserve_prefix=is_protected,
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

    async def _summarize_message_content(
        self,
        *,
        content: str,
        base_request: ChatCompletionRequest,
        summarize_fn,
        max_size: int,
        preserve_prefix: bool,
    ) -> str:
        summary = await self._hierarchical_summarize(
            content=content,
            base_request=base_request,
            summarize_fn=summarize_fn,
            max_size=max_size,
        )
        if not preserve_prefix:
            return summary

        head_keep = min(2500, max(500, int(max_size * 0.15)))
        tail_keep = min(1200, max(200, int(max_size * 0.08)))
        prefix = content[:head_keep]
        suffix = content[-tail_keep:] if len(content) > (head_keep + tail_keep) else ""
        compacted = (
            "[PROTECTED CONTEXT COMPACTION]\n"
            "Critical instruction/source content was oversized and compacted.\n"
            "Most important bullet points:\n"
            f"{summary}\n"
            "\n[PRESERVED PREFIX]\n"
            f"{prefix}"
        )
        if suffix:
            compacted += f"\n\n[PRESERVED SUFFIX]\n{suffix}"
        return compacted

    async def _hierarchical_summarize(
        self,
        *,
        content: str,
        base_request: ChatCompletionRequest,
        summarize_fn,
        max_size: int,
    ) -> str:
        """
        Recursively summarize content in a hierarchical manner.

        Process:
        1. Split content into chunks
        2. Summarize each chunk
        3. Group summaries and summarize groups (map-reduce pattern)
        4. Return final summary
        """
        target_chunk_size = max(
            self.chunk_size_chars,
            int(max_size * _HUGE_BLOCK_CHUNK_RATIO),
        )
        overlap_size = max(100, int(target_chunk_size * _HUGE_BLOCK_OVERLAP_RATIO))
        if len(content.encode("utf-8")) > max_size:
            chunks = self._split_into_overlapping_chunks(
                content,
                chunk_size=target_chunk_size,
                overlap=overlap_size,
            )
        else:
            chunks = self._split_into_chunks(content, self.chunk_size_chars)

        if len(chunks) == 1:
            # Single chunk, no hierarchical processing needed
            self.logger.debug(
                f"[HierarchicalSummarizer] Single chunk ({len(chunks[0])} chars), "
                f"applying simple summarization"
            )
            return await summarize_fn(
                base_request=base_request,
                content=(
                    "Summarize to the most important bullet points. "
                    "Preserve key facts, IDs, constraints, and decisions.\n\n"
                    f"{content}"
                ),
                access_token="",
                outer_span=None,
            )

        # Step 1: Summarize each chunk
        self.logger.debug(
            f"[HierarchicalSummarizer] Summarizing {len(chunks)} chunks at level 0"
        )
        chunk_tasks = [
            summarize_fn(
                base_request=base_request,
                content=(
                    "Summarize this section into the most important bullet points. "
                    "Keep critical facts and constraints.\n\n"
                    f"{chunk}"
                ),
                access_token="",
                outer_span=None,
            )
            for chunk in chunks
        ]
        chunk_summaries = list(await asyncio.gather(*chunk_tasks))
        self.logger.debug(
            f"[HierarchicalSummarizer] Summarized {len(chunk_summaries)}/{len(chunks)} chunks"
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
            grouped_inputs = []
            for i in range(0, len(chunk_summaries), self.group_size):
                group = chunk_summaries[i : i + self.group_size]
                group_text = "\n\n".join(
                    [f"[Part {j + 1}]\n{s}" for j, s in enumerate(group)]
                )
                grouped_inputs.append(group_text)

            group_tasks = [
                summarize_fn(
                    base_request=base_request,
                    content=(
                        "Combine these section summaries into the most important bullet points. "
                        "Remove duplication and keep key facts/constraints.\n\n"
                        f"{group_text}"
                    ),
                    access_token="",
                    outer_span=None,
                )
                for group_text in grouped_inputs
            ]
            grouped_summaries = list(await asyncio.gather(*group_tasks))

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

    @staticmethod
    def _split_into_overlapping_chunks(
        content: str, chunk_size: int, overlap: int
    ) -> List[str]:
        if len(content) <= chunk_size:
            return [content]
        if overlap >= chunk_size:
            overlap = max(1, chunk_size // 4)

        chunks: List[str] = []
        step = max(1, chunk_size - overlap)
        start = 0
        while start < len(content):
            end = min(len(content), start + chunk_size)
            chunks.append(content[start:end])
            if end >= len(content):
                break
            start += step
        return chunks

    @staticmethod
    def _classify_message_source(message: Message) -> str:
        text = (message.content or "").lower()
        if message.role == MessageRole.TOOL:
            return "tool_output"
        if "json_schema" in text or "response_format" in text or '"schema"' in text:
            return "schema"
        if "tool" in text and ("result" in text or "structuredcontent" in text):
            return "tool_output"
        return "single_message_content"


class AdaptiveCompressor(CompressionStrategy):
    """
    Adaptive compression that combines multiple strategies intelligently.

    Strategy selection:
    1. First pass: Compact JSON and whitespace
    2. Second pass: Offload oversized context into bullet summaries + retrieval refs
    3. Third pass: Apply hierarchical summarization to large messages
    4. Fourth pass: Apply sliding window compression to conversation history
    5. Final pass: Drop oldest tool-related messages if still oversized
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

        stats.metadata["oversized_sources"] = self._diagnose_oversized_sources(
            request, max_size
        )

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

        # Step 2: Offload oversized context to compact bullet summaries.
        self.logger.info(
            "[AdaptiveCompressor] Step 2: Context offload to bullet summaries"
        )
        request, offload_meta = self._offload_context_to_bullets(request, max_size)
        current_size = self._get_payload_size_from_request(request)
        if offload_meta.get("items"):
            stats.metadata["offloaded_context"] = offload_meta

        if current_size <= max_size:
            stats.compressed_size = current_size
            stats.compression_ratio = current_size / original_size
            stats.messages_summarized = int(offload_meta.get("count", 0))
            stats.metadata["steps_used"] = ["compact", "offload_context"]
            self.logger.info(f"[AdaptiveCompressor] {stats.summary()}")
            return request, stats

        # Step 3: Apply hierarchical summarization to large messages
        self.logger.info("[AdaptiveCompressor] Step 3: Hierarchical summarization")
        request, hier_stats = await self.hierarchical.compress(
            request, max_size, summarize_fn
        )
        current_size = self._get_payload_size_from_request(request)

        if current_size <= max_size:
            stats.compressed_size = current_size
            stats.compression_ratio = current_size / original_size
            stats.messages_summarized = (
                int(offload_meta.get("count", 0)) + hier_stats.messages_summarized
            )
            stats.metadata["steps_used"] = [
                "compact",
                "offload_context",
                "hierarchical",
            ]
            self.logger.info(f"[AdaptiveCompressor] {stats.summary()}")
            return request, stats

        # Step 4: Apply sliding window compression
        self.logger.info("[AdaptiveCompressor] Step 4: Sliding window compression")
        request, window_stats = await self.sliding_window.compress(
            request, max_size, summarize_fn
        )
        current_size = self._get_payload_size_from_request(request)

        if current_size <= max_size:
            stats.compressed_size = current_size
            stats.compression_ratio = current_size / original_size
            stats.messages_summarized = (
                int(offload_meta.get("count", 0))
                + hier_stats.messages_summarized
                + window_stats.messages_summarized
            )
            stats.metadata["steps_used"] = [
                "compact",
                "offload_context",
                "hierarchical",
                "sliding_window",
            ]
            self.logger.info(f"[AdaptiveCompressor] {stats.summary()}")
            return request, stats

        # Step 5: Drop oldest tool-related messages
        self.logger.info("[AdaptiveCompressor] Step 5: Dropping old messages")
        request = self._drop_oldest_messages(request, max_size)
        current_size = self._get_payload_size_from_request(request)

        stats.compressed_size = current_size
        stats.compression_ratio = current_size / original_size
        stats.messages_summarized = (
            int(offload_meta.get("count", 0))
            + hier_stats.messages_summarized
            + window_stats.messages_summarized
        )
        stats.metadata["steps_used"] = [
            "compact",
            "offload_context",
            "hierarchical",
            "sliding_window",
            "drop",
        ]
        self.logger.info(f"[AdaptiveCompressor] {stats.summary()}")

        return request, stats

    def _offload_context_to_bullets(
        self, request: ChatCompletionRequest, max_size: int
    ) -> Tuple[ChatCompletionRequest, Dict[str, Any]]:
        messages = request.messages or []
        current_size = self._get_payload_size_from_request(request)
        threshold = max(2048, int(max_size * 0.35))
        offloaded_items: List[Dict[str, Any]] = []

        if current_size <= max_size:
            return request, {"count": 0, "items": []}

        candidates: List[Tuple[int, int, Message]] = []
        for idx, message in enumerate(messages):
            if _is_protected_message(message, idx, len(messages)):
                continue
            if not isinstance(message.content, str) or not message.content:
                continue
            content_bytes = len(message.content.encode("utf-8"))
            if content_bytes < threshold:
                continue
            candidates.append((content_bytes, idx, message))

        candidates.sort(reverse=True)

        for content_bytes, idx, message in candidates:
            if current_size <= max_size:
                break
            original_content = message.content or ""
            ref_id = self._build_context_ref_id(idx, message.role, original_content)
            source = HierarchicalSummarizer._classify_message_source(message)
            bullets = self._content_bullets(original_content)
            summary_block = self._build_offload_summary(
                ref_id=ref_id,
                source=source,
                content_bytes=content_bytes,
                bullets=bullets,
            )

            message.content = summary_block
            current_size = self._get_payload_size_from_request(request)
            offloaded_items.append(
                {
                    "ref_id": ref_id,
                    "index": idx,
                    "role": str(message.role),
                    "source": source,
                    "original_bytes": content_bytes,
                    "summary_bullets": bullets,
                }
            )

        if offloaded_items:
            self.logger.warning(
                "[AdaptiveCompressor] Offloaded %d oversized context blocks to summary refs",
                len(offloaded_items),
            )

        return request, {"count": len(offloaded_items), "items": offloaded_items}

    @staticmethod
    def _build_context_ref_id(index: int, role: MessageRole, content: str) -> str:
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:10]
        role_name = str(role).split(".")[-1].lower()
        return f"ctx_{index}_{role_name}_{digest}"

    @staticmethod
    def _build_offload_summary(
        *, ref_id: str, source: str, content_bytes: int, bullets: List[str]
    ) -> str:
        bullet_lines = "\n".join(f"- {item}" for item in bullets[:6])
        if not bullet_lines:
            bullet_lines = "- Context omitted due to payload limits"
        return (
            "[CONTEXT OFFLOAD SUMMARY]\n"
            f"- ref_id: {ref_id}\n"
            f"- source: {source}\n"
            f"- original_size_bytes: {content_bytes}\n"
            "- retrieval_tool: request_context(ref_id)\n"
            "- retrieval_hint: Ask for specific ref_id if exact raw payload is required.\n"
            "- key_points:\n"
            f"{bullet_lines}"
        )

    @staticmethod
    def _content_bullets(content: str) -> List[str]:
        text = (content or "").strip()
        if not text:
            return []

        bullets: List[str] = []
        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    keys = list(parsed.keys())
                    if keys:
                        bullets.append("top-level keys: " + ", ".join(keys[:10]))
                    for key in keys[:4]:
                        value = parsed.get(key)
                        if isinstance(value, dict):
                            bullets.append(f"{key}: object with {len(value)} keys")
                        elif isinstance(value, list):
                            bullets.append(f"{key}: list with {len(value)} entries")
                        elif isinstance(value, str):
                            bullets.append(f"{key}: text ({len(value)} chars)")
                        else:
                            bullets.append(f"{key}: {type(value).__name__}")
                elif isinstance(parsed, list):
                    bullets.append(f"array payload with {len(parsed)} entries")
                    if parsed:
                        bullets.append(f"first item type: {type(parsed[0]).__name__}")
            except Exception:
                pass

        if not bullets:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            for line in lines[:6]:
                if len(line) > 180:
                    line = line[:177] + "..."
                bullets.append(line)

        return bullets[:6]

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
            if _is_protected_message(message, idx, len(messages)):
                continue
            if message.role == MessageRole.ASSISTANT and bool(message.tool_calls):
                end = idx + 1
                while (
                    end < len(messages)
                    and messages[end].role == MessageRole.TOOL
                    and not _is_protected_message(messages[end], end, len(messages))
                ):
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

    def _diagnose_oversized_sources(
        self, request: ChatCompletionRequest, max_size: int
    ) -> List[Dict[str, Any]]:
        diagnostics: List[Dict[str, Any]] = []
        messages = request.messages or []

        for idx, message in enumerate(messages):
            content = message.content if isinstance(message.content, str) else ""
            message_bytes = len(content.encode("utf-8"))
            tool_arg_bytes = 0
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    args = getattr(
                        getattr(tool_call, "function", None), "arguments", ""
                    )
                    if isinstance(args, str):
                        tool_arg_bytes += len(args.encode("utf-8"))

            total_bytes = message_bytes + tool_arg_bytes
            if total_bytes <= max(2048, int(max_size * 0.1)):
                continue

            source = HierarchicalSummarizer._classify_message_source(message)
            if tool_arg_bytes > message_bytes and tool_arg_bytes > 0:
                source = "tool_call_arguments"
            if source == "single_message_content" and re.search(
                r"schema|response_format|properties|required|json",
                content,
                re.IGNORECASE,
            ):
                source = "schema"

            diagnostics.append(
                {
                    "index": idx,
                    "role": str(message.role),
                    "source": source,
                    "content_bytes": message_bytes,
                    "tool_args_bytes": tool_arg_bytes,
                    "total_bytes": total_bytes,
                    "protected": _is_protected_message(message, idx, len(messages)),
                }
            )

        diagnostics.sort(key=lambda item: item["total_bytes"], reverse=True)
        top = diagnostics[:5]
        if top:
            self.logger.warning(
                "[AdaptiveCompressor] Oversized source diagnostics (top=%d): %s",
                len(top),
                top,
            )
        return top


# Default compressor instance
_default_compressor: Optional[AdaptiveCompressor] = None


def get_default_compressor() -> AdaptiveCompressor:
    """Get or create the default adaptive compressor."""
    global _default_compressor
    if _default_compressor is None:
        _default_compressor = AdaptiveCompressor(logger_instance=logger)
    return _default_compressor
