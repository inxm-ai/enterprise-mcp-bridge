"""
LLM client module for handling direct communication with the LLM API.
"""

import json
import logging
import os
import time
import uuid
from typing import AsyncGenerator, List, Optional
import aiohttp
from app.vars import LLM_MAX_PAYLOAD_BYTES, TOOL_CHUNK_SIZE
from opentelemetry import trace

from app.tgi.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Choice,
    DeltaMessage,
    Message,
    MessageRole,
    Usage,
)
from app.utils import mask_token
from fastapi import HTTPException

from app.tgi.model_formats import BaseModelFormat, get_model_format_for

logger = logging.getLogger("uvicorn.error")
tracer = trace.get_tracer(__name__)


class LLMClient:
    """Client for communicating with the LLM API."""

    def __init__(self, model_format: Optional[BaseModelFormat] = None):
        self.logger = logger
        self.tgi_url = os.environ.get("TGI_URL", "")
        self.tgi_token = os.environ.get("TGI_TOKEN", "")
        self.model_format = model_format or get_model_format_for()

        # Ensure TGI_URL doesn't end with slash for consistent URL building
        if self.tgi_url.endswith("/"):
            self.tgi_url = self.tgi_url[:-1]

        self.logger.info(f"[LLMClient] Initialized with URL: {self.tgi_url}")
        if self.tgi_token:
            self.logger.info(
                mask_token(
                    f"[LLMClient] Using authentication token: {self.tgi_token[:10]}...",
                    self.tgi_token[:10],
                )
            )
        else:
            self.logger.info("[LLMClient] No authentication token configured")

    def _get_headers(self, access_token="fake") -> dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "mcp-rest-server/1.0",
        }

        if self.tgi_token:
            headers["Authorization"] = f"Bearer {self.tgi_token}"
        else:
            headers["Authorization"] = f"Bearer {access_token}"

        return headers

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

    def _serialize_payload(self, payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _payload_size(self, serialized_payload: str) -> int:
        return len(serialized_payload.encode("utf-8"))

    def _minify_messages_for_payload(self, request: ChatCompletionRequest) -> None:
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

    def _truncate_messages_until_fit(
        self, request: ChatCompletionRequest, limit: int
    ) -> tuple[dict, str, int]:
        """Iteratively truncate the largest assistant/tool messages."""
        max_chars = TOOL_CHUNK_SIZE
        candidates: List[Message] = [
            message
            for message in request.messages or []
            if message.role in (MessageRole.TOOL, MessageRole.ASSISTANT)
            and isinstance(message.content, str)
        ]
        candidates.sort(key=lambda msg: len(msg.content or ""), reverse=True)

        payload = request.model_dump(exclude_none=True)
        serialized = self._serialize_payload(payload)
        size = self._payload_size(serialized)

        for message in candidates:
            if not message.content or len(message.content) <= max_chars:
                continue
            original_len = len(message.content)
            message.content = f"{message.content[:max_chars]}...[truncated {original_len - max_chars} chars]"
            payload = request.model_dump(exclude_none=True)
            serialized = self._serialize_payload(payload)
            size = self._payload_size(serialized)
            self.logger.warning(
                "[LLMClient] Truncated %s message to fit payload (removed %s chars)",
                message.role,
                original_len - max_chars,
            )
            if size <= limit:
                break

        return payload, serialized, size

    def _find_oldest_droppable_message_index(
        self, request: ChatCompletionRequest
    ) -> int | None:
        for idx, message in enumerate(request.messages or []):
            if idx == 0:
                continue  # keep first message (likely system)
            if message.role == MessageRole.TOOL:
                return idx
        for idx, message in enumerate(request.messages or []):
            if idx == 0:
                continue
            if message.role == MessageRole.ASSISTANT and bool(message.tool_calls):
                return idx
        for idx, message in enumerate(request.messages or []):
            if idx == 0:
                continue
            if getattr(message, "name", None) == "mcp_tool_retry_hint":
                return idx
        return None

    def _drop_messages_until_fit(
        self, request: ChatCompletionRequest, limit: int
    ) -> tuple[dict, str, int]:
        payload = request.model_dump(exclude_none=True)
        serialized = self._serialize_payload(payload)
        size = self._payload_size(serialized)

        while size > limit and len(request.messages) > 1:
            drop_index = self._find_oldest_droppable_message_index(request)
            if drop_index is None:
                break
            dropped = request.messages.pop(drop_index)
            self.logger.warning(
                "[LLMClient] Dropped message role=%s name=%s to reduce payload",
                dropped.role,
                getattr(dropped, "name", ""),
            )
            payload = request.model_dump(exclude_none=True)
            serialized = self._serialize_payload(payload)
            size = self._payload_size(serialized)

        return payload, serialized, size

    def _ensure_payload_size(
        self, request: ChatCompletionRequest, payload: dict
    ) -> tuple[dict, str, int]:
        serialized = self._serialize_payload(payload)
        size = self._payload_size(serialized)

        if size <= LLM_MAX_PAYLOAD_BYTES:
            return payload, serialized, size

        self.logger.warning(
            "[LLMClient] Payload size %s exceeds limit %s, applying compaction",
            size,
            LLM_MAX_PAYLOAD_BYTES,
        )

        # First pass: compact whitespace and JSON formatting
        self._minify_messages_for_payload(request)
        payload = request.model_dump(exclude_none=True)
        serialized = self._serialize_payload(payload)
        size = self._payload_size(serialized)

        if size <= LLM_MAX_PAYLOAD_BYTES:
            return payload, serialized, size

        # Second pass: truncate largest assistant/tool messages
        payload, serialized, size = self._truncate_messages_until_fit(
            request, LLM_MAX_PAYLOAD_BYTES
        )
        if size <= LLM_MAX_PAYLOAD_BYTES:
            return payload, serialized, size

        # Final pass: drop oldest tool-related messages if still too large
        payload, serialized, size = self._drop_messages_until_fit(
            request, LLM_MAX_PAYLOAD_BYTES
        )

        if size > LLM_MAX_PAYLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="LLM payload size remains above configured limit after trimming",
            )

        return payload, serialized, size

    def _prepare_payload(self, request: ChatCompletionRequest) -> tuple[dict, str, int]:
        payload = self._generate_llm_payload(request)
        return self._ensure_payload_size(request, payload)

    def _generate_llm_payload(self, request: ChatCompletionRequest) -> dict:
        """Generate the payload for the LLM API request."""

        self.model_format.prepare_request(request)

        payload = request.model_dump(exclude_none=True)
        return payload

    def create_completion_id(self) -> str:
        """Generate a unique completion ID."""
        return f"chatcmpl-{uuid.uuid4().hex[:29]}"

    def create_usage_stats(
        self, prompt_tokens: int = 0, completion_tokens: int = 0
    ) -> Usage:
        """Create usage statistics."""
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    async def stream_completion(
        self,
        request: ChatCompletionRequest,
        access_token: str,
        parent_span,
    ) -> AsyncGenerator[str, None]:
        """Stream completion from the actual LLM."""
        # Remove all tracing to avoid context cleanup issues with async generators
        try:
            # Prepare payload with size-aware compaction
            payload, serialized_payload, payload_size = self._prepare_payload(request)
            self.logger.debug(
                "[LLMClient] Prepared streaming payload (bytes=%s, messages=%s)",
                payload_size,
                len(payload.get("messages", [])),
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.tgi_url}/chat/completions",
                    headers=self._get_headers(access_token),
                    data=serialized_payload,
                ) as response:

                    if not response.ok:
                        error_text = await response.text()
                        error_msg = f"LLM API error: {response.status} {error_text}"
                        self.logger.error(f"[LLMClient] {error_msg}")

                        # Return error as streaming response
                        error_chunk = ChatCompletionChunk(
                            id=self.create_completion_id(),
                            created=int(time.time()),
                            model=request.model or "unknown",
                            choices=[
                                Choice(
                                    index=0,
                                    delta=DeltaMessage(content=f"Error: {error_msg}"),
                                    finish_reason="stop",
                                )
                            ],
                        )
                        yield f"data: {error_chunk.model_dump_json()}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # Count streamed lines for observability (without tracing)
                    line_count = 0
                    self.logger.debug(
                        f"[LLMClient] Response OK, starting to stream (model={request.model}, content={response.content})"
                    )
                    async for line in response.content:
                        line_str = line.decode("utf-8").strip()
                        if line_str:
                            line_count += 1
                            yield f"{line_str}\n"

                    self.logger.debug(
                        f"[LLMClient] Streamed {line_count} lines from LLM"
                    )

        except GeneratorExit:
            # Handle generator closure (e.g., client disconnect or normal completion)
            # This is normal when streaming completes or when starting a new iteration in tool execution
            self.logger.debug(
                "[LLMClient] Generator closed - normal completion or iteration change"
            )
            # Don't re-raise - this is expected behavior during tool execution iterations
            return

        except ConnectionError as e:
            error_msg = "Connection error occurred while streaming from LLM."
            self.logger.error(f"[LLMClient] {error_msg}: {str(e)}")
            raise

        except Exception as e:
            error_msg = f"Error streaming from LLM: {str(e)}"
            self.logger.error(f"[LLMClient] {error_msg} - Stack trace:", exc_info=True)

            # Return error as streaming response
            error_chunk = ChatCompletionChunk(
                id=self.create_completion_id(),
                created=int(time.time()),
                model=request.model or "unknown",
                choices=[
                    Choice(
                        index=0,
                        delta=DeltaMessage(content=f"Error: {error_msg}"),
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        # Method end - no span cleanup needed

    async def non_stream_completion(
        self,
        request: ChatCompletionRequest,
        access_token: str,
        parent_span,
    ) -> ChatCompletionResponse:
        """Get non-streaming completion from the actual LLM."""
        with tracer.start_as_current_span("non_stream_llm_completion") as span:
            span.set_attribute("llm.url", self.tgi_url)
            span.set_attribute("llm.model", request.model or "unknown")

            try:
                # Convert request to dict for JSON serialization, ensure stream=False
                request.stream = False
                payload, serialized_payload, payload_size = self._prepare_payload(
                    request
                )
                self.logger.debug(
                    "[LLMClient] Prepared non-stream payload (bytes=%s, messages=%s)",
                    payload_size,
                    len(payload.get("messages", [])),
                )

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.tgi_url}/chat/completions",
                        headers=self._get_headers(access_token),
                        data=serialized_payload,
                    ) as response:

                        if not response.ok:
                            error_text = await response.text()
                            error_msg = f"LLM API error: {response.status} {error_text}"
                            self.logger.error(f"[LLMClient] {error_msg}")
                            span.set_attribute("error", True)
                            span.set_attribute("error.message", error_msg)
                            raise HTTPException(
                                status_code=response.status, detail=error_msg
                            )

                        # Parse response
                        response_data = await response.json()
                        return ChatCompletionResponse(**response_data)

            except HTTPException:
                raise
            except Exception as e:
                error_msg = f"Error calling LLM: {str(e)}"
                self.logger.error(f"[LLMClient] {error_msg}")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise HTTPException(status_code=500, detail=error_msg)

    async def summarize_text(
        self,
        base_request: ChatCompletionRequest,
        content: str,
        access_token: str,
        outer_span,
    ) -> str:
        """
        Summarize text using the LLM.

        Args:
            base_request: Base request to use for summarization
            content: Content to summarize
            access_token: Access token for authentication
            outer_span: Parent span for tracing

        Returns:
            Summarized text
        """
        await self.ask(
            base_prompt="# System Prompt: Summarization Expert\n\nYou are a **summarization expert**. Your task is to read the **user's question** and the **assistant's reply** (which may include tool outputs), and then produce a concise, accurate summary of the reply that directly addresses the user's question.  \n\n---\n\n## Key Instructions\n\n### 1. Stay Aligned with the User's Question\n- Only summarize the information that is relevant to what the user asked.  \n- If the assistant's reply contains extraneous content (e.g., HTML markup in emails, raw metadata, or formatting noise), **ignore it** unless the user explicitly requested it.  \n\n### 2. Context-Sensitive Relevance\n- If the user asked about the *content* (e.g., “Summarize the email”), focus only on the meaningful text.  \n- If the user asked about *structure or metadata* (e.g., “Which senders use HTML emails?”), then the presence of HTML or metadata details is essential and should be included in the summary.  \n\n### 3. Clarity & Brevity\n- Rewrite in plain, natural language.  \n- Strip out technical noise, boilerplate, or irrelevant tool artifacts.  \n- Preserve essential details (facts, names, actions, outcomes).  \n\n### 4. Prioritization\n- Always privilege the **user's intent** over the assistant's full reply.  \n- Keep summaries **short but complete**: capture the key points, not every detail.  \n\n---\n\n## Examples\n\n- **User asks:** “Summarize this email.”  \n  - **Assistant reply (tool output):** Includes full HTML source.  \n  - **Your summary:** Only the human-readable body text of the email.  \n\n- **User asks:** “Which senders use HTML emails?”  \n  - **Assistant reply:** Includes headers and HTML details.\n  - **Your summary:** Mention the senders and the fact they use HTML formatting.\n",
            base_request=base_request,
            assistant_statement=content,
            access_token=access_token,
            outer_span=outer_span,
        )

    async def ask(
        self,
        base_prompt: str,
        base_request: ChatCompletionRequest,
        outer_span,
        question: str = None,
        assistant_statement: str = None,
        access_token: str = None,
    ) -> str:
        """
        Ask a question, get a reply

        Args:
            base_prompt: A base system prompt to use for the question
            base_request: Base request to use for summarization
            question: The question to ask
            access_token: Access token for authentication
            outer_span: Parent span for tracing

        Returns:
            Summarized text
        """

        messages_history = [
            Message(
                role=MessageRole.SYSTEM,
                content=base_prompt,
            ),
        ]
        if question:
            messages_history.append(Message(role=MessageRole.USER, content=question))
        if assistant_statement:
            messages_history.append(
                Message(role=MessageRole.ASSISTANT, content=assistant_statement)
            )

        llm_request = ChatCompletionRequest(
            messages=messages_history,
            model=base_request.model,
            stream=True,
            temperature=base_request.temperature,
            max_tokens=base_request.max_tokens,
            top_p=base_request.top_p,
        )

        llm_stream_generator = self.stream_completion(
            llm_request, access_token, outer_span
        )

        result = ""

        async for raw_chunk in llm_stream_generator:
            with tracer.start_as_current_span("process_stream_chunk") as chunk_span:
                try:
                    if raw_chunk.startswith("data: "):
                        chunk_data = raw_chunk[len("data: ") :].strip()
                        if chunk_data == "[DONE]":
                            chunk_span.set_attribute("stream.done", True)
                            break
                        chunk_span.set_attribute("stream.done", False)
                        chunk = json.loads(chunk_data)
                    else:
                        continue
                except Exception as e:
                    self.logger.error(f"[LLMClient] Error parsing streamed chunk: {e}")
                    continue

                # OpenAI compatible chunk: check for choices[0].delta.content
                choices = chunk.get("choices", [])
                if not choices:
                    # weird, let's ignore
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})
                if "content" in delta and delta["content"]:
                    result += delta["content"]
                    continue
        return result
