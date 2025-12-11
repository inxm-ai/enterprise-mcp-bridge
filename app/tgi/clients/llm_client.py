"""
LLM client module for handling direct communication with the LLM API.
"""

import logging
import os
import time
import uuid
from typing import AsyncGenerator, List, Optional
import aiohttp
from app.vars import LLM_MAX_PAYLOAD_BYTES, TGI_MODEL_NAME
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

from app.tgi.models.model_formats import BaseModelFormat, get_model_format_for
from app.tgi.protocols.chunk_reader import chunk_reader
from app.tgi.context_compressor import get_default_compressor

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

    def _serialize_payload(self, payload: dict) -> str:
        """Serialize payload to JSON."""
        import json

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _payload_size(self, serialized_payload: str) -> int:
        """Calculate payload size in bytes."""
        return len(serialized_payload.encode("utf-8"))

    def _filter_llm_payload(self, payload: dict) -> dict:
        """
        Remove non-LLM fields from the outgoing payload to avoid API errors.
        """
        payload.pop("persist_inner_thinking", None)
        return payload

    async def _prepare_payload(
        self, request: ChatCompletionRequest, access_token: str = ""
    ) -> tuple[dict, str, int]:
        """
        Prepare and compress payload if needed using adaptive compression strategy.

        Args:
            request: ChatCompletionRequest to prepare
            access_token: Access token for summarization operations

        Returns:
            Tuple of (payload_dict, serialized_payload, payload_size)
        """
        payload = self._generate_llm_payload(request)
        serialized = self._serialize_payload(payload)
        size = self._payload_size(serialized)

        if size <= LLM_MAX_PAYLOAD_BYTES:
            return payload, serialized, size

        # Payload exceeds limit, apply compression
        self.logger.warning(
            "[LLMClient] Payload size %s exceeds limit %s, applying adaptive compression",
            size,
            LLM_MAX_PAYLOAD_BYTES,
        )

        compressor = get_default_compressor()
        compressed_request, stats = await compressor.compress(
            request,
            max_size=LLM_MAX_PAYLOAD_BYTES,
            summarize_fn=self.summarize_text,
        )
        request = compressed_request

        payload = self._generate_llm_payload(request)
        serialized = self._serialize_payload(payload)
        size = self._payload_size(serialized)

        self.logger.info(f"[LLMClient] Compression result: {stats.summary()}")

        if size > LLM_MAX_PAYLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"LLM payload size {size} remains above limit {LLM_MAX_PAYLOAD_BYTES} after compression",
            )

        return payload, serialized, size

    def _generate_llm_payload(self, request: ChatCompletionRequest) -> dict:
        """Generate the payload for the LLM API request."""

        self.model_format.prepare_request(request)

        payload = self._filter_llm_payload(request.model_dump(exclude_none=True))

        # tool_choice is only valid when tools are specified
        # OpenAI API returns 400 error if tool_choice is sent without tools
        if not payload.get("tools"):
            payload.pop("tool_choice", None)

        # model parameter must not be empty string
        # OpenAI API returns 400 error "you must provide a model parameter"
        if payload.get("model") == "" or payload.get("model") is None:
            payload["model"] = TGI_MODEL_NAME

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
            # Prepare payload with size-aware compression
            self.logger.info("[LLMClient] Preparing payload for streaming request")
            payload, serialized_payload, payload_size = await self._prepare_payload(
                request, access_token
            )
            self.logger.debug(
                "[LLMClient] Prepared streaming payload (bytes=%s, messages=%s)",
                payload_size,
                len(payload.get("messages", [])),
            )

            self.logger.debug(
                f"[LLMClient] Opening HTTP session to {self.tgi_url}/chat/completions"
            )
            async with aiohttp.ClientSession() as session:
                self.logger.debug("[LLMClient] Sending POST request to LLM")
                async with session.post(
                    f"{self.tgi_url}/chat/completions",
                    headers=self._get_headers(access_token),
                    data=serialized_payload,
                ) as response:
                    self.logger.debug(
                        f"[LLMClient] Received response, status={response.status}"
                    )
                    self.logger.debug(
                        f"[LLMClient] Response headers: {dict(getattr(response, 'headers', {}))}"
                    )
                    self.logger.debug(
                        f"[LLMClient] Response content-type: {getattr(response, 'content_type', None)}"
                    )
                    self.logger.debug(
                        f"[LLMClient] Response content-length: {getattr(response, 'content_length', None)}"
                    )

                    if not response.ok:
                        error_text = await response.text()
                        error_msg = f"LLM API error: {response.status} {error_text}"
                        self.logger.error(f"[LLMClient] {error_msg}")
                        self.logger.debug(f"[LLMClient] Payload: {serialized_payload}")
                        if parent_span is not None:
                            parent_span.set_attribute("error", True)
                            parent_span.set_attribute("error.message", error_msg)

                        # Return error as streaming response
                        error_chunk = ChatCompletionChunk(
                            id=self.create_completion_id(),
                            created=int(time.time()),
                            model=request.model or TGI_MODEL_NAME,
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

                    # Stream chunks immediately without buffering
                    # Ensure proper SSE format by adding \n if not already present
                    chunk_count = 0
                    self.logger.info(
                        f"[LLMClient] Response OK, starting to stream chunks (model={request.model})"
                    )
                    self.logger.debug(
                        "[LLMClient] About to iterate over response.content"
                    )
                    self.logger.debug(
                        f"[LLMClient] Response content object: {response.content}"
                    )

                    async for chunk in response.content:
                        self.logger.debug(
                            "[LLMClient] Received raw chunk from HTTP response"
                        )
                        chunk_str = chunk.decode("utf-8")
                        if chunk_str:
                            chunk_count += 1
                            if chunk_count == 1:
                                self.logger.debug(
                                    f"[LLMClient] Received first chunk from LLM, length={len(chunk_str)}"
                                )
                            if chunk_count % 10 == 0:
                                self.logger.info(
                                    f"[LLMClient] Received {chunk_count} chunks so far"
                                )
                            self.logger.debug(
                                f"[LLMClient] Chunk {chunk_count}: {chunk_str[:100]}..."
                            )
                            # Ensure proper SSE format with \n\n terminator
                            # Most chunks will end with \n, so we add one more \n
                            if not chunk_str.endswith("\n\n"):
                                if chunk_str.endswith("\n"):
                                    chunk_str += "\n"
                                else:
                                    chunk_str += "\n\n"
                            yield chunk_str
                        else:
                            self.logger.debug(
                                "[LLMClient] Received empty chunk, skipping"
                            )

                    self.logger.info(
                        f"[LLMClient] Streaming completed, total chunks={chunk_count}"
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
                model=request.model or TGI_MODEL_NAME,
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
            span.set_attribute("llm.model", request.model or TGI_MODEL_NAME)

            try:
                # Convert request to dict for JSON serialization, ensure stream=False
                request.stream = False
                payload, serialized_payload, payload_size = await self._prepare_payload(
                    request, access_token
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
                self.logger.error(f"[LLMClient] {error_msg}", exc_info=True)
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise HTTPException(status_code=500, detail=error_msg)

    async def summarize_conversation(
        self,
        conversation: List[Message],
        base_request: ChatCompletionRequest,
        access_token: str,
        outer_span,
    ):
        """
        Summarize the conversation so far using the LLM.

        Args:
            base_request: Base request to use for summarization
            access_token: Access token for authentication
            outer_span: Parent span for tracing

        Returns:
            Summarized text
        """

        def stringify_messages(messages: List[Message]) -> str:
            return "\n".join(
                [
                    f"## {message.role.value.capitalize()}\n{message.content}"
                    for message in messages
                    if message.content and message.role != MessageRole.SYSTEM
                ]
            )

        await self.ask(
            base_prompt="# System Prompt: Summarization Expert\n\nYou are a **summarization expert**. Your task is to read the **user's question** and the **assistant's reply** (which may include tool outputs), and then produce a concise, accurate summary of the reply that directly addresses the user's question.  \n\n---\n\n## Key Instructions\n\n### 1. Stay Aligned with the User's Question\n- Only summarize the information that is relevant to what the user asked.  \n- If the assistant's reply contains extraneous content (e.g., HTML markup in emails, raw metadata, or formatting noise), **ignore it** unless the user explicitly requested it.  \n\n### 2. Context-Sensitive Relevance\n- If the user asked about the *content* (e.g., “Summarize the email”), focus only on the meaningful text.  \n- If the user asked about *structure or metadata* (e.g., “Which senders use HTML emails?”), then the presence of HTML or metadata details is essential and should be included in the summary.  \n\n### 3. Clarity & Brevity\n- Rewrite in plain, natural language.  \n- Strip out technical noise, boilerplate, or irrelevant tool artifacts.  \n- Preserve essential details (facts, names, actions, outcomes).  \n\n### 4. Prioritization\n- Always privilege the **user's intent** over the assistant's full reply.  \n- Keep summaries **short but complete**: capture the key points, not every detail.  \n\n---\n\n## Examples\n\n- **User asks:** “Summarize this email.”  \n  - **Assistant reply (tool output):** Includes full HTML source.  \n  - **Your summary:** Only the human-readable body text of the email.  \n\n- **User asks:** “Which senders use HTML emails?”  \n  - **Assistant reply:** Includes headers and HTML details.\n  - **Your summary:** Mention the senders and the fact they use HTML formatting.\n",
            base_request=base_request,
            question=stringify_messages(conversation),
            access_token=access_token,
            outer_span=outer_span,
        )

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
            model=base_request.model or TGI_MODEL_NAME,
            stream=True,
            temperature=base_request.temperature,
            max_tokens=base_request.max_tokens,
            top_p=base_request.top_p,
        )

        llm_stream_generator = self.stream_completion(
            llm_request, access_token, outer_span
        )

        result = ""
        async with chunk_reader(llm_stream_generator) as reader:
            async for content_piece in reader.as_str():
                result += content_piece

        return result
