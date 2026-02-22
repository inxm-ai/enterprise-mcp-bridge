"""
LLM client module for handling direct communication with the LLM API.
"""

import logging
import os
import time
import uuid
import json
from typing import AsyncGenerator, List, Optional
from openai import AsyncOpenAI, APIConnectionError
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

        self.client = AsyncOpenAI(
            api_key=self.tgi_token or "fake-token", base_url=self.tgi_url
        )

    def _payload_size(self, request: ChatCompletionRequest) -> int:
        """Calculate payload size in bytes."""
        return len(json.dumps(request.model_dump(exclude_none=True)).encode("utf-8"))

    def _message_size_summary(self, request: ChatCompletionRequest) -> List[dict]:
        summaries: List[dict] = []
        for index, message in enumerate(request.messages or []):
            content = message.content if message.content is not None else ""
            size = len(str(content).encode("utf-8"))
            summaries.append(
                {
                    "index": index,
                    "role": getattr(message, "role", "unknown"),
                    "bytes": size,
                }
            )
        summaries.sort(key=lambda item: item.get("bytes", 0), reverse=True)
        return summaries[:5]

    def _message_json_section_summary(
        self, request: ChatCompletionRequest
    ) -> List[dict]:
        sections: List[dict] = []
        for index, message in enumerate(request.messages or []):
            content = message.content if message.content is not None else ""
            if not isinstance(content, str):
                continue
            stripped = content.strip()
            if not stripped.startswith("{"):
                continue
            try:
                parsed = json.loads(stripped)
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue
            for key, value in parsed.items():
                size = len(
                    json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
                )
                sections.append(
                    {
                        "index": index,
                        "role": getattr(message, "role", "unknown"),
                        "section": key,
                        "bytes": size,
                    }
                )
        sections.sort(key=lambda item: item.get("bytes", 0), reverse=True)
        return sections[:8]

    def _sanitize_message_contents(self, request: ChatCompletionRequest) -> None:
        """Ensure all message contents are strings to satisfy API validation."""
        for message in request.messages or []:
            if message.content is None:
                message.content = ""

    def _build_request_params(self, request: ChatCompletionRequest) -> dict:
        """Build OpenAI request params, removing invalid tool fields."""
        self._sanitize_message_contents(request)
        params = request.model_dump(exclude_none=True)
        tools = params.get("tools")
        if tools and not params.get("tool_choice"):
            if len(tools) == 1:
                params["tool_choice"] = tools[0]
            else:
                params["tool_choice"] = "auto"
        if not tools:
            params.pop("tool_choice", None)
            params.pop("tools", None)
        params.pop("persist_inner_thinking", None)
        return params

    async def _prepare_payload(
        self, request: ChatCompletionRequest, access_token: str = ""
    ) -> ChatCompletionRequest:
        """
        Prepare and compress payload if needed using adaptive compression strategy.

        Args:
            request: ChatCompletionRequest to prepare
            access_token: Access token for summarization operations

        Returns:
            ChatCompletionRequest
        """
        # Ensure model is set
        if not request.model:
            request.model = TGI_MODEL_NAME
        self._sanitize_message_contents(request)

        size = self._payload_size(request)

        if size <= LLM_MAX_PAYLOAD_BYTES:
            return request

        # Payload exceeds limit, apply compression
        self.logger.warning(
            "[LLMClient] Payload size %s exceeds limit %s, applying adaptive compression",
            size,
            LLM_MAX_PAYLOAD_BYTES,
        )
        self.logger.warning(
            "[LLMClient] Largest message contents before compression: %s",
            self._message_size_summary(request),
        )
        section_summary = self._message_json_section_summary(request)
        if section_summary:
            self.logger.warning(
                "[LLMClient] Largest JSON sections before compression: %s",
                section_summary,
            )

        compressor = get_default_compressor()
        compressed_request, stats = await compressor.compress(
            request,
            max_size=LLM_MAX_PAYLOAD_BYTES,
            summarize_fn=self.summarize_text,
        )
        request = compressed_request

        size = self._payload_size(request)

        self.logger.info(f"[LLMClient] Compression result: {stats.summary()}")
        oversized_sources = stats.metadata.get("oversized_sources") or []
        if oversized_sources:
            self.logger.warning(
                "[LLMClient] Oversized payload sources after analysis: %s",
                oversized_sources,
            )

        if size > LLM_MAX_PAYLOAD_BYTES:
            if oversized_sources:
                top = oversized_sources[0]
                self.logger.error(
                    "[LLMClient] Payload still too large; primary source role=%s index=%s source=%s total_bytes=%s",
                    top.get("role"),
                    top.get("index"),
                    top.get("source"),
                    top.get("total_bytes"),
                )
            raise HTTPException(
                status_code=413,
                detail=f"LLM payload size {size} remains above limit {LLM_MAX_PAYLOAD_BYTES} after compression",
            )

        return request

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
            self.logger.info("[LLMClient] Preparing payload for streaming request.")
            request = await self._prepare_payload(request, access_token)

            self.logger.debug(f"[LLMClient] Opening stream to {self.tgi_url}")

            # Convert Pydantic model to dict and filter invalid tool fields
            params = self._build_request_params(request)
            params["stream"] = True
            self.logger.debug(f"[LLMClient] Streaming request params: {params}")

            stream = await self.client.chat.completions.create(**params)

            async for chunk in stream:
                self.logger.debug(f"[LLMClient] Received openai chunk: {chunk}")
                # Convert openai chunk to dict
                chunk_dict = chunk.model_dump(mode="json", exclude_none=True)

                # Yield formatted SSE
                import json

                yield f"data: {json.dumps(chunk_dict)}\n\n"

            yield "data: [DONE]\n\n"

        except GeneratorExit:
            self.logger.debug(
                "[LLMClient] Generator closed - normal completion or iteration change"
            )
            return

        except APIConnectionError as e:
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
                request.stream = False
                request = await self._prepare_payload(request, access_token)

                params = self._build_request_params(request)

                response = await self.client.chat.completions.create(**params)

                # Convert openai ChatCompletion to our ChatCompletionResponse
                return ChatCompletionResponse(**response.model_dump(mode="json"))

            except Exception as e:
                self.logger.error(
                    f"[LLMClient] Error in non-stream completion: {e}", exc_info=True
                )
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise

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
            return "\n".join([f"{m.role}: {m.content}" for m in messages])

        return await self.ask(
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
        return await self.ask(
            base_prompt=(
                "# System Prompt: Content Summarization\n\n"
                "You are a **summarization expert**. Summarize the content provided by "
                "the user. Preserve key facts, identifiers, lists, tables, and any "
                "structured data. If the content includes JSON or other structured "
                "data, keep the structure and retain all items; do not invent new "
                "data or claim data is missing. Use concise, plain language when "
                "summarizing narrative text.\n"
            ),
            base_request=base_request,
            question=content,
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

        request = ChatCompletionRequest(
            model=base_request.model,
            messages=messages_history,
            stream=False,
        )

        response = await self.non_stream_completion(request, access_token, outer_span)

        # Handle response from ChatCompletionResponse or raw dict
        if isinstance(response, ChatCompletionResponse):
            choices = response.choices or []
            if choices:
                message = choices[0].message
                if message and message.content is not None:
                    return message.content
                delta = choices[0].delta
                if delta and delta.content is not None:
                    return delta.content
            return ""
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        return ""
