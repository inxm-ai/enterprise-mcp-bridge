"""
LLM client module for handling direct communication with the LLM API.
"""

import json
import logging
import os
import time
import uuid
from typing import AsyncGenerator
import aiohttp
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

logger = logging.getLogger("uvicorn.error")
tracer = trace.get_tracer(__name__)


class LLMClient:
    """Client for communicating with the LLM API."""

    def __init__(self):
        self.logger = logger
        self.tgi_url = os.environ.get("TGI_URL", "")
        self.tgi_token = os.environ.get("TGI_TOKEN", "")

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
            # Convert request to dict for JSON serialization
            payload = request.model_dump(exclude_none=True)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.tgi_url}/chat/completions",
                    headers=self._get_headers(access_token),
                    json=payload,
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
                payload = request.model_dump(exclude_none=True)
                payload["stream"] = False

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.tgi_url}/chat/completions",
                        headers=self._get_headers(access_token),
                        json=payload,
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
        user_request: str,
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
            base_prompt="You are a summarization expert. Read the question from the user, and then summarize the reply by the assistant. The reply from the assistant is coming from a tool, and might contain content not relevant to the original question. For instance, emails contain html, if the user wants a summary you are only interested in the text. If the user wants to know which senders send html emails than the information is important",
            base_request=base_request,
            question=user_request,
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
