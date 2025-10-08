"""
Message summarization service for managing conversation history length.
"""

import json
import logging
from typing import List, Optional
from opentelemetry import trace

from app.tgi.models import (
    ChatCompletionRequest,
    Message,
    MessageRole,
)
from app.tgi.clients.llm_client import LLMClient
from app.vars import TGI_MODEL_NAME

logger = logging.getLogger("uvicorn.error")
tracer = trace.get_tracer(__name__)


class MessageSummarizationService:
    """Service for summarizing conversation history to manage token limits."""

    def __init__(self, llm_client: LLMClient):
        self.logger = logger
        self.llm_client = llm_client

    async def summarize_messages(
        self,
        messages: List[Message],
        access_token: Optional[str] = None,
        parent_span=None,
    ) -> List[Message]:
        """
        Summarize messages while preserving:
        - System prompt (first message if it's a system message)
        - Last tool result (last message if it's a tool message)
        - Everything in between summarized into a single assistant message

        Args:
            messages: List of messages to potentially summarize
            access_token: Optional access token for LLM calls
            parent_span: Optional parent span for tracing

        Returns:
            Summarized list of messages
        """
        with tracer.start_as_current_span("summarize_messages") as span:
            span.set_attribute("messages.count", len(messages))

            # Need at least 4 messages to benefit from summarization:
            # system + user + assistant + tool (minimum conversation)
            if len(messages) < 4:
                self.logger.debug(
                    f"[MessageSummarizationService] Only {len(messages)} messages, skipping summarization"
                )
                span.set_attribute("summarization.skipped", True)
                span.set_attribute("summarization.reason", "too_few_messages")
                return messages

            # Identify system prompt (first message if system role)
            system_prompt = None
            start_idx = 0
            if messages[0].role == MessageRole.SYSTEM:
                system_prompt = messages[0]
                start_idx = 1
                self.logger.debug(
                    "[MessageSummarizationService] Found system prompt to preserve"
                )

            # Identify last tool result (last message if tool role)
            last_tool_result = None
            end_idx = len(messages)
            if messages[-1].role == MessageRole.TOOL:
                last_tool_result = messages[-1]
                end_idx = len(messages) - 1
                self.logger.debug(
                    "[MessageSummarizationService] Found last tool result to preserve"
                )

            # Check if there's anything to summarize
            messages_to_summarize = messages[start_idx:end_idx]
            if not messages_to_summarize:
                self.logger.debug(
                    "[MessageSummarizationService] No messages to summarize after preserving system and tool"
                )
                span.set_attribute("summarization.skipped", True)
                span.set_attribute("summarization.reason", "no_content_to_summarize")
                return messages

            span.set_attribute("messages.to_summarize", len(messages_to_summarize))

            # Build summarization prompt
            summarization_prompt = self._build_summarization_prompt(
                messages_to_summarize
            )

            # Call LLM to summarize
            try:
                summarization_request = ChatCompletionRequest(
                    messages=[
                        Message(
                            role=MessageRole.USER,
                            content=summarization_prompt,
                        )
                    ],
                    model=TGI_MODEL_NAME,
                    stream=False,
                    temperature=0.3,  # Lower temperature for more focused summarization
                    max_tokens=500,  # Limit summary length
                )

                response = await self.llm_client.non_stream_completion(
                    summarization_request, access_token, parent_span
                )

                if not response.choices or not response.choices[0].message:
                    self.logger.warning(
                        "[MessageSummarizationService] Failed to get summarization response"
                    )
                    span.set_attribute("summarization.success", False)
                    return messages

                summary_content = response.choices[0].message.content
                if not summary_content:
                    self.logger.warning(
                        "[MessageSummarizationService] Empty summary received"
                    )
                    span.set_attribute("summarization.success", False)
                    return messages

                self.logger.info(
                    f"[MessageSummarizationService] Successfully summarized {len(messages_to_summarize)} messages into summary of {len(summary_content)} characters"
                )
                span.set_attribute("summarization.success", True)
                span.set_attribute("summary.length", len(summary_content))

                # Build new message list
                new_messages = []

                # Add system prompt if present
                if system_prompt:
                    new_messages.append(system_prompt)

                # Add summary as an assistant message
                new_messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=f"[Conversation Summary]\n{summary_content}",
                    )
                )

                # Add last tool result if present
                if last_tool_result:
                    new_messages.append(last_tool_result)

                span.set_attribute("messages.result_count", len(new_messages))
                self.logger.debug(
                    f"[MessageSummarizationService] Reduced from {len(messages)} to {len(new_messages)} messages"
                )

                return new_messages

            except Exception as e:
                self.logger.error(
                    f"[MessageSummarizationService] Error during summarization: {str(e)}"
                )
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                # Return original messages on error
                return messages

    def _build_summarization_prompt(self, messages: List[Message]) -> str:
        """Build a prompt that asks the LLM to summarize the conversation."""
        # Format messages for summarization
        formatted_messages = []
        for msg in messages:
            role = msg.role.value
            content = msg.content or ""

            # Handle tool calls
            if msg.tool_calls:
                tool_info = []
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                        tool_info.append(
                            f"{tc.function.name}({json.dumps(args, ensure_ascii=False)})"
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"[MessageSummarizationService] Failed to parse tool arguments for summarization: {str(e)}"
                        )
                        tool_info.append(f"{tc.function.name}({tc.function.arguments})")
                content = f"{content}\nTool calls: {', '.join(tool_info)}"

            # Handle tool results
            if msg.role == MessageRole.TOOL:
                tool_name = msg.name or "unknown_tool"
                content = f"Tool '{tool_name}' result: {content}"

            formatted_messages.append(f"{role.upper()}: {content}")

        conversation_text = "\n\n".join(formatted_messages)

        prompt = f"""You are helping to summarize a conversation to save tokens. Please provide a concise summary of the following conversation that captures the key points, questions asked, actions taken, and important results.

Keep the summary factual and informative but brief. Focus on what was discussed and what was accomplished.

Conversation to summarize:
{conversation_text}

Please provide a concise summary:"""

        return prompt

    def should_summarize(self, messages: List[Message], threshold: int = 15) -> bool:
        """
        Determine if messages should be summarized based on count.

        Args:
            messages: List of messages to check
            threshold: Message count threshold for triggering summarization

        Returns:
            True if summarization should occur
        """
        count = len(messages)
        should = count >= threshold

        if should:
            self.logger.debug(
                f"[MessageSummarizationService] Message count ({count}) exceeds threshold ({threshold}), summarization recommended"
            )
        return should
