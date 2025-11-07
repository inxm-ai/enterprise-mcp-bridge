"""
Prompt service module for handling MCP prompt operations.
"""

import logging
from typing import List, Optional, Dict, Any, Iterable
from opentelemetry import trace
from fastapi import HTTPException

from app.tgi.models import Message, MessageRole
from app.session import MCPSessionBase

logger = logging.getLogger("uvicorn.error")
tracer = trace.get_tracer(__name__)


class PromptService:
    """Service for handling MCP prompt operations."""

    prompt_cache = {}

    def __init__(self):
        self.logger = logger
        self.prompt_cache = {}

    async def find_prompt_by_name_or_role(
        self, session: MCPSessionBase, prompt_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find a prompt by name or by role=system, or return the first available prompt.

        Args:
            session: The MCP session to use
            prompt_name: Optional specific prompt name to search for

        Returns:
            The found prompt data or None if no prompts available
        """
        with tracer.start_as_current_span("find_prompt") as span:
            span.set_attribute("prompt.requested_name", prompt_name or "")

            if prompt_name in self.prompt_cache:
                span.set_attribute("prompt.found_name", prompt_name)
                span.set_attribute("prompt.found", True)
                span.set_attribute("prompt.cache.found", True)
                self.logger.debug(f"[PromptService] Found cached prompt: {prompt_name}")
                return self.prompt_cache[prompt_name]

            span.set_attribute("prompt.cache.found", False)
            try:
                # Get all available prompts
                prompts_result = await session.list_prompts()
                prompts = self._extract_prompts(prompts_result)

                if not prompts:
                    self.logger.debug(
                        f"[PromptService] No prompts available from MCP server: {prompts_result}"
                    )
                    span.set_attribute("prompt.found", False)
                    return None

                self.logger.debug(
                    f"[PromptService] Found {len(prompts)} prompts available"
                )

                # If specific prompt name requested, search for it
                if prompt_name:
                    for prompt in prompts:
                        if prompt["name"] == prompt_name:
                            span.set_attribute("prompt.found_name", prompt["name"])
                            span.set_attribute("prompt.found", True)
                            self.logger.debug(
                                f"[PromptService] Found requested prompt: {prompt_name}"
                            )
                            self.prompt_cache[prompt_name] = prompt
                            return prompt

                    # Prompt not found
                    span.set_attribute("prompt.found", False)
                    self.logger.warning(
                        f"[PromptService] Requested prompt '{prompt_name}' not found"
                    )
                    return None

                # Look for 'system' prompt or one with role=system
                for prompt in prompts:
                    if prompt["name"] == "system":
                        span.set_attribute("prompt.found_name", prompt["name"])
                        span.set_attribute("prompt.found", True)
                        self.logger.debug("[PromptService] Found 'system' prompt")
                        self.prompt_cache[prompt["name"]] = prompt
                        return prompt

                # Look for any prompt with role information
                for prompt in prompts:
                    if prompt["description"]:
                        if (
                            "role=system" in prompt["description"].lower()
                            or "system" in prompt["description"].lower()
                        ):
                            span.set_attribute("prompt.found_name", prompt["name"])
                            span.set_attribute("prompt.found", True)
                            self.logger.debug(
                                f"[PromptService] Found system-role prompt: {prompt['name']}"
                            )
                            self.prompt_cache[prompt["name"]] = prompt
                            return prompt

                # Prefer prompts that don't require arguments as fallback
                def _requires_args(p) -> bool:
                    try:
                        args = p.get("arguments", None)
                        if args is None:
                            return False
                        # If it's a list and non-empty, assume it requires args
                        if isinstance(args, (list, tuple)):
                            return len(args) > 0
                        # If it's a dict-like, check for required keys
                        if isinstance(args, dict):
                            required = args.get("required")
                            return bool(required)
                        # Fallback: any truthy value means it likely needs args
                        return bool(args)
                    except Exception:
                        return True

                no_arg_prompts = [p for p in prompts if not _requires_args(p)]
                if no_arg_prompts:
                    chosen = no_arg_prompts[0]
                    span.set_attribute(
                        "prompt.found_name", chosen.get("name", "unknown")
                    )
                    span.set_attribute("prompt.found", True)
                    self.logger.debug(
                        f"[PromptService] Using first available no-arg prompt: {chosen.get('name', 'unknown')}"
                    )
                    self.prompt_cache[chosen.get("name", "unknown")] = chosen
                    return chosen

                # As a last resort, return the first prompt
                if prompts:
                    first_prompt = prompts[0]
                    span.set_attribute("prompt.found_name", first_prompt["name"])
                    span.set_attribute("prompt.found", True)
                    self.logger.debug(
                        f"[PromptService] Using first available prompt (may require args): {first_prompt['name']}"
                    )
                    self.prompt_cache[first_prompt["name"]] = first_prompt
                    return first_prompt

                span.set_attribute("prompt.found", False)
                return None

            except Exception as e:
                self.logger.error(f"[PromptService] Error finding prompt: {str(e)}")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise

    def _extract_prompts(self, prompts_result: Any) -> List[Dict[str, Any]]:
        """Normalize various list_prompts return types into a list of dictionaries."""

        if prompts_result is None:
            return []

        if isinstance(prompts_result, dict):
            raw_prompts = prompts_result.get("prompts", [])
        elif hasattr(prompts_result, "prompts"):
            raw_prompts = getattr(prompts_result, "prompts")
        else:
            raw_prompts = prompts_result

        if raw_prompts is None:
            return []

        if isinstance(raw_prompts, dict):
            raw_prompts = raw_prompts.values()

        if not isinstance(raw_prompts, Iterable) or isinstance(
            raw_prompts, (str, bytes)
        ):
            raw_prompts = [raw_prompts]

        normalized: List[Dict[str, Any]] = []
        for prompt in raw_prompts:
            normalized_prompt = self._normalize_prompt_entry(prompt)
            if normalized_prompt:
                normalized.append(normalized_prompt)
        return normalized

    def _normalize_prompt_entry(self, prompt: Any) -> Optional[Dict[str, Any]]:
        if prompt is None:
            return None

        if isinstance(prompt, dict):
            return prompt

        if hasattr(prompt, "to_dict") and callable(getattr(prompt, "to_dict")):
            try:
                return prompt.to_dict()
            except Exception:
                pass

        if hasattr(prompt, "model_dump") and callable(getattr(prompt, "model_dump")):
            try:
                return prompt.model_dump()
            except Exception:
                pass

        if hasattr(prompt, "dict") and callable(getattr(prompt, "dict")):
            try:
                return prompt.dict()
            except Exception:
                pass

        if hasattr(prompt, "__dataclass_fields__"):
            try:
                return {
                    field: getattr(prompt, field, None)
                    for field in prompt.__dataclass_fields__
                }
            except Exception:
                pass

        if hasattr(prompt, "__dict__"):
            data = {
                key: value
                for key, value in vars(prompt).items()
                if not key.startswith("_")
            }
            if data:
                return data

        # Fallback: attempt to build dict from known attributes
        known_keys = [
            "name",
            "title",
            "description",
            "arguments",
            "template",
            "metadata",
        ]
        constructed = {
            key: getattr(prompt, key) for key in known_keys if hasattr(prompt, key)
        }
        return constructed or None

    async def get_prompt_content(
        self, session: MCPSessionBase, prompt: Dict[str, Any]
    ) -> str:
        """
        Get the actual content of a prompt by calling it.

        Args:
            session: The MCP session to use
            prompt: The prompt object to execute

        Returns:
            The prompt content as a string
        """
        with tracer.start_as_current_span("get_prompt_content") as span:
            span.set_attribute("prompt.name", prompt["name"])

            try:
                if "template" in prompt and prompt["template"]:
                    content = prompt["template"]["content"]
                    self.logger.debug(
                        f"[PromptService] Retrieved prompt content: {len(content)} characters"
                    )
                    span.set_attribute("prompt.content_length", len(content))
                    return content

                # Call the prompt to get its content
                result = await session.call_prompt(prompt["name"], {})

                if result.isError:
                    error_msg = f"Error getting prompt content: {result}"
                    self.logger.error(f"[PromptService] {error_msg}")
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", error_msg)
                    raise HTTPException(status_code=500, detail=error_msg)

                # Extract content from the result
                content = ""
                if hasattr(result, "messages") and result.messages:
                    for message in result.messages:
                        if hasattr(message, "content") and hasattr(
                            message.content, "text"
                        ):
                            content += message.content.text + "\n"
                        elif hasattr(message, "text"):
                            content += message.text + "\n"

                content = content.strip()
                self.logger.debug(
                    f"[PromptService] Retrieved prompt content: {len(content)} characters"
                )
                span.set_attribute("prompt.content_length", len(content))
                return content

            except HTTPException as he:
                # Gracefully ignore prompts that require arguments we don't have
                detail = getattr(he, "detail", "")
                if isinstance(detail, str) and "Missing required arguments" in detail:
                    self.logger.warning(
                        "[PromptService] Prompt requires arguments; skipping content retrieval"
                    )
                    return ""
                raise
            except Exception as e:
                msg = str(e)
                if "Missing required arguments" in msg:
                    self.logger.warning(
                        "[PromptService] Prompt requires arguments; skipping content retrieval"
                    )
                    return ""
                error_msg = f"Error retrieving prompt content: {msg}"
                self.logger.error(f"[PromptService] {error_msg}")
                span.set_attribute("error", True)
                span.set_attribute("error.message", msg)
                raise HTTPException(status_code=500, detail=error_msg)

    async def prepare_messages(
        self,
        session: MCPSessionBase,
        messages: List[Message],
        prompt_name: Optional[str] = None,
        parent_span=None,
    ) -> List[Message]:
        """
        Prepare messages by adding system prompt if needed.

        Args:
            session: The MCP session to use
            messages: Original messages from the request
            prompt_name: Optional specific prompt name to use
            parent_span: Optional parent span for tracing

        Returns:
            Messages with system prompt prepended if found
        """
        with tracer.start_as_current_span("prepare_messages") as span:
            span.set_attribute("messages.count", len(messages))
            span.set_attribute("prompt_name", prompt_name or "none")

            try:
                # Find appropriate prompt
                prompt = await self.find_prompt_by_name_or_role(session, prompt_name)

                prepared_messages = []

                # Add system prompt if found and no system message exists
                has_system_message = any(
                    msg.role == MessageRole.SYSTEM for msg in messages
                )

                span.set_attribute("has_system_message", has_system_message)

                if prompt and not has_system_message:
                    prompt_content = await self.get_prompt_content(session, prompt)
                    if prompt_content:
                        system_message = Message(
                            role=MessageRole.SYSTEM,
                            content=prompt_content.replace("\\n", "\n"),
                        )
                        prepared_messages.append(system_message)
                        self.logger.debug(
                            f"[PromptService] Added system prompt: {prompt['name']}"
                        )

                # Add original messages
                prepared_messages.extend(messages)

                self.logger.debug(
                    f"[PromptService] Prepared {len(prepared_messages)} messages"
                )

                return prepared_messages

            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                self.logger.error(f"[PromptService] Error preparing messages: {str(e)}")
                # Return original messages on error
                return messages
