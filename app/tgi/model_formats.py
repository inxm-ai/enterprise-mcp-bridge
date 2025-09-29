from __future__ import annotations

import json
import logging
from abc import ABC
from typing import Any, Dict, Optional, Type

from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.tool_resolution import ToolCallFormat, ToolResolutionStrategy
from app.vars import TGI_MODEL_FORMAT, TOOL_INJECTION_MODE

logger = logging.getLogger("uvicorn.error")


class BaseModelFormat(ABC):
    """Interface describing model-specific behavior for chat completions."""

    name: str

    def prepare_request(self, request: ChatCompletionRequest) -> None:
        """Adjust a chat completion request prior to serialization."""

    def build_tool_message(
        self,
        tool_call_format: ToolCallFormat,
        tool_result: Dict[str, Any],
        content: str,
    ) -> Message:
        """Return the message that should be appended after a tool executes."""
        return Message(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_result.get("tool_call_id"),
            name=tool_result.get("name"),
        )

    def create_tool_resolution_strategy(self) -> ToolResolutionStrategy:
        """Provide the tool resolution strategy to use for this model."""
        return ToolResolutionStrategy()


_FORMAT_REGISTRY: Dict[str, Type[BaseModelFormat]] = {}


def register_model_format(cls: Type[BaseModelFormat]) -> Type[BaseModelFormat]:
    if not getattr(cls, "name", None):
        raise ValueError("Model format must define a name")
    key = cls.name.lower()
    if key in _FORMAT_REGISTRY:
        raise ValueError(f"Model format '{cls.name}' already registered")
    _FORMAT_REGISTRY[key] = cls
    return cls


@register_model_format
class ChatGPTModelFormat(BaseModelFormat):
    name = "chat-gpt/v1"

    def prepare_request(self, request: ChatCompletionRequest) -> None:  # noqa: D401
        if not getattr(request, "tool_choice", None):
            return

        # map simple string values into the object format
        if isinstance(request.tool_choice, str):
            tc = request.tool_choice.strip()
            if tc.lower() in ("auto", "none", "required"):
                # /lowercased for consistency
                request.tool_choice = tc.lower()
                return

            request.tool_choice = {
                "type": "function",
                "function": {"name": tc},
            }
            return


@register_model_format
class ClaudeModelFormat(BaseModelFormat):
    name = "claude/v1"

    def prepare_request(self, request: ChatCompletionRequest) -> None:
        if not request.tools:
            return

        tools = [self._format_tool(tool) for tool in request.tools if tool.function]
        if not tools:
            return

        injection = (
            "You have access to the following tools:\n"
            + "\n".join(tools)
            + "\nEnd each of your tool calls, and only tool calls, with <stop/>."
        )

        messages = list(request.messages or [])
        system_index: Optional[int] = None
        for idx, message in enumerate(messages):
            if message.role == MessageRole.SYSTEM:
                system_index = idx
                break

        if system_index is None:
            system_message = Message(role=MessageRole.SYSTEM, content=injection)
            messages.insert(0, system_message)
        else:
            system_message = messages[system_index]
            existing = system_message.content or ""
            if injection not in existing:
                separator = "\n\n" if existing else ""
                system_message.content = f"{existing}{separator}{injection}"
            messages[system_index] = system_message

        request.messages = messages
        request.tools = []
        stops = list(request.stop or [])
        if "<stop/>" not in stops:
            stops.append("<stop/>")
        request.stop = stops

    def build_tool_message(
        self,
        tool_call_format: ToolCallFormat,
        tool_result: Dict[str, Any],
        content: str,
    ) -> Message:
        if tool_call_format == ToolCallFormat.CLAUDE_XML:
            name = tool_result.get("name")
            xml_tag = f"<{name}_result>{content}</{name}_result><stop/>"
            return Message(
                role=MessageRole.ASSISTANT,
                content=xml_tag,
                tool_call_id=tool_result.get("tool_call_id"),
                name=name,
            )
        return super().build_tool_message(tool_call_format, tool_result, content)

    @staticmethod
    def _format_tool(tool) -> str:
        parameters = tool.function.parameters or {}
        return f"<{tool.function.name}>{json.dumps(parameters, ensure_ascii=False, separators=(',', ':'))}</{tool.function.name}>"


def _guess_by_model_name(model_name: Optional[str]) -> Optional[str]:
    if not model_name:
        return None
    lowered = model_name.lower()
    if "claude" in lowered:
        return ClaudeModelFormat.name
    if "gpt" in lowered or "chat" in lowered:
        return ChatGPTModelFormat.name
    return None


def get_model_format_for(
    model_name: Optional[str] = None, *, override: Optional[str] = None
) -> BaseModelFormat:
    """Resolve the model format for a given model name or configuration."""
    if override:
        key = override.lower()
    else:
        guessed = _guess_by_model_name(model_name)
        if guessed:
            key = guessed.lower()
        else:
            env_value = (TGI_MODEL_FORMAT or "").strip()
            if env_value:
                key = env_value.lower()
            else:
                legacy_mode = (TOOL_INJECTION_MODE or "openai").lower()
                key = (
                    ClaudeModelFormat.name
                    if legacy_mode == "claude"
                    else ChatGPTModelFormat.name
                )
                key = key.lower()

    fmt_cls = _FORMAT_REGISTRY.get(key)
    if not fmt_cls:
        logger.warning("Unknown TGI_MODEL_FORMAT '%s'; defaulting to chat-gpt/v1", key)
        fmt_cls = _FORMAT_REGISTRY[ChatGPTModelFormat.name.lower()]
    return fmt_cls()


def available_model_formats() -> Dict[str, Type[BaseModelFormat]]:
    """Return the registered model formats."""
    return dict(_FORMAT_REGISTRY)
