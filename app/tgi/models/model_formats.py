from __future__ import annotations

import logging
from abc import ABC
from typing import Any, Dict, Optional, Type

from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.services.tools.tool_resolution import (
    ToolCallFormat,
    ToolResolutionStrategy,
)

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


def get_model_format_for(
    model_name: Optional[str] = None, *, override: Optional[str] = None
) -> BaseModelFormat:
    """Resolve the model format for a given model name or configuration."""
    # Default to ChatGPTModelFormat as it is the only one left and standard for OpenAI
    return ChatGPTModelFormat()


def available_model_formats() -> Dict[str, Type[BaseModelFormat]]:
    """Return the registered model formats."""
    return dict(_FORMAT_REGISTRY)
