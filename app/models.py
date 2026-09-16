from pydantic import BaseModel
from typing import Any, Optional, Dict
from mcp.server.mcpserver.prompts import base

from app.utils.mcp_fields import MISSING, is_error, read, structured_content


class MCPRequest(BaseModel):
    action: str


class RunToolRequest(BaseModel):
    tool_name: str
    args: Optional[Dict] = None


class RunToolResultContent(BaseModel):
    text: str
    # Any JSON value: protocol 2026-07-28 no longer limits structured content to objects.
    structuredContent: Optional[Any]

    def __init__(self, resultEntry):
        super().__init__(
            text=str(read(resultEntry, "text", "text", "")),
            structuredContent=structured_content(resultEntry),
        )


def error_finder(result):
    if is_error(result):
        return True
    if hasattr(result, "error") and result.error:
        return True
    if (
        hasattr(result, "content")
        and isinstance(result.content, list)
        and any(is_error(item) for item in result.content)
    ):
        return True
    return False


def content_resolver(result, isError):
    """The content blocks of a result, whether it is an SDK model, a bridge model or a dict."""
    error = read(result, "error", "error")
    if isError and error:
        return [RunToolResultContent({"text": error})]
    content = read(result, "content", "content")
    if isinstance(content, list):
        return [RunToolResultContent(item) for item in content]
    text = read(result, "text", "text", MISSING)
    if text is not MISSING:
        return [RunToolResultContent({"text": text})]
    if isError:
        return [RunToolResultContent({"text": "An error occurred"})]
    return []


class MessageContent(BaseModel):
    type: str
    text: str


class Message(BaseModel):
    role: str
    content: MessageContent


class RunPromptResult(BaseModel):
    isError: bool
    meta: Optional[str]
    description: Optional[str]
    messages: list[Message]

    def map_message(self, message: base.Message) -> Message:
        content = MessageContent(
            type=message.content.type if hasattr(message.content, "type") else "text",
            text=(
                message.content.text
                if hasattr(message.content, "text")
                else str(message.content)
            ),
        )
        return Message(
            role=message.role if hasattr(message, "role") else "unknown",
            content=content,
        )

    def __init__(self, result):
        try:
            super().__init__(
                isError=False,
                meta=result.meta if hasattr(result, "meta") else None,
                description=(
                    result.description if hasattr(result, "description") else None
                ),
                messages=(
                    map(self.map_message, result.messages)
                    if hasattr(result, "messages")
                    else []
                ),
            )
        except Exception as e:
            super().__init__(
                isError=True,
                meta=None,
                description=f"Error processing prompt result: {e}",
                messages=[],
            )


class RunToolsResult(BaseModel):
    isError: bool
    content: list[RunToolResultContent]
    structuredContent: Optional[Any]

    def __init__(self, result):
        isError = error_finder(result)
        content = content_resolver(result, isError)
        super().__init__(
            isError=isError,
            content=content,
            structuredContent=structured_content(result),
        )
