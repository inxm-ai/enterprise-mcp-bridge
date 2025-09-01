from pydantic import BaseModel
from typing import Optional, Dict
from mcp.server.fastmcp.prompts import base


class MCPRequest(BaseModel):
    action: str


class RunToolRequest(BaseModel):
    tool_name: str
    args: Optional[Dict] = None


class RunToolResultContent(BaseModel):
    text: str
    structuredContent: Optional[Dict]

    def __init__(self, resultEntry):
        print(f"[RunToolResultContent] Initializing with resultEntry: {resultEntry}")
        super().__init__(
            text=resultEntry.text,
            structuredContent=(
                resultEntry.structuredContent
                if hasattr(resultEntry, "structuredContent")
                else None
            ),
        )


def error_finder(result):
    if hasattr(result, "isError") and result.isError:
        return result.isError
    if hasattr(result, "error") and result.error:
        return True
    if (
        hasattr(result, "content")
        and isinstance(result.content, list)
        and any(getattr(item, "isError", False) for item in result.content)
    ):
        return True
    return False


def content_resolver(result, isError):
    if isError and hasattr(result, "error"):
        return [RunToolResultContent({"text": result.error})]
    if hasattr(result, "content") and isinstance(result.content, list):
        return [RunToolResultContent(item) for item in result.content]
    if hasattr(result, "text"):
        return [RunToolResultContent({"text": result.text})]
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
        print(f"[RunPromptResult] Initializing with result: {result}")
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
            print(f"[RunPromptResult] Error during initialization: {e}")
            super().__init__(
                isError=True,
                meta=None,
                description=f"Error processing prompt result: {e}",
                messages=[],
            )


class RunToolsResult(BaseModel):
    isError: bool
    content: list[RunToolResultContent]
    structuredContent: Optional[Dict]

    def __init__(self, result):
        isError = error_finder(result)
        content = content_resolver(result, isError)
        super().__init__(
            isError=isError,
            content=content,
            structuredContent=(
                result.structuredContent
                if hasattr(result, "structuredContent")
                else None
            ),
        )
