from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class UiCreateRequest(BaseModel):
    id: str
    name: str
    prompt: str
    tools: Optional[List[str]] = None


class UiUpdateRequest(BaseModel):
    prompt: str
    tools: Optional[List[str]] = None


class UiChatSessionCreateRequest(BaseModel):
    tools: Optional[List[str]] = None


class UiChatMessageRequest(BaseModel):
    message: str
    tools: Optional[List[str]] = None
    tool_choice: Optional[Any] = None


class UiChatSessionResponse(BaseModel):
    session_id: str
    base_version: int
    draft_version: int
    expires_at: str


class UiPublishResponse(BaseModel):
    status: str
    version: int
    published_at: str
    record: Dict[str, Any]
