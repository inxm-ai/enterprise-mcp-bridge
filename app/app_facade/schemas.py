from typing import List, Optional

from pydantic import BaseModel


class UiCreateRequest(BaseModel):
    id: str
    name: str
    prompt: str
    tools: Optional[List[str]] = None


class UiUpdateRequest(BaseModel):
    prompt: str
    tools: Optional[List[str]] = None
