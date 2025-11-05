from typing import List, Optional

from pydantic import BaseModel


class DashboardCreateRequest(BaseModel):
    id: str
    name: str
    prompt: str
    tools: Optional[List[str]] = None


class DashboardUpdateRequest(BaseModel):
    prompt: str
    tools: Optional[List[str]] = None
