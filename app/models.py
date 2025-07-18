from pydantic import BaseModel
from typing import Optional, Dict

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
            structuredContent=resultEntry.structuredContent if hasattr(resultEntry, 'structuredContent') else None
        )

class RunToolsResult(BaseModel):
    isError: bool
    content: list[RunToolResultContent]
    structuredContent: Optional[Dict]
    def __init__(self, result):
        print(f"[RunToolsResult] Initializing with result: {result}")
        super().__init__(
            isError=result.isError, 
            content=[RunToolResultContent(item) for item in result.content],
            structuredContent=result.structuredContent if hasattr(result, 'structuredContent') else None
        )
