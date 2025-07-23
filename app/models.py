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

def error_finder(result):
    if hasattr(result, 'isError') and result.isError:
        return result.isError
    if hasattr(result, 'error') and result.error:
        return True
    if hasattr(result, 'content') and isinstance(result.content, list) and any(getattr(item, 'isError', False) for item in result.content):
        return True
    return False

def content_resolver(result, isError):
    if isError and hasattr(result, 'error'):
        return [RunToolResultContent({'text': result.error})]
    if hasattr(result, 'content') and isinstance(result.content, list):
        return [RunToolResultContent(item) for item in result.content]
    if hasattr(result, 'text'):
        return [RunToolResultContent({'text': result.text})]
    if isError:
        return [RunToolResultContent({'text': 'An error occurred'})]
    return []

class RunToolsResult(BaseModel):
    isError: bool
    content: list[RunToolResultContent]
    structuredContent: Optional[Dict]
    def __init__(self, result):
        print(f"[RunToolsResult] Initializing with result: {result}")
        isError = error_finder(result)
        content = content_resolver(result, isError)
        super().__init__(
            isError=isError,
            content=content,
            structuredContent=result.structuredContent if hasattr(result, 'structuredContent') else None
        )
