from fastapi import Header, Request
from typing import Optional
import logging
from app.vars import TOKEN_NAME, TOKEN_COOKIE_NAME, TOKEN_SOURCE

logger = logging.getLogger("uvicorn.error")

async def get_access_token(
    request: Request,
    header_token: Optional[str] = Header(None, alias=TOKEN_NAME),
) -> Optional[str]:
    """
    Dependency to retrieve the access token based on the configured TOKEN_SOURCE.
    """
    if TOKEN_SOURCE == "cookie":
        cookie_value = request.cookies.get(TOKEN_COOKIE_NAME)
        if not cookie_value:
            logger.warning(f"Cookie {TOKEN_COOKIE_NAME} not found in request cookies: {request.cookies.keys()}")
        return cookie_value
    # Default to header
    return header_token
