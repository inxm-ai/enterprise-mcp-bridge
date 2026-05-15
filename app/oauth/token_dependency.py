import logging
from typing import Optional

from fastapi import Header, Request

from app.vars import TOKEN_COOKIE_NAME, TOKEN_NAME, TOKEN_SOURCE

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
        if cookie_value:
            return cookie_value
        logger.debug(
            f"Cookie {TOKEN_COOKIE_NAME} not found; falling back to {TOKEN_NAME} header"
        )
    # Default to header (also fallback when TOKEN_SOURCE=cookie but cookie is absent)
    return header_token
