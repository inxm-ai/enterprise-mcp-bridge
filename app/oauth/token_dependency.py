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
        if header_token:
            logger.debug(
                f"Cookie {TOKEN_COOKIE_NAME} not found; falling back to {TOKEN_NAME} header"
            )
        else:
            logger.warning(
                f"Cookie {TOKEN_COOKIE_NAME} not found and {TOKEN_NAME} header is also missing"
            )
    # Default to header (also fallback when TOKEN_SOURCE=cookie but cookie is absent)
    if header_token:
        return header_token
    # Accept standard Authorization: Bearer <token> as a fallback (desktop clients
    # bypass oauth2-proxy and send this instead of X-Auth-Request-Access-Token).
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:]
    return None
