"""Incoming headers that carry the caller's credentials.

Used to keep the caller's token away from MCP servers configured with
``forward_access_token: false``, even when header forwarding or
header-to-input mapping would otherwise pass these headers on.
"""

from typing import Optional

from app.vars import TOKEN_NAME

_CREDENTIAL_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "x-auth-request-access-token",
        "x-forwarded-access-token",
        "x-amzn-oidc-accesstoken",
        "x-amzn-oidc-data",
    }
)


def is_caller_credential_header(
    name: str, value: Optional[str] = None, access_token: Optional[str] = None
) -> bool:
    """Whether a header carries (or may carry) the caller's credentials.

    Matches the known credential headers, the configured TOKEN_NAME, and any
    header whose value contains the caller's access token.
    """
    lowered = (name or "").lower()
    if lowered in _CREDENTIAL_HEADERS or lowered == TOKEN_NAME.lower():
        return True
    return bool(access_token and value and access_token in value)
