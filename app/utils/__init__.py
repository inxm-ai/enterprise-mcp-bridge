import hashlib
from typing import Optional


def mask_token(text: str, token: str) -> str:
    return text.replace(token, "[REDACTED]") if token else text


def token_fingerprint(token: Optional[str]) -> str:
    """Provide a stable, low-leak token identifier for logs."""
    if not token:
        return "<empty>"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
    return f"len={len(token)} sha256={digest}"
