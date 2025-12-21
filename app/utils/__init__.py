import hashlib
from typing import Optional


def mask_token(text: str, token: str) -> str:
    return text.replace(token, f"{token[:4]}****") if token else text


def token_fingerprint(token: Optional[str]) -> str:
    """Provide a stable, low-leak token identifier for logs."""
    if not token:
        return "<empty>"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
    token_len = len(token)
    head = token[:6]
    tail = token[-6:] if token_len > 6 else token
    return f"len={token_len} sha256={digest} head={head} tail={tail}"
