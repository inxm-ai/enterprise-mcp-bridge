"""Minimal propagate API stub used for unit testing."""

from typing import Any, MutableMapping


def inject(carrier: MutableMapping[str, str], context: Any = None) -> None:
    """No-op: the stub has no active span context to propagate."""
    return None


__all__ = ["inject"]
