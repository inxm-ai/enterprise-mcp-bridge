"""Minimal baggage API stub used for unit testing."""

from typing import Any, Mapping


def get_all(context: Any = None) -> Mapping[str, object]:
    return {}


__all__ = ["get_all"]
