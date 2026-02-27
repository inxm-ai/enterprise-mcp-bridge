"""
Pure utility functions for navigating and mutating nested dictionaries
using dotted-path notation (e.g. ``"agents.planner.result"``).

All functions are stateless and side-effect-free (except the mutating
variants which modify the dict in-place).
"""

from typing import Any


def get_path_value(data: dict, path: str) -> Any:
    """Resolve a dotted path from a dict.  Returns ``None`` when any segment is missing."""
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def set_path_value(data: dict, path: str, value: Any) -> None:
    """Set a dotted path on a dict, creating nested dicts as needed."""
    parts = path.split(".")
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current.get(part), dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def path_exists(data: dict, path: str) -> bool:
    """Check if a dotted path exists in a dict."""
    current: Any = data
    parts = path.split(".")
    for idx, part in enumerate(parts):
        if not isinstance(current, dict) or part not in current:
            return False
        if idx == len(parts) - 1:
            return True
        current = current.get(part)
    return False


def has_value(data: dict, path: str) -> bool:
    """Check if a dotted path exists **and** has a non-empty value.

    Empty strings, lists, dicts and tuples are treated as missing.
    """
    parts = path.split(".")
    current: Any = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False
    if current is None:
        return False
    if isinstance(current, (list, dict, str, tuple)) and len(current) == 0:
        return False
    return True


def delete_path(data: dict, path: str) -> None:
    """Remove a dotted path from a nested dict if present."""
    parts = path.split(".")
    current = data
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


def set_nested_value(target: dict, path_parts: list[str], value: Any) -> None:
    """Set a value on *target* following the given path parts (not dotted string)."""
    current = target
    for part in path_parts[:-1]:
        if not isinstance(current.get(part), dict):
            current[part] = {}
        current = current[part]
    current[path_parts[-1]] = value
