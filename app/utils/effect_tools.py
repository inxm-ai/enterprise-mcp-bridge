"""Decide whether a tool has side effects (and must be simulated in a dry run).

Effect tools are configured as glob patterns (``EFFECT_TOOLS`` or a server's
``effect_tools``). The special entry ``"auto"`` additionally classifies every
tool automatically. Auto classification is deliberately conservative: a tool
counts as read-only only when its MCP annotations say so, or — without a
usable annotation — when its name starts with a well-known read verb. A write
tool misclassified as read-only would run for real during a dry run, so
anything unknown is an effect.
"""

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from typing import Any, Iterable, Optional

from app.utils.mcp_fields import read

AUTO_EFFECT_TOOLS = "auto"

READ_ONLY_VERBS = (
    "get",
    "list",
    "search",
    "read",
    "fetch",
    "find",
    "describe",
    "query",
    "lookup",
    "show",
    "view",
    "count",
    "check",
    "validate",
    "preview",
    "explain",
    "summarize",
    "whoami",
)

_NAMESPACE_SEPARATOR_RE = re.compile(r"[./]")


def _is_auto_entry(pattern: str) -> bool:
    return pattern.strip().lower() == AUTO_EFFECT_TOOLS


def auto_effect_enabled(patterns: Optional[Iterable[str]]) -> bool:
    """Whether the effect-tool list asks for automatic classification."""
    return any(_is_auto_entry(pattern) for pattern in patterns or ())


def _annotations(tool_def: Any) -> Any:
    if tool_def is None:
        return None
    annotations = read(tool_def, "annotations", "annotations")
    if annotations is None and isinstance(tool_def, dict):
        # OpenAI-style tool specs nest the MCP fields under "function".
        function = tool_def.get("function")
        if isinstance(function, dict):
            annotations = function.get("annotations")
    return annotations


def _annotation_verdict(tool_def: Any) -> Optional[bool]:
    """True/False when annotations decide the effect, None when they don't."""
    annotations = _annotations(tool_def)
    if annotations is None:
        return None
    # Identity checks on purpose: only real booleans count, so a missing hint
    # (None) or a test double's invented attribute is not a verdict.
    read_only = read(annotations, "read_only_hint", "readOnlyHint")
    if read_only is True:
        return False
    destructive = read(annotations, "destructive_hint", "destructiveHint")
    if read_only is False or destructive is True:
        return True
    return None


def _strip_namespace(tool_name: str) -> str:
    return _NAMESPACE_SEPARATOR_RE.split(tool_name)[-1]


def _starts_with_read_verb(name: str) -> bool:
    lowered = name.lower()
    for verb in READ_ONLY_VERBS:
        if not lowered.startswith(verb):
            continue
        rest = name[len(verb) :]
        if not rest or rest[0] in "_-":
            return True
        # camelCase boundary: "getUser", but neither "Getaway" nor "GETTER".
        if rest[0].isupper() and name[len(verb) - 1].islower():
            return True
    return False


def classify_effect(tool_name: str, tool_def: Any = None) -> bool:
    """Automatic classification: True unless the tool is confidently read-only."""
    verdict = _annotation_verdict(tool_def)
    if verdict is not None:
        return verdict
    if not isinstance(tool_name, str) or not tool_name:
        return True
    return not _starts_with_read_verb(_strip_namespace(tool_name))


def is_effect_tool(
    tool_name: str, tool_def: Any, patterns: Optional[Iterable[str]]
) -> bool:
    """Whether a tool is an effect tool under the configured patterns.

    Explicit globs and ``"auto"`` combine: a tool is an effect when it matches
    an explicit glob or when auto classification is enabled and says so.
    """
    patterns = list(patterns or ())
    globs = [pattern for pattern in patterns if not _is_auto_entry(pattern)]
    if (
        isinstance(tool_name, str)
        and tool_name
        and any(fnmatchcase(tool_name, pattern) for pattern in globs)
    ):
        return True
    if auto_effect_enabled(patterns):
        return classify_effect(tool_name, tool_def)
    return False
