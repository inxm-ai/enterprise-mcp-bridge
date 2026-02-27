"""Pure / near-pure helper functions for prompt building, runtime-context
sanitisation, chat-history trimming, payload capping, and lightweight
JSON parsing.

All public symbols are plain functions — no class is needed since none of
them carry state beyond the constants defined in this module.
"""

import contextlib
import copy
import json
import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from fastapi import HTTPException

from app.tgi.models import Message, MessageRole

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Constants (previously inline in GeneratedUIService)
# ---------------------------------------------------------------------------
MAX_RUNTIME_CONTEXT_ENTRIES = 20
MAX_RUNTIME_CONTEXT_DEPTH = 3
MAX_RUNTIME_CONTEXT_TEXT = 2000
MAX_RUNTIME_CONSOLE_EVENTS = 20

SCRIPT_KEYS = ("service_script", "components_script", "test_script", "dummy_data")


# ---------------------------------------------------------------------------
# Runtime-context sanitisation
# ---------------------------------------------------------------------------


def trim_runtime_text(value: Any, limit: int = MAX_RUNTIME_CONTEXT_TEXT) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated]"


def sanitize_runtime_value(value: Any, depth: int = 0) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return trim_runtime_text(value)
    if depth >= MAX_RUNTIME_CONTEXT_DEPTH:
        with contextlib.suppress(Exception):
            return trim_runtime_text(json.dumps(value, ensure_ascii=False, default=str))
        return trim_runtime_text(value)
    if isinstance(value, list):
        return [
            sanitize_runtime_value(item, depth + 1)
            for item in value[:MAX_RUNTIME_CONTEXT_ENTRIES]
        ]
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= 40:
                break
            cleaned[str(key)] = sanitize_runtime_value(item, depth + 1)
        return cleaned
    return trim_runtime_text(value)


def sanitize_runtime_action(
    draft_action: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(draft_action, dict):
        return None
    action_type = trim_runtime_text(draft_action.get("type"))
    if action_type != "runtime_service_exchanges":
        return None

    raw_entries = draft_action.get("entries")
    if not isinstance(raw_entries, list):
        raw_entries = []

    entries: List[Dict[str, Any]] = []
    for raw in raw_entries[:MAX_RUNTIME_CONTEXT_ENTRIES]:
        if not isinstance(raw, dict):
            continue
        tool = trim_runtime_text(raw.get("tool"), limit=160)
        error = trim_runtime_text(raw.get("error"))
        request_body = sanitize_runtime_value(raw.get("request_body"))
        request_options = sanitize_runtime_value(raw.get("request_options"))
        response_payload = sanitize_runtime_value(raw.get("response_payload"))
        cursor_raw = raw.get("cursor")
        timestamp_raw = raw.get("timestamp")

        cursor: Optional[int]
        try:
            parsed_cursor = int(cursor_raw)
            cursor = parsed_cursor if parsed_cursor >= 0 else None
        except (TypeError, ValueError):
            cursor = None

        timestamp: Optional[Union[int, str]] = None
        if isinstance(timestamp_raw, (int, float)):
            timestamp = int(timestamp_raw)
        elif timestamp_raw is not None:
            timestamp = trim_runtime_text(timestamp_raw, limit=80)

        if not tool and not error and response_payload is None:
            continue

        entry: Dict[str, Any] = {
            "tool": tool,
            "request_body": request_body,
            "request_options": request_options,
            "response_payload": response_payload,
            "error": error,
            "mocked": bool(raw.get("mocked")),
        }
        if cursor is not None:
            entry["cursor"] = cursor
        if timestamp is not None:
            entry["timestamp"] = timestamp
        entries.append(entry)

    raw_console_events = draft_action.get("console_events")
    if not isinstance(raw_console_events, list):
        raw_console_events = []
    console_events: List[Dict[str, Any]] = []
    for raw_event in raw_console_events[:MAX_RUNTIME_CONSOLE_EVENTS]:
        if not isinstance(raw_event, dict):
            continue
        kind = trim_runtime_text(raw_event.get("kind"), limit=80)
        message = trim_runtime_text(raw_event.get("message"))
        stack = trim_runtime_text(raw_event.get("stack"))
        filename = trim_runtime_text(raw_event.get("filename"), limit=200)
        line_value = raw_event.get("line")
        column_value = raw_event.get("column")
        timestamp_value = raw_event.get("timestamp")
        line: Optional[int] = None
        column: Optional[int] = None
        timestamp2: Optional[int] = None
        with contextlib.suppress(TypeError, ValueError):
            parsed = int(line_value)
            line = parsed if parsed >= 0 else None
        with contextlib.suppress(TypeError, ValueError):
            parsed = int(column_value)
            column = parsed if parsed >= 0 else None
        with contextlib.suppress(TypeError, ValueError):
            parsed = int(timestamp_value)
            timestamp2 = parsed if parsed >= 0 else None
        if not kind and not message and not stack:
            continue
        event: Dict[str, Any] = {
            "kind": kind,
            "message": message,
            "stack": stack,
            "filename": filename,
        }
        if line is not None:
            event["line"] = line
        if column is not None:
            event["column"] = column
        if timestamp2 is not None:
            event["timestamp"] = timestamp2
        console_events.append(event)

    if not entries and not console_events:
        return None

    cursor_raw = draft_action.get("cursor")
    cursor = 0
    with contextlib.suppress(TypeError, ValueError):
        cursor = max(0, int(cursor_raw))

    captured_at = trim_runtime_text(draft_action.get("captured_at"), limit=80)
    return {
        "type": action_type,
        "cursor": cursor,
        "captured_at": captured_at,
        "entries": entries,
        "console_events": console_events,
    }


# ---------------------------------------------------------------------------
# Payload-size helpers
# ---------------------------------------------------------------------------


def payload_bytes(value: Any) -> int:
    with contextlib.suppress(Exception):
        return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
    return len(str(value).encode("utf-8"))


# ---------------------------------------------------------------------------
# Runtime-context → prompt enrichment
# ---------------------------------------------------------------------------


def runtime_context_for_prompt(
    runtime_context: Optional[Dict[str, Any]],
    *,
    limit: int = 8,
    max_console_events: int = MAX_RUNTIME_CONSOLE_EVENTS,
    max_bytes: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(runtime_context, dict):
        return None
    entries = runtime_context.get("entries")
    console_events = runtime_context.get("console_events")

    prompt_entries: List[Dict[str, Any]] = []
    if isinstance(entries, list):
        for entry in entries[-limit:]:
            if not isinstance(entry, dict):
                continue
            prompt_entries.append(
                {
                    "tool": entry.get("tool"),
                    "request_body": entry.get("request_body"),
                    "request_options": entry.get("request_options"),
                    "response_payload": entry.get("response_payload"),
                    "error": entry.get("error"),
                    "mocked": bool(entry.get("mocked")),
                }
            )

    prompt_console_events: List[Dict[str, Any]] = []
    if isinstance(console_events, list):
        for event in console_events[-max_console_events:]:
            if not isinstance(event, dict):
                continue
            prompt_console_events.append(
                {
                    "kind": event.get("kind"),
                    "message": event.get("message"),
                    "stack": event.get("stack"),
                    "filename": event.get("filename"),
                    "line": event.get("line"),
                    "column": event.get("column"),
                }
            )

    payload: Dict[str, Any] = {}
    if prompt_entries:
        payload["service_exchanges"] = prompt_entries
    if prompt_console_events:
        payload["console_events"] = prompt_console_events
    if max_bytes and max_bytes > 0:
        while payload and payload_bytes(payload) > max_bytes:
            if prompt_entries:
                prompt_entries.pop(0)
                if prompt_entries:
                    payload["service_exchanges"] = prompt_entries
                else:
                    payload.pop("service_exchanges", None)
                continue
            if prompt_console_events:
                prompt_console_events.pop(0)
                if prompt_console_events:
                    payload["console_events"] = prompt_console_events
                else:
                    payload.pop("console_events", None)
                continue
            break
    return payload or None


def context_state_for_prompt(
    current_state: Any, *, max_bytes: Optional[int] = None
) -> Any:
    """Return a bounded representation of previous/current UI state for prompts."""
    sanitized = sanitize_runtime_value(current_state)
    if not max_bytes or max_bytes <= 0:
        return sanitized
    if payload_bytes(sanitized) <= max_bytes:
        return sanitized
    return trim_runtime_text(
        json.dumps(sanitized, ensure_ascii=False, default=str),
        limit=max_bytes,
    )


def prompt_with_runtime_context(
    *,
    prompt: str,
    runtime_context: Optional[Dict[str, Any]],
    purpose: str,
) -> str:
    prompt_entries = runtime_context_for_prompt(runtime_context)
    if not prompt_entries:
        return prompt

    guidance = (
        "Use these observed runtime service exchanges to align data shapes, edge cases, "
        "and test realism."
    )
    if purpose == "dummy_data":
        guidance = (
            "Use these observed runtime service exchanges to shape realistic dummy data. "
            "Match field names and nesting from observed response_payload values. "
            "Also incorporate console warnings/errors as edge-case hints."
        )

    return (
        f"{prompt}\n\n"
        "Observed runtime context (single-use for this request):\n"
        f"{json.dumps(prompt_entries, ensure_ascii=False)}\n\n"
        f"{guidance}"
    )


# ---------------------------------------------------------------------------
# Chat history helpers
# ---------------------------------------------------------------------------


def to_chat_history_messages(
    history: Sequence[Dict[str, Any]],
) -> List[Message]:
    messages: List[Message] = []
    for item in history or []:
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "")
        if role == MessageRole.USER.value:
            messages.append(Message(role=MessageRole.USER, content=content))
        elif role == MessageRole.ASSISTANT.value:
            messages.append(Message(role=MessageRole.ASSISTANT, content=content))
    return messages


def history_entry(
    *,
    action: str,
    prompt: str,
    tools: List[str],
    user_id: str,
    generated_at: str,
    payload_metadata: Dict[str, Any],
    payload_html: Optional[Dict[str, Any]] = None,
    payload_scripts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entry = {
        "action": action,
        "prompt": prompt,
        "tools": tools,
        "user_id": user_id,
        "generated_at": generated_at,
        "payload_metadata": payload_metadata,
        "payload_html": payload_html,
    }
    if payload_scripts is not None:
        entry["payload_scripts"] = payload_scripts
    return entry


def history_for_prompt(
    history_entries: Sequence[Any],
    *,
    max_entries: Optional[int] = None,
    max_bytes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for entry in history_entries or []:
        if not isinstance(entry, dict):
            continue
        entry_copy = dict(entry)
        entry_copy.pop("payload_html", None)
        entry_copy.pop("payload_scripts", None)
        sanitized.append(entry_copy)

    if max_entries and max_entries > 0 and len(sanitized) > max_entries:
        sanitized = sanitized[-max_entries:]

    if max_bytes and max_bytes > 0:
        while sanitized and payload_bytes(sanitized) > max_bytes:
            sanitized.pop(0)
    return sanitized


# ---------------------------------------------------------------------------
# Script-change tracking
# ---------------------------------------------------------------------------


def changed_scripts(
    new_payload: Dict[str, Any],
    previous_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    changed: Dict[str, Any] = {}
    previous_payload = previous_payload or {}
    for key in SCRIPT_KEYS:
        if key not in new_payload and key not in previous_payload:
            continue
        new_value = new_payload.get(key)
        old_value = previous_payload.get(key)
        if new_value != old_value:
            changed[key] = new_value
    return changed


def scripts_from_history(history_entries: Sequence[Any]) -> Dict[str, Any]:
    scripts: Dict[str, Any] = {}
    for entry in reversed(history_entries or []):
        if not isinstance(entry, dict):
            continue
        payload_scripts = entry.get("payload_scripts")
        if not isinstance(payload_scripts, dict):
            continue
        for key in SCRIPT_KEYS:
            if key in payload_scripts and key not in scripts:
                scripts[key] = payload_scripts[key]
        if len(scripts) == len(SCRIPT_KEYS):
            break
    return scripts


# ---------------------------------------------------------------------------
# Tool / message payload capping
# ---------------------------------------------------------------------------


def cap_tools_for_prompt(
    tools: Optional[List[Dict[str, Any]]],
    *,
    max_tools: Optional[int],
    max_bytes: Optional[int],
) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return tools

    capped = copy.deepcopy(list(tools))
    original_count = len(capped)
    original_bytes_val = payload_bytes(capped)

    if max_tools and max_tools > 0 and len(capped) > max_tools:
        capped = capped[:max_tools]

    if max_bytes and max_bytes > 0 and payload_bytes(capped) > max_bytes:
        for tool in capped:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if isinstance(function, dict):
                function.pop("outputSchema", None)
            tool.pop("outputSchema", None)

    if max_bytes and max_bytes > 0:
        while len(capped) > 1 and payload_bytes(capped) > max_bytes:
            capped.pop()

    if len(capped) != original_count or payload_bytes(capped) != original_bytes_val:
        logger.info(
            "[GeneratedUI] Tool payload capped: count %s -> %s, bytes %s -> %s",
            original_count,
            len(capped),
            original_bytes_val,
            payload_bytes(capped),
        )

    return capped


def cap_message_payload_for_prompt(
    message_payload: Dict[str, Any],
    *,
    max_bytes: Optional[int],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    if not isinstance(message_payload, dict):
        return message_payload, None
    if not max_bytes or max_bytes <= 0:
        return message_payload, None

    capped = copy.deepcopy(message_payload)
    original_bytes_val = payload_bytes(capped)
    steps_applied: List[str] = []
    if payload_bytes(capped) <= max_bytes:
        return capped, None

    context_obj = capped.get("context")
    if isinstance(context_obj, dict):
        context_obj.pop("runtime_console_events", None)
        steps_applied.append("drop_runtime_console_events")
    if payload_bytes(capped) <= max_bytes:
        return (
            capped,
            {
                "original_bytes": original_bytes_val,
                "final_bytes": payload_bytes(capped),
                "budget_bytes": max_bytes,
                "steps": steps_applied,
            },
        )

    context_obj = capped.get("context")
    if isinstance(context_obj, dict):
        exchanges = context_obj.get("runtime_service_exchanges")
        if isinstance(exchanges, list):
            removed = 0
            while exchanges and payload_bytes(capped) > max_bytes:
                exchanges.pop(0)
                removed += 1
            if not exchanges:
                context_obj.pop("runtime_service_exchanges", None)
            if removed:
                steps_applied.append(f"trim_runtime_service_exchanges:{removed}")
    if payload_bytes(capped) <= max_bytes:
        return (
            capped,
            {
                "original_bytes": original_bytes_val,
                "final_bytes": payload_bytes(capped),
                "budget_bytes": max_bytes,
                "steps": steps_applied,
            },
        )

    context_obj = capped.get("context")
    if isinstance(context_obj, dict):
        history = context_obj.get("history")
        if isinstance(history, list):
            removed = 0
            while history and payload_bytes(capped) > max_bytes:
                history.pop(0)
                removed += 1
            if not history:
                context_obj.pop("history", None)
            if removed:
                steps_applied.append(f"trim_history:{removed}")
    if payload_bytes(capped) <= max_bytes:
        return (
            capped,
            {
                "original_bytes": original_bytes_val,
                "final_bytes": payload_bytes(capped),
                "budget_bytes": max_bytes,
                "steps": steps_applied,
            },
        )

    context_obj = capped.get("context")
    if isinstance(context_obj, dict) and "current_state" in context_obj:
        context_obj["current_state"] = context_state_for_prompt(
            context_obj.get("current_state"),
            max_bytes=max(512, max_bytes // 6),
        )
        steps_applied.append("compact_current_state")
    if payload_bytes(capped) <= max_bytes:
        return (
            capped,
            {
                "original_bytes": original_bytes_val,
                "final_bytes": payload_bytes(capped),
                "budget_bytes": max_bytes,
                "steps": steps_applied,
            },
        )

    context_obj = capped.get("context")
    if isinstance(context_obj, dict):
        context_obj.pop("current_state", None)
        if not context_obj:
            capped.pop("context", None)
        steps_applied.append("drop_current_state")

    if payload_bytes(capped) > max_bytes:
        logger.warning(
            "[GeneratedUI] message payload still above budget after capping: bytes=%s budget=%s",
            payload_bytes(capped),
            max_bytes,
        )
    else:
        logger.info(
            "[GeneratedUI] Message payload capped to %s bytes (budget=%s)",
            payload_bytes(capped),
            max_bytes,
        )
    return (
        capped,
        {
            "original_bytes": original_bytes_val,
            "final_bytes": payload_bytes(capped),
            "budget_bytes": max_bytes,
            "steps": steps_applied,
        },
    )


# ---------------------------------------------------------------------------
# Lightweight JSON extraction / parsing
# ---------------------------------------------------------------------------


def extract_content(response: Any) -> str:
    if response is None:
        raise HTTPException(status_code=502, detail="Generation response was empty")

    if isinstance(response, dict):
        content = response.get("content")
        if isinstance(content, str):
            return content

    choices = getattr(response, "choices", None)
    if choices:
        first = choices[0]
        message = getattr(first, "message", None)
        if message and getattr(message, "content", None):
            return message.content
        delta = getattr(first, "delta", None)
        if delta and getattr(delta, "content", None):
            return delta.content

    raise HTTPException(
        status_code=502,
        detail="Unable to extract content from generation response",
    )


def extract_json_block(text: str) -> Optional[str]:
    start = None
    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(text):
        if start is None:
            if char == "{":
                start = idx
                depth = 1
            continue

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"' and not escape:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : idx + 1]
    return None


def parse_json(payload_str: str) -> Dict[str, Any]:
    try:
        return json.loads(payload_str)
    except json.JSONDecodeError:
        candidate = extract_json_block(payload_str)
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=502,
                    detail="Generated content is not valid JSON",
                ) from exc
        raise HTTPException(
            status_code=502,
            detail="Generated content is not valid JSON",
        )


# ---------------------------------------------------------------------------
# Generic value → JSON-safe conversion
# ---------------------------------------------------------------------------


def to_json_value(value: Any) -> Any:
    """Recursively convert *value* to a JSON-serialisable structure."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(v) for v in value]
    if hasattr(value, "model_dump"):
        with contextlib.suppress(Exception):
            return to_json_value(value.model_dump())
    if hasattr(value, "__dict__"):
        with contextlib.suppress(Exception):
            return to_json_value(vars(value))
    return str(value)
