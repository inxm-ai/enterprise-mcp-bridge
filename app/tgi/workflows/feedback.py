"""
Pure functions and a ``FeedbackService`` for user-feedback handling in
workflow agents.

Pure functions cover parsing, normalisation and payload construction.
``FeedbackService`` encapsulates the async LLM calls needed for
rendering feedback questions and deciding whether to rerun agents.
"""

import json
import logging
import re
from typing import Any, Callable, Optional

from app.elicitation import canonicalize_elicitation_payload, parse_user_feedback_tag
from app.tgi.workflows.context_builder import (
    ContextCompressionReport,
    analyze_context_compression,
    compact_large_structure,
    create_context_summary,
)
from app.tgi.workflows.contextual_llm_helper import run_context_aware_llm_helper
from app.tgi.workflows.dict_utils import get_path_value
from app.tgi.workflows.models import WorkflowExecutionState
from app.tgi.workflows.reroute import (
    is_external_reroute_target,
    parse_workflow_reroute_target,
    resolve_external_with_values,
)

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Pure parsing / normalisation helpers
# ---------------------------------------------------------------------------


def format_feedback_block(payload: dict[str, Any]) -> str:
    """Wrap *payload* in a ``<user_feedback_needed>`` tag."""
    try:
        canonical = canonicalize_elicitation_payload(payload)
        serialized = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        serialized = json.dumps({"message": payload.get("message") or ""})
    return f"<user_feedback_needed>{serialized}</user_feedback_needed>"


def feedback_expected_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the ``expected_responses`` list from a feedback payload."""
    expected = payload.get("expected_responses")
    if isinstance(expected, list):
        return expected
    meta = payload.get("meta") or {}
    if not isinstance(meta, dict):
        return []
    expected = meta.get("expected_responses")
    if isinstance(expected, list):
        return expected
    return []


def legacy_feedback_spec_view(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Keep legacy state readers working while the canonical payload
    is the source of truth.
    """
    legacy = dict(payload or {})
    if "question" not in legacy:
        legacy["question"] = payload.get("message") or ""
    if "expected_responses" not in legacy:
        legacy["expected_responses"] = feedback_expected_entries(payload)
    return legacy


def set_feedback_expected_entries(
    payload: dict[str, Any], expected: list[dict[str, Any]]
) -> None:
    """Set ``expected_responses`` inside the meta section of *payload*."""
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    meta["expected_responses"] = expected
    payload["meta"] = meta


def parse_feedback_json_value(value: Any) -> Any:
    """Attempt to JSON-parse a string value; return as-is otherwise."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        parsed = json.loads(stripped)
    except Exception:
        return value
    return parsed


def parse_feedback_payload(raw: Optional[str]) -> Optional[dict[str, Any]]:
    """Parse a raw JSON feedback payload, returning ``None`` on failure."""
    if not raw or not isinstance(raw, str):
        return None
    stripped = raw.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        parsed = json.loads(stripped)
    except Exception:
        return None
    if isinstance(parsed, dict):
        return canonicalize_elicitation_payload(parsed)
    return None


def normalize_feedback_option(item: Any) -> Optional[tuple[str, str]]:
    """Normalise a single feedback option to ``(key, value)``."""
    if item is None:
        return None
    if isinstance(item, dict):
        if "key" in item:
            key = item.get("key")
            value = item.get("value") or item.get("label") or item.get("name")
            if value is None:
                value = key
            return str(key), str(value)
        if "id" in item:
            key = item.get("id")
            value = (
                item.get("name") or item.get("description") or item.get("label") or key
            )
            return str(key), str(value)
        if item:
            key, value = next(iter(item.items()))
            return str(key), str(value)
        return None
    if isinstance(item, (str, int, float, bool)):
        return str(item), str(item)
    return None


def normalize_feedback_options(items: Any) -> list[dict[str, str]]:
    """Normalise a list of option items to ``[{"key": ..., "value": ...}]``."""
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in items:
        parsed = normalize_feedback_option(item)
        if not parsed:
            continue
        key, value = parsed
        normalized.append({"key": key, "value": value})
    return normalized


def normalize_feedback_rerun_decision(response: Optional[str]) -> str:
    """Normalise the LLM's ``USE_PREVIOUS``/``RERUN`` decision."""
    if not response:
        return "RERUN"
    text = response.strip().upper()
    if "USE_PREVIOUS" in text or "USE PREVIOUS" in text or "SKIP" in text:
        return "USE_PREVIOUS"
    if "RERUN" in text or "RE-RUN" in text or "RUN AGAIN" in text:
        return "RERUN"
    return "RERUN"


def normalize_user_query_summary(summary: Optional[str]) -> Optional[str]:
    """Clean up an LLM-generated user query summary."""
    if not summary:
        return None
    cleaned = summary.strip().strip("`")
    cleaned = re.sub(
        r"^(summary|combined query|rewritten request)\s*[:\-]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = " ".join(cleaned.split())
    return cleaned or None


def parse_feedback_args(text: str) -> tuple[list[str], dict[str, str]]:
    """Parse comma-separated positional and keyword arguments."""
    args: list[str] = []
    kwargs: dict[str, str] = {}
    if not text:
        return args, kwargs
    parts: list[str] = []
    current: list[str] = []
    in_quote: Optional[str] = None
    escape = False
    for ch in text:
        if escape:
            current.append(ch)
            escape = False
            continue
        if ch == "\\":
            escape = True
            current.append(ch)
            continue
        if ch in ("'", '"'):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
            current.append(ch)
            continue
        if ch == "," and in_quote is None:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(ch)
    if current:
        part = "".join(current).strip()
        if part:
            parts.append(part)
    for part in parts:
        if "=" in part:
            name, value = part.split("=", 1)
            name = name.strip()
            value = value.strip()
            kwargs[name] = strip_feedback_arg_value(value)
        else:
            args.append(strip_feedback_arg_value(part))
    return args, kwargs


def strip_feedback_arg_value(value: str) -> str:
    """Strip surrounding quotes from a feedback argument value."""
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
        return cleaned[1:-1]
    return cleaned


def resolve_feedback_each_value(
    each: Any,
    agent_context: dict[str, Any],
    shared_context: dict[str, Any],
) -> Any:
    """Resolve feedback ``each`` values from agent or shared context."""
    if each is None:
        return None
    if isinstance(each, list):
        return each
    if isinstance(each, dict):
        return each
    if not isinstance(each, str):
        return None
    value = get_path_value(agent_context, each)
    if value is not None:
        return parse_feedback_json_value(value)
    value = get_path_value(shared_context, each)
    if value is not None:
        return parse_feedback_json_value(value)
    agents_ctx = (
        shared_context.get("agents") if isinstance(shared_context, dict) else None
    )
    if isinstance(agents_ctx, dict):
        value = get_path_value(agents_ctx, each)
        if value is not None:
            return parse_feedback_json_value(value)
    return None


def build_feedback_choices_from_payload(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build choices from an already-parsed feedback payload."""
    choices: list[dict[str, Any]] = []
    expected = feedback_expected_entries(payload)
    if not isinstance(expected, list):
        return choices
    for entry in expected:
        if not isinstance(entry, dict):
            continue
        choice: dict[str, Any] = {
            "id": entry.get("id"),
            "to": entry.get("to"),
        }
        if "with" in entry:
            with_fields = entry.get("with") or []
            if isinstance(with_fields, str):
                with_fields = [with_fields]
            with_fields = [w for w in with_fields if isinstance(w, str)]
            choice["with"] = with_fields
        if "each" in entry:
            choice["each"] = entry.get("each")
        if "options" in entry:
            choice["options"] = entry.get("options") or []
        for key, value in entry.items():
            if key in {"id", "to", "with", "each", "options"}:
                continue
            choice[key] = value
        choices.append(choice)
    return choices


def resolve_external_with_in_feedback_payload(
    payload: dict[str, Any],
    agent_context: dict[str, Any],
    shared_context: dict[str, Any],
) -> dict[str, Any]:
    """Resolve external ``with`` values inside a feedback payload."""
    expected = feedback_expected_entries(payload)
    if not isinstance(expected, list):
        return payload
    for entry in expected:
        if not isinstance(entry, dict):
            continue
        target = entry.get("to")
        if not is_external_reroute_target(target):
            continue
        with_fields = entry.get("with")
        if isinstance(with_fields, str):
            with_fields = [with_fields]
        if not isinstance(with_fields, list):
            continue
        fields = [w for w in with_fields if isinstance(w, str)]
        if not fields:
            continue
        entry["with"] = resolve_external_with_values(
            fields, agent_context, shared_context
        )
    set_feedback_expected_entries(payload, expected)
    return payload


def collect_feedback_context(
    ask_config: dict[str, Any],
    agent_context: dict[str, Any],
    shared_context: dict[str, Any],
) -> dict[str, Any]:
    """Gather context values referenced by feedback ``each``."""
    context_payload: dict[str, Any] = {}
    expected = ask_config.get("expected_responses") or []
    if isinstance(expected, dict):
        expected = [expected]
    for group in expected:
        if not isinstance(group, dict):
            continue
        for choice_id, cfg in group.items():
            if not isinstance(cfg, dict):
                continue
            each = cfg.get("each")
            if not each:
                continue
            value = resolve_feedback_each_value(each, agent_context, shared_context)
            if value is not None:
                context_payload[str(choice_id)] = value
    return context_payload


def build_feedback_choices(
    ask_config: dict[str, Any],
    agent_context: dict[str, Any],
    shared_context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build feedback choices from an ``ask`` config dict."""
    choices: list[dict[str, Any]] = []
    expected = ask_config.get("expected_responses") or []
    if isinstance(expected, dict):
        expected = [expected]
    for group in expected:
        if not isinstance(group, dict):
            continue
        for choice_id, cfg in group.items():
            if not isinstance(cfg, dict):
                continue
            target = cfg.get("to")
            with_fields = cfg.get("with") or []
            if isinstance(with_fields, str):
                with_fields = [with_fields]
            with_fields = [w for w in with_fields if isinstance(w, str)]
            extra_fields = {
                k: v for k, v in cfg.items() if k not in {"to", "with", "each"}
            }
            each = cfg.get("each")
            options_raw = resolve_feedback_each_value(
                each, agent_context, shared_context
            )
            options = normalize_feedback_options(options_raw)
            if each and not with_fields:
                logger.warning(
                    "[feedback] Feedback option '%s' has 'each' without explicit 'with'; "
                    "selection context will not be propagated.",
                    choice_id,
                )
            choice_entry: dict[str, Any] = {
                "id": str(choice_id),
                "to": target,
                "with": with_fields,
            }
            choice_entry.update(extra_fields)
            if each is not None:
                choice_entry["each"] = each
                choice_entry["options"] = options
            choices.append(choice_entry)
    return choices


def build_feedback_payload(
    question: str,
    choices: list[dict[str, Any]],
    agent_context: dict[str, Any],
    shared_context: dict[str, Any],
) -> dict[str, Any]:
    """Build a canonical feedback payload from a rendered question and choices."""
    expected_responses: list[dict[str, Any]] = []
    for choice in choices:
        entry: dict[str, Any] = {
            "id": choice.get("id"),
            "to": choice.get("to"),
        }
        if choice.get("with"):
            with_fields = choice["with"]
            if is_external_reroute_target(choice.get("to")):
                entry["with"] = resolve_external_with_values(
                    with_fields, agent_context, shared_context
                )
            else:
                entry["with"] = with_fields
        if choice.get("each") is not None:
            entry["each"] = choice.get("each")
        if "options" in choice:
            entry["options"] = choice.get("options") or []
        for key, value in choice.items():
            if key in {"id", "to", "with", "each", "options"}:
                continue
            entry[key] = value
        expected_responses.append(entry)
    selection_ids = [
        str(entry.get("id"))
        for entry in expected_responses
        if isinstance(entry.get("id"), str)
    ]
    requested_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    input_fields_meta: dict[str, dict[str, Any]] = {}
    if selection_ids:
        requested_schema["properties"] = {
            "selection": {
                "type": "string",
                "enum": selection_ids,
                "description": "Selected response id.",
            },
            "value": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "null"},
                ]
            },
        }
        requested_schema["required"] = ["selection"]
        for choice in choices:
            choice_input = choice.get("input")
            if not isinstance(choice_input, dict):
                continue
            choice_id = str(choice.get("id") or "")
            for field_name, field_def in choice_input.items():
                if field_name in {"selection", "value"}:
                    logger.warning(
                        "[feedback] Input field name '%s' for choice '%s' "
                        "collides with reserved schema property; skipping.",
                        field_name,
                        choice_id,
                    )
                    continue
                if not isinstance(field_def, dict):
                    field_def = {"type": "string"}
                if "type" not in field_def:
                    field_def["type"] = "string"
                requested_schema["properties"][field_name] = field_def
                input_fields_meta[field_name] = {
                    "for_selection": choice_id,
                }
    else:
        requested_schema["properties"] = {
            "response": {"type": "string"},
        }
        requested_schema["required"] = ["response"]
    meta: dict[str, Any] = {"expected_responses": expected_responses}
    if input_fields_meta:
        meta["input_fields"] = input_fields_meta
    return {
        "message": question,
        "requestedSchema": requested_schema,
        "meta": meta,
    }


def is_placeholder_feedback_arg(
    value: Any,
    field: str,
    agent_entry: Optional[dict[str, Any]] = None,
    shared_context: Optional[dict[str, Any]] = None,
) -> bool:
    """
    Treat args that repeat the field name (e.g. ``"plan_id"``) as placeholders.
    """
    if not isinstance(value, str) or not isinstance(field, str):
        return False
    if value != field:
        return False
    if agent_entry and get_path_value(agent_entry, field) is not None:
        return True
    if shared_context and get_path_value(shared_context, field) is not None:
        return True
    return True


def parse_feedback_action(raw: str) -> Optional[dict[str, Any]]:
    """Parse a raw feedback string into a structured action dict."""
    if not raw:
        return None
    text = (parse_user_feedback_tag(raw) or raw).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"decline", "reject", "no"}:
        return {"action": "decline"}
    if lowered in {"cancel", "dismiss"}:
        return {"action": "cancel"}
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                action = str(parsed.get("action") or "accept").lower()
                if action in {"decline", "cancel"}:
                    return {"action": action}
                content = parsed.get("content")
                if isinstance(content, dict):
                    return {"action": "accept", "content": content}
                return parsed
        except Exception:
            pass
    workflow_target = parse_workflow_reroute_target(text)
    if workflow_target:
        return {"workflow": workflow_target}
    func_match = re.match(r"^(?P<name>[\w\-]+)\s*\((?P<args>.*)\)$", text)
    if func_match:
        name = func_match.group("name")
        args_raw = func_match.group("args") or ""
        args, kwargs = parse_feedback_args(args_raw)
        return {"to": name, "args": args, "kwargs": kwargs}
    return {"to": text}


def resolve_feedback_action(
    feedback_text: str,
    agent_entry: dict[str, Any],
    shared_context: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Resolve structured feedback into an actionable result."""
    action = parse_feedback_action(feedback_text)
    if not action:
        return None
    action_type = action.get("action")
    if action_type in {"decline", "cancel"}:
        return {"action": action_type}

    if action_type == "accept" and isinstance(action.get("content"), dict):
        content = action.get("content") or {}
        if "selection" in content and "to" not in action and "id" not in action:
            action["to"] = content.get("selection")
        if "value" in content and "args" not in action:
            action["args"] = [content.get("value")]
        if "kwargs" not in action:
            action["kwargs"] = {
                key: value
                for key, value in content.items()
                if key not in {"selection", "value"}
            }

    choices = (
        agent_entry.get("elicitation_choices")
        or agent_entry.get("feedback_choices")
        or []
    )
    if not choices:
        return {"action": "accept", "target": action.get("to")}
    matched_choice: Optional[dict[str, Any]] = None
    raw_to = action.get("to")
    raw_workflow = action.get("workflow")
    raw_id = action.get("id")
    for choice in choices:
        choice_to = choice.get("to")
        choice_id = choice.get("id")
        choice_workflow = parse_workflow_reroute_target(choice_to or "")
        if raw_id and choice_id == raw_id:
            matched_choice = choice
            break
        if raw_workflow and choice_workflow == raw_workflow:
            matched_choice = choice
            break
        if raw_to and choice_to == raw_to:
            matched_choice = choice
            break
    if not matched_choice and raw_to and choices:
        for choice in choices:
            if choice.get("id") == raw_to:
                matched_choice = choice
                break
    with_fields = matched_choice.get("with") if matched_choice else []
    if isinstance(with_fields, str):
        with_fields = [with_fields]
    with_fields = [w for w in with_fields if isinstance(w, str)]
    assignments: dict[str, Any] = {}
    kwargs = action.get("kwargs") or {}
    if isinstance(kwargs, dict) and kwargs:
        for key, value in kwargs.items():
            if is_placeholder_feedback_arg(value, key, agent_entry, shared_context):
                continue
            assignments[key] = value
    args = action.get("args") or []
    if not assignments and args and with_fields:
        for idx, field in enumerate(with_fields):
            if idx >= len(args):
                break
            if is_placeholder_feedback_arg(
                args[idx], field, agent_entry, shared_context
            ):
                continue
            assignments[field] = args[idx]
    if matched_choice and matched_choice.get("options") and args:
        option_key = args[0]
        if with_fields and with_fields[0] not in assignments:
            assignments[with_fields[0]] = option_key
    # Map per-choice input field values into assignments
    choice_input = matched_choice.get("input") if matched_choice else None
    if isinstance(choice_input, dict) and isinstance(kwargs, dict):
        for field_name in choice_input:
            if field_name in assignments:
                continue
            if field_name in kwargs:
                assignments[field_name] = kwargs[field_name]
    if not matched_choice and raw_to:
        return {
            "target": raw_to,
            "workflow_target": parse_workflow_reroute_target(raw_to),
            "with_fields": with_fields,
            "assignments": assignments,
            "input_fields": choice_input,
        }
    if not matched_choice and raw_workflow:
        return {
            "workflow_target": raw_workflow,
            "with_fields": with_fields,
            "assignments": assignments,
            "input_fields": choice_input,
        }
    if not matched_choice:
        return None
    choice_target = matched_choice.get("to")
    workflow_target = parse_workflow_reroute_target(choice_target or "")
    return {
        "target": None if workflow_target else choice_target,
        "workflow_target": workflow_target,
        "with_fields": with_fields,
        "assignments": assignments,
        "input_fields": choice_input,
    }


# ---------------------------------------------------------------------------
# FeedbackService — async methods that require an LLM client
# ---------------------------------------------------------------------------


class FeedbackService:
    """
    Encapsulates the LLM calls needed for rendering feedback questions,
    summarising user queries and deciding whether to rerun an agent after
    receiving feedback.
    """

    SIZE_THRESHOLD = 12000
    PLACEHOLDER_THRESHOLD = 6
    MARKER_THRESHOLD = 4

    def __init__(self, llm_client: Any, state_store: Any = None):
        self.llm_client = llm_client
        self.state_store = state_store

    def _compression_report(
        self, *, original: Any, compacted: Any
    ) -> ContextCompressionReport:
        return analyze_context_compression(
            original,
            compacted,
            size_threshold=self.SIZE_THRESHOLD,
            placeholder_threshold=self.PLACEHOLDER_THRESHOLD,
            marker_threshold=self.MARKER_THRESHOLD,
        )

    async def _ask_with_context_budget(
        self,
        *,
        base_prompt: str,
        base_request: Any,
        question: str,
        access_token: Optional[str],
        span: Any,
        context_payload: Any,
        execution_id: Optional[str],
    ) -> str:
        compacted = compact_large_structure(context_payload, max_items=4, max_depth=4)
        report = self._compression_report(original=context_payload, compacted=compacted)

        use_subagent = bool(
            execution_id
            and self.state_store is not None
            and report.should_use_lazy_context
        )
        if use_subagent:
            helper_prompt = (
                f"{base_prompt}\n"
                "If context summaries are insufficient, use get_workflow_context to fetch exact data.\n"
                "Never invent details that are not present in context."
            )
            helper_result = await run_context_aware_llm_helper(
                llm_client=self.llm_client,
                base_request=base_request,
                access_token=access_token,
                span=span,
                system_prompt=helper_prompt,
                user_payload=question,
                state_store=self.state_store,
                execution_id=execution_id,
                max_turns=2,
            )
            response = (helper_result.text or "").strip()
            if response:
                logger.info(
                    "[FeedbackService] Used context-aware helper for prompt. "
                    "size=%d placeholders=%d markers=%d turns=%d tools=%s",
                    report.serialized_size,
                    report.placeholder_count,
                    report.compaction_marker_count,
                    helper_result.turns,
                    helper_result.used_tools,
                )
                return response

        response = await self.llm_client.ask(
            base_prompt=base_prompt,
            base_request=base_request,
            question=question,
            access_token=access_token or "",
            outer_span=span,
        )
        return (response or "").strip()

    async def render_feedback_question(
        self,
        ask_config: dict[str, Any],
        agent_context: dict[str, Any],
        shared_context: dict[str, Any],
        request: Any,
        access_token: Optional[str],
        span,
        execution_id: Optional[str] = None,
    ) -> str:
        instruction = (ask_config.get("question") or "").strip()
        if not instruction:
            return ""
        logger.info(
            "[FeedbackService] Rendering feedback question. Instruction=%s", instruction
        )
        prompt = (
            "USER_FEEDBACK_QUESTION\n"
            "Write a user-facing question based on the instruction and context.\n"
            "Use the context to include relevant option details. Return only the question."
        )
        context_summary_str = create_context_summary(shared_context, scoped=True)
        agent_snapshot = compact_large_structure(
            agent_context, max_items=4, max_depth=4
        )
        render_context = {
            "shared_context": shared_context,
            "agent_context": agent_context,
        }
        render_snapshot = compact_large_structure(
            render_context,
            max_items=4,
            max_depth=4,
        )
        compression = self._compression_report(
            original=render_context,
            compacted=render_snapshot,
        )
        plan_keys = [
            key
            for key in ("plan", "plans", "plan_id", "plan_ids")
            if isinstance(agent_context, dict) and key in agent_context
        ]
        if plan_keys:
            logger.info(
                "[FeedbackService] Feedback question agent context has plan keys: %s",
                plan_keys,
            )
        option_context = collect_feedback_context(
            ask_config, agent_context, shared_context
        )
        question_parts = [
            f"Instruction: {instruction}",
            f"Context summary: {context_summary_str}",
            f"Agent context: {json.dumps(agent_snapshot, ensure_ascii=False, default=str)}",
            (
                "Compression report: "
                f"{json.dumps({'serialized_size': compression.serialized_size, 'placeholder_count': compression.placeholder_count, 'compaction_marker_count': compression.compaction_marker_count, 'lossy': compression.lossy}, ensure_ascii=False)}"
            ),
        ]
        if option_context:
            question_parts.append(
                f"Option context: {json.dumps(option_context, ensure_ascii=False, default=str)}"
            )
        question_parts.append("Question:")
        question_payload = "\n".join(question_parts)
        logger.info(
            "[FeedbackService] Feedback question payload size=%d option_context=%s",
            len(question_payload),
            bool(option_context),
        )
        response = await self._ask_with_context_budget(
            base_prompt=prompt,
            base_request=request,
            question=question_payload,
            access_token=access_token,
            span=span,
            context_payload=render_context,
            execution_id=execution_id,
        )
        cleaned = (response or "").strip()
        if not cleaned:
            logger.info(
                "[FeedbackService] LLM returned empty response; falling back to instruction."
            )
            return instruction
        if cleaned == instruction:
            logger.info("[FeedbackService] LLM returned the instruction verbatim.")
        else:
            logger.info(
                "[FeedbackService] Feedback question rendered. Length=%d", len(cleaned)
            )
        return cleaned or instruction

    async def summarize_user_query(
        self,
        base_query: Optional[str],
        feedback: str,
        feedback_prompt: Optional[str],
        request: Any,
        access_token: Optional[str],
        span,
        execution_id: Optional[str] = None,
    ) -> Optional[str]:
        base_query = (base_query or "").strip()
        feedback = (feedback or "").strip()
        feedback_prompt = (feedback_prompt or "").strip()
        if not base_query and not feedback:
            return None
        prompt = (
            "USER_QUERY_SUMMARY\n"
            "You rewrite a conversation into a single, self-contained user request.\n"
            "Combine the original user request, the assistant clarification question, "
            "and the user's response. Resolve references like 'the first one' or 'that' "
            "using the clarification question. Keep it concise and specific.\n"
            "Return only the rewritten request with no extra text."
        )
        question_parts: list[str] = []
        if base_query:
            question_parts.append(f"Original user request: {base_query}")
        if feedback_prompt:
            question_parts.append(f"Assistant clarification: {feedback_prompt}")
        if feedback:
            question_parts.append(f"User response: {feedback}")
        question_parts.append("Rewritten request:")
        question = "\n".join(question_parts)
        summary_context = {
            "base_query": base_query,
            "feedback_prompt": feedback_prompt,
            "feedback": feedback,
        }
        summary = await self._ask_with_context_budget(
            base_prompt=prompt,
            base_request=request,
            question=question,
            access_token=access_token,
            span=span,
            context_payload=summary_context,
            execution_id=execution_id,
        )
        return normalize_user_query_summary(summary)

    async def should_rerun_feedback_agent(
        self,
        base_query: Optional[str],
        feedback: str,
        feedback_prompt: Optional[str],
        agent_name: str,
        agent_context: dict[str, Any],
        request: Any,
        access_token: Optional[str],
        span,
        execution_id: Optional[str] = None,
    ) -> bool:
        prompt = (
            "FEEDBACK_RERUN_DECISION\n"
            "Decide if the agent needs to run again after user feedback.\n"
            "Return only one token: USE_PREVIOUS or RERUN.\n"
            "Use USE_PREVIOUS only when the user is simply confirming or approving"
            " the prior output and no new data, choices, or corrections were provided.\n"
            "Use RERUN when the assistant asked the user to choose among options,"
            " provide missing details, clarify preferences, supply IDs/dates/times,"
            " or when the response changes or adds information that should alter"
            " the agent output or downstream routing.\n"
            "Example (options): If the assistant listed options and asked which one"
            " to pick, the user's reply should always result in RERUN.\n"
            "Example (confirmation): If the assistant asked 'Does that look right?'"
            " and the user replied 'Yes', return USE_PREVIOUS."
        )
        context_snapshot: dict[str, Any] = {}
        if agent_context:
            filtered = {
                key: value
                for key, value in agent_context.items()
                if key not in {"_full_tool_results"}
            }
            context_snapshot = compact_large_structure(
                filtered, max_items=3, max_depth=2
            )
        question = "\n".join(
            [
                f"Agent: {agent_name}",
                f"Original user request: {base_query or ''}",
                f"Assistant question: {feedback_prompt or ''}",
                f"User response: {feedback}",
                f"Agent context snapshot: {json.dumps(context_snapshot, ensure_ascii=False, default=str)}",
                "Decision:",
            ]
        )
        rerun_context = {
            "base_query": base_query or "",
            "feedback": feedback,
            "feedback_prompt": feedback_prompt or "",
            "agent_name": agent_name,
            "agent_context": agent_context,
        }
        response = await self._ask_with_context_budget(
            base_prompt=prompt,
            base_request=request,
            question=question,
            access_token=access_token,
            span=span,
            context_payload=rerun_context,
            execution_id=execution_id,
        )
        decision = normalize_feedback_rerun_decision(response)
        return decision != "USE_PREVIOUS"

    async def merge_feedback(
        self,
        state: WorkflowExecutionState,
        feedback: str,
        request: Any,
        access_token: Optional[str],
        span,
        save_fn: Callable[[WorkflowExecutionState], None],
    ) -> None:
        """Process user feedback and merge it into workflow state.

        Handles structured feedback (decline/cancel/selection) as well as
        free-text feedback that may trigger user query re-summarisation and
        agent rerun decisions.

        ``save_fn`` is called once at the end to persist the updated state.
        """
        from app.tgi.workflows import state_management, tag_parser

        state.awaiting_feedback = False
        state_management.reset_feedback_pause_notice(state)

        if state.current_agent:
            agent_entry = state.context["agents"].setdefault(
                state.current_agent, {"content": "", "pass_through": False}
            )
            prior_content = agent_entry.get("content", "")
            logger.info(
                "[FeedbackService.merge_feedback] Agent '%s' receiving feedback. "
                "Before merge - completed: %s, awaiting_feedback: %s, had_feedback: %s, reroute_reason: %s",
                state.current_agent,
                agent_entry.get("completed"),
                agent_entry.get("awaiting_feedback"),
                agent_entry.get("had_feedback"),
                agent_entry.get("reroute_reason"),
            )
            feedback_tag = tag_parser.extract_tag(feedback, "user_feedback")
            feedback_content = feedback_tag or feedback

            feedback_action = None
            if feedback_tag:
                feedback_action = resolve_feedback_action(
                    feedback_tag, agent_entry, state.context
                )

            if feedback_action:
                if feedback_action.get("action") in {"decline", "cancel"}:
                    agent_entry["feedback_selection"] = feedback_tag
                    agent_entry["feedback_response"] = {
                        "action": feedback_action.get("action"),
                        "content": None,
                    }
                    agent_entry.pop("awaiting_feedback", None)
                    agent_entry["had_feedback"] = True
                    agent_entry["completed"] = True
                    agent_entry["skip_feedback_rerun"] = True
                    logger.info(
                        "[FeedbackService.merge_feedback] Agent '%s' received '%s' feedback",
                        state.current_agent,
                        feedback_action.get("action"),
                    )
                else:
                    agent_entry["feedback_selection"] = feedback_tag
                    agent_entry["pending_user_reroute"] = feedback_action
                    agent_entry.pop("awaiting_feedback", None)
                    agent_entry["completed"] = False
                    agent_entry["had_feedback"] = True
                    # For structured self-reroutes with input fields (e.g. "adjust"),
                    # preserve the original intent and append the adjustment text.
                    reroute_target = feedback_action.get("target")
                    input_fields = feedback_action.get("input_fields")
                    assignments = feedback_action.get("assignments")
                    if (
                        reroute_target == state.current_agent
                        and isinstance(input_fields, dict)
                        and isinstance(assignments, dict)
                    ):
                        input_parts = [
                            str(assignments[field_name])
                            for field_name in input_fields
                            if assignments.get(field_name) is not None
                        ]
                        if input_parts:
                            base_query = state.context.get("user_query")
                            if not base_query:
                                messages = state.context.get("user_messages") or []
                                if isinstance(messages, list) and len(messages) >= 2:
                                    base_query = str(messages[-2])
                                elif messages:
                                    base_query = str(messages[-1])
                            adjustment = " ".join(input_parts).strip()
                            if base_query:
                                state.context["user_query"] = (
                                    f"{base_query}\nAdjustment: {adjustment}"
                                )
                            else:
                                state.context["user_query"] = (
                                    f"Adjustment: {adjustment}"
                                )
                    # Ensure the feedback-processing path runs before other agents
                    state.context["_resume_agent"] = state.current_agent
                    logger.info(
                        "[FeedbackService.merge_feedback] Agent '%s' received structured feedback selection",
                        state.current_agent,
                    )
            elif feedback_tag:
                logger.debug(
                    "[FeedbackService.merge_feedback] Structured feedback tag received but no matching choice found for agent '%s'",
                    state.current_agent,
                )
            else:
                agent_entry["content"] = (
                    agent_entry.get("content", "") + f" {feedback_content}"
                ).strip()
                agent_entry.pop("awaiting_feedback", None)
                agent_entry["completed"] = False
                agent_entry["had_feedback"] = True
                # Clear any stored reroute reason so the agent can make a fresh decision
                # based on the user's feedback
                agent_entry.pop("reroute_reason", None)

                feedback_prompt = agent_entry.get("feedback_prompt") or prior_content
                base_query = state.context.get("user_query")
                if not base_query:
                    messages = state.context.get("user_messages") or []
                    if isinstance(messages, list) and len(messages) >= 2:
                        base_query = str(messages[-2])
                    elif messages:
                        base_query = str(messages[-1])

                combined_query = await self.summarize_user_query(
                    base_query=base_query,
                    feedback=feedback_content,
                    feedback_prompt=feedback_prompt,
                    request=request,
                    access_token=access_token,
                    span=span,
                    execution_id=state.execution_id,
                )
                if combined_query:
                    state.context["user_query"] = combined_query

                should_rerun = await self.should_rerun_feedback_agent(
                    base_query=base_query,
                    feedback=feedback_content,
                    feedback_prompt=feedback_prompt,
                    agent_name=state.current_agent,
                    agent_context=agent_entry,
                    request=request,
                    access_token=access_token,
                    span=span,
                    execution_id=state.execution_id,
                )
                if should_rerun:
                    # Remember which agent should resume first after feedback
                    state.context["_resume_agent"] = state.current_agent
                    logger.info(
                        "[FeedbackService.merge_feedback] Agent '%s' set to rerun after feedback",
                        state.current_agent,
                    )
                else:
                    agent_entry["completed"] = True
                    agent_entry["skip_feedback_rerun"] = True
                    logger.info(
                        "[FeedbackService.merge_feedback] Agent '%s' will reuse prior output after feedback",
                        state.current_agent,
                    )

        state.context.setdefault("feedback", []).append(feedback)
        save_fn(state)
