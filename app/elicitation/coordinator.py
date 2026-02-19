from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import jsonschema

logger = logging.getLogger("uvicorn.error")

_PRIMITIVE_JSON_TYPES = {"string", "number", "integer", "boolean", "null"}
_PRIMITIVE_PY_TYPES = (str, int, float, bool, type(None))
_USER_FEEDBACK_RE = re.compile(
    r"<user_feedback[^>]*>(?P<content>.*?)</user_feedback>",
    flags=re.IGNORECASE | re.DOTALL,
)


class ElicitationError(Exception):
    pass


class UnsupportedElicitationSchemaError(ElicitationError):
    def __init__(self, message: str, payload: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.payload = payload or {}


class InvalidUserFeedbackError(ElicitationError):
    def __init__(
        self,
        message: str,
        *,
        session_key: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.session_key = session_key
        self.payload = payload or {}


class ElicitationRequiredError(ElicitationError):
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        session_key: Optional[str],
        requires_session: bool,
    ):
        super().__init__("User feedback required before continuing.")
        self.payload = payload
        self.session_key = session_key
        self.requires_session = requires_session
        self.resume_token = session_key

    def to_client_payload(self) -> dict[str, Any]:
        return {
            "error": "feedback_required",
            "awaiting_feedback": True,
            "elicitation": self.payload,
            "resume_token": self.resume_token,
            "requires_session": self.requires_session,
        }


@dataclass
class PendingElicitation:
    payload: dict[str, Any]
    created_at: float


def parse_user_feedback_tag(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    match = _USER_FEEDBACK_RE.search(text)
    if not match:
        stripped = text.strip()
        return stripped or None
    inner = (match.group("content") or "").strip()
    return inner or None


def _ensure_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _is_primitive_enum(values: Any) -> bool:
    if not isinstance(values, list) or not values:
        return False
    return all(isinstance(item, _PRIMITIVE_PY_TYPES) for item in values)


def _is_supported_field_schema(field_schema: Any) -> bool:
    if not isinstance(field_schema, dict):
        return False

    if "enum" in field_schema:
        return _is_primitive_enum(field_schema.get("enum"))

    for multi_key in ("oneOf", "anyOf"):
        if multi_key in field_schema:
            variants = field_schema.get(multi_key)
            if not isinstance(variants, list) or not variants:
                return False
            return all(_is_supported_field_schema(variant) for variant in variants)

    field_type = field_schema.get("type")
    if isinstance(field_type, str):
        return field_type in _PRIMITIVE_JSON_TYPES
    if isinstance(field_type, list):
        return bool(field_type) and all(t in _PRIMITIVE_JSON_TYPES for t in field_type)
    if field_type is None:
        # Allow "const" style primitive declarations.
        const_val = field_schema.get("const")
        if const_val is not None:
            return isinstance(const_val, _PRIMITIVE_PY_TYPES)
    return False


def validate_supported_requested_schema(schema: Any) -> None:
    if not isinstance(schema, dict):
        raise UnsupportedElicitationSchemaError("requestedSchema must be a JSON object")
    if schema.get("type") != "object":
        raise UnsupportedElicitationSchemaError(
            "Only object requestedSchema is supported."
        )
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise UnsupportedElicitationSchemaError(
            "requestedSchema.properties must be an object."
        )
    for name, field_schema in properties.items():
        if not _is_supported_field_schema(field_schema):
            raise UnsupportedElicitationSchemaError(
                "Unsupported requestedSchema field "
                f"'{name}'. Only primitive, enum, oneOf/anyOf(primitive) are allowed."
            )
    required = schema.get("required")
    if required is not None:
        if not isinstance(required, list) or not all(
            isinstance(item, str) for item in required
        ):
            raise UnsupportedElicitationSchemaError(
                "requestedSchema.required must be a list of field names."
            )


def _build_legacy_requested_schema(
    expected_responses: list[dict[str, Any]],
) -> dict[str, Any]:
    ids = [
        str(item.get("id"))
        for item in expected_responses
        if isinstance(item, dict) and item.get("id")
    ]
    if ids:
        return {
            "type": "object",
            "properties": {
                "selection": {
                    "type": "string",
                    "enum": ids,
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
            },
            "required": ["selection"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "response": {"type": "string"},
        },
        "required": ["response"],
        "additionalProperties": False,
    }


def canonicalize_elicitation_payload(
    raw: Optional[dict[str, Any]], fallback_message: str = ""
) -> dict[str, Any]:
    payload = _ensure_dict(raw)
    if payload.get("message") and isinstance(payload.get("requestedSchema"), dict):
        canonical = {
            "message": str(payload.get("message") or ""),
            "requestedSchema": payload.get("requestedSchema") or {},
            "meta": _ensure_dict(payload.get("meta")),
        }
        validate_supported_requested_schema(canonical["requestedSchema"])
        return canonical

    question = str(payload.get("question") or fallback_message or "").strip()
    expected = payload.get("expected_responses") or []
    if not isinstance(expected, list):
        expected = []

    canonical = {
        "message": question,
        "requestedSchema": _build_legacy_requested_schema(expected),
        "meta": {
            "expected_responses": expected,
            "legacy_format": True,
        },
    }
    validate_supported_requested_schema(canonical["requestedSchema"])
    return canonical


def _extract_expected_responses(payload: dict[str, Any]) -> list[dict[str, Any]]:
    expected = payload.get("expected_responses")
    if isinstance(expected, list):
        return expected
    meta = _ensure_dict(payload.get("meta"))
    expected = meta.get("expected_responses")
    if isinstance(expected, list):
        return expected
    return []


def _coerce_scalar_value(text: str, field_schema: dict[str, Any]) -> Any:
    stripped = text.strip()
    enum_values = field_schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        for enum_value in enum_values:
            if isinstance(enum_value, bool):
                if stripped.lower() == str(enum_value).lower():
                    return enum_value
            elif str(enum_value) == stripped:
                return enum_value
        return stripped

    for multi_key in ("oneOf", "anyOf"):
        options = field_schema.get(multi_key)
        if isinstance(options, list):
            for option in options:
                try:
                    candidate = _coerce_scalar_value(stripped, _ensure_dict(option))
                    jsonschema.validate(candidate, option)
                    return candidate
                except Exception:
                    continue

    field_type = field_schema.get("type")
    if isinstance(field_type, list):
        for item in field_type:
            candidate_schema = dict(field_schema)
            candidate_schema["type"] = item
            try:
                candidate = _coerce_scalar_value(stripped, candidate_schema)
                jsonschema.validate(candidate, candidate_schema)
                return candidate
            except Exception:
                continue
        return stripped

    if field_type == "boolean":
        lower = stripped.lower()
        if lower in {"true", "1", "yes", "on"}:
            return True
        if lower in {"false", "0", "no", "off"}:
            return False
        return stripped
    if field_type == "integer":
        try:
            return int(stripped)
        except ValueError:
            return stripped
    if field_type == "number":
        try:
            return float(stripped)
        except ValueError:
            return stripped
    if field_type == "null":
        return None
    return stripped


def _coerce_feedback_content_from_text(
    text: str, requested_schema: dict[str, Any]
) -> dict[str, Any]:
    properties = requested_schema.get("properties") or {}
    if not isinstance(properties, dict):
        return {"response": text}
    if not properties:
        return {}

    required = requested_schema.get("required") or []
    candidate_key: Optional[str] = None
    if isinstance(required, list) and len(required) == 1 and required[0] in properties:
        candidate_key = required[0]
    elif "response" in properties:
        candidate_key = "response"
    elif "selection" in properties:
        candidate_key = "selection"
    elif len(properties) == 1:
        candidate_key = next(iter(properties.keys()))

    if not candidate_key:
        return {"response": text}

    field_schema = _ensure_dict(properties.get(candidate_key))
    coerced_value = _coerce_scalar_value(text, field_schema)
    return {candidate_key: coerced_value}


def parse_feedback_response(
    raw_feedback: str, requested_schema: dict[str, Any]
) -> dict[str, Any]:
    content = parse_user_feedback_tag(raw_feedback) or ""
    lowered = content.strip().lower()
    if lowered in {"cancel", "dismiss"}:
        return {"action": "cancel", "content": None}
    if lowered in {"decline", "reject", "no"}:
        return {"action": "decline", "content": None}

    parsed_json = None
    if content.startswith("{") and content.endswith("}"):
        try:
            parsed_json = json.loads(content)
        except json.JSONDecodeError:
            parsed_json = None

    if isinstance(parsed_json, dict):
        action = str(parsed_json.get("action") or "accept").lower()
        if action in {"decline", "cancel"}:
            return {"action": action, "content": None}
        inner_content = parsed_json.get("content")
        if isinstance(inner_content, dict):
            payload = inner_content
        else:
            payload = parsed_json
            payload.pop("action", None)
        jsonschema.validate(payload, requested_schema)
        return {"action": "accept", "content": payload}

    payload = _coerce_feedback_content_from_text(content, requested_schema)
    jsonschema.validate(payload, requested_schema)
    return {"action": "accept", "content": payload}


class ElicitationCoordinator:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pending: dict[str, PendingElicitation] = {}
        self._queued_feedback: dict[str, str] = {}
        self._queued_responses: dict[str, dict[str, Any]] = {}

    def has_pending(self, session_key: Optional[str]) -> bool:
        if not session_key:
            return False
        with self._lock:
            return session_key in self._pending

    def get_pending(self, session_key: Optional[str]) -> Optional[dict[str, Any]]:
        if not session_key:
            return None
        with self._lock:
            pending = self._pending.get(session_key)
            return dict(pending.payload) if pending else None

    def submit_feedback(self, session_key: Optional[str], feedback: str) -> bool:
        if not session_key or not feedback:
            return False
        with self._lock:
            if session_key not in self._pending:
                return False
            self._queued_feedback[session_key] = feedback
            return True

    def submit_response(
        self, session_key: Optional[str], response: dict[str, Any]
    ) -> bool:
        if not session_key or not isinstance(response, dict):
            return False
        with self._lock:
            if session_key not in self._pending:
                return False
            self._queued_responses[session_key] = response
            return True

    def _consume_response(
        self, session_key: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        explicit = self._queued_responses.pop(session_key, None)
        if isinstance(explicit, dict):
            action = str(explicit.get("action") or "accept").lower()
            if action in {"decline", "cancel"}:
                return {"action": action, "content": None}
            content = explicit.get("content")
            if content is None:
                content = {}
            if not isinstance(content, dict):
                raise InvalidUserFeedbackError(
                    "Feedback content must be an object for accept action.",
                    session_key=session_key,
                    payload=payload,
                )
            jsonschema.validate(content, payload.get("requestedSchema") or {})
            return {"action": "accept", "content": content}

        raw_feedback = self._queued_feedback.pop(session_key, None)
        if raw_feedback is None:
            raise ElicitationRequiredError(
                payload,
                session_key=session_key,
                requires_session=False,
            )
        try:
            return parse_feedback_response(
                raw_feedback, payload.get("requestedSchema") or {}
            )
        except jsonschema.ValidationError as exc:
            raise InvalidUserFeedbackError(
                f"Invalid user feedback for schema: {exc.message}",
                session_key=session_key,
                payload=payload,
            ) from exc

    def resolve_or_raise(
        self, *, session_key: Optional[str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        canonical = canonicalize_elicitation_payload(payload)
        if not session_key:
            raise ElicitationRequiredError(
                canonical,
                session_key=None,
                requires_session=True,
            )

        with self._lock:
            self._pending[session_key] = PendingElicitation(
                payload=canonical,
                created_at=time.time(),
            )
            response = self._consume_response(session_key, canonical)
            if response.get("action") in {"accept", "decline", "cancel"}:
                self._pending.pop(session_key, None)
            return response

    @staticmethod
    def to_client_payload(exc: ElicitationRequiredError) -> dict[str, Any]:
        return exc.to_client_payload()

    @staticmethod
    def extract_expected_responses(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return _extract_expected_responses(payload)


_COORDINATOR = ElicitationCoordinator()


def get_elicitation_coordinator() -> ElicitationCoordinator:
    return _COORDINATOR
