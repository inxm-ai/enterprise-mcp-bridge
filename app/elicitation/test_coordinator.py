import json

import pytest

from app.elicitation.coordinator import (
    ElicitationCoordinator,
    ElicitationRequiredError,
    InvalidUserFeedbackError,
    UnsupportedElicitationSchemaError,
    parse_feedback_response,
    validate_supported_requested_schema,
)


def _requested_schema(field_schema: dict) -> dict:
    return {
        "type": "object",
        "properties": {"value": field_schema},
        "required": ["value"],
        "additionalProperties": False,
    }


@pytest.mark.parametrize(
    ("field_schema", "value"),
    [
        ({"type": "string"}, "hello"),
        ({"type": "number"}, 3.14),
        ({"type": "integer"}, 7),
        ({"type": "boolean"}, True),
        ({"enum": ["a", "b"]}, "a"),
        ({"enum": [1, 2, 3]}, 2),
        ({"oneOf": [{"type": "string"}, {"type": "number"}]}, "x"),
        ({"anyOf": [{"type": "boolean"}, {"type": "integer"}]}, False),
    ],
)
def test_supported_schema_subset_accepts_validation_and_feedback(field_schema, value):
    schema = _requested_schema(field_schema)
    validate_supported_requested_schema(schema)

    raw_feedback = json.dumps({"action": "accept", "content": {"value": value}})
    parsed = parse_feedback_response(raw_feedback, schema)
    assert parsed["action"] == "accept"
    assert parsed["content"]["value"] == value


@pytest.mark.parametrize(
    "field_schema",
    [
        {"type": "object", "properties": {"nested": {"type": "string"}}},
        {"type": "array", "items": {"type": "string"}},
        {"oneOf": [{"type": "string"}, {"type": "object", "properties": {}}]},
        {"anyOf": [{"type": "number"}, {"type": "array", "items": {"type": "number"}}]},
    ],
)
def test_complex_schema_subset_is_rejected(field_schema):
    with pytest.raises(UnsupportedElicitationSchemaError):
        validate_supported_requested_schema(_requested_schema(field_schema))


def test_coordinator_requires_feedback_then_accepts_resume():
    coordinator = ElicitationCoordinator()
    payload = {
        "message": "Pick",
        "requestedSchema": _requested_schema({"type": "string"}),
    }

    with pytest.raises(ElicitationRequiredError):
        coordinator.resolve_or_raise(session_key="sess-1", payload=payload)

    assert coordinator.submit_feedback(
        "sess-1", "<user_feedback>approved</user_feedback>"
    )
    result = coordinator.resolve_or_raise(session_key="sess-1", payload=payload)
    assert result == {"action": "accept", "content": {"value": "approved"}}


def test_coordinator_rejects_invalid_feedback_content():
    coordinator = ElicitationCoordinator()
    payload = {
        "message": "Pick",
        "requestedSchema": _requested_schema({"type": "integer"}),
    }

    with pytest.raises(ElicitationRequiredError):
        coordinator.resolve_or_raise(session_key="sess-2", payload=payload)

    assert coordinator.submit_feedback(
        "sess-2", "<user_feedback>not-a-number</user_feedback>"
    )
    with pytest.raises(InvalidUserFeedbackError):
        coordinator.resolve_or_raise(session_key="sess-2", payload=payload)
