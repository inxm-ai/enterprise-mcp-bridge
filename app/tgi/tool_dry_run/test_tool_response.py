import pytest
from jsonschema import ValidationError
from app.tgi.tool_dry_run.tool_response import get_tool_dry_run_response


def test_throws_if_tgi_url_unset(monkeypatch):
    # ensure TGI_URL is not set in the environment
    monkeypatch.delenv("TGI_URL", raising=False)

    with pytest.raises(ValueError, match="TGI_URL environment variable is not set"):
        get_tool_dry_run_response("some_tool", {})


def test_raises_on_invalid_tool_input(monkeypatch):
    # set TGI_URL so the function proceeds to schema validation
    monkeypatch.setenv("TGI_URL", "http://example")

    tool = {
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
    }

    invalid_input = {}

    with pytest.raises(ValidationError):
        get_tool_dry_run_response(tool, invalid_input)


def test_accepts_valid_tool_input(monkeypatch):
    # set TGI_URL so the function proceeds to schema validation
    monkeypatch.setenv("TGI_URL", "http://example")

    tool = {
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
    }

    valid_input = {"name": "Alice"}

    # Should not raise, and current implementation returns None
    assert get_tool_dry_run_response(tool, valid_input) is None
