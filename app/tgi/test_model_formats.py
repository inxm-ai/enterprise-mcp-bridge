import pytest

from app.tgi import model_formats
from app.tgi.model_formats import (
    ChatGPTModelFormat,
    ClaudeModelFormat,
    get_model_format_for,
)


@pytest.mark.parametrize(
    "env_value,expected",
    [
        ("", ChatGPTModelFormat),
        ("chat-gpt/v1", ChatGPTModelFormat),
        ("claude/v1", ClaudeModelFormat),
    ],
)
def test_get_model_format_respects_env(monkeypatch, env_value, expected):
    monkeypatch.setattr(model_formats, "TGI_MODEL_FORMAT", env_value)
    monkeypatch.setattr(model_formats, "TOOL_INJECTION_MODE", "openai")

    fmt = get_model_format_for()
    assert isinstance(fmt, expected)


def test_get_model_format_prefers_model_name(monkeypatch):
    monkeypatch.setattr(model_formats, "TGI_MODEL_FORMAT", "")
    monkeypatch.setattr(model_formats, "TOOL_INJECTION_MODE", "openai")

    fmt = get_model_format_for("claude-3-haiku")
    assert isinstance(fmt, ClaudeModelFormat)


def test_legacy_tool_injection_mode(monkeypatch):
    monkeypatch.setattr(model_formats, "TGI_MODEL_FORMAT", "")
    monkeypatch.setattr(model_formats, "TOOL_INJECTION_MODE", "claude")

    fmt = get_model_format_for()
    assert isinstance(fmt, ClaudeModelFormat)
