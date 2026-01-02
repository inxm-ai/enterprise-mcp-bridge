
from app.tgi.models.model_formats import (
    ChatGPTModelFormat,
    get_model_format_for,
)
from app.tgi.models import ChatCompletionRequest


def test_chatgpt_model_format_maps_tool_choice_string_to_function_object():
    fmt = ChatGPTModelFormat()
    req = ChatCompletionRequest(messages=[], tool_choice="get_me")
    fmt.prepare_request(req)
    assert isinstance(req.tool_choice, dict)
    assert req.tool_choice.get("type") == "function"
    assert req.tool_choice.get("function", {}).get("name") == "get_me"


def test_chatgpt_model_format_preserves_control_strings():
    fmt = ChatGPTModelFormat()
    for val in ("auto", "none", "required", "Auto"):
        req = ChatCompletionRequest(messages=[], tool_choice=val)
        fmt.prepare_request(req)
        assert isinstance(req.tool_choice, str)
        assert req.tool_choice == val.lower()


def test_get_model_format_always_returns_chatgpt():
    fmt = get_model_format_for()
    assert isinstance(fmt, ChatGPTModelFormat)

    fmt = get_model_format_for("claude-3-haiku")
    assert isinstance(fmt, ChatGPTModelFormat)
