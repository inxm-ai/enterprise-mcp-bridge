import json
from app.tgi.tool_argument_fixer_service import fix_tool_arguments
from app.tgi.models import ToolCall, ToolCallFunction


def make_tool_call(name, arguments):
    # Always serialize arguments to string for ToolCallFunction
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return ToolCall(
        id="test-id", function=ToolCallFunction(name=name, arguments=arguments)
    )


def make_tool_definition(name, properties):
    return {"name": name, "parameters": {"type": "object", "properties": properties}}


def test_fix_tool_arguments_valid():
    tool_def = make_tool_definition(
        "my_tool", {"foo": {"type": "string"}, "bar": {"type": "integer"}}
    )
    tool_call = make_tool_call("my_tool", json.dumps({"foo": "abc", "bar": 123}))
    fixed = fix_tool_arguments(tool_call, [tool_def])
    args = json.loads(fixed.function.arguments)
    assert args == {"foo": "abc", "bar": 123}


def test_fix_tool_arguments_missing_tool():
    tool_call = make_tool_call("unknown_tool", json.dumps({"foo": "abc"}))
    fixed = fix_tool_arguments(tool_call, [])
    assert fixed == tool_call


def test_fix_tool_arguments_invalid_json():
    tool_def = make_tool_definition("my_tool", {"foo": {"type": "string"}})
    tool_call = make_tool_call("my_tool", "not a json string")
    fixed = fix_tool_arguments(tool_call, [tool_def])
    assert fixed == tool_call


def test_fix_tool_arguments_dict_input():
    tool_def = make_tool_definition("my_tool", {"foo": {"type": "string"}})
    tool_call = make_tool_call("my_tool", {"foo": "abc"})
    fixed = fix_tool_arguments(tool_call, [tool_def])
    args = json.loads(fixed.function.arguments)
    assert args == {"foo": "abc"}


def test_fix_tool_arguments_extra_argument():
    tool_def = make_tool_definition("my_tool", {"foo": {"type": "string"}})
    tool_call = make_tool_call("my_tool", json.dumps({"foo": "abc", "extra": 1}))
    fixed = fix_tool_arguments(tool_call, [tool_def])
    args = json.loads(fixed.function.arguments)
    assert args == {"foo": "abc"}  # 'extra' should be dropped


# --- Tests for argument name confusion (expected to fail) ---
def test_fix_tool_arguments_camel_to_snake():
    tool_def = make_tool_definition("my_tool", {"argument_name": {"type": "string"}})
    tool_call = make_tool_call("my_tool", json.dumps({"argumentName": "value"}))
    fixed = fix_tool_arguments(tool_call, [tool_def])
    args = json.loads(fixed.function.arguments)
    # This will fail until the function is improved to map camelCase to snake_case
    assert args == {"argument_name": "value"}


def test_fix_tool_arguments_snake_to_camel():
    tool_def = make_tool_definition("my_tool", {"argumentName": {"type": "string"}})
    tool_call = make_tool_call("my_tool", json.dumps({"argument_name": "value"}))
    fixed = fix_tool_arguments(tool_call, [tool_def])
    args = json.loads(fixed.function.arguments)
    # This will fail until the function is improved to map snake_case to camelCase
    assert args == {"argumentName": "value"}
