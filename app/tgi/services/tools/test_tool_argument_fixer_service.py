import json
from app.tgi.services.tools.tool_argument_fixer_service import fix_tool_arguments
from app.tgi.models import ToolCall, ToolCallFunction


def make_tool_call(name, arguments):
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return ToolCall(
        id="test-id", function=ToolCallFunction(name=name, arguments=arguments)
    )


def make_tool_definition(name, properties):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": properties},
        },
    }


def test_fix_tool_arguments_valid():
    tool_def = make_tool_definition(
        "my_tool", {"foo": {"type": "string"}, "bar": {"type": "integer"}}
    )
    tool_call = make_tool_call("my_tool", {"foo": "abc", "bar": 123})
    fixed = fix_tool_arguments(tool_call, [tool_def])
    args = json.loads(fixed.function.arguments)
    assert args == {"foo": "abc", "bar": 123}


def test_fix_tool_arguments_missing_tool():
    tool_call = make_tool_call("unknown_tool", {"foo": "abc"})
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
    tool_call = make_tool_call("my_tool", {"foo": "abc", "extra": 1})
    fixed = fix_tool_arguments(tool_call, [tool_def])
    args = json.loads(fixed.function.arguments)
    assert args == {"foo": "abc"}  # 'extra' should be dropped


# --- Tests for argument name confusion (expected to fail) ---
def test_fix_tool_arguments_camel_to_snake():
    tool_def = make_tool_definition("my_tool", {"argument_name": {"type": "string"}})
    tool_call = make_tool_call("my_tool", {"argumentName": "value"})
    fixed = fix_tool_arguments(tool_call, [tool_def])
    args = json.loads(fixed.function.arguments)
    assert args == {"argument_name": "value"}


def test_fix_tool_arguments_snake_to_camel():
    tool_def = make_tool_definition("my_tool", {"argumentName": {"type": "string"}})
    tool_call = make_tool_call("my_tool", {"argument_name": "value"})
    fixed = fix_tool_arguments(tool_call, [tool_def])
    args = json.loads(fixed.function.arguments)
    assert args == {"argumentName": "value"}


def test_fix_tool_arguments_case_insensitive():
    tool_def = make_tool_definition("my_tool", {"ArgumentName": {"type": "string"}})
    tool_call = make_tool_call("my_tool", {"argumentname": "value"})
    fixed = fix_tool_arguments(tool_call, [tool_def])
    args = json.loads(fixed.function.arguments)
    assert args == {"ArgumentName": "value"}


def test_fix_tool_arguments_in_deeply_nested():
    tool_def = make_tool_definition(
        "my_tool",
        {
            "outer": {
                "type": "object",
                "properties": {"innerArgument": {"type": "string"}},
            }
        },
    )
    tool_call = make_tool_call("my_tool", {"outer": {"inner_argument": "value"}})
    fixed = fix_tool_arguments(tool_call, [tool_def])
    args = json.loads(fixed.function.arguments)
    assert args == {"outer": {"innerArgument": "value"}}


# --- Additional edge case tests ---
def test_fix_tool_arguments_nested_extra_argument():
    tool_def = make_tool_definition(
        "my_tool",
        {"outer": {"type": "object", "properties": {"inner": {"type": "string"}}}},
    )
    tool_call = make_tool_call(
        "my_tool", {"outer": {"inner": "v", "extra": 1}, "extra2": 2}
    )
    fixed = fix_tool_arguments(tool_call, [tool_def])
    args = json.loads(fixed.function.arguments)
    assert args == {"outer": {"inner": "v"}}


def test_fix_tool_arguments_type_coercion():
    tool_def = make_tool_definition("my_tool", {"foo": {"type": "integer"}})
    tool_call = make_tool_call("my_tool", {"foo": "123"})
    fixed = fix_tool_arguments(tool_call, [tool_def])
    args = json.loads(fixed.function.arguments)
    assert args == {"foo": 123}


def test_fix_tool_arguments_null_and_none():
    tool_def = make_tool_definition("my_tool", {"foo": {"type": "string"}})
    tool_call = make_tool_call("my_tool", {"foo": None})
    fixed = fix_tool_arguments(tool_call, [tool_def])
    args = json.loads(fixed.function.arguments)
    assert args == {"foo": None}


def test_fix_tool_arguments_empty_arguments():
    tool_def = make_tool_definition("my_tool", {"foo": {"type": "string"}})
    tool_call = make_tool_call("my_tool", {})
    fixed = fix_tool_arguments(tool_call, [tool_def])
    args = json.loads(fixed.function.arguments)
    assert args == {}


def test_fix_tool_arguments_array_argument():
    tool_def = make_tool_definition(
        "my_tool", {"items": {"type": "array", "items": {"type": "string"}}}
    )
    tool_call = make_tool_call("my_tool", {"items": ["a", "b"]})
    fixed = fix_tool_arguments(tool_call, [tool_def])
    args = json.loads(fixed.function.arguments)
    assert args == {"items": ["a", "b"]}
