import pytest
from .tools_map import map_tools


def test_map_tools():
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a given location.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
            },
        }
    ]
    result = map_tools(tools)
    assert isinstance(result, list)
    expected = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                        }
                    },
                },
            },
        }
    ]
    assert result == expected


def test_map_tools_basic_inlining():
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a given location.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"$ref": "#/$defs/TemperatureUnit"},
                },
                "$defs": {
                    "TemperatureUnit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    }
                },
            },
        }
    ]
    result = map_tools(tools)
    assert isinstance(result, list)
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "get_weather"
    params = result[0]["function"]["parameters"]
    assert "unit" in params["properties"]
    assert params["properties"]["unit"]["type"] == "string"
    assert params["properties"]["unit"]["enum"] == ["celsius", "fahrenheit"]
    assert "$defs" not in params


def test_map_tools_nested_ref():
    tools = [
        {
            "name": "nested_ref_tool",
            "description": "Tool with nested $ref.",
            "inputSchema": {
                "type": "object",
                "properties": {"outer": {"$ref": "#/$defs/Outer"}},
                "$defs": {
                    "Outer": {
                        "type": "object",
                        "properties": {"inner": {"$ref": "#/$defs/Inner"}},
                    },
                    "Inner": {"type": "string", "enum": ["foo", "bar"]},
                },
            },
        }
    ]
    result = map_tools(tools)
    params = result[0]["function"]["parameters"]
    assert "outer" in params["properties"]
    assert "inner" in params["properties"]["outer"]["properties"]
    assert params["properties"]["outer"]["properties"]["inner"]["type"] == "string"
    assert params["properties"]["outer"]["properties"]["inner"]["enum"] == [
        "foo",
        "bar",
    ]


def test_map_tools_cyclic_ref():
    tools = [
        {
            "name": "cyclic_ref_tool",
            "description": "Tool with cyclic $ref.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "a": {"$ref": "#/$defs/B"},
                },
                "$defs": {
                    "B": {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}}
                },
            },
        }
    ]
    result = map_tools(tools)
    params = result[0]["function"]["parameters"]
    # Cyclic ref should be replaced with {}
    assert "a" in params["properties"]
    assert "b" in params["properties"]["a"]["properties"]
    assert params["properties"]["a"]["properties"]["b"] == {}


def test_map_tools_missing_ref_raises():
    tools = [
        {
            "name": "missing_ref_tool",
            "description": "Tool with missing $ref.",
            "inputSchema": {
                "type": "object",
                "properties": {"foo": {"$ref": "#/$defs/NotFound"}},
                "$defs": {},
            },
        }
    ]
    with pytest.raises(ValueError):
        map_tools(tools)


def test_map_tools_no_input_schema():
    tools = [{"name": "no_input", "description": "Tool with no inputSchema."}]
    result = map_tools(tools)
    params = result[0]["function"]["parameters"]
    assert params == {}


def test_map_tools_list_with_ref():
    tools = [
        {
            "name": "list_ref_tool",
            "description": "Tool with list containing $ref.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"$ref": "#/$defs/ItemType"}}
                },
                "$defs": {"ItemType": {"type": "string", "enum": ["x", "y"]}},
            },
        }
    ]
    result = map_tools(tools)
    params = result[0]["function"]["parameters"]
    assert "items" in params["properties"]
    assert params["properties"]["items"]["items"]["type"] == "string"
    assert params["properties"]["items"]["items"]["enum"] == ["x", "y"]
