import pytest
from fastapi import HTTPException
from types import SimpleNamespace
from app.oauth.decorator import decorate_args_with_oauth_token


class DummyTool:
    def __init__(self, name, input_schema):
        self.name = name
        self.inputSchema = input_schema


class DummyTools:
    def __init__(self, tools):
        self.tools = tools


@pytest.mark.asyncio
async def test_adds_oauth_token_when_required():
    input_schema = {"properties": {"oauth_token": {"type": "string"}}}
    tool = DummyTool("test_tool", input_schema)
    tools = DummyTools([tool])
    args = {"foo": "bar"}
    result = await decorate_args_with_oauth_token(
        tools, "test_tool", args.copy(), "token123"
    )
    assert result["oauth_token"] == "token123"
    assert result["foo"] == "bar"


@pytest.mark.asyncio
async def test_raises_when_oauth_token_missing():
    input_schema = {"properties": {"oauth_token": {"type": "string"}}}
    tool = DummyTool("test_tool", input_schema)
    tools = DummyTools([tool])
    with pytest.raises(HTTPException) as exc:
        await decorate_args_with_oauth_token(tools, "test_tool", {}, None)
    assert exc.value.status_code == 401
    assert "requires oauth_token" in exc.value.detail


@pytest.mark.asyncio
async def test_does_not_add_oauth_token_when_not_required():
    input_schema = {"properties": {"other_field": {"type": "string"}}}
    tool = DummyTool("test_tool", input_schema)
    tools = DummyTools([tool])
    args = {"foo": "bar"}
    result = await decorate_args_with_oauth_token(tools, "test_tool", args.copy(), None)
    assert "oauth_token" not in result
    assert result["foo"] == "bar"


@pytest.mark.asyncio
async def test_tool_has_no_input_schema():
    tool = DummyTool("test_tool", None)
    tools = DummyTools([tool])
    args = {"foo": "bar"}
    result = await decorate_args_with_oauth_token(tools, "test_tool", args.copy(), None)
    assert result == args


@pytest.mark.asyncio
async def test_tool_not_found():
    tool = DummyTool("other_tool", {"properties": {}})
    tools = DummyTools([tool])
    args = {"foo": "bar"}
    result = await decorate_args_with_oauth_token(tools, "test_tool", args.copy(), None)
    assert result == args


@pytest.mark.asyncio
async def test_args_none_initializes_dict():
    input_schema = {"properties": {"oauth_token": {"type": "string"}}}
    tool = DummyTool("test_tool", input_schema)
    tools = DummyTools([tool])
    result = await decorate_args_with_oauth_token(tools, "test_tool", None, "token123")
    assert result["oauth_token"] == "token123"
    assert isinstance(result, dict)
