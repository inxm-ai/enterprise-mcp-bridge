import pytest
from fastapi import HTTPException
from app.oauth.decorator import decorate_args_with_oauth_token


class DummyTool:
    def __init__(self, name, input_schema):
        self.name = name
        self.inputSchema = input_schema


class DummyTools:
    def __init__(self, tools):
        self.tools = tools


@pytest.mark.asyncio
async def test_adds_oauth_token_when_required(mock_token_retriever_factory):
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
async def test_args_none_initializes_dict(mock_token_retriever_factory):
    input_schema = {"properties": {"oauth_token": {"type": "string"}}}
    tool = DummyTool("test_tool", input_schema)
    tools = DummyTools([tool])
    result = await decorate_args_with_oauth_token(tools, "test_tool", None, "token123")
    assert result["oauth_token"] == "token123"
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_no_oauth_token_when_forwarding_disabled_by_env(monkeypatch):
    """A standalone bridge with MCP_FORWARD_ACCESS_TOKEN=false never injects
    the caller's token, and doesn't fail on the tool's oauth_token either."""
    monkeypatch.setenv("MCP_FORWARD_ACCESS_TOKEN", "false")
    input_schema = {"properties": {"oauth_token": {"type": "string"}}}
    tools = DummyTools([DummyTool("test_tool", input_schema)])

    result = await decorate_args_with_oauth_token(
        tools, "test_tool", {"foo": "bar"}, "token123"
    )

    assert result == {"foo": "bar"}


class FailingRetriever:
    def __init__(self):
        self.calls = 0

    def retrieve_token(self, token):
        self.calls += 1
        raise RuntimeError("broker refused: no read-token")


@pytest.fixture
def broker_alias_with_failing_exchange(monkeypatch):
    """Keycloak with a provider alias whose broker exchange always fails, as
    on a hosted install where nobody holds broker read-token."""
    from app.oauth import decorator

    retriever = FailingRetriever()
    monkeypatch.setattr(decorator, "AUTH_PROVIDER", "keycloak")
    monkeypatch.setattr(decorator, "KEYCLOAK_PROVIDER_ALIAS", "microsoft")
    monkeypatch.setattr(
        decorator,
        "TokenRetrieverFactory",
        lambda: type("F", (), {"get": lambda self: retriever})(),
    )
    return retriever


@pytest.mark.asyncio
async def test_tool_without_oauth_token_skips_the_exchange(
    broker_alias_with_failing_exchange,
):
    tools = DummyTools(
        [DummyTool("search_nodes", {"properties": {"query": {"type": "string"}}})]
    )

    result = await decorate_args_with_oauth_token(
        tools, "search_nodes", {"query": "x"}, "kc-token"
    )

    assert result == {"query": "x"}
    assert broker_alias_with_failing_exchange.calls == 0


@pytest.mark.asyncio
async def test_tool_with_oauth_token_still_exchanges(
    broker_alias_with_failing_exchange,
):
    tools = DummyTools(
        [DummyTool("upload", {"properties": {"oauth_token": {"type": "string"}}})]
    )

    with pytest.raises(RuntimeError):
        await decorate_args_with_oauth_token(tools, "upload", {}, "kc-token")
    assert broker_alias_with_failing_exchange.calls == 1
