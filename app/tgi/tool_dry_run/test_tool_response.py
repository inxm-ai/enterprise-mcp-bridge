import pytest
from jsonschema import ValidationError
from app.tgi.tool_dry_run.tool_response import get_tool_dry_run_response

session = None  # Placeholder for MCPSessionBase


@pytest.mark.asyncio
async def test_throws_if_tgi_url_unset(monkeypatch):
    # ensure TGI_URL is not set in the environment
    monkeypatch.delenv("TGI_URL", raising=False)
    with pytest.raises(ValueError, match="TGI_URL environment variable is not set"):
        await get_tool_dry_run_response(session, "some_tool", {})


@pytest.mark.asyncio
async def test_raises_on_invalid_tool_input(monkeypatch):
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

    result = await get_tool_dry_run_response(session, tool, invalid_input)
    assert result is not None
    assert getattr(result, "isError", None) is True
    assert isinstance(result.content, list)
    assert (
        "Failed to validate input against schema: 'name' is a required property"
        in result.content[0].text
    )


@pytest.mark.asyncio
async def test_accepts_valid_tool_input(monkeypatch):
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

    # Patch PromptService and LLMClient so prompt lookup and streaming are mocked
    monkeypatch.setattr(
        "app.tgi.tool_dry_run.tool_response.PromptService", MockPromptService
    )
    monkeypatch.setattr("app.tgi.tool_dry_run.tool_response.LLMClient", DummyLLMClient)
    MockPromptService.returns = {}

    # Should not raise, and current implementation returns an MCP-style result
    result = await get_tool_dry_run_response(session, tool, valid_input)
    assert result is not None
    assert getattr(result, "isError", None) is False
    # Since no outputSchema was provided, the aggregated stream should be
    # available under content[0].text
    assert isinstance(result.content, list)
    assert "chunk" in result.content[0].text


class DummyLLMClient:
    """Mock LLMClient that records the request and returns preset chunks."""

    created_instance = None

    def __init__(self, *args, **kwargs):
        DummyLLMClient.created_instance = self
        self.stream_called = False
        # Example chunks that a real stream might return
        self.stream_return = ['data: {"chunk":1}\n\n', "data: [DONE]\n\n"]

    def stream_completion(self, request, token, span):
        # record that it was called and capture the request object
        self.stream_called = True
        self.last_request = request

        stream_source = self.stream_return
        if not hasattr(self.stream_return, "__aiter__"):
            # Wrap sync iterable into async generator
            async def _wrap_sync_iter(iterable):
                for item in iterable:
                    yield item

            stream_source = _wrap_sync_iter(self.stream_return)
        return stream_source


class MockPromptService:
    """Mock PromptService that returns values from a mapping and records calls."""

    last_instance = None
    returns = {}

    def __init__(self, *args, **kwargs):
        MockPromptService.last_instance = self
        self.calls = []

    async def find_prompt_by_name_or_role(self, session_arg, prompt_name=None):
        # Async version to match the real PromptService API
        self.calls.append((session_arg, prompt_name))
        return MockPromptService.returns.get(prompt_name)


def _make_tool():
    return {
        "name": "mytool",
        "inputSchema": {
            "type": "object",
            "properties": {"foo": {"type": "string"}},
            "required": ["foo"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {"res": {"type": "string"}},
            "required": ["res"],
        },
    }


@pytest.mark.asyncio
async def test_uses_specific_prompt_when_available(monkeypatch):
    monkeypatch.setenv("TGI_URL", "http://example")

    # Patch PromptService and LLMClient used inside the module under test
    monkeypatch.setattr(
        "app.tgi.tool_dry_run.tool_response.PromptService", MockPromptService
    )
    monkeypatch.setattr("app.tgi.tool_dry_run.tool_response.LLMClient", DummyLLMClient)

    # Configure prompt service to return a specific prompt for the tool
    MockPromptService.returns = {"dryrun_mytool": "Specific prompt content"}

    tool = _make_tool()
    valid_input = {"foo": "bar"}

    # Call under test
    result = await get_tool_dry_run_response(session, tool, valid_input)

    # Assertions
    client = DummyLLMClient.created_instance
    assert client is not None and client.stream_called
    # The system message should contain the prompt returned by the prompt service
    assert client.last_request.messages[0].content == "Specific prompt content"
    # The user message should mention the tool name and input
    assert "mytool" in client.last_request.messages[1].content
    assert "{'foo': 'bar'}" in client.last_request.messages[1].content
    # Ensure the prompt service was asked for the specific dryrun prompt only
    assert MockPromptService.last_instance.calls[0][1] == "dryrun_mytool"


@pytest.mark.asyncio
async def test_uses_default_prompt_when_specific_missing(monkeypatch):
    monkeypatch.setenv("TGI_URL", "http://example")
    monkeypatch.setattr(
        "app.tgi.tool_dry_run.tool_response.PromptService", MockPromptService
    )
    monkeypatch.setattr("app.tgi.tool_dry_run.tool_response.LLMClient", DummyLLMClient)

    # Specific missing, default available
    MockPromptService.returns = {"dryrun_default": "Default prompt content"}

    tool = _make_tool()
    valid_input = {"foo": "bar"}

    result = await get_tool_dry_run_response(session, tool, valid_input)

    client = DummyLLMClient.created_instance
    assert client is not None and client.stream_called
    # Should have tried specific first, then default
    assert MockPromptService.last_instance.calls[0][1] == "dryrun_mytool"
    assert MockPromptService.last_instance.calls[1][1] == "dryrun_default"
    assert client.last_request.messages[0].content == "Default prompt content"


@pytest.mark.asyncio
async def test_uses_hardcoded_prompt_when_none_available(monkeypatch):
    monkeypatch.setenv("TGI_URL", "http://example")
    monkeypatch.setattr(
        "app.tgi.tool_dry_run.tool_response.PromptService", MockPromptService
    )
    monkeypatch.setattr("app.tgi.tool_dry_run.tool_response.LLMClient", DummyLLMClient)

    # Nothing available from PromptService
    MockPromptService.returns = {}

    tool = _make_tool()
    valid_input = {"foo": "bar"}

    result = await get_tool_dry_run_response(session, tool, valid_input)

    client = DummyLLMClient.created_instance
    assert client is not None and client.stream_called
    # Should have tried specific then default
    assert MockPromptService.last_instance.calls[0][1] == "dryrun_mytool"
    assert MockPromptService.last_instance.calls[1][1] == "dryrun_default"
    # Final fallback is a hardcoded string
    assert (
        client.last_request.messages[0].content
        == "You are a helpful assistant that provides mock responses for tools."
    )
