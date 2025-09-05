import os
import json
import pytest
from fastapi import HTTPException
from app.session_manager import prompt_helper


class DummyCall:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    async def __call__(self, prompt_name, args):
        if self.error:
            raise self.error
        return self.result


class DummyListPrompts:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    async def __call__(self):
        if self.error:
            raise self.error
        return self.result


@pytest.fixture(autouse=True)
def clear_env():
    os.environ.pop("SYSTEM_DEFINED_PROMPTS", None)
    yield
    os.environ.pop("SYSTEM_DEFINED_PROMPTS", None)


@pytest.fixture
def system_prompts():
    prompts = [
        {
            "name": "greeting",
            "title": "Hello You",
            "description": "Get a personalized greeting.",
            "arguments": [{"name": "name"}],
            "template": {"role": "system", "content": "Hello, {name}!"},
        }
    ]
    os.environ["SYSTEM_DEFINED_PROMPTS"] = json.dumps(prompts)
    return [prompt_helper.Prompt(**p) for p in prompts]


def test_default_prompt_list_result(system_prompts):
    result = prompt_helper.default_prompt_list_result(system_prompts)
    assert "prompts" in result
    assert result["prompts"][0]["name"] == "greeting"
    assert "template" in result["prompts"][0]
    assert result["prompts"][0]["template"]["content"] == "Hello, {name}!"


def test_system_defined_prompts(system_prompts):
    prompts = prompt_helper.system_defined_prompts()
    assert len(prompts) == 1
    assert prompts[0].name == "greeting"


@pytest.mark.asyncio
async def test_list_prompts_merges_system_prompts(system_prompts):
    class ListPromptsResult:
        def __init__(self):
            self.prompts = []

    async def list_prompts():
        return ListPromptsResult()

    result = await prompt_helper.list_prompts(list_prompts)
    # result is ListPromptsResult, check its .prompts attribute
    assert any(p["name"] == "greeting" for p in result.prompts)


@pytest.mark.asyncio
async def test_list_prompts_mcp_error_method_not_found(system_prompts):
    class McpError(Exception):
        pass

    McpError.__name__ = "McpError"

    async def list_prompts():
        raise McpError("Method not found")

    result = await prompt_helper.list_prompts(list_prompts)
    assert any(p["name"] == "greeting" for p in result["prompts"])


@pytest.mark.asyncio
async def test_list_prompts_mcp_error_method_not_found_no_system():
    async def list_prompts():
        class McpError(Exception):
            pass

        McpError.__name__ = "McpError"
        raise McpError("Method not found")

    with pytest.raises(HTTPException) as exc:
        await prompt_helper.list_prompts(list_prompts)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_list_prompts_other_error():
    async def list_prompts():
        raise Exception("Other error")

    with pytest.raises(HTTPException) as exc:
        await prompt_helper.list_prompts(list_prompts)
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_call_prompt_system_defined(system_prompts):
    args = {"name": "World"}
    result = await prompt_helper.call_prompt(lambda n, a: None, "greeting", args)
    print(f"Result: {result}")
    assert result.description == "Get a personalized greeting."
    assert not result.isError
    assert result.messages[0].content.text == "Hello, World!"


@pytest.mark.asyncio
async def test_call_prompt_delegates_to_call():
    async def call(prompt_name, args):
        return "called"

    result = await prompt_helper.call_prompt(call, "not_found", {})
    assert result == "called"
