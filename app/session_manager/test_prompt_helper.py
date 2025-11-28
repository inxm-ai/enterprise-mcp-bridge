import json
import os
import pytest

from fastapi import HTTPException

from app.session_manager.prompt_helper import call_prompt


@pytest.mark.asyncio
async def test_call_prompt_with_content(monkeypatch):
    monkeypatch.setenv(
        "SYSTEM_DEFINED_PROMPTS",
        json.dumps(
            [
                {
                    "name": "greeting_content",
                    "title": "Greeting",
                    "description": "A greeting",
                    "arguments": [{"name": "name"}],
                    "template": {"role": "assistant", "content": "Hello {name}!"},
                }
            ]
        ),
    )

    res = await call_prompt(None, "greeting_content", {"name": "Alice"})
    assert not res.isError
    assert res.messages[0].content.text == "Hello Alice!"


@pytest.mark.asyncio
async def test_call_prompt_with_file(monkeypatch, tmp_path):
    # Create a test file under the app directory (relative path expected)
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_dir = os.path.join(app_dir, "session_manager", "test_files")
    os.makedirs(test_dir, exist_ok=True)
    file_path = os.path.join(test_dir, "greet.txt")
    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write("File hello {name} from file!")

    rel_path = os.path.relpath(file_path, app_dir)

    monkeypatch.setenv(
        "SYSTEM_DEFINED_PROMPTS",
        json.dumps(
            [
                {
                    "name": "greeting_file",
                    "title": "GreetingFile",
                    "description": "A greeting from file",
                    "arguments": [{"name": "name"}],
                    "template": {"role": "assistant", "file": rel_path},
                }
            ]
        ),
    )

    res = await call_prompt(None, "greeting_file", {"name": "Bob"})
    assert not res.isError
    assert res.messages[0].content.text == "File hello Bob from file!"


@pytest.mark.asyncio
async def test_file_precedence_over_content(monkeypatch):
    # Create a file and also provide content; file should take precedence
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_dir = os.path.join(app_dir, "session_manager", "test_files")
    os.makedirs(test_dir, exist_ok=True)
    file_path = os.path.join(test_dir, "precedence.txt")
    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write("FromFile {name}")
    rel_path = os.path.relpath(file_path, app_dir)

    monkeypatch.setenv(
        "SYSTEM_DEFINED_PROMPTS",
        json.dumps(
            [
                {
                    "name": "greeting_both",
                    "title": "Both",
                    "description": "Both file and content",
                    "arguments": [{"name": "name"}],
                    "template": {
                        "role": "assistant",
                        "file": rel_path,
                        "content": "FromContent {name}",
                    },
                }
            ]
        ),
    )

    res = await call_prompt(None, "greeting_both", {"name": "Carol"})
    assert not res.isError
    assert res.messages[0].content.text == "FromFile Carol"


@pytest.mark.asyncio
async def test_missing_file_raises(monkeypatch):
    monkeypatch.setenv(
        "SYSTEM_DEFINED_PROMPTS",
        json.dumps(
            [
                {
                    "name": "missing_file",
                    "title": "Missing",
                    "description": "Missing file",
                    "arguments": [],
                    "template": {"role": "assistant", "file": "does/not/exist.txt"},
                }
            ]
        ),
    )

    with pytest.raises(HTTPException):
        await call_prompt(None, "missing_file", {})

    @pytest.mark.asyncio
    async def test_absolute_path_allowed(monkeypatch):
        # Create a file in /tmp and reference it by absolute path
        abs_file = "/tmp/mcp_prompt_test.txt"
        with open(abs_file, "w", encoding="utf-8") as fh:
            fh.write("Absolute {who}")

        monkeypatch.setenv(
            "SYSTEM_DEFINED_PROMPTS",
            json.dumps(
                [
                    {
                        "name": "abs_path",
                        "title": "AbsPath",
                        "description": "Absolute path prompt",
                        "arguments": [{"name": "who"}],
                        "template": {"role": "assistant", "file": abs_file},
                    }
                ]
            ),
        )

        res = await call_prompt(None, "abs_path", {"who": "World"})
        assert not res.isError
        assert res.messages[0].content.text == "Absolute World"


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
