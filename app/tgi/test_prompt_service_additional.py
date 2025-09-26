import types
import pytest
from fastapi import HTTPException

from app.tgi.prompt_service import PromptService
from app.tgi.models import Message, MessageRole


class StubSession:
    def __init__(self, prompts=None, prompt_result=None, call_exception=None):
        self._prompts = prompts or []
        self._prompt_result = prompt_result
        self._call_exception = call_exception
        self.calls = []

    async def list_prompts(self):
        return {"prompts": list(self._prompts)}

    async def call_prompt(self, name, args):
        self.calls.append((name, args))
        if self._call_exception:
            raise self._call_exception
        return self._prompt_result


@pytest.mark.asyncio
async def test_find_prompt_by_name_returns_requested():
    prompts = [
        {"name": "alpha", "description": "first", "arguments": None},
        {"name": "beta", "description": "second", "arguments": None},
    ]
    session = StubSession(prompts=prompts)
    service = PromptService()

    prompt = await service.find_prompt_by_name_or_role(session, "beta")

    assert prompt["name"] == "beta"


@pytest.mark.asyncio
async def test_find_prompt_prefers_system_prompt():
    prompts = [
        {"name": "foo", "description": "something", "arguments": None},
        {"name": "system", "description": "role=system", "arguments": None},
    ]
    session = StubSession(prompts=prompts)
    service = PromptService()

    prompt = await service.find_prompt_by_name_or_role(session)

    assert prompt["name"] == "system"


@pytest.mark.asyncio
async def test_get_prompt_content_uses_template_when_present():
    prompt = {"name": "templated", "template": {"content": "static content"}}
    session = StubSession()
    service = PromptService()

    content = await service.get_prompt_content(session, prompt)

    assert content == "static content"
    assert session.calls == []


@pytest.mark.asyncio
async def test_get_prompt_content_from_call_result():
    class DummyContent:
        def __init__(self, text):
            self.text = text

    class DummyMessage:
        def __init__(self, text):
            self.content = DummyContent(text=text)

    prompt = {"name": "dynamic"}
    result = types.SimpleNamespace(isError=False, messages=[DummyMessage("hello"), DummyMessage("world")])
    session = StubSession(prompt_result=result)
    service = PromptService()

    content = await service.get_prompt_content(session, prompt)

    assert content == "hello\nworld"
    assert session.calls == [("dynamic", {})]


@pytest.mark.asyncio
async def test_get_prompt_content_skips_missing_argument_errors():
    prompt = {"name": "needs_args", "template": {"content": ""}}
    session = StubSession(call_exception=HTTPException(status_code=400, detail="Missing required arguments"))
    service = PromptService()

    content = await service.get_prompt_content(session, prompt)

    assert content == ""


@pytest.mark.asyncio
async def test_prepare_messages_appends_system_prompt(monkeypatch):
    prompts = [
        {"name": "system", "description": "role=system", "arguments": None, "template": {"content": "context"}},
    ]
    session = StubSession(prompts=prompts)
    service = PromptService()

    user_message = Message(role=MessageRole.USER, content="hi")
    prepared = await service.prepare_messages(session, [user_message])

    assert prepared[0].role == MessageRole.SYSTEM
    assert prepared[0].content == "context"
    assert prepared[1] == user_message


@pytest.mark.asyncio
async def test_prepare_messages_returns_original_on_error(monkeypatch):
    service = PromptService()
    session = StubSession()
    user_message = Message(role=MessageRole.USER, content="hi")

    async def raise_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "find_prompt_by_name_or_role", raise_error)

    prepared = await service.prepare_messages(session, [user_message])

    assert prepared == [user_message]
