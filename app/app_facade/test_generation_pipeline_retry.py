"""Direct tests for the shared retry/persist core in generation_pipeline.py.

``_stream_generate_and_persist`` is exercised end-to-end elsewhere via a fake
LLM stream (test_generated_service.py), but that only covers the
happy-path/persist-failure cases on the first attempt. These tests target
the behavioral surface that changed with the create/update merge: retry-
feedback injection across attempts, immediate abort on a fatal LLM error,
and the create-vs-update instruction/merge branch — by faking
``_phase_1_attempt``/``_phase_2_attempt`` directly so each scenario is
deterministic and fast.
"""

import os

import pytest

from app.app_facade.generated_service import GeneratedUIService, GeneratedUIStorage
from app.app_facade.generated_types import Actor, Scope
from app.app_facade.generation_pipeline import GenerationPipeline
from app.tgi.models import MessageRole


class DummyTGIService:
    def __init__(self):
        self.llm_client = None
        self.prompt_service = None
        self.tool_service = None


class PS:
    async def find_prompt_by_name_or_role(self, session, prompt_name=None):
        return None


class NoTools:
    async def get_all_mcp_tools(self, session, include_output_schema=True):
        return []


def _make_service(tmp_path, monkeypatch):
    storage = GeneratedUIStorage(str(tmp_path))
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    monkeypatch.setattr(
        "app.app_facade.generated_service._load_pfusch_prompt",
        lambda: "PROMPT {{DESIGN_SYSTEM_PROMPT}}",
    )
    service.tgi_service.prompt_service = PS()
    service.tgi_service.tool_service = NoTools()
    return service


def _phase2_payload():
    return {
        "type": "result",
        "success": True,
        "payload": {
            "template_parts": {
                "title": "t",
                "styles": "",
                "html": "<app-root></app-root>",
                "script": "",
            },
            "metadata": {},
        },
    }


@pytest.mark.asyncio
async def test_retry_carries_prior_failure_reason_into_next_attempt(
    tmp_path, monkeypatch
):
    service = _make_service(tmp_path, monkeypatch)
    captured_messages_by_attempt = {}

    async def fake_phase_1_attempt(
        self, *, attempt, max_attempts, messages, allowed_tools, dummy_data, access_token
    ):
        captured_messages_by_attempt[attempt] = list(messages)
        if attempt == 1:
            yield {
                "type": "result",
                "success": False,
                "reason": "boom on first try",
                "messages": messages,
            }
            return
        yield {
            "type": "result",
            "success": True,
            "payload": {
                "components_script": "export const s = 1;",
                "test_script": "test('x', () => {});",
            },
            "messages": messages,
        }

    async def fake_phase_2_attempt(self, **_kwargs):
        yield _phase2_payload()

    monkeypatch.setattr(GenerationPipeline, "_phase_1_attempt", fake_phase_1_attempt)
    monkeypatch.setattr(GenerationPipeline, "_phase_2_attempt", fake_phase_2_attempt)

    gen = service.generation_pipeline.stream_generate_ui(
        session=None,
        scope=Scope(kind="user", identifier="u1"),
        actor=Actor(user_id="u1", groups=[]),
        ui_id="retry1",
        name="n",
        prompt="build a thing",
        tools=None,
        access_token=None,
    )
    async for _ in gen:
        pass

    assert set(captured_messages_by_attempt.keys()) == {1, 2}
    second_attempt_contents = [
        m.content for m in captured_messages_by_attempt[2] if m.role == MessageRole.USER
    ]
    assert any(
        "Previous generation attempt(s) failed" in c and "boom on first try" in c
        for c in second_attempt_contents
    )
    # the failed first attempt's own messages must not leak into attempt 2
    # beyond the compact summary — i.e. attempt 2 starts from the pristine
    # base messages plus exactly one extra feedback message.
    assert len(captured_messages_by_attempt[2]) == len(
        captured_messages_by_attempt[1]
    ) + 1


@pytest.mark.asyncio
async def test_fatal_llm_error_aborts_without_further_retries(tmp_path, monkeypatch):
    service = _make_service(tmp_path, monkeypatch)
    call_count = {"n": 0}

    async def fake_phase_1_attempt_fatal(
        self, *, attempt, max_attempts, messages, allowed_tools, dummy_data, access_token
    ):
        call_count["n"] += 1
        yield {
            "type": "result",
            "success": False,
            "reason": "insufficient_quota: no budget left",
            "messages": messages,
        }

    monkeypatch.setattr(
        GenerationPipeline, "_phase_1_attempt", fake_phase_1_attempt_fatal
    )

    gen = service.generation_pipeline.stream_generate_ui(
        session=None,
        scope=Scope(kind="user", identifier="u2"),
        actor=Actor(user_id="u2", groups=[]),
        ui_id="fatal1",
        name="n",
        prompt="build a thing",
        tools=None,
        access_token=None,
    )
    events = []
    async for chunk in gen:
        events.append(chunk)

    assert call_count["n"] == 1  # no retries after a fatal error
    assert any(b"error" in e for e in events)
    assert not service.storage.exists(Scope(kind="user", identifier="u2"), "fatal1", "n")


@pytest.mark.asyncio
async def test_create_and_update_use_distinct_phase2_instructions_and_merge_correctly(
    tmp_path, monkeypatch
):
    service = _make_service(tmp_path, monkeypatch)
    instructions = []

    async def fake_phase_1_attempt(
        self, *, attempt, max_attempts, messages, allowed_tools, dummy_data, access_token
    ):
        yield {
            "type": "result",
            "success": True,
            "payload": {
                "components_script": "export const s = 1;",
                "test_script": "test('x', () => {});",
            },
            "messages": messages,
        }

    async def fake_phase_2_attempt(self, *, instruction, **_kwargs):
        instructions.append(instruction)
        yield _phase2_payload()

    monkeypatch.setattr(GenerationPipeline, "_phase_1_attempt", fake_phase_1_attempt)
    monkeypatch.setattr(GenerationPipeline, "_phase_2_attempt", fake_phase_2_attempt)

    scope = Scope(kind="user", identifier="u3")
    actor = Actor(user_id="u3", groups=[])

    create_gen = service.generation_pipeline.stream_generate_ui(
        session=None,
        scope=scope,
        actor=actor,
        ui_id="merge1",
        name="n",
        prompt="build a thing",
        tools=None,
        access_token=None,
    )
    async for _ in create_gen:
        pass

    created = service.storage.read(scope, "merge1", "n")
    assert created["metadata"]["version"] == 1
    assert len(created["metadata"]["history"]) == 1
    assert created["metadata"]["history"][0]["action"] == "create"
    assert "generate" in instructions[0]
    assert "update" not in instructions[0]

    update_gen = service.generation_pipeline.stream_update_ui(
        session=None,
        scope=scope,
        actor=actor,
        ui_id="merge1",
        name="n",
        prompt="change a thing",
        tools=None,
        access_token=None,
    )
    async for _ in update_gen:
        pass

    updated = service.storage.read(scope, "merge1", "n")
    assert updated["metadata"]["version"] == 2
    assert len(updated["metadata"]["history"]) == 2
    assert updated["metadata"]["history"][1]["action"] == "update"
    assert "update" in instructions[1]
    # the previously-created record was merged into (not replaced wholesale)
    assert updated["metadata"]["created_by"] == created["metadata"]["created_by"]
    assert updated["current"]["components_script"] == "export const s = 1;"
