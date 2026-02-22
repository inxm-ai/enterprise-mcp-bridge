import copy
import asyncio
import json
import time
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

import app.app_facade.generated_service as generated_service_module
from app.app_facade.generated_service import Actor, GeneratedUIService, Scope
from app.app_facade.generated_storage import GeneratedUIStorage


def _base_record() -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    return {
        "metadata": {
            "id": "dash1",
            "name": "overview",
            "scope": {"type": "user", "id": "user123"},
            "owner": {"type": "user", "id": "user123"},
            "created_by": "user123",
            "created_at": ts,
            "updated_at": ts,
            "version": 1,
            "published_at": ts,
            "published_by": "user123",
            "history": [],
        },
        "current": {
            "html": {
                "page": "<!DOCTYPE html><html><body><div>v1</div></body></html>",
                "snippet": "<div>v1</div>",
            },
            "metadata": {"requirements": "initial"},
            "service_script": "export class McpService {}",
            "components_script": "export const init = () => {};",
            "test_script": "import { test } from 'node:test'; test('ok', () => {});",
        },
    }


class DummyTGI:
    pass


def _new_service(tmp_path) -> tuple[GeneratedUIService, GeneratedUIStorage]:
    storage = GeneratedUIStorage(str(tmp_path))
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGI())
    return service, storage


async def _wait_for_test_state(
    storage: GeneratedUIStorage,
    scope: Scope,
    ui_id: str,
    name: str,
    session_id: str,
    expected_states: set[str],
    timeout: float = 2.0,
) -> dict:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        payload = storage.read_session(scope, ui_id, name, session_id)
        if str(payload.get("test_state")) in expected_states:
            return payload
        await asyncio.sleep(0.02)
    raise AssertionError(f"Timed out waiting for states: {expected_states}")


@pytest.mark.asyncio
async def test_stream_chat_update_prefers_patch_when_available(tmp_path, monkeypatch):
    service, storage = _new_service(tmp_path)
    scope = Scope(kind="user", identifier="user123")
    actor = Actor(user_id="user123", groups=["eng"])
    storage.write(scope, "dash1", "overview", _base_record())
    session_info = service.create_draft_session(
        scope=scope, actor=actor, ui_id="dash1", name="overview", tools=[]
    )
    session_id = session_info["session_id"]

    async def fake_select_tools(_session, _requested_tools, _prompt):
        return [{"type": "function", "function": {"name": "list_items"}}]

    async def fake_assistant(**_kwargs):
        return "Applied requested change."

    async def fake_patch(**_kwargs):
        rec = storage.read(scope, "dash1", "overview")
        patched = copy.deepcopy(rec["current"])
        patched["html"]["snippet"] = "<div>patched</div>"
        patched["html"][
            "page"
        ] = "<!DOCTYPE html><html><body><div>patched</div></body></html>"
        return {"payload": patched}

    async def fail_regenerate(**_kwargs):  # pragma: no cover - defensive
        raise AssertionError("Fallback regenerate should not run in patch success path")

    async def fake_queue_tests(**_kwargs):
        return {"status": "queued", "run_id": "run-1", "trigger": "post_update"}

    monkeypatch.setattr(service, "_select_tools", fake_select_tools)
    monkeypatch.setattr(service, "_run_assistant_message", fake_assistant)
    monkeypatch.setattr(service, "_attempt_patch_update", fake_patch)
    monkeypatch.setattr(service, "_generate_ui_payload", fail_regenerate)
    monkeypatch.setattr(service, "_queue_test_run", fake_queue_tests)

    chunks = []
    async for chunk in service.stream_chat_update(
        session=object(),
        scope=scope,
        actor=actor,
        ui_id="dash1",
        name="overview",
        session_id=session_id,
        message="Make it patched",
        tools=["list_items"],
        tool_choice="auto",
        access_token=None,
    ):
        chunks.append(chunk.decode("utf-8"))

    joined = "".join(chunks)
    assert "patch_applied" in joined
    assert "ui_updated" in joined
    assert "tests_queued" in joined
    draft = service.get_draft_ui(
        scope=scope,
        actor=actor,
        ui_id="dash1",
        name="overview",
        session_id=session_id,
        expand=False,
    )
    assert draft["current"]["html"]["snippet"] == "<div>patched</div>"


@pytest.mark.asyncio
async def test_stream_chat_update_falls_back_to_regenerate(tmp_path, monkeypatch):
    service, storage = _new_service(tmp_path)
    scope = Scope(kind="user", identifier="user123")
    actor = Actor(user_id="user123", groups=["eng"])
    storage.write(scope, "dash1", "overview", _base_record())
    session_id = service.create_draft_session(
        scope=scope, actor=actor, ui_id="dash1", name="overview", tools=[]
    )["session_id"]

    async def fake_select_tools(_session, _requested_tools, _prompt):
        return [{"type": "function", "function": {"name": "list_items"}}]

    async def fake_assistant(**_kwargs):
        return "Regenerating"

    async def no_patch(**_kwargs):
        return None

    async def fake_regenerate(**_kwargs):
        rec = storage.read(scope, "dash1", "overview")
        payload = copy.deepcopy(rec["current"])
        payload["html"]["snippet"] = "<div>regen</div>"
        payload["html"][
            "page"
        ] = "<!DOCTYPE html><html><body><div>regen</div></body></html>"
        payload["metadata"]["generation_diagnostics"] = {
            "message_payload_compaction": {
                "original_bytes": 90000,
                "final_bytes": 28000,
                "budget_bytes": 32000,
                "steps": ["trim_history:8", "compact_current_state"],
            }
        }
        return payload

    async def fake_queue_tests(**_kwargs):
        return {"status": "queued", "run_id": "run-2", "trigger": "post_update"}

    monkeypatch.setattr(service, "_select_tools", fake_select_tools)
    monkeypatch.setattr(service, "_run_assistant_message", fake_assistant)
    monkeypatch.setattr(service, "_attempt_patch_update", no_patch)
    monkeypatch.setattr(service, "_generate_ui_payload", fake_regenerate)
    monkeypatch.setattr(service, "_queue_test_run", fake_queue_tests)
    monkeypatch.setattr(generated_service_module, "APP_UI_PATCH_ONLY", False)
    monkeypatch.setattr(generated_service_module, "APP_UI_PATCH_ONLY", False)
    monkeypatch.setattr(generated_service_module, "APP_UI_PATCH_ONLY", False)

    chunks = []
    async for chunk in service.stream_chat_update(
        session=object(),
        scope=scope,
        actor=actor,
        ui_id="dash1",
        name="overview",
        session_id=session_id,
        message="Force regenerate",
        tools=["list_items"],
        tool_choice="auto",
        access_token=None,
    ):
        chunks.append(chunk.decode("utf-8"))

    joined = "".join(chunks)
    assert "regenerated_fallback" in joined
    assert "tests_queued" in joined
    assert "attempt=1/2:unknown_patch_failure" in joined
    assert "context_compaction" in joined


@pytest.mark.asyncio
async def test_stream_chat_update_patch_only_blocks_regenerate(tmp_path, monkeypatch):
    service, storage = _new_service(tmp_path)
    scope = Scope(kind="user", identifier="user123")
    actor = Actor(user_id="user123", groups=["eng"])
    storage.write(scope, "dash1", "overview", _base_record())
    session_id = service.create_draft_session(
        scope=scope, actor=actor, ui_id="dash1", name="overview", tools=[]
    )["session_id"]

    async def fake_select_tools(_session, _requested_tools, _prompt):
        return [{"type": "function", "function": {"name": "list_items"}}]

    async def fake_assistant(**_kwargs):
        return "Patch-only path"

    async def no_patch(**_kwargs):
        return None

    async def fail_regenerate(**_kwargs):  # pragma: no cover - defensive
        raise AssertionError("Regenerate should not run when patch-only is enabled")

    monkeypatch.setattr(service, "_select_tools", fake_select_tools)
    monkeypatch.setattr(service, "_run_assistant_message", fake_assistant)
    monkeypatch.setattr(service, "_attempt_patch_update", no_patch)
    monkeypatch.setattr(service, "_generate_ui_payload", fail_regenerate)
    monkeypatch.setattr(generated_service_module, "APP_UI_PATCH_ONLY", True)

    chunks = []
    async for chunk in service.stream_chat_update(
        session=object(),
        scope=scope,
        actor=actor,
        ui_id="dash1",
        name="overview",
        session_id=session_id,
        message="Patch only",
        tools=["list_items"],
        tool_choice="auto",
        access_token=None,
    ):
        chunks.append(chunk.decode("utf-8"))

    joined = "".join(chunks)
    assert "event: error" in joined
    assert "APP_UI_PATCH_ONLY=true prevents full regeneration" in joined


@pytest.mark.asyncio
async def test_stream_chat_update_patch_retry_succeeds_before_regenerate(
    tmp_path, monkeypatch
):
    service, storage = _new_service(tmp_path)
    scope = Scope(kind="user", identifier="user123")
    actor = Actor(user_id="user123", groups=["eng"])
    storage.write(scope, "dash1", "overview", _base_record())
    session_id = service.create_draft_session(
        scope=scope, actor=actor, ui_id="dash1", name="overview", tools=[]
    )["session_id"]

    attempts = {"count": 0}

    async def fake_select_tools(_session, _requested_tools, _prompt):
        return [{"type": "function", "function": {"name": "list_items"}}]

    async def fake_assistant(**_kwargs):
        return "Retry patch"

    async def flaky_patch(**_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            service._last_patch_failure_reason = "patch_tests_failed"
            return None
        rec = storage.read(scope, "dash1", "overview")
        patched = copy.deepcopy(rec["current"])
        patched["html"]["snippet"] = "<div>patched-retry</div>"
        patched["html"][
            "page"
        ] = "<!DOCTYPE html><html><body><div>patched-retry</div></body></html>"
        return {"payload": patched}

    async def fail_regenerate(**_kwargs):  # pragma: no cover - defensive
        raise AssertionError("Regenerate should not run when retry patch succeeds")

    async def fake_queue_tests(**_kwargs):
        return {"status": "queued", "run_id": "run-retry", "trigger": "post_update"}

    monkeypatch.setattr(service, "_select_tools", fake_select_tools)
    monkeypatch.setattr(service, "_run_assistant_message", fake_assistant)
    monkeypatch.setattr(service, "_attempt_patch_update", flaky_patch)
    monkeypatch.setattr(service, "_generate_ui_payload", fail_regenerate)
    monkeypatch.setattr(service, "_queue_test_run", fake_queue_tests)
    monkeypatch.setattr(generated_service_module, "APP_UI_PATCH_ONLY", False)
    monkeypatch.setattr(generated_service_module, "APP_UI_PATCH_RETRIES", 2)

    chunks = []
    async for chunk in service.stream_chat_update(
        session=object(),
        scope=scope,
        actor=actor,
        ui_id="dash1",
        name="overview",
        session_id=session_id,
        message="Retry patch",
        tools=["list_items"],
        tool_choice="auto",
        access_token=None,
    ):
        chunks.append(chunk.decode("utf-8"))

    joined = "".join(chunks)
    assert "patch_applied" in joined
    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_stream_chat_update_emits_progress_status_updates(tmp_path, monkeypatch):
    service, storage = _new_service(tmp_path)
    scope = Scope(kind="user", identifier="user123")
    actor = Actor(user_id="user123", groups=["eng"])
    storage.write(scope, "dash1", "overview", _base_record())
    session_id = service.create_draft_session(
        scope=scope, actor=actor, ui_id="dash1", name="overview", tools=[]
    )["session_id"]

    async def fake_select_tools(_session, _requested_tools, _prompt):
        return [{"type": "function", "function": {"name": "list_items"}}]

    async def fake_assistant(**_kwargs):
        return ""

    async def fake_patch(**_kwargs):
        rec = storage.read(scope, "dash1", "overview")
        patched = copy.deepcopy(rec["current"])
        patched["html"]["snippet"] = "<div>patched-status</div>"
        patched["html"][
            "page"
        ] = "<!DOCTYPE html><html><body><div>patched-status</div></body></html>"
        return {"payload": patched}

    async def fake_queue_tests(**_kwargs):
        return {"status": "queued", "run_id": "run-status", "trigger": "post_update"}

    monkeypatch.setattr(service, "_select_tools", fake_select_tools)
    monkeypatch.setattr(service, "_run_assistant_message", fake_assistant)
    monkeypatch.setattr(service, "_attempt_patch_update", fake_patch)
    monkeypatch.setattr(service, "_queue_test_run", fake_queue_tests)

    chunks = []
    async for chunk in service.stream_chat_update(
        session=object(),
        scope=scope,
        actor=actor,
        ui_id="dash1",
        name="overview",
        session_id=session_id,
        message="Show progress",
        tools=["list_items"],
        tool_choice="auto",
        access_token=None,
    ):
        chunks.append(chunk.decode("utf-8"))

    joined = "".join(chunks)
    assert "I will prepare the update request now." in joined
    assert "I will select the best tools for this update." in joined
    assert "I will run the validation tests for this updated draft now." in joined


@pytest.mark.asyncio
async def test_stream_chat_update_runtime_context_is_single_use(tmp_path, monkeypatch):
    service, storage = _new_service(tmp_path)
    scope = Scope(kind="user", identifier="user123")
    actor = Actor(user_id="user123", groups=["eng"])
    storage.write(scope, "dash1", "overview", _base_record())
    session_id = service.create_draft_session(
        scope=scope, actor=actor, ui_id="dash1", name="overview", tools=[]
    )["session_id"]

    captured_runtime = []

    async def fake_select_tools(_session, _requested_tools, _prompt):
        return [{"type": "function", "function": {"name": "list_items"}}]

    async def fake_assistant(**kwargs):
        captured_runtime.append(("assistant", kwargs.get("runtime_context")))
        return "Regenerating"

    async def no_patch(**_kwargs):
        return None

    async def fake_regenerate(**kwargs):
        captured_runtime.append(("regenerate", kwargs.get("runtime_context")))
        rec = storage.read(scope, "dash1", "overview")
        payload = copy.deepcopy(rec["current"])
        payload["html"]["snippet"] = "<div>runtime-aware</div>"
        payload["html"][
            "page"
        ] = "<!DOCTYPE html><html><body><div>runtime-aware</div></body></html>"
        return payload

    async def fake_queue_tests(**_kwargs):
        return {"status": "queued", "run_id": "run-runtime", "trigger": "post_update"}

    monkeypatch.setattr(service, "_select_tools", fake_select_tools)
    monkeypatch.setattr(service, "_run_assistant_message", fake_assistant)
    monkeypatch.setattr(service, "_attempt_patch_update", no_patch)
    monkeypatch.setattr(service, "_generate_ui_payload", fake_regenerate)
    monkeypatch.setattr(service, "_queue_test_run", fake_queue_tests)
    monkeypatch.setattr(generated_service_module, "APP_UI_PATCH_ONLY", False)

    runtime_action = {
        "type": "runtime_service_exchanges",
        "cursor": 3,
        "captured_at": "2026-02-21T12:00:00Z",
        "entries": [
            {
                "tool": "list_items",
                "request_body": {"limit": 5},
                "response_payload": {"items": [{"id": "1"}]},
            }
        ],
        "console_events": [
            {
                "kind": "console_warning",
                "message": "Received empty optional field from list_items",
            }
        ],
    }

    async for _chunk in service.stream_chat_update(
        session=object(),
        scope=scope,
        actor=actor,
        ui_id="dash1",
        name="overview",
        session_id=session_id,
        message="Use runtime context",
        tools=["list_items"],
        tool_choice="auto",
        draft_action=runtime_action,
        access_token=None,
    ):
        pass

    async for _chunk in service.stream_chat_update(
        session=object(),
        scope=scope,
        actor=actor,
        ui_id="dash1",
        name="overview",
        session_id=session_id,
        message="Second message without runtime context",
        tools=["list_items"],
        tool_choice="auto",
        draft_action=None,
        access_token=None,
    ):
        pass

    assistant_contexts = [ctx for role, ctx in captured_runtime if role == "assistant"]
    regenerate_contexts = [
        ctx for role, ctx in captured_runtime if role == "regenerate"
    ]

    assert len(assistant_contexts) == 2
    assert len(regenerate_contexts) == 2

    assert assistant_contexts[0] is not None
    assert assistant_contexts[0]["console_events"][0]["kind"] == "console_warning"
    assert assistant_contexts[1] is None

    assert regenerate_contexts[0] is not None
    assert regenerate_contexts[1] is None


@pytest.mark.asyncio
async def test_stream_chat_update_runtime_context_not_reused_after_error(
    tmp_path, monkeypatch
):
    service, storage = _new_service(tmp_path)
    scope = Scope(kind="user", identifier="user123")
    actor = Actor(user_id="user123", groups=["eng"])
    storage.write(scope, "dash1", "overview", _base_record())
    session_id = service.create_draft_session(
        scope=scope, actor=actor, ui_id="dash1", name="overview", tools=[]
    )["session_id"]

    captured_runtime = []
    attempts = {"count": 0}

    async def fake_select_tools(_session, _requested_tools, _prompt):
        return [{"type": "function", "function": {"name": "list_items"}}]

    async def fake_assistant(**kwargs):
        captured_runtime.append(("assistant", kwargs.get("runtime_context")))
        return "Regenerating"

    async def no_patch(**_kwargs):
        return None

    async def flaky_regenerate(**kwargs):
        attempts["count"] += 1
        captured_runtime.append(("regenerate", kwargs.get("runtime_context")))
        if attempts["count"] == 1:
            raise RuntimeError("forced regenerate failure")
        rec = storage.read(scope, "dash1", "overview")
        payload = copy.deepcopy(rec["current"])
        payload["html"]["snippet"] = "<div>recovered</div>"
        payload["html"][
            "page"
        ] = "<!DOCTYPE html><html><body><div>recovered</div></body></html>"
        return payload

    async def fake_queue_tests(**_kwargs):
        return {
            "status": "queued",
            "run_id": "run-after-error",
            "trigger": "post_update",
        }

    monkeypatch.setattr(service, "_select_tools", fake_select_tools)
    monkeypatch.setattr(service, "_run_assistant_message", fake_assistant)
    monkeypatch.setattr(service, "_attempt_patch_update", no_patch)
    monkeypatch.setattr(service, "_generate_ui_payload", flaky_regenerate)
    monkeypatch.setattr(service, "_queue_test_run", fake_queue_tests)
    monkeypatch.setattr(generated_service_module, "APP_UI_PATCH_ONLY", False)

    first_chunks = []
    async for chunk in service.stream_chat_update(
        session=object(),
        scope=scope,
        actor=actor,
        ui_id="dash1",
        name="overview",
        session_id=session_id,
        message="Fail once with runtime context",
        tools=["list_items"],
        tool_choice="auto",
        draft_action={
            "type": "runtime_service_exchanges",
            "entries": [{"tool": "list_items", "response_payload": {"items": []}}],
        },
        access_token=None,
    ):
        first_chunks.append(chunk.decode("utf-8"))

    second_chunks = []
    async for chunk in service.stream_chat_update(
        session=object(),
        scope=scope,
        actor=actor,
        ui_id="dash1",
        name="overview",
        session_id=session_id,
        message="Retry without runtime context",
        tools=["list_items"],
        tool_choice="auto",
        draft_action=None,
        access_token=None,
    ):
        second_chunks.append(chunk.decode("utf-8"))

    assert "event: error" in "".join(first_chunks)
    assert "event: done" in "".join(second_chunks)
    assert captured_runtime[0][0] == "assistant"
    assert captured_runtime[0][1] is not None
    assert captured_runtime[1][0] == "regenerate"
    assert captured_runtime[1][1] is not None
    assert captured_runtime[2][0] == "assistant"
    assert captured_runtime[2][1] is None
    assert captured_runtime[3][0] == "regenerate"
    assert captured_runtime[3][1] is None


@pytest.mark.asyncio
async def test_stream_chat_update_accepts_console_only_runtime_context(
    tmp_path, monkeypatch
):
    service, storage = _new_service(tmp_path)
    scope = Scope(kind="user", identifier="user123")
    actor = Actor(user_id="user123", groups=["eng"])
    storage.write(scope, "dash1", "overview", _base_record())
    session_id = service.create_draft_session(
        scope=scope, actor=actor, ui_id="dash1", name="overview", tools=[]
    )["session_id"]

    captured_runtime = []

    async def fake_select_tools(_session, _requested_tools, _prompt):
        return [{"type": "function", "function": {"name": "list_items"}}]

    async def fake_assistant(**kwargs):
        captured_runtime.append(kwargs.get("runtime_context"))
        return "Regenerating"

    async def no_patch(**_kwargs):
        return None

    async def fake_regenerate(**_kwargs):
        rec = storage.read(scope, "dash1", "overview")
        payload = copy.deepcopy(rec["current"])
        payload["html"]["snippet"] = "<div>console-only</div>"
        payload["html"][
            "page"
        ] = "<!DOCTYPE html><html><body><div>console-only</div></body></html>"
        return payload

    async def fake_queue_tests(**_kwargs):
        return {
            "status": "queued",
            "run_id": "run-console-only",
            "trigger": "post_update",
        }

    monkeypatch.setattr(service, "_select_tools", fake_select_tools)
    monkeypatch.setattr(service, "_run_assistant_message", fake_assistant)
    monkeypatch.setattr(service, "_attempt_patch_update", no_patch)
    monkeypatch.setattr(service, "_generate_ui_payload", fake_regenerate)
    monkeypatch.setattr(service, "_queue_test_run", fake_queue_tests)

    async for _chunk in service.stream_chat_update(
        session=object(),
        scope=scope,
        actor=actor,
        ui_id="dash1",
        name="overview",
        session_id=session_id,
        message="Use console warning context",
        tools=["list_items"],
        tool_choice="auto",
        draft_action={
            "type": "runtime_service_exchanges",
            "console_events": [
                {"kind": "console_warning", "message": "Potential null value"}
            ],
        },
        access_token=None,
    ):
        pass

    assert captured_runtime
    assert captured_runtime[0] is not None
    assert captured_runtime[0]["console_events"][0]["kind"] == "console_warning"


def test_publish_draft_version_conflict_returns_409(tmp_path):
    service, storage = _new_service(tmp_path)
    scope = Scope(kind="user", identifier="user123")
    actor = Actor(user_id="user123", groups=["eng"])
    record = _base_record()
    storage.write(scope, "dash1", "overview", record)

    session_id = service.create_draft_session(
        scope=scope, actor=actor, ui_id="dash1", name="overview", tools=[]
    )["session_id"]

    updated = storage.read(scope, "dash1", "overview")
    updated["metadata"]["version"] = 2
    storage.write(scope, "dash1", "overview", updated)

    with pytest.raises(HTTPException) as exc:
        service.publish_draft_session(
            scope=scope,
            actor=actor,
            ui_id="dash1",
            name="overview",
            session_id=session_id,
        )
    assert exc.value.status_code == 409
    assert exc.value.detail["error"] == "publish_conflict"


@pytest.mark.asyncio
async def test_queue_test_action_run_updates_state(tmp_path, monkeypatch):
    service, storage = _new_service(tmp_path)
    scope = Scope(kind="user", identifier="user123")
    actor = Actor(user_id="user123", groups=["eng"])
    storage.write(scope, "dash1", "overview", _base_record())
    session_id = service.create_draft_session(
        scope=scope, actor=actor, ui_id="dash1", name="overview", tools=[]
    )["session_id"]

    monkeypatch.setattr(
        service,
        "_run_tests",
        lambda *_args, **_kwargs: (True, "TAP version 13\n# pass 2\n# fail 0\n"),
    )

    queued = await service.queue_test_action(
        scope=scope,
        actor=actor,
        ui_id="dash1",
        name="overview",
        session_id=session_id,
        action="run",
        test_name=None,
        access_token=None,
    )
    assert queued["status"] == "queued"
    final_payload = await _wait_for_test_state(
        storage, scope, "dash1", "overview", session_id, {"passed"}
    )
    assert final_payload["test_summary"]["passed"] == 2
    assert final_payload["test_summary"]["failed"] == 0


@pytest.mark.asyncio
async def test_queue_test_action_run_allows_missing_service_script(
    tmp_path, monkeypatch
):
    service, storage = _new_service(tmp_path)
    scope = Scope(kind="user", identifier="user123")
    actor = Actor(user_id="user123", groups=["eng"])
    record = _base_record()
    record["current"]["service_script"] = ""
    storage.write(scope, "dash1", "overview", record)
    session_id = service.create_draft_session(
        scope=scope, actor=actor, ui_id="dash1", name="overview", tools=[]
    )["session_id"]

    monkeypatch.setattr(
        service,
        "_run_tests",
        lambda *_args, **_kwargs: (True, "TAP version 13\n# pass 1\n# fail 0\n"),
    )

    queued = await service.queue_test_action(
        scope=scope,
        actor=actor,
        ui_id="dash1",
        name="overview",
        session_id=session_id,
        action="run",
        test_name=None,
        access_token=None,
    )
    assert queued["status"] == "queued"
    final_payload = await _wait_for_test_state(
        storage, scope, "dash1", "overview", session_id, {"passed"}
    )
    assert final_payload["test_summary"]["passed"] == 1
    assert final_payload["test_summary"]["failed"] == 0


@pytest.mark.asyncio
async def test_queue_test_action_cancels_previous_run(tmp_path, monkeypatch):
    service, storage = _new_service(tmp_path)
    scope = Scope(kind="user", identifier="user123")
    actor = Actor(user_id="user123", groups=["eng"])
    storage.write(scope, "dash1", "overview", _base_record())
    session_id = service.create_draft_session(
        scope=scope, actor=actor, ui_id="dash1", name="overview", tools=[]
    )["session_id"]

    def slow_tests(*_args, **_kwargs):
        time.sleep(0.2)
        return True, "TAP version 13\n# pass 1\n# fail 0\n"

    monkeypatch.setattr(service, "_run_tests", slow_tests)

    first = await service.queue_test_action(
        scope=scope,
        actor=actor,
        ui_id="dash1",
        name="overview",
        session_id=session_id,
        action="run",
        test_name=None,
        access_token=None,
    )
    second = await service.queue_test_action(
        scope=scope,
        actor=actor,
        ui_id="dash1",
        name="overview",
        session_id=session_id,
        action="run",
        test_name=None,
        access_token=None,
    )
    assert first["run_id"] != second["run_id"]
    final_payload = await _wait_for_test_state(
        storage, scope, "dash1", "overview", session_id, {"passed"}
    )
    statuses = [
        item.get("payload", {}).get("state")
        for item in final_payload.get("test_events", [])
        if item.get("event") == "test_status"
    ]
    assert "cancelled" in statuses


@pytest.mark.asyncio
async def test_add_test_action_updates_test_script(tmp_path, monkeypatch):
    service, storage = _new_service(tmp_path)
    scope = Scope(kind="user", identifier="user123")
    actor = Actor(user_id="user123", groups=["eng"])
    storage.write(scope, "dash1", "overview", _base_record())
    session_id = service.create_draft_session(
        scope=scope, actor=actor, ui_id="dash1", name="overview", tools=[]
    )["session_id"]

    monkeypatch.setattr(
        service,
        "_run_tests",
        lambda *_args, **_kwargs: (True, "TAP version 13\n# pass 3\n# fail 0\n"),
    )

    await service.queue_test_action(
        scope=scope,
        actor=actor,
        ui_id="dash1",
        name="overview",
        session_id=session_id,
        action="add_test",
        test_name="adds coverage for cards",
        access_token=None,
    )
    final_payload = await _wait_for_test_state(
        storage, scope, "dash1", "overview", session_id, {"passed"}
    )
    script = final_payload["draft_payload"]["test_script"]
    assert "adds coverage for cards" in script


@pytest.mark.asyncio
async def test_queue_test_action_failed_run_includes_analysis_payload(
    tmp_path, monkeypatch
):
    service, storage = _new_service(tmp_path)
    scope = Scope(kind="user", identifier="user123")
    actor = Actor(user_id="user123", groups=["eng"])
    storage.write(scope, "dash1", "overview", _base_record())
    session_id = service.create_draft_session(
        scope=scope, actor=actor, ui_id="dash1", name="overview", tools=[]
    )["session_id"]

    monkeypatch.setattr(
        service,
        "_run_tests",
        lambda *_args, **_kwargs: (
            False,
            (
                "TAP version 13\n"
                "not ok 1 - Weather Dashboard Tests\n"
                "not ok 2 - weather-forecast: refetches when city changes\n"
                "# pass 1\n"
                "# fail 2\n"
            ),
        ),
    )

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    async def fake_explain(**_kwargs):
        return {
            "what_failed": "2 tests failed: Weather Dashboard Tests, weather-forecast",
            "why_it_ended": "fix_code run ended after repeated assertion-signature failures.",
            "recommended_action": "adjust_test",
            "alternative_action": "fix_code",
            "next_step_text": "Recommended next action: adjust_test. Alternative: fix_code.",
        }

    monkeypatch.setattr(service, "_explain_test_failure_for_user", fake_explain)

    run_id = "run-99999999"
    stream_key = service._test_stream_key(
        scope=scope, ui_id="dash1", name="overview", session_id=session_id
    )
    session_payload = storage.read_session(scope, "dash1", "overview", session_id)
    session_payload["test_run_id"] = run_id
    session_payload["test_state"] = "queued"
    storage.write_session(scope, "dash1", "overview", session_id, session_payload)

    await service._execute_test_run(
        scope=scope,
        ui_id="dash1",
        name="overview",
        session_id=session_id,
        stream_key=stream_key,
        run_id=run_id,
        action="run",
        trigger="manual_run",
        test_name=None,
        access_token=None,
    )

    final_payload = storage.read_session(scope, "dash1", "overview", session_id)
    result_events = [
        item.get("payload", {})
        for item in final_payload.get("test_events", [])
        if item.get("event") == "test_result"
    ]
    assert result_events
    final_result = result_events[-1]
    assert final_result["is_final"] is True
    assert final_result["recommended_action"] == "adjust_test"
    assert final_result["alternative_action"] == "fix_code"
    assert final_result["analysis"]["why_it_ended"].startswith("fix_code run ended")
    assert (
        final_payload["test_summary"]["message"]
        == "Recommended next action: adjust_test. Alternative: fix_code."
    )


@pytest.mark.asyncio
async def test_explain_test_failure_for_user_uses_fallback_without_llm(tmp_path):
    service, _storage = _new_service(tmp_path)

    analysis = await service._explain_test_failure_for_user(
        action="fix_code",
        passed=1,
        failed=2,
        failed_tests=["Weather Dashboard Tests", "forecast test"],
        output_tail="# pass 1\n# fail 2\n",
        strategy_context={
            "reason": "Repeated assertion-signature failures persisted.",
            "suggested_action": "adjust_test",
        },
        access_token=None,
    )

    assert analysis["recommended_action"] == "adjust_test"
    assert analysis["alternative_action"] == "fix_code"
    assert "Repeated assertion-signature failures persisted" in analysis["why_it_ended"]


@pytest.mark.asyncio
async def test_explain_test_failure_for_user_parses_llm_structured_response(tmp_path):
    service, _storage = _new_service(tmp_path)

    class StubLLMClient:
        def __init__(self):
            self.calls = 0

        async def non_stream_completion(self, request, _token, _span):
            self.calls += 1
            assert request.response_format is not None
            content = json.dumps(
                {
                    "what_failed": "Two weather tests failed after the latest run.",
                    "why_it_ended": "fix_code stopped because assertion-shape failures persisted.",
                    "recommended_action": "adjust_test",
                    "alternative_action": "fix_code",
                    "next_step_text": "Use adjust_test next; fall back to fix_code if needed.",
                }
            )
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {"message": type("Message", (), {"content": content})()},
                        )
                    ]
                },
            )()

    service.tgi_service.llm_client = StubLLMClient()

    analysis = await service._explain_test_failure_for_user(
        action="fix_code",
        passed=2,
        failed=2,
        failed_tests=["a", "b"],
        output_tail="TAP version 13\n# pass 2\n# fail 2\n",
        strategy_context={"reason_code": "assertion_signature_repeat"},
        access_token=None,
    )

    assert analysis["recommended_action"] == "adjust_test"
    assert analysis["alternative_action"] == "fix_code"
    assert "assertion-shape failures persisted" in analysis["why_it_ended"]
