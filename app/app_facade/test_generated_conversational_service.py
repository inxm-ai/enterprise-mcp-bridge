import copy
import asyncio
import time
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

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
        return payload

    async def fake_queue_tests(**_kwargs):
        return {"status": "queued", "run_id": "run-2", "trigger": "post_update"}

    monkeypatch.setattr(service, "_select_tools", fake_select_tools)
    monkeypatch.setattr(service, "_run_assistant_message", fake_assistant)
    monkeypatch.setattr(service, "_attempt_patch_update", no_patch)
    monkeypatch.setattr(service, "_generate_ui_payload", fake_regenerate)
    monkeypatch.setattr(service, "_queue_test_run", fake_queue_tests)

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

    assert captured_runtime[0][0] == "assistant"
    assert captured_runtime[0][1] is not None
    assert captured_runtime[0][1]["console_events"][0]["kind"] == "console_warning"
    assert captured_runtime[1][0] == "regenerate"
    assert captured_runtime[1][1] is not None
    assert captured_runtime[2][0] == "assistant"
    assert captured_runtime[2][1] is None
    assert captured_runtime[3][0] == "regenerate"
    assert captured_runtime[3][1] is None


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
