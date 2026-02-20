import copy
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

    monkeypatch.setattr(service, "_select_tools", fake_select_tools)
    monkeypatch.setattr(service, "_run_assistant_message", fake_assistant)
    monkeypatch.setattr(service, "_attempt_patch_update", fake_patch)
    monkeypatch.setattr(service, "_generate_ui_payload", fail_regenerate)

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

    monkeypatch.setattr(service, "_select_tools", fake_select_tools)
    monkeypatch.setattr(service, "_run_assistant_message", fake_assistant)
    monkeypatch.setattr(service, "_attempt_patch_update", no_patch)
    monkeypatch.setattr(service, "_generate_ui_payload", fake_regenerate)

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

    assert "regenerated_fallback" in "".join(chunks)


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
