from datetime import datetime, timedelta, timezone

from app.app_facade.generated_storage import GeneratedUIStorage
from app.app_facade.generated_types import Scope


def test_session_storage_roundtrip_and_list(tmp_path):
    storage = GeneratedUIStorage(str(tmp_path))
    scope = Scope(kind="user", identifier="user123")
    payload = {
        "session_id": "sess1",
        "editor_user_id": "user123",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "draft_payload": {"html": {"snippet": "<div>ok</div>"}},
    }

    storage.write_session(scope, "dash1", "overview", "sess1", payload)
    read_back = storage.read_session(scope, "dash1", "overview", "sess1")
    assert read_back["editor_user_id"] == "user123"
    assert storage.list_sessions(scope, "dash1", "overview") == ["sess1"]


def test_cleanup_expired_sessions_removes_only_expired(tmp_path):
    storage = GeneratedUIStorage(str(tmp_path))
    scope = Scope(kind="group", identifier="eng")
    now = datetime.now(timezone.utc)

    expired = {
        "session_id": "expired1",
        "editor_user_id": "user123",
        "created_at": now.isoformat(),
        "expires_at": (now - timedelta(minutes=1)).isoformat(),
        "draft_payload": {},
    }
    valid = {
        "session_id": "valid1",
        "editor_user_id": "user123",
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=30)).isoformat(),
        "draft_payload": {},
    }

    storage.write_session(scope, "dash1", "overview", "expired1", expired)
    storage.write_session(scope, "dash1", "overview", "valid1", valid)

    removed = storage.cleanup_expired_sessions(scope, "dash1", "overview")
    assert removed == 1
    assert storage.list_sessions(scope, "dash1", "overview") == ["valid1"]
