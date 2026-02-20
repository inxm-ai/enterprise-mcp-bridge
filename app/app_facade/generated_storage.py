import json
import os
from datetime import datetime, timezone
from typing import Optional
from typing import Any, Dict

from fastapi import HTTPException

from app.app_facade.generated_types import Scope, validate_identifier


class GeneratedUIStorage:
    def __init__(self, base_path: str):
        if not base_path:
            raise ValueError("Generated ui storage path is required")
        self.base_path = os.path.abspath(base_path)

    def _ui_dir(self, scope: Scope, ui_id: str, name: str) -> str:
        safe_scope = validate_identifier(scope.identifier, f"{scope.kind} id")
        safe_ui_id = validate_identifier(ui_id, "ui id")
        safe_name = validate_identifier(name, "ui name")
        return os.path.join(
            self.base_path, scope.kind, safe_scope, safe_ui_id, safe_name
        )

    def _file_path(self, scope: Scope, ui_id: str, name: str) -> str:
        return os.path.join(self._ui_dir(scope, ui_id, name), "ui.json")

    def _sessions_dir(self, scope: Scope, ui_id: str, name: str) -> str:
        return os.path.join(self._ui_dir(scope, ui_id, name), "sessions")

    def _session_file_path(
        self, scope: Scope, ui_id: str, name: str, session_id: str
    ) -> str:
        safe_session_id = validate_identifier(session_id, "session id")
        return os.path.join(
            self._sessions_dir(scope, ui_id, name),
            f"{safe_session_id}.json",
        )

    def read(self, scope: Scope, ui_id: str, name: str) -> Dict[str, Any]:
        file_path = self._file_path(scope, ui_id, name)
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            raise
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Stored ui payload at {file_path} is invalid JSON",
            ) from exc

    def write(
        self, scope: Scope, ui_id: str, name: str, payload: Dict[str, Any]
    ) -> None:
        file_path = self._file_path(scope, ui_id, name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        tmp_path = f"{file_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, file_path)

    def exists(self, scope: Scope, ui_id: str, name: str) -> bool:
        return os.path.exists(self._file_path(scope, ui_id, name))

    def read_session(
        self, scope: Scope, ui_id: str, name: str, session_id: str
    ) -> Dict[str, Any]:
        file_path = self._session_file_path(scope, ui_id, name, session_id)
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            raise
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Stored session payload at {file_path} is invalid JSON",
            ) from exc

    def write_session(
        self,
        scope: Scope,
        ui_id: str,
        name: str,
        session_id: str,
        payload: Dict[str, Any],
    ) -> None:
        file_path = self._session_file_path(scope, ui_id, name, session_id)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        tmp_path = f"{file_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, file_path)

    def delete_session(
        self, scope: Scope, ui_id: str, name: str, session_id: str
    ) -> bool:
        file_path = self._session_file_path(scope, ui_id, name, session_id)
        try:
            os.remove(file_path)
            return True
        except FileNotFoundError:
            return False

    def list_sessions(self, scope: Scope, ui_id: str, name: str) -> list[str]:
        sessions_dir = self._sessions_dir(scope, ui_id, name)
        if not os.path.isdir(sessions_dir):
            return []
        session_ids: list[str] = []
        for filename in os.listdir(sessions_dir):
            if not filename.endswith(".json"):
                continue
            session_id = filename[:-5]
            try:
                validate_identifier(session_id, "session id")
            except HTTPException:
                continue
            session_ids.append(session_id)
        return sorted(session_ids)

    def cleanup_expired_sessions(self, scope: Scope, ui_id: str, name: str) -> int:
        removed = 0
        now = datetime.now(timezone.utc)
        for session_id in self.list_sessions(scope, ui_id, name):
            expires_at: Optional[str] = None
            try:
                payload = self.read_session(scope, ui_id, name, session_id)
                expires_at = payload.get("expires_at")
            except FileNotFoundError:
                continue
            except Exception:
                # Corrupt session payloads are removed defensively.
                if self.delete_session(scope, ui_id, name, session_id):
                    removed += 1
                continue

            if not expires_at:
                continue
            try:
                normalized = (
                    expires_at[:-1] + "+00:00"
                    if str(expires_at).endswith("Z")
                    else str(expires_at)
                )
                expires_dt = datetime.fromisoformat(normalized)
                if expires_dt.tzinfo is None:
                    expires_dt = expires_dt.replace(tzinfo=timezone.utc)
                if expires_dt <= now and self.delete_session(
                    scope, ui_id, name, session_id
                ):
                    removed += 1
            except ValueError:
                if self.delete_session(scope, ui_id, name, session_id):
                    removed += 1
        return removed
