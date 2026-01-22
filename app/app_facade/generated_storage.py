import json
import os
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
