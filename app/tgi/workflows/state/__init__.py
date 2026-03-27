import importlib
from pathlib import Path
from typing import Optional

from app.tgi.workflows.models import WorkflowExecutionState
from app.tgi.workflows.state.postgres import PostgresWorkflowStateBackend
from app.tgi.workflows.state.sqlite import SQLiteWorkflowStateBackend
from app.vars import normalize_workflow_db_backend, resolve_workflow_db_settings


def _load_psycopg():
    try:
        return importlib.import_module("psycopg")
    except ImportError as exc:
        raise RuntimeError(
            "Postgres workflow storage requires psycopg. Install 'psycopg[binary]'."
        ) from exc


class WorkflowStateStore:
    """
    Persist workflow executions using a configurable backend.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        backend: str = "sqlite",
        database_url: str | None = None,
    ):
        normalized_backend = normalize_workflow_db_backend(backend)
        if normalized_backend == "postgres":
            if not database_url:
                raise ValueError("database_url is required when backend='postgres'.")
            self._backend = PostgresWorkflowStateBackend(
                database_url, psycopg_loader=_load_psycopg
            )
            self.db_path: Path | None = None
            self.database_url = database_url
        else:
            resolved_db_path = Path(db_path) if db_path is not None else Path(":memory:")
            self._backend = SQLiteWorkflowStateBackend(resolved_db_path)
            self.db_path = resolved_db_path
            self.database_url = None
        self.backend_name = self._backend.backend_name

    @classmethod
    def from_env(cls, default_sqlite_path: str | Path) -> "WorkflowStateStore":
        settings = resolve_workflow_db_settings(default_sqlite_path)
        if settings.backend == "postgres":
            return cls(backend="postgres", database_url=settings.database_url)
        return cls(db_path=settings.db_path, backend="sqlite")

    def _connect(self):
        """
        Backward-compatible access for tests and direct store inspection.
        """
        return self._backend._connect()

    def load_execution(self, execution_id: str) -> Optional[WorkflowExecutionState]:
        row = self._backend.load_row(execution_id)
        return WorkflowExecutionState.from_row(row)

    def save_state(self, state: WorkflowExecutionState) -> None:
        previous = self.load_execution(state.execution_id)
        if previous:
            previous_status = previous.status()
            current_status = state.status()
            if previous_status != current_status:
                state.last_change = WorkflowExecutionState.new(
                    state.execution_id, state.flow_id
                ).last_change
            elif not state.last_change:
                state.last_change = previous.last_change
        else:
            if not state.created_at:
                state.created_at = WorkflowExecutionState.new(
                    state.execution_id, state.flow_id
                ).created_at
            if not state.last_change:
                state.last_change = state.created_at

        self._backend.upsert_record(state.to_record())

    def get_or_create(self, execution_id: str, flow_id: str) -> WorkflowExecutionState:
        existing = self.load_execution(execution_id)
        if existing:
            return existing
        state = WorkflowExecutionState.new(execution_id, flow_id)
        self.save_state(state)
        return state

    def list_workflows(
        self,
        owner_id: str,
        *,
        limit: int,
        before: Optional[str] = None,
        before_id: Optional[str] = None,
        after: Optional[str] = None,
        after_id: Optional[str] = None,
    ) -> list[WorkflowExecutionState]:
        rows = self._backend.list_rows(
            owner_id,
            limit=limit,
            before=before,
            before_id=before_id,
            after=after,
            after_id=after_id,
        )

        results: list[WorkflowExecutionState] = []
        for row in rows:
            state = WorkflowExecutionState.from_row(row)
            if state:
                results.append(state)
        return results


__all__ = ["WorkflowExecutionState", "WorkflowStateStore", "_load_psycopg"]
