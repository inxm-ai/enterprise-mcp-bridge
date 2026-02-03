import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.tgi.workflows.models import WorkflowExecutionState

logger = logging.getLogger("uvicorn.error")


def _normalize_timestamp(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (
            dt.astimezone(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
    except ValueError:
        return value


class WorkflowStateStore:
    """
    Lightweight SQLite-backed state store used to persist workflow executions.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._shared_conn: sqlite3.Connection | None = None
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if str(self.db_path) == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:")
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_executions (
                    execution_id TEXT PRIMARY KEY,
                    flow_id TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    events_json TEXT NOT NULL,
                    current_agent TEXT,
                    completed INTEGER NOT NULL DEFAULT 0,
                    awaiting_feedback INTEGER NOT NULL DEFAULT 0,
                    owner_id TEXT,
                    created_at TEXT,
                    last_change TEXT
                )
            """
            )
            conn.commit()
        self._ensure_columns()
        self._backfill_missing_fields()

    def _ensure_columns(self) -> None:
        columns = set()
        with self._connect() as conn:
            cur = conn.execute("PRAGMA table_info(workflow_executions)")
            for row in cur.fetchall():
                columns.add(row[1])

            if "owner_id" not in columns:
                conn.execute("ALTER TABLE workflow_executions ADD COLUMN owner_id TEXT")
            if "created_at" not in columns:
                conn.execute(
                    "ALTER TABLE workflow_executions ADD COLUMN created_at TEXT"
                )
            if "last_change" not in columns:
                conn.execute(
                    "ALTER TABLE workflow_executions ADD COLUMN last_change TEXT"
                )
            conn.commit()

    def _backfill_missing_fields(self) -> None:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT execution_id, flow_id, context_json, owner_id, created_at, last_change "
                "FROM workflow_executions "
                "WHERE owner_id IS NULL OR owner_id = '' OR created_at IS NULL OR created_at = '' "
                "OR last_change IS NULL OR last_change = ''"
            )
            rows = cur.fetchall()
            if not rows:
                return
            for (
                execution_id,
                flow_id,
                context_json,
                owner_id,
                created_at,
                last_change,
            ) in rows:
                context = json.loads(context_json) if context_json else {"agents": {}}
                resolved_owner = owner_id or context.get("_workflow_owner_id")
                resolved_created_at = _normalize_timestamp(
                    created_at or context.get("created_at")
                )
                resolved_last_change = (
                    _normalize_timestamp(last_change or context.get("last_change"))
                    or resolved_created_at
                )
                if not resolved_created_at:
                    resolved_created_at = WorkflowExecutionState.new(
                        execution_id, flow_id or "unknown"
                    ).created_at
                if not resolved_last_change:
                    resolved_last_change = resolved_created_at
                conn.execute(
                    "UPDATE workflow_executions SET owner_id = ?, created_at = ?, last_change = ? WHERE execution_id = ?",
                    (
                        resolved_owner,
                        resolved_created_at,
                        resolved_last_change,
                        execution_id,
                    ),
                )
            conn.commit()

    def load_execution(self, execution_id: str) -> Optional[WorkflowExecutionState]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT execution_id, flow_id, context_json, events_json, current_agent, completed, awaiting_feedback, owner_id, created_at, last_change "
                "FROM workflow_executions WHERE execution_id = ?",
                (execution_id,),
            )
            row = cur.fetchone()
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

        record = state.to_record()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO workflow_executions
                (execution_id, flow_id, context_json, events_json, current_agent, completed, awaiting_feedback, owner_id, created_at, last_change)
                VALUES (:execution_id, :flow_id, :context_json, :events_json, :current_agent, :completed, :awaiting_feedback, :owner_id, :created_at, :last_change)
                ON CONFLICT(execution_id) DO UPDATE SET
                    flow_id=excluded.flow_id,
                    context_json=excluded.context_json,
                    events_json=excluded.events_json,
                    current_agent=excluded.current_agent,
                    completed=excluded.completed,
                    awaiting_feedback=excluded.awaiting_feedback,
                    owner_id=excluded.owner_id,
                    created_at=excluded.created_at,
                    last_change=excluded.last_change
                """,
                record,
            )
            conn.commit()

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
        """
        List workflow executions for a specific owner, ordered by created_at desc.
        """
        clauses = ["owner_id = ?"]
        params: list[object] = [owner_id]
        if after and after_id:
            clauses.append("(created_at > ? OR (created_at = ? AND execution_id > ?))")
            params.extend([after, after, after_id])
        elif after:
            clauses.append("created_at > ?")
            params.append(after)
        if before and before_id:
            clauses.append("(created_at < ? OR (created_at = ? AND execution_id < ?))")
            params.extend([before, before, before_id])
        elif before:
            clauses.append("created_at < ?")
            params.append(before)

        where_clause = " AND ".join(clauses)
        query = (
            "SELECT execution_id, flow_id, context_json, events_json, current_agent, completed, awaiting_feedback, owner_id, created_at, last_change "
            f"FROM workflow_executions WHERE {where_clause} "
            "ORDER BY created_at DESC, execution_id DESC LIMIT ?"
        )
        params.append(limit)

        with self._connect() as conn:
            cur = conn.execute(query, params)
            rows = cur.fetchall()

        results: list[WorkflowExecutionState] = []
        for row in rows:
            state = WorkflowExecutionState.from_row(row)
            if state:
                results.append(state)
        return results
