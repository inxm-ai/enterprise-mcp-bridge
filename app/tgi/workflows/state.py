import logging
import sqlite3
from pathlib import Path
from typing import Optional

from app.tgi.workflows.models import WorkflowExecutionState

logger = logging.getLogger("uvicorn.error")


class WorkflowStateStore:
    """
    Lightweight SQLite-backed state store used to persist workflow executions.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
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
                    awaiting_feedback INTEGER NOT NULL DEFAULT 0
                )
            """
            )
            conn.commit()

    def load_execution(self, execution_id: str) -> Optional[WorkflowExecutionState]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT execution_id, flow_id, context_json, events_json, current_agent, completed, awaiting_feedback "
                "FROM workflow_executions WHERE execution_id = ?",
                (execution_id,),
            )
            row = cur.fetchone()
        return WorkflowExecutionState.from_row(row)

    def save_state(self, state: WorkflowExecutionState) -> None:
        record = state.to_record()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO workflow_executions
                (execution_id, flow_id, context_json, events_json, current_agent, completed, awaiting_feedback)
                VALUES (:execution_id, :flow_id, :context_json, :events_json, :current_agent, :completed, :awaiting_feedback)
                ON CONFLICT(execution_id) DO UPDATE SET
                    flow_id=excluded.flow_id,
                    context_json=excluded.context_json,
                    events_json=excluded.events_json,
                    current_agent=excluded.current_agent,
                    completed=excluded.completed,
                    awaiting_feedback=excluded.awaiting_feedback
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
