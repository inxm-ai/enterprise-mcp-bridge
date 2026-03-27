import sqlite3
from pathlib import Path
from typing import Optional

from app.tgi.workflows.state.common import (
    WORKFLOW_COLUMNS,
    WORKFLOW_TABLE,
    resolve_missing_state_fields,
)


class SQLiteWorkflowStateBackend:
    backend_name = "sqlite"

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._shared_conn: sqlite3.Connection | None = None
        if str(self.db_path) != ":memory:" and not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if str(self.db_path) == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:")
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        return sqlite3.connect(self.db_path)

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {WORKFLOW_TABLE} (
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
            cur = conn.execute(f"PRAGMA table_info({WORKFLOW_TABLE})")
            for row in cur.fetchall():
                columns.add(row[1])

            if "owner_id" not in columns:
                conn.execute(f"ALTER TABLE {WORKFLOW_TABLE} ADD COLUMN owner_id TEXT")
            if "created_at" not in columns:
                conn.execute(f"ALTER TABLE {WORKFLOW_TABLE} ADD COLUMN created_at TEXT")
            if "last_change" not in columns:
                conn.execute(
                    f"ALTER TABLE {WORKFLOW_TABLE} ADD COLUMN last_change TEXT"
                )
            conn.commit()

    def _backfill_missing_fields(self) -> None:
        with self._connect() as conn:
            cur = conn.execute(
                f"SELECT execution_id, flow_id, context_json, owner_id, created_at, last_change "
                f"FROM {WORKFLOW_TABLE} "
                "WHERE owner_id IS NULL OR owner_id = '' OR created_at IS NULL OR created_at = '' "
                "OR last_change IS NULL OR last_change = ''"
            )
            rows = cur.fetchall()
            if not rows:
                return
            self._backfill_rows(conn, rows)
            conn.commit()

    def load_row(self, execution_id: str) -> Optional[tuple]:
        with self._connect() as conn:
            cur = conn.execute(
                f"SELECT {WORKFLOW_COLUMNS} FROM {WORKFLOW_TABLE} WHERE execution_id = ?",
                (execution_id,),
            )
            return cur.fetchone()

    def upsert_record(self, record: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {WORKFLOW_TABLE}
                ({WORKFLOW_COLUMNS})
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

    def list_rows(
        self,
        owner_id: str,
        *,
        limit: int,
        before: Optional[str] = None,
        before_id: Optional[str] = None,
        after: Optional[str] = None,
        after_id: Optional[str] = None,
    ) -> list[tuple]:
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
            f"SELECT {WORKFLOW_COLUMNS} FROM {WORKFLOW_TABLE} WHERE {where_clause} "
            "ORDER BY created_at DESC, execution_id DESC LIMIT ?"
        )
        params.append(limit)

        with self._connect() as conn:
            cur = conn.execute(query, params)
            return cur.fetchall()

    def _backfill_rows(self, conn: sqlite3.Connection, rows: list[tuple]) -> None:
        for (
            execution_id,
            flow_id,
            context_json,
            owner_id,
            created_at,
            last_change,
        ) in rows:
            resolved_owner, resolved_created_at, resolved_last_change = (
                resolve_missing_state_fields(
                    execution_id,
                    flow_id,
                    context_json,
                    owner_id,
                    created_at,
                    last_change,
                )
            )
            conn.execute(
                f"UPDATE {WORKFLOW_TABLE} SET owner_id = ?, created_at = ?, last_change = ? WHERE execution_id = ?",
                (
                    resolved_owner,
                    resolved_created_at,
                    resolved_last_change,
                    execution_id,
                ),
            )
