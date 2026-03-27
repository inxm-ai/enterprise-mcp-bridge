from typing import Callable, Optional

from app.tgi.workflows.state.common import (
    WORKFLOW_COLUMNS,
    WORKFLOW_TABLE,
    resolve_missing_state_fields,
)


class PostgresWorkflowStateBackend:
    backend_name = "postgres"

    def __init__(self, database_url: str, *, psycopg_loader: Callable):
        self.database_url = database_url
        self._psycopg_loader = psycopg_loader
        self.ensure_schema()

    def _connect(self):
        psycopg = self._psycopg_loader()
        return psycopg.connect(self.database_url)

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
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
                cur.execute(
                    f"ALTER TABLE {WORKFLOW_TABLE} ADD COLUMN IF NOT EXISTS owner_id TEXT"
                )
                cur.execute(
                    f"ALTER TABLE {WORKFLOW_TABLE} ADD COLUMN IF NOT EXISTS created_at TEXT"
                )
                cur.execute(
                    f"ALTER TABLE {WORKFLOW_TABLE} ADD COLUMN IF NOT EXISTS last_change TEXT"
                )
        self._backfill_missing_fields()

    def _backfill_missing_fields(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT execution_id, flow_id, context_json, owner_id, created_at, last_change "
                    f"FROM {WORKFLOW_TABLE} "
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
                    cur.execute(
                        f"UPDATE {WORKFLOW_TABLE} SET owner_id = %s, created_at = %s, last_change = %s WHERE execution_id = %s",
                        (
                            resolved_owner,
                            resolved_created_at,
                            resolved_last_change,
                            execution_id,
                        ),
                    )

    def load_row(self, execution_id: str) -> Optional[tuple]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {WORKFLOW_COLUMNS} FROM {WORKFLOW_TABLE} WHERE execution_id = %s",
                    (execution_id,),
                )
                return cur.fetchone()

    def upsert_record(self, record: dict) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {WORKFLOW_TABLE}
                    ({WORKFLOW_COLUMNS})
                    VALUES (
                        %(execution_id)s,
                        %(flow_id)s,
                        %(context_json)s,
                        %(events_json)s,
                        %(current_agent)s,
                        %(completed)s,
                        %(awaiting_feedback)s,
                        %(owner_id)s,
                        %(created_at)s,
                        %(last_change)s
                    )
                    ON CONFLICT (execution_id) DO UPDATE SET
                        flow_id = EXCLUDED.flow_id,
                        context_json = EXCLUDED.context_json,
                        events_json = EXCLUDED.events_json,
                        current_agent = EXCLUDED.current_agent,
                        completed = EXCLUDED.completed,
                        awaiting_feedback = EXCLUDED.awaiting_feedback,
                        owner_id = EXCLUDED.owner_id,
                        created_at = EXCLUDED.created_at,
                        last_change = EXCLUDED.last_change
                    """,
                    record,
                )

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
        clauses = ["owner_id = %s"]
        params: list[object] = [owner_id]
        if after and after_id:
            clauses.append(
                "(created_at > %s OR (created_at = %s AND execution_id > %s))"
            )
            params.extend([after, after, after_id])
        elif after:
            clauses.append("created_at > %s")
            params.append(after)
        if before and before_id:
            clauses.append(
                "(created_at < %s OR (created_at = %s AND execution_id < %s))"
            )
            params.extend([before, before, before_id])
        elif before:
            clauses.append("created_at < %s")
            params.append(before)

        where_clause = " AND ".join(clauses)
        query = (
            f"SELECT {WORKFLOW_COLUMNS} FROM {WORKFLOW_TABLE} WHERE {where_clause} "
            "ORDER BY created_at DESC, execution_id DESC LIMIT %s"
        )
        params.append(limit)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()
