from __future__ import annotations

from app.tgi.workflows.models import WorkflowExecutionState
import app.tgi.workflows.state as state_module
from app.tgi.workflows.state import WorkflowStateStore


def _owned_state(exec_id: str, flow_id: str, owner: str, created_at: str):
    state = WorkflowExecutionState.new(exec_id, flow_id)
    state.owner_id = owner
    state.context["_workflow_owner_id"] = owner
    state.created_at = created_at
    return state


class _FakePsycopgModule:
    def __init__(self):
        self.rows: dict[str, tuple] = {}
        self.connect_calls: list[str] = []
        self.queries: list[str] = []

    def connect(self, dsn: str):
        self.connect_calls.append(dsn)
        return _FakePostgresConnection(self)


class _FakePostgresConnection:
    def __init__(self, module: _FakePsycopgModule):
        self.module = module

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return _FakePostgresCursor(self.module)


class _FakePostgresCursor:
    def __init__(self, module: _FakePsycopgModule):
        self.module = module
        self._result: tuple | None = None
        self._results: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query: str, params=None):
        normalized = " ".join(query.split())
        self.module.queries.append(normalized)

        if normalized.startswith("CREATE TABLE IF NOT EXISTS workflow_executions"):
            self._result = None
            self._results = []
            return

        if normalized.startswith(
            "ALTER TABLE workflow_executions ADD COLUMN IF NOT EXISTS"
        ):
            self._result = None
            self._results = []
            return

        if normalized.startswith(
            "SELECT execution_id, flow_id, context_json, owner_id, created_at, last_change FROM workflow_executions"
        ):
            self._results = [
                (row[0], row[1], row[2], row[7], row[8], row[9])
                for row in self.module.rows.values()
                if row[7] in (None, "")
                or row[8] in (None, "")
                or row[9] in (None, "")
            ]
            self._result = None
            return

        if normalized.startswith(
            "UPDATE workflow_executions SET owner_id = %s, created_at = %s, last_change = %s WHERE execution_id = %s"
        ):
            owner_id, created_at, last_change, execution_id = params
            row = list(self.module.rows[execution_id])
            row[7] = owner_id
            row[8] = created_at
            row[9] = last_change
            self.module.rows[execution_id] = tuple(row)
            self._result = None
            self._results = []
            return

        if normalized.startswith(
            "SELECT execution_id, flow_id, context_json, events_json, current_agent, completed, awaiting_feedback, owner_id, created_at, last_change FROM workflow_executions WHERE execution_id = %s"
        ):
            self._result = self.module.rows.get(params[0])
            self._results = []
            return

        if normalized.startswith("INSERT INTO workflow_executions"):
            record = params
            self.module.rows[record["execution_id"]] = (
                record["execution_id"],
                record["flow_id"],
                record["context_json"],
                record["events_json"],
                record["current_agent"],
                record["completed"],
                record["awaiting_feedback"],
                record["owner_id"],
                record["created_at"],
                record["last_change"],
            )
            self._result = None
            self._results = []
            return

        if (
            normalized.startswith(
                "SELECT execution_id, flow_id, context_json, events_json, current_agent, completed, awaiting_feedback, owner_id, created_at, last_change FROM workflow_executions WHERE owner_id = %s"
            )
            and normalized.endswith("ORDER BY created_at DESC, execution_id DESC LIMIT %s")
        ):
            self._results = self._filter_list_rows(normalized, params)
            self._result = None
            return

        raise AssertionError(f"Unhandled query in fake psycopg module: {normalized}")

    def _filter_list_rows(self, normalized: str, params) -> list[tuple]:
        owner_id = params[0]
        idx = 1
        after = None
        after_id = None
        before = None
        before_id = None

        if "(created_at > %s OR (created_at = %s AND execution_id > %s))" in normalized:
            after = params[idx]
            after_id = params[idx + 2]
            idx += 3
        elif "created_at > %s" in normalized:
            after = params[idx]
            idx += 1

        if "(created_at < %s OR (created_at = %s AND execution_id < %s))" in normalized:
            before = params[idx]
            before_id = params[idx + 2]
            idx += 3
        elif "created_at < %s" in normalized:
            before = params[idx]
            idx += 1

        limit = params[idx]
        rows = [row for row in self.module.rows.values() if row[7] == owner_id]

        if after and after_id:
            rows = [
                row
                for row in rows
                if row[8] > after or (row[8] == after and row[0] > after_id)
            ]
        elif after:
            rows = [row for row in rows if row[8] > after]

        if before and before_id:
            rows = [
                row
                for row in rows
                if row[8] < before or (row[8] == before and row[0] < before_id)
            ]
        elif before:
            rows = [row for row in rows if row[8] < before]

        rows.sort(key=lambda row: (row[8], row[0]), reverse=True)
        return rows[:limit]

    def fetchone(self):
        return self._result

    def fetchall(self):
        return list(self._results)


def _fake_postgres_store(monkeypatch):
    fake_psycopg = _FakePsycopgModule()
    monkeypatch.setattr(state_module, "_load_psycopg", lambda: fake_psycopg)
    store = WorkflowStateStore(backend="postgres", database_url="postgresql://test/db")
    return store, fake_psycopg


def test_list_workflows_orders_and_pages_by_timestamp(tmp_path):
    store = WorkflowStateStore(db_path=tmp_path / "state.db")

    state_a = _owned_state("exec-1", "flow-a", "user-1", "2024-01-15T10:30:00.000000Z")
    state_b = _owned_state("exec-2", "flow-b", "user-1", "2024-01-15T10:30:00.000000Z")
    state_c = _owned_state("exec-3", "flow-c", "user-1", "2024-01-14T10:30:00.000000Z")
    state_other = _owned_state(
        "exec-9", "flow-x", "user-2", "2024-01-16T09:00:00.000000Z"
    )

    for state in (state_a, state_b, state_c, state_other):
        store.save_state(state)

    first_page = store.list_workflows(owner_id="user-1", limit=2)
    assert [state.execution_id for state in first_page] == ["exec-2", "exec-1"]

    second_page = store.list_workflows(
        owner_id="user-1",
        limit=2,
        before="2024-01-15T10:30:00.000000Z",
        before_id="exec-2",
    )
    assert [state.execution_id for state in second_page] == ["exec-1", "exec-3"]


def test_list_workflows_filters_after_timestamp(tmp_path):
    store = WorkflowStateStore(db_path=tmp_path / "state.db")

    state_a = _owned_state("exec-1", "flow-a", "user-1", "2024-01-15T10:30:00.000000Z")
    state_b = _owned_state("exec-2", "flow-b", "user-1", "2024-01-15T10:30:00.000000Z")
    state_c = _owned_state("exec-3", "flow-c", "user-1", "2024-01-14T10:30:00.000000Z")
    state_other = _owned_state(
        "exec-9", "flow-x", "user-2", "2024-01-16T09:00:00.000000Z"
    )

    for state in (state_a, state_b, state_c, state_other):
        store.save_state(state)

    newer = store.list_workflows(
        owner_id="user-1",
        limit=10,
        after="2024-01-14T10:30:00.000000Z",
    )
    assert [state.execution_id for state in newer] == ["exec-2", "exec-1"]

    newer_with_id = store.list_workflows(
        owner_id="user-1",
        limit=10,
        after="2024-01-15T10:30:00.000000Z",
        after_id="exec-1",
    )
    assert [state.execution_id for state in newer_with_id] == ["exec-2"]


def test_last_change_updates_on_status_change(tmp_path):
    store = WorkflowStateStore(db_path=tmp_path / "state.db")
    state = WorkflowExecutionState.new("exec-1", "flow-a")
    state.owner_id = "user-1"
    state.context["_workflow_owner_id"] = "user-1"
    state.created_at = "2024-01-15T10:30:00.000000Z"
    state.last_change = "2024-01-15T10:30:00.000000Z"
    store.save_state(state)

    state.awaiting_feedback = True
    store.save_state(state)
    updated = store.load_execution("exec-1")

    assert updated is not None
    assert updated.status() == "awaiting_feedback"
    assert updated.last_change != "2024-01-15T10:30:00.000000Z"

    last_change = updated.last_change
    updated.current_agent = "planner"
    store.save_state(updated)
    stable = store.load_execution("exec-1")
    assert stable is not None
    assert stable.last_change == last_change


def test_postgres_store_initializes_schema(monkeypatch):
    store, fake_psycopg = _fake_postgres_store(monkeypatch)

    assert store.backend_name == "postgres"
    assert fake_psycopg.connect_calls
    assert any(
        query.startswith("CREATE TABLE IF NOT EXISTS workflow_executions")
        for query in fake_psycopg.queries
    )
    assert any(
        query.startswith("ALTER TABLE workflow_executions ADD COLUMN IF NOT EXISTS owner_id")
        for query in fake_psycopg.queries
    )


def test_postgres_store_round_trip_and_get_or_create(monkeypatch):
    store, _ = _fake_postgres_store(monkeypatch)
    state = _owned_state("exec-1", "flow-a", "user-1", "2024-01-15T10:30:00.000000Z")
    store.save_state(state)

    loaded = store.load_execution("exec-1")
    assert loaded is not None
    assert loaded.execution_id == "exec-1"
    assert loaded.owner_id == "user-1"

    existing = store.get_or_create("exec-1", "ignored-flow")
    assert existing.execution_id == "exec-1"
    created = store.get_or_create("exec-2", "flow-b")
    assert created.execution_id == "exec-2"
    assert created.flow_id == "flow-b"


def test_postgres_list_workflows_orders_and_pages_by_timestamp(monkeypatch):
    store, _ = _fake_postgres_store(monkeypatch)

    state_a = _owned_state("exec-1", "flow-a", "user-1", "2024-01-15T10:30:00.000000Z")
    state_b = _owned_state("exec-2", "flow-b", "user-1", "2024-01-15T10:30:00.000000Z")
    state_c = _owned_state("exec-3", "flow-c", "user-1", "2024-01-14T10:30:00.000000Z")
    state_other = _owned_state(
        "exec-9", "flow-x", "user-2", "2024-01-16T09:00:00.000000Z"
    )

    for state in (state_a, state_b, state_c, state_other):
        store.save_state(state)

    first_page = store.list_workflows(owner_id="user-1", limit=2)
    assert [state.execution_id for state in first_page] == ["exec-2", "exec-1"]

    second_page = store.list_workflows(
        owner_id="user-1",
        limit=2,
        before="2024-01-15T10:30:00.000000Z",
        before_id="exec-2",
    )
    assert [state.execution_id for state in second_page] == ["exec-1", "exec-3"]


def test_postgres_list_workflows_filters_after_timestamp(monkeypatch):
    store, _ = _fake_postgres_store(monkeypatch)

    state_a = _owned_state("exec-1", "flow-a", "user-1", "2024-01-15T10:30:00.000000Z")
    state_b = _owned_state("exec-2", "flow-b", "user-1", "2024-01-15T10:30:00.000000Z")
    state_c = _owned_state("exec-3", "flow-c", "user-1", "2024-01-14T10:30:00.000000Z")
    state_other = _owned_state(
        "exec-9", "flow-x", "user-2", "2024-01-16T09:00:00.000000Z"
    )

    for state in (state_a, state_b, state_c, state_other):
        store.save_state(state)

    newer = store.list_workflows(
        owner_id="user-1",
        limit=10,
        after="2024-01-14T10:30:00.000000Z",
    )
    assert [state.execution_id for state in newer] == ["exec-2", "exec-1"]

    newer_with_id = store.list_workflows(
        owner_id="user-1",
        limit=10,
        after="2024-01-15T10:30:00.000000Z",
        after_id="exec-1",
    )
    assert [state.execution_id for state in newer_with_id] == ["exec-2"]


def test_postgres_last_change_updates_on_status_change(monkeypatch):
    store, _ = _fake_postgres_store(monkeypatch)
    state = WorkflowExecutionState.new("exec-1", "flow-a")
    state.owner_id = "user-1"
    state.context["_workflow_owner_id"] = "user-1"
    state.created_at = "2024-01-15T10:30:00.000000Z"
    state.last_change = "2024-01-15T10:30:00.000000Z"
    store.save_state(state)

    state.awaiting_feedback = True
    store.save_state(state)
    updated = store.load_execution("exec-1")

    assert updated is not None
    assert updated.status() == "awaiting_feedback"
    assert updated.last_change != "2024-01-15T10:30:00.000000Z"
