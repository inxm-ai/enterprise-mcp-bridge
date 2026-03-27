"""
Integration tests for the Postgres workflow state backend.

These tests spin up a real Postgres container via testcontainers and verify
that the actual SQL queries, schema migration, and store operations work
correctly against a live database — not just against a fake in-memory cursor.

Requires Docker to be running.  Tests are automatically skipped when the
`testcontainers` package is not installed or when Docker is unavailable.
"""
from __future__ import annotations

import pytest

from app.tgi.workflows.models import WorkflowExecutionState
from app.tgi.workflows.state import WorkflowStateStore

# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

POSTGRES_IMAGE = "postgres:16"


def _owned_state(exec_id: str, flow_id: str, owner: str, created_at: str) -> WorkflowExecutionState:
    state = WorkflowExecutionState.new(exec_id, flow_id)
    state.owner_id = owner
    state.context["_workflow_owner_id"] = owner
    state.created_at = created_at
    return state


# ---------------------------------------------------------------------------
# Module-scoped container fixture — started once, shared across all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def postgres_url():
    """
    Start a Postgres container and yield a plain `postgresql://` connection URL
    suitable for psycopg3.  The container is stopped when all tests in the
    module have finished.

    The fixture is skipped automatically if Docker is unavailable or if the
    `testcontainers` package is not installed.
    """
    pytest.importorskip(
        "testcontainers.postgres",
        reason="testcontainers[postgres] is not installed",
    )
    from testcontainers.postgres import PostgresContainer

    try:
        with PostgresContainer(POSTGRES_IMAGE, driver=None) as container:
            yield container.get_connection_url()
    except Exception as exc:
        pytest.skip(f"Docker unavailable or container failed to start: {exc}")


@pytest.fixture
def store(postgres_url: str) -> WorkflowStateStore:
    """A fresh WorkflowStateStore backed by the shared Postgres container."""
    return WorkflowStateStore(backend="postgres", database_url=postgres_url)


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


def test_schema_is_created_on_init(store: WorkflowStateStore) -> None:
    """
    The store must create the `workflow_executions` table on first connect so
    that subsequent reads and writes succeed without any manual migration step.
    """
    assert store.backend_name == "postgres"
    # Round-trip a record — this proves the table exists
    state = WorkflowExecutionState.new("schema-check", "flow-x")
    store.save_state(state)
    loaded = store.load_execution("schema-check")
    assert loaded is not None
    assert loaded.execution_id == "schema-check"


def test_schema_init_is_idempotent(postgres_url: str) -> None:
    """
    Creating a second store against the same database must not raise even
    though the table already exists (CREATE TABLE IF NOT EXISTS + ADD COLUMN
    IF NOT EXISTS must both be idempotent).
    """
    WorkflowStateStore(backend="postgres", database_url=postgres_url)
    WorkflowStateStore(backend="postgres", database_url=postgres_url)


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


def test_round_trip_save_and_load(store: WorkflowStateStore) -> None:
    state = _owned_state("rt-1", "flow-a", "user-1", "2024-06-01T12:00:00.000000Z")
    store.save_state(state)

    loaded = store.load_execution("rt-1")
    assert loaded is not None
    assert loaded.execution_id == "rt-1"
    assert loaded.flow_id == "flow-a"
    assert loaded.owner_id == "user-1"


def test_load_returns_none_for_missing_execution(store: WorkflowStateStore) -> None:
    assert store.load_execution("does-not-exist") is None


def test_upsert_updates_existing_record(store: WorkflowStateStore) -> None:
    state = _owned_state("upsert-1", "flow-a", "user-1", "2024-06-01T12:00:00.000000Z")
    store.save_state(state)

    state.current_agent = "planner"
    store.save_state(state)

    loaded = store.load_execution("upsert-1")
    assert loaded is not None
    assert loaded.current_agent == "planner"


# ---------------------------------------------------------------------------
# get_or_create
# ---------------------------------------------------------------------------


def test_get_or_create_returns_existing(store: WorkflowStateStore) -> None:
    state = _owned_state("goc-1", "flow-a", "user-1", "2024-06-01T12:00:00.000000Z")
    store.save_state(state)

    existing = store.get_or_create("goc-1", "ignored-flow")
    assert existing.execution_id == "goc-1"
    assert existing.flow_id == "flow-a"  # original, not "ignored-flow"


def test_get_or_create_creates_new(store: WorkflowStateStore) -> None:
    created = store.get_or_create("goc-new", "flow-b")
    assert created.execution_id == "goc-new"
    assert created.flow_id == "flow-b"
    assert store.load_execution("goc-new") is not None


# ---------------------------------------------------------------------------
# last_change tracking
# ---------------------------------------------------------------------------


def test_last_change_updates_when_status_changes(store: WorkflowStateStore) -> None:
    state = WorkflowExecutionState.new("lc-1", "flow-a")
    state.owner_id = "user-1"
    state.context["_workflow_owner_id"] = "user-1"
    state.created_at = "2024-01-15T10:30:00.000000Z"
    state.last_change = "2024-01-15T10:30:00.000000Z"
    store.save_state(state)

    state.awaiting_feedback = True
    store.save_state(state)

    updated = store.load_execution("lc-1")
    assert updated is not None
    assert updated.status() == "awaiting_feedback"
    assert updated.last_change != "2024-01-15T10:30:00.000000Z"


def test_last_change_stable_when_status_unchanged(store: WorkflowStateStore) -> None:
    state = WorkflowExecutionState.new("lc-2", "flow-a")
    state.owner_id = "user-1"
    state.context["_workflow_owner_id"] = "user-1"
    state.created_at = "2024-01-15T10:30:00.000000Z"
    state.last_change = "2024-01-15T10:30:00.000000Z"
    store.save_state(state)

    state.awaiting_feedback = True
    store.save_state(state)
    after_first_status_change = store.load_execution("lc-2")
    assert after_first_status_change is not None
    recorded_last_change = after_first_status_change.last_change

    after_first_status_change.current_agent = "planner"
    store.save_state(after_first_status_change)

    stable = store.load_execution("lc-2")
    assert stable is not None
    assert stable.last_change == recorded_last_change


# ---------------------------------------------------------------------------
# Pagination — list_workflows
# ---------------------------------------------------------------------------


def test_list_workflows_orders_by_timestamp_desc(store: WorkflowStateStore) -> None:
    owner = "list-user-order"
    state_a = _owned_state("lo-1", "flow-a", owner, "2024-01-15T10:30:00.000000Z")
    state_b = _owned_state("lo-2", "flow-b", owner, "2024-01-15T10:30:00.000000Z")
    state_c = _owned_state("lo-3", "flow-c", owner, "2024-01-14T10:30:00.000000Z")
    state_other = _owned_state("lo-9", "flow-x", "other-user", "2024-01-16T09:00:00.000000Z")

    for s in (state_a, state_b, state_c, state_other):
        store.save_state(s)

    first_page = store.list_workflows(owner_id=owner, limit=2)
    assert [s.execution_id for s in first_page] == ["lo-2", "lo-1"]


def test_list_workflows_before_cursor_pagination(store: WorkflowStateStore) -> None:
    owner = "list-user-before"
    state_a = _owned_state("lb-1", "flow-a", owner, "2024-01-15T10:30:00.000000Z")
    state_b = _owned_state("lb-2", "flow-b", owner, "2024-01-15T10:30:00.000000Z")
    state_c = _owned_state("lb-3", "flow-c", owner, "2024-01-14T10:30:00.000000Z")

    for s in (state_a, state_b, state_c):
        store.save_state(s)

    second_page = store.list_workflows(
        owner_id=owner,
        limit=2,
        before="2024-01-15T10:30:00.000000Z",
        before_id="lb-2",
    )
    assert [s.execution_id for s in second_page] == ["lb-1", "lb-3"]


def test_list_workflows_after_cursor_pagination(store: WorkflowStateStore) -> None:
    owner = "list-user-after"
    state_a = _owned_state("la-1", "flow-a", owner, "2024-01-15T10:30:00.000000Z")
    state_b = _owned_state("la-2", "flow-b", owner, "2024-01-15T10:30:00.000000Z")
    state_c = _owned_state("la-3", "flow-c", owner, "2024-01-14T10:30:00.000000Z")

    for s in (state_a, state_b, state_c):
        store.save_state(s)

    newer = store.list_workflows(
        owner_id=owner,
        limit=10,
        after="2024-01-14T10:30:00.000000Z",
    )
    assert [s.execution_id for s in newer] == ["la-2", "la-1"]


def test_list_workflows_after_cursor_with_id_tiebreak(store: WorkflowStateStore) -> None:
    owner = "list-user-after-id"
    state_a = _owned_state("lai-1", "flow-a", owner, "2024-01-15T10:30:00.000000Z")
    state_b = _owned_state("lai-2", "flow-b", owner, "2024-01-15T10:30:00.000000Z")

    for s in (state_a, state_b):
        store.save_state(s)

    newer_with_id = store.list_workflows(
        owner_id=owner,
        limit=10,
        after="2024-01-15T10:30:00.000000Z",
        after_id="lai-1",
    )
    assert [s.execution_id for s in newer_with_id] == ["lai-2"]


def test_list_workflows_excludes_other_owners(store: WorkflowStateStore) -> None:
    owner_a = "iso-user-a"
    owner_b = "iso-user-b"
    store.save_state(_owned_state("iso-1", "flow", owner_a, "2024-06-01T12:00:00.000000Z"))
    store.save_state(_owned_state("iso-2", "flow", owner_b, "2024-06-01T12:00:00.000000Z"))

    results = store.list_workflows(owner_id=owner_a, limit=10)
    assert all(s.owner_id == owner_a for s in results)
    assert not any(s.execution_id == "iso-2" for s in results)
