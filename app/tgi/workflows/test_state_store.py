from app.tgi.workflows.models import WorkflowExecutionState
from app.tgi.workflows.state import WorkflowStateStore


def _owned_state(exec_id: str, flow_id: str, owner: str, created_at: str):
    state = WorkflowExecutionState.new(exec_id, flow_id)
    state.owner_id = owner
    state.context["_workflow_owner_id"] = owner
    state.created_at = created_at
    return state


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
