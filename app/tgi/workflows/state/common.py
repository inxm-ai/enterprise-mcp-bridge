import json
from datetime import datetime, timezone
from typing import Optional

from app.tgi.workflows.models import WorkflowExecutionState

WORKFLOW_TABLE = "workflow_executions"
WORKFLOW_COLUMNS = (
    "execution_id, flow_id, context_json, events_json, current_agent, "
    "completed, awaiting_feedback, owner_id, created_at, last_change"
)


def normalize_workflow_timestamp(value: Optional[str]) -> Optional[str]:
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


def resolve_missing_state_fields(
    execution_id: str,
    flow_id: str,
    context_json: str,
    owner_id: Optional[str],
    created_at: Optional[str],
    last_change: Optional[str],
) -> tuple[Optional[str], str, str]:
    context = json.loads(context_json) if context_json else {"agents": {}}
    resolved_owner = owner_id or context.get("_workflow_owner_id")
    resolved_created_at = normalize_workflow_timestamp(
        created_at or context.get("created_at")
    )
    resolved_last_change = (
        normalize_workflow_timestamp(last_change or context.get("last_change"))
        or resolved_created_at
    )
    if not resolved_created_at:
        resolved_created_at = WorkflowExecutionState.new(
            execution_id, flow_id or "unknown"
        ).created_at
    if not resolved_last_change:
        resolved_last_change = resolved_created_at
    return resolved_owner, resolved_created_at, resolved_last_change
