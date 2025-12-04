import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class WorkflowAgentDef:
    agent: str
    description: str
    pass_through: bool = False
    depends_on: List[str] = field(default_factory=list)
    when: Optional[str] = None
    reroute: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    tools: Optional[List[str]] = None


@dataclass
class WorkflowDefinition:
    flow_id: str
    root_intent: str
    agents: List[WorkflowAgentDef]


@dataclass
class WorkflowExecutionState:
    execution_id: str
    flow_id: str
    context: Dict[str, Any] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)
    current_agent: Optional[str] = None
    completed: bool = False
    awaiting_feedback: bool = False

    @classmethod
    def new(cls, execution_id: str, flow_id: str) -> "WorkflowExecutionState":
        # Always include an agents map to attach outputs in a consistent shape
        return cls(
            execution_id=execution_id,
            flow_id=flow_id,
            context={"agents": {}},
            events=[],
            current_agent=None,
            completed=False,
            awaiting_feedback=False,
        )

    def to_record(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "flow_id": self.flow_id,
            "context_json": json.dumps(self.context),
            "events_json": json.dumps(self.events),
            "current_agent": self.current_agent,
            "completed": int(self.completed),
            "awaiting_feedback": int(self.awaiting_feedback),
        }

    @classmethod
    def from_row(cls, row: Optional[tuple]) -> Optional["WorkflowExecutionState"]:
        if not row:
            return None
        (
            execution_id,
            flow_id,
            context_json,
            events_json,
            current_agent,
            completed,
            awaiting_feedback,
        ) = row
        context = json.loads(context_json) if context_json else {"agents": {}}
        # Ensure agents map exists
        context.setdefault("agents", {})
        return cls(
            execution_id=execution_id,
            flow_id=flow_id,
            context=context,
            events=json.loads(events_json) if events_json else [],
            current_agent=current_agent,
            completed=bool(completed),
            awaiting_feedback=bool(awaiting_feedback),
        )
