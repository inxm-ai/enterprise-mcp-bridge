import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class WorkflowAgentDef:
    """
    Definition of an agent within a workflow.

    Attributes:
        agent: Unique identifier for the agent within the workflow.
        description: Human-readable description of what the agent does.
        pass_through: Controls response visibility. Can be:
            - False: Don't show intermediate responses (default)
            - True: Show all responses
            - str: Show responses with this specific guideline/instruction
        depends_on: List of agent names that must complete before this agent runs.
        when: Optional condition expression for whether this agent should run.
        reroute: Configuration for rerouting to other agents based on conditions.
            Supports tool-result triggers with `on` entries like
            "tool:tool_name:success" or "tool:tool_name:error".
        tools: Tool configurations. Can be:
            - None: Use all available tools
            - []: Disable tools
            - List of str: Use only these tool names
            - List of dict: Advanced tool configs with settings/args, e.g.:
              {"plan": {"settings": {"streaming": true}, "args": {"x": "agent.field"}}}
        returns: List of field names to extract from tool results and store in context.
        on_tool_error: Agent name to reroute to when a tool call fails. If set and a
            tool error is detected, the workflow will automatically reroute to this
            agent even if the LLM doesn't emit a reroute tag.
        context: Controls how much workflow context is provided to the agent:
            - True (default): full context is provided (current behavior)
            - False: no workflow context is provided
            - List[str]: only the referenced fields from other agents are provided,
              using the same notation as arg mappings (e.g., "agent.field.nested").
            - "user_prompt": passes only the original user prompt captured when the
              workflow started.
        stop_point: If True, no further agents will be executed after this agent
            completes. Useful for terminal agents like summaries or error handlers.
    """

    agent: str
    description: str
    pass_through: Union[bool, str] = False
    context: Union[bool, List[str], str] = True
    depends_on: List[str] = field(default_factory=list)
    when: Optional[str] = None
    reroute: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    tools: Optional[List[Union[str, Dict[str, Any]]]] = None
    returns: Optional[List[str]] = None
    on_tool_error: Optional[str] = None
    stop_point: bool = False

    @property
    def should_pass_through(self) -> bool:
        """Whether responses should be passed through to the user."""
        return bool(self.pass_through)

    @property
    def pass_through_guideline(self) -> Optional[str]:
        """
        Get the pass-through guideline if specified as a string.

        Returns:
            The guideline string if pass_through is a string, None otherwise.
        """
        if isinstance(self.pass_through, str):
            return self.pass_through
        return None


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
