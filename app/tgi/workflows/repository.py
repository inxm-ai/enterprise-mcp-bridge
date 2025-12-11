import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional

from app.tgi.workflows.models import WorkflowAgentDef, WorkflowDefinition

logger = logging.getLogger("uvicorn.error")


class WorkflowRepository:
    """
    Load workflow definitions from a directory pointed to by WORKFLOWS_PATH.
    """

    def __init__(self, workflows_path: Optional[str] = None):
        base_path = workflows_path or os.environ.get("WORKFLOWS_PATH")
        if not base_path:
            raise ValueError("WORKFLOWS_PATH must be configured to use workflows.")
        path_obj = Path(base_path)
        if not path_obj.exists() or not path_obj.is_dir():
            raise ValueError(
                f"Workflow path '{base_path}' does not exist or is not a directory."
            )
        self.base_path = path_obj
        self._definitions = self._load_definitions()

    def _load_definitions(self) -> Dict[str, WorkflowDefinition]:
        definitions: Dict[str, WorkflowDefinition] = {}
        for file in self.base_path.glob("*.json"):
            try:
                payload = json.loads(file.read_text(encoding="utf-8"))
                flow_id = payload.get("flow_id")
                root_intent = payload.get("root_intent") or ""
                agents_payload = payload.get("agents") or []
                if not flow_id or not root_intent or not agents_payload:
                    logger.debug(
                        f"[WorkflowRepository] Skipping incomplete workflow file {file}"
                    )
                    continue
                agents = []
                for agent in agents_payload:
                    tools_field = agent.get("tools", None)
                    tools_value = list(tools_field) if tools_field is not None else None
                    # pass_through can be boolean or string (guideline)
                    pass_through_value = agent.get("pass_through", False)
                    # context controls how much workflow context is provided
                    context_value = agent.get("context", True)
                    # returns is a list of field names to capture from tool results
                    returns_value = agent.get("returns", None)
                    # on_tool_error is the agent to reroute to on tool failure
                    on_tool_error_value = agent.get("on_tool_error", None)
                    agents.append(
                        WorkflowAgentDef(
                            agent=agent.get("agent"),
                            description=agent.get("description", ""),
                            pass_through=pass_through_value,
                            context=context_value,
                            depends_on=list(agent.get("depends_on", []) or []),
                            when=agent.get("when"),
                            reroute=agent.get("reroute"),
                            tools=tools_value,
                            returns=returns_value,
                            on_tool_error=on_tool_error_value,
                        )
                    )
                definitions[flow_id] = WorkflowDefinition(
                    flow_id=flow_id, root_intent=root_intent, agents=agents
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    f"[WorkflowRepository] Failed to load workflow from {file}: {exc}"
                )
        return definitions

    def get(self, flow_id: str) -> WorkflowDefinition:
        if flow_id not in self._definitions:
            raise ValueError(f"Workflow '{flow_id}' not found")
        return self._definitions[flow_id]

    def match_workflow(self, user_text: str) -> Optional[WorkflowDefinition]:
        """
        Naive intent matching: counts overlapping tokens between the user text and
        the workflow root intent (split on underscores/whitespace).
        """
        if not self._definitions:
            return None
        text = (user_text or "").lower()
        best_score = 0
        best_def = None
        for wf in self._definitions.values():
            tokens = [
                t for t in re.split(r"[^a-zA-Z0-9]+", wf.root_intent.lower()) if t
            ]
            score = sum(1 for token in tokens if token in text)
            if score > best_score:
                best_score = score
                best_def = wf
        return best_def if best_score else None

    def all(self) -> Dict[str, WorkflowDefinition]:
        return dict(self._definitions)
