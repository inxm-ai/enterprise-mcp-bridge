import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.tgi.protocols.chunk_reader import ChunkFormat
from app.tgi.workflows.models import WorkflowExecutionState
from app.vars import TGI_MODEL_NAME


class WorkflowChunkFormatter:
    """
    Formats workflow updates into streaming-compatible chunks.

    The formatter can emit OpenAI-style SSE chunks (used by the chat/completions
    endpoint) or JSON-RPC A2A envelopes. Agentic metadata is embedded in the
    OpenAI payload under the ``agentic`` key so downstream formatters (like the
    A2A converter) can preserve workflow context.
    """

    def __init__(
        self,
        model_name: str = TGI_MODEL_NAME,
        default_format: ChunkFormat = ChunkFormat.OPENAI,
    ):
        self.model_name = model_name
        self.default_format = default_format

    def ensure_task_id(self, state: WorkflowExecutionState, reset: bool = False) -> str:
        """
        Guarantee a task id is present in the workflow context.

        Args:
            state: Current workflow execution state.
            reset: When True, forces creation of a new task id (used after
                   user feedback to start a new task thread).

        Returns:
            The active task id string.
        """
        if reset or not state.context.get("task_id"):
            state.context["task_id"] = str(uuid.uuid4())
        return str(state.context["task_id"])

    def format_chunk(
        self,
        *,
        state: WorkflowExecutionState,
        content: str,
        status: str = "in_progress",
        role: str = "assistant",
        kind: str = "task",
        metadata: Optional[Dict[str, Any]] = None,
        finish_reason: Optional[str] = None,
        chunk_format: Optional[ChunkFormat] = None,
        request_id: Optional[str] = None,
        parts: Optional[list[dict[str, Any]]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build a single streaming chunk.

        Args:
            state: Workflow execution state (provides ids and stored task id).
            content: Human-readable content to stream.
            status: High-level workflow status (submitted, in_progress, completed, error).
            role: Message role for chat-style consumers.
            kind: Domain kind (e.g., "task").
            metadata: Extra metadata to attach to the agentic envelope.
            finish_reason: OpenAI finish reason when known.
            chunk_format: Target chunk format (defaults to default_format).
            request_id: Request id used for A2A JSON-RPC envelopes.
            parts: Optional history parts payload for A2A formatting.
            error: Optional error description payload.

        Returns:
            A formatted chunk string ready for SSE streaming.
        """
        fmt = chunk_format or self.default_format
        task_id = self.ensure_task_id(state)
        metadata = metadata or {}
        created = int(time.time())
        agentic_meta = {
            "context_id": state.execution_id,
            "workflow_id": state.flow_id,
            "task_id": task_id,
            "status": status,
            "kind": kind,
            "metadata": metadata,
            "role": role,
        }
        if error:
            agentic_meta["error"] = error

        delta: Dict[str, Any] = {"content": content}
        if role:
            delta["role"] = role

        payload: Dict[str, Any] = {
            "id": task_id,
            "model": self.model_name,
            "created": created,
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
            "agentic": agentic_meta,
        }

        if fmt == ChunkFormat.A2A:
            return self._to_a2a(payload, request_id or task_id, parts=parts)

        serialized = json.dumps(payload, ensure_ascii=False)
        # Guard SSE transport from malformed surrogate code points coming
        # from model output. Without this, UTF-8 encoding can fail mid-stream.
        safe_serialized = serialized.encode("utf-8", errors="replace").decode("utf-8")
        return f"data: {safe_serialized}\n\n"

    def _to_a2a(
        self,
        payload: Dict[str, Any],
        request_id: str,
        parts: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """
        Convert an OpenAI-style payload (with agentic metadata) into an A2A chunk.
        """
        agentic = payload.get("agentic", {}) if isinstance(payload, dict) else {}
        delta = (
            payload.get("choices", [{}])[0].get("delta", {})
            if isinstance(payload, dict)
            else {}
        )
        content = delta.get("content") or ""
        context_id = (
            agentic.get("context_id") or agentic.get("workflow_id") or payload.get("id")
        )
        task_id = agentic.get("task_id") or payload.get("id") or request_id
        history_parts = parts or [{"kind": "text", "text": content}]
        status_state = agentic.get("status") or "in_progress"

        history_entry = {
            "role": agentic.get("role") or "assistant",
            "parts": history_parts,
            "messageId": task_id,
            "taskId": task_id,
            "contextId": context_id,
        }

        result = {
            "id": task_id,
            "contextId": context_id,
            "status": {
                "state": status_state,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "history": [history_entry],
            "kind": agentic.get("kind") or "task",
            "metadata": agentic.get("metadata") or {},
        }

        if agentic.get("error"):
            result["error"] = agentic["error"]

        a2a_dict = {"jsonrpc": "2.0", "result": result, "id": request_id}
        return f"data: {json.dumps(a2a_dict, ensure_ascii=False)}\n\n"
