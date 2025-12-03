import json

import pytest

from app.tgi.protocols.chunk_reader import ChunkFormat, chunk_reader
from app.tgi.workflows.chunk_formatter import WorkflowChunkFormatter
from app.tgi.workflows.models import WorkflowExecutionState


def _strip_prefix(chunk: str) -> str:
    return chunk.split("data: ", 1)[1].strip()


def test_openai_envelope_includes_agentic_metadata():
    state = WorkflowExecutionState.new("exec-openai", "flow-openai")
    formatter = WorkflowChunkFormatter()

    chunk = formatter.format_chunk(
        state=state, content="Hello agentic", status="submitted"
    )

    payload = json.loads(_strip_prefix(chunk))
    assert payload["choices"][0]["delta"]["content"] == "Hello agentic"
    assert payload["agentic"]["context_id"] == "exec-openai"
    assert payload["agentic"]["workflow_id"] == "flow-openai"
    assert payload["agentic"]["status"] == "submitted"
    assert payload["id"] == payload["agentic"]["task_id"]


@pytest.mark.asyncio
async def test_formatter_converts_to_a2a_with_context_and_task_ids():
    state = WorkflowExecutionState.new("exec-a2a", "flow-a2a")
    formatter = WorkflowChunkFormatter()
    chunk = formatter.format_chunk(state=state, content="A2A hello", status="submitted")

    async def _gen():
        yield chunk
        yield "data: [DONE]\n\n"

    async with chunk_reader(_gen()) as reader:
        converted = [
            payload
            async for payload in reader.as_json(ChunkFormat.A2A, request_id="req-123")
        ]

    assert converted
    payload = json.loads(_strip_prefix(converted[0]))
    assert payload["id"] == "req-123"
    result = payload["result"]
    assert result["contextId"] == "exec-a2a"
    assert result["id"] == state.context["task_id"]
    assert result["history"][0]["parts"][0]["text"] == "A2A hello"


def test_error_metadata_preserved():
    state = WorkflowExecutionState.new("exec-err", "flow-err")
    formatter = WorkflowChunkFormatter()
    chunk = formatter.format_chunk(
        state=state,
        content="Something failed",
        status="error",
        error={"code": "agent_failed", "message": "boom"},
    )

    payload = json.loads(_strip_prefix(chunk))
    assert payload["agentic"]["status"] == "error"
    assert payload["agentic"]["error"]["code"] == "agent_failed"
