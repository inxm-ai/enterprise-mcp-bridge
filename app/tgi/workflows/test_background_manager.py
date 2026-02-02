import asyncio
import json
import logging

import pytest

from app.tgi.workflows.background_manager import WorkflowBackgroundManager


def _submitted_chunk(execution_id: str = "exec-1", flow_id: str = "flow") -> str:
    payload = {
        "choices": [
            {
                "delta": {
                    "content": f'<workflow_execution_id for="{flow_id}">{execution_id}</workflow_execution_id>'
                }
            }
        ],
        "agentic": {"status": "submitted"},
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _normal_chunk(content: str = "hello") -> str:
    payload = {
        "choices": [{"delta": {"content": content}}],
        "agentic": {"status": "in_progress"},
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@pytest.mark.asyncio
async def test_background_manager_publishes_and_counts():
    manager = WorkflowBackgroundManager(logger=logging.getLogger("test-bg"))
    chunks = [
        _submitted_chunk(),
        _normal_chunk("first"),
        "data: [DONE]\n\n",
    ]

    async def stream_factory():
        async def _stream():
            for chunk in chunks:
                yield chunk

        return _stream()

    run = await manager.get_or_start("exec-1", stream_factory)

    received = []
    async with manager.subscribe("exec-1") as queue:
        assert queue is not None
        while True:
            idx, chunk, recorded = await asyncio.wait_for(queue.get(), timeout=1)
            if chunk is None:
                break
            received.append((idx, chunk, recorded))

    await asyncio.wait_for(run.done.wait(), timeout=1)

    assert run.counter == 1
    assert received[0][2] is False  # submitted chunk is unrecorded
    assert received[1][2] is True
    assert "[DONE]" in received[2][1]


@pytest.mark.asyncio
async def test_background_manager_cancel():
    manager = WorkflowBackgroundManager(logger=logging.getLogger("test-bg"))

    async def stream_factory():
        async def _stream():
            while True:
                await asyncio.sleep(0.01)
                yield _normal_chunk("tick")

        return _stream()

    run = await manager.get_or_start("exec-cancel", stream_factory)

    async with manager.subscribe("exec-cancel") as queue:
        assert queue is not None
        idx, chunk, recorded = await asyncio.wait_for(queue.get(), timeout=1)
        assert chunk is not None
        assert recorded is True

        cancelled = await manager.cancel("exec-cancel")
        assert cancelled is True

        while True:
            idx, chunk, recorded = await asyncio.wait_for(queue.get(), timeout=1)
            if chunk is None:
                break

    await asyncio.wait_for(run.done.wait(), timeout=1)
    assert run.cancelled is True
    assert manager.is_running("exec-cancel") is False
