import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from typing import AsyncGenerator, Awaitable, Callable, Optional


StreamFactory = Callable[[], Awaitable[AsyncGenerator[str, None]]]


@dataclass
class BackgroundWorkflowRun:
    execution_id: str
    counter: int
    task: Optional[asyncio.Task] = None
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    error: Optional[BaseException] = None
    cancelled: bool = False


class WorkflowBackgroundManager:
    """
    Run workflow streams in a background task and publish chunks to subscribers.
    """

    def __init__(self, logger: logging.Logger, max_queue_size: int = 1000):
        self._logger = logger
        self._max_queue_size = max_queue_size
        self._runs: dict[str, BackgroundWorkflowRun] = {}
        self._lock = asyncio.Lock()

    def is_running(self, execution_id: str) -> bool:
        run = self._runs.get(execution_id)
        return bool(run and run.task and not run.task.done())

    async def get_or_start(
        self,
        execution_id: str,
        stream_factory: StreamFactory,
        initial_event_count: int = 0,
    ) -> BackgroundWorkflowRun:
        async with self._lock:
            run = self._runs.get(execution_id)
            if run and run.task and not run.task.done():
                return run

            run = BackgroundWorkflowRun(
                execution_id=execution_id, counter=max(0, initial_event_count)
            )
            self._runs[execution_id] = run
            run.task = asyncio.create_task(
                self._run_stream(run, stream_factory),
                name=f"workflow-bg-{execution_id}",
            )
            return run

    @contextlib.asynccontextmanager
    async def subscribe(self, execution_id: str):
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue_size)
        run = self._runs.get(execution_id)
        if run:
            run.subscribers.add(queue)
        try:
            yield queue if run else None
        finally:
            if run:
                run.subscribers.discard(queue)

    async def cancel(self, execution_id: str) -> bool:
        async with self._lock:
            run = self._runs.get(execution_id)

        if not run or not run.task or run.task.done():
            return False

        run.cancelled = True
        run.task.cancel()
        try:
            await run.task
        except asyncio.CancelledError:
            pass
        return True

    async def _run_stream(
        self, run: BackgroundWorkflowRun, stream_factory: StreamFactory
    ) -> None:
        stream = None
        try:
            stream = await stream_factory()
            if not stream or not hasattr(stream, "__aiter__"):
                self._logger.warning(
                    "[WorkflowBackground] Stream factory did not return a stream for %s",
                    run.execution_id,
                )
                return

            async for chunk in stream:
                chunk_text = self._coerce_chunk(chunk)
                is_unrecorded = self._is_unrecorded_chunk(chunk_text)
                if not is_unrecorded:
                    run.counter += 1
                self._publish(
                    run,
                    run.counter,
                    chunk_text,
                    recorded=not is_unrecorded,
                )
        except asyncio.CancelledError:
            run.cancelled = True
            if stream and hasattr(stream, "aclose"):
                with contextlib.suppress(Exception):
                    await stream.aclose()
            raise
        except Exception as exc:  # pragma: no cover - defensive
            run.error = exc
            self._logger.warning(
                "[WorkflowBackground] Background workflow failed for %s: %s",
                run.execution_id,
                exc,
            )
        finally:
            run.done.set()
            self._publish(run, run.counter, None, recorded=False)
            await self._remove_run(run.execution_id)

    def _publish(
        self,
        run: BackgroundWorkflowRun,
        idx: int,
        chunk: Optional[str],
        *,
        recorded: bool,
    ) -> None:
        for queue in list(run.subscribers):
            try:
                queue.put_nowait((idx, chunk, recorded))
            except asyncio.QueueFull:
                self._logger.debug(
                    "[WorkflowBackground] Dropping chunk for %s (queue full)",
                    run.execution_id,
                )

    async def _remove_run(self, execution_id: str) -> None:
        async with self._lock:
            run = self._runs.get(execution_id)
            if run and run.task and run.task.done():
                self._runs.pop(execution_id, None)

    def _coerce_chunk(self, chunk: object) -> str:
        if isinstance(chunk, str):
            return chunk
        try:
            return json.dumps(chunk, ensure_ascii=False)
        except Exception:
            return str(chunk)

    def _is_unrecorded_chunk(self, chunk: str) -> bool:
        if "[DONE]" in chunk:
            return True
        if not chunk.startswith("data: "):
            return False
        payload_text = chunk[len("data: ") :].strip()
        try:
            payload = json.loads(payload_text)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        agentic = payload.get("agentic")
        if not isinstance(agentic, dict):
            return False
        if agentic.get("status") != "submitted":
            return False
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str) and "<workflow_execution_id" in content:
                return True
        return False
