"""Test runner service: execution, streaming, fix loops, and failure analysis.

The ``TestRunnerService`` encapsulates all methods for managing test
execution, streaming test events, executing fix strategies, and
explaining test failures.
It receives a reference to the parent ``GeneratedUIService`` for access
to shared helpers (session loading, storage, LLM client, etc.).
"""

import asyncio
import contextlib
import copy
import json
import logging
import re
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

from fastapi import HTTPException

from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.app_facade.generated_schemas import generation_response_format
from app.app_facade.generated_types import (
    Actor,
    Scope,
)
from app.app_facade.test_fix_tools import _parse_tap_output, run_tool_driven_test_fix
from app.app_facade.prompt_helpers import (
    parse_json,
    trim_runtime_text,
    to_chat_history_messages,
)

logger = logging.getLogger("uvicorn.error")

UI_MODEL_HEADERS = {"x-inxm-model-capability": "code-generation"}

MAX_TEST_EVENTS = 200
TEST_ACTIONS = ("run", "fix_code", "adjust_test", "delete_test", "add_test")
TestAction = Literal["run", "fix_code", "adjust_test", "delete_test", "add_test"]
TEST_FAILURE_ACTION_CHOICES = ["run", "fix_code", "adjust_test"]

_TEST_FAILURE_EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "what_failed": {"type": "string"},
        "why_it_ended": {"type": "string"},
        "recommended_action": {"type": "string", "enum": TEST_FAILURE_ACTION_CHOICES},
        "alternative_action": {"type": "string", "enum": TEST_FAILURE_ACTION_CHOICES},
        "next_step_text": {"type": "string"},
    },
    "required": [
        "what_failed",
        "why_it_ended",
        "recommended_action",
        "alternative_action",
        "next_step_text",
    ],
    "additionalProperties": False,
}


def _generation_response_format(schema=None, name: str = "generated_ui"):
    return generation_response_format(schema=schema, name=name)


def _sse_event(event: str, payload: Dict[str, Any]) -> bytes:
    return (
        f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode(
            "utf-8"
        )
    )


def _assistant_status_event(status: str) -> bytes:
    return _sse_event(
        "assistant",
        {
            "delta": status,
            "is_status": True,
        },
    )


class TestRunnerService:
    """Manages test execution, streaming, fix loops, and failure analysis."""

    def __init__(self, *, service):
        self.service = service
        self.storage = service.storage
        self.tgi_service = service.tgi_service
        self._test_event_subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._test_run_tasks: Dict[str, asyncio.Task] = {}
        self._test_run_locks: Dict[str, asyncio.Lock] = {}
        self._test_run_seq = 0

    def _test_stream_key(
        self, *, scope: Scope, ui_id: str, name: str, session_id: str
    ) -> str:
        return f"{scope.kind}:{scope.identifier}:{ui_id}:{name}:{session_id}"

    def _test_lock(self, stream_key: str) -> asyncio.Lock:
        lock = self._test_run_locks.get(stream_key)
        if lock is None:
            lock = asyncio.Lock()
            self._test_run_locks[stream_key] = lock
        return lock

    def _next_test_run_id(self) -> str:
        self._test_run_seq += 1
        return f"run-{self._test_run_seq:08d}"

    def _trim_output(self, output: str, limit: int = 4000) -> str:
        text = output or ""
        if len(text) <= limit:
            return text
        failure_lines: List[str] = []
        seen_failures = set()
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith("not ok ") and line not in seen_failures:
                seen_failures.add(line)
                failure_lines.append(line)

        summary_lines: List[str] = []
        for pattern in (
            r"^# tests\s+\d+",
            r"^# suites\s+\d+",
            r"^# pass\s+\d+",
            r"^# fail\s+\d+",
        ):
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                summary_lines.append(match.group(0))

        prefix_parts = [f"...(trimmed {len(text) - limit} chars)"]
        if failure_lines:
            preview = failure_lines[:8]
            if len(failure_lines) > 8:
                preview.append(f"... ({len(failure_lines) - 8} more failing entries)")
            prefix_parts.append(
                "Failed tests:\n" + "\n".join(f"- {line}" for line in preview)
            )
        if summary_lines:
            prefix_parts.append("Summary:\n" + "\n".join(summary_lines))

        prefix = "\n".join(prefix_parts).strip() + "\n\n"
        remaining = max(800, limit - len(prefix))
        head_budget = max(200, remaining // 3)
        tail_budget = max(400, remaining - head_budget)
        head = text[:head_budget]
        tail = text[-tail_budget:]
        return (
            f"{prefix}" f"--- OUTPUT HEAD ---\n{head}\n" f"--- OUTPUT TAIL ---\n{tail}"
        )

    async def _broadcast_test_event(
        self, stream_key: str, event: str, payload: Dict[str, Any]
    ) -> None:
        queues = list(self._test_event_subscribers.get(stream_key) or [])
        for queue in queues:
            try:
                queue.put_nowait((event, payload))
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    _ = queue.get_nowait()
                with contextlib.suppress(Exception):
                    queue.put_nowait((event, payload))

    def _append_test_event_to_session(
        self,
        session_payload: Dict[str, Any],
        *,
        event: str,
        payload: Dict[str, Any],
    ) -> None:
        events = session_payload.setdefault("test_events", [])
        events.append({"event": event, "payload": payload})
        session_payload["test_events"] = events[-MAX_TEST_EVENTS:]

    async def _update_test_state(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        session_id: str,
        stream_key: str,
        run_id: Optional[str],
        state: str,
        trigger: str,
        message: str,
        draft_version: Optional[int] = None,
        completed_at: Optional[str] = None,
        summary_updates: Optional[Dict[str, Any]] = None,
        skip_if_run_mismatch: bool = True,
    ) -> Optional[Dict[str, Any]]:
        payload: Optional[Dict[str, Any]] = None
        lock = self._test_lock(stream_key)
        async with lock:
            try:
                session_payload = self.service._load_session(
                    scope=scope, ui_id=ui_id, name=name, session_id=session_id
                )
            except HTTPException:
                return None

            current_run_id = str(session_payload.get("test_run_id") or "")
            run_mismatch = bool(run_id and current_run_id and current_run_id != run_id)
            if skip_if_run_mismatch and run_mismatch:
                return None

            now = self.service._now()
            if draft_version is None:
                draft_version = int(session_payload.get("draft_version") or 1)

            payload = {
                "run_id": run_id or current_run_id,
                "state": state,
                "trigger": trigger,
                "draft_version": draft_version,
                "message": message,
                "started_at": (session_payload.get("test_summary") or {}).get(
                    "started_at"
                ),
                "completed_at": completed_at,
            }

            if not run_mismatch:
                if run_id:
                    session_payload["test_run_id"] = run_id
                session_payload["test_state"] = state
                session_payload["updated_at"] = now
                summary = session_payload.setdefault("test_summary", {})
                summary.update(
                    {
                        "message": message,
                        "trigger": trigger,
                        "completed_at": completed_at,
                    }
                )
                if summary_updates:
                    summary.update(summary_updates)
                payload["started_at"] = summary.get("started_at")
                session_payload["test_summary"] = summary

            self._append_test_event_to_session(
                session_payload,
                event="test_status",
                payload=payload,
            )
            self.storage.write_session(scope, ui_id, name, session_id, session_payload)

        if payload:
            await self._broadcast_test_event(stream_key, "test_status", payload)
        return payload

    async def _append_test_event(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        session_id: str,
        stream_key: str,
        event: str,
        payload: Dict[str, Any],
        run_id: Optional[str] = None,
        skip_if_run_mismatch: bool = True,
    ) -> bool:
        lock = self._test_lock(stream_key)
        async with lock:
            try:
                session_payload = self.service._load_session(
                    scope=scope, ui_id=ui_id, name=name, session_id=session_id
                )
            except HTTPException:
                return False

            current_run_id = str(session_payload.get("test_run_id") or "")
            if (
                skip_if_run_mismatch
                and run_id
                and current_run_id
                and current_run_id != run_id
            ):
                return False

            if event == "test_output":
                session_payload["last_test_output_tail"] = str(
                    payload.get("output_tail") or ""
                )
            self._append_test_event_to_session(
                session_payload,
                event=event,
                payload=payload,
            )
            self.storage.write_session(scope, ui_id, name, session_id, session_payload)

        await self._broadcast_test_event(stream_key, event, payload)
        return True

    async def queue_test_action(
        self,
        *,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        session_id: str,
        action: TestAction,
        test_name: Optional[str],
        access_token: Optional[str],
    ) -> Dict[str, Any]:
        if action not in TEST_ACTIONS:
            raise HTTPException(status_code=400, detail="Invalid test action")
        session_payload = self.service._load_session(
            scope=scope, ui_id=ui_id, name=name, session_id=session_id
        )
        self.service._assert_session_owner(session_payload, actor)
        return await self._queue_test_run(
            scope=scope,
            ui_id=ui_id,
            name=name,
            session_id=session_id,
            action=action,
            trigger=f"manual_{action}",
            test_name=test_name,
            access_token=access_token,
        )

    async def stream_test_events(
        self,
        *,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        session_id: str,
    ) -> AsyncIterator[bytes]:
        session_payload = self.service._load_session(
            scope=scope, ui_id=ui_id, name=name, session_id=session_id
        )
        self.service._assert_session_owner(session_payload, actor)
        stream_key = self._test_stream_key(
            scope=scope, ui_id=ui_id, name=name, session_id=session_id
        )
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        subscribers = self._test_event_subscribers.setdefault(stream_key, [])
        subscribers.append(queue)
        try:
            for item in list(session_payload.get("test_events") or []):
                if not isinstance(item, dict):
                    continue
                event = str(item.get("event") or "").strip()
                payload = item.get("payload")
                if not event or not isinstance(payload, dict):
                    continue
                yield _sse_event(event, payload)
            while True:
                try:
                    event, payload = await asyncio.wait_for(queue.get(), timeout=10.0)
                except asyncio.TimeoutError:
                    yield b": keepalive\n\n"
                    continue
                if not isinstance(event, str) or not isinstance(payload, dict):
                    continue
                yield _sse_event(event, payload)
        finally:
            current = self._test_event_subscribers.get(stream_key) or []
            self._test_event_subscribers[stream_key] = [
                q for q in current if q != queue
            ]
            if not self._test_event_subscribers[stream_key]:
                self._test_event_subscribers.pop(stream_key, None)

    async def _queue_test_run(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        session_id: str,
        action: TestAction,
        trigger: str,
        test_name: Optional[str],
        access_token: Optional[str],
    ) -> Dict[str, Any]:
        stream_key = self._test_stream_key(
            scope=scope, ui_id=ui_id, name=name, session_id=session_id
        )
        lock = self._test_lock(stream_key)
        events_to_emit: List[Tuple[str, Dict[str, Any]]] = []
        previous_task: Optional[asyncio.Task] = None
        run_id = self._next_test_run_id()

        async with lock:
            session_payload = self.service._load_session(
                scope=scope, ui_id=ui_id, name=name, session_id=session_id
            )
            previous_task = self._test_run_tasks.get(stream_key)
            previous_run_id = str(session_payload.get("test_run_id") or "")
            draft_version = int(session_payload.get("draft_version") or 1)

            if previous_task and not previous_task.done() and previous_run_id:
                previous_task.cancel()
                cancelled_payload = {
                    "run_id": previous_run_id,
                    "state": "cancelled",
                    "trigger": trigger,
                    "draft_version": draft_version,
                    "message": "Superseded by a newer test run",
                    "started_at": None,
                    "completed_at": self.service._now(),
                }
                self._append_test_event_to_session(
                    session_payload,
                    event="test_status",
                    payload=cancelled_payload,
                )
                events_to_emit.append(("test_status", cancelled_payload))

            session_payload["test_run_id"] = run_id
            session_payload["test_state"] = "queued"
            summary = session_payload.setdefault("test_summary", {})
            summary.update(
                {
                    "message": "Test run queued",
                    "trigger": trigger,
                    "passed": int(summary.get("passed") or 0),
                    "failed": int(summary.get("failed") or 0),
                    "failed_tests": list(summary.get("failed_tests") or []),
                    "started_at": None,
                    "completed_at": None,
                }
            )
            session_payload["test_summary"] = summary
            session_payload["updated_at"] = self.service._now()
            queued_payload = {
                "run_id": run_id,
                "state": "queued",
                "trigger": trigger,
                "draft_version": draft_version,
                "message": "Test run queued",
                "started_at": None,
                "completed_at": None,
            }
            self._append_test_event_to_session(
                session_payload,
                event="test_status",
                payload=queued_payload,
            )
            events_to_emit.append(("test_status", queued_payload))
            self.storage.write_session(scope, ui_id, name, session_id, session_payload)

        for event, payload in events_to_emit:
            await self._broadcast_test_event(stream_key, event, payload)

        task = asyncio.create_task(
            self._execute_test_run(
                scope=scope,
                ui_id=ui_id,
                name=name,
                session_id=session_id,
                stream_key=stream_key,
                run_id=run_id,
                action=action,
                trigger=trigger,
                test_name=test_name,
                access_token=access_token,
            )
        )
        self._test_run_tasks[stream_key] = task

        def _cleanup(done_task: asyncio.Task) -> None:
            current = self._test_run_tasks.get(stream_key)
            if current is done_task:
                self._test_run_tasks.pop(stream_key, None)

        task.add_done_callback(_cleanup)
        return {"status": "queued", "run_id": run_id, "trigger": trigger}

    async def _commit_test_action_payload(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        session_id: str,
        stream_key: str,
        run_id: str,
        updated_payload: Dict[str, Any],
        update_mode: str,
    ) -> Optional[int]:
        lock = self._test_lock(stream_key)
        async with lock:
            session_payload = self.service._load_session(
                scope=scope, ui_id=ui_id, name=name, session_id=session_id
            )
            if str(session_payload.get("test_run_id") or "") != run_id:
                return None
            session_payload["draft_payload"] = updated_payload
            session_payload["updated_at"] = self.service._now()
            session_payload["draft_version"] = (
                int(session_payload.get("draft_version") or 1) + 1
            )
            draft_version = int(session_payload["draft_version"])
            ui_payload = {
                "session_id": session_id,
                "draft_version": draft_version,
                "update_mode": update_mode,
            }
            self._append_test_event_to_session(
                session_payload,
                event="ui_updated",
                payload=ui_payload,
            )
            self.storage.write_session(scope, ui_id, name, session_id, session_payload)
        await self._broadcast_test_event(stream_key, "ui_updated", ui_payload)
        return draft_version

    async def _emit_tool_or_test_event(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        session_id: str,
        stream_key: str,
        run_id: str,
        event_data: Dict[str, Any],
    ) -> None:
        event = str(event_data.get("event") or "").strip()
        if event not in {"tool_start", "test_result"}:
            return
        payload = dict(event_data)
        payload.pop("event", None)
        payload["run_id"] = run_id
        if event == "tool_start":
            why = payload.get("why") or payload.get("fix_explanation")
            if why:
                payload["fix_explanation"] = str(why)
                payload["why"] = str(why)
                logger.info(
                    "[GeneratedUI] Test run %s why for tool %s: %s",
                    run_id,
                    payload.get("tool") or "unknown",
                    self._trim_output(str(why), limit=600),
                )
        await self._append_test_event(
            scope=scope,
            ui_id=ui_id,
            name=name,
            session_id=session_id,
            stream_key=stream_key,
            event=event,
            payload=payload,
            run_id=run_id,
        )

    def _delete_tests_by_name(self, test_script: str, test_names: Sequence[str]) -> str:
        updated = test_script or ""
        for test_name in test_names:
            clean_name = (test_name or "").strip()
            if not clean_name:
                continue
            escaped = re.escape(clean_name)
            block_pattern = re.compile(
                rf"(?ms)^\s*(?:test|it)\(\s*(['\"])({escaped})\1\s*,\s*(?:async\s*)?\([^)]*\)\s*=>\s*\{{.*?^\s*\}}\s*\);\s*"
            )
            single_line_pattern = re.compile(
                rf"(?m)^\s*(?:test|it)\(\s*(['\"])({escaped})\1\s*,.*\);\s*$"
            )
            updated = block_pattern.sub("", updated)
            updated = single_line_pattern.sub("", updated)
        return updated

    def _append_named_test(self, test_script: str, test_name: str) -> str:
        clean_name = (test_name or "").strip()
        if not clean_name:
            return test_script or ""
        existing_pattern = re.compile(
            rf"(?:test|it)\(\s*(['\"]){re.escape(clean_name)}\1\s*,"
        )
        base = test_script or ""
        if existing_pattern.search(base):
            return base

        if "from 'node:test'" not in base and 'from "node:test"' not in base:
            base = "import { test } from 'node:test';\n" + base
        elif "from 'node:test'" in base or 'from "node:test"' in base:
            import_pattern = re.compile(
                r"import\s*\{([^}]*)\}\s*from\s*['\"]node:test['\"];?"
            )
            match = import_pattern.search(base)
            if match:
                symbols = [
                    part.strip() for part in match.group(1).split(",") if part.strip()
                ]
                if "test" not in symbols:
                    symbols.append("test")
                    replacement = (
                        "import { "
                        + ", ".join(sorted(set(symbols)))
                        + " } from 'node:test';"
                    )
                    base = base[: match.start()] + replacement + base[match.end() :]

        stub = (
            f"\n\ntest({json.dumps(clean_name)}, async () => {{\n"
            "  // User-requested test placeholder.\n"
            "});\n"
        )
        return base.rstrip() + stub

    async def _execute_strategy_fix_action(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        session_id: str,
        stream_key: str,
        run_id: str,
        session_payload: Dict[str, Any],
        action: TestAction,
        access_token: Optional[str],
    ) -> Tuple[
        bool,
        str,
        int,
        int,
        List[str],
        Optional[Dict[str, Any]],
        Dict[str, Any],
    ]:
        draft_payload = copy.deepcopy(session_payload.get("draft_payload", {}))
        service_script = str(draft_payload.get("service_script") or "")
        components_script = str(draft_payload.get("components_script") or "")
        test_script = str(draft_payload.get("test_script") or "")
        dummy_data = draft_payload.get("dummy_data")
        history_messages = to_chat_history_messages(session_payload.get("messages", []))

        event_queue: asyncio.Queue = asyncio.Queue()
        strategy_context: Dict[str, Any] = {
            "reason_code": "",
            "reason": "",
            "suggested_action": "",
            "tool_why": [],
        }

        def _capture_strategy_context(event_data: Dict[str, Any]) -> None:
            event_name = str(event_data.get("event") or "").strip()
            if event_name == "tool_start":
                why = str(
                    event_data.get("why") or event_data.get("fix_explanation") or ""
                ).strip()
                if why:
                    tool_why = strategy_context.setdefault("tool_why", [])
                    if why not in tool_why:
                        tool_why.append(trim_runtime_text(why, limit=240))
                    strategy_context["tool_why"] = tool_why[-6:]
                return
            if event_name != "test_result":
                return
            status = str(event_data.get("status") or "").strip().lower()
            if status not in {"handoff", "contract_failed", "failed", "error"}:
                return
            reason_code = str(event_data.get("reason_code") or "").strip()
            reason = str(event_data.get("reason") or "").strip()
            suggested_action = str(event_data.get("suggested_action") or "").strip()
            if reason_code:
                strategy_context["reason_code"] = reason_code
            if reason:
                strategy_context["reason"] = trim_runtime_text(reason, limit=300)
            if suggested_action:
                strategy_context["suggested_action"] = suggested_action

        strategy_mode: Literal["fix_code", "adjust_test"]
        strategy_mode = "fix_code" if action == "fix_code" else "adjust_test"
        fix_task = asyncio.create_task(
            run_tool_driven_test_fix(
                tgi_service=self.tgi_service,
                service_script=service_script,
                components_script=components_script,
                test_script=test_script,
                dummy_data=dummy_data,
                messages=history_messages,
                allowed_tools=list(session_payload.get("last_tools") or []),
                access_token=access_token,
                max_attempts=15,
                event_queue=event_queue,
                extra_headers=UI_MODEL_HEADERS,
                strategy_mode=strategy_mode,
            )
        )
        try:
            while not fix_task.done():
                try:
                    event_data = event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.05)
                    continue
                _capture_strategy_context(event_data)
                await self._emit_tool_or_test_event(
                    scope=scope,
                    ui_id=ui_id,
                    name=name,
                    session_id=session_id,
                    stream_key=stream_key,
                    run_id=run_id,
                    event_data=event_data,
                )
            (
                fix_success,
                fixed_service,
                fixed_components,
                fixed_test,
                fixed_dummy_data,
                _updated_messages,
            ) = await fix_task
        except asyncio.CancelledError:
            fix_task.cancel()
            raise

        while not event_queue.empty():
            event_data = event_queue.get_nowait()
            _capture_strategy_context(event_data)
            await self._emit_tool_or_test_event(
                scope=scope,
                ui_id=ui_id,
                name=name,
                session_id=session_id,
                stream_key=stream_key,
                run_id=run_id,
                event_data=event_data,
            )

        final_success, final_output = await asyncio.to_thread(
            self.service._run_tests,
            fixed_service,
            fixed_components,
            fixed_test,
            fixed_dummy_data,
            None,
        )
        passed, failed, failed_tests = _parse_tap_output(final_output or "")
        updated_payload = None
        if fix_success and action in {"fix_code", "adjust_test"}:
            updated_payload = copy.deepcopy(draft_payload)
            updated_payload["service_script"] = fixed_service
            updated_payload["components_script"] = fixed_components
            updated_payload["test_script"] = fixed_test
            updated_payload["dummy_data"] = fixed_dummy_data
        return (
            final_success,
            final_output,
            int(passed),
            int(failed),
            list(failed_tests),
            updated_payload,
            strategy_context,
        )

    async def _execute_test_run(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        session_id: str,
        stream_key: str,
        run_id: str,
        action: TestAction,
        trigger: str,
        test_name: Optional[str],
        access_token: Optional[str],
    ) -> None:
        started_at = self.service._now()
        await self._update_test_state(
            scope=scope,
            ui_id=ui_id,
            name=name,
            session_id=session_id,
            stream_key=stream_key,
            run_id=run_id,
            state="running",
            trigger=trigger,
            message="Running tests...",
            summary_updates={"started_at": started_at, "completed_at": None},
        )

        try:
            session_payload = self.service._load_session(
                scope=scope, ui_id=ui_id, name=name, session_id=session_id
            )
            if str(session_payload.get("test_run_id") or "") != run_id:
                return

            draft_payload = copy.deepcopy(session_payload.get("draft_payload", {}))
            service_script = str(draft_payload.get("service_script") or "")
            components_script = str(draft_payload.get("components_script") or "")
            test_script = str(draft_payload.get("test_script") or "")
            dummy_data = draft_payload.get("dummy_data")

            if not components_script or not test_script:
                completed_at = self.service._now()
                await self._update_test_state(
                    scope=scope,
                    ui_id=ui_id,
                    name=name,
                    session_id=session_id,
                    stream_key=stream_key,
                    run_id=run_id,
                    state="error",
                    trigger=trigger,
                    message="Draft is missing components_script or test_script",
                    completed_at=completed_at,
                    summary_updates={
                        "started_at": started_at,
                        "completed_at": completed_at,
                    },
                )
                return

            updated_payload: Optional[Dict[str, Any]] = None
            success = False
            output = ""
            passed = 0
            failed = 0
            failed_tests: List[str] = []
            strategy_context: Optional[Dict[str, Any]] = None

            if action == "run":
                success, output = await asyncio.to_thread(
                    self.service._run_tests,
                    service_script,
                    components_script,
                    test_script,
                    dummy_data,
                    test_name,
                )
                passed, failed, failed_tests = _parse_tap_output(output or "")
            elif action in {"fix_code", "adjust_test"}:
                (
                    success,
                    output,
                    passed,
                    failed,
                    failed_tests,
                    updated_payload,
                    strategy_context,
                ) = await self._execute_strategy_fix_action(
                    scope=scope,
                    ui_id=ui_id,
                    name=name,
                    session_id=session_id,
                    stream_key=stream_key,
                    run_id=run_id,
                    session_payload=session_payload,
                    action=action,
                    access_token=access_token,
                )
            elif action == "delete_test":
                target_tests = list(
                    (session_payload.get("test_summary") or {}).get("failed_tests")
                    or []
                )
                if test_name and test_name.strip():
                    target_tests = [test_name.strip()]
                updated_test_script = self._delete_tests_by_name(
                    test_script, target_tests
                )
                if updated_test_script == test_script:
                    output = "No matching tests found to delete."
                    success = False
                else:
                    updated_payload = copy.deepcopy(draft_payload)
                    updated_payload["test_script"] = updated_test_script
                    success, output = await asyncio.to_thread(
                        self.service._run_tests,
                        service_script,
                        components_script,
                        updated_test_script,
                        dummy_data,
                        None,
                    )
                    passed, failed, failed_tests = _parse_tap_output(output or "")
            elif action == "add_test":
                if not (test_name or "").strip():
                    output = "test_name is required for add_test action."
                    success = False
                else:
                    updated_test_script = self._append_named_test(
                        test_script, test_name or ""
                    )
                    updated_payload = copy.deepcopy(draft_payload)
                    updated_payload["test_script"] = updated_test_script
                    success, output = await asyncio.to_thread(
                        self.service._run_tests,
                        service_script,
                        components_script,
                        updated_test_script,
                        dummy_data,
                        None,
                    )
                    passed, failed, failed_tests = _parse_tap_output(output or "")

            if updated_payload is not None:
                await self._commit_test_action_payload(
                    scope=scope,
                    ui_id=ui_id,
                    name=name,
                    session_id=session_id,
                    stream_key=stream_key,
                    run_id=run_id,
                    updated_payload=updated_payload,
                    update_mode=f"tests_{action}",
                )

            output_tail = self._trim_output(output)
            failure_analysis: Optional[Dict[str, str]] = None
            if not success:
                failure_analysis = await self._explain_test_failure_for_user(
                    action=action,
                    passed=int(passed),
                    failed=int(failed),
                    failed_tests=list(failed_tests),
                    output_tail=output_tail,
                    strategy_context=strategy_context,
                    access_token=access_token,
                )

            result_payload = {
                "run_id": run_id,
                "status": "passed" if success else "failed",
                "passed": int(passed),
                "failed": int(failed),
                "failed_tests": list(failed_tests),
                "message": (
                    "All tests passed"
                    if success
                    else str(
                        (failure_analysis or {}).get("next_step_text") or "Tests failed"
                    )
                ),
                "is_final": True,
            }
            if failure_analysis:
                result_payload["analysis"] = {
                    "what_failed": str(failure_analysis.get("what_failed") or ""),
                    "why_it_ended": str(failure_analysis.get("why_it_ended") or ""),
                    "next_step_text": str(failure_analysis.get("next_step_text") or ""),
                }
                result_payload["recommended_action"] = str(
                    failure_analysis.get("recommended_action") or ""
                )
                result_payload["alternative_action"] = str(
                    failure_analysis.get("alternative_action") or ""
                )
            await self._append_test_event(
                scope=scope,
                ui_id=ui_id,
                name=name,
                session_id=session_id,
                stream_key=stream_key,
                event="test_result",
                payload=result_payload,
                run_id=run_id,
            )

            await self._append_test_event(
                scope=scope,
                ui_id=ui_id,
                name=name,
                session_id=session_id,
                stream_key=stream_key,
                event="test_output",
                payload={"run_id": run_id, "output_tail": output_tail},
                run_id=run_id,
            )

            completed_at = self.service._now()
            status_message = "Tests passing" if success else "Tests failing"
            if failure_analysis:
                status_message = trim_runtime_text(
                    failure_analysis.get("next_step_text") or status_message,
                    limit=240,
                )
            await self._update_test_state(
                scope=scope,
                ui_id=ui_id,
                name=name,
                session_id=session_id,
                stream_key=stream_key,
                run_id=run_id,
                state="passed" if success else "failed",
                trigger=trigger,
                message=status_message,
                completed_at=completed_at,
                summary_updates={
                    "passed": int(passed),
                    "failed": int(failed),
                    "failed_tests": list(failed_tests),
                    "started_at": started_at,
                    "completed_at": completed_at,
                },
            )
        except asyncio.CancelledError:
            completed_at = self.service._now()
            await self._update_test_state(
                scope=scope,
                ui_id=ui_id,
                name=name,
                session_id=session_id,
                stream_key=stream_key,
                run_id=run_id,
                state="cancelled",
                trigger=trigger,
                message="Test run cancelled",
                completed_at=completed_at,
                summary_updates={"completed_at": completed_at},
                skip_if_run_mismatch=False,
            )
            raise
        except Exception as exc:
            logger.error("[GeneratedUI] Test run failed: %s", exc, exc_info=exc)
            completed_at = self.service._now()
            await self._update_test_state(
                scope=scope,
                ui_id=ui_id,
                name=name,
                session_id=session_id,
                stream_key=stream_key,
                run_id=run_id,
                state="error",
                trigger=trigger,
                message="Test run errored",
                completed_at=completed_at,
                summary_updates={
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "error": str(exc),
                },
            )

    def _assistant_text_from_response(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if choices:
            first = choices[0]
            message = getattr(first, "message", None)
            if message and getattr(message, "content", None):
                return str(message.content)

        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                first = choices[0] or {}
                message = first.get("message") or {}
                content = message.get("content")
                if content is not None:
                    return str(content)
        return ""

    def _normalize_failure_action_choice(self, value: Any, default: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in TEST_FAILURE_ACTION_CHOICES:
            return normalized
        return default

    def _fallback_test_failure_analysis(
        self,
        *,
        action: TestAction,
        passed: int,
        failed: int,
        failed_tests: Sequence[str],
        strategy_context: Optional[Dict[str, Any]],
    ) -> Dict[str, str]:
        failed_preview = ", ".join(list(failed_tests)[:3]).strip()
        if failed_preview:
            if len(list(failed_tests)) > 3:
                failed_preview += f", ... ({len(list(failed_tests)) - 3} more)"
            what_failed = f"{failed} tests failed after {passed} passed. Failing tests: {failed_preview}."
        elif failed > 0:
            what_failed = f"{failed} tests failed after {passed} passed."
        else:
            what_failed = "Tests did not complete successfully."

        strategy_reason = str((strategy_context or {}).get("reason") or "").strip()
        if strategy_reason:
            why_it_ended = f"The run ended with a strategy handoff/stop condition: {strategy_reason}"
        elif action == "fix_code":
            why_it_ended = (
                "fix_code mode only applies runtime edits; the remaining failures likely need "
                "test or fixture adjustments."
            )
        elif action == "adjust_test":
            why_it_ended = "adjust_test mode could not fully resolve the failing assertions with test/fixture edits."
        else:
            why_it_ended = "The latest test run still contains failures."

        default_recommended = {
            "run": "fix_code",
            "fix_code": "adjust_test",
            "adjust_test": "fix_code",
        }.get(action, "fix_code")
        suggested = self._normalize_failure_action_choice(
            (strategy_context or {}).get("suggested_action"),
            default_recommended,
        )
        alternative = "adjust_test" if suggested == "fix_code" else "fix_code"
        if suggested == "run":
            alternative = "fix_code"

        next_step_text = (
            f"Recommended next action: {suggested}. Alternative: {alternative}."
        )
        return {
            "what_failed": trim_runtime_text(what_failed, limit=600),
            "why_it_ended": trim_runtime_text(why_it_ended, limit=600),
            "recommended_action": suggested,
            "alternative_action": alternative,
            "next_step_text": trim_runtime_text(next_step_text, limit=240),
        }

    async def _explain_test_failure_for_user(
        self,
        *,
        action: TestAction,
        passed: int,
        failed: int,
        failed_tests: Sequence[str],
        output_tail: str,
        strategy_context: Optional[Dict[str, Any]],
        access_token: Optional[str],
    ) -> Dict[str, str]:
        fallback = self._fallback_test_failure_analysis(
            action=action,
            passed=passed,
            failed=failed,
            failed_tests=failed_tests,
            strategy_context=strategy_context,
        )
        llm_client = getattr(self.tgi_service, "llm_client", None)
        non_stream = getattr(llm_client, "non_stream_completion", None)
        if not callable(non_stream):
            return fallback

        payload = {
            "action": action,
            "passed": int(passed),
            "failed": int(failed),
            "failed_tests": list(failed_tests)[:8],
            "output_tail": output_tail,
            "strategy_context": strategy_context or {},
            "requirements": {
                "explain_what_failed": True,
                "explain_why_it_ended": True,
                "recommend_one_next_action": TEST_FAILURE_ACTION_CHOICES,
            },
        }
        request = ChatCompletionRequest(
            messages=[
                Message(
                    role=MessageRole.SYSTEM,
                    content=(
                        "You explain generated-ui test failures to end users. "
                        "Be concise, practical, and action-oriented. "
                        "Always explain why the run ended and what the user should do next."
                    ),
                ),
                Message(
                    role=MessageRole.USER,
                    content=json.dumps(payload, ensure_ascii=False),
                ),
            ],
            stream=False,
            response_format=_generation_response_format(
                schema=_TEST_FAILURE_EXPLANATION_SCHEMA,
                name="test_failure_explanation",
            ),
            extra_headers=UI_MODEL_HEADERS,
        )
        try:
            response = await non_stream(request, access_token or "", None)
            content = self._assistant_text_from_response(response)
            if not content.strip():
                return fallback
            parsed = parse_json(content)
        except Exception as exc:
            logger.warning(
                "[GeneratedUI] Failed to generate test failure explanation via LLM: %s",
                exc,
            )
            return fallback

        what_failed = str(parsed.get("what_failed") or "").strip()
        why_it_ended = str(parsed.get("why_it_ended") or "").strip()
        next_step_text = str(parsed.get("next_step_text") or "").strip()
        recommended_action = self._normalize_failure_action_choice(
            parsed.get("recommended_action"),
            fallback["recommended_action"],
        )
        alternative_action = self._normalize_failure_action_choice(
            parsed.get("alternative_action"),
            fallback["alternative_action"],
        )
        if not what_failed:
            what_failed = fallback["what_failed"]
        if not why_it_ended:
            why_it_ended = fallback["why_it_ended"]
        if not next_step_text:
            next_step_text = fallback["next_step_text"]
        if alternative_action == recommended_action:
            alternative_action = (
                "adjust_test" if recommended_action == "fix_code" else "fix_code"
            )

        return {
            "what_failed": trim_runtime_text(what_failed, limit=600),
            "why_it_ended": trim_runtime_text(why_it_ended, limit=600),
            "recommended_action": recommended_action,
            "alternative_action": alternative_action,
            "next_step_text": trim_runtime_text(next_step_text, limit=240),
        }
