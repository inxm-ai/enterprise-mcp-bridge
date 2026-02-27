"""Conversational service: chat-driven updates, patch application, assistant messaging.

The ``ConversationalService`` class handles the conversational update flow,
including streaming chat updates, composing regeneration prompts,
running assistant messages with tool use, and applying patch updates.
"""

import asyncio
import copy
import json
import logging
import os
import re
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from fastapi import HTTPException

from app.session import MCPSessionBase
from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.protocols.chunk_reader import chunk_reader
from app.tgi.services.proxied_tgi_service import ProxiedTGIService
from app.app_facade.generated_schemas import generation_response_format
from app.app_facade.generated_output_factory import GeneratedUIOutputFactory
from app.app_facade.generated_types import (
    Actor,
    Scope,
    validate_identifier,
)
from app.app_facade.test_fix_tools import _parse_tap_output
from app.app_facade.prompt_helpers import (
    SCRIPT_KEYS,
    extract_content,
    extract_json_block,
    parse_json,
    to_json_value,
    trim_runtime_text,
    sanitize_runtime_value,
    sanitize_runtime_action,
    runtime_context_for_prompt,
    context_state_for_prompt,
    payload_bytes,
    prompt_with_runtime_context,
    to_chat_history_messages,
    history_entry,
    history_for_prompt,
    changed_scripts,
    scripts_from_history,
    cap_tools_for_prompt,
    cap_message_payload_for_prompt,
)
from app.vars import (
    MCP_BASE_PATH,
    GENERATED_UI_PROMPT_DUMP,
    GENERATED_UI_INCLUDE_OUTPUT_SCHEMA,
    GENERATED_UI_MAX_HISTORY_ENTRIES,
    GENERATED_UI_MAX_HISTORY_BYTES,
    GENERATED_UI_MAX_RUNTIME_EXCHANGES,
    GENERATED_UI_MAX_RUNTIME_CONSOLE_EVENTS,
    GENERATED_UI_MAX_RUNTIME_BYTES,
    GENERATED_UI_MAX_TOOLS,
    GENERATED_UI_MAX_TOOLS_BYTES,
    GENERATED_UI_MAX_MESSAGE_PAYLOAD_BYTES,
    APP_UI_PATCH_ONLY,
    APP_UI_PATCH_RETRIES,
)

logger = logging.getLogger("uvicorn.error")

_PATCH_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "patch": {
            "type": "object",
            "properties": {
                "html": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "string"},
                        "snippet": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                "service_script": {"type": "string"},
                "components_script": {"type": "string"},
                "test_script": {"type": "string"},
                "dummy_data": {"type": "string"},
                "metadata": {
                    "type": "object",
                    "additionalProperties": True,
                },
            },
            "additionalProperties": False,
        }
    },
    "required": ["patch"],
    "additionalProperties": False,
}

UI_MODEL_HEADERS = {"x-inxm-model-capability": "code-generation"}


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


class ConversationalService:
    """Handles conversational chat update flow, patch application, and assistant messaging."""

    def __init__(self, *, service):
        self.service = service
        self.storage = service.storage
        self.tgi_service = service.tgi_service

    async def stream_chat_update(
        self,
        *,
        session: MCPSessionBase,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        session_id: str,
        message: str,
        tools: Optional[Sequence[str]],
        tool_choice: Optional[Any],
        draft_action: Optional[Dict[str, Any]] = None,
        access_token: Optional[str] = None,
    ) -> AsyncIterator[bytes]:
        if not message.strip():
            yield _sse_event("error", {"error": "message must not be empty"})
            return

        try:
            yield _assistant_status_event("I will prepare the update request now.")
            session_payload = self.service._load_session(
                scope=scope, ui_id=ui_id, name=name, session_id=session_id
            )
            self.service._assert_session_owner(session_payload, actor)
            runtime_context = sanitize_runtime_action(draft_action)

            requested_tools = (
                list(tools)
                if tools is not None
                else list(session_payload.get("last_tools") or [])
            )
            yield _assistant_status_event(
                "I will select the best tools for this update."
            )
            selected_tools = await self.service._select_tools(
                session, requested_tools, message
            )
            selected_tool_names = [
                t.get("function", {}).get("name")
                for t in (selected_tools or [])
                if isinstance(t, dict)
            ]
            selected_tool_names = [name for name in selected_tool_names if name]

            draft_payload = copy.deepcopy(session_payload.get("draft_payload", {}))
            yield _assistant_status_event(
                "I will analyze your request against the current draft now."
            )
            assistant_text = await self._run_assistant_message(
                session=session,
                draft_payload=draft_payload,
                history=session_payload.get("messages", []),
                user_message=message,
                selected_tools=selected_tools,
                tool_choice=tool_choice,
                runtime_context=runtime_context,
                access_token=access_token,
            )
            if assistant_text:
                yield _sse_event("assistant", {"delta": assistant_text})

            updated_payload: Optional[Dict[str, Any]] = None
            update_mode = "regenerated_fallback"
            patch_error: Optional[str] = None
            patch_failure_reasons: List[str] = []

            patch_enabled = os.environ.get(
                "APP_UI_PATCH_ENABLED", "true"
            ).strip().lower() in {"1", "true", "yes", "on"}
            if patch_enabled:
                max_attempts = max(1, int(APP_UI_PATCH_RETRIES))
                for attempt_index in range(max_attempts):
                    yield _assistant_status_event(
                        f"I will try a targeted patch first (attempt {attempt_index + 1}/{max_attempts})."
                    )
                    patch_attempt = await self._attempt_patch_update(
                        scope=scope,
                        ui_id=ui_id,
                        name=name,
                        draft_payload=draft_payload,
                        user_message=message,
                        assistant_message=assistant_text,
                        selected_tools=selected_tools,
                        access_token=access_token,
                        previous_metadata=session_payload.get("metadata_snapshot", {}),
                    )
                    if patch_attempt:
                        updated_payload = patch_attempt.get("payload")
                        update_mode = "patch_applied"
                        if attempt_index > 0:
                            logger.info(
                                "[GeneratedUI] Patch update succeeded on retry attempt %s/%s",
                                attempt_index + 1,
                                max_attempts,
                            )
                        break

                    patch_reason = (
                        self.service._last_patch_failure_reason
                        or "unknown_patch_failure"
                    )
                    patch_failure_reasons.append(
                        f"attempt={attempt_index + 1}/{max_attempts}:{patch_reason}"
                    )
                    logger.warning(
                        "[GeneratedUI] Patch attempt %s/%s failed: %s",
                        attempt_index + 1,
                        max_attempts,
                        patch_reason,
                    )

                if updated_payload is None:
                    attempts_text = "; ".join(patch_failure_reasons) or "no_attempts"
                    patch_error = (
                        "Patch validation failed, using full regenerate"
                        f" (attempts={attempts_text})"
                    )
                    yield _assistant_status_event(
                        "I will switch to full regeneration because patch attempts failed."
                    )
                    logger.warning(
                        "[GeneratedUI] Falling back to regenerate after patch failures: %s",
                        attempts_text,
                    )
                    if APP_UI_PATCH_ONLY:
                        raise HTTPException(
                            status_code=409,
                            detail=(
                                "Patch update failed and APP_UI_PATCH_ONLY=true prevents full regeneration "
                                f"(attempts={attempts_text})"
                            ),
                        )

            if updated_payload is None:
                previous = {
                    "metadata": copy.deepcopy(
                        session_payload.get("metadata_snapshot", {}) or {}
                    ),
                    "current": draft_payload,
                }
                yield _assistant_status_event(
                    "I will generate a full updated draft now."
                )
                regenerate_prompt = self._compose_regeneration_prompt(
                    user_message=message,
                    assistant_message=assistant_text,
                    history=session_payload.get("messages", []),
                    runtime_context=runtime_context,
                )
                updated_payload = (
                    await self.service.generation_pipeline._generate_ui_payload(
                        session=session,
                        scope=scope,
                        ui_id=ui_id,
                        name=name,
                        prompt=regenerate_prompt,
                        tools=selected_tool_names,
                        access_token=access_token,
                        previous=previous,
                        runtime_context=runtime_context,
                    )
                )

            messages_history = list(session_payload.get("messages") or [])
            messages_history.append({"role": "user", "content": message})
            messages_history.append({"role": "assistant", "content": assistant_text})

            session_payload["messages"] = messages_history
            session_payload["draft_payload"] = updated_payload
            session_payload["last_tools"] = selected_tool_names
            session_payload["updated_at"] = self.service._now()
            session_payload["draft_version"] = (
                int(session_payload.get("draft_version") or 1) + 1
            )
            self.storage.write_session(scope, ui_id, name, session_id, session_payload)

            ui_event_payload = {
                "session_id": session_id,
                "draft_version": session_payload["draft_version"],
                "update_mode": update_mode,
                "tools": selected_tool_names,
            }
            if patch_error:
                ui_event_payload["warning"] = patch_error
            metadata_obj = (
                updated_payload.get("metadata")
                if isinstance(updated_payload, dict)
                else None
            )
            if isinstance(metadata_obj, dict):
                diagnostics_obj = metadata_obj.get("generation_diagnostics")
                if isinstance(diagnostics_obj, dict):
                    prompt_compaction = diagnostics_obj.get(
                        "message_payload_compaction"
                    )
                    if isinstance(prompt_compaction, dict):
                        ui_event_payload["context_compaction"] = prompt_compaction
            yield _sse_event("ui_updated", ui_event_payload)
            yield _assistant_status_event(
                "I will run the validation tests for this updated draft now."
            )
            queued = await self.service.test_runner._queue_test_run(
                scope=scope,
                ui_id=ui_id,
                name=name,
                session_id=session_id,
                action="run",
                trigger="post_update",
                test_name=None,
                access_token=access_token,
            )
            yield _sse_event(
                "tests_queued",
                {
                    "run_id": queued.get("run_id"),
                    "trigger": queued.get("trigger"),
                    "draft_version": session_payload["draft_version"],
                },
            )
            yield _sse_event(
                "done",
                {
                    "session_id": session_id,
                    "draft_version": session_payload["draft_version"],
                    "update_mode": update_mode,
                },
            )
            yield b"data: [DONE]\n\n"
        except HTTPException as exc:
            yield _sse_event("error", {"error": exc.detail})
        except Exception as exc:
            logger.error(
                "[GeneratedUI] Conversational update failed: %s", exc, exc_info=exc
            )
            yield _sse_event(
                "error", {"error": "Failed to process conversational update"}
            )
        finally:
            runtime_context = None

    def _compose_regeneration_prompt(
        self,
        *,
        user_message: str,
        assistant_message: str,
        history: Sequence[Dict[str, Any]],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        tail = list(history or [])[-6:]
        prompt = (
            "Update the existing UI using the conversational request.\n\n"
            f"User request:\n{user_message}\n\n"
            f"Assistant analysis:\n{assistant_message}\n\n"
            "Recent conversation history (JSON):\n"
            f"{json.dumps(tail, ensure_ascii=False)}\n\n"
            "Component data-loading constraints:\n"
            "- Keep data fetching component-owned by default.\n"
            "- Avoid root-level fan-out Promise.all() across unrelated UI blocks.\n"
            "- Avoid a single global loading gate that blocks the whole screen when sections are independent.\n"
            "- Keep per-component loading/error/data states with local placeholders.\n"
            "- Use namespaced public events plus targeted refetch in affected components only.\n"
            "- In runtime catch blocks, always log with console.error including component/service context.\n"
            "- Keep tests deterministic with mocked service/fetch responses and never seed fetched domain data directly via test state/event payloads."
        )
        return prompt_with_runtime_context(
            prompt=prompt,
            runtime_context=runtime_context,
            purpose="regeneration",
        )

    async def _run_assistant_message(
        self,
        *,
        session: MCPSessionBase,
        draft_payload: Dict[str, Any],
        history: Sequence[Dict[str, Any]],
        user_message: str,
        selected_tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Any],
        runtime_context: Optional[Dict[str, Any]] = None,
        access_token: Optional[str],
    ) -> str:
        system_prompt = (
            "You are an assistant helping a user iteratively edit a generated web UI. "
            "Keep answers concise and implementation-focused. If tools are available, "
            "use them to gather facts before proposing UI changes. "
            "When this workflow will continue automatically, do not ask for permission to proceed. "
            "State the next action assertively with phrasing that starts with 'I will ...'. "
            "Preserve component-owned data loading and partial rendering: each data-owning "
            "component should manage its own loading/error/data states. "
            "Do not introduce root-level Promise.all() fan-out for unrelated components or a "
            "global blocking loading state unless the whole view is intentionally atomic. "
            "Use namespaced public events for targeted refetch in affected components only. "
            "In runtime catch blocks, always log with console.error and include useful context. "
            "Keep tests deterministic with mocked service/fetch responses and avoid direct test seeding of fetched domain data. "
            "If runtime service exchange context is provided, use it to narrow solutioning, "
            "match real payload shapes, and improve proposed test data quality."
        )
        messages: List[Message] = [
            Message(role=MessageRole.SYSTEM, content=system_prompt)
        ]
        messages.append(
            Message(
                role=MessageRole.USER,
                content=(
                    "Current draft context:\n"
                    + json.dumps(
                        {
                            "html": (draft_payload.get("html") or {}),
                            "metadata": (draft_payload.get("metadata") or {}),
                        },
                        ensure_ascii=False,
                    )
                ),
            )
        )
        runtime_prompt_context = runtime_context_for_prompt(
            runtime_context,
            limit=GENERATED_UI_MAX_RUNTIME_EXCHANGES,
            max_console_events=GENERATED_UI_MAX_RUNTIME_CONSOLE_EVENTS,
            max_bytes=GENERATED_UI_MAX_RUNTIME_BYTES,
        )
        if runtime_prompt_context:
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=(
                        "Observed runtime context for this request only:\n"
                        + json.dumps(runtime_prompt_context, ensure_ascii=False)
                    ),
                )
            )
        bounded_history = history_for_prompt(
            history,
            max_entries=GENERATED_UI_MAX_HISTORY_ENTRIES,
            max_bytes=GENERATED_UI_MAX_HISTORY_BYTES,
        )
        for item in bounded_history:
            role = str(item.get("role") or "").lower()
            content = str(item.get("content") or "")
            if role == MessageRole.USER.value:
                messages.append(Message(role=MessageRole.USER, content=content))
            elif role == MessageRole.ASSISTANT.value:
                messages.append(Message(role=MessageRole.ASSISTANT, content=content))
        messages.append(Message(role=MessageRole.USER, content=user_message))

        request = ChatCompletionRequest(
            messages=messages,
            stream=False,
            tools=selected_tools if selected_tools else None,
            tool_choice=tool_choice if tool_choice is not None else "auto",
            extra_headers=UI_MODEL_HEADERS,
        )

        response = await self.tgi_service._non_stream_chat_with_tools(
            session,
            messages,
            selected_tools or [],
            request,
            access_token,
            None,
        )
        return self.service.test_runner._assistant_text_from_response(response)

    async def _attempt_patch_update(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        draft_payload: Dict[str, Any],
        user_message: str,
        assistant_message: str,
        access_token: Optional[str],
        previous_metadata: Dict[str, Any],
        selected_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        self.service._last_patch_failure_reason = None

        def _fail(
            reason: str, detail: Optional[str] = None
        ) -> Optional[Dict[str, Any]]:
            message = reason if not detail else f"{reason}: {detail}"
            self.service._last_patch_failure_reason = message
            logger.warning(
                "[GeneratedUI] Patch update failed (%s) scope=%s:%s ui_id=%s name=%s",
                message,
                scope.kind,
                scope.identifier,
                ui_id,
                name,
            )
            return None

        try:
            system_prompt = (
                "You are a UI patch planner. Return valid JSON only in this shape: "
                '{"patch":{"html":{"page":"...","snippet":"..."},"service_script":"...","components_script":"...","metadata":{...}}}. '
                "Only include fields that need changes. Do not include markdown fences. "
                "Preserve component-owned data loading and partial rendering. Do not rewrite to "
                "one root-level Promise.all() fan-out or a single full-screen blocking loader "
                "for independent components. Keep targeted event-driven refetch behavior. "
                "In runtime catch blocks, always log with console.error and include component/service context. "
                "Keep tests deterministic with mocked service/fetch responses and avoid test-only seeding of fetched domain data."
            )
            payload = {
                "user_message": user_message,
                "assistant_message": assistant_message,
                "current": {
                    "html": draft_payload.get("html"),
                    "service_script": draft_payload.get("service_script"),
                    "components_script": draft_payload.get("components_script"),
                    "metadata": draft_payload.get("metadata"),
                },
            }
            request = ChatCompletionRequest(
                messages=[
                    Message(role=MessageRole.SYSTEM, content=system_prompt),
                    Message(
                        role=MessageRole.USER,
                        content=json.dumps(payload, ensure_ascii=False),
                    ),
                ],
                stream=False,
                response_format=_generation_response_format(
                    schema=_PATCH_UPDATE_SCHEMA,
                    name="generated_ui_patch",
                ),
                extra_headers=UI_MODEL_HEADERS,
            )

            response = await self.tgi_service.llm_client.non_stream_completion(
                request, access_token or "", None
            )
            content = self.service.test_runner._assistant_text_from_response(response)
            if not content:
                return _fail("empty_patch_response")

            try:
                parsed = parse_json(content)
            except Exception as exc:
                return _fail("invalid_patch_json", str(exc))
            patch = parsed.get("patch")
            if not isinstance(patch, dict):
                return _fail("missing_patch_object")

            candidate = copy.deepcopy(draft_payload)
            html_patch = patch.get("html")
            if isinstance(html_patch, dict):
                html_target = candidate.setdefault("html", {})
                for key in ("page", "snippet"):
                    value = html_patch.get(key)
                    if isinstance(value, str) and value.strip():
                        html_target[key] = value

            for key in (
                "service_script",
                "components_script",
                "test_script",
                "dummy_data",
            ):
                value = patch.get(key)
                if isinstance(value, str):
                    candidate[key] = value

            metadata_patch = patch.get("metadata")
            if isinstance(metadata_patch, dict):
                current_metadata = candidate.get("metadata")
                if not isinstance(current_metadata, dict):
                    current_metadata = {}
                merged_metadata = {**current_metadata, **metadata_patch}
                candidate["metadata"] = merged_metadata

            previous = {"metadata": previous_metadata, "current": draft_payload}
            self.service._normalise_payload(
                candidate,
                scope,
                ui_id,
                name,
                user_message,
                previous,
            )

            test_script = candidate.get("test_script") or draft_payload.get(
                "test_script"
            )
            service_script = candidate.get("service_script") or draft_payload.get(
                "service_script"
            )
            components_script = candidate.get("components_script") or draft_payload.get(
                "components_script"
            )
            if (
                isinstance(service_script, str)
                and isinstance(components_script, str)
                and isinstance(test_script, str)
                and test_script.strip()
            ):
                success, _ = self.service._run_tests(
                    service_script,
                    components_script,
                    test_script,
                    candidate.get("dummy_data"),
                )
                if not success:
                    logger.warning(
                        "[GeneratedUI] Patch candidate failed tests; invoking iterative fixer loop"
                    )
                    fix_messages = [
                        Message(role=MessageRole.USER, content=user_message),
                    ]
                    if isinstance(assistant_message, str) and assistant_message.strip():
                        fix_messages.append(
                            Message(
                                role=MessageRole.ASSISTANT,
                                content=assistant_message,
                            )
                        )
                    (
                        fix_success,
                        fixed_service,
                        fixed_components,
                        fixed_test,
                        fixed_dummy_data,
                        _updated_messages,
                    ) = await self.service._iterative_test_fix(
                        service_script=service_script,
                        components_script=components_script,
                        test_script=test_script,
                        dummy_data=candidate.get("dummy_data"),
                        messages=fix_messages,
                        allowed_tools=selected_tools,
                        access_token=access_token,
                        max_attempts=8,
                    )
                    if fix_success:
                        candidate["service_script"] = fixed_service
                        candidate["components_script"] = fixed_components
                        candidate["test_script"] = fixed_test
                        candidate["dummy_data"] = fixed_dummy_data
                        logger.info(
                            "[GeneratedUI] Patch candidate repaired via iterative fixer loop"
                        )
                        return {"payload": candidate}
                    return _fail("patch_tests_failed_fix_loop_failed")

            return {"payload": candidate}
        except Exception as exc:
            return _fail("patch_exception", f"{type(exc).__name__}: {exc}")
