"""Conversational service: chat-driven updates, patch application, assistant messaging.

The ``ConversationalService`` class handles the conversational update flow,
including streaming chat updates, composing regeneration prompts,
running assistant messages with tool use, and applying patch updates.
"""

import copy
import json
import logging
import os
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Sequence,
)

from fastapi import HTTPException

from app.app_facade.env_utils import positive_int_env
from app.app_facade.generated_schemas import generation_response_format
from app.app_facade.patch_ops import (
    PATCH_UPDATE_SCHEMA,
    apply_patch_operations,
    enforce_runtime_script_integrity,
    legacy_patch_to_operations,
    normalize_patch_response,
)
from app.app_facade.generated_types import (
    Actor,
    Scope,
)
from app.app_facade.prompt_helpers import (
    changed_scripts,
    history_for_prompt,
    parse_json,
    prompt_with_runtime_context,
    runtime_context_for_prompt,
    sanitize_runtime_action,
)
from app.app_facade.sse import assistant_status_event, sse_event
from app.app_facade.test_fix_tools import _parse_tap_output
from app.session import MCPSessionBase
from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.vars import (
    APP_UI_PATCH_ONLY,
    APP_UI_PATCH_FIX_ATTEMPTS,
    APP_UI_PATCH_RETRIES,
    GENERATED_UI_MAX_HISTORY_BYTES,
    GENERATED_UI_MAX_HISTORY_ENTRIES,
    GENERATED_UI_MAX_RUNTIME_BYTES,
    GENERATED_UI_MAX_RUNTIME_CONSOLE_EVENTS,
    GENERATED_UI_MAX_RUNTIME_EXCHANGES,
)

logger = logging.getLogger("uvicorn.error")

_PATCH_UPDATE_SCHEMA = PATCH_UPDATE_SCHEMA
_TARGETED_REWRITE_TARGETS = (
    "service_script",
    "components_script",
    "test_script",
    "dummy_data",
)
_TARGETED_REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "enum": list(_TARGETED_REWRITE_TARGETS),
                    },
                    "content": {"type": "string"},
                },
                "required": ["target", "content"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["files"],
    "additionalProperties": False,
}

UI_MODEL_HEADERS = {"x-inxm-model-capability": "code-generation"}

_COMPONENTS_SCRIPT_CONTEXT_CHARS = positive_int_env(
    "CONVERSATIONAL_COMPONENTS_SCRIPT_CONTEXT_CHARS", 6000
)
_PATCH_ASSISTANT_CONTEXT_CHARS = positive_int_env(
    "CONVERSATIONAL_PATCH_ASSISTANT_CONTEXT_CHARS", 2000
)
_TAP_FAILURE_DIAGNOSTIC_KEYS = (
    "not ok ",
    "error:",
    "code:",
    "failureType:",
    "expected:",
    "actual:",
    "operator:",
)


def _generation_response_format(schema=None, name: str = "generated_ui"):
    return generation_response_format(schema=schema, name=name)


_sse_event = sse_event
_assistant_status_event = assistant_status_event


def _test_failure_signature(output: str) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    """Return stable TAP failure details without paths, stacks, or timings."""
    _, failed, failure_names = _parse_tap_output(output)
    diagnostics = tuple(
        line
        for raw_line in (output or "").splitlines()
        if (line := raw_line.strip()).startswith(_TAP_FAILURE_DIAGNOSTIC_KEYS)
    )
    return failed, tuple(failure_names), diagnostics


def _sync_patched_template_script(
    candidate: Dict[str, Any], draft_payload: Dict[str, Any]
) -> bool:
    """Keep the legacy template script mirror aligned after component edits.

    Older generated payloads store the same source in both
    ``components_script`` and ``template_parts.script``. Normalization treats
    divergent values as separate modules and appends the template script. For
    a conversational component patch, the unchanged template value is stale,
    not an additional module, so update that mirror before normalization.
    """
    candidate_components = candidate.get("components_script")
    draft_components = draft_payload.get("components_script")
    if (
        not isinstance(candidate_components, str)
        or candidate_components == draft_components
    ):
        return False

    candidate_parts = candidate.get("template_parts")
    draft_parts = draft_payload.get("template_parts")
    if not isinstance(candidate_parts, dict) or not isinstance(draft_parts, dict):
        return False
    if candidate_parts.get("script") != draft_parts.get("script"):
        return False

    candidate_parts["script"] = candidate_components
    return True


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
                last_patch_failure: Optional[str] = None
                for attempt_index in range(max_attempts):
                    rewrite_files = attempt_index > 0 and last_patch_failure is not None
                    if rewrite_files:
                        yield _assistant_status_event(
                            "I will rewrite only the affected source files because "
                            f"the operation patch failed (attempt {attempt_index + 1}/{max_attempts})."
                        )
                    else:
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
                        failure_feedback=last_patch_failure,
                        rewrite_files=rewrite_files,
                    )
                    if patch_attempt:
                        candidate = patch_attempt.get("payload")
                        scripts_changed = bool(
                            changed_scripts(candidate, draft_payload)
                        )
                        html_changed = (candidate or {}).get("html") != (
                            draft_payload or {}
                        ).get("html")
                        if not scripts_changed and not html_changed:
                            # Patch was a complete no-op — didn't change scripts or HTML.
                            # Treat as failure so we fall through to full regeneration.
                            patch_reason = "patch_no_changes"
                            last_patch_failure = (
                                "previous patch produced no effective changes"
                            )
                            patch_failure_reasons.append(
                                f"attempt={attempt_index + 1}/{max_attempts}:{patch_reason}"
                            )
                            logger.warning(
                                "[GeneratedUI] Patch attempt %s/%s produced no changes, treating as failure",
                                attempt_index + 1,
                                max_attempts,
                            )
                            continue
                        updated_payload = candidate
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
                    last_patch_failure = patch_reason
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
            "Do not emit source files, large code blocks, or a rewritten component: a separate "
            "patch planner applies the edit after your analysis. Describe the intended change "
            "briefly and identify the exact symbol or expression that should change. "
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
        components_script_raw = draft_payload.get("components_script") or ""
        messages.append(
            Message(
                role=MessageRole.USER,
                content=(
                    "Current draft context:\n"
                    + json.dumps(
                        {
                            "html": (draft_payload.get("html") or {}),
                            "components_script": components_script_raw[:_COMPONENTS_SCRIPT_CONTEXT_CHARS],
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
        failure_feedback: Optional[str] = None,
        rewrite_files: bool = False,
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
            patch_system_prompt = (
                "You are a UI patch planner. Return valid JSON only with operations "
                "at the root in this shape: "
                '{"operations":[{"target":"components_script","op":"replace","search":"<exact current text>","content":"<replacement>"}],"metadata":{...}}. '
                "Never wrap the response in a 'patch' property. "
                "Targets: service_script, components_script, test_script, dummy_data, html_page, html_snippet. "
                "Ops: 'replace' swaps an exact 'search' string (copied VERBATIM from the current file, "
                "unique enough to match once) with 'content'; 'insert_before' and 'insert_after' add content "
                "beside an exact existing search anchor; 'append' adds 'content' at the end of the target; "
                "'set' replaces the whole target and must only be used to create a missing file or when "
                "nearly all of it changes. Prefer several small 'replace' operations over one big 'set'; "
                "never re-emit unchanged code. Do not include markdown fences. "
                "Never use replace to add new code by inventing old text that is not in the current target. "
                "For additions, use insert_before/insert_after with a verbatim existing anchor. "
                "For a localized before/after request, replace the smallest unique expression "
                "(for example encodeURIComponent(saga.name) -> encodeURIComponent(saga.id)); "
                "do not return the enclosing function or component. "
                "service_script and components_script are bundled into ONE module at runtime: "
                "never add an import for an identifier that either file already imports, and never "
                "register a pfusch component name that already exists — change existing components "
                "with 'replace' on their current code. "
                "When the user request involves adding, changing, or fixing tests, you MUST include operations on test_script. "
                "When component operations change UI labels, rendered content, DOM structure, "
                "or behavior, inspect the current test_script and include exact replacements for "
                "every affected assertion. A component patch that leaves current tests stale is incomplete. "
                "Update existing affected tests instead of adding redundant new tests unless the user "
                "explicitly requests more coverage. The DOM test stubs do not parse SVG or other "
                "descendants assigned through innerHTML: assert the container's innerHTML/markup "
                "instead of querying those synthetic descendants. "
                "Preserve component-owned data loading and partial rendering. Do not rewrite to "
                "one root-level Promise.all() fan-out or a single full-screen blocking loader "
                "for independent components. Keep targeted event-driven refetch behavior. "
                "In runtime catch blocks, always log with console.error and include component/service context. "
                "Keep tests deterministic with mocked service/fetch responses and avoid test-only seeding of fetched domain data."
            )
            rewrite_system_prompt = (
                "You are a UI source-file editor. Exact-string operations already "
                "failed, so return complete replacement content only for affected "
                "files as valid JSON: "
                '{"files":[{"target":"components_script","content":"<complete file>"},'
                '{"target":"test_script","content":"<complete file>"}]}. '
                "Allowed targets: service_script, components_script, test_script, "
                "dummy_data. Do not return markdown or unchanged files. For a UI "
                "structure or behavior change, return both the complete "
                "components_script and complete test_script. Preserve all unrelated "
                "state, loading logic, data ownership, event-driven refetching, and "
                "component registrations. Do not add a second pfusch import or a "
                "second registration of an existing component. service_script and "
                "components_script share one runtime module. Update existing tests "
                "for the new UI rather than adding redundant tests. The DOM stubs do "
                "not parse descendants assigned through innerHTML; assert stored "
                "markup on the owning container. pfuschTest returns a "
                "PfuschNodeCollection: query it with get(selector), inspect length, "
                "and use at(index)/first for individual results. Do not invent "
                "unsupported query helpers. Keep tests deterministic and mock "
                "service/fetch responses."
            )
            system_prompt = (
                rewrite_system_prompt if rewrite_files else patch_system_prompt
            )
            payload = {
                "user_message": user_message,
                "assistant_message": assistant_message[
                    :_PATCH_ASSISTANT_CONTEXT_CHARS
                ],
                "current": {
                    "html": draft_payload.get("html"),
                    "service_script": draft_payload.get("service_script"),
                    "components_script": draft_payload.get("components_script"),
                    "test_script": draft_payload.get("test_script"),
                    "dummy_data": draft_payload.get("dummy_data"),
                    "metadata": draft_payload.get("metadata"),
                },
            }
            if failure_feedback:
                failure_instruction = (
                    "Produce complete corrected affected files; do not return patch "
                    "anchors or repeat the failed operations."
                    if rewrite_files
                    else (
                        "Fix the described problem; when a search string was not "
                        "found, do not repeat or extend the missing text. Copy an "
                        "exact existing anchor from the current file and use "
                        "insert_before/insert_after when adding new code."
                    )
                )
                payload["previous_attempt_failure"] = (
                    f"{failure_feedback}. {failure_instruction}"
                )
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
                    schema=(
                        _TARGETED_REWRITE_SCHEMA
                        if rewrite_files
                        else _PATCH_UPDATE_SCHEMA
                    ),
                    name=(
                        "generated_ui_targeted_rewrite"
                        if rewrite_files
                        else "generated_ui_patch"
                    ),
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
            if rewrite_files:
                files = parsed.get("files") if isinstance(parsed, dict) else None
                if not isinstance(files, list) or not files:
                    return _fail("invalid_targeted_rewrite_response")
                seen_targets = set()
                operations = []
                for index, file_obj in enumerate(files):
                    if not isinstance(file_obj, dict):
                        return _fail(
                            "invalid_targeted_rewrite_response",
                            f"files[{index}] is not an object",
                        )
                    target = file_obj.get("target")
                    content_value = file_obj.get("content")
                    if (
                        target not in _TARGETED_REWRITE_TARGETS
                        or not isinstance(content_value, str)
                        or not content_value.strip()
                    ):
                        return _fail(
                            "invalid_targeted_rewrite_response",
                            f"files[{index}] has invalid target/content",
                        )
                    if target in seen_targets:
                        return _fail(
                            "invalid_targeted_rewrite_response",
                            f"target '{target}' appears more than once",
                        )
                    seen_targets.add(target)
                    operations.append(
                        {"target": target, "op": "set", "content": content_value}
                    )
                metadata_patch = None
            else:
                patch, patch_shape_error = normalize_patch_response(parsed)
                if patch is None:
                    return _fail("invalid_patch_response", patch_shape_error)

                operations = patch.get("operations")
                if not isinstance(operations, list):
                    # Leniency: accept legacy whole-file patch objects from models
                    # that ignore the operations schema.
                    operations = legacy_patch_to_operations(patch)

                metadata_patch = patch.get("metadata")
            if operations:
                candidate, apply_errors = apply_patch_operations(
                    draft_payload, operations
                )
                if candidate is None:
                    return _fail("patch_apply_failed", "; ".join(apply_errors))
                if _sync_patched_template_script(candidate, draft_payload):
                    logger.info(
                        "[GeneratedUI] Synchronized patched components_script "
                        "to legacy template_parts.script"
                    )
            elif isinstance(metadata_patch, dict) and metadata_patch:
                # Metadata-only patch: nothing to apply to scripts/html.
                candidate = copy.deepcopy(draft_payload)
            else:
                return _fail("patch_no_operations")
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

            # Normalization can materialize legacy template parts into the
            # runtime scripts. Validate the final normalized candidate, not
            # only the pre-normalized patch result.
            integrity_notes, integrity_errors = enforce_runtime_script_integrity(
                candidate
            )
            if integrity_notes:
                logger.info(
                    "[GeneratedUI] Patch import auto-dedup applied: %s",
                    "; ".join(integrity_notes),
                )
            if integrity_errors:
                return _fail(
                    "patch_integrity_failed", "; ".join(integrity_errors)
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
                success, candidate_test_output = self.service._run_tests(
                    service_script,
                    components_script,
                    test_script,
                    candidate.get("dummy_data"),
                )
                if not success:
                    test_script_changed = candidate.get(
                        "test_script"
                    ) != draft_payload.get("test_script")
                    baseline_failed_the_same_way = False
                    if not test_script_changed:
                        baseline_success, baseline_test_output = self.service._run_tests(
                            str(draft_payload.get("service_script") or ""),
                            str(draft_payload.get("components_script") or ""),
                            str(draft_payload.get("test_script") or ""),
                            draft_payload.get("dummy_data"),
                        )
                        candidate_signature = _test_failure_signature(
                            candidate_test_output
                        )
                        baseline_signature = _test_failure_signature(
                            baseline_test_output
                        )
                        candidate_failed = candidate_signature[0]
                        baseline_failed_the_same_way = (
                            not baseline_success
                            and candidate_failed > 0
                            and candidate_signature == baseline_signature
                        )
                    if baseline_failed_the_same_way:
                        logger.warning(
                            "[GeneratedUI] Patch candidate retained the draft's existing "
                            "test failures; accepting the non-regressing patch"
                        )
                        return {"payload": candidate}
                    if not test_script_changed:
                        _, failed_count, failed_tests = _parse_tap_output(
                            candidate_test_output
                        )
                        failed_tests_text = (
                            ", ".join(failed_tests[:3]) or "unknown generated test"
                        )
                        return _fail(
                            "patch_tests_failed_test_update_required",
                            (
                                f"failed_count={failed_count}; "
                                f"failed_tests={failed_tests_text}; "
                                "the component changed but test_script did not. "
                                "Return the component operations again and add exact "
                                "test_script replacements for the affected assertions"
                            ),
                        )
                    logger.warning(
                        "[GeneratedUI] Patch candidate failed tests; invoking iterative fixer loop"
                    )
                    candidate_before_fix = copy.deepcopy(candidate)
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
                        max_attempts=max(1, APP_UI_PATCH_FIX_ATTEMPTS),
                    )
                    if fix_success:
                        candidate["service_script"] = fixed_service
                        candidate["components_script"] = fixed_components
                        candidate["test_script"] = fixed_test
                        candidate["dummy_data"] = fixed_dummy_data
                        if _sync_patched_template_script(
                            candidate, candidate_before_fix
                        ):
                            logger.info(
                                "[GeneratedUI] Synchronized repaired "
                                "components_script to legacy template_parts.script"
                            )
                        # Match the publish path now. A repaired component must
                        # survive normalization without the stale legacy script
                        # being appended as a second module.
                        self.service._normalise_payload(
                            candidate,
                            scope,
                            ui_id,
                            name,
                            user_message,
                            previous,
                        )
                        repair_notes, repair_errors = (
                            enforce_runtime_script_integrity(candidate)
                        )
                        if repair_notes:
                            logger.info(
                                "[GeneratedUI] Repaired patch import auto-dedup "
                                "applied: %s",
                                "; ".join(repair_notes),
                            )
                        if repair_errors:
                            return _fail(
                                "patch_integrity_failed_after_fix",
                                "; ".join(repair_errors),
                            )
                        logger.info(
                            "[GeneratedUI] Patch candidate repaired via iterative fixer loop"
                        )
                        return {"payload": candidate}
                    return _fail("patch_tests_failed_fix_loop_failed")

            return {"payload": candidate}
        except Exception as exc:
            return _fail("patch_exception", f"{type(exc).__name__}: {exc}")
