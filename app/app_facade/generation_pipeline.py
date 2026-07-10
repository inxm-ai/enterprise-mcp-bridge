"""Generation pipeline: initial create, stream generate, stream update, update.

The ``GenerationPipeline`` class owns the full UI generation and update
lifecycle, including phased attempts, dummy-data augmentation, and
test orchestration during generation.

``stream_generate_ui`` (create) and ``stream_update_ui`` (update) share a
single implementation, ``_stream_generate_and_persist``; the public
methods only differ in their preconditions and how the record is persisted.
"""

import copy
import json
import logging
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)

from fastapi import HTTPException

from app.session import MCPSessionBase
from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.protocols.chunk_reader import chunk_reader
from app.app_facade.generated_schemas import generation_response_format
from app.app_facade.generated_phase1 import run_phase1_attempt
from app.app_facade.generated_phase2 import run_phase2_attempt
from app.app_facade.generated_types import (
    Actor,
    Scope,
)
from app.app_facade.patch_ops import enforce_runtime_script_integrity
from app.app_facade.sse import sse_event
from app.app_facade.prompt_helpers import (
    parse_json,
    runtime_context_for_prompt,
    context_state_for_prompt,
    prompt_with_runtime_context,
    history_entry,
    history_for_prompt,
    changed_scripts,
    cap_tools_for_prompt,
    cap_message_payload_for_prompt,
)
from app.vars import (
    GENERATED_UI_MAX_HISTORY_ENTRIES,
    GENERATED_UI_MAX_HISTORY_BYTES,
    GENERATED_UI_MAX_RUNTIME_EXCHANGES,
    GENERATED_UI_MAX_RUNTIME_CONSOLE_EVENTS,
    GENERATED_UI_MAX_RUNTIME_BYTES,
    GENERATED_UI_MAX_TOOLS,
    GENERATED_UI_MAX_TOOLS_BYTES,
    GENERATED_UI_MAX_MESSAGE_PAYLOAD_BYTES,
)

logger = logging.getLogger("uvicorn.error")

UI_MODEL_HEADERS = {"x-inxm-model-capability": "code-generation"}

_DUMMY_DATA_TEST_USAGE_GUIDANCE = (
    "Dummy data module for tests is available as ./dummy_data.js. "
    "Tests MUST import { dummyData, dummyDataSchemaHints, dummyDataGatewayHints } from './dummy_data.js' and use "
    "svc.test.addResolved(toolName, dummyData[toolName]) for final resolved results, "
    "or globalThis.fetch.addRoute(...) when validating raw transport/extraction paths. "
    "If dummyDataGatewayHints?.[toolName]?.mcp_server_id exists, prefer calling "
    "svc.call(dummyDataGatewayHints[toolName].mcp_server_id, args, ...) so gateway tools route correctly. "
    "If dummyDataSchemaHints[toolName] exists, that tool is missing output schema; "
    "the client should ask for schema and regenerate dummy data before relying on that fixture. "
    "Tests MUST NOT throw or fail solely because a schema hint exists; "
    "when hints are present, either inject explicit per-test resolved mocks for asserted fields "
    "or assert resilient UI behavior without assuming unavailable schema fields. "
    "Never import './dummy_data.js' in service_script/components_script; "
    "it is test-only and not browser-delivered at runtime. "
    "Do NOT inject fetched domain data directly via component initial state or "
    "test-only event payloads; components must fetch/refetch themselves. "
    "When asserting concrete field values in tests, derive expectations from a normalized shape (e.g. "
    "const normalized = data.current_air_quality || data; const pm25 = normalized.pm2_5) instead of assuming flat paths. "
    "Do NOT hardcode dynamic time/value literals when fixture payload already provides source-of-truth fields; "
    "assert against transformed fixture values."
)

_PHASE1_CONTRACT_MESSAGE = (
    "PHASE 1 CONTRACT (STRICT): Return ONLY a JSON object for logic generation "
    "with keys: components_script (required), test_script (required), "
    "service_script (optional). Do NOT return template_parts, html, or metadata."
)

# How many characters of each retained failure reason to carry into the next
# retry's messages (see the retry-feedback loop in
# ``_stream_generate_and_persist``), and how many recent reasons to keep.
_PHASE1_RETRY_FEEDBACK_REASON_CHARS = 400
_PHASE1_RETRY_FEEDBACK_MAX_REASONS = 2

# Substrings that identify non-retryable LLM failures: retrying will not help
# (context overflow) or only wastes quota (auth / quota errors).
_FATAL_LLM_ERROR_MARKERS = (
    "maximum context length",
    "context window",
    "insufficient_quota",
    "invalid_api_key",
    "authentication_error",
    "permission_denied",
)


def _generation_response_format(schema=None, name: str = "generated_ui"):
    return generation_response_format(schema=schema, name=name)


def _is_fatal_llm_error(text: str) -> bool:
    lower = (text or "").lower()
    return any(marker in lower for marker in _FATAL_LLM_ERROR_MARKERS)


_sse_event = sse_event


class GenerationPipeline:
    """Owns the full UI generation and update lifecycle."""

    def __init__(self, *, service):
        self.service = service
        self.storage = service.storage
        self.tgi_service = service.tgi_service

    async def create_ui(
        self,
        *,
        session: MCPSessionBase,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        prompt: str,
        tools: Optional[Iterable[str]],
        access_token: Optional[str],
    ) -> Dict[str, Any]:
        if self.storage.exists(scope, ui_id, name):
            raise HTTPException(
                status_code=409,
                detail="Ui already exists for this id and name",
            )

        if scope.kind == "user" and actor.user_id != scope.identifier:
            raise HTTPException(
                status_code=403,
                detail="User uis may only be created by the owning user",
            )

        if scope.kind == "group" and scope.identifier not in set(actor.groups or []):
            raise HTTPException(
                status_code=403,
                detail="Group uis may only be created by group members",
            )

        generated = await self._generate_ui_payload(
            session=session,
            scope=scope,
            ui_id=ui_id,
            name=name,
            prompt=prompt,
            tools=list(tools or []),
            access_token=access_token,
            previous=None,
        )

        timestamp = self.service._now()
        payload_scripts = changed_scripts(generated, None)
        record = {
            "metadata": {
                "id": ui_id,
                "name": name,
                "scope": {"type": scope.kind, "id": scope.identifier},
                "owner": {"type": scope.kind, "id": scope.identifier},
                "created_by": actor.user_id,
                "created_at": timestamp,
                "updated_at": timestamp,
                "version": 1,
                "published_at": timestamp,
                "published_by": actor.user_id,
                "history": [
                    history_entry(
                        action="create",
                        prompt=prompt,
                        tools=list(tools or []),
                        user_id=actor.user_id,
                        generated_at=timestamp,
                        payload_metadata=generated.get("metadata", {}),
                        payload_html=generated.get("html", {}),
                        payload_scripts=payload_scripts or None,
                    )
                ],
            },
            "current": generated,
        }

        self.storage.write(scope, ui_id, name, record)
        return record

    async def _phase_1_attempt(
        self,
        *,
        attempt: int,
        max_attempts: int,
        messages: List[Message],
        allowed_tools: List[Dict[str, Any]],
        dummy_data: Optional[str],
        access_token: Optional[str],
    ) -> AsyncIterator[Union[bytes, Dict[str, Any]]]:
        async for item in run_phase1_attempt(
            attempt=attempt,
            max_attempts=max_attempts,
            messages=messages,
            allowed_tools=allowed_tools,
            dummy_data=dummy_data,
            access_token=access_token,
            tgi_service=self.tgi_service,
            parse_json=parse_json,
            run_tests=self.service._run_tests,
            iterative_test_fix=self.service._iterative_test_fix,
            chunk_reader=chunk_reader,
            ui_model_headers=UI_MODEL_HEADERS,
        ):
            yield item

    async def _phase_2_attempt(
        self,
        *,
        system_prompt: str,
        prompt: str,
        logic_payload: Dict[str, Any],
        access_token: Optional[str],
        instruction: str,
    ) -> AsyncIterator[Union[bytes, Dict[str, Any]]]:
        async for item in run_phase2_attempt(
            system_prompt=system_prompt,
            prompt=prompt,
            logic_payload=logic_payload,
            access_token=access_token,
            instruction=instruction,
            tgi_service=self.tgi_service,
            parse_json=parse_json,
            chunk_reader=chunk_reader,
            ui_model_headers=UI_MODEL_HEADERS,
        ):
            yield item

    def _reusable_dummy_data(
        self,
        *,
        previous: Optional[Dict[str, Any]],
        allowed_tools: Optional[List[Dict[str, Any]]],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Return the previous dummy-data module when it already covers all
        selected tools, so updates skip the sampling + LLM fixture pipeline.

        Fresh runtime observations disable reuse: they usually exist because
        real payload shapes diverged from the fixtures.
        """
        if not previous:
            return None
        if runtime_context_for_prompt(runtime_context):
            return None
        candidate = (previous.get("current") or {}).get("dummy_data")
        if not isinstance(candidate, str) or not candidate.strip():
            return None
        if not self.service.tool_sampler.dummy_data_covers_tools(
            candidate, allowed_tools
        ):
            return None
        return candidate

    async def _prepare_dummy_data(
        self,
        *,
        session: MCPSessionBase,
        scope: Scope,
        ui_id: str,
        name: str,
        prompt: str,
        allowed_tools: Optional[List[Dict[str, Any]]],
        access_token: Optional[str],
        previous: Optional[Dict[str, Any]] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Generate (or reuse) the dummy-data module and augment tools with
        derived output schemas. Returns ``(dummy_data, allowed_tools, reused)``."""
        reusable = self._reusable_dummy_data(
            previous=previous,
            allowed_tools=allowed_tools,
            runtime_context=runtime_context,
        )
        if reusable is not None:
            logger.info(
                "[GenerationPipeline] Reusing existing dummy data module for %s/%s (covers all selected tools)",
                ui_id,
                name,
            )
            dummy_data = reusable
        else:
            dummy_data = await self.service.tool_sampler._generate_dummy_data(
                session=session,
                scope=scope,
                ui_id=ui_id,
                name=name,
                prompt=prompt,
                allowed_tools=allowed_tools,
                access_token=access_token,
                runtime_context=runtime_context,
            )

        allowed_tools, derived_schema_count = (
            await self.service.tool_sampler._augment_tools_with_derived_output_schemas(
                allowed_tools=allowed_tools,
                dummy_data_module=dummy_data,
            )
        )
        if derived_schema_count:
            logger.info(
                "[GenerationPipeline] Added %s derived output schemas from dummy data to allowed tools",
                derived_schema_count,
            )
        return dummy_data, allowed_tools, reusable is not None

    async def stream_generate_ui(
        self,
        *,
        session: MCPSessionBase,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        prompt: str,
        tools: Optional[Iterable[str]],
        access_token: Optional[str],
    ) -> AsyncIterator[bytes]:
        """Stream UI creation as Server-Sent Events (SSE)."""
        logger.info(
            f"[stream_generate_ui] Starting stream for ui_id={ui_id}, name={name}, scope={scope.kind}:{scope.identifier}"
        )
        if self.storage.exists(scope, ui_id, name):
            logger.warning(f"[stream_generate_ui] UI already exists: {ui_id}/{name}")
            yield _sse_event("error", {"error": "Ui already exists for this id and name"})
            return

        if scope.kind == "user" and actor.user_id != scope.identifier:
            logger.warning(
                f"[stream_generate_ui] Permission denied: user {actor.user_id} cannot create UI for user {scope.identifier}"
            )
            yield _sse_event(
                "error", {"error": "User uis may only be created by the owning user"}
            )
            return

        if scope.kind == "group" and scope.identifier not in set(actor.groups or []):
            logger.warning(
                f"[stream_generate_ui] Permission denied: user {actor.user_id} not in group {scope.identifier}"
            )
            yield _sse_event(
                "error", {"error": "Group uis may only be created by group members"}
            )
            return

        async for chunk in self._stream_generate_and_persist(
            session=session,
            scope=scope,
            actor=actor,
            ui_id=ui_id,
            name=name,
            prompt=prompt,
            tools=tools,
            access_token=access_token,
            existing=None,
        ):
            yield chunk

    async def stream_update_ui(
        self,
        *,
        session: MCPSessionBase,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        prompt: str,
        tools: Optional[Iterable[str]],
        access_token: Optional[str],
    ) -> AsyncIterator[bytes]:
        """Stream UI updates as SSE, using the same pipeline as create."""
        logger.info(
            f"[stream_update_ui] Starting stream for ui_id={ui_id}, name={name}, scope={scope.kind}:{scope.identifier}"
        )
        try:
            existing = self.storage.read(scope, ui_id, name)
        except FileNotFoundError:
            logger.warning(f"[stream_update_ui] UI not found: {ui_id}/{name}")
            yield _sse_event("error", {"error": "Ui not found"})
            return

        try:
            self.service._assert_scope_consistency(existing, scope, name)
            self.service._ensure_update_permissions(existing, scope, actor)
        except HTTPException as exc:
            yield _sse_event("error", {"error": exc.detail})
            yield _sse_event("log", {"message": "Page creation failed"})
            return

        async for chunk in self._stream_generate_and_persist(
            session=session,
            scope=scope,
            actor=actor,
            ui_id=ui_id,
            name=name,
            prompt=prompt,
            tools=tools,
            access_token=access_token,
            existing=existing,
        ):
            yield chunk

    async def _stream_generate_and_persist(
        self,
        *,
        session: MCPSessionBase,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        prompt: str,
        tools: Optional[Iterable[str]],
        access_token: Optional[str],
        existing: Optional[Dict[str, Any]],
    ) -> AsyncIterator[bytes]:
        """Shared streaming implementation for create (``existing is None``)
        and update flows: phase 1 (logic + tests, with retries), phase 2
        (presentation), persistence, final ``done`` event."""
        mode = "update" if existing is not None else "create"
        log_tag = f"[stream_{'update' if existing is not None else 'generate'}_ui]"
        requested_tools = list(tools or [])

        attempt = 0
        max_attempts = 3
        messages: List[Message] = []
        allowed_tools: List[Dict[str, Any]] = []
        payload_obj: Dict[str, Any] = {}
        logic_payload: Dict[str, Any] = {}
        dummy_data: Optional[str] = None
        phase1_failure_reasons: List[str] = []
        phase1_fatal_error: bool = False

        try:
            system_prompt = await self.service._build_system_prompt(session)
            logger.info(f"{log_tag} System prompt built, length={len(system_prompt)}")

            allowed_tools = await self.service._select_tools(
                session, requested_tools, prompt
            )
            allowed_tools = cap_tools_for_prompt(
                allowed_tools,
                max_tools=GENERATED_UI_MAX_TOOLS,
                max_bytes=GENERATED_UI_MAX_TOOLS_BYTES,
            )

            message_payload: Dict[str, Any] = {
                "ui": {
                    "id": ui_id,
                    "name": name,
                    "scope": {"type": scope.kind, "id": scope.identifier},
                },
                "request": {
                    "prompt": prompt,
                    "tools": [t["function"]["name"] for t in (allowed_tools or [])],
                    "requested_tools": requested_tools,
                },
            }
            if existing is not None:
                previous_metadata = existing.get("metadata", {})
                message_payload["context"] = {
                    "original_prompt": self.service._initial_prompt(previous_metadata),
                    "history": history_for_prompt(
                        previous_metadata.get("history", []),
                        max_entries=GENERATED_UI_MAX_HISTORY_ENTRIES,
                        max_bytes=GENERATED_UI_MAX_HISTORY_BYTES,
                    ),
                    "current_state": context_state_for_prompt(
                        existing.get("current", {}),
                        max_bytes=max(2048, GENERATED_UI_MAX_HISTORY_BYTES // 2),
                    ),
                }
            message_payload, _prompt_compaction = cap_message_payload_for_prompt(
                message_payload,
                max_bytes=GENERATED_UI_MAX_MESSAGE_PAYLOAD_BYTES,
            )

            messages = [
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(
                    role=MessageRole.USER,
                    content=json.dumps(message_payload, ensure_ascii=False),
                ),
            ]

            yield _sse_event("log", {"message": "Generating dummy data for tests..."})

            dummy_data, allowed_tools, dummy_reused = await self._prepare_dummy_data(
                session=session,
                scope=scope,
                ui_id=ui_id,
                name=name,
                prompt=prompt,
                allowed_tools=allowed_tools,
                access_token=access_token,
                previous=existing,
            )
            if dummy_reused:
                yield _sse_event(
                    "log", {"message": "Reusing existing test fixtures (unchanged tools)"}
                )

            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=(
                        f"{_DUMMY_DATA_TEST_USAGE_GUIDANCE} "
                        f"Tools: {[t['function']['name'] for t in (allowed_tools or [])]}"
                    ),
                )
            )
            messages.append(
                Message(role=MessageRole.USER, content=_PHASE1_CONTRACT_MESSAGE)
            )

            # --- PHASE 1: GENERATE LOGIC AND TESTS ---
            phase1_success = False
            while attempt < max_attempts:
                attempt += 1
                # Clone messages for this attempt so failed-attempt history is
                # discarded; only a compact failure summary carries over.
                attempt_messages = copy.deepcopy(messages)
                if phase1_failure_reasons:
                    recent = " | ".join(
                        reason[:_PHASE1_RETRY_FEEDBACK_REASON_CHARS]
                        for reason in phase1_failure_reasons[
                            -_PHASE1_RETRY_FEEDBACK_MAX_REASONS:
                        ]
                    )
                    attempt_messages.append(
                        Message(
                            role=MessageRole.USER,
                            content=(
                                f"Previous generation attempt(s) failed: {recent}. "
                                "Generate a corrected payload that avoids these failure modes."
                            ),
                        )
                    )

                async for item in self._phase_1_attempt(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    messages=attempt_messages,
                    allowed_tools=allowed_tools,
                    dummy_data=dummy_data,
                    access_token=access_token,
                ):
                    if isinstance(item, bytes):
                        yield item
                    elif isinstance(item, dict) and item.get("type") == "result":
                        if item["success"]:
                            phase1_success = True
                            logic_payload = item["payload"]
                            # Keep the successful attempt history
                            messages = item["messages"]
                        else:
                            reason = (
                                item.get("reason")
                                or item.get("error")
                                or "unknown phase 1 failure"
                            )
                            phase1_failure_reasons.append(f"attempt {attempt}: {reason}")
                            logger.warning(
                                "%s Phase 1 attempt %s failed: %s",
                                log_tag,
                                attempt,
                                reason,
                            )
                            yield _sse_event(
                                "log",
                                {
                                    "message": f"Phase 1 attempt {attempt} failed",
                                    "reason": reason,
                                },
                            )
                            if _is_fatal_llm_error(reason):
                                # Non-retryable error (context overflow, quota
                                # exceeded, invalid auth). Retrying will not help.
                                logger.error(
                                    "%s Fatal LLM error on attempt %s, aborting retries: %s",
                                    log_tag,
                                    attempt,
                                    reason,
                                )
                                phase1_fatal_error = True
                        break

                if phase1_success or phase1_fatal_error:
                    break

            if not phase1_success:
                detail = (
                    " | ".join(phase1_failure_reasons) if phase1_failure_reasons else ""
                )
                logger.error(
                    "%s Failed to generate valid logic after %s attempts. Reasons: %s",
                    log_tag,
                    max_attempts,
                    detail or "none captured",
                )
                error_message = (
                    f"Failed to generate valid logic after {max_attempts} attempts"
                )
                if detail:
                    error_message = f"{error_message}: {detail}"
                yield _sse_event("error", {"error": error_message})
                return

            # --- PHASE 2: GENERATE PRESENTATION ---
            presentation_payload: Dict[str, Any] = {}
            instruction = (
                "Tests passed. Now "
                + ("update" if existing is not None else "generate")
                + " presentation template parts only. "
                "Return `template_parts` and `metadata`. "
                "Do not return `html.page` or `html.snippet`."
            )

            phase2_system_prompt = await self.service._build_phase2_system_prompt(
                session
            )

            async for item in self._phase_2_attempt(
                system_prompt=phase2_system_prompt,
                prompt=prompt,
                logic_payload=logic_payload,
                access_token=access_token,
                instruction=instruction,
            ):
                if isinstance(item, bytes):
                    yield item
                elif isinstance(item, dict) and item.get("type") == "result":
                    if item["success"]:
                        presentation_payload = item["payload"]
                    else:
                        return

            # Merge
            payload_obj = {**logic_payload, **presentation_payload}
            self.service._normalise_payload(
                payload_obj, scope, ui_id, name, prompt, existing
            )

        except HTTPException as exc:
            logger.error(
                f"{log_tag} HTTPException during {mode}: {exc.detail}",
                exc_info=exc,
            )
            yield _sse_event("error", {"error": exc.detail})
            yield _sse_event("log", {"message": "Page creation failed"})
            return
        except Exception as exc:
            logger.error(
                f"{log_tag} Exception during {mode}: {str(exc)}",
                exc_info=exc,
            )
            fallback = (
                "Failed to update ui"
                if existing is not None
                else "Failed to initialize generation"
            )
            yield _sse_event("error", {"error": fallback})
            yield _sse_event("log", {"message": "Page creation failed"})
            return

        # Build final record and persist
        logger.info(f"{log_tag} Building final record")
        timestamp = self.service._now()
        if existing is None:
            payload_scripts = changed_scripts(payload_obj, None)
            record = {
                "metadata": {
                    "id": ui_id,
                    "name": name,
                    "scope": {"type": scope.kind, "id": scope.identifier},
                    "owner": {"type": scope.kind, "id": scope.identifier},
                    "created_by": actor.user_id,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                    "version": 1,
                    "published_at": timestamp,
                    "published_by": actor.user_id,
                    "history": [
                        history_entry(
                            action="create",
                            prompt=prompt,
                            tools=list(tools or []),
                            user_id=actor.user_id,
                            generated_at=timestamp,
                            payload_metadata=payload_obj.get("metadata", {}),
                            payload_html=payload_obj.get("html", {}),
                            payload_scripts=payload_scripts or None,
                        )
                    ],
                },
                "current": payload_obj,
            }
        else:
            existing.setdefault("metadata", {})
            metadata = existing["metadata"]
            self.service._ensure_version_metadata(metadata)
            metadata["updated_at"] = timestamp
            metadata["updated_by"] = actor.user_id
            metadata["version"] = self.service._current_version(metadata) + 1
            metadata["published_at"] = timestamp
            metadata["published_by"] = actor.user_id
            history = metadata.setdefault("history", [])
            payload_scripts = changed_scripts(payload_obj, existing.get("current", {}))
            history.append(
                history_entry(
                    action="update",
                    prompt=prompt,
                    tools=list(tools or []),
                    user_id=actor.user_id,
                    generated_at=timestamp,
                    payload_metadata=payload_obj.get("metadata", {}),
                    payload_html=payload_obj.get("html", {}),
                    payload_scripts=payload_scripts or None,
                )
            )
            existing["current"] = payload_obj
            record = existing

        logger.info(f"{log_tag} Persisting record to storage")
        try:
            self.storage.write(scope, ui_id, name, record)
            logger.info(f"{log_tag} Record persisted successfully")
        except Exception as exc:
            logger.error(
                f"{log_tag} Failed to persist record: {str(exc)}", exc_info=exc
            )
            yield _sse_event(
                "error", {"error": f"Failed to persist generated ui: {str(exc)}"}
            )
            yield _sse_event("log", {"message": "Page creation failed"})
            return

        logger.info(f"{log_tag} Sending final done event")
        expanded_record = record.copy()
        expanded_record["current"] = self.service._expand_payload(record["current"])
        final_payload = json.dumps(
            {
                "status": "created" if existing is None else "updated",
                "record": expanded_record,
            },
            ensure_ascii=False,
        )
        yield _sse_event("log", {"message": "Page successfully generated"})
        yield f"event: done\ndata: {final_payload}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
        logger.info(f"{log_tag} Stream completed successfully")

    async def update_ui(
        self,
        *,
        session: MCPSessionBase,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        prompt: str,
        tools: Optional[Iterable[str]],
        access_token: Optional[str],
    ) -> Dict[str, Any]:
        try:
            existing = self.storage.read(scope, ui_id, name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Ui not found") from exc

        self.service._assert_scope_consistency(existing, scope, name)
        self.service._ensure_update_permissions(existing, scope, actor)

        generated = await self._generate_ui_payload(
            session=session,
            scope=scope,
            ui_id=ui_id,
            name=name,
            prompt=prompt,
            tools=list(tools or []),
            access_token=access_token,
            previous=existing,
        )

        timestamp = self.service._now()
        existing.setdefault("metadata", {})
        metadata = existing["metadata"]
        self.service._ensure_version_metadata(metadata)
        metadata["updated_at"] = timestamp
        metadata["updated_by"] = actor.user_id
        metadata["version"] = self.service._current_version(metadata) + 1
        metadata["published_at"] = timestamp
        metadata["published_by"] = actor.user_id
        history = metadata.setdefault("history", [])
        payload_scripts = changed_scripts(generated, existing.get("current", {}))
        history.append(
            history_entry(
                action="update",
                prompt=prompt,
                tools=list(tools or []),
                user_id=actor.user_id,
                generated_at=timestamp,
                payload_metadata=generated.get("metadata", {}),
                payload_html=generated.get("html", {}),
                payload_scripts=payload_scripts or None,
            )
        )

        existing["current"] = generated

        self.storage.write(scope, ui_id, name, existing)
        return existing

    async def _generate_ui_payload(
        self,
        *,
        session: MCPSessionBase,
        scope: Scope,
        ui_id: str,
        name: str,
        prompt: str,
        tools: List[str],
        access_token: Optional[str],
        previous: Optional[Dict[str, Any]],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        system_prompt = await self.service._build_system_prompt(session)
        prompt_with_runtime = prompt_with_runtime_context(
            prompt=prompt,
            runtime_context=runtime_context,
            purpose="generation",
        )
        message_payload = {
            "ui": {
                "id": ui_id,
                "name": name,
                "scope": {"type": scope.kind, "id": scope.identifier},
            },
            "request": {
                "prompt": prompt_with_runtime,
                "tools": tools,
            },
        }

        if previous:
            previous_metadata = previous.get("metadata", {})
            bounded_history = history_for_prompt(
                previous_metadata.get("history", []),
                max_entries=GENERATED_UI_MAX_HISTORY_ENTRIES,
                max_bytes=GENERATED_UI_MAX_HISTORY_BYTES,
            )
            message_payload["context"] = {
                "original_prompt": self.service._initial_prompt(previous_metadata),
                "history": bounded_history,
                "current_state": context_state_for_prompt(
                    previous.get("current", {}),
                    max_bytes=max(2048, GENERATED_UI_MAX_HISTORY_BYTES // 2),
                ),
            }
        runtime_prompt_context = runtime_context_for_prompt(
            runtime_context,
            limit=GENERATED_UI_MAX_RUNTIME_EXCHANGES,
            max_console_events=GENERATED_UI_MAX_RUNTIME_CONSOLE_EVENTS,
            max_bytes=GENERATED_UI_MAX_RUNTIME_BYTES,
        )
        if runtime_prompt_context:
            context_obj = message_payload.setdefault("context", {})
            if runtime_prompt_context.get("service_exchanges"):
                context_obj["runtime_service_exchanges"] = runtime_prompt_context.get(
                    "service_exchanges"
                )
            if runtime_prompt_context.get("console_events"):
                context_obj["runtime_console_events"] = runtime_prompt_context.get(
                    "console_events"
                )

        message_payload, prompt_compaction = cap_message_payload_for_prompt(
            message_payload,
            max_bytes=GENERATED_UI_MAX_MESSAGE_PAYLOAD_BYTES,
        )

        messages = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(
                role=MessageRole.USER,
                content=json.dumps(message_payload, ensure_ascii=False),
            ),
        ]

        allowed_tools = await self.service._select_tools(session, tools, prompt)
        allowed_tools = cap_tools_for_prompt(
            allowed_tools,
            max_tools=GENERATED_UI_MAX_TOOLS,
            max_bytes=GENERATED_UI_MAX_TOOLS_BYTES,
        )

        dummy_data, allowed_tools, _dummy_reused = await self._prepare_dummy_data(
            session=session,
            scope=scope,
            ui_id=ui_id,
            name=name,
            prompt=prompt_with_runtime,
            allowed_tools=allowed_tools,
            access_token=access_token,
            previous=previous,
            runtime_context=runtime_context,
        )

        messages.append(
            Message(
                role=MessageRole.USER,
                content=(
                    f"{_DUMMY_DATA_TEST_USAGE_GUIDANCE} "
                    f"Tools: {[t['function']['name'] for t in (allowed_tools or [])]}"
                ),
            )
        )

        chat_request = ChatCompletionRequest(
            messages=messages,
            tools=allowed_tools if allowed_tools else None,
            stream=True,
            response_format=_generation_response_format(),
            extra_headers=UI_MODEL_HEADERS,
        )
        self.service._maybe_dump_chat_request(
            chat_request=chat_request,
            scope=scope,
            ui_id=ui_id,
            name=name,
            prompt=prompt,
            tools=tools,
            message_payload=message_payload,
        )

        # Use streaming to collect the response
        content = ""
        stream_source = self.tgi_service.llm_client.stream_completion(
            chat_request, access_token or "", None
        )

        async with chunk_reader(stream_source) as reader:
            async for parsed in reader.as_parsed():
                if parsed.is_done:
                    break
                if parsed.content:
                    content += parsed.content

        if not content:
            raise HTTPException(status_code=502, detail="Generation response was empty")

        payload = parse_json(content)
        payload["dummy_data"] = payload.get("dummy_data") or dummy_data
        integrity_notes, integrity_errors = enforce_runtime_script_integrity(payload)
        if integrity_notes:
            logger.info(
                "[_generate_ui_payload] Deduplicated runtime imports: %s",
                "; ".join(integrity_notes),
            )
        if integrity_errors:
            # Single-shot path has no retry loop; surface loudly so the
            # follow-up test run/fixer addresses it with full context.
            logger.warning(
                "[_generate_ui_payload] Runtime integrity issues in generated payload: %s",
                "; ".join(integrity_errors),
            )
        self.service._normalise_payload(payload, scope, ui_id, name, prompt, previous)
        if prompt_compaction:
            metadata_obj = payload.get("metadata")
            if not isinstance(metadata_obj, dict):
                metadata_obj = {}
                payload["metadata"] = metadata_obj
            diagnostics_obj = metadata_obj.get("generation_diagnostics")
            if not isinstance(diagnostics_obj, dict):
                diagnostics_obj = {}
                metadata_obj["generation_diagnostics"] = diagnostics_obj
            diagnostics_obj["message_payload_compaction"] = prompt_compaction
        return payload
