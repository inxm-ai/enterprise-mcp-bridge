"""Generation pipeline: initial create, stream generate, stream update, update.

The ``GenerationPipeline`` class owns the full UI generation and update
lifecycle, including phased attempts, dummy-data augmentation, and
test orchestration during generation.
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


def _generation_response_format(schema=None, name: str = "generated_ui"):
    return generation_response_format(schema=schema, name=name)


def _is_fatal_llm_error(text: str) -> bool:
    return any(
        marker in text for marker in ("maximum context length", "context window")
    )


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

        # TODO: remove after testing
        logger.info(
            "[create_ui] Initial generation starting point:\n"
            "--- Dummy Data ---\n%s\n"
            "--- Service Script ---\n%s\n"
            "--- Components Script ---\n%s\n"
            "--- Test Script ---\n%s\n",
            generated.get("dummy_data", ""),
            generated.get("service_script", ""),
            generated.get("components_script", ""),
            generated.get("test_script", ""),
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
        """
        Stream UI generation as Server-Sent Events (SSE).

        Yields bytes that are already formatted as SSE messages. The stream
        will emit keepalive comments roughly every 10 seconds during idle
        model streaming to avoid client timeouts.
        """
        logger.info(
            f"[stream_generate_ui] Starting stream for ui_id={ui_id}, name={name}, scope={scope.kind}:{scope.identifier}"
        )
        requested_tools = list(tools or [])

        # Basic existence and permission checks similar to create_ui
        if self.storage.exists(scope, ui_id, name):
            logger.warning(f"[stream_generate_ui] UI already exists: {ui_id}/{name}")
            # SSE error message and stop
            payload = json.dumps({"error": "Ui already exists for this id and name"})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            return

        if scope.kind == "user" and actor.user_id != scope.identifier:
            logger.warning(
                f"[stream_generate_ui] Permission denied: user {actor.user_id} cannot create UI for user {scope.identifier}"
            )
            payload = json.dumps(
                {"error": "User uis may only be created by the owning user"}
            )
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            return

        if scope.kind == "group" and scope.identifier not in set(actor.groups or []):
            logger.warning(
                f"[stream_generate_ui] Permission denied: user {actor.user_id} not in group {scope.identifier}"
            )
            payload = json.dumps(
                {"error": "Group uis may only be created by group members"}
            )
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            return

        # Wrap initialization in try-catch to ensure we yield error messages
        # if anything fails before streaming starts
        logger.info("[stream_generate_ui] Building system prompt and selecting tools")

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
            logger.info(
                f"[stream_generate_ui] System prompt built, length={len(system_prompt)}"
            )

            allowed_tools = await self.service._select_tools(
                session, requested_tools, prompt
            )

            message_payload = {
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

            initial_message = Message(
                role=MessageRole.USER,
                content=json.dumps(message_payload, ensure_ascii=False),
            )

            messages = [
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                initial_message,
            ]

            yield f"event: log\ndata: {json.dumps({'message': 'Generating dummy data for tests...'})}\n\n".encode(
                "utf-8"
            )

            dummy_data = await self.service.tool_sampler._generate_dummy_data(
                session=session,
                scope=scope,
                ui_id=ui_id,
                name=name,
                prompt=prompt,
                allowed_tools=allowed_tools,
                access_token=access_token,
            )

            allowed_tools, derived_schema_count = (
                await self.service.tool_sampler._augment_tools_with_derived_output_schemas(
                    allowed_tools=allowed_tools,
                    dummy_data_module=dummy_data,
                )
            )
            if derived_schema_count:
                logger.info(
                    "[stream_generate_ui] Added %s derived output schemas from dummy data to allowed tools",
                    derived_schema_count,
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
                Message(
                    role=MessageRole.USER,
                    content=(
                        "PHASE 1 CONTRACT (STRICT): Return ONLY a JSON object for logic generation "
                        "with keys: components_script (required), test_script (required), "
                        "service_script (optional). Do NOT return template_parts, html, or metadata."
                    ),
                )
            )

            # --- PHASE 1: GENERATE LOGIC AND TESTS ---
            phase1_success = False
            while attempt < max_attempts:
                attempt += 1
                # Clone messages for this attempt to avoid polluting history on retry
                attempt_messages = copy.deepcopy(messages)

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
                            # Update main messages with the successful history
                            messages = item["messages"]
                        else:
                            reason = (
                                item.get("reason")
                                or item.get("error")
                                or "unknown phase 1 failure"
                            )
                            phase1_failure_reasons.append(
                                f"attempt {attempt}: {reason}"
                            )
                            logger.warning(
                                "[stream_generate_ui] Phase 1 attempt %s failed: %s",
                                attempt,
                                reason,
                            )
                            yield f"event: log\ndata: {json.dumps({'message': f'Phase 1 attempt {attempt} failed', 'reason': reason})}\n\n".encode(
                                "utf-8"
                            )
                            if _is_fatal_llm_error(reason):
                                # Non-retryable error (quota exceeded, invalid auth, etc.).
                                # Stop immediately – retrying will not help and only wastes quota.
                                logger.error(
                                    "[stream_generate_ui] Fatal LLM error on attempt %s, aborting retries: %s",
                                    attempt,
                                    reason,
                                )
                                phase1_fatal_error = True
                            # Should we update messages here?
                            # If we update messages here, we are keeping failed attempt history, which defeats "discarding".
                            # The user requested "cleanly discarded once an attempt completed".
                            # So we DO NOT update `messages` on failure.
                            # The next attempt will start from original `messages` state (fresh retry).
                            pass
                        break

                if phase1_success or phase1_fatal_error:
                    break

            if not phase1_success:
                detail = (
                    " | ".join(phase1_failure_reasons) if phase1_failure_reasons else ""
                )
                logger.error(
                    "[stream_generate_ui] Failed to generate valid logic after %s attempts. Reasons: %s",
                    max_attempts,
                    detail or "none captured",
                )
                error_message = (
                    f"Failed to generate valid logic after {max_attempts} attempts"
                )
                if detail:
                    error_message = f"{error_message}: {detail}"
                yield f"event: error\ndata: {json.dumps({'error': error_message})}\n\n".encode(
                    "utf-8"
                )
                return

            # --- PHASE 2: GENERATE PRESENTATION ---
            presentation_payload: Dict[str, Any] = {}
            instruction = (
                "Tests passed. Now generate presentation template parts only. "
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
                payload_obj, scope, ui_id, name, prompt, None
            )

        except HTTPException as exc:
            logger.error(
                f"[stream_generate_ui] HTTPException during initialization: {exc.detail}",
                exc_info=exc,
            )
            payload = json.dumps({"error": exc.detail})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            yield f"event: log\ndata: {json.dumps({'message': 'Page creation failed'})}\n\n".encode(
                "utf-8"
            )
            return
        except Exception as exc:
            logger.error(
                f"[stream_generate_ui] Exception during initialization: {str(exc)}",
                exc_info=exc,
            )
            payload = json.dumps({"error": "Failed to initialize generation"})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            yield f"event: log\ndata: {json.dumps({'message': 'Page creation failed'})}\n\n".encode(
                "utf-8"
            )
            return

        # Build final record and persist
        logger.info("[stream_generate_ui] Building final record")
        timestamp = self.service._now()
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

        # persist
        logger.info("[stream_generate_ui] Persisting record to storage")
        try:
            self.storage.write(scope, ui_id, name, record)
            logger.info("[stream_generate_ui] Record persisted successfully")
        except Exception as e:
            logger.error(
                f"[stream_generate_ui] Failed to persist record: {str(e)}", exc_info=e
            )
            payload = json.dumps({"error": f"Failed to persist generated ui: {str(e)}"})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            yield f"event: log\ndata: {json.dumps({'message': 'Page creation failed'})}\n\n".encode(
                "utf-8"
            )
            return

        # Final event with the created record
        logger.info("[stream_generate_ui] Sending final done event")

        # We want to send the expanded record to the client so they can run it
        expanded_record = record.copy()
        expanded_record["current"] = self.service._expand_payload(record["current"])

        final_payload = json.dumps(
            {"status": "created", "record": expanded_record}, ensure_ascii=False
        )
        yield f"event: log\ndata: {json.dumps({'message': 'Page successfully generated'})}\n\n".encode(
            "utf-8"
        )
        yield f"event: done\ndata: {final_payload}\n\n".encode("utf-8")
        # Send proper SSE done marker
        yield b"data: [DONE]\n\n"
        logger.info("[stream_generate_ui] Stream completed successfully")

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
        """
        Stream UI updates as Server-Sent Events (SSE).

        Uses the same generation/test pipeline as create, but applies the
        results to an existing UI record with update history.
        """
        logger.info(
            f"[stream_update_ui] Starting stream for ui_id={ui_id}, name={name}, scope={scope.kind}:{scope.identifier}"
        )
        requested_tools = list(tools or [])

        try:
            existing = self.storage.read(scope, ui_id, name)
        except FileNotFoundError:
            logger.warning(f"[stream_update_ui] UI not found: {ui_id}/{name}")
            payload = json.dumps({"error": "Ui not found"})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            return

        try:
            self.service._assert_scope_consistency(existing, scope, name)
            self.service._ensure_update_permissions(existing, scope, actor)
        except HTTPException as exc:
            payload = json.dumps({"error": exc.detail})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            yield f"event: log\ndata: {json.dumps({'message': 'Page creation failed'})}\n\n".encode(
                "utf-8"
            )
            return

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
            logger.info(
                f"[stream_update_ui] System prompt built, length={len(system_prompt)}"
            )

            allowed_tools = await self.service._select_tools(
                session, requested_tools, prompt
            )

            previous_metadata = existing.get("metadata", {})
            message_payload = {
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
                "context": {
                    "original_prompt": self.service._initial_prompt(previous_metadata),
                    "history": history_for_prompt(previous_metadata.get("history", [])),
                    "current_state": context_state_for_prompt(
                        existing.get("current", {})
                    ),
                },
            }

            initial_message = Message(
                role=MessageRole.USER,
                content=json.dumps(message_payload, ensure_ascii=False),
            )

            messages = [
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                initial_message,
            ]

            yield f"event: log\ndata: {json.dumps({'message': 'Generating dummy data for tests...'})}\n\n".encode(
                "utf-8"
            )

            dummy_data = await self.service.tool_sampler._generate_dummy_data(
                session=session,
                scope=scope,
                ui_id=ui_id,
                name=name,
                prompt=prompt,
                allowed_tools=allowed_tools,
                access_token=access_token,
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
                Message(
                    role=MessageRole.USER,
                    content=(
                        "PHASE 1 CONTRACT (STRICT): Return ONLY a JSON object for logic generation "
                        "with keys: components_script (required), test_script (required), "
                        "service_script (optional). Do NOT return template_parts, html, or metadata."
                    ),
                )
            )

            # --- PHASE 1: GENERATE LOGIC AND TESTS ---
            phase1_success = False
            while attempt < max_attempts:
                attempt += 1
                attempt_messages = copy.deepcopy(messages)

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
                            messages = item["messages"]
                        else:
                            reason = (
                                item.get("reason")
                                or item.get("error")
                                or "unknown phase 1 failure"
                            )
                            phase1_failure_reasons.append(
                                f"attempt {attempt}: {reason}"
                            )
                            logger.warning(
                                "[stream_update_ui] Phase 1 attempt %s failed: %s",
                                attempt,
                                reason,
                            )
                            yield f"event: log\ndata: {json.dumps({'message': f'Phase 1 attempt {attempt} failed', 'reason': reason})}\n\n".encode(
                                "utf-8"
                            )
                            if _is_fatal_llm_error(reason):
                                # Non-retryable error (quota exceeded, invalid auth, etc.).
                                # Stop immediately – retrying will not help and only wastes quota.
                                logger.error(
                                    "[stream_update_ui] Fatal LLM error on attempt %s, aborting retries: %s",
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
                    "[stream_update_ui] Failed to generate valid logic after %s attempts. Reasons: %s",
                    max_attempts,
                    detail or "none captured",
                )
                error_message = (
                    f"Failed to generate valid logic after {max_attempts} attempts"
                )
                if detail:
                    error_message = f"{error_message}: {detail}"
                yield f"event: error\ndata: {json.dumps({'error': error_message})}\n\n".encode(
                    "utf-8"
                )
                return

            # --- PHASE 2: GENERATE PRESENTATION ---
            presentation_payload: Dict[str, Any] = {}
            instruction = (
                "Tests passed. Now update presentation template parts only. "
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

            payload_obj = {**logic_payload, **presentation_payload}
            self.service._normalise_payload(
                payload_obj, scope, ui_id, name, prompt, existing
            )

        except HTTPException as exc:
            logger.error(
                f"[stream_update_ui] HTTPException during update: {exc.detail}",
                exc_info=exc,
            )
            payload = json.dumps({"error": exc.detail})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            yield f"event: log\ndata: {json.dumps({'message': 'Page creation failed'})}\n\n".encode(
                "utf-8"
            )
            return
        except Exception as exc:
            logger.error(
                f"[stream_update_ui] Exception during update: {str(exc)}",
                exc_info=exc,
            )
            payload = json.dumps({"error": "Failed to update ui"})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            yield f"event: log\ndata: {json.dumps({'message': 'Page creation failed'})}\n\n".encode(
                "utf-8"
            )
            return

        # Build final record and persist
        logger.info("[stream_update_ui] Building updated record")
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

        logger.info("[stream_update_ui] Persisting record to storage")
        try:
            self.storage.write(scope, ui_id, name, existing)
            logger.info("[stream_update_ui] Record persisted successfully")
        except Exception as exc:
            logger.error(
                f"[stream_update_ui] Failed to persist record: {str(exc)}",
                exc_info=exc,
            )
            payload = json.dumps(
                {"error": f"Failed to persist generated ui: {str(exc)}"}
            )
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            yield f"event: log\ndata: {json.dumps({'message': 'Page creation failed'})}\n\n".encode(
                "utf-8"
            )
            return

        logger.info("[stream_update_ui] Sending final done event")
        expanded_record = existing.copy()
        expanded_record["current"] = self.service._expand_payload(existing["current"])
        final_payload = json.dumps(
            {"status": "updated", "record": expanded_record}, ensure_ascii=False
        )
        yield f"event: log\ndata: {json.dumps({'message': 'Page successfully generated'})}\n\n".encode(
            "utf-8"
        )
        yield f"event: done\ndata: {final_payload}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
        logger.info("[stream_update_ui] Stream completed successfully")

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

        dummy_data = await self.service.tool_sampler._generate_dummy_data(
            session=session,
            scope=scope,
            ui_id=ui_id,
            name=name,
            prompt=prompt_with_runtime,
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
                "[_generate_ui_payload] Added %s derived output schemas from dummy data to allowed tools",
                derived_schema_count,
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
