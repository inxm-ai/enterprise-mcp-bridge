import logging
import os
import re
import asyncio
import json
import subprocess
import tempfile
import shutil
import copy
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from typing import AsyncIterator

from fastapi import HTTPException

from app.session import MCPSessionBase
from app.vars import (
    MCP_BASE_PATH,
    GENERATED_UI_PROMPT_DUMP,
    APP_UI_SESSION_TTL_MINUTES,
    APP_UI_PATCH_ENABLED,
)
from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.protocols.chunk_reader import chunk_reader
from app.tgi.services.proxied_tgi_service import ProxiedTGIService
from app.app_facade.generated_output_factory import GeneratedUIOutputFactory
from app.app_facade.generated_phase1 import run_phase1_attempt
from app.app_facade.generated_phase2 import run_phase2_attempt
from app.app_facade.generated_dummy_data import DummyDataGenerator
from app.app_facade.generated_schemas import (
    generation_response_format,
)
from app.app_facade.generated_storage import GeneratedUIStorage
from app.app_facade.generated_types import (
    Actor,
    Scope,  # re-exported for route/tests import compatibility
    validate_identifier,  # noqa: F401  # re-exported for route/tests import compatibility
)
from app.app_facade.test_fix_tools import run_tool_driven_test_fix


logger = logging.getLogger("uvicorn.error")

DEFAULT_DESIGN_PROMPT = (
    "Use lightweight, responsive layouts. Prefer utility-first styling via Tailwind "
    "CSS conventions when no explicit design system guidance is provided."
)

UI_MODEL_HEADERS = {"x-inxm-model-capability": "code-generation"}

SCRIPT_KEYS = ("service_script", "components_script", "test_script", "dummy_data")


def _sse_event(event: str, payload: Dict[str, Any]) -> bytes:
    return (
        f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode(
            "utf-8"
        )
    )


def _load_pfusch_prompt() -> str:
    """Load the pfusch ui prompt from the markdown file and replace placeholders."""
    prompt_path = os.path.join(os.path.dirname(__file__), "pfusch_ui_prompt.md")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()
        # Replace the MCP_BASE_PATH placeholder
        return prompt_content.replace("{{MCP_BASE_PATH}}", MCP_BASE_PATH)
    except Exception as e:
        logger.error(f"Error loading pfusch prompt: {e}")
        raise e


def _load_pfusch_phase2_prompt() -> str:
    """Load the pfusch phase 2 presentation prompt."""
    prompt_path = os.path.join(os.path.dirname(__file__), "pfusch_ui_phase2_prompt.md")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()
        return prompt_content
    except Exception as e:
        logger.error(f"Error loading pfusch phase 2 prompt: {e}")
        raise e


def _generation_response_format(
    schema: Optional[Dict[str, Any]] = None, name: str = "generated_ui"
) -> Dict[str, Any]:
    return generation_response_format(schema=schema, name=name)


class GeneratedUIService:
    def __init__(
        self,
        *,
        storage: GeneratedUIStorage,
        tgi_service: Optional[ProxiedTGIService] = None,
    ):
        self.storage = storage
        self.tgi_service = tgi_service or ProxiedTGIService()
        self.output_factory = GeneratedUIOutputFactory()
        self.dummy_data_generator = DummyDataGenerator(self.tgi_service)

    def _current_version(self, metadata: Dict[str, Any]) -> int:
        raw = metadata.get("version")
        try:
            parsed = int(raw)
            return parsed if parsed > 0 else 1
        except (TypeError, ValueError):
            return 1

    def _ensure_version_metadata(self, metadata: Dict[str, Any]) -> None:
        version = self._current_version(metadata)
        metadata["version"] = version
        if not metadata.get("published_at"):
            metadata["published_at"] = (
                metadata.get("updated_at") or metadata.get("created_at") or self._now()
            )
        if not metadata.get("published_by"):
            metadata["published_by"] = metadata.get("updated_by") or metadata.get(
                "created_by"
            )

    def _run_tests(
        self,
        service_code: str,
        components_code: str,
        test_code: str,
        dummy_data: Optional[str] = None,
    ) -> Tuple[bool, str]:
        helpers_dir = os.path.join(os.path.dirname(__file__), "node_test_helpers")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy helpers including pfusch.js
            for filename in ["domstubs.js", "pfusch.js"]:
                src = os.path.join(helpers_dir, filename)
                dst = os.path.join(tmpdir, filename)
                if os.path.exists(src):
                    shutil.copy(src, dst)

            with open(os.path.join(tmpdir, "package.json"), "w", encoding="utf-8") as f:
                f.write('{"type":"module"}\n')

            mocked_components_code = (
                (components_code or "")
                .replace(
                    "https://matthiaskainer.github.io/pfusch/pfusch.min.js",
                    "./pfusch.js",
                )
                .replace(
                    "https://matthiaskainer.github.io/pfusch/pfusch.js", "./pfusch.js"
                )
            )
            combined_code = (service_code or "") + "\n\n" + mocked_components_code

            with open(os.path.join(tmpdir, "app.js"), "w", encoding="utf-8") as f:
                f.write(combined_code)

            with open(os.path.join(tmpdir, "user_test.js"), "w", encoding="utf-8") as f:
                f.write(test_code or "")

            if dummy_data is not None:
                with open(
                    os.path.join(tmpdir, "dummy_data.js"), "w", encoding="utf-8"
                ) as f:
                    f.write(dummy_data or "")

            test_wrapper = (
                "import { setupDomStubs, pfuschTest } from './domstubs.js';\n"
                "if (typeof globalThis.HTMLElement === 'undefined') {\n"
                "  setupDomStubs();\n"
                "}\n"
                "await import('./user_test.js');\n"
            )
            with open(os.path.join(tmpdir, "test.js"), "w", encoding="utf-8") as f:
                f.write(test_wrapper)

            # Run node --test
            try:
                # set path to include current directory so imports work if needed
                env = os.environ.copy()
                env["NODE_PATH"] = tmpdir
                result = subprocess.run(
                    ["node", "--test", "test.js"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=30,  # Safety timeout
                    env=env,
                )
                output = result.stdout + "\n" + result.stderr
                if result.returncode != 0:
                    output_tail = output[-4000:] if len(output) > 4000 else output
                    if len(output) > 4000:
                        output_tail = (
                            f"...(trimmed {len(output) - 4000} chars)\n{output_tail}"
                        )
                    logger.error(
                        "[GeneratedUI] Test run failed. Output tail:\n%s", output_tail
                    )
                return result.returncode == 0, output
            except subprocess.TimeoutExpired:
                return False, "Tests timed out after 30 seconds."
            except Exception as e:
                return False, f"Error running tests: {str(e)}"

    async def _generate_dummy_data(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        prompt: str,
        allowed_tools: Optional[List[Dict[str, Any]]],
        access_token: Optional[str],
    ) -> str:
        if not allowed_tools:
            return "export const dummyData = {};\n"

        tool_specs: List[Dict[str, Any]] = []
        for tool in allowed_tools or []:
            if isinstance(tool, dict):
                function = tool.get("function", {})
                tool_name = function.get("name")
                output_schema = function.get("outputSchema") or tool.get("outputSchema")
            else:
                function = getattr(tool, "function", None)
                tool_name = None
                output_schema = getattr(tool, "outputSchema", None)
                if function and hasattr(function, "name"):
                    tool_name = function.name
                if function and hasattr(function, "outputSchema"):
                    output_schema = getattr(function, "outputSchema")
            if tool_name:
                tool_specs.append(
                    {
                        "name": tool_name,
                        "outputSchema": output_schema,
                    }
                )

        return await self.dummy_data_generator.generate_dummy_data(
            prompt=prompt,
            tool_specs=tool_specs,
            ui_model_headers=UI_MODEL_HEADERS,
        )

    async def _iterative_test_fix(
        self,
        *,
        service_script: str,
        components_script: str,
        test_script: str,
        dummy_data: Optional[str],
        messages: List[Message],
        allowed_tools: Optional[List[Dict[str, Any]]],
        access_token: Optional[str],
        max_attempts: int = 25,
        event_queue: Optional[asyncio.Queue] = None,
    ) -> Tuple[bool, str, str, str, Optional[str], List[Message]]:
        return await run_tool_driven_test_fix(
            tgi_service=self.tgi_service,
            service_script=service_script,
            components_script=components_script,
            test_script=test_script,
            dummy_data=dummy_data,
            messages=messages,
            allowed_tools=allowed_tools,
            access_token=access_token,
            max_attempts=max_attempts,
            event_queue=event_queue,
            extra_headers=UI_MODEL_HEADERS,
        )

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

        timestamp = self._now()
        payload_scripts = self._changed_scripts(generated, None)
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
                    self._history_entry(
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
            parse_json=self._parse_json,
            run_tests=self._run_tests,
            iterative_test_fix=self._iterative_test_fix,
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
            parse_json=self._parse_json,
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

        try:
            system_prompt = await self._build_system_prompt(session)
            logger.info(
                f"[stream_generate_ui] System prompt built, length={len(system_prompt)}"
            )

            allowed_tools = await self._select_tools(session, requested_tools, prompt)

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

            dummy_data = await self._generate_dummy_data(
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
                        "Dummy data module for tests is available as ./dummy_data.js. "
                        "Tests MUST import { dummyData } from './dummy_data.js' and use "
                        "dummyData[toolName] as the mocked response.json(). "
                        f"Tools: {[t['function']['name'] for t in (allowed_tools or [])]}"
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
                            # Should we update messages here?
                            # If we update messages here, we are keeping failed attempt history, which defeats "discarding".
                            # The user requested "cleanly discarded once an attempt completed".
                            # So we DO NOT update `messages` on failure.
                            # The next attempt will start from original `messages` state (fresh retry).
                            pass
                        break

                if phase1_success:
                    break

            if not phase1_success:
                yield f"event: error\ndata: {json.dumps({'error': 'Failed to generate valid logic after 3 attempts'})}\n\n".encode(
                    "utf-8"
                )
                return

            # --- PHASE 2: GENERATE PRESENTATION ---
            presentation_payload: Dict[str, Any] = {}
            instruction = "Tests passed. Now generate the HTML page that uses these components. Return the `html` object and `metadata`."

            phase2_system_prompt = await self._build_phase2_system_prompt(session)

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
            self._normalise_payload(payload_obj, scope, ui_id, name, prompt, None)

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
        timestamp = self._now()
        payload_scripts = self._changed_scripts(payload_obj, None)
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
                    self._history_entry(
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
        expanded_record["current"] = self._expand_payload(record["current"])

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
            self._assert_scope_consistency(existing, scope, name)
            self._ensure_update_permissions(existing, scope, actor)
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

        try:
            system_prompt = await self._build_system_prompt(session)
            logger.info(
                f"[stream_update_ui] System prompt built, length={len(system_prompt)}"
            )

            allowed_tools = await self._select_tools(session, requested_tools, prompt)

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
                    "original_prompt": self._initial_prompt(previous_metadata),
                    "history": self._history_for_prompt(
                        previous_metadata.get("history", [])
                    ),
                    "current_state": existing.get("current", {}),
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

            dummy_data = await self._generate_dummy_data(
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
                        "Dummy data module for tests is available as ./dummy_data.js. "
                        "Tests MUST import { dummyData } from './dummy_data.js' and use "
                        "dummyData[toolName] as the mocked response.json(). "
                        f"Tools: {[t['function']['name'] for t in (allowed_tools or [])]}"
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
                        break

                if phase1_success:
                    break

            if not phase1_success:
                yield f"event: error\ndata: {json.dumps({'error': 'Failed to generate valid logic after 3 attempts'})}\n\n".encode(
                    "utf-8"
                )
                return

            # --- PHASE 2: GENERATE PRESENTATION ---
            presentation_payload: Dict[str, Any] = {}
            instruction = (
                "Tests passed. Now update the HTML page that uses these components. "
                "Return the `html` object and `metadata`."
            )

            phase2_system_prompt = await self._build_phase2_system_prompt(session)

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
            self._normalise_payload(payload_obj, scope, ui_id, name, prompt, existing)

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
        timestamp = self._now()
        existing.setdefault("metadata", {})
        metadata = existing["metadata"]
        self._ensure_version_metadata(metadata)
        metadata["updated_at"] = timestamp
        metadata["updated_by"] = actor.user_id
        metadata["version"] = self._current_version(metadata) + 1
        metadata["published_at"] = timestamp
        metadata["published_by"] = actor.user_id
        history = metadata.setdefault("history", [])
        payload_scripts = self._changed_scripts(
            payload_obj, existing.get("current", {})
        )
        history.append(
            self._history_entry(
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
        expanded_record["current"] = self._expand_payload(existing["current"])
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

        self._assert_scope_consistency(existing, scope, name)
        self._ensure_update_permissions(existing, scope, actor)

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

        timestamp = self._now()
        existing.setdefault("metadata", {})
        metadata = existing["metadata"]
        self._ensure_version_metadata(metadata)
        metadata["updated_at"] = timestamp
        metadata["updated_by"] = actor.user_id
        metadata["version"] = self._current_version(metadata) + 1
        metadata["published_at"] = timestamp
        metadata["published_by"] = actor.user_id
        history = metadata.setdefault("history", [])
        payload_scripts = self._changed_scripts(generated, existing.get("current", {}))
        history.append(
            self._history_entry(
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

    def _expand_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.output_factory.expand_payload(payload)

    def get_ui(
        self,
        *,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        expand: bool = False,
    ) -> Dict[str, Any]:
        try:
            existing = self.storage.read(scope, ui_id, name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Ui not found") from exc

        self._assert_scope_consistency(existing, scope, name)
        existing.setdefault("metadata", {})
        self._ensure_version_metadata(existing["metadata"])

        # Only expand payload when explicitly requested (for HTML rendering)
        if expand and "current" in existing:
            existing["current"] = self._expand_payload(existing["current"])

        return existing

    def reset_last_change(
        self,
        *,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
    ) -> Dict[str, Any]:
        """
        Reset the last change by removing the last history entry and
        restoring the previous state as current.

        Requires update permissions. Will fail if there is only one history entry.
        """
        try:
            existing = self.storage.read(scope, ui_id, name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Ui not found") from exc

        self._assert_scope_consistency(existing, scope, name)
        self._ensure_update_permissions(existing, scope, actor)

        metadata = existing.get("metadata", {})
        history = metadata.get("history", [])

        if len(history) <= 1:
            raise HTTPException(
                status_code=400,
                detail="Cannot reset: only one history entry exists (initial creation)",
            )

        # Remove the last history entry
        last_entry = history.pop()

        # Get the new last entry (which becomes current)
        new_last_entry = history[-1]

        # Reconstruct the current state from the new last entry
        new_current = {
            "html": new_last_entry.get("payload_html", {}),
            "metadata": new_last_entry.get("payload_metadata", {}),
        }
        scripts = self._scripts_from_history(history)
        if scripts:
            new_current.update(scripts)

        # Update timestamps
        timestamp = self._now()
        metadata["updated_at"] = timestamp
        metadata["updated_by"] = actor.user_id

        # Update the record
        existing["current"] = new_current
        existing["metadata"] = metadata

        # Persist the changes
        self.storage.write(scope, ui_id, name, existing)

        logger.info(
            f"Reset UI {ui_id}/{name} - removed history entry from {last_entry.get('generated_at')}"
        )

        return existing

    def create_draft_session(
        self,
        *,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        tools: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        try:
            existing = self.storage.read(scope, ui_id, name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Ui not found") from exc

        self._assert_scope_consistency(existing, scope, name)
        self._ensure_update_permissions(existing, scope, actor)
        existing.setdefault("metadata", {})
        self._ensure_version_metadata(existing["metadata"])
        self.storage.cleanup_expired_sessions(scope, ui_id, name)

        ttl_minutes = int(
            os.environ.get(
                "APP_UI_SESSION_TTL_MINUTES", str(APP_UI_SESSION_TTL_MINUTES)
            )
        )
        created_at = self._now()
        expires_at = datetime.now(timezone.utc).timestamp() + (ttl_minutes * 60)
        expires_iso = datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat()
        session_id = validate_identifier(uuid.uuid4().hex, "session id")
        base_version = self._current_version(existing["metadata"])

        payload = {
            "session_id": session_id,
            "editor_user_id": actor.user_id,
            "created_at": created_at,
            "updated_at": created_at,
            "expires_at": expires_iso,
            "base_version": base_version,
            "draft_version": 1,
            "draft_payload": copy.deepcopy(existing.get("current", {})),
            "messages": [],
            "last_tools": list(tools or []),
            "metadata_snapshot": copy.deepcopy(existing.get("metadata", {})),
        }

        self.storage.write_session(scope, ui_id, name, session_id, payload)
        return {
            "session_id": session_id,
            "base_version": base_version,
            "draft_version": 1,
            "expires_at": expires_iso,
        }

    def _session_is_expired(self, session_payload: Dict[str, Any]) -> bool:
        raw = session_payload.get("expires_at")
        if not raw:
            return False
        value = str(raw)
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        try:
            expires_dt = datetime.fromisoformat(value)
        except ValueError:
            return True
        if expires_dt.tzinfo is None:
            expires_dt = expires_dt.replace(tzinfo=timezone.utc)
        return expires_dt <= datetime.now(timezone.utc)

    def _load_session(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        session_id: str,
    ) -> Dict[str, Any]:
        self.storage.cleanup_expired_sessions(scope, ui_id, name)
        try:
            session_payload = self.storage.read_session(scope, ui_id, name, session_id)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail="Draft session not found"
            ) from exc
        if self._session_is_expired(session_payload):
            self.storage.delete_session(scope, ui_id, name, session_id)
            raise HTTPException(status_code=404, detail="Draft session expired")
        return session_payload

    def _assert_session_owner(
        self, session_payload: Dict[str, Any], actor: Actor
    ) -> None:
        if session_payload.get("editor_user_id") != actor.user_id:
            raise HTTPException(status_code=403, detail="Draft session access denied")

    def get_draft_ui(
        self,
        *,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        session_id: str,
        expand: bool = False,
    ) -> Dict[str, Any]:
        session_payload = self._load_session(
            scope=scope, ui_id=ui_id, name=name, session_id=session_id
        )
        self._assert_session_owner(session_payload, actor)

        metadata = copy.deepcopy(session_payload.get("metadata_snapshot", {}))
        metadata["draft_version"] = session_payload.get("draft_version", 1)
        metadata["session_id"] = session_payload.get("session_id")
        self._ensure_version_metadata(metadata)

        current = copy.deepcopy(session_payload.get("draft_payload", {}))
        if expand:
            current = self._expand_payload(current)

        return {"metadata": metadata, "current": current}

    def discard_draft_session(
        self,
        *,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        session_id: str,
    ) -> bool:
        session_payload = self._load_session(
            scope=scope, ui_id=ui_id, name=name, session_id=session_id
        )
        self._assert_session_owner(session_payload, actor)
        return self.storage.delete_session(scope, ui_id, name, session_id)

    def publish_draft_session(
        self,
        *,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        session_id: str,
    ) -> Dict[str, Any]:
        try:
            existing = self.storage.read(scope, ui_id, name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Ui not found") from exc

        session_payload = self._load_session(
            scope=scope, ui_id=ui_id, name=name, session_id=session_id
        )
        self._assert_session_owner(session_payload, actor)

        self._assert_scope_consistency(existing, scope, name)
        self._ensure_update_permissions(existing, scope, actor)

        existing.setdefault("metadata", {})
        metadata = existing["metadata"]
        self._ensure_version_metadata(metadata)
        current_version = self._current_version(metadata)
        base_version = int(session_payload.get("base_version") or 0)

        if base_version != current_version:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "publish_conflict",
                    "detail": "Draft base version is stale. Refresh draft session.",
                    "current_version": current_version,
                    "base_version": base_version,
                },
            )

        timestamp = self._now()
        draft_payload = copy.deepcopy(session_payload.get("draft_payload", {}))
        self._normalise_payload(
            draft_payload,
            scope,
            ui_id,
            name,
            "Publish conversational draft",
            existing,
        )
        payload_scripts = self._changed_scripts(
            draft_payload, existing.get("current", {})
        )

        metadata["updated_at"] = timestamp
        metadata["updated_by"] = actor.user_id
        metadata["version"] = current_version + 1
        metadata["published_at"] = timestamp
        metadata["published_by"] = actor.user_id
        history = metadata.setdefault("history", [])
        history.append(
            self._history_entry(
                action="publish",
                prompt="Publish conversational draft",
                tools=list(session_payload.get("last_tools") or []),
                user_id=actor.user_id,
                generated_at=timestamp,
                payload_metadata=draft_payload.get("metadata", {}),
                payload_html=draft_payload.get("html", {}),
                payload_scripts=payload_scripts or None,
            )
        )

        existing["current"] = draft_payload
        self.storage.write(scope, ui_id, name, existing)
        self.storage.delete_session(scope, ui_id, name, session_id)
        return existing

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
        access_token: Optional[str],
    ) -> AsyncIterator[bytes]:
        if not message.strip():
            yield _sse_event("error", {"error": "message must not be empty"})
            return

        try:
            session_payload = self._load_session(
                scope=scope, ui_id=ui_id, name=name, session_id=session_id
            )
            self._assert_session_owner(session_payload, actor)

            requested_tools = (
                list(tools)
                if tools is not None
                else list(session_payload.get("last_tools") or [])
            )
            selected_tools = await self._select_tools(session, requested_tools, message)
            selected_tool_names = [
                t.get("function", {}).get("name")
                for t in (selected_tools or [])
                if isinstance(t, dict)
            ]
            selected_tool_names = [name for name in selected_tool_names if name]

            draft_payload = copy.deepcopy(session_payload.get("draft_payload", {}))
            assistant_text = await self._run_assistant_message(
                session=session,
                draft_payload=draft_payload,
                history=session_payload.get("messages", []),
                user_message=message,
                selected_tools=selected_tools,
                tool_choice=tool_choice,
                access_token=access_token,
            )
            if assistant_text:
                yield _sse_event("assistant", {"delta": assistant_text})

            updated_payload: Optional[Dict[str, Any]] = None
            update_mode = "regenerated_fallback"
            patch_error: Optional[str] = None

            patch_enabled = (
                os.environ.get(
                    "APP_UI_PATCH_ENABLED", "true" if APP_UI_PATCH_ENABLED else "false"
                ).lower()
                == "true"
            )
            if patch_enabled:
                patch_attempt = await self._attempt_patch_update(
                    scope=scope,
                    ui_id=ui_id,
                    name=name,
                    draft_payload=draft_payload,
                    user_message=message,
                    assistant_message=assistant_text,
                    access_token=access_token,
                    previous_metadata=session_payload.get("metadata_snapshot", {}),
                )
                if patch_attempt:
                    updated_payload = patch_attempt.get("payload")
                    update_mode = "patch_applied"
                else:
                    patch_error = "Patch validation failed, using full regenerate"

            if updated_payload is None:
                previous = {
                    "metadata": copy.deepcopy(
                        session_payload.get("metadata_snapshot", {}) or {}
                    ),
                    "current": draft_payload,
                }
                regenerate_prompt = self._compose_regeneration_prompt(
                    user_message=message,
                    assistant_message=assistant_text,
                    history=session_payload.get("messages", []),
                )
                updated_payload = await self._generate_ui_payload(
                    session=session,
                    scope=scope,
                    ui_id=ui_id,
                    name=name,
                    prompt=regenerate_prompt,
                    tools=selected_tool_names,
                    access_token=access_token,
                    previous=previous,
                )

            messages_history = list(session_payload.get("messages") or [])
            messages_history.append({"role": "user", "content": message})
            messages_history.append({"role": "assistant", "content": assistant_text})

            session_payload["messages"] = messages_history
            session_payload["draft_payload"] = updated_payload
            session_payload["last_tools"] = selected_tool_names
            session_payload["updated_at"] = self._now()
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
            yield _sse_event("ui_updated", ui_event_payload)
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

    def _compose_regeneration_prompt(
        self,
        *,
        user_message: str,
        assistant_message: str,
        history: Sequence[Dict[str, Any]],
    ) -> str:
        tail = list(history or [])[-6:]
        return (
            "Update the existing UI using the conversational request.\n\n"
            f"User request:\n{user_message}\n\n"
            f"Assistant analysis:\n{assistant_message}\n\n"
            "Recent conversation history (JSON):\n"
            f"{json.dumps(tail, ensure_ascii=False)}"
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
        access_token: Optional[str],
    ) -> str:
        system_prompt = (
            "You are an assistant helping a user iteratively edit a generated web UI. "
            "Keep answers concise and implementation-focused. If tools are available, "
            "use them to gather facts before proposing UI changes."
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
        for item in history or []:
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
        return self._assistant_text_from_response(response)

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
    ) -> Optional[Dict[str, Any]]:
        try:
            system_prompt = (
                "You are a UI patch planner. Return valid JSON only in this shape: "
                '{"patch":{"html":{"page":"...","snippet":"..."},"service_script":"...","components_script":"...","metadata":{...}}}. '
                "Only include fields that need changes. Do not include markdown fences."
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
                extra_headers=UI_MODEL_HEADERS,
            )

            response = await self.tgi_service.llm_client.non_stream_completion(
                request, access_token or "", None
            )
            content = self._assistant_text_from_response(response)
            if not content:
                return None

            parsed = self._parse_json(content)
            patch = parsed.get("patch")
            if not isinstance(patch, dict):
                return None

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
            self._normalise_payload(
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
                success, _ = self._run_tests(
                    service_script,
                    components_script,
                    test_script,
                    candidate.get("dummy_data"),
                )
                if not success:
                    return None

            return {"payload": candidate}
        except Exception:
            return None

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
    ) -> Dict[str, Any]:
        system_prompt = await self._build_system_prompt(session)
        message_payload = {
            "ui": {
                "id": ui_id,
                "name": name,
                "scope": {"type": scope.kind, "id": scope.identifier},
            },
            "request": {
                "prompt": prompt,
                "tools": tools,
            },
        }

        if previous:
            previous_metadata = previous.get("metadata", {})
            message_payload["context"] = {
                "original_prompt": self._initial_prompt(previous_metadata),
                "history": self._history_for_prompt(
                    previous_metadata.get("history", [])
                ),
                "current_state": previous.get("current", {}),
            }

        messages = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(
                role=MessageRole.USER,
                content=json.dumps(message_payload, ensure_ascii=False),
            ),
        ]

        allowed_tools = await self._select_tools(session, tools, prompt)

        dummy_data = await self._generate_dummy_data(
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
                    "Dummy data module for tests is available as ./dummy_data.js. "
                    "Tests MUST import { dummyData } from './dummy_data.js' and use "
                    "dummyData[toolName] as the mocked response.json(). "
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
        self._maybe_dump_chat_request(
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

        payload = self._parse_json(content)
        payload["dummy_data"] = payload.get("dummy_data") or dummy_data
        self._normalise_payload(payload, scope, ui_id, name, prompt, previous)
        return payload

    async def _build_system_prompt(self, session: MCPSessionBase) -> str:
        prompt_service = self.tgi_service.prompt_service
        design_prompt_content = ""
        try:
            design_prompt = await prompt_service.find_prompt_by_name_or_role(
                session, prompt_name="design-system"
            )
            if design_prompt:
                design_prompt_content = await prompt_service.get_prompt_content(
                    session, design_prompt
                )
        except Exception:
            design_prompt_content = ""

        combined_design = design_prompt_content or DEFAULT_DESIGN_PROMPT

        # Load the pfusch prompt from file and replace the design system placeholder
        pfusch_prompt = _load_pfusch_prompt()
        return pfusch_prompt.replace("{{DESIGN_SYSTEM_PROMPT}}", combined_design)

    async def _build_phase2_system_prompt(self, session: MCPSessionBase) -> str:
        prompt_service = self.tgi_service.prompt_service
        design_prompt_content = ""
        try:
            design_prompt = await prompt_service.find_prompt_by_name_or_role(
                session, prompt_name="design-system"
            )
            if design_prompt:
                design_prompt_content = await prompt_service.get_prompt_content(
                    session, design_prompt
                )
        except Exception:
            design_prompt_content = ""

        combined_design = design_prompt_content or DEFAULT_DESIGN_PROMPT

        # Load the pfusch phase 2 prompt from file and replace the design system placeholder
        pfusch_prompt = _load_pfusch_phase2_prompt()
        return pfusch_prompt.replace("{{DESIGN_SYSTEM_PROMPT}}", combined_design)

    async def _select_tools(
        self, session: MCPSessionBase, requested_tools: Sequence[str], prompt: str = ""
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Select relevant tools for ui generation.

        If specific tools are requested, use those. Otherwise, intelligently
        filter tools based on the prompt to reduce context size.
        """
        # Get all tools with output schema for UI generation
        available = await self.tgi_service.tool_service.get_all_mcp_tools(
            session, include_output_schema=True
        )

        if not available:
            return None

        # If specific tools requested, filter to those
        if requested_tools:
            selected: List[Dict[str, Any]] = []
            for tool in available:
                tool_name: Optional[str] = None
                if isinstance(tool, dict):
                    tool_name = tool.get("function", {}).get("name")
                else:
                    tool_name = getattr(tool, "function", None)
                    if tool_name and hasattr(tool_name, "name"):
                        tool_name = tool_name.name
                if tool_name and tool_name in requested_tools:
                    selected.append(tool)
            return selected if selected else None

        # Otherwise, intelligently pre-select most relevant tools
        return self._filter_relevant_tools(available, prompt)

    def _filter_relevant_tools(
        self, tools: List[Dict[str, Any]], prompt: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Filter tools to most relevant ones based on prompt keywords.

        Uses simple keyword matching to reduce context size. If the prompt
        mentions specific domain terms, prioritize tools with those terms.
        """
        if not prompt or len(tools) <= 10:
            # If prompt is empty or tool count is manageable, return all
            return tools

        prompt_lower = prompt.lower()
        scored_tools = []

        for tool in tools:
            if not isinstance(tool, dict):
                continue

            function = tool.get("function", {})
            name = function.get("name", "")
            description = function.get("description", "")

            # Skip the meta tool "describe_tool"
            if name == "describe_tool":
                scored_tools.append((tool, 100))  # Always include
                continue

            # Score based on keyword matches
            score = 0

            # Check if tool name appears in prompt
            if name.lower() in prompt_lower:
                score += 50

            # Check for partial name matches (e.g., "absence" matches "list_absence_types")
            name_parts = name.lower().replace("_", " ").split()
            for part in name_parts:
                if len(part) > 3 and part in prompt_lower:
                    score += 10

            # Check if description keywords appear in prompt
            desc_words = description.lower().replace("_", " ").split()
            for word in desc_words:
                if len(word) > 4 and word in prompt_lower:
                    score += 5

            # Prioritize list/get operations for uis
            if any(
                prefix in name.lower()
                for prefix in ["list_", "get_", "fetch_", "retrieve_"]
            ):
                score += 3

            # Deprioritize create/update/delete operations unless explicitly mentioned
            if any(
                prefix in name.lower()
                for prefix in ["create_", "update_", "delete_", "remove_"]
            ):
                if not any(
                    word in prompt_lower
                    for word in [
                        "create",
                        "update",
                        "delete",
                        "edit",
                        "modify",
                        "remove",
                    ]
                ):
                    score -= 10

            scored_tools.append((tool, score))

        # Sort by score descending
        scored_tools.sort(key=lambda x: x[1], reverse=True)

        # Take top tools, ensuring we include describe_tool
        max_tools = 15  # Reasonable limit for context size
        selected = [tool for tool, score in scored_tools[:max_tools] if score > 0]

        # If we filtered too aggressively, include some more
        if len(selected) < 5 and len(scored_tools) > len(selected):
            selected = [tool for tool, score in scored_tools[:10]]

        logger.info(
            f"[GeneratedUI] Filtered {len(tools)} tools to {len(selected)} based on prompt relevance"
        )

        return selected if selected else tools

    def _extract_content(self, response: Any) -> str:
        if response is None:
            raise HTTPException(status_code=502, detail="Generation response was empty")

        if isinstance(response, dict):
            content = response.get("content")
            if isinstance(content, str):
                return content

        choices = getattr(response, "choices", None)
        if choices:
            first = choices[0]
            message = getattr(first, "message", None)
            if message and getattr(message, "content", None):
                return message.content
            delta = getattr(first, "delta", None)
            if delta and getattr(delta, "content", None):
                return delta.content

        raise HTTPException(
            status_code=502,
            detail="Unable to extract content from generation response",
        )

    def _parse_json(self, payload_str: str) -> Dict[str, Any]:
        try:
            return json.loads(payload_str)
        except json.JSONDecodeError:
            candidate = self._extract_json_block(payload_str)
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise HTTPException(
                        status_code=502,
                        detail="Generated content is not valid JSON",
                    ) from exc
            raise HTTPException(
                status_code=502,
                detail="Generated content is not valid JSON",
            )

    def _extract_json_block(self, text: str) -> Optional[str]:
        start = None
        depth = 0
        in_string = False
        escape = False
        for idx, char in enumerate(text):
            if start is None:
                if char == "{":
                    start = idx
                    depth = 1
                continue

            if escape:
                escape = False
                continue

            if char == "\\":
                escape = True
                continue

            if char == '"' and not escape:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : idx + 1]
        return None

    def _normalise_payload(
        self,
        payload: Dict[str, Any],
        scope: Scope,
        ui_id: str,
        name: str,
        prompt: str,
        previous: Optional[Dict[str, Any]],
    ) -> None:
        self.output_factory.normalise_payload(
            payload=payload,
            scope=scope,
            ui_id=ui_id,
            name=name,
            prompt=prompt,
            previous=previous,
        )

    def _maybe_dump_chat_request(
        self,
        *,
        chat_request: ChatCompletionRequest,
        scope: Scope,
        ui_id: str,
        name: str,
        prompt: str,
        tools: List[str],
        message_payload: Dict[str, Any],
    ) -> None:
        dump_target = GENERATED_UI_PROMPT_DUMP
        if not dump_target:
            return
        try:
            path = Path(dump_target)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

            def _safe(value: str) -> str:
                return re.sub(r"[^A-Za-z0-9_-]", "_", value or "")

            file_name = (
                f"{timestamp}_{_safe(scope.kind)}-{_safe(scope.identifier)}_"
                f"{_safe(ui_id)}_{_safe(name)}.json"
            )

            if path.exists() and path.is_dir():
                target_path = path / file_name
            elif path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
                target_path = path
            else:
                path.mkdir(parents=True, exist_ok=True)
                target_path = path / file_name

            payload = {
                "timestamp": timestamp,
                "scope": {"type": scope.kind, "id": scope.identifier},
                "ui_id": ui_id,
                "name": name,
                "prompt": prompt,
                "requested_tools": tools,
                "message_payload": message_payload,
                "chat_request": chat_request.model_dump(exclude_none=True),
            }

            with target_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)

            logger.info("[GeneratedUI] Chat request preview dumped to %s", target_path)
        except Exception as exc:  # pragma: no cover - debug helper
            logger.error(
                "[GeneratedUI] Failed to dump chat request preview: %s",
                exc,
                exc_info=exc,
            )

    def _extract_body(self, html: str) -> Optional[str]:
        return self.output_factory.extract_body(html)

    def _wrap_snippet(self, snippet: str) -> str:
        return self.output_factory.wrap_snippet(snippet)

    def _assert_scope_consistency(
        self, existing: Dict[str, Any], scope: Scope, name: str
    ) -> None:
        metadata = existing.get("metadata", {})
        stored_scope = metadata.get("scope", {})
        if (
            stored_scope.get("type") != scope.kind
            or stored_scope.get("id") != scope.identifier
        ):
            raise HTTPException(
                status_code=403,
                detail="Scope mismatch for stored ui",
            )
        if metadata.get("name") and metadata.get("name") != name:
            raise HTTPException(status_code=403, detail="Ui name mismatch")

    def _ensure_update_permissions(
        self, existing: Dict[str, Any], scope: Scope, actor: Actor
    ) -> None:
        if not actor.is_owner(scope):
            raise HTTPException(status_code=403, detail="Access denied for update")

    def _initial_prompt(self, metadata: Dict[str, Any]) -> Optional[str]:
        history = metadata.get("history") or []
        if history:
            first = history[0]
            if isinstance(first, dict):
                return first.get("prompt")
        return None

    def _changed_scripts(
        self,
        new_payload: Dict[str, Any],
        previous_payload: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        changed: Dict[str, Any] = {}
        previous_payload = previous_payload or {}
        for key in SCRIPT_KEYS:
            if key not in new_payload and key not in previous_payload:
                continue
            new_value = new_payload.get(key)
            old_value = previous_payload.get(key)
            if new_value != old_value:
                changed[key] = new_value
        return changed

    def _scripts_from_history(self, history_entries: Sequence[Any]) -> Dict[str, Any]:
        scripts: Dict[str, Any] = {}
        for entry in reversed(history_entries or []):
            if not isinstance(entry, dict):
                continue
            payload_scripts = entry.get("payload_scripts")
            if not isinstance(payload_scripts, dict):
                continue
            for key in SCRIPT_KEYS:
                if key in payload_scripts and key not in scripts:
                    scripts[key] = payload_scripts[key]
            if len(scripts) == len(SCRIPT_KEYS):
                break
        return scripts

    def _history_entry(
        self,
        *,
        action: str,
        prompt: str,
        tools: List[str],
        user_id: str,
        generated_at: str,
        payload_metadata: Dict[str, Any],
        payload_html: Optional[Dict[str, Any]] = None,
        payload_scripts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        entry = {
            "action": action,
            "prompt": prompt,
            "tools": tools,
            "user_id": user_id,
            "generated_at": generated_at,
            "payload_metadata": payload_metadata,
            "payload_html": payload_html,
        }
        if payload_scripts is not None:
            entry["payload_scripts"] = payload_scripts
        return entry

    def _history_for_prompt(
        self, history_entries: Sequence[Any]
    ) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for entry in history_entries or []:
            if not isinstance(entry, dict):
                continue
            entry_copy = dict(entry)
            entry_copy.pop("payload_html", None)
            entry_copy.pop("payload_scripts", None)
            sanitized.append(entry_copy)
        return sanitized

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
