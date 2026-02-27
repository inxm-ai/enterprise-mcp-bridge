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
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from fastapi import HTTPException

from app.session import MCPSessionBase
from app.vars import (
    MCP_BASE_PATH,
    GENERATED_UI_PROMPT_DUMP,
    APP_UI_SESSION_TTL_MINUTES,
    GENERATED_UI_FIX_CODE_FIRST,
    GENERATED_UI_INCLUDE_OUTPUT_SCHEMA,
    GENERATED_UI_MAX_TOOLS,
)
from app.tgi.models import (
    ChatCompletionRequest,
    Message,
    MessageRole,
)  # noqa: F401 – MessageRole re-exported
from app.tgi.services.proxied_tgi_service import ProxiedTGIService
from app.app_facade.generated_output_factory import (
    GeneratedUIOutputFactory,
    MCP_SERVICE_TEST_HELPER_SCRIPT,
)
from app.app_facade.generated_dummy_data import (
    DummyDataGenerator,
)
from app.app_facade.generated_schemas import (
    generation_response_format,
)
from app.app_facade.generated_storage import GeneratedUIStorage
from app.app_facade.generated_types import (
    Actor,
    Scope,  # re-exported for route/tests import compatibility
    validate_identifier,  # noqa: F401  # re-exported for route/tests import compatibility
)
from app.app_facade.test_fix_tools import _parse_tap_output, run_tool_driven_test_fix
from app.app_facade.tool_sampling import ToolSampler
from app.app_facade.test_runner_service import TestRunnerService
from app.app_facade.generation_pipeline import GenerationPipeline
from app.app_facade.conversational_service import ConversationalService
from app.app_facade.gateway_explorer import GatewayExplorer
from app.app_facade.prompt_helpers import (
    history_entry,
    changed_scripts,
    scripts_from_history,
)


logger = logging.getLogger("uvicorn.error")

# Error codes / substrings that identify non-retryable LLM API failures.
# When one of these appears in a phase failure reason we should stop retrying
# immediately – hammering the API will not help and only wastes quota.
_FATAL_LLM_ERROR_SUBSTRINGS = (
    "insufficient_quota",
    "invalid_api_key",
    "authentication_error",
    "permission_denied",
)


def _is_fatal_llm_error(reason: str) -> bool:
    """Return True when *reason* indicates a non-retryable LLM API error."""
    lower = reason.lower()
    return any(sub in lower for sub in _FATAL_LLM_ERROR_SUBSTRINGS)


DEFAULT_DESIGN_PROMPT = (
    "Use lightweight, responsive layouts. Prefer utility-first styling via Tailwind "
    "CSS conventions when no explicit design system guidance is provided."
)

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


def _positive_int_env(name: str, default: int) -> int:
    try:
        parsed = int(os.environ.get(name, str(default)))
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


NODE_TEST_TIMEOUT_MS = _positive_int_env("GENERATED_UI_NODE_TEST_TIMEOUT_MS", 8000)


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
        self.gateway_explorer = GatewayExplorer(tgi_service=self.tgi_service)
        self.tool_sampler = ToolSampler(
            tgi_service=self.tgi_service,
            dummy_data_generator=self.dummy_data_generator,
        )
        self.test_runner = TestRunnerService(service=self)
        self.generation_pipeline = GenerationPipeline(service=self)
        self.conversational_service = ConversationalService(service=self)
        self._last_patch_failure_reason: Optional[str] = None

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
        test_name: Optional[str] = None,
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
            resolved_service_code = service_code or ""
            # Always inject the global McpService/service safety prelude in tests.
            # It writes only to globalThis and avoids top-level symbol collisions.
            service_prelude = MCP_SERVICE_TEST_HELPER_SCRIPT
            combined_code = f"{service_prelude}\n\n{resolved_service_code}\n\n{mocked_components_code}"

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
                cmd = [
                    "node",
                    "--test",
                    "--test-force-exit",
                    "--test-timeout",
                    str(NODE_TEST_TIMEOUT_MS),
                    "test.js",
                ]
                if test_name:
                    cmd.extend(["--test-name-pattern", test_name])
                result = subprocess.run(
                    cmd,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=30,  # Safety timeout
                    env=env,
                )
                output = result.stdout + "\n" + result.stderr
                if result.returncode != 0:
                    passed, failed, failed_tests = _parse_tap_output(output)
                    output_excerpt = self.test_runner._trim_output(output)
                    failed_preview = ", ".join(failed_tests[:5]) if failed_tests else ""
                    if len(failed_tests) > 5:
                        failed_preview += f", ... ({len(failed_tests) - 5} more)"
                    logger.error(
                        "[GeneratedUI] Test run failed (passed=%s, failed=%s). "
                        "Failing tests: %s\nOutput excerpt:\n%s",
                        passed,
                        failed,
                        failed_preview or "unknown",
                        output_excerpt,
                    )
                return result.returncode == 0, output
            except subprocess.TimeoutExpired:
                return False, "Tests timed out after 30 seconds."
            except Exception as e:
                return False, f"Error running tests: {str(e)}"

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
        post_success_validator: Optional[
            Callable[[str, str, str, Optional[str]], Tuple[bool, Optional[str]]]
        ] = None,
    ) -> Tuple[bool, str, str, str, Optional[str], List[Message]]:
        if not GENERATED_UI_FIX_CODE_FIRST:
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
                post_success_validator=post_success_validator,
            )

        def _score_candidate(
            svc: str, comps: str, tests: str, fixtures: Optional[str]
        ) -> Tuple[bool, int, int]:
            try:
                ok, output = self._run_tests(svc, comps, tests, fixtures)
            except Exception:
                return False, -1, 10**9
            passed, failed, _ = _parse_tap_output(output or "")
            if not ok:
                return False, passed, failed
            if not post_success_validator:
                return True, passed, failed
            try:
                validation_ok, validation_reason = post_success_validator(
                    svc, comps, tests, fixtures
                )
            except Exception as exc:
                logger.error(
                    "[iterative_test_fix] Post-success validator errored while scoring candidate: %s",
                    exc,
                    exc_info=exc,
                )
                return False, passed, max(1, failed)
            if validation_ok:
                return True, passed, failed
            logger.info(
                "[iterative_test_fix] Candidate rejected by post-success validator: %s",
                (validation_reason or "post_success_validation_failed").strip(),
            )
            return False, passed, max(1, failed)

        test_source = test_script or ""
        likely_pfusch_collection_mismatch = bool(
            re.search(r"\bpfuschTest\s*\(", test_source)
            and re.search(r"\bcomp\.(?:state|shadowRoot)\b", test_source)
        )

        stage_a_attempts = min(max_attempts, 12)
        if likely_pfusch_collection_mismatch:
            stage_a_attempts = min(max_attempts, 6)
            logger.info(
                "[iterative_test_fix] Detected pfuschTest collection access via comp.state/comp.shadowRoot; "
                "reducing code-first attempts before adjust-test fallback"
            )
        stage_b_attempts = min(8, max(0, max_attempts - stage_a_attempts))

        best = {
            "service_script": service_script,
            "components_script": components_script,
            "test_script": test_script,
            "dummy_data": dummy_data,
            "messages": list(messages),
            "ok": False,
            "passed": -1,
            "failed": 10**9,
        }

        initial_ok, initial_passed, initial_failed = _score_candidate(
            service_script, components_script, test_script, dummy_data
        )
        best["ok"] = initial_ok
        best["passed"] = initial_passed
        best["failed"] = initial_failed

        logger.info(
            "[iterative_test_fix] Starting code-first strategy (stage_a=%s, stage_b=%s)",
            stage_a_attempts,
            stage_b_attempts,
        )

        stage_a = await run_tool_driven_test_fix(
            tgi_service=self.tgi_service,
            service_script=service_script,
            components_script=components_script,
            test_script=test_script,
            dummy_data=dummy_data,
            messages=messages,
            allowed_tools=allowed_tools,
            access_token=access_token,
            max_attempts=stage_a_attempts,
            event_queue=event_queue,
            extra_headers=UI_MODEL_HEADERS,
            strategy_mode="fix_code",
            post_success_validator=post_success_validator,
        )
        (
            stage_a_success,
            stage_a_service,
            stage_a_components,
            stage_a_test,
            stage_a_dummy_data,
            stage_a_messages,
        ) = stage_a

        stage_a_ok, stage_a_passed, stage_a_failed = _score_candidate(
            stage_a_service, stage_a_components, stage_a_test, stage_a_dummy_data
        )
        if (
            stage_a_passed > best["passed"]
            or (stage_a_passed == best["passed"] and stage_a_failed < best["failed"])
            or (stage_a_ok and not best["ok"])
        ):
            best = {
                "service_script": stage_a_service,
                "components_script": stage_a_components,
                "test_script": stage_a_test,
                "dummy_data": stage_a_dummy_data,
                "messages": stage_a_messages,
                "ok": stage_a_ok,
                "passed": stage_a_passed,
                "failed": stage_a_failed,
            }

        if stage_a_success and stage_a_ok:
            logger.info("[iterative_test_fix] Code-first stage succeeded")
            return stage_a
        if stage_a_success and not stage_a_ok:
            logger.warning(
                "[iterative_test_fix] Code-first stage reported success but failed post-success validation; continuing to fallback stage"
            )

        if stage_b_attempts <= 0:
            logger.warning(
                "[iterative_test_fix] Code-first stage failed and no fallback attempts available; returning best snapshot"
            )
            return (
                bool(best["ok"]),
                str(best["service_script"]),
                str(best["components_script"]),
                str(best["test_script"]),
                best["dummy_data"],
                list(best["messages"]),
            )

        logger.info(
            "[iterative_test_fix] Code-first stage failed, starting fallback adjust-test stage"
        )
        stage_b = await run_tool_driven_test_fix(
            tgi_service=self.tgi_service,
            service_script=stage_a_service,
            components_script=stage_a_components,
            test_script=stage_a_test,
            dummy_data=stage_a_dummy_data,
            messages=stage_a_messages,
            allowed_tools=allowed_tools,
            access_token=access_token,
            max_attempts=stage_b_attempts,
            event_queue=event_queue,
            extra_headers=UI_MODEL_HEADERS,
            strategy_mode="adjust_test",
            post_success_validator=post_success_validator,
        )
        (
            stage_b_success,
            stage_b_service,
            stage_b_components,
            stage_b_test,
            stage_b_dummy_data,
            stage_b_messages,
        ) = stage_b

        stage_b_ok, stage_b_passed, stage_b_failed = _score_candidate(
            stage_b_service, stage_b_components, stage_b_test, stage_b_dummy_data
        )
        if (
            stage_b_passed > best["passed"]
            or (stage_b_passed == best["passed"] and stage_b_failed < best["failed"])
            or (stage_b_ok and not best["ok"])
        ):
            best = {
                "service_script": stage_b_service,
                "components_script": stage_b_components,
                "test_script": stage_b_test,
                "dummy_data": stage_b_dummy_data,
                "messages": stage_b_messages,
                "ok": stage_b_ok,
                "passed": stage_b_passed,
                "failed": stage_b_failed,
            }

        if stage_b_success and stage_b_ok:
            logger.info("[iterative_test_fix] Fallback adjust-test stage succeeded")
            return stage_b
        if stage_b_success and not stage_b_ok:
            logger.warning(
                "[iterative_test_fix] Fallback adjust-test stage reported success but failed post-success validation"
            )

        logger.warning(
            "[iterative_test_fix] All fix stages failed. Returning best snapshot (passed=%s, failed=%s)",
            best["passed"],
            best["failed"],
        )
        return (
            bool(best["ok"]),
            str(best["service_script"]),
            str(best["components_script"]),
            str(best["test_script"]),
            best["dummy_data"],
            list(best["messages"]),
        )

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
        scripts = scripts_from_history(history)
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
            "test_state": "idle",
            "test_run_id": "",
            "test_summary": {
                "passed": 0,
                "failed": 0,
                "failed_tests": [],
                "message": "No tests run yet",
                "trigger": "none",
                "started_at": None,
                "completed_at": None,
            },
            "test_events": [],
            "last_test_output_tail": "",
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

    def _ensure_test_session_fields(self, session_payload: Dict[str, Any]) -> bool:
        changed = False
        if "test_state" not in session_payload:
            session_payload["test_state"] = "idle"
            changed = True
        if "test_run_id" not in session_payload:
            session_payload["test_run_id"] = ""
            changed = True
        if "last_test_output_tail" not in session_payload:
            session_payload["last_test_output_tail"] = ""
            changed = True
        if "test_events" not in session_payload or not isinstance(
            session_payload.get("test_events"), list
        ):
            session_payload["test_events"] = []
            changed = True

        summary = session_payload.get("test_summary")
        if not isinstance(summary, dict):
            summary = {}
            changed = True
        defaults = {
            "passed": 0,
            "failed": 0,
            "failed_tests": [],
            "message": "No tests run yet",
            "trigger": "none",
            "started_at": None,
            "completed_at": None,
        }
        for key, value in defaults.items():
            if key not in summary:
                summary[key] = value
                changed = True
        session_payload["test_summary"] = summary
        return changed

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
        if self._ensure_test_session_fields(session_payload):
            self.storage.write_session(scope, ui_id, name, session_id, session_payload)
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
        payload_scripts = changed_scripts(draft_payload, existing.get("current", {}))

        metadata["updated_at"] = timestamp
        metadata["updated_by"] = actor.user_id
        metadata["version"] = current_version + 1
        metadata["published_at"] = timestamp
        metadata["published_by"] = actor.user_id
        history = metadata.setdefault("history", [])
        history.append(
            history_entry(
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

    def _rewrite_gateway_tool_calls_in_script(
        self,
        script: str,
        *,
        gateway_routes: Dict[str, str],
    ) -> Tuple[str, int]:
        """Replace literal tool-name strings in ``svc.call`` / ``mcp.call`` with gateway route paths.

        Only rewrites calls where the first argument is a *string literal*
        that matches a key in *gateway_routes*.  Dynamic / variable-based calls
        are left untouched.

        Returns ``(rewritten_script, replacement_count)``.
        """
        replacements = 0

        def _replace(m: re.Match) -> str:
            nonlocal replacements
            prefix = m.group(1)   # e.g. "svc.call(" or "mcp.call("
            quote = m.group(2)    # the quote character (' or ")
            tool = m.group(3)     # the tool name
            suffix = m.group(4)   # rest after closing quote
            route = gateway_routes.get(tool)
            if route is None:
                return m.group(0)
            replacements += 1
            return f"{prefix}{quote}{route}{quote}{suffix}"

        pattern = r"""((?:svc|mcp)\.call\()(['"])([\w./-]+)\2([\s,)])"""
        rewritten = re.sub(pattern, _replace, script)
        return rewritten, replacements

    async def _build_system_prompt(self, session: MCPSessionBase) -> str:
        """Build the legacy all-in-one generation prompt used outside phase 1/2 flow."""
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
            session, include_output_schema=GENERATED_UI_INCLUDE_OUTPUT_SCHEMA
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
            safe_file_name = Path(file_name).name

            if path.exists() and path.is_dir():
                base_dir = path.resolve()
                target_path = (base_dir / safe_file_name).resolve()
                if target_path.parent != base_dir:
                    raise ValueError("Invalid dump path")
            elif path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
                target_path = path.resolve()
            else:
                path.mkdir(parents=True, exist_ok=True)
                base_dir = path.resolve()
                target_path = (base_dir / safe_file_name).resolve()
                if target_path.parent != base_dir:
                    raise ValueError("Invalid dump path")

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

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
