import asyncio
import difflib
import json
import logging
import re
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from fastapi import HTTPException

from app.tgi.models import ChatCompletionRequest, Message, MessageRole

from app.app_facade.generated_schemas import (
    generation_logic_schema,
    generation_response_format,
)


logger = logging.getLogger("uvicorn.error")


def _trim_text(value: Optional[str], max_len: int = 1200) -> str:
    text = (value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}...(trimmed {len(text) - max_len} chars)"


_TIMEOUT_RISK_PATTERNS = (
    (re.compile(r"await\s+new\s+Promise\s*\(", re.IGNORECASE), "await_new_promise"),
    (re.compile(r"while\s*\(\s*true\s*\)", re.IGNORECASE), "while_true_loop"),
    (re.compile(r"for\s*\(\s*;\s*;\s*\)", re.IGNORECASE), "for_ever_loop"),
)


def _detect_timeout_risks(
    test_script: Optional[str], components_script: Optional[str]
) -> List[str]:
    risks: List[str] = []
    test_text = test_script or ""
    components_text = components_script or ""

    for pattern, label in _TIMEOUT_RISK_PATTERNS:
        if pattern.search(test_text):
            risks.append(f"test_script:{label}")

    if re.search(r"setInterval\s*\(", test_text, re.IGNORECASE):
        risks.append("test_script:setInterval_usage")

    if re.search(r"setInterval\s*\(", components_text, re.IGNORECASE) and not re.search(
        r"clearInterval\s*\(", components_text, re.IGNORECASE
    ):
        risks.append("components_script:setInterval_without_clearInterval")

    return risks


def _detect_quality_risks(
    test_script: Optional[str],
    components_script: Optional[str],
    service_script: Optional[str] = None,
    gateway_tool_names: Optional[Set[str]] = None,
) -> List[str]:
    risks: List[str] = []
    test_text = test_script or ""
    components_text = components_script or ""
    service_text = service_script or ""
    runtime_text = f"{service_text}\n{components_text}"

    # Common regression pattern in generated pfusch components:
    # state destructuring in component callback parameters can break when
    # framework invocation doesn't pass the expected object.
    if re.search(
        r"=\s*\(\s*\{\s*state\b[^)]*\)\s*=>", components_text, re.IGNORECASE
    ) or re.search(
        r"function\s+\w+\s*\(\s*\{\s*state\b[^)]*\)", components_text, re.IGNORECASE
    ):
        risks.append("components_script:destructured_state_callback_signature")

    uses_fetch = bool(re.search(r"\bfetch\s*\(", test_text, re.IGNORECASE))
    uses_tool_http_path = bool(
        re.search(r"/tools/|/api/.*/tools/", test_text, re.IGNORECASE)
    )
    uses_service_calls = bool(
        re.search(r"\b(?:service|svc)\s*\.\s*call\s*\(", test_text, re.IGNORECASE)
    )
    has_fixture_stubs = bool(
        re.search(
            r"dummyData|\.test\.addResponse\s*\(|(?:globalThis\.)?fetch\.addRoute\s*\(",
            test_text,
            re.IGNORECASE,
        )
    )
    if (uses_fetch or uses_tool_http_path) and not has_fixture_stubs:
        risks.append("test_script:direct_network_without_fixtures")
    if uses_service_calls and not has_fixture_stubs:
        risks.append("test_script:service_calls_without_fixture_stubs")

    # Reject explicit test-coupled hints that bypass runtime data loading.
    if re.search(
        r"don['’]t\s+fetch|tests?\s+will\s+provide\s+state\s+directly",
        f"{components_text}\n{test_text}",
        re.IGNORECASE,
    ):
        risks.append("components_script:test_coupled_no_fetch_hint")

    # Runtime scripts (service/components) must not depend on test-only fixture modules.
    if re.search(
        r"import\s+(?:[\s\S]*?\s+from\s+)?[\"']\.\/dummy_data\.js[\"']",
        runtime_text,
        re.IGNORECASE,
    ):
        risks.append("runtime_script:dummy_data_import")
    if re.search(r"\bdummyData\b", runtime_text):
        risks.append("runtime_script:dummy_data_reference")
    if re.search(r"\.test\.addResponse\s*\(", runtime_text, re.IGNORECASE):
        risks.append("runtime_script:test_mock_api_usage")

    gateway_names = {str(name or "").strip() for name in (gateway_tool_names or set())}
    gateway_names = {name for name in gateway_names if name}
    if gateway_names:
        plain_gateway_call = re.compile(
            r"\b[\w$.]+\s*\.\s*(?:call|callTool)\s*\(\s*(['\"])([A-Za-z_][A-Za-z0-9_-]*)\1"
        )
        for match in plain_gateway_call.finditer(runtime_text):
            called_name = str(match.group(2) or "").strip()
            if called_name in gateway_names:
                risks.append("runtime_script:gateway_direct_tool_name_call")
                break

    return risks


def _extract_dummy_data_payload(dummy_data: Optional[str]) -> Dict[str, Any]:
    text = (dummy_data or "").strip()
    if not text:
        return {}

    marker = "export const dummyData ="
    start = text.find(marker)
    if start < 0:
        return {}

    remainder = text[start + len(marker) :].lstrip()
    if not remainder.startswith("{"):
        return {}

    end_marker = "export const dummyDataSchemaHints"
    end = remainder.find(end_marker)
    candidate = remainder[:end].strip() if end >= 0 else remainder
    candidate = candidate.rstrip()
    if candidate.endswith(";"):
        candidate = candidate[:-1].rstrip()

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _collect_object_keys_recursive(value: Any) -> Set[str]:
    keys: Set[str] = set()
    if isinstance(value, dict):
        for item_key, item_value in value.items():
            keys.add(item_key)
            keys.update(_collect_object_keys_recursive(item_value))
        return keys
    if isinstance(value, list):
        for item in value:
            keys.update(_collect_object_keys_recursive(item))
    return keys


def _normalize_schema_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _build_dummy_data_key_index(dummy_payload: Dict[str, Any]) -> Dict[str, Set[str]]:
    index: Dict[str, Set[str]] = {}
    for tool_name, tool_payload in dummy_payload.items():
        if isinstance(tool_payload, (dict, list)):
            keys = _collect_object_keys_recursive(tool_payload)
            if keys:
                index[tool_name] = keys
    return index


def _find_schema_drift_candidate(
    field_name: str, valid_keys: Set[str]
) -> Optional[str]:
    if not field_name or not valid_keys or field_name in valid_keys:
        return None

    normalized_field = _normalize_schema_key(field_name)
    if not normalized_field:
        return None

    normalized_to_keys: Dict[str, List[str]] = {}
    for key in sorted(valid_keys):
        normalized_to_keys.setdefault(_normalize_schema_key(key), []).append(key)

    exact_normalized = normalized_to_keys.get(normalized_field)
    if exact_normalized:
        return exact_normalized[0]

    near_matches = difflib.get_close_matches(
        field_name, sorted(valid_keys), n=1, cutoff=0.74
    )
    if near_matches:
        return near_matches[0]

    near_normalized = difflib.get_close_matches(
        normalized_field,
        sorted(normalized_to_keys.keys()),
        n=1,
        cutoff=0.74,
    )
    if near_normalized:
        return normalized_to_keys[near_normalized[0]][0]

    return None


def _extract_js_property_chain_leafs(text: str, root_pattern: str) -> List[str]:
    leafs: List[str] = []
    chain_pattern = re.compile(
        rf"{root_pattern}((?:\s*\?*\.\s*[A-Za-z_]\w*)+)",
        re.IGNORECASE,
    )
    for match in chain_pattern.finditer(text):
        chain = match.group(1)
        segments = re.findall(r"[A-Za-z_]\w*", chain)
        if segments:
            leafs.append(segments[-1])
    return leafs


def _extract_mock_object_literal_keys(test_text: str) -> List[Tuple[str, str, str]]:
    extracted: List[Tuple[str, str, str]] = []

    add_resolved_pattern = re.compile(
        r"addResolved\s*\(\s*['\"]([A-Za-z_]\w*)['\"]\s*,\s*\{([\s\S]{0,1200}?)\}\s*\)",
        re.IGNORECASE,
    )
    fallback_pattern = re.compile(
        r"dummyData\.([A-Za-z_]\w*)\s*\?\?\s*\{([\s\S]{0,1200}?)\}",
        re.IGNORECASE,
    )
    key_pattern = re.compile(r"(?:^|[,{]\s*)(?:['\"])?([A-Za-z_]\w*)(?:['\"])?\s*:")

    for tool_name, object_body in add_resolved_pattern.findall(test_text):
        for key in key_pattern.findall(object_body):
            extracted.append((tool_name, key, "addResolved"))

    for tool_name, object_body in fallback_pattern.findall(test_text):
        for key in key_pattern.findall(object_body):
            extracted.append((tool_name, key, "fallback"))

    return extracted


def _detect_schema_contract_risks(
    *,
    test_script: Optional[str],
    components_script: Optional[str],
    dummy_data: Optional[str],
) -> List[str]:
    risks: List[str] = []
    test_text = test_script or ""
    components_text = components_script or ""
    dummy_payload = _extract_dummy_data_payload(dummy_data)

    key_index = _build_dummy_data_key_index(dummy_payload)
    all_dummy_keys: Set[str] = set()
    for keys in key_index.values():
        all_dummy_keys.update(keys)

    component_leafs = _extract_js_property_chain_leafs(
        components_text,
        r"state\s*\.\s*data",
    )
    for field_name in component_leafs:
        candidate = _find_schema_drift_candidate(field_name, all_dummy_keys)
        if candidate:
            risks.append("schema_field_drift:components:" f"{field_name}->{candidate}")

    test_dummy_refs_pattern = re.compile(
        r"dummyData\.([A-Za-z_]\w*)((?:\s*\?*\.\s*[A-Za-z_]\w*)+)",
        re.IGNORECASE,
    )
    for tool_name, chain in test_dummy_refs_pattern.findall(test_text):
        tool_keys = key_index.get(tool_name)
        if not tool_keys:
            continue
        segments = re.findall(r"[A-Za-z_]\w*", chain)
        if not segments:
            continue
        field_name = segments[-1]
        candidate = _find_schema_drift_candidate(field_name, tool_keys)
        if candidate:
            risks.append(
                "schema_field_drift:test_ref:" f"{tool_name}.{field_name}->{candidate}"
            )

    for tool_name, key_name, source in _extract_mock_object_literal_keys(test_text):
        tool_keys = key_index.get(tool_name)
        if not tool_keys:
            continue
        candidate = _find_schema_drift_candidate(key_name, tool_keys)
        if candidate:
            risks.append(
                "schema_field_drift:test_mock:"
                f"{source}:{tool_name}.{key_name}->{candidate}"
            )

    if re.search(
        r"assert\.ok\([^\n]*includes\(['\"][0-9]+['\"]\)[^\n]*\|\|[^\n]*includes\(['\"]°C['\"]\)",
        test_text,
        re.IGNORECASE,
    ):
        risks.append("weak_assertion:temperature_literal_or_unit_only")

    if re.search(
        r"assert\.ok\([^\n]*includes\(['\"]°C['\"]\)[^\n]*\|\|",
        test_text,
        re.IGNORECASE,
    ):
        risks.append("weak_assertion:unit_only_disjunction")

    hard_fail_schema_hint_patterns = (
        r"dummyDataSchemaHints[\s\S]{0,200}?throw\s+new\s+Error",
        r"dummyDataSchemaHints[\s\S]{0,200}?assert\.(?:fail|ok\s*\(\s*false)",
    )
    if any(
        re.search(pattern, test_text, re.IGNORECASE)
        for pattern in hard_fail_schema_hint_patterns
    ):
        risks.append("hard_fail_schema_hint_guard")

    return sorted(set(risks))


def _apply_hard_fail_schema_hint_autofix(test_script: str) -> Tuple[str, bool]:
    text = test_script
    changed = False
    block_pattern = re.compile(
        r"if\s*\(\s*dummyDataSchemaHints\?\.[A-Za-z_]\w*\s*\)\s*\{([\s\S]{0,260}?)\}",
        re.IGNORECASE,
    )
    failing_signal = re.compile(
        r"throw\s+new\s+Error|assert\.(?:fail|ok\s*\(\s*false)",
        re.IGNORECASE,
    )

    def _replace(match: re.Match[str]) -> str:
        nonlocal changed
        block_body = match.group(1)
        if not failing_signal.search(block_body):
            return match.group(0)
        changed = True
        condition = match.group(0).split("{", 1)[0].rstrip()
        return (
            f"{condition}{{\n"
            "    // schema hint is informational; avoid hard-failing this test\n"
            "  }"
        )

    return block_pattern.sub(_replace, text), changed


def _apply_schema_field_drift_autofix(
    test_script: str,
    schema_contract_risks: Sequence[str],
) -> Tuple[str, List[str]]:
    text = test_script
    notes: List[str] = []
    replacements: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    for risk in schema_contract_risks:
        marker = "schema_field_drift:"
        if not risk.startswith(marker) or "->" not in risk:
            continue
        before_after = risk.rsplit(":", 1)[-1]
        source, candidate = before_after.split("->", 1)
        if "." in source:
            field_name = source.rsplit(".", 1)[-1].strip()
        else:
            field_name = source.strip()
        replacement = candidate.strip()
        if not field_name or not replacement or field_name == replacement:
            continue
        pair = (field_name, replacement)
        if pair in seen:
            continue
        seen.add(pair)
        replacements.append(pair)

    for field_name, replacement in replacements:
        key_pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(field_name)}(?=\s*:)")
        prop_pattern = re.compile(rf"(?<=\.){re.escape(field_name)}\b")
        next_text = key_pattern.sub(replacement, text)
        next_text = prop_pattern.sub(replacement, next_text)
        if next_text != text:
            notes.append(f"{field_name}->{replacement}")
            text = next_text

    return text, notes


def _apply_schema_contract_autofix(
    *,
    test_script: str,
    schema_contract_risks: Sequence[str],
) -> Tuple[str, List[str]]:
    notes: List[str] = []
    text = test_script

    text, hint_changed = _apply_hard_fail_schema_hint_autofix(text)
    if hint_changed:
        notes.append("removed_hard_fail_schema_hint_guard")

    text, drift_notes = _apply_schema_field_drift_autofix(text, schema_contract_risks)
    if drift_notes:
        notes.extend([f"schema_field_drift_fix:{item}" for item in drift_notes])

    return text, notes


async def run_phase1_attempt(
    *,
    attempt: int,
    max_attempts: int,
    messages: List[Message],
    allowed_tools: List[Dict[str, Any]],
    dummy_data: Optional[str],
    access_token: Optional[str],
    tgi_service: Any,
    parse_json: Callable[[str], Dict[str, Any]],
    run_tests: Callable[[str, str, str, Optional[str]], Tuple[bool, str]],
    iterative_test_fix: Callable[
        ...,
        Awaitable[Tuple[bool, str, str, str, Optional[str], List[Message]]],
    ],
    chunk_reader: Callable[..., Any],
    ui_model_headers: Optional[Dict[str, str]],
    gateway_tool_names: Optional[Set[str]] = None,
    rewrite_runtime_scripts: Optional[Callable[[str, str], Tuple[str, str]]] = None,
) -> AsyncIterator[Union[bytes, Dict[str, Any]]]:
    """
    Executes a single attempt of Phase 1 logic generation.
    """
    logger.info(f"[stream_generate_ui] Phase 1 Attempt {attempt}/{max_attempts}")

    yield f"event: log\ndata: {json.dumps({'message': f'Phase 1: Generating logic and tests (Attempt {attempt})'})}\n\n".encode(
        "utf-8"
    )

    chat_request = ChatCompletionRequest(
        messages=messages,
        tools=allowed_tools if allowed_tools else None,
        stream=True,
        response_format=generation_response_format(
            schema=generation_logic_schema, name="generated_logic"
        ),
        extra_headers=ui_model_headers,
    )

    content = ""
    tool_calls_accumulated: Dict[int, dict] = {}
    tool_calls_seen = 0
    last_finish_reason = None

    try:
        stream_source = tgi_service.llm_client.stream_completion(
            chat_request, access_token or "", None
        )
    except Exception as exc:
        logger.error(f"[stream_generate_ui] LLM stream creation failed: {exc}")
        payload = json.dumps({"error": str(exc)})
        yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
        yield {
            "type": "result",
            "success": False,
            "error": str(exc),
            "reason": f"llm_stream_creation_failed: {exc}",
        }
        return

    error_in_attempt = False
    stream_error: Optional[str] = None
    try:
        async with chunk_reader(stream_source) as reader:
            async for parsed in reader.as_parsed():
                if getattr(parsed, "is_keepalive", False):
                    yield parsed.raw.encode("utf-8")
                    continue
                if parsed.is_done:
                    break
                if getattr(parsed, "content", None):
                    content += parsed.content
                if parsed.tool_calls:
                    tool_calls_seen += len(parsed.tool_calls)
                if parsed.accumulated_tool_calls:
                    tool_calls_accumulated = parsed.accumulated_tool_calls
                if parsed.finish_reason:
                    last_finish_reason = parsed.finish_reason
    except Exception as exc:
        stream_error = str(exc)
        logger.error(f"Streaming failed: {exc}")
        error_in_attempt = True

    if error_in_attempt:
        payload = json.dumps({"error": "Streaming failed"})
        yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
        yield {
            "type": "result",
            "success": False,
            "error": "Streaming failed",
            "reason": f"streaming_failed: {stream_error or 'unknown'}",
        }
        return

    if not content and tool_calls_accumulated:
        tool_call_names = sorted(
            {
                call.get("name")
                for call in tool_calls_accumulated.values()
                if call.get("name")
            }
        )
        logger.info(
            "[stream_generate_ui] Phase 1 stream returned tool calls without content. "
            "tool_calls=%s, seen=%s, finish_reason=%s",
            tool_call_names or "unknown",
            tool_calls_seen,
            last_finish_reason or "unknown",
        )

        def _extract_payload_candidate_from_tool_args(raw_args: Any) -> Optional[str]:
            candidate_obj: Optional[Dict[str, Any]] = None
            if isinstance(raw_args, dict):
                if isinstance(raw_args.get("input"), dict):
                    candidate_obj = raw_args.get("input")
                else:
                    candidate_obj = raw_args
            elif isinstance(raw_args, str):
                raw_text = raw_args.strip()
                if not raw_text:
                    return None
                try:
                    parsed = json.loads(raw_text)
                except Exception:
                    return None
                if isinstance(parsed, dict):
                    candidate_obj = parsed

            if not isinstance(candidate_obj, dict):
                return None

            # Only recover from tool arguments when they already carry the
            # phase-1 required payload shape.
            if (
                isinstance(candidate_obj.get("components_script"), str)
                and candidate_obj.get("components_script", "").strip()
                and isinstance(candidate_obj.get("test_script"), str)
                and candidate_obj.get("test_script", "").strip()
            ):
                return json.dumps(candidate_obj, ensure_ascii=False)

            return None

        for idx in sorted(tool_calls_accumulated.keys()):
            args = tool_calls_accumulated[idx].get("arguments")
            if args is None:
                continue
            candidate = _extract_payload_candidate_from_tool_args(args)
            if candidate:
                content = candidate
                logger.info(
                    "[stream_generate_ui] Phase 1 attempt %s recovered payload from tool-call arguments",
                    attempt,
                )
                break

    logger.info(
        "[stream_generate_ui] Phase 1 attempt %s stream summary: content_chars=%s, tool_calls_seen=%s, finish_reason=%s",
        attempt,
        len(content),
        tool_calls_seen,
        last_finish_reason or "unknown",
    )

    if not content:
        reason = (
            "no_content_generated "
            f"(tool_calls_seen={tool_calls_seen}, finish_reason={last_finish_reason or 'unknown'})"
        )
        logger.warning(
            "[stream_generate_ui] Phase 1 attempt %s failed: %s", attempt, reason
        )
        payload = json.dumps({"error": "No content generated"})
        yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
        yield {
            "type": "result",
            "success": False,
            "error": "No content generated",
            "reason": reason,
        }
        return

    try:
        current_payload = parse_json(content)
        service_script = current_payload.get("service_script")
        components_script = current_payload.get("components_script")
        test_script = current_payload.get("test_script")

        # Recovery path: if the model produced template_parts.script instead of
        # components_script, treat it as the phase-1 components script.
        if not components_script:
            template_parts = current_payload.get("template_parts")
            if isinstance(template_parts, dict):
                candidate_script = template_parts.get("script")
                if isinstance(candidate_script, str) and candidate_script.strip():
                    components_script = candidate_script
                    current_payload["components_script"] = components_script
                    logger.info(
                        "[stream_generate_ui] Phase 1 attempt %s recovered components_script from template_parts.script",
                        attempt,
                    )

        if not (components_script and test_script):
            missing_parts = []
            if not components_script:
                missing_parts.append("components_script")
            if not test_script:
                missing_parts.append("test_script")
            raise HTTPException(
                status_code=502,
                detail=f"missing_required_scripts: {', '.join(missing_parts)}",
            )

        if callable(rewrite_runtime_scripts):
            rewritten_service, rewritten_components = rewrite_runtime_scripts(
                service_script or "",
                components_script or "",
            )
            service_script = rewritten_service
            components_script = rewritten_components
            current_payload["service_script"] = service_script
            current_payload["components_script"] = components_script

        # TODO: remove after testing
        logger.info(
            "[stream_generate_ui] Phase 1 initial generation (attempt %s):\n"
            "--- Dummy Data ---\n%s\n"
            "--- Service Script ---\n%s\n"
            "--- Components Script ---\n%s\n"
            "--- Test Script ---\n%s\n",
            attempt,
            dummy_data or "",
            service_script or "",
            components_script or "",
            test_script or "",
        )

        timeout_risks = _detect_timeout_risks(test_script, components_script)
        if timeout_risks:
            reason = f"timeout_risk_patterns_detected: {', '.join(timeout_risks)}"
            logger.warning(
                "[stream_generate_ui] Phase 1 attempt %s failed: %s", attempt, reason
            )
            messages.append(Message(role=MessageRole.ASSISTANT, content=content))
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=(
                        "Your generated scripts include timeout-risk patterns. "
                        "Avoid open-ended async waits/polling in tests and regenerate."
                    ),
                )
            )
            yield {
                "type": "result",
                "success": False,
                "messages": messages,
                "reason": reason,
            }
            return

        quality_risks = _detect_quality_risks(
            test_script,
            components_script,
            service_script,
            gateway_tool_names=gateway_tool_names,
        )
        if quality_risks:
            reason = f"quality_risk_patterns_detected: {', '.join(quality_risks)}"
            logger.warning(
                "[stream_generate_ui] Phase 1 attempt %s failed: %s", attempt, reason
            )
            messages.append(Message(role=MessageRole.ASSISTANT, content=content))
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=(
                        "Your generated scripts include quality-risk patterns that cause "
                        "non-deterministic tests or runtime state errors. "
                        "Regenerate with deterministic mocked tests (service-level and/or "
                        "fetch-route mocking), no direct state seeding for fetched data, "
                        "and safe component callback signatures."
                    ),
                )
            )
            if "runtime_script:gateway_direct_tool_name_call" in quality_risks:
                messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=(
                            "Gateway routing violation: discovered domain tools must not be "
                            "called by plain name (e.g. svc.call('search')). Use full MCP "
                            "tool route keys (svc.call('/api/<server>/tools/<tool>', ...)) "
                            "or gateway metadata so runtime routing resolves through call_tool."
                        ),
                    )
                )
            yield {
                "type": "result",
                "success": False,
                "messages": messages,
                "reason": reason,
            }
            return

        schema_contract_risks = _detect_schema_contract_risks(
            test_script=test_script,
            components_script=components_script,
            dummy_data=dummy_data,
        )
        if schema_contract_risks:
            test_script, autofix_notes = _apply_schema_contract_autofix(
                test_script=test_script,
                schema_contract_risks=schema_contract_risks,
            )
            if autofix_notes:
                logger.info(
                    "[stream_generate_ui] Phase 1 attempt %s applied schema auto-fix hints: %s",
                    attempt,
                    ", ".join(autofix_notes),
                )
                schema_contract_risks = _detect_schema_contract_risks(
                    test_script=test_script,
                    components_script=components_script,
                    dummy_data=dummy_data,
                )

        def _schema_post_success_validator(
            current_service_script: str,
            current_components_script: str,
            current_test_script: str,
            current_dummy_data: Optional[str],
        ) -> Tuple[bool, Optional[str]]:
            del current_service_script
            current_risks = _detect_schema_contract_risks(
                test_script=current_test_script,
                components_script=current_components_script,
                dummy_data=current_dummy_data,
            )
            if not current_risks:
                return True, None
            return (
                False,
                "schema_contract_risk_after_fix: " f"{', '.join(current_risks)}",
            )

        schema_contract_reason: Optional[str] = None
        if schema_contract_risks:
            schema_contract_reason = (
                "schema_contract_risk_detected: " f"{', '.join(schema_contract_risks)}"
            )
            logger.warning(
                "[stream_generate_ui] Phase 1 attempt %s detected schema contract risks: %s",
                attempt,
                schema_contract_reason,
            )
            yield f"event: log\ndata: {json.dumps({'message': 'Schema contract issues detected, attempting auto-fix', 'reason': schema_contract_reason})}\n\n".encode(
                "utf-8"
            )

        yield f"event: log\ndata: {json.dumps({'message': 'Running initial tests...'})}\n\n".encode(
            "utf-8"
        )

        success, output = run_tests(
            service_script,
            components_script,
            test_script,
            dummy_data,
        )

        if schema_contract_reason:
            schema_output = (
                "Schema contract validation failed for generated scripts. "
                f"{schema_contract_reason}"
            )
            output = f"{schema_output}\n\n{output or ''}".strip()
            if success:
                logger.warning(
                    "[stream_generate_ui] Phase 1 attempt %s schema contract gate failed; continuing iterative fix in the same attempt",
                    attempt,
                )
                yield f"event: log\ndata: {json.dumps({'message': 'Schema contract gate failed; continuing iterative fix in this attempt'})}\n\n".encode(
                    "utf-8"
                )
            success = False

        if not success:
            output_tail = _trim_text(output)
            logger.warning(
                "[stream_generate_ui] Phase 1 attempt %s tests failed. Output tail:\n%s",
                attempt,
                output_tail or "<empty>",
            )
            yield f"event: log\ndata: {json.dumps({'message': 'Initial checks failed (tests and/or schema contract)', 'output_tail': output_tail})}\n\n".encode(
                "utf-8"
            )
            yield f"event: log\ndata: {json.dumps({'message': 'Tests failed, starting iterative fix with tools...'})}\n\n".encode(
                "utf-8"
            )

            messages.append(Message(role=MessageRole.ASSISTANT, content=content))
            if schema_contract_reason:
                messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=(
                            "Schema contract issues were detected and must be fixed while repairing tests. "
                            "Use exact dummyData/schema keys in components and tests, remove mismatched fallback "
                            "shapes, avoid weak assertions that pass on units/labels only, and never throw/fail "
                            "tests solely because dummyDataSchemaHints entries exist. "
                            f"Details: {schema_contract_reason}"
                        ),
                    )
                )
            tool_events_queue: asyncio.Queue = asyncio.Queue()

            async def stream_tool_events() -> Tuple[
                bool,
                str,
                str,
                str,
                Optional[str],
                List[Message],
            ]:
                return await iterative_test_fix(
                    service_script=service_script,
                    components_script=components_script,
                    test_script=test_script,
                    dummy_data=dummy_data,
                    messages=messages,
                    allowed_tools=allowed_tools,
                    access_token=access_token,
                    max_attempts=25,
                    event_queue=tool_events_queue,
                    post_success_validator=_schema_post_success_validator,
                )

            fix_task = asyncio.create_task(stream_tool_events())

            updated_messages = messages
            fix_success = False
            fixed_service = ""
            fixed_components = ""
            fixed_test = ""
            fixed_dummy_data = dummy_data

            try:
                while not fix_task.done():
                    try:
                        event_data = tool_events_queue.get_nowait()
                        if event_data.get("event") == "tool_start":
                            why = event_data.get("why") or event_data.get(
                                "fix_explanation"
                            )
                            payload = {
                                "tool": event_data["tool"],
                                "description": event_data["description"],
                            }
                            if why:
                                payload["fix_explanation"] = str(why)
                                payload["why"] = str(why)
                                logger.info(
                                    "[stream_generate_ui] Phase 1 why for tool %s: %s",
                                    event_data["tool"],
                                    _trim_text(str(why), 600),
                                )
                            yield f"event: tool_start\ndata: {json.dumps(payload)}\n\n".encode(
                                "utf-8"
                            )
                        elif event_data.get("event") == "test_result":
                            yield f"event: test_result\ndata: {json.dumps(event_data)}\n\n".encode(
                                "utf-8"
                            )
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.05)

                (
                    fix_success,
                    fixed_service,
                    fixed_components,
                    fixed_test,
                    fixed_dummy_data,
                    updated_messages,
                ) = await fix_task

                while not tool_events_queue.empty():
                    event_data = tool_events_queue.get_nowait()
                    if event_data.get("event") == "tool_start":
                        why = event_data.get("why") or event_data.get("fix_explanation")
                        payload = {
                            "tool": event_data["tool"],
                            "description": event_data["description"],
                        }
                        if why:
                            payload["fix_explanation"] = str(why)
                            payload["why"] = str(why)
                            logger.info(
                                "[stream_generate_ui] Phase 1 why for tool %s: %s",
                                event_data["tool"],
                                _trim_text(str(why), 600),
                            )
                        yield f"event: tool_start\ndata: {json.dumps(payload)}\n\n".encode(
                            "utf-8"
                        )
                    elif event_data.get("event") == "test_result":
                        yield f"event: test_result\ndata: {json.dumps(event_data)}\n\n".encode(
                            "utf-8"
                        )

            except Exception:
                fix_task.cancel()
                raise

            if fix_success:
                post_fix_schema_risks = _detect_schema_contract_risks(
                    test_script=fixed_test,
                    components_script=fixed_components,
                    dummy_data=fixed_dummy_data or dummy_data,
                )
                if post_fix_schema_risks:
                    fix_reason = (
                        "schema_contract_risk_after_fix: "
                        f"{', '.join(post_fix_schema_risks)}"
                    )
                    logger.warning(
                        "[stream_generate_ui] Phase 1 attempt %s failed after iterative fix (defensive fallback): %s",
                        attempt,
                        fix_reason,
                    )
                    yield f"event: log\ndata: {json.dumps({'message': 'Iterative fix completed but schema contract checks still fail', 'reason': fix_reason})}\n\n".encode(
                        "utf-8"
                    )
                    yield {
                        "type": "result",
                        "success": False,
                        "messages": updated_messages,
                        "reason": f"{fix_reason}; initial_test_output_tail={output_tail}",
                    }
                    return

                yield f"event: log\ndata: {json.dumps({'message': 'Tests fixed and passing!'})}\n\n".encode(
                    "utf-8"
                )
                logic_payload = {
                    "service_script": fixed_service,
                    "components_script": fixed_components,
                    "test_script": fixed_test,
                    "dummy_data": fixed_dummy_data or dummy_data,
                }

                if (
                    updated_messages
                    and updated_messages[-1].role == MessageRole.ASSISTANT
                ):
                    clean_content = json.dumps(
                        {
                            "service_script": fixed_service,
                            "components_script": fixed_components,
                            "test_script": fixed_test,
                        },
                        ensure_ascii=False,
                    )
                    updated_messages[-1].content = clean_content

                yield {
                    "type": "result",
                    "success": True,
                    "payload": logic_payload,
                    "messages": updated_messages,
                }
                return
            fix_reason = "tool_fix_failed_after_initial_test_failure"
            logger.warning(
                "[stream_generate_ui] Phase 1 attempt %s failed: %s",
                attempt,
                fix_reason,
            )
            yield f"event: log\ndata: {json.dumps({'message': 'Tool-based fix failed, regenerating from scratch...'})}\n\n".encode(
                "utf-8"
            )
            yield {
                "type": "result",
                "success": False,
                "messages": updated_messages,
                "reason": f"{fix_reason}; initial_test_output_tail={output_tail}",
            }
            return

        yield f"event: log\ndata: {json.dumps({'message': 'Tests passed!'})}\n\n".encode(
            "utf-8"
        )
        logic_payload = {
            **current_payload,
            "dummy_data": dummy_data,
        }
        messages.append(Message(role=MessageRole.ASSISTANT, content=content))
        yield {
            "type": "result",
            "success": True,
            "payload": logic_payload,
            "messages": messages,
        }
        return

    except (json.JSONDecodeError, HTTPException) as exc:
        reason = f"payload_parse_or_validation_error: {exc}"
        logger.warning(
            "[stream_generate_ui] Phase 1 attempt %s failed: %s",
            attempt,
            reason,
        )
        messages.append(Message(role=MessageRole.ASSISTANT, content=content))
        messages.append(
            Message(role=MessageRole.USER, content=f"Error: {exc}. Fix it.")
        )
        yield {
            "type": "result",
            "success": False,
            "messages": messages,
            "reason": reason,
        }
        return
    except Exception as exc:
        reason = f"unexpected_phase1_exception: {exc}"
        logger.error(
            "[stream_generate_ui] Phase 1 attempt %s failed: %s",
            attempt,
            reason,
            exc_info=exc,
        )
        yield {
            "type": "result",
            "success": False,
            "reason": reason,
        }
        return
