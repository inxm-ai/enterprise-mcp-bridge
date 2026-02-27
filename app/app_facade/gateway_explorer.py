"""Gateway and generic tool-exploration logic.

The ``GatewayExplorer`` class encapsulates all methods for discovering
servers, tools, and sub-tools behind both gateway-style and generic
exploration-style MCP endpoints.
"""

import asyncio
import contextlib
import json
import logging
import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

from app.session import MCPSessionBase
from app.utils import token_fingerprint
from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.services.proxied_tgi_service import ProxiedTGIService
from app.app_facade.generated_schemas import generation_response_format
from app.vars import (
    GENERATED_UI_EXPLORE_TOOLS,
    GENERATED_UI_EXPLORE_TOOLS_MAX_CALLS,
    GENERATED_UI_GATEWAY_CALL_TOOL,
    GENERATED_UI_GATEWAY_GET_TOOL,
    GENERATED_UI_GATEWAY_LIST_SERVERS,
    GENERATED_UI_GATEWAY_LIST_TOOLS,
    GENERATED_UI_GATEWAY_PROMPT_ARG_MAX_CHARS,
    GENERATED_UI_GATEWAY_ROLE_ARGS,
    GENERATED_UI_GATEWAY_SERVER_ID_FIELDS,
    GENERATED_UI_GATEWAY_SERVER_ID_URL_REGEX,
)
from app.app_facade.prompt_helpers import to_json_value, extract_content, parse_json

logger = logging.getLogger("uvicorn.error")

UI_MODEL_HEADERS = {"x-inxm-model-capability": "code-generation"}


def _generation_response_format(schema=None, name: str = "generated_ui"):
    return generation_response_format(schema=schema, name=name)


_TOOL_EXPLORE_NAME_PATTERNS = re.compile(
    r"(list[_-]?tools?|get[_-]?tools?|available[_-]?tools?|discover|capabilities|catalog|registry)",
    re.IGNORECASE,
)
# Description keywords that hint at a discovery tool.
_TOOL_EXPLORE_DESC_KEYWORDS = {
    "list",
    "available",
    "tools",
    "capabilities",
    "catalog",
    "discover",
    "registry",
    "provides",
    "exposes",
}

_TOOL_EXPLORE_PLAN_SYSTEM_PROMPT = (
    "You are planning a tool exploration step. Given a set of available MCP tools "
    "and a user prompt, identify which tools should be called to discover additional "
    "sub-tools, capabilities, or resources that are not directly listed but are "
    "accessible through a callable interface.\n"
    "Return a JSON object. For each tool to call, provide the tool name and the "
    "arguments to pass. Only include tools that are likely to reveal additional "
    "capabilities relevant to the user's request.\n"
    "If no exploration is needed, return an empty calls array."
)
_TOOL_EXPLORE_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "arguments": {"type": "object"},
                    "reason": {"type": "string"},
                },
                "required": ["tool_name", "arguments"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["calls"],
    "additionalProperties": False,
}

_TOOL_EXPLORE_PARSE_SYSTEM_PROMPT = (
    "You are analyzing the result of a tool call that was made to discover "
    "additional tools or capabilities from an MCP server.\n"
    "Extract any tool definitions found in the response. Each discovered tool "
    "should have a name, description, and optionally an inputSchema and outputSchema.\n"
    "If the response does not contain tool definitions, return an empty tools array.\n"
    "Only extract actual tool definitions — not data records, resources, or other entities."
)
_TOOL_EXPLORE_PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "inputSchema": {"type": "object"},
                    "outputSchema": {"type": "object"},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["tools"],
    "additionalProperties": False,
}

# Tools that should never be called during exploration
_TOOL_EXPLORE_EXCLUDED = {
    "describe_tool",
    "select-from-tool-response",
}

_GATEWAY_ARG_PLACEHOLDERS = {
    "${prompt}": "prompt",
    "${server_id}": "server_id",
    "${tool_name}": "tool_name",
}


class GatewayExplorer:
    """Discovers servers, tools, and sub-tools behind gateway / generic MCPs."""

    def __init__(self, *, tgi_service: ProxiedTGIService):
        self.tgi_service = tgi_service

    # ------------------------------------------------------------------
    # Tool exploration – discover sub-tools via callable interfaces
    # ------------------------------------------------------------------

    # Gateway pattern: detect MCPs that expose domain tools through a
    # gateway interface.  The four canonical roles (list_servers, list_tools,
    # get_tool, call_tool) are mapped to actual tool names via env vars.
    # The class property builds the forward+reverse maps once at class-load
    # time; if the env vars change at runtime the service must be restarted.

    @staticmethod
    def _build_gateway_maps() -> tuple:
        """Return (role→name, name→role) dicts from env-configured names."""
        role_to_name: Dict[str, str] = {
            "list_servers": GENERATED_UI_GATEWAY_LIST_SERVERS,
            "list_tools": GENERATED_UI_GATEWAY_LIST_TOOLS,
            "get_tool": GENERATED_UI_GATEWAY_GET_TOOL,
            "call_tool": GENERATED_UI_GATEWAY_CALL_TOOL,
        }
        name_to_role: Dict[str, str] = {v: k for k, v in role_to_name.items()}
        return role_to_name, name_to_role

    _gateway_role_to_name, _gateway_name_to_role = _build_gateway_maps.__func__()

    def _gateway_prompt_for_args(self, prompt: str) -> str:
        max_chars = GENERATED_UI_GATEWAY_PROMPT_ARG_MAX_CHARS
        if max_chars <= 0:
            return ""
        prompt_text = prompt if isinstance(prompt, str) else str(prompt or "")
        return prompt_text[:max_chars]

    def _resolve_gateway_arg_template_value(
        self,
        *,
        value: Any,
        context: Dict[str, Any],
    ) -> Tuple[bool, Any]:
        """Resolve placeholder values inside a gateway args template.

        Returns ``(include, resolved)``. ``include=False`` means the key should
        be omitted from the final argument payload.
        """
        if isinstance(value, str):
            ctx_key = _GATEWAY_ARG_PLACEHOLDERS.get(value)
            if ctx_key is not None:
                resolved = context.get(ctx_key)
                return (False, None) if resolved is None else (True, resolved)

            resolved_value = value
            for placeholder, placeholder_ctx_key in _GATEWAY_ARG_PLACEHOLDERS.items():
                if placeholder not in resolved_value:
                    continue
                replacement = context.get(placeholder_ctx_key)
                resolved_value = resolved_value.replace(
                    placeholder,
                    "" if replacement is None else str(replacement),
                )
            if resolved_value != value:
                return True, resolved_value
            return True, value

        if isinstance(value, dict):
            resolved_dict: Dict[str, Any] = {}
            for key, child in value.items():
                include, resolved_child = self._resolve_gateway_arg_template_value(
                    value=child,
                    context=context,
                )
                if include:
                    resolved_dict[str(key)] = resolved_child
            return True, resolved_dict

        if isinstance(value, list):
            resolved_list: List[Any] = []
            for child in value:
                include, resolved_child = self._resolve_gateway_arg_template_value(
                    value=child,
                    context=context,
                )
                if include:
                    resolved_list.append(resolved_child)
            return True, resolved_list

        return True, value

    def _build_gateway_call_args(
        self,
        *,
        role: str,
        defaults: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build call args for a gateway role from defaults + configured template."""
        args: Dict[str, Any] = dict(defaults or {})
        template = GENERATED_UI_GATEWAY_ROLE_ARGS.get(role)
        if template is None:
            return args
        if not isinstance(template, dict):
            logger.warning(
                "[_build_gateway_call_args] Ignoring non-object template for role %s",
                role,
            )
            return args

        for key, template_value in template.items():
            include, resolved = self._resolve_gateway_arg_template_value(
                value=template_value,
                context=context,
            )
            if include:
                args[str(key)] = resolved
            else:
                args.pop(str(key), None)
        return args

    @staticmethod
    def _value_by_dotted_path(data: Dict[str, Any], path: str) -> Any:
        current: Any = data
        for part in (path or "").split("."):
            if not part:
                continue
            if not isinstance(current, dict):
                return None
            if part not in current:
                return None
            current = current.get(part)
        return current

    def _extract_server_id_from_url_like(self, value: str) -> Optional[str]:
        text = (value or "").strip()
        if not text:
            return None

        pattern = (GENERATED_UI_GATEWAY_SERVER_ID_URL_REGEX or "").strip()
        if pattern:
            try:
                match = re.search(pattern, text)
            except re.error as exc:
                logger.warning(
                    "[_extract_server_id_from_url_like] Invalid regex '%s': %s",
                    pattern,
                    exc,
                )
                match = None
            if match:
                with contextlib.suppress(Exception):
                    if "server_id" in match.re.groupindex:
                        sid = match.group("server_id")
                        if sid:
                            return str(sid)
                if match.groups():
                    sid = match.group(1)
                    if sid:
                        return str(sid)

        fallback = re.search(r"/api/([^/]+)/tools/[^/?#]+", text)
        if fallback:
            sid = fallback.group(1)
            if sid:
                return sid
        return None

    def _extract_server_id_from_tool_summary(
        self, summary: Dict[str, Any]
    ) -> Optional[str]:
        if not isinstance(summary, dict):
            return None

        for path in GENERATED_UI_GATEWAY_SERVER_ID_FIELDS:
            raw = self._value_by_dotted_path(summary, path)
            if raw is None:
                continue
            text = raw if isinstance(raw, str) else str(raw)
            text = text.strip()
            if not text:
                continue
            if "/" in text:
                sid = self._extract_server_id_from_url_like(text)
                if sid:
                    return sid
            return text

        return None

    def _detect_gateway_pattern(
        self, tools: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Return a map of gateway tool defs if the tool set looks like a gateway MCP.

        Returns ``None`` when the gateway pattern is not detected.  Otherwise a
        dict keyed by **role** (``list_servers``, ``list_tools``, ``get_tool``,
        ``call_tool``) whose values are the original tool dicts.

        Tool names are resolved through the ``GENERATED_UI_GATEWAY_*`` env
        vars so that non-standard gateway MCPs can be supported.
        """
        by_role: Dict[str, Dict[str, Any]] = {}
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name", "")
            role = self._gateway_name_to_role.get(name)
            if role:
                by_role[role] = t

        # We require at least list_tools + call_tool as the minimum viable
        # gateway.  list_servers and get_tool are optional.
        if "list_tools" in by_role and "call_tool" in by_role:
            return by_role
        return None

    def _extract_text_from_result(self, result: Any) -> str:
        """Pull text content from a tool-call result."""
        text_parts: List[str] = []

        content_items = (
            result.get("content")
            if isinstance(result, dict)
            else getattr(result, "content", None)
        )
        if isinstance(content_items, list):
            for item in content_items:
                t = (
                    item.get("text")
                    if isinstance(item, dict)
                    else getattr(item, "text", None)
                )
                if t:
                    text_parts.append(str(t))

        structured = (
            result.get("structuredContent")
            if isinstance(result, dict)
            else getattr(result, "structuredContent", None)
        )
        if structured is not None:
            text_parts.append(json.dumps(to_json_value(structured), ensure_ascii=False))

        return "\n".join(text_parts)

    @staticmethod
    def _result_is_error(result: Any) -> bool:
        """Return True if the tool-call result indicates an error."""
        if isinstance(result, dict):
            return bool(result.get("isError"))
        return bool(getattr(result, "isError", False))

    def _json_from_result(self, result: Any) -> Any:
        """Parse the first JSON payload found in a tool-call result."""
        text = self._extract_text_from_result(result)
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None

    def _parse_json_values_from_text(self, text: str) -> List[Any]:
        """Parse one or more JSON values from a text blob."""
        stripped = text.strip()
        if not stripped:
            return []

        try:
            return [json.loads(stripped)]
        except Exception:
            pass

        decoder = json.JSONDecoder()
        values: List[Any] = []
        idx = 0
        while idx < len(stripped):
            while idx < len(stripped) and stripped[idx].isspace():
                idx += 1
            if idx >= len(stripped):
                break
            try:
                value, next_idx = decoder.raw_decode(stripped, idx)
            except json.JSONDecodeError:
                break
            values.append(value)
            idx = next_idx
        if values:
            return values

        # Also support newline-delimited JSON payloads.
        line_values: List[Any] = []
        for line in stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                line_values.append(json.loads(line))
            except Exception:
                continue
        return line_values

    def _extract_server_entries_from_payload(
        self, payload: Any
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Extract server entry dicts from known gateway list-servers payload shapes."""
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)], True

        if not isinstance(payload, dict):
            return [], False

        if payload.get("id") or payload.get("name"):
            return [payload], True

        for key in ("result", "servers", "data"):
            if key not in payload:
                continue
            nested = payload.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)], True
            if isinstance(nested, dict):
                if nested.get("id") or nested.get("name"):
                    return [nested], True
                return [], True
            if nested is None:
                return [], True
            return [], True

        return [], False

    def _extract_gateway_server_entries(
        self, result: Any
    ) -> Optional[List[Dict[str, Any]]]:
        """Best-effort extraction of gateway server entries from tool call results.

        Returns ``None`` when the payload shape is not recognized.
        Returns a list (possibly empty) when the payload is valid.
        """
        payloads: List[Any] = []

        structured = (
            result.get("structuredContent")
            if isinstance(result, dict)
            else getattr(result, "structuredContent", None)
        )
        if structured is not None:
            payloads.append(to_json_value(structured))

        content_items = (
            result.get("content")
            if isinstance(result, dict)
            else getattr(result, "content", None)
        )
        if isinstance(content_items, list):
            for item in content_items:
                t = (
                    item.get("text")
                    if isinstance(item, dict)
                    else getattr(item, "text", None)
                )
                if isinstance(t, str):
                    payloads.extend(self._parse_json_values_from_text(t))

        if not payloads:
            text = self._extract_text_from_result(result)
            if text:
                payloads.extend(self._parse_json_values_from_text(text))

        if not payloads:
            return None

        parsed_valid_payload = False
        entries: List[Dict[str, Any]] = []
        for payload in payloads:
            extracted, is_valid_payload = self._extract_server_entries_from_payload(
                payload
            )
            if is_valid_payload:
                parsed_valid_payload = True
                entries.extend(extracted)

        if not parsed_valid_payload:
            return None
        return entries

    @staticmethod
    def _compact_gateway_server_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Return a compact server record suitable for prompt context."""
        compact: Dict[str, Any] = {}
        for key in ("id", "name", "description", "url", "auth_provider", "released"):
            if key in entry:
                compact[key] = entry.get(key)
        if not compact:
            return {"id": str(entry.get("id") or entry.get("name") or "unknown")}
        return compact

    def _extract_tool_entries_from_payload(
        self, payload: Any
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Extract tool summary dicts from known gateway list-tools payload shapes."""
        if isinstance(payload, list):
            return [
                item for item in payload if isinstance(item, dict) and item.get("name")
            ], True

        if not isinstance(payload, dict):
            return [], False

        if payload.get("name"):
            return [payload], True

        for key in ("result", "tools", "data"):
            if key not in payload:
                continue
            nested = payload.get(key)
            if isinstance(nested, list):
                return [
                    item
                    for item in nested
                    if isinstance(item, dict) and item.get("name")
                ], True
            if isinstance(nested, dict):
                if nested.get("name"):
                    return [nested], True
                return [], True
            if nested is None:
                return [], True
            return [], True

        return [], False

    def _extract_gateway_tool_entries(
        self, result: Any
    ) -> Optional[List[Dict[str, Any]]]:
        """Best-effort extraction of gateway tool entries from tool call results.

        Returns ``None`` when the payload shape is not recognized.
        Returns a list (possibly empty) when the payload is valid.
        """
        payloads: List[Any] = []

        structured = (
            result.get("structuredContent")
            if isinstance(result, dict)
            else getattr(result, "structuredContent", None)
        )
        if structured is not None:
            payloads.append(to_json_value(structured))

        content_items = (
            result.get("content")
            if isinstance(result, dict)
            else getattr(result, "content", None)
        )
        if isinstance(content_items, list):
            for item in content_items:
                t = (
                    item.get("text")
                    if isinstance(item, dict)
                    else getattr(item, "text", None)
                )
                if isinstance(t, str):
                    payloads.extend(self._parse_json_values_from_text(t))

        if not payloads:
            text = self._extract_text_from_result(result)
            if text:
                payloads.extend(self._parse_json_values_from_text(text))

        if not payloads:
            return None

        parsed_valid_payload = False
        entries: List[Dict[str, Any]] = []
        for payload in payloads:
            extracted, is_valid_payload = self._extract_tool_entries_from_payload(
                payload
            )
            if is_valid_payload:
                parsed_valid_payload = True
                entries.extend(extracted)

        if not parsed_valid_payload:
            return None
        return entries

    async def gateway_discover_servers(
        self,
        *,
        session: MCPSessionBase,
        gw_list_servers: str,
        prompt: str,
        servers_info: List[Dict[str, Any]],
        access_token: Optional[str],
    ) -> Tuple[List[str], int]:
        """Call the list-servers gateway tool (with one retry on error) and return ``(server_ids, calls_made)``."""
        calls_made = 0
        server_ids: List[str] = []

        for attempt in range(2):  # up to 2 attempts
            if attempt > 0:
                logger.info(
                    "[_gateway_discover_servers] Retrying %s (attempt %d)",
                    gw_list_servers,
                    attempt + 1,
                )
                await asyncio.sleep(1)
            try:
                args = self._build_gateway_call_args(
                    role="list_servers",
                    defaults={},
                    context={
                        "prompt": self._gateway_prompt_for_args(prompt),
                    },
                )
                result = await session.call_tool(gw_list_servers, args, access_token)
                calls_made += 1

                if self._result_is_error(result):
                    err_text = self._extract_text_from_result(result)
                    logger.warning(
                        "[_gateway_discover_servers] %s returned isError=True (attempt %d), "
                        "raw content: %s",
                        gw_list_servers,
                        attempt + 1,
                        err_text[:500],
                    )
                    continue  # retry on error

                entries = self._extract_gateway_server_entries(result)
                if entries is not None:
                    seen_server_ids: set[str] = set()
                    for entry in entries:
                        sid = entry.get("id") or entry.get("name")
                        if not sid:
                            continue
                        sid_str = str(sid)
                        if sid_str in seen_server_ids:
                            continue
                        seen_server_ids.add(sid_str)
                        server_ids.append(sid_str)
                        servers_info.append(self._compact_gateway_server_entry(entry))
                    if server_ids:
                        logger.info(
                            "[_gateway_discover_servers] Discovered %d servers: %s",
                            len(server_ids),
                            server_ids,
                        )
                    else:
                        logger.info(
                            "[_gateway_discover_servers] %s returned valid but empty list (attempt %d)",
                            gw_list_servers,
                            attempt + 1,
                        )
                    # Valid response (even if empty) — do NOT retry
                    break
                else:
                    # Could not parse a list from the response — retry
                    raw_text = self._extract_text_from_result(result)
                    logger.warning(
                        "[_gateway_discover_servers] %s returned unparseable response (attempt %d): %s",
                        gw_list_servers,
                        attempt + 1,
                        raw_text[:300] if raw_text else "(empty)",
                    )
            except Exception as exc:
                calls_made += 1
                logger.warning(
                    "[_gateway_discover_servers] %s failed (attempt %d): %s",
                    gw_list_servers,
                    attempt + 1,
                    exc,
                )

        return server_ids, calls_made

    async def gateway_probe_servers_from_prompt(
        self,
        *,
        session: MCPSessionBase,
        gw_list_tools: str,
        prompt: str,
        tool_summaries_by_server: Dict[str, List[Dict[str, Any]]],
        servers_info: List[Dict[str, Any]],
        access_token: Optional[str],
    ) -> Tuple[List[str], int]:
        """Extract likely server IDs from the prompt and probe each via list-tools.

        Returns ``(server_ids, calls_made)`` for servers that returned tools.
        """
        import re as _re

        # Common MCP server / service names to look for in the prompt
        _KNOWN_HINTS = [
            "github",
            "gitlab",
            "bitbucket",
            "jira",
            "confluence",
            "atlassian",
            "linear",
            "slack",
            "notion",
            "google",
            "microsoft",
            "azure",
            "aws",
            "vercel",
            "heroku",
            "sentry",
            "datadog",
            "pagerduty",
            "stripe",
            "twilio",
            "sendgrid",
            "salesforce",
            "hubspot",
            "zendesk",
        ]
        prompt_lower = prompt.lower()
        candidates: List[str] = []
        for hint in _KNOWN_HINTS:
            if hint in prompt_lower:
                candidates.append(hint)

        # Also try to extract words that look like server IDs (e.g., "my-server")
        for word in _re.findall(r"\b[a-z][a-z0-9_-]{2,20}\b", prompt_lower):
            if word not in candidates and word not in (
                "the",
                "and",
                "for",
                "with",
                "that",
                "this",
                "from",
                "are",
                "was",
                "were",
                "been",
                "have",
                "has",
                "had",
                "not",
                "but",
                "all",
                "can",
                "will",
                "just",
                "more",
                "some",
                "than",
                "them",
                "very",
                "when",
                "what",
                "your",
                "each",
                "make",
                "like",
                "over",
                "such",
                "into",
                "our",
                "also",
                "its",
                "use",
                "how",
                "new",
                "get",
                "set",
                "show",
                "list",
                "create",
                "update",
                "delete",
                "work",
                "dashboard",
                "tool",
                "tools",
                "server",
                "servers",
                "issues",
                "pull",
                "requests",
                "data",
                "user",
            ):
                # Skip — too many false positives from common words
                pass

        if not candidates:
            logger.info(
                "[_gateway_probe_servers_from_prompt] No server hints found in prompt"
            )
            return [], 0

        logger.info(
            "[_gateway_probe_servers_from_prompt] Probing %d candidates from prompt: %s",
            len(candidates),
            candidates,
        )

        calls_made = 0
        found_ids: List[str] = []
        for candidate in candidates:
            if calls_made >= 4:  # limit probing budget
                break
            try:
                args = self._build_gateway_call_args(
                    role="list_tools",
                    defaults={"server_id": candidate},
                    context={
                        "prompt": self._gateway_prompt_for_args(prompt),
                        "server_id": candidate,
                    },
                )
                result = await session.call_tool(gw_list_tools, args, access_token)
                calls_made += 1

                if self._result_is_error(result):
                    logger.info(
                        "[_gateway_probe_servers_from_prompt] %s(server_id=%s) error: %s",
                        gw_list_tools,
                        candidate,
                        self._extract_text_from_result(result)[:200],
                    )
                    continue

                items = self._extract_gateway_tool_entries(result)
                if items is None:
                    raw = self._extract_text_from_result(result)
                    logger.warning(
                        "[_gateway_probe_servers_from_prompt] %s(server_id=%s) returned non-JSON: %s",
                        gw_list_tools,
                        candidate,
                        raw[:500] if raw else "(empty)",
                    )
                    continue
                tools = [i for i in items if isinstance(i, dict) and i.get("name")]
                if tools:
                    found_ids.append(candidate)
                    tool_summaries_by_server[candidate] = tools
                    servers_info.append({"id": candidate, "name": candidate})
                    logger.info(
                        "[_gateway_probe_servers_from_prompt] Found %d tools on '%s'",
                        len(tools),
                        candidate,
                    )
            except Exception as exc:
                calls_made += 1
                logger.info(
                    "[_gateway_probe_servers_from_prompt] %s(server_id=%s) failed: %s",
                    gw_list_tools,
                    candidate,
                    exc,
                )

        return found_ids, calls_made

    async def explore_gateway(
        self,
        *,
        session: MCPSessionBase,
        gateway: Dict[str, Dict[str, Any]],
        prompt: str,
        access_token: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Deterministic multi-step exploration of a gateway MCP.

        Returns ``(discovered_tools, exploration_context)`` where each
        discovered tool carries ``_gateway`` metadata and the context dict
        is meant to be injected into the code-generator prompt.
        """
        discovered: List[Dict[str, Any]] = []
        servers_info: List[Dict[str, Any]] = []
        max_calls = GENERATED_UI_EXPLORE_TOOLS_MAX_CALLS

        logger.info(
            "[_explore_gateway] access_token=%s, session_type=%s",
            token_fingerprint(access_token),
            type(session).__name__,
        )

        # Resolve configured tool names for each gateway role
        gw_list_servers = self._gateway_role_to_name["list_servers"]
        gw_list_tools = self._gateway_role_to_name["list_tools"]
        gw_get_tool = self._gateway_role_to_name["get_tool"]
        gw_call_tool = self._gateway_role_to_name["call_tool"]

        # ---- Step 1: discover servers ---------------------------------
        server_ids: List[str] = []
        calls_made = 0
        if "list_servers" in gateway:
            server_ids, calls_made = await self.gateway_discover_servers(
                session=session,
                gw_list_servers=gw_list_servers,
                prompt=prompt,
                servers_info=servers_info,
                access_token=access_token,
            )

        else:
            logger.info(
                "[_explore_gateway] No %s tool; skipping server discovery",
                gw_list_servers,
            )

        # ---- Step 1b: fallback — try list_tools without server_id -----
        tool_summaries_by_server: Dict[str, List[Dict[str, Any]]] = {}
        if not server_ids:
            logger.info(
                "[_explore_gateway] No servers discovered; trying %s() without server_id",
                gw_list_tools,
            )
            try:
                list_tools_args = self._build_gateway_call_args(
                    role="list_tools",
                    defaults={},
                    context={
                        "prompt": self._gateway_prompt_for_args(prompt),
                    },
                )
                result = await session.call_tool(
                    gw_list_tools,
                    list_tools_args,
                    access_token,
                )
                calls_made += 1
                if self._result_is_error(result):
                    logger.warning(
                        "[_explore_gateway] %s() returned error: %s",
                        gw_list_tools,
                        self._extract_text_from_result(result)[:200],
                    )
                else:
                    items = self._extract_gateway_tool_entries(result)
                    if items is None:
                        raw = self._extract_text_from_result(result)
                        logger.warning(
                            "[_explore_gateway] %s() returned non-JSON response: %s",
                            gw_list_tools,
                            raw[:500] if raw else "(empty)",
                        )
                    else:
                        flat_tools = [
                            i for i in items if isinstance(i, dict) and i.get("name")
                        ]
                        if flat_tools:
                            # Group by server_id if present, else use "_default"
                            for ft in flat_tools:
                                sid = (
                                    self._extract_server_id_from_tool_summary(ft)
                                    or "_default"
                                )
                                s = str(sid)
                                tool_summaries_by_server.setdefault(s, []).append(ft)
                            server_ids = list(tool_summaries_by_server.keys())
                            logger.info(
                                "[_explore_gateway] Fallback discovered %d tools across servers: %s",
                                len(flat_tools),
                                server_ids,
                            )
            except Exception as exc:
                logger.warning(
                    "[_explore_gateway] %s() (no server_id) failed: %s",
                    gw_list_tools,
                    exc,
                )

        # ---- Step 1c: last resort — try known server IDs from prompt --
        if not server_ids:
            server_ids, calls_made_1c = await self.gateway_probe_servers_from_prompt(
                session=session,
                gw_list_tools=gw_list_tools,
                prompt=prompt,
                tool_summaries_by_server=tool_summaries_by_server,
                servers_info=servers_info,
                access_token=access_token,
            )
            calls_made += calls_made_1c

        if not server_ids:
            logger.info("[_explore_gateway] No servers or tools discovered")
            return [], {}

        logger.info("[_explore_gateway] Discovered servers: %s", server_ids)

        # ---- Step 2: list tools per server ----------------------------
        for sid in server_ids:
            if sid in tool_summaries_by_server:
                continue  # Already populated by fallback step 1b
            if calls_made >= max_calls:
                break
            logger.info(
                "[_explore_gateway] Calling %s(server_id=%s)", gw_list_tools, sid
            )
            try:
                list_tools_args = self._build_gateway_call_args(
                    role="list_tools",
                    defaults={"server_id": sid},
                    context={
                        "prompt": self._gateway_prompt_for_args(prompt),
                        "server_id": sid,
                    },
                )
                result = await session.call_tool(
                    gw_list_tools,
                    list_tools_args,
                    access_token,
                )
                calls_made += 1
                if self._result_is_error(result):
                    logger.warning(
                        "[_explore_gateway] %s(server_id=%s) returned error: %s",
                        gw_list_tools,
                        sid,
                        self._extract_text_from_result(result)[:500],
                    )
                    continue
                items = self._extract_gateway_tool_entries(result)
                if items is None:
                    raw = self._extract_text_from_result(result)
                    logger.warning(
                        "[_explore_gateway] %s(server_id=%s) returned non-JSON response: %s",
                        gw_list_tools,
                        sid,
                        raw[:500] if raw else "(empty)",
                    )
                    continue
                tool_summaries_by_server[sid] = [
                    i for i in items if isinstance(i, dict) and i.get("name")
                ]
            except Exception as exc:
                logger.warning(
                    "[_explore_gateway] %s(%s) failed: %s", gw_list_tools, sid, exc
                )

        # ---- Step 3: get full tool definitions (optional) -------------
        has_get_tool = "get_tool" in gateway
        for sid, summaries in tool_summaries_by_server.items():
            for tsummary in summaries:
                tname = tsummary.get("name")
                if not tname:
                    continue

                tdesc = tsummary.get("description") or ""
                input_schema = tsummary.get("inputSchema") or {}
                output_schema = tsummary.get("outputSchema") or None

                # If we have get_tool and budget, fetch the full definition
                if has_get_tool and calls_made < max_calls and not input_schema:
                    try:
                        logger.info(
                            "[_explore_gateway] Calling %s(%s, %s)",
                            gw_get_tool,
                            sid,
                            tname,
                        )
                        get_tool_args = self._build_gateway_call_args(
                            role="get_tool",
                            defaults={
                                "server_id": sid,
                                "tool_name": tname,
                            },
                            context={
                                "prompt": self._gateway_prompt_for_args(prompt),
                                "server_id": sid,
                                "tool_name": tname,
                            },
                        )
                        result = await session.call_tool(
                            gw_get_tool,
                            get_tool_args,
                            access_token,
                        )
                        calls_made += 1
                        if self._result_is_error(result):
                            logger.warning(
                                "[_explore_gateway] %s(%s, %s) returned error: %s",
                                gw_get_tool,
                                sid,
                                tname,
                                self._extract_text_from_result(result)[:500],
                            )
                        else:
                            detail = self._json_from_result(result)
                            if detail is None:
                                raw = self._extract_text_from_result(result)
                                logger.warning(
                                    "[_explore_gateway] %s(%s, %s) returned non-JSON response: %s",
                                    gw_get_tool,
                                    sid,
                                    tname,
                                    raw[:500] if raw else "(empty)",
                                )
                            elif isinstance(detail, dict):
                                tdesc = detail.get("description") or tdesc
                                input_schema = detail.get("inputSchema") or input_schema
                                output_schema = (
                                    detail.get("outputSchema") or output_schema
                                )
                    except Exception as exc:
                        logger.warning(
                            "[_explore_gateway] %s(%s, %s) failed: %s",
                            gw_get_tool,
                            sid,
                            tname,
                            exc,
                        )

                tool_def: Dict[str, Any] = {
                    "type": "function",
                    "function": {
                        "name": tname,
                        "description": tdesc,
                        "parameters": input_schema,
                    },
                    # Marker: this tool is accessed through the gateway
                    "_gateway": {
                        "server_id": sid,
                        "via_tool": gw_call_tool,
                    },
                }
                if output_schema:
                    tool_def["function"]["outputSchema"] = output_schema

                discovered.append(tool_def)

        logger.info(
            "[_explore_gateway] Discovered %d tools across %d servers (%d API calls)",
            len(discovered),
            len(server_ids),
            calls_made,
        )

        # Build the exploration context for the code-generator prompt
        exploration_context = self._build_exploration_context(
            servers_info=servers_info,
            discovered=discovered,
        )
        return discovered, exploration_context

    def _build_exploration_context(
        self,
        *,
        servers_info: List[Dict[str, Any]],
        discovered: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a structured context dict for the code-generator prompt."""
        tool_listing: List[Dict[str, Any]] = []
        for t in discovered:
            fn = t.get("function", {})
            gw = t.get("_gateway", {})
            entry: Dict[str, Any] = {
                "name": fn.get("name"),
                "description": fn.get("description"),
                "server_id": gw.get("server_id"),
            }
            params = fn.get("parameters")
            if params:
                entry["inputSchema"] = params
            out = fn.get("outputSchema")
            if out:
                entry["outputSchema"] = out
            tool_listing.append(entry)

        gw_call_tool = self._gateway_role_to_name["call_tool"]
        return {
            "gateway_pattern": True,
            "invocation_guidance": (
                "The MCP server is a gateway/proxy. Domain tools are NOT available as "
                "top-level tool names. Instead, call them through the `{gw}` "
                "gateway tool:\n"
                "  svc.callTool('{gw}', {{\n"
                "    server_id: '<server_id>',\n"
                "    tool_name: '<tool_name>',\n"
                "    input_data: {{ ... }}\n"
                "  }})\n"
                "The `input_data` object must match the tool's inputSchema. "
                "Do NOT try to call discovered tool names directly — they are "
                "only reachable via {gw}."
            ).format(gw=gw_call_tool),
            "servers": servers_info,
            "discovered_tools": tool_listing,
        }

    def _is_exploration_candidate(self, tool: Dict[str, Any]) -> bool:
        """Return True if *tool* looks like it might expose sub-tools."""
        function = tool.get("function", {})
        name = function.get("name", "")
        description = function.get("description", "")
        if name in _TOOL_EXPLORE_EXCLUDED:
            return False
        if _TOOL_EXPLORE_NAME_PATTERNS.search(name):
            return True
        desc_lower = (description or "").lower()
        matching = sum(1 for kw in _TOOL_EXPLORE_DESC_KEYWORDS if kw in desc_lower)
        return matching >= 2

    async def _plan_exploration_calls(
        self,
        *,
        tools: List[Dict[str, Any]],
        prompt: str,
        access_token: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Ask the LLM which tools to call for sub-tool discovery."""
        llm_client = getattr(self.tgi_service, "llm_client", None)
        if not llm_client:
            return []
        non_stream = getattr(llm_client, "non_stream_completion", None)
        if not callable(non_stream):
            return []

        candidates = [t for t in tools if self._is_exploration_candidate(t)]
        if not candidates:
            logger.info(
                "[_plan_exploration_calls] No exploration candidate tools found"
            )
            return []

        tool_summaries = []
        for t in candidates:
            fn = t.get("function", {})
            tool_summaries.append(
                {
                    "name": fn.get("name"),
                    "description": fn.get("description"),
                    "parameters": fn.get("parameters"),
                }
            )

        request = ChatCompletionRequest(
            messages=[
                Message(
                    role=MessageRole.SYSTEM, content=_TOOL_EXPLORE_PLAN_SYSTEM_PROMPT
                ),
                Message(
                    role=MessageRole.USER,
                    content=json.dumps(
                        {
                            "user_prompt": prompt,
                            "candidate_tools": tool_summaries,
                        },
                        ensure_ascii=False,
                    ),
                ),
            ],
            tools=None,
            stream=False,
            response_format=_generation_response_format(
                schema=_TOOL_EXPLORE_PLAN_SCHEMA,
                name="tool_exploration_plan",
            ),
            extra_headers=UI_MODEL_HEADERS,
        )

        try:
            response = await non_stream(request, access_token or "", None)
        except Exception as exc:
            logger.warning("[_plan_exploration_calls] LLM call failed: %s", exc)
            return []

        try:
            text = extract_content(response)
            parsed = parse_json(text)
            calls = parsed.get("calls", [])
            if not isinstance(calls, list):
                return []
            # Cap to configured max
            return calls[:GENERATED_UI_EXPLORE_TOOLS_MAX_CALLS]
        except Exception as exc:
            logger.warning("[_plan_exploration_calls] Failed to parse plan: %s", exc)
            return []

    def _parse_discovered_tools_from_result(self, result: Any) -> List[Dict[str, Any]]:
        """Best-effort extraction of tool definitions from a raw tool call result."""
        text = self._extract_text_from_result(result)
        if not text:
            return []

        # Try to parse as JSON and look for tool-like structures
        discovered: List[Dict[str, Any]] = []
        try:
            payload = json.loads(text)
        except Exception:
            payload = None

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and item.get("name"):
                    discovered.append(self._normalize_discovered_tool(item))
        elif isinstance(payload, dict):
            # Could be {"tools": [...]} or similar wrapper
            for key in ("tools", "items", "capabilities", "results", "data", "result"):
                items = payload.get(key)
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and item.get("name"):
                            discovered.append(self._normalize_discovered_tool(item))
                    if discovered:
                        break
            # Single tool definition at top level
            if not discovered and payload.get("name"):
                discovered.append(self._normalize_discovered_tool(payload))

        return discovered

    def _normalize_discovered_tool(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a raw discovered tool dict into the OpenAI function-calling format."""
        input_schema = (
            raw.get("inputSchema")
            or raw.get("input_schema")
            or raw.get("parameters")
            or {}
        )
        output_schema = raw.get("outputSchema") or raw.get("output_schema") or None
        tool: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": raw.get("name", "unknown"),
                "description": raw.get("description") or "",
                "parameters": input_schema,
            },
        }
        if output_schema:
            tool["function"]["outputSchema"] = output_schema
        return tool

    async def _parse_discovered_tools_with_llm(
        self,
        *,
        tool_name: str,
        result: Any,
        access_token: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Use the LLM to extract tool definitions from an ambiguous result."""
        llm_client = getattr(self.tgi_service, "llm_client", None)
        if not llm_client:
            return []
        non_stream = getattr(llm_client, "non_stream_completion", None)
        if not callable(non_stream):
            return []

        # Build a text representation of the result
        result_text = json.dumps(to_json_value(result), ensure_ascii=False)
        # Cap to avoid oversized prompts
        if len(result_text) > 16000:
            result_text = result_text[:16000] + "... (truncated)"

        request = ChatCompletionRequest(
            messages=[
                Message(
                    role=MessageRole.SYSTEM, content=_TOOL_EXPLORE_PARSE_SYSTEM_PROMPT
                ),
                Message(
                    role=MessageRole.USER,
                    content=json.dumps(
                        {
                            "source_tool": tool_name,
                            "result": result_text,
                        },
                        ensure_ascii=False,
                    ),
                ),
            ],
            tools=None,
            stream=False,
            response_format=_generation_response_format(
                schema=_TOOL_EXPLORE_PARSE_SCHEMA,
                name="tool_exploration_parse",
            ),
            extra_headers=UI_MODEL_HEADERS,
        )

        try:
            response = await non_stream(request, access_token or "", None)
        except Exception as exc:
            logger.warning(
                "[_parse_discovered_tools_with_llm] LLM call failed: %s", exc
            )
            return []

        try:
            text = extract_content(response)
            parsed = parse_json(text)
            raw_tools = parsed.get("tools", [])
            if not isinstance(raw_tools, list):
                return []
            return [
                self._normalize_discovered_tool(t)
                for t in raw_tools
                if isinstance(t, dict) and t.get("name")
            ]
        except Exception as exc:
            logger.warning(
                "[_parse_discovered_tools_with_llm] Failed to parse: %s", exc
            )
            return []

    async def explore_generic(
        self,
        *,
        session: MCPSessionBase,
        tools: List[Dict[str, Any]],
        prompt: str,
        access_token: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """LLM-planned exploration for non-gateway tool sets."""
        planned_calls = await self._plan_exploration_calls(
            tools=tools, prompt=prompt, access_token=access_token
        )
        if not planned_calls:
            logger.info("[_explore_generic] No exploration calls planned")
            return tools, None

        logger.info(
            "[_explore_generic] Planned %d exploration calls: %s",
            len(planned_calls),
            [c.get("tool_name") for c in planned_calls],
        )

        existing_names: set = set()
        for t in tools:
            fn = t.get("function", {})
            n = fn.get("name")
            if n:
                existing_names.add(n)

        all_discovered: List[Dict[str, Any]] = []

        for call_spec in planned_calls:
            tool_name = call_spec.get("tool_name")
            arguments = call_spec.get("arguments", {})
            if not tool_name or tool_name in _TOOL_EXPLORE_EXCLUDED:
                continue
            if tool_name not in existing_names:
                logger.warning(
                    "[_explore_generic] Planned tool '%s' not in available tools, skipping",
                    tool_name,
                )
                continue

            logger.info(
                "[_explore_generic] Calling '%s' with args=%s", tool_name, arguments
            )
            try:
                result = await session.call_tool(tool_name, arguments, access_token)
            except Exception as exc:
                logger.warning(
                    "[_explore_generic] Call to '%s' failed: %s", tool_name, exc
                )
                continue

            discovered = self._parse_discovered_tools_from_result(result)
            if not discovered:
                discovered = await self._parse_discovered_tools_with_llm(
                    tool_name=tool_name, result=result, access_token=access_token
                )

            for dtool in discovered:
                dname = dtool.get("function", {}).get("name")
                if dname and dname not in existing_names:
                    all_discovered.append(dtool)
                    existing_names.add(dname)
                    logger.info("[_explore_generic] Discovered new tool: '%s'", dname)

        if all_discovered:
            logger.info(
                "[_explore_generic] Discovered %d new tools: %s",
                len(all_discovered),
                [t.get("function", {}).get("name") for t in all_discovered],
            )
            return list(tools) + all_discovered, None

        logger.info("[_explore_generic] No new tools discovered")
        return tools, None

    async def explore_tools(
        self,
        *,
        session: MCPSessionBase,
        tools: List[Dict[str, Any]],
        prompt: str,
        access_token: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Explore tools by calling discovery/listing tools to find sub-tools.

        When GENERATED_UI_EXPLORE_TOOLS is enabled, this method:
        1. Detects the gateway pattern (get_servers/get_tools/call_tool)
        2. For gateways: deterministically explores servers and their tools
        3. For others: uses LLM-planned exploration calls
        4. Returns (augmented_tools, exploration_context)

        The exploration_context (if any) is a dict that should be injected
        into the code-generator prompt so it knows how to invoke discovered
        tools (e.g. via the ``call_tool`` gateway).

        This runs *before* dummy data generation so that discovered tools
        can also get proper dummy data.
        """
        if not GENERATED_UI_EXPLORE_TOOLS:
            return tools, None
        if not tools:
            return tools, None

        logger.info(
            "[_explore_tools] Starting tool exploration for %d tools", len(tools)
        )

        # Fast path: detect the gateway pattern
        gateway = self._detect_gateway_pattern(tools)
        if gateway:
            logger.info(
                "[_explore_tools] Gateway pattern detected — using deterministic exploration"
            )

            # Strip gateway meta-tools from the tool list — they are
            # infrastructure tools and must not be sent to the LLM for
            # UI/dummy-data generation.
            gateway_tool_names = set(self._gateway_role_to_name.values())
            non_gateway_tools = [
                t
                for t in tools
                if t.get("function", {}).get("name") not in gateway_tool_names
            ]
            logger.info(
                "[_explore_tools] Stripped %d gateway meta-tools, %d remaining",
                len(tools) - len(non_gateway_tools),
                len(non_gateway_tools),
            )

            discovered, context = await self.explore_gateway(
                session=session,
                gateway=gateway,
                prompt=prompt,
                access_token=access_token,
            )
            if discovered:
                # Merge discovered tools with remaining non-gateway tools
                existing_names = {
                    t.get("function", {}).get("name") for t in non_gateway_tools
                }
                augmented = list(non_gateway_tools)
                for dt in discovered:
                    dname = dt.get("function", {}).get("name")
                    if dname and dname not in existing_names:
                        augmented.append(dt)
                        existing_names.add(dname)
                return augmented, context
            return non_gateway_tools, context

        # Fallback: LLM-planned exploration for non-gateway MCPs
        return await self.explore_generic(
            session=session,
            tools=tools,
            prompt=prompt,
            access_token=access_token,
        )
