import json
import logging
import html as _html_escape
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence
from typing import AsyncIterator

from fastapi import HTTPException

from app.session import MCPSessionBase
from app.vars import MCP_BASE_PATH
from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.protocols.chunk_reader import chunk_reader
from app.tgi.services.proxied_tgi_service import ProxiedTGIService


IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

DEFAULT_DESIGN_PROMPT = (
    "Use lightweight, responsive layouts. Prefer utility-first styling via Tailwind "
    "CSS conventions when no explicit design system guidance is provided."
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
        logging.error(f"Error loading pfusch prompt: {e}")
        raise e


def _get_fallback_prompt() -> str:
    """Fallback prompt if the markdown file cannot be loaded."""
    return (
        "You are a microsite and ui designer that produces structured JSON. "
        "All interactive behaviour must be implemented with pfusch, a minimal progressive enhancement "
        "library that works directly in the browser. Follow these rules:\n"
        "- Start with semantic HTML that works without JavaScript, then enhance it.\n"
        "- Load pfusch using a module script tag: "
        '"<script type=\\"module\\">import { pfusch, html, css, script } from '
        "'https://matthiaskainer.github.io/pfusch/pfusch.min.js'; ... </script>\".\n"
        f"- The Base Url for all API calls is `{MCP_BASE_PATH}/tools/<tool_name>`, "
        "they all need POST to return data, and the body is the MCP input.\n"
        "Output strictly in JSON with top-level keys `html` and `metadata`."
    )


# JSON Schema describing the expected structure of the generated UI payload.
# This is intentionally permissive for fields the service will normalise later
# (for example either `html.page` or `html.snippet` may be provided).
generation_ui_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "GeneratedUi",
    "type": "object",
    "properties": {
        "html": {
            "type": "object",
            "description": "HTML output. Either a full `page` (document) or a `snippet` (embed) is acceptable.",
            "properties": {
                "page": {
                    "type": "string",
                    "description": "Complete HTML document as a string",
                },
                "snippet": {
                    "type": "string",
                    "description": "HTML fragment suitable for embedding",
                },
            },
            "additionalProperties": True,
        },
        "metadata": {
            "type": "object",
            "description": "Auxiliary metadata about the generated ui",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "scope": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["user", "group"]},
                        "id": {"type": "string"},
                    },
                    "required": ["type", "id"],
                    "additionalProperties": False,
                },
                "requirements": {"type": "string"},
                "original_requirements": {"type": "string"},
                "components": {"type": "array", "items": {"type": "string"}},
                "pfusch_components": {"type": "array", "items": {"type": "string"}},
                "created_by": {"type": "string"},
                "created_at": {"type": "string", "format": "date-time"},
                "updated_at": {"type": "string", "format": "date-time"},
                "history": {"type": "array", "items": {"type": "object"}},
            },
            "additionalProperties": True,
        },
    },
    "required": ["html"],
    "additionalProperties": True,
}


@dataclass(frozen=True)
class Scope:
    kind: str  # "group" or "user"
    identifier: str


@dataclass(frozen=True)
class Actor:
    user_id: str
    groups: Sequence[str]

    def is_owner(self, scope: Scope) -> bool:
        if scope.kind == "user":
            return self.user_id == scope.identifier
        return scope.identifier in set(self.groups or [])


def validate_identifier(value: str, field_label: str) -> str:
    if not value:
        raise HTTPException(status_code=400, detail=f"{field_label} must not be empty")
    if not IDENTIFIER_RE.fullmatch(value):
        raise HTTPException(
            status_code=400,
            detail=f"{field_label} must match pattern {IDENTIFIER_RE.pattern}",
        )
    return value


class GeneratedUIStorage:
    def __init__(self, base_path: str):
        if not base_path:
            raise ValueError("Generated ui storage path is required")
        self.base_path = os.path.abspath(base_path)

    def _ui_dir(self, scope: Scope, ui_id: str, name: str) -> str:
        safe_scope = validate_identifier(scope.identifier, f"{scope.kind} id")
        safe_ui_id = validate_identifier(ui_id, "ui id")
        safe_name = validate_identifier(name, "ui name")
        return os.path.join(
            self.base_path, scope.kind, safe_scope, safe_ui_id, safe_name
        )

    def _file_path(self, scope: Scope, ui_id: str, name: str) -> str:
        return os.path.join(self._ui_dir(scope, ui_id, name), "ui.json")

    def read(self, scope: Scope, ui_id: str, name: str) -> Dict[str, Any]:
        file_path = self._file_path(scope, ui_id, name)
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            raise
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Stored ui payload at {file_path} is invalid JSON",
            ) from exc

    def write(
        self, scope: Scope, ui_id: str, name: str, payload: Dict[str, Any]
    ) -> None:
        file_path = self._file_path(scope, ui_id, name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        tmp_path = f"{file_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, file_path)

    def exists(self, scope: Scope, ui_id: str, name: str) -> bool:
        return os.path.exists(self._file_path(scope, ui_id, name))


class GeneratedUIService:
    def __init__(
        self,
        *,
        storage: GeneratedUIStorage,
        tgi_service: Optional[ProxiedTGIService] = None,
    ):
        self.storage = storage
        self.tgi_service = tgi_service or ProxiedTGIService()

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
        record = {
            "metadata": {
                "id": ui_id,
                "name": name,
                "scope": {"type": scope.kind, "id": scope.identifier},
                "owner": {"type": scope.kind, "id": scope.identifier},
                "created_by": actor.user_id,
                "created_at": timestamp,
                "updated_at": timestamp,
                "history": [
                    self._history_entry(
                        action="create",
                        prompt=prompt,
                        tools=list(tools or []),
                        user_id=actor.user_id,
                        generated_at=timestamp,
                        payload_metadata=generated.get("metadata", {}),
                    )
                ],
            },
            "current": generated,
        }

        self.storage.write(scope, ui_id, name, record)
        return record

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
        will emit keepalive comments when the underlying stream yields parsed
        items without content to avoid client timeouts.
        """
        # Basic existence and permission checks similar to create_ui
        if self.storage.exists(scope, ui_id, name):
            # SSE error message and stop
            payload = json.dumps({"error": "Ui already exists for this id and name"})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            return

        if scope.kind == "user" and actor.user_id != scope.identifier:
            payload = json.dumps(
                {"error": "User uis may only be created by the owning user"}
            )
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            return

        if scope.kind == "group" and scope.identifier not in set(actor.groups or []):
            payload = json.dumps(
                {"error": "Group uis may only be created by group members"}
            )
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            return

        system_prompt = await self._build_system_prompt(session)
        message_payload = {
            "ui": {
                "id": ui_id,
                "name": name,
                "scope": {"type": scope.kind, "id": scope.identifier},
            },
            "request": {"prompt": prompt, "tools": list(tools or [])},
        }

        messages = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(
                role=MessageRole.USER,
                content=json.dumps(message_payload, ensure_ascii=False),
            ),
        ]

        allowed_tools = await self._select_tools(session, list(tools or []), prompt)

        chat_request = ChatCompletionRequest(
            messages=messages,
            tools=allowed_tools if allowed_tools else None,
            stream=True,
            response_format={
                "type": "json_schema",
                "json_schema": generation_ui_schema,
            },
        )

        # Stream source from LLM
        content = ""
        stream_source = self.tgi_service.llm_client.stream_completion(
            chat_request, access_token or "", None
        )

        async with chunk_reader(stream_source) as reader:
            # Read parsed chunks. If a parsed item has no content, emit a keepalive.
            async for parsed in reader.as_parsed():
                # If the parsed chunk marks completion, stop looping
                if parsed.is_done:
                    break

                # When parsed contains content, append and send partial update
                if getattr(parsed, "content", None):
                    content_piece = parsed.content
                    content += content_piece
                    # send a chunk SSE with the partial content
                    # wrap it in a JSON object to make it easy for clients
                    payload = json.dumps({"chunk": content_piece})
                    yield f"data: {payload}\n\n".encode("utf-8")
                else:
                    # Emit a keepalive comment (SSE comment) so clients don't timeout.
                    # Comments start with ':' and are ignored by SSE parsers but keep
                    # the connection alive.
                    yield b":\n\n"

        if not content:
            # No content generated
            payload = json.dumps({"error": "Generation response was empty"})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            return

        # Parse and normalise the final payload
        try:
            payload_obj = self._parse_json(content)
            self._normalise_payload(payload_obj, scope, ui_id, name, prompt, None)
        except HTTPException as exc:
            payload = json.dumps({"error": exc.detail})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            return

        # Build final record and persist
        timestamp = self._now()
        record = {
            "metadata": {
                "id": ui_id,
                "name": name,
                "scope": {"type": scope.kind, "id": scope.identifier},
                "owner": {"type": scope.kind, "id": scope.identifier},
                "created_by": actor.user_id,
                "created_at": timestamp,
                "updated_at": timestamp,
                "history": [
                    self._history_entry(
                        action="create",
                        prompt=prompt,
                        tools=list(tools or []),
                        user_id=actor.user_id,
                        generated_at=timestamp,
                        payload_metadata=payload_obj.get("metadata", {}),
                    )
                ],
            },
            "current": payload_obj,
        }

        # persist
        try:
            self.storage.write(scope, ui_id, name, record)
        except Exception as e:
            payload = json.dumps({"error": f"Failed to persist generated ui: {str(e)}"})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            return

        # Final event with the created record
        final_payload = json.dumps(
            {"status": "created", "record": record}, ensure_ascii=False
        )
        yield f"event: done\ndata: {final_payload}\n\n[DONE]".encode("utf-8")

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
        metadata["updated_at"] = timestamp
        metadata["updated_by"] = actor.user_id
        history = metadata.setdefault("history", [])
        history.append(
            self._history_entry(
                action="update",
                prompt=prompt,
                tools=list(tools or []),
                user_id=actor.user_id,
                generated_at=timestamp,
                payload_metadata=generated.get("metadata", {}),
            )
        )

        existing["current"] = generated

        self.storage.write(scope, ui_id, name, existing)
        return existing

    def get_ui(
        self,
        *,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
    ) -> Dict[str, Any]:
        try:
            existing = self.storage.read(scope, ui_id, name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Ui not found") from exc

        self._assert_scope_consistency(existing, scope, name)
        self._ensure_update_permissions(existing, scope, actor)
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
                "history": previous_metadata.get("history", []),
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

        chat_request = ChatCompletionRequest(
            messages=messages,
            tools=allowed_tools if allowed_tools else None,
            stream=True,
            response_format={
                "type": "json_schema",
                "json_schema": generation_ui_schema,
            },
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

        logging.info(
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
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=502,
                detail="Generated payload must be a JSON object",
            )

        html_section = payload.get("html")

        # If the model returned a bare string for the html section, accept it
        if isinstance(html_section, str):
            html_section = {"page": html_section}

        # If there is no explicit `html` key, attempt to salvage common
        # alternatives (top-level `page`, `snippet`, `content`, `body`, etc.)
        if html_section is None:
            maybe_page = (
                payload.get("page")
                or payload.get("html_page")
                or payload.get("page_html")
            )
            maybe_snippet = (
                payload.get("snippet")
                or payload.get("body")
                or payload.get("content")
                or payload.get("text")
            )

            if isinstance(maybe_page, str) or isinstance(maybe_snippet, str):
                html_section = {}
                if isinstance(maybe_page, str):
                    html_section["page"] = maybe_page
                if isinstance(maybe_snippet, str):
                    html_section["snippet"] = maybe_snippet
            else:
                # As a last resort, search the payload for a single string
                # value that looks like HTML and treat it as a page.
                for val in payload.values():
                    if isinstance(val, str) and ("<" in val and ">" in val):
                        html_section = {"page": val}
                        break

        if not isinstance(html_section, dict):
            # As a last-resort fallback, synthesize a minimal HTML page using
            # the original prompt so the API can return something usable
            # instead of failing with 502. Log a warning so this can be
            # investigated.
            logger = logging.getLogger(__name__)
            logger.warning(
                "Generated payload missing 'html' object; synthesizing fallback page. payload keys: %s",
                list(payload.keys()),
            )
            try:
                prompt_snippet = _html_escape.escape(prompt or "")
                synthesized_snippet = (
                    f'<div class="generated-fallback">{prompt_snippet}</div>'
                )
                html_section = {
                    "page": self._wrap_snippet(synthesized_snippet),
                    "snippet": synthesized_snippet,
                }
            except Exception:
                raise HTTPException(
                    status_code=502,
                    detail="Generated payload must include an 'html' object",
                )

        snippet = html_section.get("snippet")
        page = html_section.get("page")

        if not snippet and page:
            snippet = self._extract_body(page) or page
            html_section["snippet"] = snippet
        if not page and snippet:
            html_section["page"] = self._wrap_snippet(snippet)

        payload["html"] = html_section

        metadata = payload.setdefault("metadata", {})
        metadata.setdefault("id", ui_id)
        metadata.setdefault("name", name)
        metadata.setdefault("scope", {"type": scope.kind, "id": scope.identifier})
        metadata.setdefault("requirements", prompt)

        if previous and "metadata" in previous:
            previous_metadata = previous.get("metadata", {})
            history = previous_metadata.get("history") or []
            original_prompt = None
            if history:
                first_entry = history[0]
                if isinstance(first_entry, dict):
                    original_prompt = first_entry.get("prompt")
            if original_prompt:
                metadata.setdefault("original_requirements", original_prompt)

    def _extract_body(self, html: str) -> Optional[str]:
        match = re.search(
            r"<body[^>]*>(.*?)</body>", html, flags=re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return None

    def _wrap_snippet(self, snippet: str) -> str:
        return (
            '<!DOCTYPE html><html lang="en"><head>'
            '<meta charset="utf-8"/>'
            '<meta name="viewport" content="width=device-width, initial-scale=1"/>'
            "<title>Generated Ui</title>"
            '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/pfusch/dist/pfusch.css"/>'
            "</head><body>"
            f"{snippet}"
            "</body></html>"
        )

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

    def _history_entry(
        self,
        *,
        action: str,
        prompt: str,
        tools: List[str],
        user_id: str,
        generated_at: str,
        payload_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "action": action,
            "prompt": prompt,
            "tools": tools,
            "user_id": user_id,
            "generated_at": generated_at,
            "payload_metadata": payload_metadata,
        }

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
