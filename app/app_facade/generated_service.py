import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from fastapi import HTTPException

from app.session import MCPSessionBase
from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.services.proxied_tgi_service import ProxiedTGIService


IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

DEFAULT_DESIGN_PROMPT = (
    "Use lightweight, responsive layouts. Prefer utility-first styling via Tailwind "
    "CSS conventions when no explicit design system guidance is provided."
)

DEFAULT_PFUSCH_PROMPT = (
    "You are a microsite and dashboard designer that produces structured JSON. "
    "All interactive behaviour must be implemented with pfusch, a minimal progressive enhancement "
    "library that works directly in the browser. Follow these rules:\n"
    "- Start with semantic HTML that works without JavaScript, then enhance it.\n"
    "- Load pfusch using a module script tag: "
    '"<script type=\\"module\\">import { pfusch, html, css, script } from '
    "'https://matthiaskainer.github.io/pfusch/pfusch.min.js'; ... </script>\".\n"
    "- When you need shared styles from a design system, include "
    '"<link rel=\\"stylesheet\\" href=\\"...\\" data-pfusch>" so pfusch '
    "components inherit them.\n"
    "- Define interactivity by registering custom elements with pfusch and using "
    "its html/css/script helpers. Do not use React, JSX, frameworks, or build steps.\n"
    "- Forms should remain standard HTML forms; enhance them by mutating pfusch "
    "state or subscribing to events rather than replacing native behaviour.\n"
    "- Use pfusch triggers/events for component communication instead of global "
    "framework state.\n"
    "- Example usage:\n"
    '  <div class=\\"dashboard\\">\\n'
    "    <feedback-panel>\\n"
    "      <form method='post' action='/api/feedback'>\\n"
    "        <label>Comment<input name='comment' placeholder='Say hi'></label>\\n"
    "        <button type='submit'>Send</button>\\n"
    "      </form>\\n"
    "    </feedback-panel>\\n"
    "  </div>\\n"
    '  <script type=\\"module\\">\\n'
    "    import { pfusch, html, script } from 'https://matthiaskainer.github.io/pfusch/pfusch.min.js';\\n"
    "    pfusch('feedback-panel', { status: 'idle', history: [] }, (state, trigger, helpers) => [\\n"
    "      script(function() {\\n"
    "        const [form] = helpers.children('form');\\n"
    "        if (!form) return;\\n"
    "        const textarea = form.querySelector('textarea, input[name=\\\\'comment\\\\']');\\n"
    "        state.subscribe('status', (value) => {\\n"
    "          this.dataset.status = value;\\n"
    "        });\\n"
    "        form.addEventListener('submit', async (event) => {\\n"
    "          event.preventDefault();\\n"
    "          state.status = 'saving';\\n"
    "          const payload = new FormData(form);\\n"
    "          await fetch(form.action || '#', { method: form.method || 'post', body: payload });\\n"
    "          const text = textarea ? textarea.value : '';\\n"
    "          state.history = [{ text, at: Date.now() }, ...state.history].slice(0, 5);\\n"
    "          state.status = 'saved';\\n"
    "          trigger('submitted', { text });\\n"
    "        });\\n"
    "      }),\\n"
    "      html.slot(),\\n"
    "      html.div({ class: 'status' }, state.status),\\n"
    "      html.ul(\\n"
    "        ...state.history.map((item) =>\\n"
    "          html.li(\\n"
    "            html.time(new Date(item.at).toLocaleTimeString()),\\n"
    "            html.span(' ', item.text)\\n"
    "          )\\n"
    "        )\\n"
    "      )\\n"
    "    ]);\\n"
    "    window.addEventListener('feedback-panel.submitted', (event) => {\\n"
    "      console.log('Feedback event', event.detail);\\n"
    "    });\\n"
    "  </script>\n"
    "Output strictly in JSON with top-level keys `html` and `metadata`. Within "
    "`html`, provide `page` (a complete HTML document with pfusch imports) and "
    "`snippet` (only the enhanced dashboard content). `metadata` must capture "
    "requirements, pfusch components used, and guidance for future updates. "
    "Do not include Markdown fences or explanatory prose—return only JSON."
)


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
            raise ValueError("Generated dashboard storage path is required")
        self.base_path = os.path.abspath(base_path)

    def _dashboard_dir(self, scope: Scope, dashboard_id: str, name: str) -> str:
        safe_scope = validate_identifier(scope.identifier, f"{scope.kind} id")
        safe_dashboard_id = validate_identifier(dashboard_id, "dashboard id")
        safe_name = validate_identifier(name, "dashboard name")
        return os.path.join(
            self.base_path, scope.kind, safe_scope, safe_dashboard_id, safe_name
        )

    def _file_path(self, scope: Scope, dashboard_id: str, name: str) -> str:
        return os.path.join(
            self._dashboard_dir(scope, dashboard_id, name), "dashboard.json"
        )

    def read(self, scope: Scope, dashboard_id: str, name: str) -> Dict[str, Any]:
        file_path = self._file_path(scope, dashboard_id, name)
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError as exc:
            raise
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Stored dashboard payload at {file_path} is invalid JSON",
            ) from exc

    def write(
        self, scope: Scope, dashboard_id: str, name: str, payload: Dict[str, Any]
    ) -> None:
        file_path = self._file_path(scope, dashboard_id, name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        tmp_path = f"{file_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, file_path)

    def exists(self, scope: Scope, dashboard_id: str, name: str) -> bool:
        return os.path.exists(self._file_path(scope, dashboard_id, name))


class GeneratedUIService:
    def __init__(
        self,
        *,
        storage: GeneratedUIStorage,
        tgi_service: Optional[ProxiedTGIService] = None,
    ):
        self.storage = storage
        self.tgi_service = tgi_service or ProxiedTGIService()

    async def create_dashboard(
        self,
        *,
        session: MCPSessionBase,
        scope: Scope,
        actor: Actor,
        dashboard_id: str,
        name: str,
        prompt: str,
        tools: Optional[Iterable[str]],
        access_token: Optional[str],
    ) -> Dict[str, Any]:
        if self.storage.exists(scope, dashboard_id, name):
            raise HTTPException(
                status_code=409,
                detail="Dashboard already exists for this id and name",
            )

        if scope.kind == "user" and actor.user_id != scope.identifier:
            raise HTTPException(
                status_code=403,
                detail="User dashboards may only be created by the owning user",
            )

        if scope.kind == "group" and scope.identifier not in set(actor.groups or []):
            raise HTTPException(
                status_code=403,
                detail="Group dashboards may only be created by group members",
            )

        generated = await self._generate_dashboard_payload(
            session=session,
            scope=scope,
            dashboard_id=dashboard_id,
            name=name,
            prompt=prompt,
            tools=list(tools or []),
            access_token=access_token,
            previous=None,
        )

        timestamp = self._now()
        record = {
            "metadata": {
                "id": dashboard_id,
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

        self.storage.write(scope, dashboard_id, name, record)
        return record

    async def update_dashboard(
        self,
        *,
        session: MCPSessionBase,
        scope: Scope,
        actor: Actor,
        dashboard_id: str,
        name: str,
        prompt: str,
        tools: Optional[Iterable[str]],
        access_token: Optional[str],
    ) -> Dict[str, Any]:
        try:
            existing = self.storage.read(scope, dashboard_id, name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dashboard not found") from exc

        self._assert_scope_consistency(existing, scope, name)
        self._ensure_update_permissions(existing, scope, actor)

        generated = await self._generate_dashboard_payload(
            session=session,
            scope=scope,
            dashboard_id=dashboard_id,
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

        self.storage.write(scope, dashboard_id, name, existing)
        return existing

    def get_dashboard(
        self,
        *,
        scope: Scope,
        actor: Actor,
        dashboard_id: str,
        name: str,
    ) -> Dict[str, Any]:
        try:
            existing = self.storage.read(scope, dashboard_id, name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dashboard not found") from exc

        self._assert_scope_consistency(existing, scope, name)
        self._ensure_update_permissions(existing, scope, actor)
        return existing

    async def _generate_dashboard_payload(
        self,
        *,
        session: MCPSessionBase,
        scope: Scope,
        dashboard_id: str,
        name: str,
        prompt: str,
        tools: List[str],
        access_token: Optional[str],
        previous: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        system_prompt = await self._build_system_prompt(session)
        message_payload = {
            "dashboard": {
                "id": dashboard_id,
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

        allowed_tools = await self._select_tools(session, tools)

        chat_request = ChatCompletionRequest(
            messages=messages,
            tools=allowed_tools if allowed_tools else None,
            stream=False,
        )

        response = await self.tgi_service.well_planned_chat_completion(
            session, chat_request, access_token
        )

        content = self._extract_content(response)
        payload = self._parse_json(content)
        self._normalise_payload(payload, scope, dashboard_id, name, prompt, previous)
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
        return (
            f"{DEFAULT_PFUSCH_PROMPT}\n\nDesign system guidelines:\n{combined_design}"
        )

    async def _select_tools(
        self, session: MCPSessionBase, requested_tools: Sequence[str]
    ) -> Optional[List[Dict[str, Any]]]:
        if not requested_tools:
            return None
        available = await self.tgi_service.tool_service.get_all_mcp_tools(session)
        selected: List[Dict[str, Any]] = []
        for tool in available or []:
            tool_name: Optional[str] = None
            if isinstance(tool, dict):
                tool_name = tool.get("function", {}).get("name")
            else:
                tool_name = getattr(tool, "function", None)
                if tool_name and hasattr(tool_name, "name"):
                    tool_name = tool_name.name
            if tool_name and tool_name in requested_tools:
                selected.append(tool)
        return selected

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
        dashboard_id: str,
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
        if isinstance(html_section, str):
            html_section = {"page": html_section}
        if not isinstance(html_section, dict):
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
        metadata.setdefault("id", dashboard_id)
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
            "<title>Generated Dashboard</title>"
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
                detail="Scope mismatch for stored dashboard",
            )
        if metadata.get("name") and metadata.get("name") != name:
            raise HTTPException(status_code=403, detail="Dashboard name mismatch")

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
