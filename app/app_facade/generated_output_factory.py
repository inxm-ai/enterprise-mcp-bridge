import copy
import html as _html_escape
import logging
import re
from typing import Any, Dict, Optional

from fastapi import HTTPException

from app.app_facade.generated_types import Scope


logger = logging.getLogger("uvicorn.error")

SNIPPET_PLACEHOLDER = "<!-- include:snippet -->"
SERVICE_SCRIPT_PLACEHOLDER = "<!-- include:service_script -->"
COMPONENTS_SCRIPT_PLACEHOLDER = "<!-- include:components_script -->"
SCRIPT_BLOCK_TEMPLATE = (
    '<script type="module">\n'
    "  <!-- include:service_script -->\n\n"
    "  <!-- include:components_script -->\n"
    "</script>"
)
SCRIPT_BLOCK_RE = re.compile(
    r"<script[^>]*>.*?(?:include:service_script|include:components_script).*?</script>",
    re.IGNORECASE | re.DOTALL,
)
SCRIPT_TAG_RE = re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)


class GeneratedUIOutputFactory:
    def _script_block_with_placeholders(self) -> str:
        return SCRIPT_BLOCK_TEMPLATE

    def _script_block_with_code(
        self, service_script: str, components_script: str
    ) -> str:
        return (
            '<script type="module">\n'
            f"{service_script or ''}\n\n"
            f"{components_script or ''}\n"
            "</script>"
        )

    def _strip_placeholder_script_block(self, html: str) -> str:
        return SCRIPT_BLOCK_RE.sub("", html or "").strip()

    def _strip_all_script_blocks(self, html: str) -> str:
        return SCRIPT_TAG_RE.sub("", html or "").strip()

    def _ensure_page_script_block(self, page: str) -> str:
        if SCRIPT_BLOCK_RE.search(page or ""):
            return page
        cleaned = (
            (page or "")
            .replace(SERVICE_SCRIPT_PLACEHOLDER, "")
            .replace(COMPONENTS_SCRIPT_PLACEHOLDER, "")
        )
        script_block = self._script_block_with_placeholders()
        if re.search(r"</body>", cleaned, flags=re.IGNORECASE):
            return re.sub(
                r"</body>",
                f"{script_block}\n</body>",
                cleaned,
                count=1,
                flags=re.IGNORECASE,
            )
        return f"{cleaned}\n{script_block}"

    def _ensure_snippet_script_block(self, snippet: str) -> str:
        if SCRIPT_BLOCK_RE.search(snippet or ""):
            return snippet
        cleaned = self._strip_all_script_blocks(snippet)
        cleaned = cleaned.replace(SERVICE_SCRIPT_PLACEHOLDER, "").replace(
            COMPONENTS_SCRIPT_PLACEHOLDER, ""
        )
        script_block = self._script_block_with_placeholders()
        return f"{cleaned}\n{script_block}"

    def _expand_snippet(
        self, snippet: str, service_script: str, components_script: str
    ) -> str:
        if not snippet:
            return snippet
        if (
            SERVICE_SCRIPT_PLACEHOLDER in snippet
            or COMPONENTS_SCRIPT_PLACEHOLDER in snippet
        ):
            expanded = snippet.replace(SERVICE_SCRIPT_PLACEHOLDER, service_script or "")
            expanded = expanded.replace(
                COMPONENTS_SCRIPT_PLACEHOLDER, components_script or ""
            )
            if "<script" not in expanded:
                return f"{expanded}\n{self._script_block_with_code(service_script, components_script)}"
            return expanded
        return f"{snippet}\n{self._script_block_with_code(service_script, components_script)}"

    def expand_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expand the HTML payload by injecting snippet, service, and component
        content into their placeholders. Returns a deep copy of the payload with expansions.
        """
        expanded = copy.deepcopy(payload)

        html_section = expanded.get("html", {})
        page = html_section.get("page")

        snippet = html_section.get("snippet")
        service_script = expanded.get("service_script", "")
        components_script = expanded.get("components_script", "")
        snippet_with_placeholders = None
        if isinstance(snippet, str):
            snippet_with_placeholders = self._ensure_snippet_script_block(snippet)

        if page and isinstance(page, str):
            page = self._ensure_page_script_block(page)
            if isinstance(snippet_with_placeholders, str):
                page_snippet = self._strip_placeholder_script_block(
                    snippet_with_placeholders
                )
                page = page.replace(SNIPPET_PLACEHOLDER, page_snippet)
            page = page.replace(SERVICE_SCRIPT_PLACEHOLDER, service_script or "")
            page = page.replace(COMPONENTS_SCRIPT_PLACEHOLDER, components_script or "")

            html_section["page"] = page

        if isinstance(snippet_with_placeholders, str):
            html_section["snippet"] = self._expand_snippet(
                snippet_with_placeholders, service_script, components_script
            )

        return expanded

    def normalise_payload(
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
                    "page": self.wrap_snippet(synthesized_snippet),
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
            snippet = self.extract_body(page) or page
            html_section["snippet"] = snippet
        if not page and snippet:
            html_section["page"] = self.wrap_snippet(snippet)

        snippet = html_section.get("snippet")
        page = html_section.get("page")
        if isinstance(snippet, str):
            html_section["snippet"] = self._ensure_snippet_script_block(snippet)
        if isinstance(page, str):
            html_section["page"] = self._ensure_page_script_block(page)

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

    def extract_body(self, html: str) -> Optional[str]:
        match = re.search(
            r"<body[^>]*>(.*?)</body>", html, flags=re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return None

    def wrap_snippet(self, snippet: str) -> str:
        return (
            '<!DOCTYPE html><html lang="en"><head>'
            '<meta charset="utf-8"/>'
            '<meta name="viewport" content="width=device-width, initial-scale=1"/>'
            "<title>Generated Ui</title>"
            "</head><body>"
            f"{snippet}"
            "</body></html>"
        )
