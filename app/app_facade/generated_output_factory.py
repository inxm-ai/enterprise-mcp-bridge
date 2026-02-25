import copy
import html as _html_escape
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import HTTPException

from app.app_facade.generated_types import Scope
from app.vars import MCP_BASE_PATH


logger = logging.getLogger("uvicorn.error")

SNIPPET_PLACEHOLDER = "<!-- include:snippet -->"
SERVICE_SCRIPT_PLACEHOLDER = "<!-- include:service_script -->"
COMPONENTS_SCRIPT_PLACEHOLDER = "<!-- include:components_script -->"
SCRIPT_BLOCK_TEMPLATE = (
    '<script type="module">\n'
    "  <!-- include:runtime_bridge -->\n"
    "  <!-- include:service_script -->\n\n"
    "  <!-- include:components_script -->\n"
    "</script>"
)
SCRIPT_BLOCK_RE = re.compile(
    r"<script[^>]*>.*?(?:include:service_script|include:components_script).*?</script\s*>",
    re.IGNORECASE | re.DOTALL,
)
SCRIPT_TAG_RE = re.compile(r"<script[^>]*>.*?</script\s*>", re.IGNORECASE | re.DOTALL)
RUNTIME_BRIDGE_MARKER = "generated-ui-runtime"
RUNTIME_BRIDGE_PLACEHOLDER = "<!-- include:runtime_bridge -->"
MCP_SERVICE_RUNTIME_MARKER = "generated-mcp-service-helper"
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _load_script_template(
    name: str, replacements: Optional[Dict[str, str]] = None
) -> str:
    template_path = _TEMPLATES_DIR / name
    try:
        content = template_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to load script template %s: %s", template_path, exc)
        raise

    for marker, value in (replacements or {}).items():
        content = content.replace(marker, value)

    if not content.endswith("\n"):
        content += "\n"
    return content


RUNTIME_BRIDGE_SCRIPT = _load_script_template(
    "generated_ui_runtime_bridge.js",
    {"{{RUNTIME_BRIDGE_MARKER}}": RUNTIME_BRIDGE_MARKER},
)
_MCP_SERVICE_CLASS_SOURCE = _load_script_template(
    "generated_mcp_service_class.js",
    {"{{MCP_BASE_PATH}}": MCP_BASE_PATH},
)
MCP_SERVICE_HELPER_SCRIPT = (
    f"/* {MCP_SERVICE_RUNTIME_MARKER} */\n"
    "(() => {\n"
    f"{_MCP_SERVICE_CLASS_SOURCE}\n"
    "  const needsGeneratedClass =\n"
    "    typeof globalThis.McpService !== 'function'\n"
    "    || typeof globalThis.McpService.prototype?.call !== 'function';\n"
    "  if (needsGeneratedClass) {\n"
    "    globalThis.McpService = __GeneratedMcpService;\n"
    "  }\n"
    "  const needsServiceInstance =\n"
    "    !globalThis.service || typeof globalThis.service.call !== 'function';\n"
    "  if (needsServiceInstance) {\n"
    "    globalThis.service = new globalThis.McpService();\n"
    "  }\n"
    "})();\n"
)
MCP_SERVICE_TEST_HELPER_SCRIPT = (
    "/* generated-mcp-service-helper-test */\n"
    "(() => {\n"
    f"{_MCP_SERVICE_CLASS_SOURCE}\n"
    "  const __needsGeneratedClassForTests =\n"
    "    typeof globalThis.McpService !== 'function'\n"
    "    || typeof globalThis.McpService.prototype?.call !== 'function';\n"
    "  if (__needsGeneratedClassForTests) {\n"
    "    globalThis.McpService = __GeneratedMcpService;\n"
    "  }\n"
    "  if (!globalThis.service || typeof globalThis.service.call !== 'function') {\n"
    "    globalThis.service = new globalThis.McpService();\n"
    "  }\n"
    "})();\n"
)
RUNTIME_BOOTSTRAP_SCRIPT = f"{RUNTIME_BRIDGE_SCRIPT}{MCP_SERVICE_HELPER_SCRIPT}"


class GeneratedUIOutputFactory:
    def _infer_root_tag_from_script(self, script: str) -> Optional[str]:
        if not isinstance(script, str) or not script.strip():
            return None
        try:
            match = re.search(
                r"\bpfusch\s*\(\s*['\"]([a-z][a-z0-9-]*)['\"]",
                script,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1)
        except Exception:
            return None
        return None

    def _script_block_with_placeholders(self) -> str:
        return SCRIPT_BLOCK_TEMPLATE

    def _script_block_with_code(
        self, service_script: str, components_script: str
    ) -> str:
        return (
            '<script type="module">\n'
            f"{RUNTIME_BOOTSTRAP_SCRIPT}"
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
            page = page.replace(RUNTIME_BRIDGE_PLACEHOLDER, RUNTIME_BOOTSTRAP_SCRIPT)

            html_section["page"] = page

        if isinstance(snippet_with_placeholders, str):
            snippet_expanded = self._expand_snippet(
                snippet_with_placeholders, service_script, components_script
            )
            html_section["snippet"] = snippet_expanded.replace(
                RUNTIME_BRIDGE_PLACEHOLDER, RUNTIME_BOOTSTRAP_SCRIPT
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

        if not isinstance(payload.get("template_parts"), dict):
            legacy_title = payload.get("title")
            legacy_styles = payload.get("styles")
            legacy_script = payload.get("script")
            legacy_html = payload.get("html")

            has_legacy_parts = any(
                isinstance(value, str) and value.strip()
                for value in (legacy_title, legacy_styles, legacy_script)
            )
            if has_legacy_parts or isinstance(legacy_html, str):
                snippet_html = legacy_html if isinstance(legacy_html, str) else ""
                if not snippet_html.strip():
                    inferred_tag = self._infer_root_tag_from_script(
                        str(legacy_script or "")
                    )
                    snippet_html = (
                        f"<{inferred_tag}></{inferred_tag}>"
                        if inferred_tag
                        else '<div class="generated-root"></div>'
                    )

                payload["template_parts"] = {
                    "title": str(legacy_title or "Generated Ui"),
                    "styles": str(legacy_styles or ""),
                    "html": snippet_html,
                    "script": str(legacy_script or ""),
                }
                logger.warning(
                    "Generated payload used legacy top-level parts; auto-upgraded to template_parts. keys=%s",
                    list(payload.keys()),
                )

        template_parts = payload.get("template_parts")
        if isinstance(template_parts, dict):
            title = _html_escape.escape(
                str(template_parts.get("title") or "Generated Ui")
            )
            styles = str(template_parts.get("styles") or "")
            body_html = str(template_parts.get("html") or "")
            parts_script = str(template_parts.get("script") or "")

            if parts_script:
                existing_components = payload.get("components_script")
                if isinstance(existing_components, str) and existing_components.strip():
                    if parts_script not in existing_components:
                        payload["components_script"] = (
                            f"{existing_components.rstrip()}\n\n{parts_script}"
                        )
                else:
                    payload["components_script"] = parts_script

            if not isinstance(payload.get("html"), dict):
                payload["html"] = {
                    "page": (
                        '<!DOCTYPE html><html lang="en"><head>'
                        '<meta charset="utf-8"/>'
                        '<meta name="viewport" content="width=device-width, initial-scale=1"/>'
                        f"<title>{title}</title>"
                        f"<style data-pfusch>{styles}</style>"
                        "</head><body>"
                        f"{SNIPPET_PLACEHOLDER}\n"
                        f"{self._script_block_with_placeholders()}"
                        "</body></html>"
                    ),
                    "snippet": body_html,
                }

        html_section = payload.get("html")

        # If the model returned a bare string for the html section, accept it
        if isinstance(html_section, str):
            if html_section.strip():
                html_section = {"page": html_section}
            else:
                html_section = None

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
