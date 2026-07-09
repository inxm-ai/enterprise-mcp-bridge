"""Operation-based patching for conversational UI edits.

Instead of asking the model to re-emit whole files inside a JSON payload
(slow, token-hungry, and brittle because of JSON escaping), the patch
planner returns a list of small edit operations that are applied to the
current draft payload server-side:

* ``replace`` – replace an exact ``search`` string with ``content``
* ``append``  – append ``content`` to the end of the target
* ``set``     – replace the whole target with ``content`` (new/small files)

Application is all-or-nothing: if any operation fails, the draft stays
untouched and the collected errors are fed back into the next patch
attempt so the model can correct its search strings.
"""

import copy
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("uvicorn.error")

# Targets addressable by patch operations. ``html_page``/``html_snippet``
# map into the nested ``html`` object of the payload.
SCRIPT_TARGETS = (
    "service_script",
    "components_script",
    "test_script",
    "dummy_data",
)
HTML_TARGETS = ("html_page", "html_snippet")
PATCH_TARGETS = SCRIPT_TARGETS + HTML_TARGETS

PATCH_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "patch": {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target": {
                                "type": "string",
                                "enum": list(PATCH_TARGETS),
                            },
                            "op": {
                                "type": "string",
                                "enum": ["replace", "append", "set"],
                            },
                            "search": {
                                "type": "string",
                                "description": (
                                    "Exact text copied verbatim from the current "
                                    "target (required for op=replace)"
                                ),
                            },
                            "content": {
                                "type": "string",
                                "description": (
                                    "Replacement text (replace), appended text "
                                    "(append), or full new content (set)"
                                ),
                            },
                            "replace_all": {
                                "type": "boolean",
                                "description": "Replace every occurrence of search (default false)",
                            },
                        },
                        "required": ["target", "op", "content"],
                        "additionalProperties": False,
                    },
                },
                "metadata": {
                    "type": "object",
                    "additionalProperties": True,
                },
            },
            "required": ["operations"],
            "additionalProperties": False,
        }
    },
    "required": ["patch"],
    "additionalProperties": False,
}


def _read_target(payload: Dict[str, Any], target: str) -> Optional[str]:
    if target in SCRIPT_TARGETS:
        value = payload.get(target)
    elif target == "html_page":
        value = (payload.get("html") or {}).get("page")
    elif target == "html_snippet":
        value = (payload.get("html") or {}).get("snippet")
    else:
        return None
    return value if isinstance(value, str) else None


def _write_target(payload: Dict[str, Any], target: str, value: str) -> None:
    if target in SCRIPT_TARGETS:
        payload[target] = value
    elif target == "html_page":
        payload.setdefault("html", {})["page"] = value
    elif target == "html_snippet":
        payload.setdefault("html", {})["snippet"] = value


def _preview(text: str, limit: int = 120) -> str:
    text = text or ""
    return text if len(text) <= limit else f"{text[:limit]}..."


def legacy_patch_to_operations(patch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a legacy whole-file patch object into ``set`` operations."""
    operations: List[Dict[str, Any]] = []
    html_patch = patch.get("html")
    if isinstance(html_patch, dict):
        for key, target in (("page", "html_page"), ("snippet", "html_snippet")):
            value = html_patch.get(key)
            if isinstance(value, str) and value.strip():
                operations.append({"target": target, "op": "set", "content": value})
    for key in SCRIPT_TARGETS:
        value = patch.get(key)
        if isinstance(value, str):
            operations.append({"target": key, "op": "set", "content": value})
    return operations


def apply_patch_operations(
    payload: Dict[str, Any],
    operations: List[Any],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Apply *operations* to a copy of *payload*.

    Returns ``(candidate, errors)``. ``candidate`` is ``None`` when any
    operation failed (all-or-nothing); ``errors`` then describes each
    failure precisely enough for the model to retry with corrected input.
    """
    errors: List[str] = []
    if not isinstance(operations, list) or not operations:
        return None, ["no_operations_provided"]

    candidate = copy.deepcopy(payload)
    for index, op_raw in enumerate(operations):
        if not isinstance(op_raw, dict):
            errors.append(f"op[{index}]: not_an_object")
            continue
        target = op_raw.get("target")
        op = op_raw.get("op")
        content = op_raw.get("content")
        if target not in PATCH_TARGETS:
            errors.append(f"op[{index}]: unknown_target '{target}'")
            continue
        if not isinstance(content, str):
            errors.append(f"op[{index}]: missing_content for target '{target}'")
            continue

        current = _read_target(candidate, target)
        if op == "set":
            _write_target(candidate, target, content)
            continue
        if op == "append":
            base = current or ""
            joined = f"{base}\n{content}" if base and not base.endswith("\n") else f"{base}{content}"
            _write_target(candidate, target, joined)
            continue
        if op == "replace":
            search = op_raw.get("search")
            if not isinstance(search, str) or not search:
                errors.append(
                    f"op[{index}]: replace requires a non-empty 'search' string "
                    f"(target '{target}')"
                )
                continue
            if current is None:
                errors.append(
                    f"op[{index}]: target '{target}' is empty/missing; use op='set' "
                    "to create it"
                )
                continue
            occurrences = current.count(search)
            if occurrences == 0:
                errors.append(
                    f"op[{index}]: search text not found in '{target}': "
                    f"'{_preview(search)}'. Copy the text verbatim from the "
                    "current file content."
                )
                continue
            if op_raw.get("replace_all"):
                updated = current.replace(search, content)
            else:
                updated = current.replace(search, content, 1)
            _write_target(candidate, target, updated)
            continue
        errors.append(f"op[{index}]: unknown_op '{op}'")

    if errors:
        return None, errors
    return candidate, []


# ---------------------------------------------------------------------------
# Runtime-script integrity: service_script and components_script are
# concatenated into ONE ES module (app.js in tests, the inline module script
# in production), so duplicate import identifiers or repeated pfusch
# component registrations across them are fatal SyntaxErrors / silent bugs.
# ---------------------------------------------------------------------------

_IMPORT_LINE_RE = re.compile(r"^[ \t]*import\b")
_NAMED_IMPORTS_RE = re.compile(r"\{([^}]*)\}")
_DEFAULT_IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z_$][\w$]*)\s*(?:,|\s+from\b)")
_NAMESPACE_IMPORT_RE = re.compile(r"\*\s*as\s+([A-Za-z_$][\w$]*)")
_PFUSCH_REGISTRATION_RE = re.compile(r"\bpfusch\(\s*['\"]([A-Za-z][\w-]*)['\"]")


def _import_line_identifiers(line: str) -> List[str]:
    identifiers: List[str] = []
    default_match = _DEFAULT_IMPORT_RE.match(line)
    if default_match:
        identifiers.append(default_match.group(1))
    namespace_match = _NAMESPACE_IMPORT_RE.search(line)
    if namespace_match:
        identifiers.append(namespace_match.group(1))
    named_match = _NAMED_IMPORTS_RE.search(line)
    if named_match:
        for part in named_match.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            # `orig as alias` binds the alias
            alias = part.split(" as ")[-1].strip()
            if alias:
                identifiers.append(alias)
    return identifiers


def _dedupe_imports_in_text(
    text: str, declared: Set[str], seen_side_effect_imports: Set[str]
) -> Tuple[str, List[str]]:
    """Remove/trim import lines whose identifiers were already declared."""
    notes: List[str] = []
    out_lines: List[str] = []
    for line in text.split("\n"):
        if not _IMPORT_LINE_RE.match(line) or " from " not in f" {line} ":
            side_effect = _IMPORT_LINE_RE.match(line) and re.search(
                r"import\s+['\"]", line
            )
            if side_effect:
                normalized = line.strip()
                if normalized in seen_side_effect_imports:
                    notes.append(f"dropped duplicate side-effect import: {normalized}")
                    continue
                seen_side_effect_imports.add(normalized)
            out_lines.append(line)
            continue

        identifiers = _import_line_identifiers(line)
        if not identifiers:
            out_lines.append(line)
            continue
        duplicated = [name for name in identifiers if name in declared]
        fresh = [name for name in identifiers if name not in declared]
        declared.update(fresh)
        if not duplicated:
            out_lines.append(line)
            continue
        if not fresh:
            notes.append(
                f"dropped duplicate import of {{{', '.join(duplicated)}}}"
            )
            continue
        named_match = _NAMED_IMPORTS_RE.search(line)
        if named_match:
            rewritten = (
                line[: named_match.start()]
                + "{ "
                + ", ".join(fresh)
                + " }"
                + line[named_match.end() :]
            )
            notes.append(
                f"removed already-imported {{{', '.join(duplicated)}}} from import"
            )
            out_lines.append(rewritten)
        else:
            # Cannot safely trim non-named imports; keep the line untouched.
            out_lines.append(line)
    return "\n".join(out_lines), notes


def sanitize_runtime_imports(
    service_script: str, components_script: str
) -> Tuple[str, str, List[str]]:
    """Drop duplicate import declarations across the concatenated runtime
    scripts (service first, then components — the bundling order)."""
    declared: Set[str] = set()
    seen_side_effect: Set[str] = set()
    sanitized_service, service_notes = _dedupe_imports_in_text(
        service_script or "", declared, seen_side_effect
    )
    sanitized_components, component_notes = _dedupe_imports_in_text(
        components_script or "", declared, seen_side_effect
    )
    notes = [f"service_script: {note}" for note in service_notes] + [
        f"components_script: {note}" for note in component_notes
    ]
    return sanitized_service, sanitized_components, notes


def detect_duplicate_component_registrations(runtime_text: str) -> List[str]:
    """Return pfusch component names registered more than once."""
    names = _PFUSCH_REGISTRATION_RE.findall(runtime_text or "")
    counts: Dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return sorted(name for name, count in counts.items() if count > 1)


def enforce_runtime_script_integrity(
    payload: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """Auto-fix duplicate imports in *payload* (mutates) and report
    unfixable integrity issues.

    Returns ``(fix_notes, errors)`` — non-empty ``errors`` means the payload
    must be rejected (e.g. a component is registered twice)."""
    service = payload.get("service_script")
    components = payload.get("components_script")
    service_text = service if isinstance(service, str) else ""
    components_text = components if isinstance(components, str) else ""

    sanitized_service, sanitized_components, notes = sanitize_runtime_imports(
        service_text, components_text
    )
    if isinstance(service, str) and sanitized_service != service:
        payload["service_script"] = sanitized_service
    if isinstance(components, str) and sanitized_components != components:
        payload["components_script"] = sanitized_components

    duplicates = detect_duplicate_component_registrations(
        f"{sanitized_service}\n{sanitized_components}"
    )
    errors = [
        (
            f"component '{name}' is registered more than once via pfusch(...); "
            "modify the existing component with a 'replace' operation instead of "
            "adding another copy"
        )
        for name in duplicates
    ]
    return notes, errors
