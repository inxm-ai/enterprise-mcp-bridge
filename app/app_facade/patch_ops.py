"""Operation-based patching for conversational UI edits.

Instead of asking the model to re-emit whole files inside a JSON payload
(slow, token-hungry, and brittle because of JSON escaping), the patch
planner returns a list of small edit operations that are applied to the
current draft payload server-side:

* ``replace`` – replace an exact ``search`` string with ``content``
* ``insert_before`` / ``insert_after`` – add content beside an exact anchor
* ``append``  – append ``content`` to the end of the target
* ``set``     – replace the whole target with ``content`` (new/small files)

Application is all-or-nothing: if any operation fails, the draft stays
untouched and the collected errors are fed back into the next patch
attempt so the model can correct its search strings.
"""

import copy
import json
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
PATCH_RESPONSE_ENVELOPES = ("result", "data", "output")
PATCH_RESPONSE_MAX_JSON_LAYERS = 3

PATCH_UPDATE_SCHEMA = {
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
                        "enum": [
                            "replace",
                            "insert_before",
                            "insert_after",
                            "append",
                            "set",
                        ],
                    },
                    "search": {
                        "type": "string",
                        "description": (
                            "Exact text copied verbatim from the current "
                            "target (required for replace/insert operations)"
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Replacement or inserted text, appended text "
                            "(append), or full new content (set)"
                        ),
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": (
                            "Replace every occurrence of search (default false)"
                        ),
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


def _whitespace_normalized_spans(text: str, search: str) -> List[Tuple[int, int]]:
    """Find search spans that differ only in whitespace formatting."""
    search_parts = re.split(r"\s+", search.strip())
    if not search_parts or any(not part for part in search_parts):
        return []
    pattern = r"\s+".join(re.escape(part) for part in search_parts)
    return [match.span() for match in re.finditer(pattern, text)]


def _shared_unique_suffix_anchor_span(
    text: str, search: str, content: str
) -> Optional[Tuple[int, int]]:
    """Find a shared closing-line anchor for a missing replacement block.

    Models sometimes describe a replacement as ``old block + closing anchor``
    even when the old block is absent, while correctly preserving the exact
    closing anchor in the replacement. If that complete line is shared by
    search/content and occurs exactly once, replacing just the anchor safely
    turns the operation into an insertion before that boundary.
    """
    search_lines = search.splitlines(keepends=True)
    content_lines = content.splitlines(keepends=True)
    shared_reversed: List[str] = []
    for search_line, content_line in zip(
        reversed(search_lines), reversed(content_lines)
    ):
        if search_line != content_line:
            break
        shared_reversed.append(search_line)
    if not shared_reversed:
        return None

    anchor = "".join(reversed(shared_reversed))
    if not anchor.strip():
        return None
    missing_prefix = search[: -len(anchor)]
    inserted_prefix = content[: -len(anchor)]
    if not missing_prefix.strip() or not inserted_prefix.strip():
        return None

    start = text.find(anchor)
    if start < 0 or text.find(anchor, start + 1) >= 0:
        return None
    return start, start + len(anchor)


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


def normalize_patch_response(
    parsed: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Normalize safe patch response variants emitted by different models.

    The JSON schema asks for ``{"patch": {...}}``, but some compatible model
    endpoints strip that outer object or add a generic result envelope. Keep
    the accepted variants explicit so arbitrary JSON is never treated as code.
    """
    return _normalize_patch_response(parsed, json_layer=0)


def _normalize_patch_response(
    parsed: Any,
    *,
    json_layer: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if isinstance(parsed, str):
        if json_layer >= PATCH_RESPONSE_MAX_JSON_LAYERS:
            return None, (
                "embedded_patch_json_exceeded_"
                f"{PATCH_RESPONSE_MAX_JSON_LAYERS}_layers"
            )
        try:
            decoded = json.loads(parsed)
        except json.JSONDecodeError as exc:
            return None, (
                "patch_type=str but value is not JSON: "
                f"{exc.msg} at position {exc.pos}"
            )
        return _normalize_patch_response(decoded, json_layer=json_layer + 1)

    if not isinstance(parsed, dict):
        return None, f"root_type={type(parsed).__name__}; expected an object"

    enveloped_candidates = tuple(
        parsed[key]
        for key in PATCH_RESPONSE_ENVELOPES
        if isinstance(parsed.get(key), dict)
    )
    candidates = (parsed, *enveloped_candidates)

    for candidate in candidates:
        patch = candidate.get("patch")
        if isinstance(patch, dict):
            return patch, None
        if isinstance(patch, list):
            return {"operations": patch}, None
        if isinstance(patch, str):
            return _normalize_patch_response(patch, json_layer=json_layer)

        operations = candidate.get("operations")
        if isinstance(operations, list):
            normalized = {"operations": operations}
            metadata = candidate.get("metadata")
            if isinstance(metadata, dict):
                normalized["metadata"] = metadata
            return normalized, None

        if all(key in candidate for key in ("target", "op", "content")):
            return {"operations": [candidate]}, None

        if legacy_patch_to_operations(candidate):
            return candidate, None

    returned_keys = sorted(str(key) for key in parsed.keys())
    keys_preview = ",".join(returned_keys[:8]) or "<empty>"
    invalid_patch_detail = (
        f"; patch_type={type(parsed.get('patch')).__name__}"
        if "patch" in parsed
        else ""
    )
    return (
        None,
        f"top_level_keys={keys_preview}{invalid_patch_detail}; "
        "expected 'patch', 'operations', "
        "a single operation, or a supported result envelope",
    )


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
        if op in {"replace", "insert_before", "insert_after"}:
            search = op_raw.get("search")
            if not isinstance(search, str) or not search:
                errors.append(
                    f"op[{index}]: {op} requires a non-empty 'search' string "
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
                normalized_spans = _whitespace_normalized_spans(current, search)
                if len(normalized_spans) == 1:
                    start, end = normalized_spans[0]
                    if op == "insert_before":
                        updated = f"{current[:start]}{content}{current[start:]}"
                    elif op == "insert_after":
                        updated = f"{current[:end]}{content}{current[end:]}"
                    else:
                        updated = f"{current[:start]}{content}{current[end:]}"
                    _write_target(candidate, target, updated)
                    logger.info(
                        "[GeneratedUI] Applied whitespace-normalized %s "
                        "to %s for op[%s]",
                        op,
                        target,
                        index,
                    )
                    continue
                if len(normalized_spans) > 1:
                    errors.append(
                        f"op[{index}]: whitespace-normalized search text matches "
                        f"{len(normalized_spans)} times in '{target}': "
                        f"'{_preview(search)}'. Include more surrounding context."
                    )
                    continue
                if op == "replace":
                    suffix_span = _shared_unique_suffix_anchor_span(
                        current, search, content
                    )
                    if suffix_span is not None:
                        start, end = suffix_span
                        updated = f"{current[:start]}{content}{current[end:]}"
                        _write_target(candidate, target, updated)
                        logger.info(
                            "[GeneratedUI] Rebased missing replacement block "
                            "onto its unique shared suffix anchor in %s for op[%s]",
                            target,
                            index,
                        )
                        continue
                errors.append(
                    f"op[{index}]: search text not found in '{target}': "
                    f"'{_preview(search)}'. Copy the text verbatim from the "
                    "current file content."
                )
                continue
            replace_all = bool(op_raw.get("replace_all"))
            if occurrences > 1 and not replace_all:
                errors.append(
                    f"op[{index}]: ambiguous search text matches {occurrences} "
                    f"times in '{target}': '{_preview(search)}'. Include more "
                    "surrounding context to make it match only once, or set "
                    "replace_all=true if every occurrence should change."
                )
                continue
            if op == "insert_before":
                if replace_all:
                    updated = current.replace(search, f"{content}{search}")
                else:
                    updated = current.replace(search, f"{content}{search}", 1)
            elif op == "insert_after":
                if replace_all:
                    updated = current.replace(search, f"{search}{content}")
                else:
                    updated = current.replace(search, f"{search}{content}", 1)
            elif replace_all:
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
# Captures the binding clause between `import` and `from '<source>'` so the
# module specifier is available for source-aware dedup (see
# ``sanitize_runtime_imports`` docstring for why the specifier matters).
_IMPORT_CLAUSE_RE = re.compile(
    r"^(?P<prefix>[ \t]*import\s+)(?P<clause>.+?)\s+from\s+"
    r"(?P<quote>['\"])(?P<source>[^'\"]+)(?P=quote)(?P<suffix>\s*;?\s*)$"
)
_CLAUSE_NAMESPACE_RE = re.compile(r"\*\s*as\s+([A-Za-z_$][\w$]*)")
_CLAUSE_NAMED_RE = re.compile(r"\{([^}]*)\}")
_CLAUSE_DEFAULT_RE = re.compile(r"^([A-Za-z_$][\w$]*)")
_PFUSCH_REGISTRATION_RE = re.compile(r"\bpfusch\(\s*['\"]([A-Za-z][\w-]*)['\"]")


def _parse_import_clause(
    clause: str,
) -> Tuple[Optional[str], Optional[str], List[Tuple[str, str]]]:
    """Split an import clause into ``(default_id, namespace_id, named_pairs)``.

    ``named_pairs`` is a list of ``(raw_text, local_name)`` so the original
    ``orig as alias`` text can be reconstructed verbatim when rebuilding a
    trimmed clause.
    """
    clause = clause.strip()
    namespace_match = _CLAUSE_NAMESPACE_RE.search(clause)
    namespace_id = namespace_match.group(1) if namespace_match else None

    named_pairs: List[Tuple[str, str]] = []
    named_match = _CLAUSE_NAMED_RE.search(clause)
    if named_match:
        for part in named_match.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            local = part.split(" as ")[-1].strip()
            if local:
                named_pairs.append((part, local))

    default_id: Optional[str] = None
    if not clause.startswith("{") and not clause.startswith("*"):
        default_match = _CLAUSE_DEFAULT_RE.match(clause)
        if default_match:
            default_id = default_match.group(1)

    return default_id, namespace_id, named_pairs


def _rebuild_import_clause(
    default_id: Optional[str],
    namespace_id: Optional[str],
    named_pairs: List[Tuple[str, str]],
) -> str:
    parts: List[str] = []
    if default_id:
        parts.append(default_id)
    if namespace_id:
        parts.append(f"* as {namespace_id}")
    if named_pairs:
        parts.append("{ " + ", ".join(raw for raw, _local in named_pairs) + " }")
    return ", ".join(parts)


def _dedupe_imports_in_text(
    text: str,
    declared: Dict[str, str],
    seen_exact_lines: Set[str],
    conflicts: List[str],
) -> Tuple[str, List[str]]:
    """Remove import bindings that exactly duplicate an already-declared
    ``(identifier, source module)`` pair; leave genuine cross-module naming
    conflicts untouched but recorded in *conflicts* (see
    ``sanitize_runtime_imports``)."""
    notes: List[str] = []
    out_lines: List[str] = []
    for line in text.split("\n"):
        if not _IMPORT_LINE_RE.match(line):
            out_lines.append(line)
            continue

        clause_match = _IMPORT_CLAUSE_RE.match(line)
        if not clause_match:
            # Side-effect import (`import 'x';`) or a shape this sanitizer
            # doesn't parse (e.g. multi-line braces) — only dedupe on exact
            # text match, never guess at identifiers without a known source.
            normalized = line.strip()
            if normalized.startswith("import"):
                if normalized in seen_exact_lines:
                    notes.append(f"dropped duplicate import line: {normalized}")
                    continue
                seen_exact_lines.add(normalized)
            out_lines.append(line)
            continue

        source = clause_match.group("source")
        default_id, namespace_id, named_pairs = _parse_import_clause(
            clause_match.group("clause")
        )
        bindings: List[Tuple[str, str]] = []  # (kind, name)
        if default_id:
            bindings.append(("default", default_id))
        if namespace_id:
            bindings.append(("namespace", namespace_id))
        bindings.extend(("named", local) for _raw, local in named_pairs)
        if not bindings:
            out_lines.append(line)
            continue

        # A genuine conflict is the SAME identifier bound to a DIFFERENT
        # source module. Silently keeping either side would change which
        # module a reference resolves to, so leave the line untouched and
        # surface it as an unfixable error instead of guessing.
        line_has_conflict = False
        for _kind, name in bindings:
            prior_source = declared.get(name)
            if prior_source is not None and prior_source != source:
                conflicts.append(
                    f"identifier '{name}' imported from both '{prior_source}' "
                    f"and '{source}'"
                )
                line_has_conflict = True
        if line_has_conflict:
            for _kind, name in bindings:
                declared.setdefault(name, source)
            out_lines.append(line)
            continue

        # No conflict on this line — safe to drop any binding that exactly
        # repeats an already-declared (identifier, source) pair, regardless
        # of whether it's the default, namespace, or a named binding.
        dropped: List[str] = []
        kept_default: Optional[str] = None
        kept_namespace: Optional[str] = None
        kept_named: List[Tuple[str, str]] = []
        if default_id is not None:
            if default_id in declared:
                dropped.append(default_id)
            else:
                declared[default_id] = source
                kept_default = default_id
        if namespace_id is not None:
            if namespace_id in declared:
                dropped.append(namespace_id)
            else:
                declared[namespace_id] = source
                kept_namespace = namespace_id
        for raw, local in named_pairs:
            if local in declared:
                dropped.append(local)
            else:
                declared[local] = source
                kept_named.append((raw, local))

        if not dropped:
            out_lines.append(line)
            continue

        if not kept_default and not kept_namespace and not kept_named:
            notes.append(
                f"dropped duplicate import of {{{', '.join(dropped)}}} from "
                f"'{source}'"
            )
            continue

        new_clause = _rebuild_import_clause(kept_default, kept_namespace, kept_named)
        rebuilt = (
            f"{clause_match.group('prefix')}{new_clause} from "
            f"{clause_match.group('quote')}{source}{clause_match.group('quote')}"
            f"{clause_match.group('suffix')}"
        )
        notes.append(
            f"removed already-imported {{{', '.join(dropped)}}} from '{source}' import"
        )
        out_lines.append(rebuilt)
    return "\n".join(out_lines), notes


def sanitize_runtime_imports(
    service_script: str, components_script: str
) -> Tuple[str, str, List[str], List[str]]:
    """Drop duplicate import declarations across the concatenated runtime
    scripts (service first, then components — the bundling order).

    Only an import that repeats an IDENTICAL ``(identifier, source module)``
    pair is treated as a safe, redundant duplicate and removed. When the
    same local identifier is imported from two *different* source modules
    (e.g. ``{ format }`` from both ``./dates.js`` and ``./numbers.js``),
    that's a genuine naming conflict, not a redundant duplicate — silently
    dropping either side would silently change which module later references
    resolve to. Conflicts are left untouched in the returned scripts and
    reported in ``conflicts`` instead, so the caller can reject the payload
    and ask the model to alias one of the imports.

    Returns ``(sanitized_service, sanitized_components, notes, conflicts)``.
    """
    declared: Dict[str, str] = {}
    seen_exact_lines: Set[str] = set()
    conflicts: List[str] = []
    sanitized_service, service_notes = _dedupe_imports_in_text(
        service_script or "", declared, seen_exact_lines, conflicts
    )
    sanitized_components, component_notes = _dedupe_imports_in_text(
        components_script or "", declared, seen_exact_lines, conflicts
    )
    notes = [f"service_script: {note}" for note in service_notes] + [
        f"components_script: {note}" for note in component_notes
    ]
    return sanitized_service, sanitized_components, notes, conflicts


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

    sanitized_service, sanitized_components, notes, conflicts = (
        sanitize_runtime_imports(service_text, components_text)
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
    errors.extend(
        f"import conflict: {conflict}; alias one of the imports to a distinct "
        "local name so both modules can be referenced"
        for conflict in conflicts
    )
    return notes, errors
