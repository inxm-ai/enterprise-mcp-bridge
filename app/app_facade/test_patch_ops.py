import json
import os
from types import SimpleNamespace

import pytest

from app.app_facade.generated_service import GeneratedUIService, GeneratedUIStorage
from app.app_facade.generated_types import Scope
from app.app_facade.patch_ops import (
    PATCH_UPDATE_SCHEMA,
    apply_patch_operations,
    detect_duplicate_component_registrations,
    enforce_runtime_script_integrity,
    legacy_patch_to_operations,
    sanitize_runtime_imports,
)


class DummyTGIService:
    def __init__(self):
        self.llm_client = None
        self.prompt_service = None
        self.tool_service = None


DRAFT = {
    "html": {"page": "<html><body><div>old</div></body></html>", "snippet": "<div>old</div>"},
    "service_script": "export function getItems() { return svc.call('list_items'); }",
    "components_script": "pfusch('app-root', {}, (state) => [html.div('hello')]);",
    "test_script": "test('renders', () => {});",
    "dummy_data": "export const dummyData = {};",
    "metadata": {},
}


def test_apply_patch_operations_replace_and_append():
    candidate, errors = apply_patch_operations(
        DRAFT,
        [
            {
                "target": "components_script",
                "op": "replace",
                "search": "html.div('hello')",
                "content": "html.div('world')",
            },
            {
                "target": "test_script",
                "op": "append",
                "content": "test('new case', () => {});",
            },
        ],
    )
    assert errors == []
    assert "html.div('world')" in candidate["components_script"]
    assert "html.div('hello')" not in candidate["components_script"]
    assert candidate["test_script"].endswith("test('new case', () => {});")
    # original untouched
    assert "html.div('hello')" in DRAFT["components_script"]


def test_apply_patch_operations_replace_all_and_html_targets():
    candidate, errors = apply_patch_operations(
        {"html": {"page": "a b a", "snippet": "x"}},
        [
            {"target": "html_page", "op": "replace", "search": "a", "content": "c", "replace_all": True},
            {"target": "html_snippet", "op": "set", "content": "<div>new</div>"},
        ],
    )
    assert errors == []
    assert candidate["html"]["page"] == "c b c"
    assert candidate["html"]["snippet"] == "<div>new</div>"


def test_apply_patch_operations_set_creates_missing_target():
    candidate, errors = apply_patch_operations(
        {"html": {}},
        [{"target": "test_script", "op": "set", "content": "test('x', () => {});"}],
    )
    assert errors == []
    assert candidate["test_script"] == "test('x', () => {});"


def test_apply_patch_operations_is_all_or_nothing():
    candidate, errors = apply_patch_operations(
        DRAFT,
        [
            {
                "target": "components_script",
                "op": "replace",
                "search": "html.div('hello')",
                "content": "html.div('world')",
            },
            {
                "target": "service_script",
                "op": "replace",
                "search": "does-not-exist-anywhere",
                "content": "whatever",
            },
        ],
    )
    assert candidate is None
    assert len(errors) == 1
    assert "search text not found" in errors[0]
    assert "service_script" in errors[0]


def test_apply_patch_operations_rejects_ambiguous_search_without_replace_all():
    payload = {"components_script": "html.div('x'); html.div('x'); html.div('y');"}
    candidate, errors = apply_patch_operations(
        payload,
        [
            {
                "target": "components_script",
                "op": "replace",
                "search": "html.div('x');",
                "content": "html.div('z');",
            }
        ],
    )
    assert candidate is None
    assert len(errors) == 1
    assert "ambiguous search text matches 2 times" in errors[0]
    assert "components_script" in errors[0]
    # payload itself must remain untouched — no silent first-match patch
    assert payload["components_script"] == "html.div('x'); html.div('x'); html.div('y');"


def test_apply_patch_operations_ambiguous_search_allowed_with_replace_all():
    payload = {"components_script": "html.div('x'); html.div('x'); html.div('y');"}
    candidate, errors = apply_patch_operations(
        payload,
        [
            {
                "target": "components_script",
                "op": "replace",
                "search": "html.div('x');",
                "content": "html.div('z');",
                "replace_all": True,
            }
        ],
    )
    assert errors == []
    assert candidate["components_script"] == "html.div('z'); html.div('z'); html.div('y');"


def test_apply_patch_operations_rejects_empty_and_invalid():
    candidate, errors = apply_patch_operations(DRAFT, [])
    assert candidate is None
    assert errors == ["no_operations_provided"]

    candidate, errors = apply_patch_operations(
        DRAFT,
        [
            {"target": "nope", "op": "set", "content": "x"},
            {"target": "test_script", "op": "replace", "content": "x"},
        ],
    )
    assert candidate is None
    assert any("unknown_target" in e for e in errors)
    assert any("non-empty 'search'" in e for e in errors)


def test_legacy_patch_to_operations_converts_whole_file_keys():
    ops = legacy_patch_to_operations(
        {
            "html": {"page": "<html>new</html>", "snippet": ""},
            "components_script": "new comp",
            "metadata": {"x": 1},
        }
    )
    assert {"target": "html_page", "op": "set", "content": "<html>new</html>"} in ops
    assert {"target": "components_script", "op": "set", "content": "new comp"} in ops
    # empty snippet and metadata are not converted into operations
    assert all(op["target"] != "html_snippet" for op in ops)


def test_patch_update_schema_shape():
    patch_props = PATCH_UPDATE_SCHEMA["properties"]["patch"]
    assert "operations" in patch_props["properties"]
    assert patch_props["required"] == ["operations"]


@pytest.mark.asyncio
async def test_attempt_patch_update_applies_operations_and_passes_feedback():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    captured = {}

    async def fake_non_stream_completion(request, _token, _span):
        captured["request"] = request
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "patch": {
                                    "operations": [
                                        {
                                            "target": "components_script",
                                            "op": "replace",
                                            "search": "html.div('hello')",
                                            "content": "html.div('patched')",
                                        }
                                    ]
                                }
                            }
                        )
                    }
                }
            ]
        }

    service.tgi_service.llm_client = SimpleNamespace(
        non_stream_completion=fake_non_stream_completion
    )
    # no test_script execution path needed
    service._run_tests = lambda *_a, **_k: (True, "ok")

    result = await service.conversational_service._attempt_patch_update(
        scope=Scope(kind="user", identifier="u1"),
        ui_id="ui1",
        name="dash",
        draft_payload=dict(DRAFT),
        user_message="change greeting",
        assistant_message="I will change it",
        access_token=None,
        previous_metadata={},
        failure_feedback="op[0]: search text not found in 'components_script'",
    )

    assert result is not None
    assert "html.div('patched')" in result["payload"]["components_script"]

    user_payload = json.loads(captured["request"].messages[1].content)
    assert "previous_attempt_failure" in user_payload
    assert "search text not found" in user_payload["previous_attempt_failure"]
    # operations vocabulary must be in the system prompt
    assert "'replace'" in captured["request"].messages[0].content


@pytest.mark.asyncio
async def test_attempt_patch_update_records_apply_failure_reason():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    async def fake_non_stream_completion(_request, _token, _span):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "patch": {
                                    "operations": [
                                        {
                                            "target": "components_script",
                                            "op": "replace",
                                            "search": "not-in-file",
                                            "content": "x",
                                        }
                                    ]
                                }
                            }
                        )
                    }
                }
            ]
        }

    service.tgi_service.llm_client = SimpleNamespace(
        non_stream_completion=fake_non_stream_completion
    )

    result = await service.conversational_service._attempt_patch_update(
        scope=Scope(kind="user", identifier="u1"),
        ui_id="ui1",
        name="dash",
        draft_payload=dict(DRAFT),
        user_message="change greeting",
        assistant_message="",
        access_token=None,
        previous_metadata={},
    )

    assert result is None
    assert service._last_patch_failure_reason.startswith("patch_apply_failed")
    assert "search text not found" in service._last_patch_failure_reason


PFUSCH_IMPORT = (
    "import { pfusch, html, css, script } from "
    "'https://matthiaskainer.github.io/pfusch/pfusch.min.js';"
)


def test_sanitize_runtime_imports_drops_exact_duplicate():
    components = f"{PFUSCH_IMPORT}\npfusch('a-b', {{}}, () => []);\n{PFUSCH_IMPORT}\npfusch('c-d', {{}}, () => []);"
    service, sanitized, notes, conflicts = sanitize_runtime_imports("", components)
    assert sanitized.count("import { pfusch") == 1
    assert "pfusch('a-b'" in sanitized and "pfusch('c-d'" in sanitized
    assert notes
    assert conflicts == []


def test_sanitize_runtime_imports_across_service_and_components():
    service = f"{PFUSCH_IMPORT}\nexport function x() {{}}"
    components = f"{PFUSCH_IMPORT}\npfusch('a-b', {{}}, () => []);"
    sanitized_service, sanitized_components, notes, conflicts = sanitize_runtime_imports(
        service, components
    )
    assert sanitized_service == service
    assert "import" not in sanitized_components.split("\n")[0]
    assert "pfusch('a-b'" in sanitized_components
    assert notes
    assert conflicts == []


def test_sanitize_runtime_imports_trims_partial_overlap():
    service = "import { pfusch } from './pfusch.js';"
    components = "import { pfusch, html } from './pfusch.js';\npfusch('a-b');"
    _s, sanitized_components, notes, conflicts = sanitize_runtime_imports(
        service, components
    )
    first_line = sanitized_components.split("\n")[0]
    assert "html" in first_line and " pfusch," not in first_line
    assert notes
    assert conflicts == []


def test_sanitize_runtime_imports_keeps_distinct_imports():
    service = "import { a } from './x.js';"
    components = "import { b } from './y.js';\nimport './side.js';"
    s, c, notes, conflicts = sanitize_runtime_imports(service, components)
    assert s == service
    assert c == components
    assert notes == []
    assert conflicts == []


def test_sanitize_runtime_imports_does_not_drop_same_name_different_source():
    # Regression for a silent-corruption bug: 'format' bound from two
    # different modules must NOT be treated as a redundant duplicate — the
    # second import must survive untouched so components' reference to
    # `format` keeps resolving to its own module.
    service = "import { format } from './dates.js';"
    components = "import { format, parse } from './numbers.js';\nparse(format(1));"
    sanitized_service, sanitized_components, notes, conflicts = sanitize_runtime_imports(
        service, components
    )
    assert sanitized_service == service
    assert sanitized_components == components  # left completely untouched
    assert len(conflicts) == 1
    assert "'format'" in conflicts[0]
    assert "./dates.js" in conflicts[0] and "./numbers.js" in conflicts[0]


def test_sanitize_runtime_imports_drops_duplicate_default_in_mixed_import():
    # Regression: a duplicated default binding inside a default+named import
    # must be removed too, not just the named-import portion.
    service = "import React from 'react';"
    components = "import React, { useMemo } from 'react';\nuseMemo(() => React.createElement('div'));"
    sanitized_service, sanitized_components, notes, conflicts = sanitize_runtime_imports(
        service, components
    )
    assert sanitized_service == service
    first_line = sanitized_components.split("\n")[0]
    assert first_line == "import { useMemo } from 'react';"
    assert conflicts == []
    assert notes


def test_sanitize_runtime_imports_drops_exact_duplicate_default_only():
    service = "import Foo from './foo.js';"
    components = "import Foo from './foo.js';\nFoo();"
    sanitized_service, sanitized_components, notes, conflicts = sanitize_runtime_imports(
        service, components
    )
    assert sanitized_service == service
    assert "import" not in sanitized_components.split("\n")[0]
    assert conflicts == []
    assert notes


def test_detect_duplicate_component_registrations():
    text = (
        "pfusch('issue-list', {}, () => []);\n"
        "pfusch('pr-list', {}, () => []);\n"
        "pfusch('issue-list', {}, () => []);"
    )
    assert detect_duplicate_component_registrations(text) == ["issue-list"]
    assert detect_duplicate_component_registrations("pfusch('one', {})") == []


def test_enforce_runtime_script_integrity_fixes_imports_and_flags_duplicates():
    payload = {
        "service_script": PFUSCH_IMPORT,
        "components_script": (
            f"{PFUSCH_IMPORT}\n"
            "pfusch('issue-list', {}, () => []);\n"
            "pfusch('issue-list', {}, () => []);"
        ),
    }
    notes, errors = enforce_runtime_script_integrity(payload)
    assert notes  # duplicate import removed
    assert "import" not in payload["components_script"].split("\n")[0]
    assert len(errors) == 1
    assert "issue-list" in errors[0]


def test_enforce_runtime_script_integrity_rejects_import_conflict():
    payload = {
        "service_script": "import { format } from './dates.js';",
        "components_script": "import { format } from './numbers.js';\nformat(1);",
    }
    original_components = payload["components_script"]
    notes, errors = enforce_runtime_script_integrity(payload)
    # left untouched — conflict is reported, not silently resolved either way
    assert payload["components_script"] == original_components
    assert len(errors) == 1
    assert "import conflict" in errors[0]
    assert "'format'" in errors[0]


@pytest.mark.asyncio
async def test_attempt_patch_update_rejects_duplicate_component_registration():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    duplicate_component = (
        "\npfusch('app-root', {}, (state) => [html.div('copy')]);"
    )

    async def fake_non_stream_completion(_request, _token, _span):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "patch": {
                                    "operations": [
                                        {
                                            "target": "components_script",
                                            "op": "append",
                                            "content": duplicate_component,
                                        }
                                    ]
                                }
                            }
                        )
                    }
                }
            ]
        }

    service.tgi_service.llm_client = SimpleNamespace(
        non_stream_completion=fake_non_stream_completion
    )

    result = await service.conversational_service._attempt_patch_update(
        scope=Scope(kind="user", identifier="u1"),
        ui_id="ui1",
        name="dash",
        draft_payload=dict(DRAFT),
        user_message="add a copy of the root",
        assistant_message="",
        access_token=None,
        previous_metadata={},
    )

    assert result is None
    assert service._last_patch_failure_reason.startswith("patch_integrity_failed")
    assert "app-root" in service._last_patch_failure_reason


def test_dummy_data_covers_tools():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    module = (
        "export const dummyData = "
        + json.dumps({"list_items": [{"id": 1}], "get_weather": {"temperature_c": 20}})
        + ";\nexport const dummyDataSchemaHints = {};"
    )
    tools = [
        {"function": {"name": "list_items"}},
        {"function": {"name": "get_weather"}},
        {"function": {"name": "describe_tool"}},  # excluded from sampling
    ]
    assert service.tool_sampler.dummy_data_covers_tools(module, tools) is True

    tools.append({"function": {"name": "delete_item"}})
    assert service.tool_sampler.dummy_data_covers_tools(module, tools) is False
    assert service.tool_sampler.dummy_data_covers_tools("", tools) is False


def test_pipeline_reuses_dummy_data_only_when_covered():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    module = (
        "export const dummyData = "
        + json.dumps({"list_items": []})
        + ";\nexport const dummyDataSchemaHints = {};"
    )
    previous = {"current": {"dummy_data": module}}
    tools = [{"function": {"name": "list_items"}}]

    assert (
        service.generation_pipeline._reusable_dummy_data(
            previous=previous, allowed_tools=tools
        )
        == module
    )
    # uncovered tool -> regenerate
    assert (
        service.generation_pipeline._reusable_dummy_data(
            previous=previous,
            allowed_tools=tools + [{"function": {"name": "get_weather"}}],
        )
        is None
    )
    # fresh runtime observations -> regenerate
    assert (
        service.generation_pipeline._reusable_dummy_data(
            previous=previous,
            allowed_tools=tools,
            runtime_context={
                "entries": [{"tool": "list_items", "response_payload": {"a": 1}}]
            },
        )
        is None
    )
    # create flow (no previous) -> regenerate
    assert (
        service.generation_pipeline._reusable_dummy_data(
            previous=None, allowed_tools=tools
        )
        is None
    )
