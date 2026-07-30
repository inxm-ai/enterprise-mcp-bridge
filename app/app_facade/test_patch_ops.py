import json
import os
from types import SimpleNamespace

import pytest

from app.app_facade.generated_service import GeneratedUIService, GeneratedUIStorage
from app.app_facade.conversational_service import _sync_patched_template_script
from app.app_facade.generated_types import Scope
from app.app_facade.patch_ops import (
    PATCH_UPDATE_SCHEMA,
    apply_patch_operations,
    detect_duplicate_component_registrations,
    enforce_runtime_script_integrity,
    legacy_patch_to_operations,
    normalize_patch_response,
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


def test_sync_patched_template_script_updates_unchanged_legacy_mirror():
    original = "import { pfusch } from './pfusch.js';\npfusch('saga-details');"
    patched = (
        "import { pfusch } from './pfusch.js';\n"
        "const buildGraphLayout = () => ({});\n"
        "pfusch('saga-details');"
    )
    draft = {
        "components_script": original,
        "template_parts": {"script": original, "title": "Saga Details"},
    }
    candidate = {
        "components_script": patched,
        "template_parts": {"script": original, "title": "Saga Details"},
    }

    assert _sync_patched_template_script(candidate, draft) is True
    assert candidate["template_parts"]["script"] == patched
    assert candidate["template_parts"]["title"] == "Saga Details"


def test_sync_patched_template_script_preserves_independently_changed_script():
    draft = {
        "components_script": "old components",
        "template_parts": {"script": "old template"},
    }
    candidate = {
        "components_script": "new components",
        "template_parts": {"script": "independently patched template"},
    }

    assert _sync_patched_template_script(candidate, draft) is False
    assert candidate["template_parts"]["script"] == "independently patched template"


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


def test_apply_patch_operations_accepts_unique_whitespace_only_difference():
    payload = {
        "components_script": (
            ".events-table {\n"
            "  width: 100%;\n"
            "  border-collapse: collapse;\n"
            "}\n"
        )
    }
    candidate, errors = apply_patch_operations(
        payload,
        [
            {
                "target": "components_script",
                "op": "replace",
                "search": (
                    ".events-table { width: 100%; "
                    "border-collapse: collapse; }"
                ),
                "content": ".graph-wrap { overflow-x: auto; }",
            }
        ],
    )

    assert errors == []
    assert candidate is not None
    assert ".graph-wrap { overflow-x: auto; }" in candidate["components_script"]
    assert ".events-table" not in candidate["components_script"]


def test_apply_patch_operations_inserts_around_exact_anchor():
    payload = {"components_script": "before\n  `,\nafter"}
    candidate, errors = apply_patch_operations(
        payload,
        [
            {
                "target": "components_script",
                "op": "insert_before",
                "search": "  `,",
                "content": "    .graph-wrap { overflow-x: auto; }\n",
            },
            {
                "target": "components_script",
                "op": "insert_after",
                "search": "before",
                "content": "\ninserted",
            },
        ],
    )

    assert errors == []
    assert candidate is not None
    assert "before\ninserted\n" in candidate["components_script"]
    assert ".graph-wrap { overflow-x: auto; }\n  `," in candidate[
        "components_script"
    ]


def test_apply_patch_operations_rebases_missing_css_block_on_shared_suffix():
    payload = {
        "components_script": (
            "css`\n"
            "    .empty-state { text-align: center; }\n"
            "  `,\n"
            "html.div('content')"
        )
    }
    candidate, errors = apply_patch_operations(
        payload,
        [
            {
                "target": "components_script",
                "op": "replace",
                "search": (
                    "    .events-toggle:hover { "
                    "background: var(--surface-secondary, #f9fafb); }\n"
                    "  `,"
                ),
                "content": (
                    "    .graph-wrap { overflow-x: auto; }\n"
                    "    .graph-node { fill: var(--surface-primary, #fff); }\n"
                    "  `,"
                ),
            }
        ],
    )

    assert errors == []
    assert candidate is not None
    assert ".graph-wrap { overflow-x: auto; }" in candidate["components_script"]
    assert ".events-toggle:hover" not in candidate["components_script"]
    assert candidate["components_script"].count("  `,") == 1


def test_apply_patch_operations_does_not_rebase_ambiguous_shared_suffix():
    payload = {"components_script": "css`\n  `,\ncss`\n  `,"}
    candidate, errors = apply_patch_operations(
        payload,
        [
            {
                "target": "components_script",
                "op": "replace",
                "search": ".missing { color: red; }\n  `,",
                "content": ".graph { color: blue; }\n  `,",
            }
        ],
    )

    assert candidate is None
    assert len(errors) == 1
    assert "search text not found" in errors[0]


def test_apply_patch_operations_rejects_ambiguous_whitespace_match():
    payload = {
        "components_script": (
            ".item {\n  color: red;\n}\n"
            ".item {\n    color: red;\n}\n"
        )
    }
    candidate, errors = apply_patch_operations(
        payload,
        [
            {
                "target": "components_script",
                "op": "replace",
                "search": ".item { color: red; }",
                "content": ".item { color: blue; }",
            }
        ],
    )

    assert candidate is None
    assert len(errors) == 1
    assert "whitespace-normalized search text matches 2 times" in errors[0]


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


@pytest.mark.parametrize(
    "response",
    [
        {
            "operations": [
                {
                    "target": "components_script",
                    "op": "replace",
                    "search": "saga.name",
                    "content": "saga.id",
                }
            ]
        },
        {
            "patch": [
                {
                    "target": "components_script",
                    "op": "replace",
                    "search": "saga.name",
                    "content": "saga.id",
                }
            ]
        },
        {
            "result": {
                "patch": {
                    "operations": [
                        {
                            "target": "components_script",
                            "op": "replace",
                            "search": "saga.name",
                            "content": "saga.id",
                        }
                    ]
                }
            }
        },
        {
            "target": "components_script",
            "op": "replace",
            "search": "saga.name",
            "content": "saga.id",
        },
        {
            "patch": json.dumps(
                {
                    "patch": {
                        "operations": [
                            {
                                "target": "components_script",
                                "op": "replace",
                                "search": "saga.name",
                                "content": "saga.id",
                            }
                        ]
                    }
                }
            )
            + "\n"
        },
    ],
)
def test_normalize_patch_response_accepts_supported_envelopes(response):
    patch, error = normalize_patch_response(response)

    assert error is None
    assert patch is not None
    assert patch["operations"][0]["content"] == "saga.id"


def test_normalize_patch_response_reports_unknown_shape():
    patch, error = normalize_patch_response({"not_patch": {"x": 1}})

    assert patch is None
    assert error is not None
    assert "top_level_keys=not_patch" in error
    assert "expected 'patch', 'operations'" in error


def test_normalize_patch_response_rejects_non_json_patch_string():
    patch, error = normalize_patch_response({"patch": "replace saga.name"})

    assert patch is None
    assert error is not None
    assert "patch_type=str but value is not JSON" in error


def test_patch_update_schema_shape():
    assert "operations" in PATCH_UPDATE_SCHEMA["properties"]
    assert "patch" not in PATCH_UPDATE_SCHEMA["properties"]
    assert PATCH_UPDATE_SCHEMA["required"] == ["operations"]


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
                                "operations": [
                                    {
                                        "target": "components_script",
                                        "op": "replace",
                                        "search": "html.div('hello')",
                                        "content": "html.div('patched')",
                                    }
                                ]
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
    assert "'insert_before'" in captured["request"].messages[0].content
    assert "Never use replace to add new code by inventing old text" in captured[
        "request"
    ].messages[0].content
    assert "Never wrap the response in a 'patch' property" in captured[
        "request"
    ].messages[0].content
    assert "DOM test stubs do not parse SVG" in captured["request"].messages[0].content
    assert "container's innerHTML/markup" in captured["request"].messages[0].content


@pytest.mark.asyncio
async def test_attempt_patch_update_does_not_append_stale_template_component():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    original = (
        f"{PFUSCH_IMPORT}\n\n"
        "pfusch('saga-details', {}, () => [html.div('events')]);"
    )
    draft = {
        **DRAFT,
        "components_script": original,
        "template_parts": {
            "title": "Saga Details",
            "styles": "",
            "html": "<saga-details></saga-details>",
            "script": original,
        },
    }

    async def fake_non_stream_completion(_request, _token, _span):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "operations": [
                                    {
                                        "target": "components_script",
                                        "op": "insert_before",
                                        "search": "pfusch('saga-details'",
                                        "content": (
                                            "const buildGraphLayout = () => ({});\n\n"
                                        ),
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }

    service.tgi_service.llm_client = SimpleNamespace(
        non_stream_completion=fake_non_stream_completion
    )

    tested_candidates = []

    def passing_tests(_service, components, _tests, _dummy):
        tested_candidates.append(components)
        return True, "ok"

    service._run_tests = passing_tests

    result = await service.conversational_service._attempt_patch_update(
        scope=Scope(kind="user", identifier="u1"),
        ui_id="saga-details",
        name="dash",
        draft_payload=draft,
        user_message="add graph helpers",
        assistant_message="I will add graph helpers above the existing component.",
        access_token=None,
        previous_metadata={},
    )

    assert result is not None
    components = result["payload"]["components_script"]
    assert components.count("import { pfusch") == 1
    assert components.count("pfusch('saga-details'") == 1
    assert components.count("buildGraphLayout") == 1
    assert result["payload"]["template_parts"]["script"] == components
    assert tested_candidates == [components]


@pytest.mark.asyncio
async def test_attempt_patch_update_syncs_template_after_iterative_repair(
    monkeypatch,
):
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    original = (
        f"{PFUSCH_IMPORT}\n\n"
        "pfusch('sagas-overview', {}, () => [html.div('overview')]);"
    )
    patched = original.replace(
        "html.div('overview')",
        "html.a({ href: '/details' }, 'overview')",
    )
    repaired = patched.replace(
        "html.a({ href: '/details' }, 'overview')",
        "html.a({ class: 'instance-saga-link', href: '/details' }, 'overview')",
    )
    draft = {
        **DRAFT,
        "components_script": original,
        "test_script": "test('renders instance link', () => {});",
        "template_parts": {
            "title": "Saga Overview",
            "styles": "",
            "html": "<sagas-overview></sagas-overview>",
            "script": original,
        },
    }

    async def fake_non_stream_completion(_request, _token, _span):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "operations": [
                                    {
                                        "target": "components_script",
                                        "op": "set",
                                        "content": patched,
                                    },
                                    {
                                        "target": "test_script",
                                        "op": "set",
                                        "content": (
                                            "test('renders linked instance', () => {});"
                                        ),
                                    },
                                ]
                            }
                        )
                    }
                }
            ]
        }

    service.tgi_service.llm_client = SimpleNamespace(
        non_stream_completion=fake_non_stream_completion
    )
    monkeypatch.setattr(
        service,
        "_run_tests",
        lambda *_args, **_kwargs: (False, "# pass 0\n# fail 1\n"),
    )

    async def repair_candidate(**_kwargs):
        return True, "", repaired, "test('fixed', () => {});", None, []

    monkeypatch.setattr(service, "_iterative_test_fix", repair_candidate)

    result = await service.conversational_service._attempt_patch_update(
        scope=Scope(kind="user", identifier="u1"),
        ui_id="ui1",
        name="dash",
        draft_payload=draft,
        user_message="link saga instances to details",
        assistant_message="I will add the existing detail link to each instance.",
        access_token=None,
        previous_metadata={},
    )

    assert result is not None
    payload = result["payload"]
    assert payload["components_script"] == repaired
    assert payload["template_parts"]["script"] == repaired
    service._normalise_payload(
        payload,
        Scope(kind="user", identifier="u1"),
        "ui1",
        "dash",
        "Publish conversational draft",
        {"metadata": {}, "current": draft},
    )
    assert payload["components_script"] == repaired
    assert payload["components_script"].count("pfusch('sagas-overview'") == 1


@pytest.mark.asyncio
async def test_attempt_patch_update_targeted_rewrite_replaces_complete_files():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    original_components = (
        f"{PFUSCH_IMPORT}\n"
        "pfusch('saga-details', {}, () => [html.table([])]);"
    )
    rewritten_components = (
        f"{PFUSCH_IMPORT}\n"
        "const buildGraphLayout = () => ({ nodes: [] });\n"
        "pfusch('saga-details', {}, () => [html.canvas({ class: 'graph-canvas' })]);"
    )
    rewritten_tests = "test('renders graph canvas', () => {});"
    draft = {
        **DRAFT,
        "components_script": original_components,
        "test_script": "test('renders events table', () => {});",
        "template_parts": {
            "title": "Saga Details",
            "styles": "",
            "html": "<saga-details></saga-details>",
            "script": original_components,
        },
    }
    captured = {}

    async def fake_non_stream_completion(request, _token, _span):
        captured["request"] = request
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "files": [
                                    {
                                        "target": "components_script",
                                        "content": rewritten_components,
                                    },
                                    {
                                        "target": "test_script",
                                        "content": rewritten_tests,
                                    },
                                ]
                            }
                        )
                    }
                }
            ]
        }

    service.tgi_service.llm_client = SimpleNamespace(
        non_stream_completion=fake_non_stream_completion
    )
    service._run_tests = lambda *_args, **_kwargs: (True, "ok")

    result = await service.conversational_service._attempt_patch_update(
        scope=Scope(kind="user", identifier="u1"),
        ui_id="saga-details",
        name="dash",
        draft_payload=draft,
        user_message="replace the events table with a canvas graph",
        assistant_message="I will preserve card-owned data and drawing.",
        access_token=None,
        previous_metadata={},
        failure_feedback="patch_apply_failed: fabricated CSS anchor",
        rewrite_files=True,
    )

    assert result is not None
    assert result["payload"]["components_script"] == rewritten_components
    assert result["payload"]["test_script"] == rewritten_tests
    assert result["payload"]["template_parts"]["script"] == rewritten_components
    request = captured["request"]
    assert request.response_format is not None
    assert "complete replacement content" in request.messages[0].content
    assert "query it with get(selector)" in request.messages[0].content
    assert "Do not invent unsupported query helpers" in request.messages[0].content
    user_payload = json.loads(request.messages[1].content)
    assert "do not return patch anchors" in user_payload["previous_attempt_failure"]


@pytest.mark.asyncio
async def test_attempt_patch_update_applies_double_encoded_model_response():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    operation = {
        "target": "components_script",
        "op": "replace",
        "search": "html.div('hello')",
        "content": "html.div('patched')",
    }
    double_encoded_response = json.dumps(
        {
            "patch": json.dumps(
                {"patch": {"operations": [operation]}},
                ensure_ascii=False,
            )
            + "\n"
        },
        ensure_ascii=False,
    )

    async def fake_non_stream_completion(_request, _token, _span):
        return {
            "choices": [{"message": {"content": double_encoded_response}}],
        }

    service.tgi_service.llm_client = SimpleNamespace(
        non_stream_completion=fake_non_stream_completion
    )
    service._run_tests = lambda *_args, **_kwargs: (True, "ok")

    result = await service.conversational_service._attempt_patch_update(
        scope=Scope(kind="user", identifier="u1"),
        ui_id="ui1",
        name="dash",
        draft_payload=dict(DRAFT),
        user_message="change greeting",
        assistant_message="I will change only the expression.",
        access_token=None,
        previous_metadata={},
    )

    assert result is not None
    assert "html.div('patched')" in result["payload"]["components_script"]
    assert service._last_patch_failure_reason is None


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


@pytest.mark.asyncio
async def test_attempt_patch_update_accepts_same_preexisting_test_failure(monkeypatch):
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
    existing_failure = (
        "TAP version 13\n"
        "not ok 1 - existing unrelated failure\n"
        "# tests 1\n"
        "# pass 0\n"
        "# fail 1\n"
    )
    test_runs = []

    def failing_tests(service_script, components_script, test_script, dummy_data):
        test_runs.append(components_script)
        return False, existing_failure

    async def fail_fix_loop(**_kwargs):  # pragma: no cover - defensive
        raise AssertionError("A non-regressing patch must not enter the fixer")

    monkeypatch.setattr(service, "_run_tests", failing_tests)
    monkeypatch.setattr(service, "_iterative_test_fix", fail_fix_loop)

    result = await service.conversational_service._attempt_patch_update(
        scope=Scope(kind="user", identifier="u1"),
        ui_id="ui1",
        name="dash",
        draft_payload=dict(DRAFT),
        user_message="change greeting",
        assistant_message="I will change the expression only.",
        access_token=None,
        previous_metadata={},
    )

    assert result is not None
    assert "html.div('patched')" in result["payload"]["components_script"]
    assert len(test_runs) == 2
    assert "html.div('patched')" in test_runs[0]
    assert "html.div('hello')" in test_runs[1]


@pytest.mark.asyncio
async def test_component_regression_requests_test_patch_before_fixer(monkeypatch):
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    async def fake_non_stream_completion(_request, _token, _span):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "operations": [
                                    {
                                        "target": "components_script",
                                        "op": "replace",
                                        "search": "html.div('hello')",
                                        "content": "html.div('DAG')",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }

    service.tgi_service.llm_client = SimpleNamespace(
        non_stream_completion=fake_non_stream_completion
    )
    candidate_failure = (
        "TAP version 13\n"
        "not ok 1 - toggles event details\n"
        "# tests 1\n"
        "# pass 0\n"
        "# fail 1\n"
    )
    test_results = iter(((False, candidate_failure), (True, "ok")))
    monkeypatch.setattr(
        service,
        "_run_tests",
        lambda *_args, **_kwargs: next(test_results),
    )

    async def fail_fix_loop(**_kwargs):  # pragma: no cover - defensive
        raise AssertionError("Planner retry must precede the generic fixer")

    monkeypatch.setattr(service, "_iterative_test_fix", fail_fix_loop)

    result = await service.conversational_service._attempt_patch_update(
        scope=Scope(kind="user", identifier="u1"),
        ui_id="ui1",
        name="dash",
        draft_payload=dict(DRAFT),
        user_message="replace the table with a DAG",
        assistant_message="I will replace only the event presentation.",
        access_token=None,
        previous_metadata={},
    )

    assert result is None
    assert service._last_patch_failure_reason.startswith(
        "patch_tests_failed_test_update_required"
    )
    assert "failed_tests=toggles event details" in service._last_patch_failure_reason
    assert "add exact test_script replacements" in service._last_patch_failure_reason


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
