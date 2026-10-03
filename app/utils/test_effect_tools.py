from types import SimpleNamespace

import pytest

from app.utils.effect_tools import (
    auto_effect_enabled,
    classify_effect,
    is_effect_tool,
)


@pytest.mark.parametrize(
    "tool_name, expected_effect",
    [
        # Read verbs followed by "_", "-", end of name or a camelCase boundary.
        ("get_issue", False),
        ("list-zones", False),
        ("search", False),
        ("read_file", False),
        ("fetch_page", False),
        ("find_users", False),
        ("describe_table", False),
        ("query_database", False),
        ("lookup_dns", False),
        ("show_status", False),
        ("view_page", False),
        ("count_rows", False),
        ("check_health", False),
        ("validate_config", False),
        ("preview_changes", False),
        ("explain_plan", False),
        ("summarize_thread", False),
        ("whoami", False),
        ("getUser", False),
        ("listZones", False),
        ("GetUser", False),
        ("GET_USER", False),
        # Case-insensitive verb match.
        ("List_Zones", False),
        # Namespace prefix before the last "." or "/" is ignored.
        ("cloudflare.list_zones", False),
        ("notion/search", False),
        ("a.b/get_page", False),
        ("cloudflare.delete_zone", True),
        ("notion/create-page", True),
        # Verb prefixes that are not a word boundary are not read verbs.
        ("checkout", True),
        ("counter_increment", True),
        ("getaway", True),
        ("GETTER", True),
        ("listen", True),
        ("readme_update", True),
        # Writes and unknown names are effects (conservative default).
        ("create_page", True),
        ("delete_zone", True),
        ("update_record", True),
        ("send_message", True),
        ("execute", True),
        ("sync_things", True),
        ("deploy-worker", True),
        ("get.delete_all", True),
        ("", True),
    ],
)
def test_classify_effect_by_name(tool_name, expected_effect):
    assert classify_effect(tool_name, None) is expected_effect


@pytest.mark.parametrize(
    "annotations, tool_name, expected_effect",
    [
        # readOnlyHint=True wins over a write-looking name.
        ({"readOnlyHint": True}, "delete_cache_entry", False),
        ({"readOnlyHint": True, "destructiveHint": True}, "purge", False),
        # readOnlyHint=False / destructiveHint=True win over a read-looking name.
        ({"readOnlyHint": False}, "get_and_bump_counter", True),
        ({"destructiveHint": True}, "list_and_prune", True),
        # Annotations without a usable hint fall back to the name.
        ({}, "get_issue", False),
        ({}, "create_issue", True),
        ({"destructiveHint": False}, "create_issue", True),
        ({"destructiveHint": False}, "get_issue", False),
        ({"readOnlyHint": None}, "get_issue", False),
        ({"readOnlyHint": "true"}, "create_issue", True),
        ({"title": "Get issue"}, "create_issue", True),
    ],
)
def test_classify_effect_annotations_win(annotations, tool_name, expected_effect):
    tool_def = {"name": tool_name, "annotations": annotations}
    assert classify_effect(tool_name, tool_def) is expected_effect


def test_classify_effect_reads_sdk_style_annotation_objects():
    # SDK v2 models expose snake_case attributes.
    read_only = SimpleNamespace(
        name="purge", annotations=SimpleNamespace(read_only_hint=True)
    )
    destructive = SimpleNamespace(
        name="get_x",
        annotations=SimpleNamespace(read_only_hint=None, destructive_hint=True),
    )
    camel = SimpleNamespace(name="purge", annotations={"readOnlyHint": True})

    assert classify_effect("purge", read_only) is False
    assert classify_effect("get_x", destructive) is True
    assert classify_effect("purge", camel) is False


def test_classify_effect_reads_annotations_nested_in_function_spec():
    tool_def = {"function": {"name": "purge", "annotations": {"readOnlyHint": True}}}
    assert classify_effect("purge", tool_def) is False


def test_classify_effect_without_annotations_uses_name():
    assert classify_effect("get_issue", {"name": "get_issue"}) is False
    assert classify_effect("create_issue", {"annotations": None}) is True


def test_auto_effect_enabled():
    assert auto_effect_enabled(["auto"])
    assert auto_effect_enabled(["create_*", " AUTO "])
    assert not auto_effect_enabled(["create_*"])
    assert not auto_effect_enabled([])
    assert not auto_effect_enabled(None)


def test_is_effect_tool_explicit_globs_only():
    patterns = ["create_*", "merge_pull_request"]

    assert is_effect_tool("create_page", None, patterns)
    assert is_effect_tool("merge_pull_request", None, patterns)
    # No auto: unknown names are not effects, exactly as before.
    assert not is_effect_tool("delete_zone", None, patterns)
    assert not is_effect_tool("get_page", None, patterns)
    assert not is_effect_tool("delete_zone", None, [])


def test_is_effect_tool_combines_globs_and_auto():
    patterns = ["get_and_mutate", "auto"]
    read_only = {"annotations": {"readOnlyHint": True}}

    # Explicit glob wins even over a read-only classification.
    assert is_effect_tool("get_and_mutate", read_only, patterns)
    # Otherwise auto decides.
    assert is_effect_tool("delete_zone", None, patterns)
    assert not is_effect_tool("list_zones", None, patterns)
    assert not is_effect_tool("purge", read_only, patterns)


def test_auto_entry_is_not_a_glob():
    assert is_effect_tool("auto", None, ["auto"])  # classified, not matched
    assert not is_effect_tool("get_auto", None, ["auto"])
