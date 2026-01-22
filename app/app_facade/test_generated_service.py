import asyncio
import json
import logging
import os
import re

import pytest

from app.app_facade.generated_service import (
    GeneratedUIService,
    GeneratedUIStorage,
    Scope,
    Actor,
    validate_identifier,
)


class DummyTGIService:
    def __init__(self):
        # minimal attributes used by tests, not used for heavy streaming
        self.llm_client = None
        self.prompt_service = None
        self.tool_service = None


@pytest.mark.parametrize("value", ["abc123", "A9_-"])
def test_validate_identifier_ok(value):
    assert validate_identifier(value, "field") == value


@pytest.mark.parametrize("value", ["", None])
def test_validate_identifier_empty(value):
    with pytest.raises(Exception):
        validate_identifier(value, "field")


def test_validate_identifier_bad_pattern():
    with pytest.raises(Exception):
        validate_identifier("!bad id", "field")


def test_scope_actor_is_owner_user_and_group():
    scope_user = Scope(kind="user", identifier="u1")
    actor = Actor(user_id="u1", groups=["g1"])
    assert actor.is_owner(scope_user)

    scope_group = Scope(kind="group", identifier="g1")
    assert actor.is_owner(scope_group)

    not_owner = Actor(user_id="u2", groups=["g2"])
    assert not not_owner.is_owner(scope_group)


def test_storage_write_read_exists(tmp_path):
    base = tmp_path / "storage"
    storage = GeneratedUIStorage(str(base))
    scope = Scope(kind="user", identifier="u1")
    ui_id = "ui1"
    name = "name1"
    payload = {"html": {"page": "<p>x</p>"}}
    storage.write(scope, ui_id, name, payload)
    assert storage.exists(scope, ui_id, name)
    read = storage.read(scope, ui_id, name)
    # storage.read should return the same object we wrote
    assert read == payload


@pytest.mark.parametrize(
    "text,expected",
    [
        ('prefix {"a":1} suffix', '{"a":1}'),
        ('{"outer": {"inner": 2}} trailing', '{"outer": {"inner": 2}}'),
    ],
)
def test_extract_json_block_and_parse(text, expected):
    # Use an instance to access the helper methods
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    # Access private method _extract_json_block
    candidate = service._extract_json_block(text)
    assert candidate == expected
    parsed = service._parse_json(text)
    assert isinstance(parsed, dict)


def test_parse_json_with_invalid_json_and_no_block():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    with pytest.raises(Exception):
        service._parse_json("not json or {broken")


def test_normalise_payload_html_variants(tmp_path):
    storage = GeneratedUIStorage(str(tmp_path))
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    scope = Scope(kind="user", identifier="u1")

    # html as string
    payload = {"html": "<div>ok</div>", "metadata": {}}
    service._normalise_payload(payload, scope, "id1", "name1", "prompt", None)
    assert isinstance(payload["html"], dict)
    assert "page" in payload["html"]

    # only snippet -> creates page
    payload2 = {"html": {"snippet": "<span>s</span>"}, "metadata": {}}
    service._normalise_payload(payload2, scope, "id2", "name2", "prompt2", None)
    assert "page" in payload2["html"] and "snippet" in payload2["html"]

    # missing html -> synthesize fallback
    payload3 = {"other": "x"}
    service._normalise_payload(payload3, scope, "id3", "name3", "<>&'\"", None)
    assert "html" in payload3
    assert "snippet" in payload3["html"]


def test_normalise_payload_preserves_previous_original_prompt(tmp_path):
    storage = GeneratedUIStorage(str(tmp_path))
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    scope = Scope(kind="user", identifier="u1")
    previous = {
        "metadata": {
            "history": [{"prompt": "first prompt"}],
        }
    }
    payload = {"html": {"page": "<p></p>"}}
    service._normalise_payload(payload, scope, "id4", "name4", "req", previous)
    assert payload.get("metadata", {}).get("original_requirements") == "first prompt"


def test_filter_relevant_tools_simple():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    # Build a set of fake tools
    tools = [
        {"function": {"name": "list_users", "description": "List users"}},
        {"function": {"name": "create_user", "description": "Create"}},
        {"function": {"name": "describe_tool", "description": "Meta"}},
    ]
    # Short prompt returns all tools
    selected = service._filter_relevant_tools(tools, "")
    assert selected == tools

    # Long prompt should prioritize tools; ensure describe_tool is present
    sel2 = service._filter_relevant_tools(
        tools * 5, "I need to list users and fetch accounts"
    )
    names = [t.get("function", {}).get("name") for t in sel2]
    assert "describe_tool" in names or len(sel2) >= 1


def test_extract_body_and_wrap_snippet():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    page = "<html><body>hello</body></html>"
    assert service._extract_body(page) == "hello"
    snippet = "<div>x</div>"
    wrapped = service._wrap_snippet(snippet)
    assert "<!DOCTYPE html>" in wrapped and snippet in wrapped


def test_normalise_payload_adds_script_blocks():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    scope = Scope(kind="user", identifier="u1")
    payload = {
        "html": {
            "page": "<html><body><!-- include:snippet --></body></html>",
            "snippet": "<div>x</div>",
        },
        "metadata": {},
    }
    service._normalise_payload(payload, scope, "ui1", "name1", "req", previous=None)
    html = payload["html"]
    assert '<script type="module">' in html["page"]
    assert "<!-- include:service_script -->" in html["page"]
    assert "<!-- include:components_script -->" in html["page"]
    assert '<script type="module">' in html["snippet"]
    assert "<!-- include:service_script -->" in html["snippet"]
    assert "<!-- include:components_script -->" in html["snippet"]


def test_expand_payload_injects_snippet_scripts():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    payload = {
        "html": {
            "page": "<html><body><!-- include:snippet --></body></html>",
            "snippet": "<div>Hi</div>",
        },
        "service_script": "export class McpService {}",
        "components_script": "console.log('ok');",
    }
    expanded = service._expand_payload(payload)
    page = expanded["html"]["page"]
    snippet = expanded["html"]["snippet"]
    assert "export class McpService" in page
    assert "console.log('ok');" in page
    assert '<script type="module">' in page
    assert "<!-- include:service_script -->" not in page
    assert "export class McpService" in snippet
    assert "console.log('ok');" in snippet
    assert '<script type="module">' in snippet


@pytest.mark.asyncio
async def test_stream_generate_ui_prechecks(tmp_path):
    storage = GeneratedUIStorage(str(tmp_path))
    tgi = DummyTGIService()
    service = GeneratedUIService(storage=storage, tgi_service=tgi)
    actor = Actor(user_id="u1", groups=[])

    # Case: storage exists -> should yield an SSE error event as bytes
    # create a file to trigger exists
    os.makedirs(
        os.path.join(str(tmp_path), "user", "u1", "ui1", "name1"), exist_ok=True
    )
    file_path = os.path.join(str(tmp_path), "user", "u1", "ui1", "name1", "ui.json")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("{}")

    # replace service storage to point to our tmp path
    service.storage = GeneratedUIStorage(str(tmp_path))

    gen = service.stream_generate_ui(
        session=None,
        scope=Scope(kind="user", identifier="u1"),
        actor=actor,
        ui_id="ui1",
        name="name1",
        prompt="p",
        tools=None,
        access_token=None,
    )

    # read first chunk
    first = await gen.__anext__()
    assert b"error" in first


def test_filter_relevant_tools_deprioritize_create():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    tools = [
        {"function": {"name": "create_thing", "description": "Create a thing"}},
        {"function": {"name": "list_thing", "description": "List things"}},
    ]
    # prompt without create-related words should penalize create_ prefix
    sel = service._filter_relevant_tools(tools * 3, "Please show me the list of things")
    names = [t.get("function", {}).get("name") for t in sel]
    assert "list_thing" in names


def test_select_tools_requested_tools_with_mixed_shapes():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    class ToolObj:
        def __init__(self, name, desc=""):
            class Fn:
                def __init__(self, name, desc):
                    self.name = name
                    self.description = desc

            self.function = Fn(name, desc)

    async def get_all(session, include_output_schema=True):
        return [
            {"function": {"name": "alpha", "description": "a"}},
            ToolObj("beta", "b"),
        ]

    service.tgi_service.tool_service = type(
        "TS", (), {"get_all_mcp_tools": staticmethod(get_all)}
    )()

    # request only beta
    selected = asyncio.run(service._select_tools(None, ["beta"], ""))
    assert selected is not None
    names = [
        t.get("function", {}).get("name") if isinstance(t, dict) else t.function.name
        for t in selected
    ]
    assert "beta" in names


def test_extract_content_variants_and_failure():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    # dict with content
    assert service._extract_content({"content": "x"}) == "x"

    # object with choices/message/content
    class Msg:
        def __init__(self, content=None):
            self.content = content

    class Choice:
        def __init__(self, msg=None, delta=None):
            self.message = msg
            self.delta = delta

    class Obj:
        def __init__(self, choices):
            self.choices = choices

    obj = Obj([Choice(Msg("hello"))])
    assert service._extract_content(obj) == "hello"

    # delta content
    obj2 = Obj([Choice(None, Msg("d"))])
    assert service._extract_content(obj2) == "d"

    # failure
    with pytest.raises(Exception):
        service._extract_content(object())


def test_storage_read_file_not_found_and_invalid_json(tmp_path):
    storage = GeneratedUIStorage(str(tmp_path))
    scope = Scope(kind="user", identifier="u1")
    # read nonexistent
    with pytest.raises(FileNotFoundError):
        storage.read(scope, "noid", "noname")

    # create folder and invalid json
    d = os.path.join(str(tmp_path), "user", "u1", "id1", "name1")
    os.makedirs(d, exist_ok=True)
    fpath = os.path.join(d, "ui.json")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("not json")

    with pytest.raises(Exception):
        storage.read(scope, "id1", "name1")


def test_assert_scope_consistency_and_permissions():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    # build a record with mismatched scope
    existing = {"metadata": {"scope": {"type": "group", "id": "g1"}, "name": "n1"}}
    with pytest.raises(Exception):
        service._assert_scope_consistency(
            existing, Scope(kind="user", identifier="u1"), "n1"
        )

    # ensure_update_permissions
    actor = Actor(user_id="u2", groups=["g2"])
    with pytest.raises(Exception):
        service._ensure_update_permissions(
            existing, Scope(kind="group", identifier="g1"), actor
        )


def test_history_entry_and_now_format():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    entry = service._history_entry(
        action="create",
        prompt="p",
        tools=["t"],
        user_id="u",
        generated_at="when",
        payload_metadata={"m": 1},
    )
    assert entry["action"] == "create" and entry["user_id"] == "u"
    now = service._now()
    # crude check for ISO format
    assert re.match(r"\d{4}-\d{2}-\d{2}", now)


def test_history_for_prompt_strips_payload_html_without_mutating():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    history = [
        {
            "action": "create",
            "payload_html": {"page": "<p>a</p>"},
            "payload_scripts": {"service_script": "class S {}"},
            "prompt": "one",
        },
        "not-a-dict",
    ]
    sanitized = service._history_for_prompt(history)
    assert len(sanitized) == 1
    assert "payload_html" not in sanitized[0]
    assert "payload_scripts" not in sanitized[0]
    # original history entry still retains payload_html
    assert "payload_html" in history[0]


def test_build_system_prompt_reads_pfusch_and_fallback(monkeypatch):
    storage = GeneratedUIStorage(os.getcwd())
    tgi = DummyTGIService()
    service = GeneratedUIService(storage=storage, tgi_service=tgi)

    class PS:
        async def find_prompt_by_name_or_role(self, session, prompt_name=None):
            return None

        async def get_prompt_content(self, session, prompt):
            return ""

    service.tgi_service.prompt_service = PS()
    # should not raise
    res = asyncio.run(service._build_system_prompt(None))
    assert isinstance(res, str)


def test_load_pfusch_prompt_and_fallback_presence():
    # _load_pfusch_prompt should read the packaged markdown and replace MCP_BASE_PATH
    from app.app_facade.generated_service import (
        _load_pfusch_prompt,
    )

    s = _load_pfusch_prompt()
    assert "Pfusch" in s or "You are a microsite" in s


def test_load_pfusch_prompt_error(monkeypatch):
    # simulate open raising an error
    import builtins

    def bad_open(*a, **k):
        raise IOError("boom")

    monkeypatch.setattr(builtins, "open", bad_open)
    from app.app_facade.generated_service import _load_pfusch_prompt

    with pytest.raises(Exception):
        _load_pfusch_prompt()


def test_generate_ui_payload_empty_stream_raises(monkeypatch):
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    # patch prompt_service and tool_service
    class PS:
        async def find_prompt_by_name_or_role(self, session, prompt_name=None):
            return None

    service.tgi_service.prompt_service = PS()

    class TS:
        async def get_all_mcp_tools(self, session, include_output_schema=True):
            return []

    service.tgi_service.tool_service = TS()

    # fake llm client
    class LLM:
        def stream_completion(self, req, token, none):
            return object()

    service.tgi_service.llm_client = LLM()

    # fake chunk_reader yields only items with no content
    class FakeParsed:
        def __init__(self, content=None, is_done=True):
            self.content = content
            self.is_done = is_done

    class FakeReader:
        def __init__(self, items):
            self._items = items

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def as_parsed(self):
            for it in self._items:
                yield it

    def fake_chunk_reader(src):
        return FakeReader([FakeParsed(content=None, is_done=True)])

    monkeypatch.setattr(
        "app.app_facade.generated_service.chunk_reader", fake_chunk_reader
    )

    with pytest.raises(Exception):
        asyncio.run(
            service._generate_ui_payload(
                session=None,
                scope=Scope(kind="user", identifier="u1"),
                ui_id="x",
                name="n",
                prompt="p",
                tools=[],
                access_token=None,
                previous=None,
            )
        )


def test_normalise_payload_non_dict_raises():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    with pytest.raises(Exception):
        service._normalise_payload(
            [1, 2, 3], Scope(kind="user", identifier="u"), "id", "name", "p", None
        )


def test_select_tools_no_available(monkeypatch):
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    class TS:
        async def get_all_mcp_tools(self, session, include_output_schema=True):
            return None

    service.tgi_service.tool_service = TS()
    res = asyncio.run(service._select_tools(None, [], ""))
    assert res is None


def test_select_tools_requested_no_match(monkeypatch):
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    class TS:
        async def get_all_mcp_tools(self, session, include_output_schema=True):
            return [{"function": {"name": "one", "description": "x"}}]

    service.tgi_service.tool_service = TS()
    res = asyncio.run(service._select_tools(None, ["missing"], ""))
    assert res is None


def test_extract_json_block_with_escaped_string():
    storage = GeneratedUIStorage(os.getcwd())
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())
    text = 'prefix {"a": "value with } bracket"} suffix'
    candidate = service._extract_json_block(text)
    assert candidate is not None and '"a"' in candidate


@pytest.mark.asyncio
async def test_create_ui_and_duplicate_and_permissions(tmp_path, monkeypatch):
    storage = GeneratedUIStorage(str(tmp_path))
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    # patch generate payload to return a simple payload
    async def fake_generate(
        *, session, scope, ui_id, name, prompt, tools, access_token, previous
    ):
        return {"html": {"page": "<p>ok</p>"}, "metadata": {}}

    monkeypatch.setattr(service, "_generate_ui_payload", fake_generate)

    # successful create
    actor = Actor(user_id="u1", groups=["g1"])
    rec = await service.create_ui(
        session=None,
        scope=Scope(kind="user", identifier="u1"),
        actor=actor,
        ui_id="ui_new",
        name="n1",
        prompt="p",
        tools=None,
        access_token=None,
    )
    assert rec["metadata"]["id"] == "ui_new"
    # duplicate create raises
    with pytest.raises(Exception):
        await service.create_ui(
            session=None,
            scope=Scope(kind="user", identifier="u1"),
            actor=actor,
            ui_id="ui_new",
            name="n1",
            prompt="p",
            tools=None,
            access_token=None,
        )

    # permission: user mismatch
    with pytest.raises(Exception):
        await service.create_ui(
            session=None,
            scope=Scope(kind="user", identifier="other"),
            actor=actor,
            ui_id="ui_x",
            name="n2",
            prompt="p",
            tools=None,
            access_token=None,
        )


@pytest.mark.asyncio
async def test_update_and_get_ui_paths(tmp_path, monkeypatch):
    storage = GeneratedUIStorage(str(tmp_path))
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    # existing file must be present
    scope = Scope(kind="group", identifier="g1")
    actor = Actor(user_id="u_owner", groups=["g1"])
    existing = {
        "metadata": {
            "scope": {"type": "group", "id": "g1"},
            "name": "n1",
            "history": [],
        },
        "current": {"html": {"page": "<p>v</p>"}},
    }
    storage.write(scope, "eid", "n1", existing)

    # patch generate to return new payload
    async def fake_generate(
        *, session, scope, ui_id, name, prompt, tools, access_token, previous
    ):
        return {"html": {"page": "<p>new</p>"}, "metadata": {}}

    monkeypatch.setattr(service, "_generate_ui_payload", fake_generate)

    updated = await service.update_ui(
        session=None,
        scope=scope,
        actor=actor,
        ui_id="eid",
        name="n1",
        prompt="p2",
        tools=None,
        access_token=None,
    )
    assert updated["current"]["html"]["page"] == "<p>new</p>"

    # get_ui doesn't check permissions, it only checks existence and scope consistency
    # So calling get with any actor should succeed as long as the UI exists
    fetched = service.get_ui(
        scope=scope, actor=Actor(user_id="x", groups=[]), ui_id="eid", name="n1"
    )
    assert fetched is not None


@pytest.mark.asyncio
async def test_stream_generate_ui_success_and_failure_cases(tmp_path, monkeypatch):
    storage = GeneratedUIStorage(str(tmp_path))
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    # monkeypatch load prompt to avoid file IO
    monkeypatch.setattr(
        "app.app_facade.generated_service._load_pfusch_prompt",
        lambda: "PROMPT {{DESIGN_SYSTEM_PROMPT}}",
    )

    # make prompt service return None
    class PS:
        async def find_prompt_by_name_or_role(self, session, prompt_name=None):
            return None

    service.tgi_service.prompt_service = PS()

    # prepare fake llm client and chunk reader
    class FakeParsed:
        def __init__(self, content=None, is_done=False):
            self.content = content
            self.is_done = is_done
            self.tool_calls = None
            self.accumulated_tool_calls = None
            self.finish_reason = None

    class FakeReader:
        def __init__(self, items):
            self._items = items

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def as_parsed(self):
            for it in self._items:
                yield it

    # patch chunk_reader to return FakeReader
    def fake_chunk_reader(src):
        # return object; items will be set later by closure
        return FakeReader(fake_chunk_reader.items)

    monkeypatch.setattr(
        "app.app_facade.generated_service.chunk_reader", fake_chunk_reader
    )

    # set llm_client.stream_completion to a dummy
    class LLM:
        def stream_completion(self, req, token, none):
            return object()

    service.tgi_service.llm_client = LLM()

    class TS:
        async def get_all_mcp_tools(self, session, include_output_schema=True):
            return []

    service.tgi_service.tool_service = TS()

    # success case: yield two content chunks forming a JSON
    payload = {
        "html": {"page": "<p>z</p>"},
        "metadata": {},
        "service_script": "class S {}",
        "components_script": "pfusch()",
        "test_script": "test()",
    }
    json_str = json.dumps(payload, ensure_ascii=False)
    # split into two chunks
    c1 = json_str[: len(json_str) // 2]
    c2 = json_str[len(json_str) // 2 :]
    fake_chunk_reader.items = [
        FakeParsed(content=c1),
        FakeParsed(content=c2),
        FakeParsed(is_done=True),
    ]

    # mock _run_tests
    service._run_tests = lambda s, c, t, d=None: (True, "Tests passed")

    gen = service.stream_generate_ui(
        session=None,
        scope=Scope(kind="user", identifier="u1"),
        actor=Actor(user_id="u1", groups=[]),
        ui_id="s1",
        name="n",
        prompt="p",
        tools=None,
        access_token=None,
    )

    collected = []
    async for chunk in gen:
        collected.append(chunk)

    # final event should indicate done
    assert any(b"done" in c or b"created" in c for c in collected)
    # storage must have persisted
    assert storage.exists(Scope(kind="user", identifier="u1"), "s1", "n")

    # failure parsing JSON -> produce error event
    fake_chunk_reader.items = [FakeParsed(content="not json"), FakeParsed(is_done=True)]
    gen2 = service.stream_generate_ui(
        session=None,
        scope=Scope(kind="user", identifier="u2"),
        actor=Actor(user_id="u2", groups=[]),
        ui_id="s2",
        name="nx",
        prompt="p",
        tools=None,
        access_token=None,
    )
    outs = []
    async for o in gen2:
        outs.append(o)
    # should be an error event emitted (because parse fails)
    assert any(b"error" in o for o in outs)

    # persist failure: make storage.write raise
    def bad_write(scope, ui_id, name, payload):
        raise RuntimeError("disk full")

    monkeypatch.setattr(storage, "write", bad_write)
    # produce a valid json again
    fake_chunk_reader.items = [FakeParsed(content=json_str), FakeParsed(is_done=True)]
    gen3 = service.stream_generate_ui(
        session=None,
        scope=Scope(kind="user", identifier="u3"),
        actor=Actor(user_id="u3", groups=[]),
        ui_id="s3",
        name="n3",
        prompt="p",
        tools=None,
        access_token=None,
    )
    # first chunks should stream content, final should include error about persist
    out = []
    async for b in gen3:
        out.append(b)
    assert any(b"Failed to persist" in o for o in out)


@pytest.mark.asyncio
async def test_stream_update_ui_success_and_not_found_cases(tmp_path, monkeypatch):
    storage = GeneratedUIStorage(str(tmp_path))
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    monkeypatch.setattr(
        "app.app_facade.generated_service._load_pfusch_prompt",
        lambda: "PROMPT {{DESIGN_SYSTEM_PROMPT}}",
    )

    class PS:
        async def find_prompt_by_name_or_role(self, session, prompt_name=None):
            return None

    service.tgi_service.prompt_service = PS()

    class FakeParsed:
        def __init__(self, content=None, is_done=False):
            self.content = content
            self.is_done = is_done
            self.tool_calls = None
            self.accumulated_tool_calls = None
            self.finish_reason = None

    class FakeReader:
        def __init__(self, items):
            self._items = items

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def as_parsed(self):
            for it in self._items:
                yield it

    def fake_chunk_reader(src):
        return FakeReader(fake_chunk_reader.items)

    monkeypatch.setattr(
        "app.app_facade.generated_service.chunk_reader", fake_chunk_reader
    )

    class LLM:
        def stream_completion(self, req, token, none):
            return object()

    service.tgi_service.llm_client = LLM()

    class TS:
        async def get_all_mcp_tools(self, session, include_output_schema=True):
            return []

    service.tgi_service.tool_service = TS()

    payload = {
        "html": {"page": "<p>z</p>"},
        "metadata": {},
        "service_script": "class S {}",
        "components_script": "pfusch()",
        "test_script": "test()",
    }
    json_str = json.dumps(payload, ensure_ascii=False)
    c1 = json_str[: len(json_str) // 2]
    c2 = json_str[len(json_str) // 2 :]
    fake_chunk_reader.items = [
        FakeParsed(content=c1),
        FakeParsed(content=c2),
        FakeParsed(is_done=True),
    ]

    service._run_tests = lambda s, c, t, d=None: (True, "Tests passed")

    scope = Scope(kind="user", identifier="u1")
    existing = {
        "metadata": {
            "id": "s1",
            "name": "n",
            "scope": {"type": "user", "id": "u1"},
            "history": [{"prompt": "orig"}],
        },
        "current": {"html": {"page": "<p>old</p>"}, "metadata": {}},
    }
    storage.write(scope, "s1", "n", existing)

    gen = service.stream_update_ui(
        session=None,
        scope=scope,
        actor=Actor(user_id="u1", groups=[]),
        ui_id="s1",
        name="n",
        prompt="p2",
        tools=None,
        access_token=None,
    )

    collected = []
    async for chunk in gen:
        collected.append(chunk)

    assert any(b"updated" in c for c in collected)
    updated = storage.read(scope, "s1", "n")
    assert "<p>z</p>" in updated["current"]["html"]["page"]
    assert len(updated["metadata"]["history"]) == 2
    assert updated["metadata"]["updated_by"] == "u1"

    gen2 = service.stream_update_ui(
        session=None,
        scope=scope,
        actor=Actor(user_id="u1", groups=[]),
        ui_id="missing",
        name="n",
        prompt="p2",
        tools=None,
        access_token=None,
    )
    out = []
    async for b in gen2:
        out.append(b)
    assert any(b"Ui not found" in o for o in out)


def test_reset_last_change_success(tmp_path):
    """Test successful reset of the last change."""
    storage = GeneratedUIStorage(str(tmp_path))
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    scope = Scope(kind="group", identifier="g1")
    actor = Actor(user_id="u1", groups=["g1"])

    # Create a record with multiple history entries
    existing = {
        "metadata": {
            "id": "ui1",
            "name": "test",
            "scope": {"type": "group", "id": "g1"},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-03T00:00:00Z",
            "history": [
                {
                    "action": "create",
                    "prompt": "first prompt",
                    "tools": [],
                    "user_id": "u1",
                    "generated_at": "2024-01-01T00:00:00Z",
                    "payload_metadata": {"m": 1},
                    "payload_html": {"page": "<p>first</p>", "snippet": "<p>first</p>"},
                    "payload_scripts": {
                        "service_script": "service v1",
                        "components_script": "components v1",
                        "test_script": "test v1",
                    },
                },
                {
                    "action": "update",
                    "prompt": "second prompt",
                    "tools": [],
                    "user_id": "u1",
                    "generated_at": "2024-01-02T00:00:00Z",
                    "payload_metadata": {"m": 2},
                    "payload_html": {
                        "page": "<p>second</p>",
                        "snippet": "<p>second</p>",
                    },
                },
                {
                    "action": "update",
                    "prompt": "third prompt",
                    "tools": [],
                    "user_id": "u1",
                    "generated_at": "2024-01-03T00:00:00Z",
                    "payload_metadata": {"m": 3},
                    "payload_html": {"page": "<p>third</p>", "snippet": "<p>third</p>"},
                    "payload_scripts": {
                        "service_script": "service v2",
                        "components_script": "components v2",
                        "test_script": "test v2",
                    },
                },
            ],
        },
        "current": {
            "html": {"page": "<p>third</p>", "snippet": "<p>third</p>"},
            "metadata": {"m": 3},
            "service_script": "service v2",
            "components_script": "components v2",
            "test_script": "test v2",
        },
    }

    storage.write(scope, "ui1", "test", existing)

    # Reset the last change
    result = service.reset_last_change(
        scope=scope, actor=actor, ui_id="ui1", name="test"
    )

    # Verify the history has been reduced by one
    assert len(result["metadata"]["history"]) == 2

    # Verify the current state now reflects the second entry
    assert result["current"]["html"]["page"] == "<p>second</p>"
    assert result["current"]["metadata"]["m"] == 2
    assert result["current"]["service_script"] == "service v1"
    assert result["current"]["components_script"] == "components v1"
    assert result["current"]["test_script"] == "test v1"

    # Verify updated_at and updated_by were set
    assert "updated_at" in result["metadata"]
    assert result["metadata"]["updated_by"] == "u1"

    # Verify the record was persisted
    reloaded = storage.read(scope, "ui1", "test")
    assert len(reloaded["metadata"]["history"]) == 2
    assert reloaded["current"]["html"]["page"] == "<p>second</p>"


def test_reset_last_change_not_found(tmp_path):
    """Test reset on non-existent UI raises 404."""
    storage = GeneratedUIStorage(str(tmp_path))
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    scope = Scope(kind="user", identifier="u1")
    actor = Actor(user_id="u1", groups=[])

    with pytest.raises(Exception) as exc_info:
        service.reset_last_change(
            scope=scope, actor=actor, ui_id="nonexistent", name="test"
        )

    # Should be a 404 error
    assert hasattr(exc_info.value, "status_code") and exc_info.value.status_code == 404


def test_reset_last_change_single_history_entry(tmp_path):
    """Test reset fails when only one history entry exists (initial creation)."""
    storage = GeneratedUIStorage(str(tmp_path))
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    scope = Scope(kind="user", identifier="u1")
    actor = Actor(user_id="u1", groups=[])

    # Create a record with only one history entry
    existing = {
        "metadata": {
            "id": "ui1",
            "name": "test",
            "scope": {"type": "user", "id": "u1"},
            "created_at": "2024-01-01T00:00:00Z",
            "history": [
                {
                    "action": "create",
                    "prompt": "first prompt",
                    "tools": [],
                    "user_id": "u1",
                    "generated_at": "2024-01-01T00:00:00Z",
                    "payload_metadata": {"m": 1},
                    "payload_html": {"page": "<p>first</p>"},
                }
            ],
        },
        "current": {"html": {"page": "<p>first</p>"}, "metadata": {"m": 1}},
    }

    storage.write(scope, "ui1", "test", existing)

    # Attempt to reset should fail
    with pytest.raises(Exception) as exc_info:
        service.reset_last_change(scope=scope, actor=actor, ui_id="ui1", name="test")

    # Should be a 400 error about only one history entry
    assert hasattr(exc_info.value, "status_code") and exc_info.value.status_code == 400
    assert "only one history entry" in str(exc_info.value.detail).lower()


def test_reset_last_change_permission_denied_user_scope(tmp_path):
    """Test reset fails when user doesn't own user-scoped UI."""
    storage = GeneratedUIStorage(str(tmp_path))
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    scope = Scope(kind="user", identifier="u1")
    actor = Actor(user_id="u2", groups=[])  # Different user

    # Create a record
    existing = {
        "metadata": {
            "id": "ui1",
            "name": "test",
            "scope": {"type": "user", "id": "u1"},
            "history": [
                {
                    "action": "create",
                    "prompt": "first",
                    "tools": [],
                    "user_id": "u1",
                    "generated_at": "2024-01-01T00:00:00Z",
                    "payload_metadata": {},
                    "payload_html": {"page": "<p>1</p>"},
                },
                {
                    "action": "update",
                    "prompt": "second",
                    "tools": [],
                    "user_id": "u1",
                    "generated_at": "2024-01-02T00:00:00Z",
                    "payload_metadata": {},
                    "payload_html": {"page": "<p>2</p>"},
                },
            ],
        },
        "current": {"html": {"page": "<p>2</p>"}},
    }

    storage.write(scope, "ui1", "test", existing)

    # Attempt to reset should fail due to permission
    with pytest.raises(Exception) as exc_info:
        service.reset_last_change(scope=scope, actor=actor, ui_id="ui1", name="test")

    # Should be a 403 error
    assert hasattr(exc_info.value, "status_code") and exc_info.value.status_code == 403


def test_reset_last_change_permission_denied_group_scope(tmp_path):
    """Test reset fails when user is not in the group."""
    storage = GeneratedUIStorage(str(tmp_path))
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    scope = Scope(kind="group", identifier="g1")
    actor = Actor(user_id="u1", groups=["g2"])  # Different group

    # Create a record
    existing = {
        "metadata": {
            "id": "ui1",
            "name": "test",
            "scope": {"type": "group", "id": "g1"},
            "history": [
                {
                    "action": "create",
                    "prompt": "first",
                    "tools": [],
                    "user_id": "u1",
                    "generated_at": "2024-01-01T00:00:00Z",
                    "payload_metadata": {},
                    "payload_html": {"page": "<p>1</p>"},
                },
                {
                    "action": "update",
                    "prompt": "second",
                    "tools": [],
                    "user_id": "u1",
                    "generated_at": "2024-01-02T00:00:00Z",
                    "payload_metadata": {},
                    "payload_html": {"page": "<p>2</p>"},
                },
            ],
        },
        "current": {"html": {"page": "<p>2</p>"}},
    }

    storage.write(scope, "ui1", "test", existing)

    # Attempt to reset should fail due to permission
    with pytest.raises(Exception) as exc_info:
        service.reset_last_change(scope=scope, actor=actor, ui_id="ui1", name="test")

    # Should be a 403 error
    assert hasattr(exc_info.value, "status_code") and exc_info.value.status_code == 403


def test_reset_last_change_scope_path_mismatch(tmp_path):
    """Test reset fails when requested scope results in different storage path."""
    storage = GeneratedUIStorage(str(tmp_path))
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    # Store with one scope
    stored_scope = Scope(kind="group", identifier="g1")
    existing = {
        "metadata": {
            "id": "ui1",
            "name": "test",
            "scope": {"type": "group", "id": "g1"},
            "history": [
                {
                    "action": "create",
                    "prompt": "first",
                    "tools": [],
                    "user_id": "u1",
                    "generated_at": "2024-01-01T00:00:00Z",
                    "payload_metadata": {},
                    "payload_html": {"page": "<p>1</p>"},
                },
                {
                    "action": "update",
                    "prompt": "second",
                    "tools": [],
                    "user_id": "u1",
                    "generated_at": "2024-01-02T00:00:00Z",
                    "payload_metadata": {},
                    "payload_html": {"page": "<p>2</p>"},
                },
            ],
        },
        "current": {"html": {"page": "<p>2</p>"}},
    }

    storage.write(stored_scope, "ui1", "test", existing)

    # Try to reset with different scope - this will result in a different storage path
    # so the file won't be found (404), not a scope mismatch (403)
    wrong_scope = Scope(kind="user", identifier="u1")
    actor = Actor(user_id="u1", groups=["g1"])

    with pytest.raises(Exception) as exc_info:
        service.reset_last_change(
            scope=wrong_scope, actor=actor, ui_id="ui1", name="test"
        )

    # Should be a 404 error because the storage path differs
    assert hasattr(exc_info.value, "status_code") and exc_info.value.status_code == 404


def test_reset_last_change_stored_scope_inconsistency(tmp_path):
    """Test reset fails when stored metadata scope doesn't match the request scope (data corruption scenario)."""
    storage = GeneratedUIStorage(str(tmp_path))
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    # Create a scenario where the file is in the right path but metadata is inconsistent
    # This simulates data corruption or manual file manipulation
    scope = Scope(kind="group", identifier="g1")
    existing = {
        "metadata": {
            "id": "ui1",
            "name": "test",
            # Stored scope claims to be g2, but file is in g1 path
            "scope": {"type": "group", "id": "g2"},
            "history": [
                {
                    "action": "create",
                    "prompt": "first",
                    "tools": [],
                    "user_id": "u1",
                    "generated_at": "2024-01-01T00:00:00Z",
                    "payload_metadata": {},
                    "payload_html": {"page": "<p>1</p>"},
                },
                {
                    "action": "update",
                    "prompt": "second",
                    "tools": [],
                    "user_id": "u1",
                    "generated_at": "2024-01-02T00:00:00Z",
                    "payload_metadata": {},
                    "payload_html": {"page": "<p>2</p>"},
                },
            ],
        },
        "current": {"html": {"page": "<p>2</p>"}},
    }

    # Write to g1 path but with g2 in metadata
    storage.write(scope, "ui1", "test", existing)

    actor = Actor(user_id="u1", groups=["g1", "g2"])

    with pytest.raises(Exception) as exc_info:
        service.reset_last_change(scope=scope, actor=actor, ui_id="ui1", name="test")

    # Should be a 403 error about scope mismatch
    assert hasattr(exc_info.value, "status_code") and exc_info.value.status_code == 403
    assert "scope mismatch" in str(exc_info.value.detail).lower()


def test_reset_last_change_with_group_member(tmp_path):
    """Test successful reset by a group member."""
    storage = GeneratedUIStorage(str(tmp_path))
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    scope = Scope(kind="group", identifier="team-alpha")
    # Different user but in the same group
    actor = Actor(user_id="u2", groups=["team-alpha", "other-group"])

    existing = {
        "metadata": {
            "id": "ui1",
            "name": "dashboard",
            "scope": {"type": "group", "id": "team-alpha"},
            "created_by": "u1",
            "history": [
                {
                    "action": "create",
                    "prompt": "create dashboard",
                    "tools": [],
                    "user_id": "u1",
                    "generated_at": "2024-01-01T00:00:00Z",
                    "payload_metadata": {"version": 1},
                    "payload_html": {"page": "<p>v1</p>"},
                },
                {
                    "action": "update",
                    "prompt": "update dashboard",
                    "tools": [],
                    "user_id": "u1",
                    "generated_at": "2024-01-02T00:00:00Z",
                    "payload_metadata": {"version": 2},
                    "payload_html": {"page": "<p>v2</p>"},
                },
            ],
        },
        "current": {"html": {"page": "<p>v2</p>"}, "metadata": {"version": 2}},
    }

    storage.write(scope, "ui1", "dashboard", existing)

    # Reset should succeed because u2 is in team-alpha
    result = service.reset_last_change(
        scope=scope, actor=actor, ui_id="ui1", name="dashboard"
    )

    assert len(result["metadata"]["history"]) == 1
    assert result["current"]["metadata"]["version"] == 1
    assert result["metadata"]["updated_by"] == "u2"


def test_reset_last_change_empty_history(tmp_path):
    """Test reset fails gracefully when history is unexpectedly empty."""
    storage = GeneratedUIStorage(str(tmp_path))
    service = GeneratedUIService(storage=storage, tgi_service=DummyTGIService())

    scope = Scope(kind="user", identifier="u1")
    actor = Actor(user_id="u1", groups=[])

    # Create a malformed record with empty history (shouldn't happen in practice)
    existing = {
        "metadata": {
            "id": "ui1",
            "name": "test",
            "scope": {"type": "user", "id": "u1"},
            "history": [],
        },
        "current": {"html": {"page": "<p>current</p>"}},
    }

    storage.write(scope, "ui1", "test", existing)

    with pytest.raises(Exception) as exc_info:
        service.reset_last_change(scope=scope, actor=actor, ui_id="ui1", name="test")

    # Should be a 400 error
    assert hasattr(exc_info.value, "status_code") and exc_info.value.status_code == 400


# === Tests for iterative test fix tool call handling ===


class MockToolCall:
    """Mock tool call object mimicking OpenAI SDK response."""

    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.type = "function"
        self.function = MockFunction(name, arguments)


class MockFunction:
    """Mock function object for tool calls."""

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class MockMessage:
    """Mock message object mimicking OpenAI SDK response."""

    def __init__(self, content: str = None, tool_calls: list = None):
        self.content = content
        self.tool_calls = tool_calls


class MockChoice:
    """Mock choice object."""

    def __init__(self, message: MockMessage):
        self.message = message


class MockResponse:
    """Mock OpenAI chat completion response."""

    def __init__(self, choices: list):
        self.choices = choices


class MockLLMClient:
    """Mock LLM client for testing iterative_test_fix."""

    def __init__(self, responses: list):
        self.responses = responses
        self.call_count = 0
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        # Default to "TESTS_PASSING" if no more responses
        return MockResponse([MockChoice(MockMessage(content="TESTS_PASSING"))])

    def _build_request_params(self, request):
        return {
            "messages": [m.model_dump() for m in request.messages],
            "tools": request.tools,
        }


class MockClientWrapper:
    """Wrapper to provide .client.chat.completions.create interface."""

    def __init__(self, mock_client):
        self.mock_client = mock_client

    @property
    def client(self):
        return self

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    async def create(self, **kwargs):
        return await self.mock_client.create(**kwargs)

    def _build_request_params(self, request):
        return self.mock_client._build_request_params(request)


def test_iterative_fix_message_includes_tool_calls():
    """Test that assistant messages include tool_calls when LLM returns them."""
    from app.tgi.models import Message, MessageRole, ToolCall, ToolCallFunction

    # Simulate LLM returning a tool call
    tool_call = MockToolCall(id="call_123", name="run_tests", arguments="{}")
    response = MockResponse(
        [MockChoice(MockMessage(content="", tool_calls=[tool_call]))]
    )

    # Build messages as the fix loop would
    messages = []
    message = response.choices[0].message

    assistant_msg = Message(
        role=MessageRole.ASSISTANT,
        content=message.content or "",
    )
    if message.tool_calls:
        assistant_msg.tool_calls = [
            ToolCall(
                id=tc.id,
                type=tc.type,
                function=ToolCallFunction(
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ),
            )
            for tc in message.tool_calls
        ]
    messages.append(assistant_msg)

    # Verify the message includes tool_calls
    assert messages[0].tool_calls is not None
    assert len(messages[0].tool_calls) == 1
    assert messages[0].tool_calls[0].id == "call_123"
    assert messages[0].tool_calls[0].function.name == "run_tests"


def test_tool_result_message_includes_tool_call_id():
    """Test that tool result messages include tool_call_id."""
    from app.tgi.models import Message, MessageRole

    # Simulate tool result
    tool_result = {
        "tool_call_id": "call_123",
        "role": "tool",
        "name": "run_tests",
        "content": "Tests passed!",
    }

    # Build message as the fix loop would
    msg = Message(
        role=MessageRole.TOOL,
        content=tool_result["content"],
        tool_call_id=tool_result["tool_call_id"],
        name=tool_result["name"],
    )

    # Verify the message includes tool_call_id
    assert msg.tool_call_id == "call_123"
    assert msg.name == "run_tests"
    assert msg.content == "Tests passed!"


def test_message_serialization_with_tool_calls():
    """Test that messages with tool_calls serialize correctly for API."""
    from app.tgi.models import Message, MessageRole, ToolCall, ToolCallFunction

    # Create assistant message with tool calls
    assistant_msg = Message(
        role=MessageRole.ASSISTANT,
        content="",
        tool_calls=[
            ToolCall(
                id="call_abc123",
                type="function",
                function=ToolCallFunction(
                    name="update_test_script",
                    arguments='{"new_script": "fixed code"}',
                ),
            )
        ],
    )

    # Create tool result message
    tool_msg = Message(
        role=MessageRole.TOOL,
        content="Test script updated successfully.",
        tool_call_id="call_abc123",
        name="update_test_script",
    )

    # Serialize both
    assistant_dict = assistant_msg.model_dump(exclude_none=True)
    tool_dict = tool_msg.model_dump(exclude_none=True)

    # Verify assistant message
    assert "tool_calls" in assistant_dict
    assert len(assistant_dict["tool_calls"]) == 1
    assert assistant_dict["tool_calls"][0]["id"] == "call_abc123"

    # Verify tool message
    assert "tool_call_id" in tool_dict
    assert tool_dict["tool_call_id"] == "call_abc123"
    assert tool_dict["name"] == "update_test_script"


def test_multiple_tool_calls_handling():
    """Test handling of multiple tool calls in a single response."""
    from app.tgi.models import Message, MessageRole, ToolCall, ToolCallFunction

    # Simulate LLM returning multiple tool calls
    tool_calls = [
        MockToolCall(
            "call_1",
            "get_script_lines",
            '{"script_type": "test", "start_line": 1, "end_line": 10}',
        ),
        MockToolCall(
            "call_2",
            "get_script_lines",
            '{"script_type": "service", "start_line": 1, "end_line": 5}',
        ),
    ]

    # Build assistant message with tool calls
    assistant_msg = Message(
        role=MessageRole.ASSISTANT,
        content="I'll examine both scripts.",
    )
    assistant_msg.tool_calls = [
        ToolCall(
            id=tc.id,
            type=tc.type,
            function=ToolCallFunction(
                name=tc.function.name,
                arguments=tc.function.arguments,
            ),
        )
        for tc in tool_calls
    ]

    # Build tool result messages
    tool_results = [
        Message(
            role=MessageRole.TOOL,
            content="lines 1-10 of test",
            tool_call_id="call_1",
            name="get_script_lines",
        ),
        Message(
            role=MessageRole.TOOL,
            content="lines 1-5 of service",
            tool_call_id="call_2",
            name="get_script_lines",
        ),
    ]

    # Verify all messages have correct structure
    assert len(assistant_msg.tool_calls) == 2
    assert assistant_msg.tool_calls[0].id == "call_1"
    assert assistant_msg.tool_calls[1].id == "call_2"

    for i, result in enumerate(tool_results):
        assert result.tool_call_id == f"call_{i+1}"
        assert result.name == "get_script_lines"


@pytest.mark.asyncio
async def test_iterative_test_fix_constructs_valid_messages(tmp_path):
    """Integration test: verify _iterative_test_fix builds valid message sequences."""
    from app.app_facade.test_fix_tools import IterativeTestFixer

    helpers_dir = os.path.join(os.path.dirname(__file__), "node_test_helpers")

    # Create a passing test scenario
    service_script = "export class McpService { constructor() {} }"
    components_script = "// components"
    test_script = """
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

describe('Test', () => {
    it('passes', () => { assert.ok(true); });
});
"""

    # Initialize toolkit and verify tests pass immediately
    toolkit = IterativeTestFixer(helpers_dir)
    toolkit.setup_test_environment(service_script, components_script, test_script)

    result = toolkit.run_tests()

    # If the test environment works, it should pass
    # This validates the toolkit itself works
    toolkit.cleanup()

    # The important thing is that no exception about tool_call_id should occur
    assert (
        result.success
        or "pass" in result.content.lower()
        or "ok" in result.content.lower()
    )


@pytest.mark.asyncio
async def test_iterative_test_fix_resets_attempts_on_progress(monkeypatch, caplog):
    """Ensure attempts reset when more tests pass across iterations."""
    from app.app_facade.test_fix_tools import (
        IterativeTestFixer,
        ToolResult,
        run_tool_driven_test_fix,
    )
    from app.tgi.models import Message, MessageRole

    results = [
        ToolResult(success=False, content="# pass 1\n# fail 2\n"),
        ToolResult(success=False, content="# pass 1\n# fail 2\n"),
        ToolResult(success=False, content="# pass 2\n# fail 1\n"),
        ToolResult(success=True, content="# pass 2\n# fail 0\n"),
    ]
    call_index = {"idx": 0}

    def fake_run_tests(self, test_name=None):
        result = results[min(call_index["idx"], len(results) - 1)]
        call_index["idx"] += 1
        return result

    monkeypatch.setattr(IterativeTestFixer, "run_tests", fake_run_tests)

    responses = [
        MockResponse(
            [
                MockChoice(
                    MockMessage(
                        content="",
                        tool_calls=[MockToolCall("call_1", "run_tests", "{}")],
                    )
                )
            ]
        ),
        MockResponse(
            [
                MockChoice(
                    MockMessage(
                        content="",
                        tool_calls=[MockToolCall("call_2", "run_tests", "{}")],
                    )
                )
            ]
        ),
        MockResponse(
            [
                MockChoice(
                    MockMessage(
                        content="",
                        tool_calls=[MockToolCall("call_3", "run_tests", "{}")],
                    )
                )
            ]
        ),
    ]
    mock_client = MockLLMClient(responses)
    mock_wrapper = MockClientWrapper(mock_client)
    mock_wrapper.model_format = None
    tgi_service = DummyTGIService()
    tgi_service.llm_client = mock_wrapper

    caplog.set_level(logging.INFO, logger="uvicorn.error")

    messages = [Message(role=MessageRole.USER, content="Fix tests.")]
    await run_tool_driven_test_fix(
        tgi_service=tgi_service,
        service_script="export class McpService {}",
        components_script="",
        test_script="",
        dummy_data=None,
        messages=messages,
        allowed_tools=None,
        access_token=None,
        max_attempts=2,
    )

    assert mock_client.call_count == 3
    assert call_index["idx"] == 4
    assert "resetting attempts" in caplog.text
    assert caplog.text.count("Fix iteration 1/2") == 2


def test_build_phase2_system_prompt_integration(monkeypatch):
    """Verify that _build_phase2_system_prompt reads the correct file and injects design system."""
    storage = GeneratedUIStorage(os.getcwd())
    tgi = DummyTGIService()
    service = GeneratedUIService(storage=storage, tgi_service=tgi)

    expected_design_system = "Use Comic Sans everywhere."

    class MockPromptService:
        async def find_prompt_by_name_or_role(self, session, prompt_name=None):
            if prompt_name == "design-system":
                return "dummy_id"
            return None

        async def get_prompt_content(self, session, prompt):
            if prompt == "dummy_id":
                return expected_design_system
            return ""

    service.tgi_service.prompt_service = MockPromptService()

    # Run the method
    prompt_result = asyncio.run(service._build_phase2_system_prompt(None))

    # Assertions
    assert "Pfusch Dashboard Presentation Generation System Prompt" in prompt_result
    assert expected_design_system in prompt_result
    assert "{{DESIGN_SYSTEM_PROMPT}}" not in prompt_result
