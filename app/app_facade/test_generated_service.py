import asyncio
import json
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
        _get_fallback_prompt,
    )

    s = _load_pfusch_prompt()
    assert "Pfusch" in s or "You are a microsite" in s
    fb = _get_fallback_prompt()
    assert "MCP_BASE_PATH" in fb or "Base Url" in fb


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

    # get with wrong actor -> should raise
    with pytest.raises(Exception):
        service.get_ui(
            scope=scope, actor=Actor(user_id="x", groups=[]), ui_id="eid", name="n1"
        )


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
    payload = {"html": {"page": "<p>z</p>"}, "metadata": {}}
    json_str = json.dumps(payload, ensure_ascii=False)
    # split into two chunks
    c1 = json_str[: len(json_str) // 2]
    c2 = json_str[len(json_str) // 2 :]
    fake_chunk_reader.items = [
        FakeParsed(content=c1),
        FakeParsed(content=c2),
        FakeParsed(is_done=True),
    ]

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
