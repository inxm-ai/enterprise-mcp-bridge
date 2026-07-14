from app.utils.traced_requests import traced_request


class _FakeSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class _FakeTracer:
    def __init__(self):
        self.span = _FakeSpan()

    def start_as_current_span(self, name):
        return self

    def __enter__(self):
        return self.span

    def __exit__(self, exc_type, exc, tb):
        return False


def test_traced_request_sets_user_id_when_present():
    tracer = _FakeTracer()

    with traced_request(
        tracer=tracer,
        operation="op",
        session_value="session-1",
        group=None,
        start_message="starting",
        user_id="user-42",
    ):
        pass

    assert tracer.span.attributes.get("user.id") == "user-42"


def test_traced_request_omits_user_id_when_absent():
    tracer = _FakeTracer()

    with traced_request(
        tracer=tracer,
        operation="op",
        session_value="session-1",
        group=None,
        start_message="starting",
    ):
        pass

    assert "user.id" not in tracer.span.attributes
