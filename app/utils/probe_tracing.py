"""Keep Kubernetes probes and Prometheus scrapes out of the trace store.

Probe traffic arrives every few seconds on every pod and, traced, makes up a
large share of all spans even on an idle install. Two mechanisms drop it:

* ``probe_excluded_urls`` builds the ``excluded_urls`` list for
  ``FastAPIInstrumentor``, so the ASGI middleware skips the request span and
  its ``http send``/``http receive`` children for ``/metrics``, the
  ``/healthz`` endpoints and the (base-path) root used as liveness target.
* ``untraced`` runs a block under an unsampled parent, so spans created inside
  it (e.g. the MCP SDK's ``MCP send initialize``/``tools/list`` from the child
  health check) are never recorded. It also marks the context with
  ``_PROBE_SUPPRESSED_KEY``, which ``ProbeSuppressingSampler`` turns into a
  drop whatever sampler the deployment configured (``always_on`` ignores the
  unsampled parent).
"""

import re
import secrets
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional

# Importable without OTel, like server.py's guarded block: an install missing
# a package starts untraced instead of failing here.
try:
    from opentelemetry import context as otel_context
    from opentelemetry import trace
except ImportError:  # pragma: no cover - exercised in a subprocess test
    otel_context = trace = None  # type: ignore[assignment]

try:
    from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
except ImportError:  # pragma: no cover - API-only install has no SDK sampler
    Decision = Sampler = SamplingResult = None  # type: ignore[assignment,misc]

# Application-owned marker for "inside untraced()"; OTel's own suppression
# keys are private, so the sampler checks this one instead.
_PROBE_SUPPRESSED_KEY = (
    otel_context.create_key("enterprise-mcp-bridge.probe-suppressed")
    if otel_context is not None
    else None
)

if Sampler is not None:

    class ProbeSuppressingSampler(Sampler):
        """Drop spans started inside ``untraced()``; defer everything else.

        Wraps the sampler the SDK resolved from the usual OTel settings
        (``OTEL_TRACES_SAMPLER``/``_ARG``), so normal traffic is sampled
        exactly as configured.
        """

        def __init__(self, delegate: "Sampler") -> None:
            self._delegate = delegate

        @property
        def delegate(self) -> "Sampler":
            return self._delegate

        def should_sample(
            self,
            parent_context,
            trace_id,
            name,
            kind=None,
            attributes=None,
            links=None,
            trace_state=None,
        ) -> "SamplingResult":
            if otel_context.get_value(_PROBE_SUPPRESSED_KEY, parent_context):
                return SamplingResult(Decision.DROP)
            return self._delegate.should_sample(
                parent_context,
                trace_id,
                name,
                kind=kind,
                attributes=attributes,
                links=links,
                trace_state=trace_state,
            )

        def get_description(self) -> str:
            return self._delegate.get_description()

else:  # pragma: no cover - exercised in a subprocess test
    ProbeSuppressingSampler = None  # type: ignore[assignment,misc]

try:
    from opentelemetry.instrumentation.utils import suppress_instrumentation
except ImportError:

    @contextmanager
    def suppress_instrumentation() -> Iterator[None]:  # type: ignore[misc]
        """No instrumentation installed, so there is nothing to suppress."""
        yield


# Paths, relative to a base path, that only probes and scrapers call. The
# empty entry is the base-path root itself (``home``), used as liveness probe.
_PROBE_SUBPATHS = ("", "healthz", "healthz/child", "metrics")


def probe_excluded_urls(
    base_paths: Iterable[Optional[str]], extra: Optional[str] = None
) -> str:
    """Return a comma-separated regex list for ``excluded_urls``.

    The instrumentation matches each regex with ``re.search`` against
    ``scheme://host[:port]/path`` (no query string), so every pattern is
    anchored on the authority and the end of the path: a tool named
    ``metrics`` under ``/tools/metrics`` stays traced. ``extra`` (the user's
    ``OTEL_PYTHON_FASTAPI_EXCLUDED_URLS``/``OTEL_PYTHON_EXCLUDED_URLS``) is
    appended unchanged.
    """
    prefixes = sorted({(path or "").strip().rstrip("/") for path in base_paths} | {""})
    # The instrumentation splits the list on commas, so a comma in a base path
    # is written as \x2c, which the regex still matches as a comma.
    prefix_alt = "|".join(
        re.escape(prefix).replace(",", r"\x2c") for prefix in prefixes
    )
    sub_alt = "|".join(re.escape(sub) for sub in _PROBE_SUBPATHS if sub)
    pattern = rf"^[a-z]+://[^/]+(?:{prefix_alt})(?:/(?:{sub_alt}))?/?$"
    extra = (extra or "").strip()
    return f"{pattern},{extra}" if extra else pattern


@contextmanager
def untraced() -> Iterator[None]:
    """Run the block without recording any span.

    The tracing SDK does not honour the suppress-instrumentation key for
    manually created spans (the MCP SDK starts its own), so the block also
    runs under a valid but unsampled parent: the default parent-based sampler
    then drops every span started inside it. The probe-suppression key lets
    ``ProbeSuppressingSampler`` drop them under any other sampler too. The
    suppress key covers instrumentations (httpx, requests, ...) that check
    it. Without the OTel API nothing is traced, so the block just runs.
    """
    if trace is None:
        yield
        return
    parent = trace.NonRecordingSpan(
        trace.SpanContext(
            trace_id=secrets.randbits(128) or 1,
            span_id=secrets.randbits(64) or 1,
            is_remote=False,
            trace_flags=trace.TraceFlags(trace.TraceFlags.DEFAULT),
        )
    )
    token = otel_context.attach(
        otel_context.set_value(
            _PROBE_SUPPRESSED_KEY, True, trace.set_span_in_context(parent)
        )
    )
    try:
        with suppress_instrumentation():
            yield
    finally:
        otel_context.detach(token)
