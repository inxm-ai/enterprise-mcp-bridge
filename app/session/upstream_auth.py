"""Typed "the upstream wants the person to sign in" errors for remote MCP servers.

A remote MCP server that refuses a request for lack of authorization answers
the MCP POST with HTTP 401/403 and a ``WWW-Authenticate`` challenge. The MCP
SDK's streamable HTTP transport turns any such non-2xx answer into the
JSON-RPC error ``-32603 "Server returned an error response"`` and drops the
status, the headers and the body, so the bridge used to report an opaque,
retryable 502.

The bridge recovers that information itself:

- ``capture_auth_challenge`` is a response event hook on the HTTP client the
  bridge builds for every remote transport. The SDK sends each request in the
  context of the task that issued it, so the hook can write what it saw to a
  holder that the issuing ``send_request`` call set in a context variable.
- ``watch_auth_challenges`` wraps a ``ClientSession``'s ``send_request``: when
  the request fails with an ``MCPError`` and its last HTTP answer was an auth
  refusal, it raises ``UpstreamAuthRequiredError`` instead. A JSON-RPC error
  whose message plainly says authorization is required is treated the same.

Only bounded, sanitised fields leave this module: the status, the challenge's
``error`` code, ``scope`` and ``error_description`` (or the body's short
``message``), never bodies, headers or tokens.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from mcp.shared.exceptions import MCPError

logger = logging.getLogger("uvicorn.error")

UPSTREAM_AUTH_REQUIRED = "upstream_auth_required"
MCP_INTERNAL_ERROR = -32603
AUTH_STATUSES = (401, 403)
# A JSON-RPC error message that says this, on its own, is an auth refusal.
_AUTH_MESSAGE_RE = re.compile(
    r"\b(?:(?:authori[sz]ation|authentication|login|sign[ -]?in)[ _-]required"
    r"|unauthori[sz]ed|invalid_token)\b",
    re.IGNORECASE,
)
_MAX_HEADER_CHARS = 4096
_MAX_BODY_BYTES = 8192
_MAX_TEXT_CHARS = 200
_MAX_SCOPES = 32
_PARAM_RE = re.compile(
    r'([A-Za-z][A-Za-z0-9_-]*)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^\s,"]+))'
)
_CODE_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_SCOPE_RE = re.compile(r"^[A-Za-z0-9:._/-]{1,128}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
# Credential-looking strings never leave the bridge, even inside a message.
_SECRET_RES = (
    re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\b[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\b[A-Za-z0-9_-]{40,}={0,2}"),
)


@dataclass
class AuthChallenge:
    """What an upstream auth refusal said, bounded and sanitised."""

    status: Optional[int]
    error: Optional[str] = None
    description: Optional[str] = None
    scopes: list[str] = field(default_factory=list)


class UpstreamAuthRequiredError(MCPError):
    """The remote MCP server refused the request until the person signs in.

    An ``MCPError`` so MCP transports keep the SDK's exception contract; its
    message is the safe sentence callers show to the person.
    """

    def __init__(
        self, challenge: AuthChallenge, server: Optional[str], code: int
    ) -> None:
        self.challenge = challenge
        self.server = server
        super().__init__(
            code=code,
            message=auth_required_message(challenge, server),
            data={"code": UPSTREAM_AUTH_REQUIRED, "upstream_status": challenge.status},
        )

    @property
    def http_status(self) -> int:
        # 403 is the upstream's own "you are signed in but may not"; everything
        # else that says "authorization required" is a 401.
        return 403 if self.challenge.status == 403 else 401

    def detail(self, phase: str) -> dict[str, Any]:
        detail: dict[str, Any] = {
            "error": UPSTREAM_AUTH_REQUIRED,
            "message": self.message,
            "phase": phase,
            "exception_type": type(self).__name__,
            "upstream_status": self.challenge.status,
        }
        if self.server:
            detail["server"] = self.server
        if self.challenge.error:
            detail["upstream_error"] = self.challenge.error
        if self.challenge.description:
            detail["upstream_message"] = self.challenge.description
        if self.challenge.scopes:
            detail["required_scopes"] = list(self.challenge.scopes)
        return detail


def sanitize_text(value: Any) -> Optional[str]:
    """A short single-line text, with anything credential-like removed."""
    if not isinstance(value, str):
        return None
    text = _CONTROL_RE.sub(" ", value)
    for secret in _SECRET_RES:
        text = secret.sub("[redacted]", text)
    text = " ".join(text.split())
    if not text:
        return None
    if len(text) > _MAX_TEXT_CHARS:
        text = text[: _MAX_TEXT_CHARS - 1].rstrip() + "…"
    return text


def _code(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and _CODE_RE.match(value) else None


def _scopes(value: Any) -> list[str]:
    scopes: list[str] = []
    if not isinstance(value, str):
        return scopes
    for scope in value.split():
        if _SCOPE_RE.match(scope) and scope not in scopes:
            scopes.append(scope)
        if len(scopes) >= _MAX_SCOPES:
            break
    return scopes


def parse_www_authenticate(value: Optional[str]) -> dict[str, str]:
    """The auth-params of a ``WWW-Authenticate`` value (lower-cased names)."""
    if not value or len(value) > _MAX_HEADER_CHARS:
        return {}
    params: dict[str, str] = {}
    for name, quoted, token in _PARAM_RE.findall(value):
        raw = quoted if quoted or not token else token
        params.setdefault(name.lower(), re.sub(r"\\(.)", r"\1", raw))
    return params


def _body_fields(body: bytes) -> tuple[Optional[str], Optional[str]]:
    """``(error code, message)`` from a small JSON error body, if it has them."""
    try:
        parsed = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    error = parsed.get("error")
    if isinstance(error, dict):  # A JSON-RPC error object.
        return None, sanitize_text(error.get("message"))
    message = parsed.get("error_description") or parsed.get("message")
    return _code(error), sanitize_text(message)


async def challenge_from_response(response) -> AuthChallenge:
    """Read the bounded challenge of an HTTP 401/403 response."""
    params = parse_www_authenticate(response.headers.get("www-authenticate"))
    error = _code(params.get("error"))
    description = sanitize_text(params.get("error_description"))
    content_type = response.headers.get("content-type", "").lower()
    try:
        length = int(response.headers.get("content-length", ""))
    except ValueError:
        length = None
    if (
        content_type.startswith("application/json")
        and length is not None
        and length <= _MAX_BODY_BYTES
    ):
        try:
            # Read in full and cached: the SDK reads the same body afterwards.
            body_error, body_message = _body_fields(await response.aread())
        except Exception:
            body_error, body_message = None, None
        error = error or body_error
        description = description or body_message
    return AuthChallenge(
        status=response.status_code,
        error=error,
        description=description,
        scopes=_scopes(params.get("scope")),
    )


def challenge_from_headers(status: int, headers) -> AuthChallenge:
    """The challenge of an already-raised ``HTTPStatusError`` (headers only)."""
    params = parse_www_authenticate(headers.get("www-authenticate"))
    return AuthChallenge(
        status=status,
        error=_code(params.get("error")),
        description=sanitize_text(params.get("error_description")),
        scopes=_scopes(params.get("scope")),
    )


def auth_required_message(challenge: AuthChallenge, server: Optional[str]) -> str:
    name = server or "The remote service"
    if challenge.status == 403 or challenge.error == "insufficient_scope":
        sentence = (
            f"{name} refused this request: the connected account lacks access. "
            f"Connect (sign in to) {name} with an account that has the needed "
            "permissions, then try again."
        )
    else:
        sentence = (
            f"{name} requires you to connect your account (sign in) before this "
            "can run. Connect your account, then try again."
        )
    if challenge.description:
        sentence += f' The service said: "{challenge.description}"'
    return sentence


# Set by ``watch_auth_challenges`` for one request; written by the hook.
_capture: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "inxm_upstream_auth_capture", default=None
)


def _is_jsonrpc_request(request) -> bool:
    """Whether an HTTP request carries a JSON-RPC request (not a notification)."""
    if request.method != "POST":
        return False
    try:
        payload = json.loads(request.content)
    except Exception:
        return False
    return isinstance(payload, dict) and "id" in payload and "method" in payload


async def capture_auth_challenge(response) -> None:
    """httpx response hook: remember the last answer to this task's MCP request."""
    holder = _capture.get()
    if not _is_jsonrpc_request(response.request):
        return
    if response.status_code not in AUTH_STATUSES:
        # A later answer (e.g. after an OAuth refresh) supersedes a refusal.
        if holder is not None:
            holder["challenge"] = None
        return
    challenge = await challenge_from_response(response)
    logger.warning(
        "[RemoteMCP] Upstream auth challenge status=%s error=%s scopes=%s "
        "has_www_authenticate=%s",
        challenge.status,
        challenge.error,
        " ".join(challenge.scopes) or None,
        "www-authenticate" in response.headers,
    )
    if holder is not None:
        holder["challenge"] = challenge
        holder["refused"].set()


def add_auth_challenge_hook(client) -> Any:
    """Install ``capture_auth_challenge`` on an httpx client; returns the client."""
    hooks = client.event_hooks
    responses = list(hooks.get("response", []))
    if capture_auth_challenge not in responses:
        responses.append(capture_auth_challenge)
    client.event_hooks = {**hooks, "response": responses}
    return client


@contextmanager
def _capturing() -> Iterator[dict]:
    holder: dict = {"challenge": None, "refused": asyncio.Event()}
    token = _capture.set(holder)
    try:
        yield holder
    finally:
        _capture.reset(token)


def _server_label() -> Optional[str]:
    try:
        from app.utils.mcp_operation import downstream_server_label

        return downstream_server_label()
    except Exception:  # pragma: no cover - label is best effort
        return None


async def _until_answered_or_refused(call, holder: dict):
    """Run ``call``; give up on it once the upstream refused it.

    The legacy SSE transport answers requests on its event stream; a refused
    POST makes the SDK end its writer without ever answering the request.
    """
    task = asyncio.ensure_future(call)
    refused = asyncio.ensure_future(holder["refused"].wait())
    try:
        await asyncio.wait({task, refused}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        refused.cancel()
        if not task.done():
            task.cancel()
            # Waits without raising the task's own cancellation here.
            await asyncio.wait({task})
    if task.cancelled():
        raise MCPError(code=MCP_INTERNAL_ERROR, message="Upstream refused the request")
    return task.result()


def watch_auth_challenges(session, *, unanswered_on_refusal: bool = False) -> Any:
    """Make ``session`` raise ``UpstreamAuthRequiredError`` for auth refusals.

    Wraps the instance's ``send_request`` (every typed request goes through
    it); a session without one (test doubles) is returned unchanged. With
    ``unanswered_on_refusal`` (legacy SSE) a refused request is not waited for.
    """
    original = getattr(session, "send_request", None)
    if original is None or getattr(original, "_inxm_auth_watch", False):
        return session

    async def send_request(*args, **kwargs):
        with _capturing() as holder:
            try:
                if unanswered_on_refusal:
                    return await _until_answered_or_refused(
                        original(*args, **kwargs), holder
                    )
                return await original(*args, **kwargs)
            except UpstreamAuthRequiredError:
                raise
            except MCPError as exc:
                challenge = holder["challenge"]
                if challenge is None and _AUTH_MESSAGE_RE.search(exc.message or ""):
                    challenge = AuthChallenge(
                        status=None, description=sanitize_text(exc.message)
                    )
                if challenge is None:
                    raise
                raise UpstreamAuthRequiredError(
                    challenge, _server_label(), exc.code
                ) from exc

    send_request._inxm_auth_watch = True
    session.send_request = send_request
    return session
