"""A blob resource reaches the caller as its bytes, not as the base64 the protocol carries."""

import base64
import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from mcp import types

from app.routes import get_resource_details

DECK = b"PK\x03\x04" + bytes(range(256)) * 3
PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


class _SessionContext:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _session_serving(contents):
    session = SimpleNamespace()

    async def list_resources():
        return SimpleNamespace(
            resources=[SimpleNamespace(name="deck.pptx", uri="pptx://deck/abc")]
        )

    async def read_resource(uri):
        assert uri == "pptx://deck/abc"
        return types.ReadResourceResult(contents=[contents])

    session.list_resources = list_resources
    session.read_resource = read_resource
    return session


async def _body(response) -> bytes:
    if hasattr(response, "body_iterator"):
        return b"".join([chunk async for chunk in response.body_iterator])
    return response.body


@pytest.mark.asyncio
async def test_blob_resource_is_streamed_decoded():
    contents = types.BlobResourceContents(
        uri="pptx://deck/abc",
        mimeType=PPTX,
        blob=base64.b64encode(DECK).decode("ascii"),
    )
    with patch(
        "app.routes.mcp_session_context",
        lambda *args, **kwargs: _SessionContext(_session_serving(contents)),
    ):
        response = await get_resource_details(
            "deck.pptx",
            request=MagicMock(headers={}),
            access_token=None,
            x_inxm_mcp_session_header=None,
            x_inxm_mcp_session_cookie=None,
            group=None,
        )

    assert response.media_type == PPTX
    body = await _body(response)
    assert body[:2] == b"PK"
    assert hashlib.sha256(body).hexdigest() == hashlib.sha256(DECK).hexdigest()


@pytest.mark.asyncio
async def test_zero_byte_blob_is_served_empty_under_its_media_type():
    """An empty file is still a file: the declared type with an empty body, not a 204."""
    contents = types.BlobResourceContents(uri="pptx://deck/abc", mimeType=PPTX, blob="")
    with patch(
        "app.routes.mcp_session_context",
        lambda *args, **kwargs: _SessionContext(_session_serving(contents)),
    ):
        response = await get_resource_details(
            "deck.pptx",
            request=MagicMock(headers={}),
            access_token=None,
            x_inxm_mcp_session_header=None,
            x_inxm_mcp_session_cookie=None,
            group=None,
        )

    assert response.status_code == 200
    assert response.media_type == PPTX
    assert await _body(response) == b""


@pytest.mark.asyncio
async def test_text_resource_is_still_served_as_text():
    contents = types.TextResourceContents(
        uri="pptx://deck/abc", mimeType="text/plain", text="hello"
    )
    with patch(
        "app.routes.mcp_session_context",
        lambda *args, **kwargs: _SessionContext(_session_serving(contents)),
    ):
        response = await get_resource_details(
            "deck.pptx",
            request=MagicMock(headers={}),
            access_token=None,
            x_inxm_mcp_session_header=None,
            x_inxm_mcp_session_cookie=None,
            group=None,
        )

    assert response.body == b"hello"
