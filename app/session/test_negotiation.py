"""The downstream session opens with the handshake by default and with server/discover on request."""

import logging

import pytest

from app.session import client_strategy


class _Session:
    def __init__(self, discover_supported: bool):
        self.calls = []
        self.protocol_version = None
        self.server_info = None
        if discover_supported:
            self.discover = self._discover

    async def initialize(self):
        self.calls.append("initialize")
        self.protocol_version = "2025-11-25"

    async def _discover(self):
        self.calls.append("discover")
        self.protocol_version = "2026-07-28"


@pytest.mark.asyncio
async def test_legacy_negotiation_handshakes_even_when_discover_exists(monkeypatch):
    monkeypatch.setattr(client_strategy, "MCP_PROTOCOL_NEGOTIATION", "legacy")
    session = _Session(discover_supported=True)

    await client_strategy._negotiate(session)

    assert session.calls == ["initialize"]


@pytest.mark.asyncio
async def test_auto_negotiation_uses_the_sdk_policy(monkeypatch):
    monkeypatch.setattr(client_strategy, "MCP_PROTOCOL_NEGOTIATION", "auto")
    seen = []

    async def fake_negotiate_auto(session):
        seen.append(session)
        await session.discover()

    monkeypatch.setattr(client_strategy, "negotiate_auto", fake_negotiate_auto)
    session = _Session(discover_supported=True)

    await client_strategy._negotiate(session)

    assert seen == [session] and session.calls == ["discover"]


@pytest.mark.asyncio
async def test_the_negotiated_revision_is_logged(monkeypatch, caplog):
    monkeypatch.setattr(client_strategy, "MCP_PROTOCOL_NEGOTIATION", "legacy")
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    session = _Session(discover_supported=False)

    await client_strategy._negotiate(session)

    assert "[MCP] Protocol 2025-11-25 negotiated with unnamed server" in caplog.text
