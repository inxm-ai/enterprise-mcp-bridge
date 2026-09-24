from abc import ABC, abstractmethod
import os
import time
from typing import Callable

from app.session import MCPSessionBase


class SessionManagerBase(ABC):
    @abstractmethod
    def get(self, session_id: str) -> MCPSessionBase | None:
        pass  # pragma: no cover

    @abstractmethod
    def set(self, session_id: str, session: MCPSessionBase):
        pass  # pragma: no cover

    @abstractmethod
    def pop(self, session_id: str, default=None) -> MCPSessionBase | None:
        pass  # pragma: no cover

    @abstractmethod
    def pop_idle(self, max_idle_seconds: float) -> list[tuple[str, MCPSessionBase]]:
        """Remove and return every session unused for longer than max_idle_seconds."""
        pass  # pragma: no cover


def session_manager(
    name: str = os.getenv("MCP_SESSION_MANAGER", "InMemorySessionManager")
) -> SessionManagerBase:
    if name == "InMemorySessionManager":
        return InMemorySessionManager()
    cls = globals().get(name)
    if cls and issubclass(cls, SessionManagerBase):
        return cls()
    else:
        raise ValueError(f"Unknown session manager type: {name}")


class InMemorySessionManager(SessionManagerBase):
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._sessions: dict[str, MCPSessionBase | None] = {}
        self._last_used: dict[str, float] = {}
        self._clock = clock

    def get(self, session_id: str) -> MCPSessionBase | None:
        session = self._sessions.get(session_id)
        if session is not None:
            self._last_used[session_id] = self._clock()
        return session

    def set(self, session_id: str, session: MCPSessionBase):
        self._sessions[session_id] = session
        self._last_used[session_id] = self._clock()

    def pop(self, session_id: str, default=None) -> MCPSessionBase | None:
        self._last_used.pop(session_id, None)
        return self._sessions.pop(session_id, default)

    def pop_idle(self, max_idle_seconds: float) -> list[tuple[str, MCPSessionBase]]:
        cutoff = self._clock() - max_idle_seconds
        idle_ids = [
            session_id
            for session_id, last_used in self._last_used.items()
            if last_used < cutoff
        ]
        return [
            (session_id, session)
            for session_id in idle_ids
            if (session := self.pop(session_id)) is not None
        ]
