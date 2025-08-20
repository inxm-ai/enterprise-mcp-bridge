from abc import ABC, abstractmethod
import os

from app.session import MCPSessionBase


class SessionManagerBase(ABC):
    @abstractmethod
    def get(self, session_id: str) -> MCPSessionBase | None:
        pass

    @abstractmethod
    def set(self, session_id: str, session: MCPSessionBase):
        pass

    @abstractmethod
    def pop(self, session_id: str, default=None) -> MCPSessionBase | None:
        pass


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
    def __init__(self):
        self._sessions: dict[str, MCPSessionBase | None] = {}

    def get(self, session_id: str) -> MCPSessionBase | None:
        return self._sessions.get(session_id)

    def set(self, session_id: str, session: MCPSessionBase):
        self._sessions[session_id] = session

    def pop(self, session_id: str, default=None) -> MCPSessionBase | None:
        return self._sessions.pop(session_id, default)
