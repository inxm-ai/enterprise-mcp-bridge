from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TodoState(str, Enum):
    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"


@dataclass
class TodoItem:
    id: str
    name: str
    goal: str
    needed_info: Optional[Any] = None
    expected_result: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    state: TodoState = TodoState.TODO
    result: Optional[Any] = None
    parent_id: Optional[str] = None
    root_id: Optional[str] = None
    depth: int = 0


class TodoManager:
    """In-memory manager for todo items and history.

    Simple, testable manager. Not persisted.
    """

    def __init__(self):
        self._todos: List[TodoItem] = []
        self._history: List[Dict[str, Any]] = []

    def add_todos(self, todos: List[TodoItem]) -> None:
        for t in todos:
            self._todos.append(t)
            self._history.append({"event": "added", "id": t.id, "todo": t})

    def list_todos(self) -> List[TodoItem]:
        return list(self._todos)

    def index_of(self, todo_id: str) -> Optional[int]:
        for idx, t in enumerate(self._todos):
            if t.id == todo_id:
                return idx
        return None

    def get_todo(self, todo_id: str) -> Optional[TodoItem]:
        for t in self._todos:
            if t.id == todo_id:
                return t
        return None

    def start_todo(self, todo_id: str) -> None:
        t = self.get_todo(todo_id)
        if not t:
            raise KeyError(todo_id)
        t.state = TodoState.IN_PROGRESS
        self._history.append({"event": "start", "id": t.id})

    def finish_todo(self, todo_id: str, result: Any) -> None:
        t = self.get_todo(todo_id)
        if not t:
            raise KeyError(todo_id)
        t.state = TodoState.DONE
        t.result = result
        self._history.append({"event": "finish", "id": t.id, "result": result})

    def insert_todos_after(self, todo_id: str, todos: List[TodoItem]) -> None:
        if not todos:
            return
        idx = self.index_of(todo_id)
        if idx is None:
            raise KeyError(todo_id)
        insert_at = idx + 1
        for offset, todo in enumerate(todos):
            self._todos.insert(insert_at + offset, todo)
            self._history.append({"event": "added", "id": todo.id, "todo": todo})

    def close_todos_after(self, todo_id: str, reason: Optional[str] = None) -> int:
        idx = self.index_of(todo_id)
        if idx is None:
            raise KeyError(todo_id)
        closed = 0
        for todo in self._todos[idx + 1 :]:
            if todo.state == TodoState.DONE:
                continue
            todo.state = TodoState.DONE
            todo.result = {"closed": True, "reason": reason}
            self._history.append(
                {"event": "closed", "id": todo.id, "result": todo.result}
            )
            closed += 1
        return closed

    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def clear(self) -> None:
        self._todos = []
        self._history = []
