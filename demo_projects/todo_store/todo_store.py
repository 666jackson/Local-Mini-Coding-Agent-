"""A tiny JSON-backed todo store."""

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path


@dataclass
class Todo:
    id: int
    title: str
    due: date
    done: bool = False


class TodoNotFound(KeyError):
    """Raised when a todo id does not exist."""


class TodoStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.todos: dict = {}

    def add(self, title: str, due: date) -> Todo:
        new_id = max(self.todos, default=0) + 1
        todo = Todo(new_id, title, due)
        self.todos[new_id] = todo
        return todo

    def remove(self, todo_id: int) -> None:
        """Delete a todo; raise TodoNotFound if it does not exist."""
        self.todos.pop(todo_id, None)

    def save(self) -> None:
        data = [asdict(t) | {"due": t.due.isoformat()} for t in self.todos.values()]
        self.path.write_text(json.dumps(data, indent=2))

    def load(self) -> None:
        data = json.loads(self.path.read_text())
        self.todos = {d["id"]: Todo(**d) for d in data}

    def overdue(self, today: date) -> list:
        return [t for t in self.todos.values() if not t.done and t.due < today]
