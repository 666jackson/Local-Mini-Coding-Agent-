from datetime import date

import pytest

from todo_store import TodoNotFound, TodoStore


def test_add_assigns_ids(tmp_path):
    s = TodoStore(tmp_path / "t.json")
    assert s.add("a", date(2024, 1, 1)).id == 1
    assert s.add("b", date(2024, 1, 2)).id == 2


def test_roundtrip_keeps_date_type(tmp_path):
    s = TodoStore(tmp_path / "t.json")
    s.add("write report", date(2024, 5, 1))
    s.save()
    s2 = TodoStore(tmp_path / "t.json")
    s2.load()
    assert s2.todos[1].due == date(2024, 5, 1)
    assert isinstance(s2.todos[1].due, date)


def test_overdue_after_reload(tmp_path):
    s = TodoStore(tmp_path / "t.json")
    s.add("old", date(2020, 1, 1))
    s.add("future", date(2999, 1, 1))
    s.save()
    s2 = TodoStore(tmp_path / "t.json")
    s2.load()
    assert [t.title for t in s2.overdue(date(2024, 1, 1))] == ["old"]


def test_remove(tmp_path):
    s = TodoStore(tmp_path / "t.json")
    s.add("a", date(2024, 1, 1))
    s.remove(1)
    assert s.todos == {}


def test_remove_missing_raises(tmp_path):
    s = TodoStore(tmp_path / "t.json")
    with pytest.raises(TodoNotFound):
        s.remove(42)
