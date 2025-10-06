from app.tgi.behaviours.todos.todo_manager import TodoManager, TodoItem, TodoState


def test_add_and_list_todos():
    m = TodoManager()
    t1 = TodoItem(id="1", name="first", goal="do first")
    t2 = TodoItem(id="2", name="second", goal="do second")
    m.add_todos([t1, t2])

    todos = m.list_todos()
    assert len(todos) == 2
    assert todos[0].id == "1"
    assert todos[1].id == "2"


def test_start_and_finish_todo_and_history():
    m = TodoManager()
    t = TodoItem(id="x", name="task-x", goal="x goal")
    m.add_todos([t])

    assert m.get_todo("x").state == TodoState.TODO

    m.start_todo("x")
    assert m.get_todo("x").state == TodoState.IN_PROGRESS

    result = {"ok": True}
    m.finish_todo("x", result)
    assert m.get_todo("x").state == TodoState.DONE
    assert m.get_todo("x").result == result

    hist = m.history()
    # should contain added, start, finish events
    assert any(h.get("event") == "added" for h in hist)
    assert any(h.get("event") == "start" for h in hist)
    assert any(h.get("event") == "finish" for h in hist)


def test_clear():
    m = TodoManager()
    t = TodoItem(id="a", name="a", goal="g")
    m.add_todos([t])
    assert len(m.list_todos()) == 1
    m.clear()
    assert len(m.list_todos()) == 0
    assert m.history() == []
