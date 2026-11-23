"""示範腳本：存檔、讀回來、查過期、刪除不存在的項目，印出「結果」和「正確答案」。

mini-agent 會在修改前、修改後各跑一次，並排顯示差別。每行格式：`例子 => 結果 => 正確答案`。
（錯誤只比對例外類別，訊息文字不比對。）
"""

import tempfile
from datetime import date
from pathlib import Path

from todo_store import TodoStore


def show(label: str, fn, expected: str) -> None:
    """執行一個例子；就算程式當掉也印出來。"""
    try:
        result = str(fn())
    except Exception as e:  # noqa: BLE001
        result = f"{type(e).__name__}: {e}"
    print(f"{label} => {result} => {expected}")


with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "todos.json"
    store = TodoStore(path)
    store.add("write report", date(2024, 5, 1))
    store.save()

    reloaded = TodoStore(path)
    reloaded.load()
    show("存檔再讀回來，due 的型別", lambda: type(reloaded.todos[1].due).__name__, "date")
    show("2024-06-01 時過期的項目", lambda: [t.title for t in reloaded.overdue(date(2024, 6, 1))], "['write report']")
    show("刪除不存在的 id 99", lambda: reloaded.remove(99) or "沒有任何反應", "TodoNotFound")
