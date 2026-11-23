"""JSONL 稽核紀錄：每次工具呼叫一行，方便事後追查與評測統計。"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLog:
    """把事件 append 到 JSONL 檔；path 為 None 時只存在記憶體。"""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.events: list[dict[str, Any]] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        """寫入一筆事件，回傳寫入的內容。"""
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            **fields,
        }
        self.events.append(entry)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return entry

    def count(self, event: str, **match: Any) -> int:
        """統計符合條件的事件數，評測用。"""
        return sum(
            1 for e in self.events if e["event"] == event and all(e.get(k) == v for k, v in match.items())
        )


class Timer:
    """簡單計時器（毫秒）。"""

    def __init__(self) -> None:
        self.start = time.perf_counter()

    def ms(self) -> int:
        return int((time.perf_counter() - self.start) * 1000)
