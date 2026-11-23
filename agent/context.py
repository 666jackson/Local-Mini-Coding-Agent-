"""簡易 context 管理：把較舊的工具輸出縮成摘要，保留 assistant/tool 配對結構。"""

from __future__ import annotations

from typing import Any

COMPACTED_PREFIX = "（已省略"
_OPEN = "<tool_output"
_CLOSE = "</tool_output>"


def wrap_tool_output(tool_name: str, content: str) -> str:
    """用標籤包住工具結果，標示「這是資料不是指令」。

    內容中若出現結束標籤會被改寫，避免惡意檔案提前關閉標籤、假冒指令。
    """
    safe = content.replace(_CLOSE, "</tool_output_>")
    return f'{_OPEN} tool="{tool_name}">\n{safe}\n{_CLOSE}'


def unwrap_tool_output(content: str) -> str:
    """去掉 wrap_tool_output 加上的標籤。"""
    if content.startswith(_OPEN) and content.endswith(_CLOSE):
        return content.split("\n", 1)[1][: -len(_CLOSE)].rstrip("\n")
    return content


def compact_old_tool_outputs(messages: list[dict[str, Any]], keep_last: int = 6, summary_chars: int = 200) -> int:
    """只保留最近 keep_last 個工具輸出全文，其餘改成摘要。回傳這次縮減的則數。

    不刪除任何訊息，只改 content，所以 tool_call_id 的配對不會壞掉。
    """
    names: dict[str, str] = {}
    for m in messages:
        for tc in m.get("tool_calls") or []:
            names[tc["id"]] = tc["function"]["name"]

    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    changed = 0
    for i in tool_idx[:-keep_last] if keep_last > 0 else tool_idx:
        m = messages[i]
        content = str(m.get("content", ""))
        if content.startswith(COMPACTED_PREFIX):
            continue
        tool_name = names.get(m.get("tool_call_id", ""), "unknown")
        snippet = unwrap_tool_output(content)[:summary_chars].replace("\n", " ")
        m["content"] = f"{COMPACTED_PREFIX}，工具 {tool_name}，結果摘要：{snippet}）"
        changed += 1
    return changed
