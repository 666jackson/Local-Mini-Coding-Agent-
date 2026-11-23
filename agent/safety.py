"""安全防護：工作區路徑邊界、禁碰檔案、指令政策、人工確認、輸出截斷。

設計原則：安全在系統層擋，不靠模型自律。就算模型被 prompt injection
騙了，這裡的檢查仍然會拒絕危險操作。所有違規都以 SafetyError 或
("deny", 原因) 回報，訊息會原樣回傳給模型。
"""

from __future__ import annotations

import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

BLOCKED_PARTS = {".env", ".git", ".ssh"}
BLOCKED_NAME_RE = re.compile(r"(^\.env$|^\.env\.(?!example$).+|id_rsa|id_ed25519|id_ecdsa|\.pem$|\.key$|\.p12$)")

ALLOW_PREFIXES: list[list[str]] = [
    ["pytest"],
    ["python", "-m", "pytest"],
    ["python3", "-m", "pytest"],
    ["ls"],
    ["cat"],
    ["git", "status"],
    ["git", "diff"],
]

# 出現在指令任何位置就拒絕的程式名稱（以 argv token 的 basename 比對）。
DENY_PROGRAMS = {
    "sudo", "su", "doas", "curl", "wget", "ssh", "scp", "sftp", "rsync", "nc", "ncat",
    "netcat", "telnet", "ftp", "chmod", "chown", "chgrp", "mkfs", "dd", "shutdown",
    "reboot", "kill", "pkill", "killall", "crontab", "sh", "bash", "zsh", "fish",
    "eval", "exec", "nohup", "env", "xargs",
}
DENY_RE = re.compile(
    r"\b(sudo|curl|wget|ssh|scp|chmod|chown|mkfs)\b|\brm\s+(-[a-zA-Z]*[rRf][a-zA-Z]*\s*)+"
)
SHELL_OPS_RE = re.compile(r"[;&|><`\n]|\$\(|\$\{")

Decision = Literal["allow", "confirm", "deny"]


class SafetyError(Exception):
    """違反安全規則。訊息會回傳給模型。"""


@dataclass
class Workspace:
    """代表 agent 唯一能碰的資料夾。所有路徑都必須經過 resolve()。"""

    root: Path
    max_read_bytes: int = 200_000
    max_write_bytes: int = 100_000

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"工作區不存在：{self.root}")

    def contains(self, p: Path) -> bool:
        """p（已 resolve）是否在工作區內。"""
        return p == self.root or self.root in p.parents

    def resolve(self, rel: str) -> Path:
        """把模型給的路徑轉成工作區內的絕對路徑；違規丟 SafetyError。

        resolve() 會展開 symlink，所以指向工作區外的連結也會被擋下。
        """
        if not isinstance(rel, str) or not rel.strip():
            raise SafetyError("路徑不可為空")
        if "\x00" in rel:
            raise SafetyError("路徑含有非法字元")
        raw = Path(rel).expanduser()
        p = (raw if raw.is_absolute() else self.root / raw).resolve()
        if not self.contains(p):
            raise SafetyError(f"路徑超出工作區：{rel}")
        self._check_blocked(p, rel)
        return p

    def _check_blocked(self, p: Path, rel: str) -> None:
        parts = p.relative_to(self.root).parts
        if any(part in BLOCKED_PARTS for part in parts) or BLOCKED_NAME_RE.search(p.name):
            raise SafetyError(f"禁止存取的檔案：{rel}")

    def display(self, p: Path) -> str:
        """回傳相對於工作區的路徑字串，給模型與稽核紀錄看。"""
        rel = p.relative_to(self.root)
        return str(rel) if str(rel) != "." else "."


def _looks_like_path(token: str) -> bool:
    return (
        token.startswith(("/", "~", "."))
        or "/" in token
        or token in BLOCKED_PARTS
        or bool(BLOCKED_NAME_RE.search(token))
    )


def _check_path_args(argv: list[str], ws: Workspace) -> str | None:
    """檢查指令參數中的路徑：指向工作區外或禁碰檔案就回傳原因。"""
    for token in argv[1:]:
        candidates = [token]
        if token.startswith("-") and "=" in token:
            candidates.append(token.split("=", 1)[1])
        for cand in candidates:
            if cand.startswith("-") or not _looks_like_path(cand):
                continue
            try:
                ws.resolve(cand)
            except SafetyError as e:
                return f"指令參數違規：{e}"
    return None


def classify_command(cmd: str, ws: Workspace | None = None) -> tuple[Decision, str]:
    """把指令分成 allow / confirm / deny 三級，並附上原因。"""
    if not isinstance(cmd, str) or not cmd.strip():
        return "deny", "空指令"
    if SHELL_OPS_RE.search(cmd):
        return "deny", "不支援 shell 運算子（; && | > < ` $( 等），請一次執行一個指令"
    if DENY_RE.search(cmd):
        return "deny", "指令在禁止清單中"
    try:
        argv = shlex.split(cmd)
    except ValueError as e:
        return "deny", f"指令解析失敗：{e}"
    if not argv:
        return "deny", "空指令"
    if os.path.basename(argv[0]) in DENY_PROGRAMS:
        return "deny", f"程式 {argv[0]} 在禁止清單中"
    if ws is not None:
        reason = _check_path_args(argv, ws)
        if reason:
            return "deny", reason
    if any(argv[: len(p)] == p for p in ALLOW_PREFIXES):
        return "allow", "白名單"
    return "confirm", "不在白名單，需要使用者確認"


ConfirmFn = Callable[[str], bool]


def deny_all(_: str) -> bool:
    """非互動環境的預設：confirm 級一律拒絕。"""
    return False


@dataclass
class CommandPolicy:
    """結合 classify_command 與人工確認。

    auto_yes 只會自動同意 confirm 級，deny 永遠不放行。
    """

    ws: Workspace
    auto_yes: bool = False
    confirm_fn: ConfirmFn = field(default=deny_all)

    def decide(self, cmd: str) -> tuple[Decision, str, bool]:
        """回傳 (判定, 原因, 是否可執行)。"""
        decision, reason = classify_command(cmd, self.ws)
        if decision == "allow":
            return decision, reason, True
        if decision == "deny":
            return decision, reason, False
        if self.auto_yes:
            return decision, "confirm 級，--yes 自動同意", True
        approved = self.confirm_fn(cmd)
        return decision, ("使用者同意" if approved else "使用者拒絕"), approved


def minimal_env() -> dict[str, str]:
    """子程序只拿到必要的環境變數，不帶任何金鑰。"""
    py_bin = str(Path(sys.executable).parent)
    path = os.pathsep.join([py_bin, os.environ.get("PATH", "/usr/bin:/bin")])
    return {
        "PATH": path,
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }


def truncate_middle(text: str, limit: int = 4000, head_ratio: float = 2 / 3) -> str:
    """超過上限時保留頭尾、中間省略，避免塞爆 context。

    指令輸出（錯誤訊息通常在最後）可用較小的 head_ratio 保留較多結尾。
    """
    if len(text) <= limit:
        return text
    head = int(limit * head_ratio)
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n...（中間省略 {omitted} 字元）...\n{text[-tail:]}"
