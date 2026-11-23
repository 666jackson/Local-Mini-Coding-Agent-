"""四個工具（read_file、write_file、run_command、search）、JSON schema 與分派。

所有工具失敗時都回傳 "ERROR: ..." 字串給模型，不丟例外中斷迴圈。
"""

from __future__ import annotations

import difflib
import fnmatch
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .audit import AuditLog, Timer
from .safety import CommandPolicy, SafetyError, Workspace, minimal_env, truncate_middle

MAX_OUTPUT_CHARS = 4000
MAX_READ_LINES = 300
MAX_SEARCH_RESULTS = 50
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", ".mypy_cache"}
# 測試檔與示範腳本：agent 改了它們就能讓「驗收」或「修改前後對照」失去意義
DEFAULT_PROTECTED = ("test_*.py", "*_test.py", "tests/*", "conftest.py", "demo.py")
TEST_CMD_RE = re.compile(r"(^|\s|/)(pytest|py\.test)(\s|$)")

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file in the workspace. Returns numbered lines. "
            "Use start_line/end_line for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                    "start_line": {"type": "integer", "description": "First line (1-based, optional)."},
                    "end_line": {"type": "integer", "description": "Last line, inclusive (optional)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the COMPLETE new content. "
            "Always send the whole file, not a fragment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                    "content": {"type": "string", "description": "Full file content."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run ONE shell-free command in the workspace, e.g. 'python -m pytest -q'. "
            "No pipes, redirects or chaining. Returns exit code, stdout and stderr.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The command line."}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Regex search over workspace files. Returns 'file:line:text', max 50 hits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Python regular expression."},
                    "glob": {"type": "string", "description": "Optional file glob, e.g. '*.py'."},
                },
                "required": ["pattern"],
            },
        },
    },
]

# 小模型常用錯的參數名稱 → 正確名稱（實際觀察：llama3.1:8b 用 filename 取代 path）
ARG_ALIASES: dict[str, str] = {
    "filename": "path", "file": "path", "file_path": "path", "filepath": "path", "file_name": "path",
    "cmd": "command", "query": "pattern", "regex": "pattern", "text": "content", "code": "content",
    "file_pattern": "glob", "start": "start_line", "end": "end_line",
}

REQUIRED_ARGS: dict[str, list[str]] = {
    s["function"]["name"]: s["function"]["parameters"]["required"] for s in TOOL_SCHEMAS
}


# ---------------------------------------------------------------- runners

@dataclass
class CommandOutput:
    """子程序執行結果。"""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class CommandRunner(Protocol):
    """執行已通過政策檢查的指令。"""

    def run(self, argv: list[str], ws: Workspace, timeout: int) -> CommandOutput: ...


def _exec(argv: list[str], cwd: Path, timeout: int) -> CommandOutput:
    """以 shell=False、最小環境變數執行，逾時與找不到程式都轉成 CommandOutput。"""
    try:
        proc = subprocess.run(
            argv,
            shell=False,
            cwd=cwd,
            env=minimal_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return CommandOutput(-1, out, f"指令逾時（{timeout} 秒）", timed_out=True)
    except FileNotFoundError:
        return CommandOutput(127, "", f"找不到程式：{argv[0]}")
    return CommandOutput(proc.returncode, proc.stdout, proc.stderr)


class LocalRunner:
    """直接在本機執行：shell=False、cwd=工作區、最小環境變數。盡力而為的隔離。"""

    def run(self, argv: list[str], ws: Workspace, timeout: int) -> CommandOutput:
        return _exec(argv, ws.root, timeout)


class DockerRunner:
    """在拋棄式容器內執行：無網路、只掛載工作區、限制資源。真正的隔離。"""

    def __init__(self, image: str = "mini-agent-sandbox:latest", memory: str = "512m", cpus: str = "1") -> None:
        self.image = image
        self.memory = memory
        self.cpus = cpus

    def build_argv(self, argv: list[str], ws: Workspace) -> list[str]:
        """組出 docker run 指令（獨立出來方便測試）。"""
        uid_gid = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "1000:1000"
        return [
            "docker", "run", "--rm", "--network", "none",
            "--memory", self.memory, "--cpus", self.cpus, "--pids-limit", "256",
            "--read-only", "--tmpfs", "/tmp:rw,size=64m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", uid_gid,
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp",
            "-v", f"{ws.root}:/work:rw", "-w", "/work",
            self.image, *argv,
        ]

    def run(self, argv: list[str], ws: Workspace, timeout: int) -> CommandOutput:
        return _exec(self.build_argv(argv, ws), ws.root, timeout + 15)


# ---------------------------------------------------------------- toolbox

@dataclass
class ToolResult:
    """一次工具呼叫的結果，content 會回給模型。"""

    name: str
    content: str
    ok: bool
    decision: str | None = None
    reason: str | None = None


class Toolbox:
    """四個工具的實作與分派，並維護「寫過檔但還沒驗證」的狀態。"""

    def __init__(
        self,
        ws: Workspace,
        policy: CommandPolicy | None = None,
        *,
        cmd_timeout: int = 30,
        runner: CommandRunner | None = None,
        audit: AuditLog | None = None,
        protected_globs: tuple[str, ...] = (),
    ) -> None:
        self.ws = ws
        self.protected_globs = protected_globs
        self.policy = policy or CommandPolicy(ws)
        self.cmd_timeout = cmd_timeout
        self.runner: CommandRunner = runner or LocalRunner()
        self.audit = audit or AuditLog()
        self.schemas = TOOL_SCHEMAS
        self.files_written: list[str] = []
        self.originals: dict[str, str | None] = {}  # 每個檔案第一次被改之前的內容；None 代表原本不存在
        self.needs_verification = False
        self.tests_passed = False
        self.tested_since_write = False  # 改檔後是否已經跑過測試（不論成敗）

    # ---------- dispatch ----------

    def dispatch(self, name: str, arguments: str | dict[str, Any] | None, *, source: str = "tool_calls") -> ToolResult:
        """解析參數並呼叫工具；任何錯誤都轉成 ERROR 字串。"""
        timer = Timer()
        args: dict[str, Any] = {}
        self._aliased: list[str] = []
        try:
            args = arguments if isinstance(arguments, dict) else {"_raw": arguments}
            args = self._parse_args(name, arguments)
            result = getattr(self, f"_tool_{name}")(**args)
        except _ToolError as e:
            result = ToolResult(name, f"ERROR: {e}", ok=False)
        except SafetyError as e:
            result = ToolResult(name, f"ERROR: 安全規則拒絕：{e}", ok=False, decision="deny", reason=str(e))
        except Exception as e:  # noqa: BLE001 - 工具不能讓迴圈崩潰
            result = ToolResult(name, f"ERROR: {type(e).__name__}: {e}", ok=False)
        result.content = truncate_middle(result.content, MAX_OUTPUT_CHARS)
        self.audit.record(
            "tool_call",
            tool=name,
            args=_summarize_args(args),
            source=source,
            decision=result.decision,
            reason=result.reason,
            ok=result.ok,
            result=result.content[:200],
            aliased=self._aliased or None,
            ms=timer.ms(),
        )
        return result

    def _parse_args(self, name: str, arguments: str | dict[str, Any] | None) -> dict[str, Any]:
        if name not in REQUIRED_ARGS:
            raise _ToolError(f"未知工具 '{name}'。可用工具：{', '.join(REQUIRED_ARGS)}")
        if isinstance(arguments, dict):
            args = arguments
        else:
            try:
                args = json.loads(arguments or "{}")
            except json.JSONDecodeError as e:
                raise _ToolError(f"arguments 不是合法 JSON（{e.msg}），請依 schema 重新呼叫") from None
        if not isinstance(args, dict):
            raise _ToolError("arguments 必須是 JSON 物件，請依 schema 重新呼叫")
        allowed = set(_schema_props(name))
        fixed: dict[str, Any] = {}
        for k, v in args.items():
            if k not in allowed and ARG_ALIASES.get(k) in allowed and ARG_ALIASES[k] not in args:
                self._aliased.append(f"{k}->{ARG_ALIASES[k]}")
                k = ARG_ALIASES[k]
            fixed[k] = v
        missing = [k for k in REQUIRED_ARGS[name] if k not in fixed]
        if missing:
            got = ", ".join(fixed) or "（沒有參數）"
            raise _ToolError(
                f"{name} 缺少必填參數：{', '.join(missing)}。你傳了：{got}；正確的參數名稱是：{', '.join(allowed)}"
            )
        return {k: v for k, v in fixed.items() if k in allowed}

    # ---------- tools ----------

    def _tool_read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> ToolResult:
        p = self.ws.resolve(path)
        if not p.is_file():
            raise _ToolError(f"檔案不存在：{path}")
        size = p.stat().st_size
        with p.open("rb") as f:
            raw = f.read(self.ws.max_read_bytes)
        if b"\x00" in raw[:4096]:
            raise _ToolError(f"{path} 看起來是二進位檔，無法讀取")
        lines = raw.decode("utf-8", errors="replace").splitlines()
        total = len(lines)
        start = max(1, int(start_line or 1))
        end = min(total, int(end_line) if end_line else start + MAX_READ_LINES - 1)
        if start > max(total, 1):
            raise _ToolError(f"start_line {start} 超過檔案行數 {total}")
        chunk = lines[start - 1 : end]
        body = "\n".join(f"{i:>4}| {line}" for i, line in enumerate(chunk, start=start))
        notes = []
        if end < total:
            notes.append(f"（只顯示第 {start}-{end} 行，共 {total} 行；用 start_line/end_line 讀其他部分）")
        if size > self.ws.max_read_bytes:
            notes.append(f"（檔案 {size} bytes 超過讀取上限，只讀了前 {self.ws.max_read_bytes} bytes）")
        return ToolResult("read_file", "\n".join([body, *notes]) if notes else body, ok=True)

    def _tool_write_file(self, path: str, content: str) -> ToolResult:
        if not isinstance(content, str):
            raise _ToolError("content 必須是字串")
        p = self.ws.resolve(path)
        rel = self.ws.display(p)
        if any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(p.name, g) for g in self.protected_globs):
            return ToolResult(
                "write_file",
                f"ERROR: {rel} 是受保護的檔案（測試檔或示範腳本），不能修改或新增。請修改被測試的程式碼。",
                ok=False, decision="deny", reason="protected file",
            )
        notes: list[str] = []
        if p.suffix == ".py":
            content, unescaped = _maybe_unescape(content)
            if unescaped:
                notes.append("（偵測到雙重跳脫的內容，已自動還原成正常換行）")
            try:
                compile(content, rel, "exec")
            except SyntaxError as e:
                bad = (e.text or "").rstrip()
                raise _ToolError(
                    f"內容有 Python 語法錯誤，檔案未寫入（原檔保持不變）：第 {e.lineno} 行 {e.msg}: {bad!r}"
                ) from None
        data = content.encode("utf-8")
        if len(data) > self.ws.max_write_bytes:
            raise _ToolError(f"內容 {len(data)} bytes 超過寫入上限 {self.ws.max_write_bytes}")
        if p.is_dir():
            raise _ToolError(f"{path} 是資料夾")
        old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
        if p.exists() and old == content:
            return ToolResult(
                "write_file",
                f"NO CHANGE: {self.ws.display(p)} already has exactly this content, nothing was written. "
                "If tests still fail, read the error message and change your approach.",
                ok=False,
            )
        self.originals.setdefault(rel, old if p.exists() else None)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        self.files_written.append(rel)
        self.needs_verification = True
        self.tests_passed = False
        self.tested_since_write = False
        diff = list(difflib.unified_diff(old.splitlines(), content.splitlines(), lineterm="", n=0))
        added = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
        removed = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))
        preview = "\n".join(d for d in diff[2:22] if not d.startswith("@@"))
        verb = "建立" if not old else "覆寫"
        summary = f"已{verb} {rel}：{len(content.splitlines())} 行（+{added} / -{removed}）"
        return ToolResult("write_file", "\n".join([summary, *notes, preview]).rstrip(), ok=True)

    def changes(self, context: int = 3) -> list[tuple[str, str]]:
        """這次執行改過的檔案，與原始內容比較的 unified diff：[(相對路徑, diff 文字)]。

        改了又改回原樣的檔案不列出。
        """
        out: list[tuple[str, str]] = []
        for rel, before in self.originals.items():
            p = self.ws.root / rel
            after = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
            if before is not None and before == after:
                continue
            diff = difflib.unified_diff(
                (before or "").splitlines(), after.splitlines(),
                fromfile=f"修改前/{rel}" if before is not None else "/dev/null", tofile=f"修改後/{rel}",
                lineterm="", n=context,
            )
            out.append((rel, "\n".join(diff)))
        return out

    def _is_protected_demo_run(self, command: str) -> bool:
        """`python demo.py`，而且 demo.py 受保護（agent 改不了）時，執行它跟跑 pytest 一樣安全。"""
        try:
            argv = shlex.split(command)
        except ValueError:
            return False
        return (
            argv in (["python", "demo.py"], ["python3", "demo.py"])
            and (self.ws.root / "demo.py").is_file()
            and any(fnmatch.fnmatch("demo.py", g) for g in self.protected_globs)
        )

    def _tool_run_command(self, command: str) -> ToolResult:
        if self._is_protected_demo_run(command):
            decision, reason, ok = "allow", "受保護的 demo.py", True
        else:
            decision, reason, ok = self.policy.decide(command)
        if not ok:
            msg = f"ERROR: 指令被拒絕（{decision}）：{reason}"
            return ToolResult("run_command", msg, ok=False, decision=decision, reason=reason)
        out = self.runner.run(shlex.split(command), self.ws, self.cmd_timeout)
        if TEST_CMD_RE.search(command) and not out.timed_out:
            self.tested_since_write = True
            self.tests_passed = out.exit_code == 0
            if self.tests_passed:
                self.needs_verification = False
        half = MAX_OUTPUT_CHARS // 2 - 100
        text = (
            f"exit_code: {out.exit_code}\n"
            f"--- stdout ---\n{truncate_middle(out.stdout, half, head_ratio=0.25)}\n"
            f"--- stderr ---\n{truncate_middle(out.stderr, half, head_ratio=0.25)}"
        )
        return ToolResult("run_command", text, ok=out.exit_code == 0, decision=decision, reason=reason)

    def _tool_search(self, pattern: str, glob: str | None = None) -> ToolResult:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise _ToolError(f"正規表示式錯誤：{e}") from None
        hits: list[str] = []
        for p in self._iter_files(glob):
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    hits.append(f"{self.ws.display(p)}:{i}:{line.strip()[:200]}")
                    if len(hits) >= MAX_SEARCH_RESULTS:
                        hits.append(f"（已達 {MAX_SEARCH_RESULTS} 筆上限，請縮小 pattern 或 glob）")
                        return ToolResult("search", "\n".join(hits), ok=True)
        return ToolResult("search", "\n".join(hits) if hits else "（沒有符合的結果）", ok=True)

    def _iter_files(self, glob: str | None) -> list[Path]:
        out: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.ws.root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and d != ".ssh")
            for fn in sorted(filenames):
                full = Path(dirpath) / fn
                rel = str(full.relative_to(self.ws.root))
                if glob and not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(fn, glob)):
                    continue
                try:
                    p = self.ws.resolve(rel)
                except SafetyError:
                    continue
                if p.is_file() and p.stat().st_size <= self.ws.max_read_bytes:
                    out.append(p)
        return out


def _maybe_unescape(content: str) -> tuple[str, bool]:
    """小模型常把檔案內容 JSON 跳脫兩次（整份檔案變成一行、充滿字面 \\n）。

    只在「幾乎沒有真正換行、卻有很多字面 \\n」且還原後是合法 Python 時才還原。
    """
    try:
        compile(content, "<check>", "exec")
        return content, False  # 本來就合法，不動
    except (SyntaxError, ValueError):
        pass
    candidates: list[str] = []
    if content.count("\\n") > content.count("\n"):
        try:  # strict=False：容許字串中混有真正的換行
            candidates.append(json.JSONDecoder(strict=False).decode(f'"{content.strip()}"'))
        except json.JSONDecodeError:
            pass
    if '\\"' in content:
        candidates.append(content.replace('\\"', '"'))  # 只有引號被跳脫、換行是真的
    for cand in candidates:
        try:
            compile(cand, "<check>", "exec")
        except (SyntaxError, ValueError):
            continue
        return (cand if cand.endswith("\n") else cand + "\n"), True
    return content, False


class _ToolError(Exception):
    """工具層的使用錯誤（參數錯、檔案不存在等），訊息給模型看。"""


def _schema_props(name: str) -> dict[str, Any]:
    for s in TOOL_SCHEMAS:
        if s["function"]["name"] == name:
            return s["function"]["parameters"]["properties"]
    return {}


def _summarize_args(args: dict[str, Any]) -> dict[str, Any]:
    """稽核紀錄不存整份檔案內容，只存長度與開頭。"""
    out: dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 200:
            out[k] = f"<{len(v)} chars> {v[:80]}..."
        else:
            out[k] = v
    return out
