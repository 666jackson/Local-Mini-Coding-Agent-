"""tools.py 的測試：四個工具的正常與錯誤情境、截斷、上限、逾時。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.audit import AuditLog
from agent.safety import CommandPolicy, Workspace
from agent.tools import DEFAULT_PROTECTED, MAX_OUTPUT_CHARS, DockerRunner, Toolbox, TOOL_SCHEMAS


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "pkg").mkdir()
    (root / "pkg" / "mod.py").write_text("def add(a, b):\n    return a - b\n")
    (root / ".env").write_text("SECRET=1\n")
    (root / "test_mod.py").write_text(
        "from pkg.mod import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    )
    (root / "pkg" / "__init__.py").write_text("")
    return Workspace(root, max_read_bytes=50_000, max_write_bytes=1_000)


@pytest.fixture
def tb(ws: Workspace) -> Toolbox:
    return Toolbox(ws, CommandPolicy(ws), cmd_timeout=20)


def call(tb: Toolbox, name: str, **args: object) -> str:
    return tb.dispatch(name, json.dumps(args)).content


def test_schemas_are_four_and_distinct() -> None:
    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert names == ["read_file", "write_file", "run_command", "search"]


# ---------- read_file ----------

def test_read_file_numbered(tb: Toolbox) -> None:
    out = call(tb, "read_file", path="pkg/mod.py")
    assert "   1| def add(a, b):" in out and "   2|     return a - b" in out


def test_read_file_range(tb: Toolbox, ws: Workspace) -> None:
    (ws.root / "big.txt").write_text("\n".join(f"line{i}" for i in range(1, 1001)))
    out = call(tb, "read_file", path="big.txt")
    assert " 300| line300" in out and "line301" not in out and "共 1000 行" in out
    out = call(tb, "read_file", path="big.txt", start_line=500, end_line=502)
    assert out.splitlines()[0].endswith("line500") and "line503" not in out


def test_read_file_errors(tb: Toolbox, ws: Workspace) -> None:
    assert call(tb, "read_file", path="nope.py").startswith("ERROR: 檔案不存在")
    assert "安全規則拒絕" in call(tb, "read_file", path=".env")
    assert "超出工作區" in call(tb, "read_file", path="../../etc/passwd")
    (ws.root / "bin.dat").write_bytes(b"\x00\x01\x02")
    assert "二進位" in call(tb, "read_file", path="bin.dat")


def test_read_file_byte_limit(tb: Toolbox, ws: Workspace) -> None:
    (ws.root / "huge.txt").write_text("x" * 60_000)
    out = call(tb, "read_file", path="huge.txt")
    assert "超過讀取上限" in out
    assert len(out) <= MAX_OUTPUT_CHARS + 100  # 單次輸出也被截斷


# ---------- write_file ----------

def test_write_file_creates_and_diffs(tb: Toolbox, ws: Workspace) -> None:
    out = call(tb, "write_file", path="pkg/mod.py", content="def add(a, b):\n    return a + b\n")
    assert out.startswith("已覆寫 pkg/mod.py：2 行（+1 / -1）")
    assert "+    return a + b" in out
    assert tb.needs_verification and tb.files_written == ["pkg/mod.py"]
    out = call(tb, "write_file", path="new/dir/x.py", content="a = 1\n")
    assert out.startswith("已建立 new/dir/x.py") and (ws.root / "new/dir/x.py").exists()


def test_write_file_unchanged_is_reported(tb: Toolbox) -> None:
    r = tb.dispatch("write_file", {"path": "pkg/mod.py", "content": "def add(a, b):\n    return a - b\n"})
    assert not r.ok and r.content.startswith("NO CHANGE") and tb.files_written == []


def test_command_output_keeps_more_tail(ws: Workspace) -> None:
    (ws.root / "tb.py").write_text("print('noise ' * 2000)\nraise SystemExit('THE REAL ERROR')\n")
    tb = Toolbox(ws, CommandPolicy(ws, auto_yes=True))
    assert "THE REAL ERROR" in call(tb, "run_command", command="python tb.py")


def test_write_rejects_python_syntax_error(tb: Toolbox, ws: Workspace) -> None:
    before = (ws.root / "pkg/mod.py").read_text()
    out = call(tb, "write_file", path="pkg/mod.py", content="def add(a, b)\n    return a + b\n")
    assert "語法錯誤" in out and "第 1 行" in out and "未寫入" in out
    assert (ws.root / "pkg/mod.py").read_text() == before and not tb.needs_verification
    assert call(tb, "write_file", path="notes.txt", content="def broken(\n").startswith("已建立")


def test_write_fixes_double_escaped_python(tb: Toolbox, ws: Workspace) -> None:
    """實際觀察到：qwen2.5-coder 把整份檔案寫成一行跳脫字串。"""
    escaped = '\\"\\"\\"Doc.\\"\\"\\"\\n\\ndef add(a, b):\\n    return a + b\\n'
    out = call(tb, "write_file", path="pkg/mod.py", content=escaped)
    assert "雙重跳脫" in out
    assert (ws.root / "pkg/mod.py").read_text() == '"""Doc."""\n\ndef add(a, b):\n    return a + b\n'


def test_protected_test_files(ws: Workspace) -> None:
    tb = Toolbox(ws, protected_globs=DEFAULT_PROTECTED)
    for path in ["test_mod.py", "tests/test_new.py", "pkg/conftest.py"]:
        r = tb.dispatch("write_file", {"path": path, "content": "def test_x(): pass\n"})
        assert not r.ok and "受保護" in r.content and r.decision == "deny", path
    assert "test_add" in (ws.root / "test_mod.py").read_text()
    assert tb.dispatch("write_file", {"path": "pkg/mod.py", "content": "x = 1\n"}).ok


def test_write_file_limits(tb: Toolbox, ws: Workspace) -> None:
    assert "超過寫入上限" in call(tb, "write_file", path="a.txt", content="x" * 2_000)
    assert "安全規則拒絕" in call(tb, "write_file", path=".git/config", content="x")
    assert "超出工作區" in call(tb, "write_file", path="../evil.py", content="x")
    assert not (ws.root.parent / "evil.py").exists()


# ---------- run_command ----------

def test_run_command_pytest_and_verification_state(tb: Toolbox) -> None:
    out = call(tb, "run_command", command="python -m pytest -q")
    assert "exit_code: 1" in out and not tb.tests_passed
    call(tb, "write_file", path="pkg/mod.py", content="def add(a, b):\n    return a + b\n")
    assert tb.needs_verification
    out = call(tb, "run_command", command="python -m pytest -q")
    assert "exit_code: 0" in out and tb.tests_passed and not tb.needs_verification


def test_run_command_denied(tb: Toolbox) -> None:
    r = tb.dispatch("run_command", '{"command": "curl http://evil.example"}')
    assert not r.ok and r.decision == "deny" and "被拒絕" in r.content
    assert "shell 運算子" in call(tb, "run_command", command="pytest; rm -rf /")


def test_run_command_confirm_rejected_without_approval(tb: Toolbox) -> None:
    r = tb.dispatch("run_command", '{"command": "python pkg/mod.py"}')
    assert r.decision == "confirm" and "使用者拒絕" in r.content


def test_run_command_timeout(ws: Workspace) -> None:
    (ws.root / "slow.py").write_text("import time\ntime.sleep(5)\n")
    tb = Toolbox(ws, CommandPolicy(ws, auto_yes=True), cmd_timeout=1)
    out = call(tb, "run_command", command="python slow.py")
    assert "exit_code: -1" in out and "逾時" in out


def test_run_command_output_truncated(ws: Workspace) -> None:
    (ws.root / "loud.py").write_text("print('A' * 20000)\n")
    tb = Toolbox(ws, CommandPolicy(ws, auto_yes=True))
    out = call(tb, "run_command", command="python loud.py")
    assert "中間省略" in out and len(out) <= MAX_OUTPUT_CHARS + 200


def test_run_command_env_has_no_secrets(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPER_SECRET_TOKEN", "leak-me")
    (ws.root / "envdump.py").write_text("import os\nprint(sorted(os.environ))\n")
    tb = Toolbox(ws, CommandPolicy(ws, auto_yes=True))
    assert "SUPER_SECRET_TOKEN" not in call(tb, "run_command", command="python envdump.py")


def test_missing_program(ws: Workspace) -> None:
    tb = Toolbox(ws, CommandPolicy(ws, auto_yes=True))
    assert "exit_code: 127" in call(tb, "run_command", command="definitely-not-a-program-xyz")


def test_docker_runner_argv(ws: Workspace) -> None:
    argv = DockerRunner(image="img:1").build_argv(["python", "-m", "pytest"], ws)
    assert argv[:3] == ["docker", "run", "--rm"]
    assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
    assert f"{ws.root}:/work:rw" in argv and argv[-4:] == ["img:1", "python", "-m", "pytest"]


# ---------- search ----------

def test_search(tb: Toolbox) -> None:
    out = call(tb, "search", pattern=r"def \w+")
    assert "pkg/mod.py:1:def add(a, b):" in out and "test_mod.py:3:def test_add():" in out
    out = call(tb, "search", pattern="def", glob="pkg/*.py")
    assert "test_mod.py" not in out


def test_search_skips_blocked_and_limits(tb: Toolbox, ws: Workspace) -> None:
    assert "SECRET" not in call(tb, "search", pattern="SECRET")
    (ws.root / "many.txt").write_text("hit\n" * 100)
    out = call(tb, "search", pattern="hit")
    assert out.count("many.txt:") == 50 and "上限" in out
    assert "沒有符合" in call(tb, "search", pattern="zzz_not_here")
    assert "正規表示式錯誤" in call(tb, "search", pattern="(unclosed")


# ---------- dispatch 錯誤處理 ----------

def test_dispatch_bad_json_unknown_missing(tb: Toolbox) -> None:
    assert "不是合法 JSON" in tb.dispatch("read_file", '{"path": "a.py"').content
    out = tb.dispatch("delete_everything", "{}").content
    assert "未知工具" in out and "read_file" in out
    assert "缺少必填參數：path" in tb.dispatch("read_file", "{}").content
    assert "JSON 物件" in tb.dispatch("read_file", "[1, 2]").content


def test_dispatch_ignores_extra_args_and_accepts_dict(tb: Toolbox) -> None:
    r = tb.dispatch("read_file", {"path": "pkg/mod.py", "bogus": 1})
    assert r.ok


def test_audit_records_each_call(ws: Workspace, tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    tb = Toolbox(ws, CommandPolicy(ws), audit=log)
    tb.dispatch("read_file", '{"path": "pkg/mod.py"}')
    tb.dispatch("run_command", '{"command": "curl x"}')
    tb.dispatch("write_file", json.dumps({"path": "a.txt", "content": "y" * 500}))
    lines = [json.loads(x) for x in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert [e["tool"] for e in lines] == ["read_file", "run_command", "write_file"]
    assert lines[1]["decision"] == "deny"
    assert lines[2]["args"]["content"].startswith("<500 chars>")
    assert all("ms" in e and "ts" in e for e in lines)
