"""CLI 逐步輸出的顯示測試（不需要 Ollama）。"""

from pathlib import Path

from rich.console import Console

import cli
from agent.tools import ToolResult


def _render(result: ToolResult, monkeypatch) -> str:
    """用一個強制輸出 ANSI 色碼的 console 印出一筆工具結果。"""
    monkeypatch.setattr(cli, "console", Console(force_terminal=True, color_system="standard", width=120, highlight=False))
    printer = cli.make_printer(quiet=False)
    with cli.console.capture() as cap:
        printer("tool_result", {"result": result})
    return cap.get()


def test_decision_tag_is_visible(monkeypatch):
    """[allow] / [deny] 要真的印出來，不能被 rich 當成標記吃掉。"""
    for decision in ("allow", "deny", "confirm"):
        out = _render(ToolResult("run_command", "ERROR: blocked", ok=False, decision=decision), monkeypatch)
        assert f"[{decision}]" in out


def test_failing_and_passing_tests_use_different_colors(monkeypatch):
    """exit_code 非 0 不能跟成功一樣是綠色。"""
    fail = _render(ToolResult("run_command", "exit_code: 1\n3 failed, 5 passed", ok=True, decision="allow"), monkeypatch)
    ok = _render(ToolResult("run_command", "exit_code: 0\n8 passed", ok=True, decision="allow"), monkeypatch)
    assert "\x1b[31mexit_code: 1" in fail
    assert "\x1b[32mexit_code: 0" in ok


def test_show_path_is_relative_under_cwd(tmp_path, monkeypatch):
    """畫面上的路徑在目前目錄底下時印相對路徑，不洩漏使用者的絕對路徑。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs" / "x").mkdir(parents=True)
    assert cli.show_path(tmp_path / "runs" / "x") == str(Path("runs/x"))
    assert cli.show_path(Path("/elsewhere/y")) == "/elsewhere/y"
