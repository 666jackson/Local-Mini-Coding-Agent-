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


# ---------------------------------------------------------------- 修改內容與修改前後對照


def _capture(monkeypatch, fn, *args) -> str:
    """用不帶色碼的寬 console 執行一個顯示函式，回傳輸出文字。"""
    monkeypatch.setattr(cli, "console", Console(width=140, highlight=False, color_system=None))
    with cli.console.capture() as cap:
        fn(*args)
    return cap.get()


def test_demo_ok_matches_exception_class_only():
    assert cli._demo_ok("14.0", "14.0")
    assert cli._demo_ok("CalcError: division by zero", "CalcError")
    assert not cli._demo_ok("ZeroDivisionError: float division by zero", "CalcError")
    assert not cli._demo_ok("6.0", "12.0")


def test_show_demo_flags_regression_even_when_other_rows_are_fixed(monkeypatch):
    """原本正確、修改後變錯的例子要被點名，不能因為其他例子修好就算過。"""
    before = "a => 20.0 => 14.0\nb => 12.0 => 12.0"
    after = "a => 14.0 => 14.0\nb => 6.0 => 12.0"
    out = _capture(monkeypatch, cli.show_demo, before, after)
    assert "正確答案" in out and "✗ 6.0" in out and "✓ 14.0" in out
    assert "還有 1 個例子不正確" in out and "1 個是原本正確、被這次修改改壞的" in out


def test_show_demo_all_correct(monkeypatch):
    out = _capture(monkeypatch, cli.show_demo, "a => 1 => 2\nb => 3 => 3", "a => 2 => 2\nb => 3 => 3")
    assert "2 個例子全部正確（修好 1 個）" in out


def test_show_demo_falls_back_to_raw_output(monkeypatch):
    """demo.py 不照格式輸出時，原樣並排，不要當掉。"""
    out = _capture(monkeypatch, cli.show_demo, "hello", "world\nmore")
    assert "hello" in out and "world" in out


def test_run_demo_before_and_after(tmp_path):
    from agent.safety import Workspace
    from agent.tools import LocalRunner

    ws = Workspace(tmp_path)
    assert cli.run_demo(ws, LocalRunner(), 10) is None  # 沒有 demo.py
    (tmp_path / "demo.py").write_text("print('x => 1 => 1')\nraise SystemExit(3)\n")
    out = cli.run_demo(ws, LocalRunner(), 10)
    assert out.startswith("x => 1 => 1") and "結束碼 3" in out


def test_show_changes_prints_diff_and_saves_file(monkeypatch, tmp_path):
    diff = "--- 修改前/a.py\n+++ 修改後/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 2"
    save = tmp_path / "changes.diff"
    out = _capture(monkeypatch, cli.show_changes, [("a.py", diff)], save)
    assert "修改內容：a.py（+1 / -1）" in out and "-x = 1" in out and "+x = 2" in out
    assert save.read_text().startswith("--- 修改前/a.py")
    assert "沒有修改任何檔案" in _capture(monkeypatch, cli.show_changes, [], None)
