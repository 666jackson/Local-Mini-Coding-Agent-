"""命令列入口。

    python cli.py --workspace demo_projects/calc "讓所有測試通過"
    python cli.py --workspace demo_projects/calc --model llama3.1:8b --max-steps 15 "..."
    python cli.py --workspace demo_projects/calc --yes "..."      # 自動同意 confirm 級
    python cli.py --workspace demo_projects/calc --docker "..."   # 容器執行 run_command
    python cli.py --workspace demo_projects/calc --copy "..."     # 在副本上跑，不改原專案
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from agent.audit import AuditLog
from agent.config import Config
from agent.llm import OpenAICompatLLM
from agent.loop import RunResult, run
from agent.safety import CommandPolicy, Workspace
from agent.tools import DEFAULT_PROTECTED, CommandRunner, DockerRunner, LocalRunner, Toolbox

ROOT = Path(__file__).resolve().parent
console = Console(highlight=False)

DECISION_STYLE = {"allow": "green", "confirm": "yellow", "deny": "bold red"}
DEMO_SCRIPT = "demo.py"  # 工作區裡有這個檔案，就在修改前後各跑一次，對照結果
MAX_DIFF_LINES = 40  # 畫面上每個檔案最多顯示幾行 diff；完整內容存在 runs/ 底下


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cfg = Config.from_env()
    p = argparse.ArgumentParser(description="地端迷你 coding agent")
    p.add_argument("task", help="要 agent 完成的任務")
    p.add_argument("--workspace", "-w", required=True, type=Path, help="agent 唯一能操作的資料夾")
    p.add_argument("--model", default=cfg.model)
    p.add_argument("--base-url", default=cfg.base_url)
    p.add_argument("--reasoning-effort", default=cfg.reasoning_effort,
                   help="思考型模型用，例如 none（qwen3 建議關閉思考）")
    p.add_argument("--max-steps", type=int, default=cfg.max_steps)
    p.add_argument("--timeout", type=int, default=cfg.cmd_timeout, help="單一指令逾時秒數")
    p.add_argument("--yes", "-y", action="store_true", help="自動同意 confirm 級指令（deny 永遠不放行）")
    p.add_argument("--docker", action="store_true", help="run_command 在無網路的容器內執行")
    p.add_argument("--docker-image", default="mini-agent-sandbox:latest")
    p.add_argument("--copy", action="store_true", help="先把工作區複製到 runs/ 底下再執行")
    p.add_argument("--allow-test-edits", action="store_true", help="允許修改測試檔（預設在系統層禁止）")
    p.add_argument("--no-require-tests", action="store_true", help="不要求測試通過才能結束（問答類任務用）")
    p.add_argument("--quiet", "-q", action="store_true", help="只印結果")
    args = p.parse_args(argv)
    args.api_key = cfg.api_key
    return args


def show_path(path: Path) -> str:
    """顯示用路徑：在目前目錄底下就印相對路徑，避免把個人的絕對路徑印在畫面或錄影裡。"""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def ask_user(cmd: str) -> bool:
    """confirm 級指令：印出來請使用者輸入 y。"""
    console.print(f"[yellow]⚠ 模型想執行不在白名單的指令：[/]{escape(cmd)}")
    try:
        return input("允許執行嗎？[y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def make_printer(quiet: bool):
    """把迴圈事件印成易讀的逐步紀錄。"""

    def on_event(kind: str, d: dict[str, Any]) -> None:
        if quiet and kind != "end":
            return
        if kind == "assistant":
            text = (d["content"] or "").strip()
            console.rule(f"[bold]第 {d['step']} 步[/]", style="dim")
            if text and d["n_calls"] == 0:
                console.print(Panel(escape(text), title="模型回覆", border_style="cyan"))
            elif text:
                console.print(f"[dim]{escape(text[:300])}[/]")
            if d["source"] == "fallback":
                console.print("[magenta]↺ 模型把工具呼叫寫在文字裡，已用 fallback 解析[/]")
        elif kind == "tool_call":
            args = d["arguments"]
            try:
                parsed = json.loads(args)
                if isinstance(parsed, dict) and isinstance(parsed.get("content"), str) and len(parsed["content"]) > 120:
                    parsed["content"] = f"<{len(parsed['content'])} chars>"
                args = json.dumps(parsed, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                pass
            console.print(f"[bold blue]→ {d['name']}[/] {escape(args[:300])}")
        elif kind == "tool_result":
            r = d["result"]
            tag = ""
            if r.decision:
                tag = f"[{DECISION_STYLE[r.decision]}]\\[{r.decision}][/] "  # 跳脫左括號，否則 rich 會把 [allow] 當成標記吃掉
            first = r.content.strip().splitlines()[0] if r.content.strip() else ""
            if r.name == "run_command" and r.content.startswith("exit_code"):
                tail = [ln for ln in r.content.splitlines() if ln.strip() and not ln.startswith("---")]
                first = f"{tail[0]} | {tail[-1][:160]}" if len(tail) > 1 else tail[0]
            style = "red" if r.content.startswith("ERROR") else "dim"
            if r.name == "run_command" and r.content.startswith("exit_code"):
                style = "green" if r.content.startswith("exit_code: 0") else "red"
            console.print(f"  {tag}[{style}]{escape(first[:200])}[/]")
        elif kind == "nudge":
            console.print(f"[yellow]✋ harness 提醒：{escape(d['text'])}[/]")

    return on_event


def run_demo(ws: Workspace, runner: CommandRunner, timeout: int) -> str | None:
    """工作區有 demo.py 就執行它並回傳輸出（含錯誤訊息）；沒有就回傳 None。"""
    if not (ws.root / DEMO_SCRIPT).is_file():
        return None
    out = runner.run(["python", DEMO_SCRIPT], ws, timeout)
    text = out.stdout.rstrip()
    if out.exit_code != 0:
        err = [ln for ln in out.stderr.strip().splitlines() if ln.strip()]
        text += f"\n（demo.py 結束碼 {out.exit_code}：{err[-1] if err else '沒有錯誤訊息'}）"
    return text.strip()


def show_changes(changes: list[tuple[str, str]], save_to: Path | None = None) -> None:
    """用紅綠色 diff 顯示 agent 改了哪些檔案；完整 diff 另存一份。"""
    if not changes:
        console.print(Panel("[dim]這次執行沒有修改任何檔案。[/]", title="修改內容", border_style="magenta"))
        return
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text("\n\n".join(d for _, d in changes) + "\n", encoding="utf-8")
    for rel, diff in changes:
        lines = diff.splitlines()
        added = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
        removed = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))
        shown = lines[2:]  # 省略 ---/+++ 標頭，檔名已經在標題上
        note = ""
        if len(shown) > MAX_DIFF_LINES:
            note = f"\n[dim]…還有 {len(shown) - MAX_DIFF_LINES} 行，完整內容：{escape(show_path(save_to)) if save_to else ''}[/]"
            shown = shown[:MAX_DIFF_LINES]
        body = Syntax("\n".join(shown), "diff", theme="ansi_dark", background_color="default", word_wrap=True)
        console.print(Panel(body, title=f"修改內容：{escape(rel)}（+{added} / -{removed}）", border_style="magenta",
                            subtitle=note.strip() or None))


def _demo_ok(result: str, expected: str) -> bool:
    """結果符合正確答案嗎？例外只比對類別名稱（`CalcError` 符合 `CalcError: 任何訊息`）。"""
    return result == expected or result.startswith(expected + ":")


def show_demo(before: str, after: str, tests_passed: bool = False) -> None:
    """並排顯示 demo.py 在修改前、修改後的輸出。

    demo.py 每行印 `例子 => 結果 => 正確答案`，表格就會用 ✓ / ✗ 標出對錯；
    只印 `例子 => 結果` 時只標出有沒有改變；其他格式就把兩次輸出原樣並排。
    """
    rows_b = [ln.split(" => ") for ln in before.splitlines()]
    rows_a = [ln.split(" => ") for ln in after.splitlines()]
    n = len(rows_b[0]) if rows_b else 0
    structured = (
        n in (2, 3)
        and len(rows_b) == len(rows_a)
        and all(len(r) == n for r in rows_b + rows_a)
        and all(b[0] == a[0] for b, a in zip(rows_b, rows_a))
    )
    table = Table(title="修改前後對照（執行 demo.py）", title_style="bold", border_style="cyan", expand=True)
    if not structured:
        table.add_column("修改前")
        table.add_column("修改後")
        table.add_row(escape(before), escape(after))
        console.print(table)
        return

    table.add_column("例子", style="bold", ratio=4)
    table.add_column("修改前", ratio=3)
    table.add_column("修改後", ratio=3)
    if n == 3:
        table.add_column("正確答案", ratio=2)
    fixed = broken = wrong = 0
    for b, a in zip(rows_b, rows_a):
        if n == 3:
            expected = b[2]
            ok_b, ok_a = _demo_ok(b[1], expected), _demo_ok(a[1], expected)
            fixed += (not ok_b) and ok_a
            broken += ok_b and not ok_a
            wrong += not ok_a
            cell_b = f"[{'dim' if ok_b else 'red'}]{'✓' if ok_b else '✗'} {escape(b[1])}[/]"
            cell_a = f"[{'green' if ok_a else 'bold red'}]{'✓' if ok_a else '✗'} {escape(a[1])}[/]"
            table.add_row(escape(b[0]), cell_b, cell_a, escape(expected))
        else:
            changed = a[1] != b[1]
            table.add_row(escape(b[0]), f"[{'yellow' if changed else 'dim'}]{escape(b[1])}[/]",
                          f"[{'yellow' if changed else 'dim'}]{escape(a[1])}{'' if changed else '（沒變）'}[/]")
    console.print(table)
    if n == 3:
        if wrong == 0:
            console.print(f"[green]demo 的 {len(rows_a)} 個例子全部正確（修好 {fixed} 個）。[/]")
        else:
            msg = f"[bold red]注意：demo 還有 {wrong} 個例子不正確"
            if broken:
                msg += f"，其中 {broken} 個是原本正確、被這次修改改壞的"
            if tests_passed:
                msg += "。測試雖然通過，但程式不一定是對的"
            console.print(msg + "。[/]")


def show_summary(res: RunResult) -> None:
    color = "green" if res.status == "done" and res.verified else ("yellow" if res.status == "done" else "red")
    lines = [
        f"狀態：[{color}]{res.status}[/]  測試驗證：{'通過' if res.verified else '未通過'}",
        f"步數：{res.steps}  工具呼叫：{res.tool_calls}  工具錯誤：{res.tool_errors}  "
        f"fallback：{res.fallback_calls}  JSON 壞掉：{res.malformed_calls}  deny：{res.denied}  "
        f"驗證提醒：{res.verify_nudges}",
        f"耗時：{res.elapsed_s} 秒",
    ]
    if res.error:
        lines.append(f"[red]錯誤：{escape(res.error)}[/]")
    console.print(Panel("\n".join(lines), title="執行結果", border_style=color))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "runs" / f"{stamp}-{args.workspace.resolve().name}"
    src = args.workspace.resolve()
    if not src.is_dir():
        console.print(f"[red]工作區不存在：{src}[/]")
        return 2
    if args.copy:
        target = run_dir / "workspace"
        shutil.copytree(src, target, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        src = target

    ws = Workspace(src)
    audit = AuditLog(run_dir / "audit.jsonl")  # 稽核紀錄放在工作區外，agent 讀不到也改不到
    policy = CommandPolicy(ws, auto_yes=args.yes, confirm_fn=ask_user)
    runner = DockerRunner(args.docker_image) if args.docker else LocalRunner()
    protected = () if args.allow_test_edits else DEFAULT_PROTECTED
    toolbox = Toolbox(ws, policy, cmd_timeout=args.timeout, runner=runner, audit=audit, protected_globs=protected)
    llm = OpenAICompatLLM(args.base_url, args.api_key, args.model, reasoning_effort=args.reasoning_effort)

    console.print(Panel(
        f"任務：{escape(args.task)}\n工作區：{escape(show_path(ws.root))}\n模型：{args.model} @ {args.base_url}\n"
        f"執行器：{'docker（--network none）' if args.docker else 'local'}  --yes：{args.yes}",
        title="mini-agent", border_style="blue",
    ))
    demo_before = run_demo(ws, runner, args.timeout)  # 在 agent 動手之前先記下原本的行為
    res = run(args.task, llm, toolbox, max_steps=args.max_steps, on_event=make_printer(args.quiet), audit=audit,
              require_passing_tests=not args.no_require_tests)
    if res.answer and args.quiet:
        console.print(escape(res.answer))
    show_changes(toolbox.changes(), save_to=run_dir / "changes.diff")
    if demo_before is not None and toolbox.originals:
        show_demo(demo_before, run_demo(ws, runner, args.timeout) or "", tests_passed=res.verified)
    show_summary(res)
    console.print(f"[dim]稽核紀錄：{escape(show_path(audit.path))}[/]")
    return 0 if res.status == "done" and res.verified else 1


if __name__ == "__main__":
    sys.exit(main())
