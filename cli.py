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

from agent.audit import AuditLog
from agent.config import Config
from agent.llm import OpenAICompatLLM
from agent.loop import RunResult, run
from agent.safety import CommandPolicy, Workspace
from agent.tools import DEFAULT_PROTECTED, DockerRunner, LocalRunner, Toolbox

ROOT = Path(__file__).resolve().parent
console = Console(highlight=False)

DECISION_STYLE = {"allow": "green", "confirm": "yellow", "deny": "bold red"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cfg = Config.from_env()
    p = argparse.ArgumentParser(description="地端迷你 coding agent")
    p.add_argument("task", help="要 agent 完成的任務")
    p.add_argument("--workspace", "-w", required=True, type=Path, help="agent 唯一能操作的資料夾")
    p.add_argument("--model", default=cfg.model)
    p.add_argument("--base-url", default=cfg.base_url)
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
                tag = f"[{DECISION_STYLE[r.decision]}][{r.decision}][/] "
            first = r.content.strip().splitlines()[0] if r.content.strip() else ""
            if r.name == "run_command" and r.content.startswith("exit_code"):
                tail = [ln for ln in r.content.splitlines() if ln.strip() and not ln.startswith("---")]
                first = f"{tail[0]} | {tail[-1][:160]}" if len(tail) > 1 else tail[0]
            style = "red" if r.content.startswith("ERROR") else "dim"
            console.print(f"  {tag}[{style}]{escape(first[:200])}[/]")
        elif kind == "nudge":
            console.print(f"[yellow]✋ harness 提醒：{escape(d['text'])}[/]")
        elif kind == "end":
            show_summary(d["result"])

    return on_event


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
    llm = OpenAICompatLLM(args.base_url, args.api_key, args.model)

    console.print(Panel(
        f"任務：{escape(args.task)}\n工作區：{ws.root}\n模型：{args.model} @ {args.base_url}\n"
        f"執行器：{'docker（--network none）' if args.docker else 'local'}  --yes：{args.yes}",
        title="mini-agent", border_style="blue",
    ))
    res = run(args.task, llm, toolbox, max_steps=args.max_steps, on_event=make_printer(args.quiet), audit=audit,
              require_passing_tests=not args.no_require_tests)
    if res.answer and args.quiet:
        console.print(escape(res.answer))
    console.print(f"[dim]稽核紀錄：{audit.path}[/]")
    return 0 if res.status == "done" and res.verified else 1


if __name__ == "__main__":
    sys.exit(main())
