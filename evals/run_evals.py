"""評測：對每個任務 × 模型 × 次數，在乾淨副本上執行 agent，再獨立驗收。

    python evals/run_evals.py --models qwen2.5-coder-16k llama3.1-16k --repeats 3

驗收前會把測試檔還原成原始版本，所以就算 agent 改了測試也無法「作弊」通過。
一次只跑一個模型，換模型前卸載，並在每次執行前檢查 GPU 記憶體上限。
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.audit import AuditLog  # noqa: E402
from agent.config import Config  # noqa: E402
from agent.gpu import check_gpu_budget, unload_model  # noqa: E402
from agent.llm import OpenAICompatLLM  # noqa: E402
from agent.loop import run  # noqa: E402
from agent.safety import CommandPolicy, Workspace, minimal_env  # noqa: E402
from agent.tools import DEFAULT_PROTECTED, Toolbox  # noqa: E402

PROJECTS = ROOT / "demo_projects"
TEST_GLOBS = ("test_*.py", "*_test.py", "conftest.py")


def snapshot_tests(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): p.read_text() for g in TEST_GLOBS for p in root.rglob(g)}


def accept(root: Path, cmd: str, original_tests: dict[str, str]) -> tuple[bool, bool]:
    """還原測試檔後執行驗收指令。回傳 (是否通過, agent 是否動過測試)。"""
    tampered = snapshot_tests(root) != original_tests
    for rel in set(snapshot_tests(root)) - set(original_tests):
        (root / rel).unlink()
    for rel, text in original_tests.items():
        (root / rel).write_text(text)
    proc = subprocess.run(shlex.split(cmd), cwd=root, env=minimal_env(), capture_output=True, timeout=120)
    return proc.returncode == 0, tampered


def run_one(task: dict[str, Any], model: str, cfg: Config, out_dir: Path, max_steps: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"eval-{task['id']}-") as tmp:
        work = Path(tmp) / task["project"]
        shutil.copytree(PROJECTS / task["project"], work, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        original = snapshot_tests(work)
        ws = Workspace(work)
        audit = AuditLog(out_dir / "audit.jsonl")
        # 評測是非互動的：confirm 級一律拒絕，只有白名單指令能跑
        tb = Toolbox(ws, CommandPolicy(ws), cmd_timeout=cfg.cmd_timeout, audit=audit, protected_globs=DEFAULT_PROTECTED)
        name, _, effort = model.partition("@")  # 例如 qwen3-16k@none 關閉思考
        llm = OpenAICompatLLM(cfg.base_url, cfg.api_key, name, reasoning_effort=effort or cfg.reasoning_effort)
        res = run(task["instruction"], llm, tb, max_steps=max_steps)
        passed, tampered = accept(work, task["accept"], original)
    events = audit.events
    return {
        "task": task["id"], "model": model, "passed": passed, "tampered_tests": tampered,
        **{k: v for k, v in asdict(res).items() if k != "answer"},
        "answer": (res.answer or "")[:500],
        "denied_cmds": [e["args"].get("command") for e in events
                        if e["event"] == "tool_call" and e.get("decision") == "deny" and e["tool"] == "run_command"],
        "env_read_blocked": sum(1 for e in events if e["event"] == "tool_call" and "禁止存取" in (e.get("result") or "")),
        "aliased_args": sum(1 for e in events if e["event"] == "tool_call" and e.get("aliased")),
    }


def summarize(rows: list[dict[str, Any]]) -> str:
    """產生 Markdown 表格：每個模型 × 任務一列。"""
    lines = ["| 模型 | 任務 | 成功率 | 平均步數 | 工具錯誤 | fallback | JSON 壞掉 | deny | 平均耗時(s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    keys = sorted({(r["model"], r["task"]) for r in rows})
    for model, task in keys:
        rs = [r for r in rows if r["model"] == model and r["task"] == task]
        lines.append(
            f"| {model} | {task} | {sum(r['passed'] for r in rs)}/{len(rs)} | {mean(r['steps'] for r in rs):.1f} | "
            f"{mean(r['tool_errors'] for r in rs):.1f} | {mean(r['fallback_calls'] for r in rs):.1f} | "
            f"{mean(r['malformed_calls'] for r in rs):.1f} | {mean(r['denied'] for r in rs):.1f} | "
            f"{mean(r['elapsed_s'] for r in rs):.1f} |"
        )
    lines.append("")
    lines.append("| 模型 | 總成功率 | 平均步數 | 平均耗時(s) |")
    lines.append("|---|---|---|---|")
    for model in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == model]
        lines.append(f"| {model} | {sum(r['passed'] for r in rs)}/{len(rs)} "
                     f"({100 * sum(r['passed'] for r in rs) / len(rs):.0f}%) | "
                     f"{mean(r['steps'] for r in rs):.1f} | {mean(r['elapsed_s'] for r in rs):.1f} |")
    return "\n".join(lines)


def main() -> int:
    cfg = Config.from_env()
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=[cfg.model], help="可用 模型@reasoning_effort，例如 qwen3-16k@none")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tasks", nargs="*", help="只跑這些任務 id")
    p.add_argument("--max-steps", type=int, default=cfg.max_steps)
    p.add_argument("--gpu-limit-mb", type=int, default=12000, help="每次執行前檢查 GPU 總用量，超過就中止")
    p.add_argument("--out", type=Path, default=ROOT / "evals" / "results")
    args = p.parse_args()

    tasks = yaml.safe_load((ROOT / "evals" / "tasks.yaml").read_text())["tasks"]
    if args.tasks:
        tasks = [t for t in tasks if t["id"] in args.tasks]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_dir = args.out / "raw" / stamp
    rows: list[dict[str, Any]] = []
    for model in args.models:
        for other in args.models:  # 確保 GPU 上同時只有一個模型
            unload_model(other.partition("@")[0])
        for task in tasks:
            for i in range(args.repeats):
                used = check_gpu_budget(args.gpu_limit_mb)
                run_dir = raw_dir / model.replace(":", "_").replace("@", "_") / f"{task['id']}-{i + 1}"
                row = run_one(task, model, cfg, run_dir, args.max_steps)
                row["gpu_mb_before"] = used
                rows.append(row)
                print(f"[{model}] {task['id']} #{i + 1}: {'PASS' if row['passed'] else 'FAIL'} "
                      f"status={row['status']} steps={row['steps']} {row['elapsed_s']}s "
                      f"denied={row['denied_cmds']}", flush=True)
        unload_model(model.partition("@")[0])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"results-{stamp}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    table = summarize(rows)
    (args.out / f"results-{stamp}.md").write_text(table + "\n")
    print("\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
