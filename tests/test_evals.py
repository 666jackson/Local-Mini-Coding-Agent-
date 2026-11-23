"""evals/run_evals.py 的純邏輯測試（不呼叫模型）。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("run_evals", ROOT / "evals" / "run_evals.py")
assert spec and spec.loader
run_evals = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_evals)


def _proj(tmp_path: Path, impl: str) -> Path:
    root = tmp_path / "p"
    root.mkdir()
    (root / "m.py").write_text(impl)
    (root / "test_m.py").write_text("from m import f\n\ndef test_f():\n    assert f() == 1\n")
    return root


def test_accept_passes_with_real_fix(tmp_path: Path) -> None:
    root = _proj(tmp_path, "def f():\n    return 1\n")
    original = run_evals.snapshot_tests(root)
    assert run_evals.accept(root, "python -m pytest -q", original) == (True, False)


def test_accept_restores_tampered_tests(tmp_path: Path) -> None:
    root = _proj(tmp_path, "def f():\n    return 2\n")
    original = run_evals.snapshot_tests(root)
    (root / "test_m.py").write_text("def test_f():\n    assert True\n")  # 作弊
    (root / "test_extra.py").write_text("def test_x():\n    assert True\n")
    passed, tampered = run_evals.accept(root, "python -m pytest -q", original)
    assert passed is False and tampered is True
    assert not (root / "test_extra.py").exists()


def test_all_demo_projects_start_red() -> None:
    """每個示範專案一開始都必須有失敗的測試，否則評測沒意義。"""
    import subprocess
    import sys

    for proj in ["calc", "textstats", "todo_store"]:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                           cwd=ROOT / "demo_projects" / proj, capture_output=True)
        assert r.returncode == 1, proj


def test_summarize_table() -> None:
    rows = [
        {"model": "m", "task": "t", "passed": True, "steps": 4, "tool_errors": 0, "fallback_calls": 0,
         "malformed_calls": 0, "denied": 0, "elapsed_s": 2.0},
        {"model": "m", "task": "t", "passed": False, "steps": 20, "tool_errors": 3, "fallback_calls": 1,
         "malformed_calls": 1, "denied": 2, "elapsed_s": 8.0},
    ]
    table = run_evals.summarize(rows)
    assert "| m | t | 1/2 | 12.0 |" in table and "1/2 (50%)" in table
