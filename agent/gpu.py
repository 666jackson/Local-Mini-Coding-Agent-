"""GPU 記憶體守門：共用機器上避免模型吃掉超過上限的顯示記憶體。"""

from __future__ import annotations

import shutil
import subprocess


def gpu_used_mb() -> int | None:
    """回傳所有 GPU 已用記憶體總和（MiB）；沒有 nvidia-smi 時回傳 None。"""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    return sum(int(x) for x in out.split() if x.strip().isdigit())


def unload_model(model: str) -> None:
    """請 Ollama 立即卸載模型，釋放顯示記憶體（失敗就忽略）。"""
    if shutil.which("ollama"):
        subprocess.run(["ollama", "stop", model], capture_output=True, timeout=60)


class GpuBudgetExceeded(RuntimeError):
    """GPU 記憶體超過設定上限。"""


def check_gpu_budget(limit_mb: int | None) -> int | None:
    """超過上限就丟 GpuBudgetExceeded；limit_mb 為 None 或量不到時不檢查。"""
    used = gpu_used_mb()
    if limit_mb is not None and used is not None and used > limit_mb:
        raise GpuBudgetExceeded(f"GPU 已用 {used} MiB，超過上限 {limit_mb} MiB")
    return used
