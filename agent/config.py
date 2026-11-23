"""從環境變數（與可選的 .env 檔）讀取設定。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """極簡 .env 讀取：只設定尚未存在的環境變數，不引入額外套件。"""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Config:
    """Agent 執行設定。"""

    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen2.5-coder:7b"
    max_steps: int = 20
    cmd_timeout: int = 30

    @classmethod
    def from_env(cls, dotenv: Path | None = None) -> "Config":
        """讀取 AGENT_* 環境變數，缺的用預設值。"""
        _load_dotenv(dotenv or Path(__file__).resolve().parent.parent / ".env")
        d = cls()
        return cls(
            base_url=os.environ.get("AGENT_BASE_URL", d.base_url),
            api_key=os.environ.get("AGENT_API_KEY", d.api_key),
            model=os.environ.get("AGENT_MODEL", d.model),
            max_steps=int(os.environ.get("AGENT_MAX_STEPS", d.max_steps)),
            cmd_timeout=int(os.environ.get("AGENT_CMD_TIMEOUT", d.cmd_timeout)),
        )
