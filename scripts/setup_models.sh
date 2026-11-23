#!/usr/bin/env bash
# 下載模型並建立固定 16K context 的別名。
# 原因：Ollama 會依顯示卡自動放大 context（llama3.1:8b 曾以 131072 載入、單一模型吃掉 30 GB），
# 而 OpenAI 相容的 /v1 端點不接受 num_ctx。固定 16K 後每個 7–8B 模型約 6–8 GB。
set -euo pipefail
cd "$(dirname "$0")/.."
for pair in "qwen2.5-coder:7b qwen2.5-coder-16k" "llama3.1:8b llama3.1-16k" "qwen3:8b qwen3-16k" "gemma4:latest gemma4-16k"; do
  set -- $pair
  ollama pull "$1"
  ollama create "$2" -f "ollama/$2.Modelfile"
done
ollama list | grep -- -16k
