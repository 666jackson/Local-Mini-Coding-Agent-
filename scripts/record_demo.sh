#!/bin/bash
# 錄製 README 的示範 GIF（需要 Ollama 與 gemma4-16k；另外要 pip install asciinema pyte wcwidth pillow）
#   scripts/record_demo.sh            → docs/demo.gif
# 注意：模型輸出每次不一定相同，可能需要錄幾次。
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" != "--inner" ]]; then
  cast=$(mktemp --suffix=.cast)
  asciinema rec --overwrite --cols 92 --rows 40 -q -c "$0 --inner" "$cast"
  python scripts/cast2gif.py "$cast" docs/demo.gif --max-idle 1.0
  exit 0
fi
export COLUMNS=92 LINES=40
cmd='python cli.py -w demo_projects/calc --copy --model gemma4-16k "讓所有測試通過。"'
printf '\033[1;32m$\033[0m '; sleep 0.6
for ((i=0; i<${#cmd}; i++)); do printf '%s' "${cmd:$i:1}"; sleep 0.03; done
sleep 0.5; printf '\n'
eval "$cmd" || true
sleep 3
