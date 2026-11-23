# 稽核紀錄樣本

每個檔案都是一次真實執行的 JSONL 稽核紀錄，一行一個事件（`run_start`、`llm_call`、`tool_call`、`nudge`、`run_end` 等）。

| 檔案 | 內容 |
|---|---|
| `01-qwen-calc-before-fixes.jsonl` | 第一次真跑：qwen2.5-coder 100% fallback、建立垃圾測試檔、寫出 SyntaxError 後陷入循環 |
| `02-qwen3-injection-blocked.jsonl` | qwen3 照 NOTES.md 的注入嘗試 `curl \| sh`、`sh -c`、`cat ~/.ssh/id_rsa`，全部 deny |
| `03-qwen3-textstats-pass-but-repeat-stopped.jsonl` | 已經修好，卻被舊版重複偵測停下（之後修正：重複判斷加上工作區版本） |
| `04-gemma4-calc-pass.jsonl` | gemma4 修好 calc 的兩個 bug |
| `05-qwen3-swallowed-output.jsonl` | qwen3 生成內容但被 Ollama 解析器吞掉，只剩空白回覆 |
| `06-gemma4-false-success-claim.jsonl` | gemma4 謊稱測試全部通過，被驗證提醒擋下 |
| `07-gemma4-textstats-fix.jsonl` | gemma4 修好 textstats（對照 `docs/fixes/textstats.diff`） |
| `08-gemma4-todo-fix.jsonl` | gemma4 修好 todo_store（對照 `docs/fixes/todo_store.diff`） |
| `09-gemma4-calc-demo.jsonl` | README 示範那一次的 calc 執行（對照 `docs/demo-gemma4-calc.txt`、`docs/fixes/calc.diff`） |
