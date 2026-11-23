# AGENTS.md

給之後維護這個 repo 的 coding agent（與人）看的說明。

## 安裝與測試

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest -q                 # 不需要 Ollama；test_loop.py 用 ScriptedLLM
```

真模型（可選）：`scripts/setup_models.sh` 建立 16K context 的 Ollama 別名，然後
`python cli.py -w demo_projects/calc --copy "讓所有測試通過"`。

## 程式結構

- `agent/safety.py`：工作區邊界、禁碰檔案、指令三級政策。**安全核心。**
- `agent/tools.py`：四個工具與 schema、參數修正、寫入前檢查、runner（local / docker）。
- `agent/loop.py`：agent 迴圈、fallback 解析、重複偵測、驗證提醒。
- `agent/context.py`：工具輸出標籤與舊輸出壓縮。
- `evals/`：評測任務與執行器。`demo_projects/`：故意埋 bug 的專案，**不要修好它們**。
  每個示範專案的 `demo.py` 印出 `例子 => 結果 => 正確答案`，CLI 在修改前後各跑一次做對照；它和測試檔一樣受保護。

## 規則

1. **修改 `agent/safety.py` 或 `agent/tools.py` 的防護邏輯前，必須先在 `tests/test_safety.py` / `tests/test_tools.py` 加上會失敗的測試**，再改程式讓它通過。不得刪除或放寬既有的安全測試。
2. 工具不得丟例外中斷迴圈：錯誤一律回傳 `ERROR: ...` 字串。
3. 不引入 agent 框架（LangChain 等）；模型一律走 OpenAI 相容 API。
4. 所有函式要有型別標註與簡短 docstring；註解與面向使用者的訊息用繁體中文，給模型看的 system prompt 用英文。
5. 單元測試不可依賴網路、Ollama 或 Docker。
6. 不要把金鑰、個資或任何內部資料寫進 repo。`demo_projects/textstats/.env` 是刻意放的假資料，用來測試 `.env` 封鎖。
7. 共用 GPU：跑真模型時用 `-16k` 別名，評測預設 `--gpu-limit-mb 12000`。
