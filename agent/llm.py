"""LLM 介面：OpenAI 相容 client（Ollama / vLLM）與測試用的 ScriptedLLM。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Union


@dataclass
class ToolCall:
    """模型要求的一次工具呼叫。arguments 保留原始 JSON 字串，解析交給 Toolbox。"""

    id: str
    name: str
    arguments: str


@dataclass
class LLMResponse:
    """與供應商無關的回應格式。"""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class LLM(Protocol):
    """迴圈只依賴這個介面，所以可以注入假的 LLM 做測試。"""

    model: str

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


class LLMError(Exception):
    """呼叫模型失敗（連線錯誤、重試用盡）。"""


class OpenAICompatLLM:
    """包裝 openai SDK。換成 vLLM 只要改 base_url。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        temperature: float = 0.0,
        timeout: float = 180.0,
        retries: int = 1,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
    ) -> None:
        from openai import OpenAI  # 延遲 import：單元測試不需要 openai 連線

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.retries = retries
        self.max_tokens = max_tokens  # 防止小模型陷入無止境的重複生成
        # 思考型模型（如 qwen3）可設 "none" 關閉思考，否則思考內容會吃光 max_tokens
        self.reasoning_effort = reasoning_effort

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        """送出一次 chat completion；連線類錯誤會退避重試。"""
        from openai import APIConnectionError, APITimeoutError, InternalServerError

        extra: dict[str, Any] = {}
        if self.reasoning_effort:
            extra["reasoning_effort"] = self.reasoning_effort
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    tools=tools,  # type: ignore[arg-type]
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **extra,
                )
                break
            except (APIConnectionError, APITimeoutError, InternalServerError) as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        else:
            raise LLMError(f"模型呼叫失敗（{type(last).__name__}）：{last}")

        choice = resp.choices[0]
        msg = choice.message
        calls = [
            ToolCall(id=tc.id or f"call_{i}", name=tc.function.name, arguments=tc.function.arguments or "{}")
            for i, tc in enumerate(msg.tool_calls or [])
        ]
        usage = {}
        if resp.usage:
            usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
        if choice.finish_reason == "length":
            usage["truncated"] = 1
        if not calls and not (msg.content or "").strip() and (usage.get("completion_tokens") or 0) > 20:
            # 模型生成了內容，但 Ollama 的工具呼叫解析器把格式壞掉的呼叫整個吞掉了
            usage["swallowed"] = 1
        return LLMResponse(content=msg.content, tool_calls=calls, usage=usage)


# ---------------------------------------------------------------- 測試用

Step = Union[LLMResponse, Callable[[list[dict[str, Any]]], LLMResponse]]


class ScriptedLLM:
    """依序回傳預先寫好的回應；步驟也可以是看 messages 決定回應的函式。

    腳本用完時回傳一段沒有工具呼叫的文字（等同模型宣布結束）。
    """

    model = "scripted-fake"

    def __init__(self, steps: list[Step]) -> None:
        self.steps = list(steps)
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        if not self.steps:
            return LLMResponse(content="（腳本結束）")
        step = self.steps.pop(0)
        return step(messages) if callable(step) else step


def tool(name: str, arguments: str | dict[str, Any], call_id: str | None = None) -> LLMResponse:
    """測試輔助：產生一個只含單一工具呼叫的回應。"""
    import json

    args = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    return LLMResponse(content=None, tool_calls=[ToolCall(id=call_id or f"c_{name}", name=name, arguments=args)])


def say(text: str) -> LLMResponse:
    """測試輔助：產生一個純文字回應。"""
    return LLMResponse(content=text)
