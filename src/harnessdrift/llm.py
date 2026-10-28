"""LLM 适配器：OpenAI 兼容接口（DeepSeek / Qwen / 任意中转）。

刻意做得薄：只管"发请求、收工具调用、收文本"。重试放在这一层，
因为它是唯一知道 HTTP 语义的地方；agent 逻辑不该感知网络抖动。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMReply:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    # 模型为什么停下来：'stop' 正常结束，'length' 撞上 max_tokens 被截断，
    # 'tool_calls' 要调工具。**截断必须能被上层看见** —— 自改进里拿一份被截断的
    # 提示词去测，得到的是一个凭空掉下来的分数，而且会被记成"这一轮的改进"。
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    @property
    def is_tool_call(self) -> bool:
        return bool(self.tool_calls)


def load_env(path: str = ".env") -> None:
    """极简 .env 读取，不引第三方依赖。已在环境里的变量优先。"""
    p = os.path.join(os.getcwd(), path)
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


class LLM:
    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 4,
    ) -> None:
        load_env()
        self.model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.temperature = float(
            temperature if temperature is not None else os.environ.get("LLM_TEMPERATURE", "0")
        )
        self.max_tokens = int(
            max_tokens if max_tokens is not None else os.environ.get("LLM_MAX_TOKENS", "1200")
        )
        self.max_retries = max_retries
        self._client = None
        # 累计用量：evaluate_tree 在每次评测前后取差，就能把"这一轮花了多少 token"
        # 落进 curve.json。不加这个的话，跑完只知道涨了几分，不知道花了多少钱。
        self.usage_total: dict[str, int] = {}

    def snapshot_usage(self) -> dict[str, int]:
        return dict(self.usage_total)

    def usage_since(self, before: dict[str, int]) -> dict[str, int]:
        """返回 before 之后新增的用量（只算增量，不含历史）。"""
        return {k: v - before.get(k, 0) for k, v in self.usage_total.items()
                if v - before.get(k, 0)}

    def _account(self, usage: dict[str, int]) -> None:
        for k, v in (usage or {}).items():
            if isinstance(v, int):
                self.usage_total[k] = self.usage_total.get(k, 0) + v

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "需要 openai 包：./.venv/Scripts/python.exe -m pip install openai"
                ) from e
            if not self.api_key:
                raise RuntimeError(
                    "LLM_API_KEY 为空。把 key 写进 harnessdrift/.env（参考 .env.example）"
                )
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> LLMReply:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in _to_oai(tools)]
            kwargs["tool_choice"] = tool_choice

        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self.client.chat.completions.create(**kwargs)
                m = r.choices[0].message
                calls = []
                for tc in (m.tool_calls or []):
                    calls.append(
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": _parse_args(tc.function.arguments),
                            "raw_arguments": tc.function.arguments,
                        }
                    )
                reply = LLMReply(
                    text=m.content or "",
                    tool_calls=calls,
                    raw=m.model_dump(),
                    usage=dict(r.usage.model_dump()) if r.usage else {},
                    finish_reason=r.choices[0].finish_reason,
                )
                self._account(reply.usage)
                return reply
            except Exception as e:  # noqa: BLE001 - 网络/限流都走同一策略
                last = e
                wait = 2**attempt
                print(f"  [llm] attempt {attempt + 1} failed: {type(e).__name__}: {e}; "
                      f"retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts") from last


def _parse_args(raw: str) -> dict[str, Any]:
    import json

    if not raw or not raw.strip():
        return {}
    try:
        v = json.loads(raw)
    except json.JSONDecodeError:
        return {"__malformed__": raw}
    return v if isinstance(v, dict) else {"__malformed__": raw}


def _to_oai(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把内部工具声明转成 OpenAI function 格式（字段名一致，这里只是显式化）。"""
    return [
        {
            "name": s["name"],
            "description": s["description"],
            "parameters": s["parameters"],
        }
        for s in specs
    ]