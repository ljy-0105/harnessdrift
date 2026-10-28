"""用量采集：不联网，用一个会记账的假 LLM 验算。

只锁两件事：
1. **第一次**评测也要记到（`before` 是空字典时不能被当成"没有统计能力"）；
2. 每次评测记的是**增量**，不是累计。
"""

from __future__ import annotations

from harnessdrift.agent import run_task  # noqa: F401  (仅为确保模块可导入)
from harnessdrift.llm import LLM, LLMReply
from harnessdrift.runner import evaluate_tree
from harnessdrift.tasks import Task


def _one_task():
    """最小任务：这里测的是记账，不是客服能力。"""
    return [Task(id="T01", message="请回复", why="用量测试",
                 expect="no_return", customers=[], orders=[])]


class _FakeLLM(LLM):
    def __init__(self) -> None:
        # 不调父类 __init__：那条路会读 env / 建 client，测试里都不需要
        import threading
        self.usage_total: dict[str, int] = {}
        self._lock = threading.Lock()
        self.calls = 0

    def chat(self, messages, tools=None, **kw):  # noqa: ANN001, ANN201
        self.calls += 1
        reply = LLMReply(text="好的", usage={"prompt_tokens": 10,
                                             "completion_tokens": 2,
                                             "total_tokens": 12})
        self._account(reply.usage)
        return reply


def test_first_eval_records_tokens():
    llm = _FakeLLM()
    ev = evaluate_tree({"systemprompt.md": "你是客服"}, _one_task(), llm, reps=1)
    assert llm.calls > 0
    assert ev.tokens.get("total_tokens") == 12 * llm.calls


def test_second_eval_records_delta_not_total():
    llm = _FakeLLM()
    evaluate_tree({"systemprompt.md": "你是客服"}, _one_task(), llm, reps=1)
    first_calls = llm.calls
    ev2 = evaluate_tree({"systemprompt.md": "你是客服"}, _one_task(), llm, reps=1)
    delta_calls = llm.calls - first_calls
    assert ev2.tokens.get("total_tokens") == 12 * delta_calls
    assert ev2.tokens.get("total_tokens") != 12 * llm.calls