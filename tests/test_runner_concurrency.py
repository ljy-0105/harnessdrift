"""并发评测不能改变结果。

并发只该改变"多久跑完"，不该改变"跑出什么"。所以这里的断言全是**等价性**：
workers=1 和 workers=8 必须算出同一份 passes、同一个 token 账。

如果哪天它们不等，说明共享状态被并发写坏了 —— 而这种 bug 不会报错，
只会让曲线上的 ± 变大，然后把噪声当成改进。
"""
from __future__ import annotations

import threading
import time

from harnessdrift.llm import LLMReply
from harnessdrift.runner import evaluate_tree
from harnessdrift.tasks import Task


class _SlowLLM:
    """每条任务睡一会儿，模拟真实调用；同时把用量记账暴露出来。"""

    def __init__(self, delay: float = 0.01) -> None:
        self.model = "fake"
        self.delay = delay
        self.calls = 0
        self.usage_total: dict[str, int] = {}
        self._lock = threading.Lock()

    def chat(self, messages, **kw):
        time.sleep(self.delay)
        with self._lock:
            self.calls += 1
            self.usage_total["total_tokens"] = self.usage_total.get("total_tokens", 0) + 7
        return LLMReply(text="好的，已为您处理。", usage={"total_tokens": 7})

    def snapshot_usage(self) -> dict[str, int]:
        with self._lock:
            return dict(self.usage_total)

    def usage_since(self, before: dict[str, int]) -> dict[str, int]:
        with self._lock:
            return {k: v - before.get(k, 0) for k, v in self.usage_total.items()
                    if v - before.get(k, 0)}


def _tasks(n: int = 8) -> list[Task]:
    return [
        Task(id=f"T{i:02d}", message=f"第 {i} 单要退", why="并发测试",
             expect="no_return", customers=[], orders=[])
        for i in range(n)
    ]


TREE = {"systemprompt.md": "# 你是客服\n", "tool_descriptions/a.md": "查订单。\n"}


def test_workers_do_not_change_the_score_or_the_order():
    tasks = _tasks()
    seq = evaluate_tree(TREE, tasks, _SlowLLM(), reps=2, workers=1)
    par = evaluate_tree(TREE, tasks, _SlowLLM(), reps=2, workers=8)
    assert seq.passes == par.passes
    # 键顺序也要一致：curve.json 和看板按这个顺序印题号
    assert list(seq.passes) == list(par.passes)
    assert seq.mean == par.mean


def test_concurrent_usage_accounting_is_exact():
    """usage_total 的 += 不是原子操作，丢一个计数账就少一笔。"""
    llm = _SlowLLM(delay=0.0)
    tasks = _tasks(20)
    ev = evaluate_tree(TREE, tasks, llm, reps=2, workers=8)
    assert llm.calls == 40, "20 题 × 2 轮 = 40 次调用，一次都不能丢"
    assert ev.tokens["total_tokens"] == 280, "7 token × 40 次"


def test_workers_actually_overlap():
    """并发要真的并发，否则这个开关只是心理安慰。"""
    tasks = _tasks(8)
    t0 = time.time()
    evaluate_tree(TREE, tasks, _SlowLLM(delay=0.15), reps=1, workers=1)
    seq_s = time.time() - t0
    t0 = time.time()
    evaluate_tree(TREE, tasks, _SlowLLM(delay=0.15), reps=1, workers=8)
    par_s = time.time() - t0
    assert par_s < seq_s / 2, f"串行 {seq_s:.2f}s，8 路并发 {par_s:.2f}s，没提速"


def test_one_crashing_task_does_not_kill_the_run():
    """一条崩了记 0 分，其余照跑 —— 并发只是把这个行为放大了。"""
    class _Boom(_SlowLLM):
        def chat(self, messages, **kw):
            if "第 3 单" in str(messages):
                raise RuntimeError("炸了")
            return super().chat(messages, **kw)

    tasks = _tasks(5)
    ev = evaluate_tree(TREE, tasks, _Boom(delay=0.0), reps=1, workers=4)
    assert ev.passes["T03"] == [0.0]
    assert "T03" in ev.errors and "炸了" in ev.errors["T03"]
    assert sum(1 for v in ev.passes.values() for p in v if p == 1.0) == 4


def test_last_failed_tracks_the_last_rep_only():
    """前一轮失败、最后一轮通过时，run --out 不能继续报旧失败。"""
    class _AlternatingLLM(_SlowLLM):
        def __init__(self) -> None:
            super().__init__(delay=0.0)
            self.n = 0

        def chat(self, messages, **kw):
            self.n += 1
            return LLMReply(text="旧答复" if self.n == 1 else "MAGIC", usage={"total_tokens": 1})

    task = Task(id="T01", message="测试", why="last_failed 测试",
                expect="no_return", customers=[], orders=[],
                answer_must_mention=["MAGIC"])
    ev = evaluate_tree(TREE, [task], _AlternatingLLM(), reps=2)
    assert ev.passes["T01"] == [0.0, 1.0]
    assert ev.last_failed == {}
