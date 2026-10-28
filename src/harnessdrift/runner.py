"""跑一棵 harness 树，把结果收成一个可比较的对象。

为什么单独一个模块：`cli run` 和 RSI 循环都要做同一件事（跑 N 轮、算均值方差、
收失败证据）。写两份一定会漂——第 2 步就吃过这个亏：CLI 里的判分和
propose 拿到的证据不是同一批对象，导致工具调用记录对不上。

这里只负责"跑 + 收"，不做任何判断。接受/拒绝哪一轮是 loop.py 的事，
报告长什么样是 report.py 的事。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .agent import RunResult, run_task
from .evaluate import TaskScore, judge
from .llm import LLM
from .tasks import Task


@dataclass
class EvalResult:
    """一次评测（同一棵树跑 reps 轮）的全部产物。"""

    n_tasks: int
    reps: int
    # passes[task_id] = [1.0, 0.0, 1.0] —— 每条题每轮的成败
    passes: dict[str, list[float]] = field(default_factory=dict)
    # 全部轮次累计的失败项计数，用来找"主因"
    failure_hist: dict[str, int] = field(default_factory=dict)
    # 最后一轮每条的失败项，用来给改进器当证据
    last_failed: dict[str, list[str]] = field(default_factory=dict)
    # 最后一轮的原始对象 (task, RunResult, TaskScore, Toolkit)，
    # propose 需要这些才能写出带工具调用的失败样本
    raw: list[tuple[Task, RunResult, TaskScore, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0
    errors: dict[str, str] = field(default_factory=dict)
    # 这次评测实际消耗的 token（调用方没有用量统计时为空）
    tokens: dict[str, int] = field(default_factory=dict)

    @property
    def totals(self) -> list[float]:
        """每轮的总分，用来算离散度。"""
        if not self.passes:
            return []
        return [sum(v[rep] for v in self.passes.values()) for rep in range(self.reps)]

    @property
    def mean(self) -> float:
        """各题均值的和。**用各题均值而不是各轮均值**，这样 reps 不一致也能比。"""
        per_task = [sum(v) / len(v) for v in self.passes.values() if v]
        return sum(per_task)

    @property
    def std(self) -> float:
        """各轮总分的总体标准差。

        只报均值会撒谎：38 条的均值 35.0 可能来自 35/35/35，也可能来自
        36/35/34。后者说明有题在抖，自改进拿它当基线会把运气当收益。
        """
        if self.reps <= 1:
            return 0.0
        t = self.totals
        mu = sum(t) / len(t)
        return (sum((x - mu) ** 2 for x in t) / len(t)) ** 0.5

    @property
    def pass_rate(self) -> float:
        return self.mean / self.n_tasks if self.n_tasks else 0.0

    @property
    def solved(self) -> int:
        """稳定通过的题数（每轮都过）。这是比 mean 更保守的口径。"""
        return sum(1 for v in self.passes.values() if v and all(x >= 1.0 for x in v))

    @property
    def flaky(self) -> dict[str, list[float]]:
        """时过时不过的题 —— 真随机，不是能力。"""
        return {k: v for k, v in self.passes.items() if len(set(v)) > 1}

    @property
    def stable_fail(self) -> dict[str, list[float]]:
        """每轮都挂的题 —— 真缺陷，是改进的主要目标。"""
        return {k: v for k, v in self.passes.items() if v and not any(x >= 1.0 for x in v)}

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_tasks": self.n_tasks,
            "reps": self.reps,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "pass_rate": round(self.pass_rate, 4),
            "solved": self.solved,
            "elapsed_s": round(self.elapsed_s, 1),
            "passes": {k: [int(x) for x in v] for k, v in self.passes.items()},
            "failure_hist": dict(sorted(self.failure_hist.items(), key=lambda kv: -kv[1])),
            "flaky": self.flaky,
            "stable_fail": list(self.stable_fail),
            "errors": self.errors,
            "tokens": self.tokens,
        }


def evaluate_tree(
    tree: dict[str, str],
    tasks: Sequence[Task],
    llm: LLM,
    *,
    reps: int = 1,
    verbose: bool = False,
    on_task: Callable[[str, int, int, float], None] | None = None,
) -> EvalResult:
    """跑 reps 轮，每轮把 tasks 全跑一遍。

    on_task(task_id, rep, turn?, value) 用来给 CLI 打进度，避免长时间无输出。
    """
    res = EvalResult(n_tasks=len(tasks), reps=reps)
    res.passes = {t.id: [] for t in tasks}
    started = time.time()
    before = llm.snapshot_usage() if hasattr(llm, "snapshot_usage") else None

    for rep in range(reps):
        for t in tasks:
            toolkit = t.build_toolkit()
            try:
                r = run_task(tree, t.message, toolkit, llm, verbose=verbose)
            except Exception as e:  # noqa: BLE001
                # 单条崩掉不能带停整个评测：那一轮已被污染的分数还不如显式记 0。
                res.errors[t.id] = f"{type(e).__name__}: {e}"
                res.passes[t.id].append(0.0)
                res.last_failed[t.id] = [f"ERROR {type(e).__name__}"]
                if on_task:
                    on_task(t.id, rep, len(tasks), 0.0)
                continue

            score = judge(t, r, toolkit)
            res.passes[t.id].append(score.value)
            if score.failed:
                res.last_failed[t.id] = [c.name for c in score.failed]
                for c in score.failed:
                    res.failure_hist[c.name] = res.failure_hist.get(c.name, 0) + 1
            if on_task:
                on_task(t.id, rep, len(tasks), score.value)
            # raw 只留最后一轮：改进器看的是"现在的树怎么错的"，
            # 前面几轮的失败已经被这一轮的代表掉了。
            if rep == reps - 1:
                res.raw.append((t, r, score, toolkit))

    res.elapsed_s = time.time() - started
    if before is not None:  # 注意是 None 不是空字典：第一次评测时 before 就是 {}
        res.tokens = llm.usage_since(before)
    return res


def format_progress(task_id: str, rep: int, total: int, value: float) -> str:
    return f"  [{'PASS' if value >= 1 else 'FAIL'}] {task_id}  ({rep + 1}/{total})"