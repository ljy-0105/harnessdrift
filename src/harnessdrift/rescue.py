"""被否决的轮做原子级抢救。

## 这个模块解决的是一个具体的浪费

`runs/attr_feed1` 里第 1 轮拿到 +4.00，但 T12 从 3/3 掉到 0/3，被一票否决，
整轮扔掉。可单点实验（`runs/probe_noavc`）显示：那一轮一共 5 个原子，只要删掉
`tool_descriptions/action_vs_confirm.md` 这一个，其余全留 —— 全量重测 **46.00 分
一分不差**，T12 回到 3/3。

也就是说否决拦对了人，但扔掉了 90% 的有效改动。这里的做法是：**不整轮丢，
逐个原子试删，删到安全题全过为止，再拿剩下的树去重测。**

## 为什么是贪心，不是穷举

n 个原子有 2^n 个子集。实际 n 一般在 3~8，但每个子集都要跑一次评测 ——
穷举是几十次全量评测，代价和收益完全不匹配。贪心的依据是「毒性通常集中在少数
原子」这个经验事实，而每一轮都可以验证：救出来的树最终要跑**全量**评测，
分数不达标就不收。所以贪心可能救不回来，但**不会救错**（验收规则和正常轮一致）。

## 一个必须说清楚的边界

抢救只在**被否决**的轮上跑，救出来还要过一遍和正常轮完全相同的验收规则
（`--accept` + 回归名单 + 安全否决）。它不会降低标准，只是给了「删掉毒原子
的候选」第二次机会。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .ablate import Atom, apply_all, split_atoms
from .runner import EvalResult, evaluate_tree
from .tasks import Task

__all__ = ["RescueAttempt", "RescueResult", "rescue_vetoed"]


@dataclass
class RescueAttempt:
    """一次「删掉这些原子然后测一下」。"""

    dropped: list[int]
    labels: list[str]
    probe_passes: int
    probe_total: int

    @property
    def clean(self) -> bool:
        return self.probe_total > 0 and self.probe_passes == self.probe_total

    def as_dict(self) -> dict[str, Any]:
        return {
            "dropped": list(self.dropped),
            "labels": list(self.labels),
            "probe_passes": self.probe_passes,
            "probe_total": self.probe_total,
            "clean": self.clean,
        }


@dataclass
class RescueResult:
    ok: bool = False
    reason: str = ""
    n_atoms: int = 0
    dropped: list[Atom] = field(default_factory=list)
    attempts: list[RescueAttempt] = field(default_factory=list)
    tree: dict[str, str] | None = None
    probe_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "n_atoms": self.n_atoms,
            "dropped": [
                {"index": a.index, "path": a.path, "label": a.label}
                for a in self.dropped
            ],
            "attempts": [a.as_dict() for a in self.attempts],
            "probe_ids": list(self.probe_ids),
        }


def _count_passes(passes: Mapping[str, Sequence[float]], ids: Sequence[str]) -> tuple[int, int]:
    """probe 题里有多少条**每一次**都过。缺题的算挂（宁可救不回来，也别虚报）。"""
    ok = 0
    for tid in ids:
        got = passes.get(tid)
        if got and all(float(x) >= 1.0 for x in got):
            ok += 1
    return ok, len(ids)


def rescue_vetoed(
    parent_tree: dict[str, str],
    cand_tree: dict[str, str],
    tasks: Sequence[Task],
    llm: Any,
    *,
    probe_ids: Sequence[str],
    reps: int = 1,
    workers: int = 1,
    max_drop: int = 3,
    max_variants: int = 12,
    verbose: bool = False,
    log: Callable[[str], None] = lambda _s: None,
) -> RescueResult:
    """试着删最少的原子，让 ``probe_ids`` 这些题全部稳定通过。

    ``probe_ids`` 是**探测器**：安全题被踩了就给 safety 那几条，普通回退就给
    被踩坏的那几条。只在这些题上评测，一次几秒，比全量便宜一个数量级。

    返回的 ``tree`` 是抢救后的树（``ok=True`` 时才有），**分数没测** ——
    全量评测留给调用方，因为那是要记进曲线、要花大钱的证据。
    """
    atoms = split_atoms(parent_tree, cand_tree)
    res = RescueResult(n_atoms=len(atoms), probe_ids=list(probe_ids))
    if not atoms:
        res.reason = "候选与父树之间没有可拆的原子（差异只在空白上）"
        return res

    probe_tasks = [t for t in tasks if t.id in set(probe_ids)]
    if not probe_tasks:
        res.reason = f"探测题 {list(probe_ids)} 不在题库里"
        return res

    dropped: list[int] = []
    best: tuple[int, int] | None = None   # (-passes, index) 用于挑"最像解"的那一支
    best_dropped: list[int] = []

    for _step in range(max(1, max_drop)):
        if len(res.attempts) >= max_variants:
            res.reason = f"达到尝试上限 max_variants={max_variants}，没救回来"
            break
        candidates = [a for a in atoms if a.index not in dropped]
        if not candidates:
            break

        step_best: tuple[int, int] | None = None
        step_best_dropped: list[int] = []

        for atom in candidates:
            if len(res.attempts) >= max_variants:
                break
            drop = [*dropped, atom.index]
            keep = [a for a in atoms if a.index not in drop]
            tree = apply_all(parent_tree, cand_tree, keep)
            ev: EvalResult = evaluate_tree(
                tree, probe_tasks, llm, reps=reps, verbose=verbose, workers=workers,
            )
            passes, total = _count_passes(ev.passes, list(probe_ids))
            labels = [a.label for a in atoms if a.index in drop]
            res.attempts.append(RescueAttempt(drop, labels, passes, total))
            log(f"    抢救试探：删 {len(drop)} 个原子 → 探测题 {passes}/{total}")

            if passes == total:
                res.ok = True
                res.dropped = [a for a in atoms if a.index in drop]
                res.tree = tree
                res.reason = (
                    f"删掉 {len(drop)} 个原子后，{len(probe_ids)} 条探测题全部稳定通过"
                )
                return res
            if step_best is None or passes > -step_best[0]:
                step_best = (-passes, atom.index)
                step_best_dropped = drop

        if step_best is None:
            break
        dropped = step_best_dropped
        best, best_dropped = step_best, step_best_dropped

    if best is not None:
        res.reason = (
            f"最多删 {len(best_dropped)} 个原子，探测题最好只到 "
            f"{-best[0]}/{len(probe_ids)}，没救回来"
        )
    elif not res.reason:
        res.reason = "没有可尝试的原子"
    return res
