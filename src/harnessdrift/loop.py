"""第 3 步：把 PROPOSE 和评测串成一个循环，产出一条能拿去看的曲线。

这个模块是整个项目里**唯一有算法含量的地方**，两个问题必须回答清楚：

1. **哪一轮算赢？**（acceptance）
2. **什么时候停？**（stopping）

两者都不是工程细节，而是 RSI 的实验设计。写错了，曲线会漂亮但没意义。

---

## 一、接受规则：不退化才留

朴素 RSI 是 `h_{t+1} = IMPROVE(h_t)`，无条件替换。直接照抄有个后果：
某一轮补丁把 38 条里的 3 条改坏，下一轮就在这棵坏树上继续改，
错误会累积，而你**找不到是哪一轮坏的**。

这里默认 `nonregress`：新树得分 **不低于** 父树才接受。
- 严格上升 → 记为 `improved`，patience 清零
- 持平   → 接受，记为 `neutral`，patience +1
  为什么持平也接受：中性补丁常常是"下一轮才能见效"的铺垫
  （比如把一条规则从提示词挪进 SKILL.md，措辞没变但引用变了）。
- 下降   → **拒绝**，父树不动，patience +1，并把"这份补丁试过且失败"
  喂回改进器，让它换个思路再提一次。
- 均值涨了、但把原本稳定通过的题改挂了 → **也拒绝**。
  只看均值是够不着的：实测第 1 轮 32.67→34.00（净 +1.33），代价是
  T12、T24 从"每轮都过"变成"每轮都挂"。题库是固定的，这 2 条是真回退。
  自改进不该拿旧能力换新能力，所以这个名单非空一律拒绝（`_broke_stable`）。

`--accept strict` 只留严格上升，`--accept always` 复现朴素 RSI 用来做对照。
**三种都跑一遍，就是"接受规则是否重要"的消融实验**，面试可以拿这个说话。

## 二、停止条件：五个都要

只设 `max_rounds` 是不够的——最常见的失败模式是循环空转，
每轮都"持平"，烧钱产出 10 份没用的补丁。所以：

- `max_rounds`     硬上限
- `patience`       连续 N 轮没有严格上升就停（默认 2）
- `perfect`        满分就停
- `proposal_fail`  连续 N 轮产不出合法补丁就停（默认 2）
  —— 说明改进器已经没想法了，再跑也是浪费
- `budget_s`       实际耗时预算

这五个条件里，**patience 是最能体现实验设计的一个**：它把"还在进步"
和"在噪声里打转"分开。

## 三、为什么用 reps 和 std 判断进步

`mean` 的粒度是 1/reps。reps=3 时最小增量是 0.33。
如果只看 mean，+0.33 可能是运气。所以"严格上升"必须**大于噪声**：

    improved = cand.mean > parent.mean + max(0, parent.std - cand.std)

即只有当增益超过父树的抖动幅度时才算真进步。否则记为 neutral。
这一条把"涨了一点但其实是同一棵树的抖动"排除掉了。
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

from .guards import assert_no_grader_verdicts
from .llm import LLM
from .paths import read_tree, tree_hash, write_tree
from .propose import Proposal, collect_failures, propose
from .runner import EvalResult, evaluate_tree
from .tasks import Task

AcceptMode = Literal["nonregress", "strict", "always"]


@dataclass
class Round:
    """一轮的完整记录。base 是第 0 轮，没有补丁。"""

    round_id: int
    kind: str  # base | accepted | rejected
    harness_dir: str
    tree_hash: str
    mean: float
    std: float
    reps: int
    n_tasks: int
    solved: int
    elapsed_s: float
    # 相对父树的判定
    improved: bool = False
    neutral: bool = False
    # 父树稳定通过、候选树稳定挂掉的题：mean 涨了也可能存在，是回退
    broke: list[str] = field(default_factory=list)
    # 这一轮用的补丁（base 为空）
    patch_text: str = ""
    changed: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failure_hist: dict[str, int] = field(default_factory=dict)
    stable_fail: list[str] = field(default_factory=list)
    flaky: dict[str, list[float]] = field(default_factory=dict)
    passes: dict[str, list[float]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.mean / self.n_tasks if self.n_tasks else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "kind": self.kind,
            "harness_dir": self.harness_dir,
            "tree_hash": self.tree_hash,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "reps": self.reps,
            "n_tasks": self.n_tasks,
            "solved": self.solved,
            "elapsed_s": round(self.elapsed_s, 1),
            "improved": self.improved,
            "neutral": self.neutral,
            "broke": self.broke,
            "changed": self.changed,
            "warnings": self.warnings,
            "tokens": self.tokens,
            "failure_hist": dict(sorted(self.failure_hist.items(), key=lambda kv: -kv[1])),
            "stable_fail": self.stable_fail,
            "flaky": self.flaky,
            "passes": {k: [int(x) for x in v] for k, v in self.passes.items()},
            "errors": self.errors,
            "patch_chars": len(self.patch_text),
        }


@dataclass
class LoopResult:
    rounds: list[Round] = field(default_factory=list)
    stop_reason: str = ""
    total_elapsed_s: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> list[Round]:
        return [r for r in self.rounds if r.kind != "rejected"]

    @property
    def best(self) -> Round:
        return max(self.accepted, key=lambda r: (r.mean, -r.std))

    @property
    def total_tokens(self) -> dict[str, int]:
        """整条曲线烧掉的 token。没有用量统计的老曲线会是空字典。"""
        agg: dict[str, int] = {}
        for r in self.rounds:
            for k, v in r.tokens.items():
                agg[k] = agg.get(k, 0) + v
        return agg

    def as_dict(self) -> dict[str, Any]:
        return {
            "when": datetime.now(timezone.utc).isoformat(),
            "config": self.config,
            "stop_reason": self.stop_reason,
            "total_elapsed_s": round(self.total_elapsed_s, 1),
            "baseline": {
                "mean": self.rounds[0].mean if self.rounds else 0.0,
                "std": self.rounds[0].std if self.rounds else 0.0,
                "solved": self.rounds[0].solved if self.rounds else 0,
                "n_tasks": self.rounds[0].n_tasks if self.rounds else 0,
                "reps": self.rounds[0].reps if self.rounds else 0,
            },
            "final": {
                "mean": self.best.mean,
                "std": self.best.std,
                "solved": self.best.solved,
                "round_id": self.best.round_id,
                "harness_dir": self.best.harness_dir,
            },
            "delta": round(self.best.mean - (self.rounds[0].mean if self.rounds else 0.0), 4),
            "rounds": [r.as_dict() for r in self.rounds],
            "total_tokens": self.total_tokens,
        }


def _to_round(n: int, kind: str, path: Path, tree: dict[str, str],
              ev: EvalResult) -> Round:
    return Round(
        round_id=n,
        kind=kind,
        harness_dir=str(path),
        tree_hash=tree_hash(tree),
        mean=ev.mean,
        std=ev.std,
        reps=ev.reps,
        n_tasks=ev.n_tasks,
        solved=ev.solved,
        elapsed_s=ev.elapsed_s,
        failure_hist=ev.failure_hist,
        stable_fail=sorted(ev.stable_fail),
        flaky=ev.flaky,
        passes=ev.passes,
        errors=ev.errors,
        tokens=ev.tokens,
    )


def _is_improved(cand_mean: float, cand_std: float,
                 parent_mean: float, parent_std: float) -> bool:
    """增益是否超过父树的抖动。

    只看 mean 的增量会把噪声当进步：parent 35.0±0.8、cand 35.3±0.5，
    +0.33 完全在父树的抖动范围内，这两次其实是同一棵树。

    参数是裸数值而不是 EvalResult：这是纯算术，纯算术就该能被单独测。
    """
    gain = cand_mean - parent_mean
    # 噪声取两者的较大值，而不是 parent_std - cand_std。
    # 减法的含义是"候选树越稳越容易算进步"，但父树的均值依然带着它自己的
    # ±0.8 不确定性，这个不确定性不会因为候选树很稳就消失。
    # 用 max(...) 才是保守的：任何一边在抖，都要求更大的增益才认账。
    noise = max(parent_std, cand_std)
    return gain > max(noise, 1e-9)


def _broke_stable(parent_passes: dict[str, list[float]],
                  cand_passes: dict[str, list[float]]) -> list[str]:
    """父树稳定通过、候选树稳定挂掉的题。

    mean 是净分，所以"修好 3 条、改坏 2 条"照样能得出 +1 的漂亮增量。
    实测第 1 轮就是这样：35.33 一路上到 34.00（净 +1），代价是 T12、T24
    从"每轮都过"变成"每轮都挂"。评测集是固定题库，这 2 条是真回退，
    不是抖动 —— 自改进不该拿旧能力换新能力。

    所以 nonregress 除了看均值，还要看这个名单：非空就拒绝。
    """
    broke: list[str] = []
    for tid, pv in parent_passes.items():
        if not pv or not all(x >= 1.0 for x in pv):
            continue                      # 父树本来就不稳，不算它
        cv = cand_passes.get(tid)
        if cv is None:
            continue                      # 候选没跑这题（--limit 子集），不判
        if not any(x >= 1.0 for x in cv):
            broke.append(tid)
    return sorted(broke)


def _accepted(mode: AcceptMode, cand_mean: float, parent_mean: float) -> bool:
    if mode == "always":
        return True
    if mode == "strict":
        return cand_mean > parent_mean
    return cand_mean >= parent_mean


def run_loop(
    base_dir: Path,
    tasks: Sequence[Task],
    out_root: Path,
    *,
    llm: LLM,
    improver: LLM,
    reps: int = 3,
    max_rounds: int = 5,
    patience: int = 2,
    proposal_patience: int = 2,
    accept: AcceptMode = "nonregress",
    budget_s: float = 0.0,
    verbose: bool = False,
    log=print,
) -> LoopResult:
    """跑一条完整的 RSI 曲线。

    base_dir 是第 0 轮；每一轮接受的新树写进 out_root/roundN，
    被拒绝的候选写进 out_root/_rejected/roundN（保留证据，不覆盖接受链）。
    """
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()

    base_tree = read_tree(base_dir)
    assert_no_grader_verdicts(base_tree)

    log(f"=== 第 0 轮：基线（{len(tasks)} 条 × {reps} 轮）===")
    base_ev = evaluate_tree(base_tree, tasks, llm, reps=reps, verbose=verbose)
    rounds: list[Round] = [_to_round(0, "base", base_dir, base_tree, base_ev)]
    log(f"基线 = {base_ev.mean:.2f}±{base_ev.std:.2f} / {base_ev.n_tasks} "
        f"({base_ev.pass_rate:.1%})  稳定通过 {base_ev.solved} 条")

    parent_tree, parent_ev = base_tree, base_ev
    parent_round = 0
    stop_reason = ""
    idle = 0          # 连续没严格上升
    patch_fails = 0   # 连续产不出合法补丁
    tried: list[str] = []  # 被拒绝过的补丁，喂回改进器避免重复

    for n in range(1, max_rounds + 1):
        if budget_s and time.time() - started > budget_s:
            stop_reason = f"超出耗时预算 {budget_s:.0f}s"
            break
        if parent_ev.mean >= len(tasks):
            stop_reason = "已满分，无可改进"
            break

        log(f"\n=== 第 {n} 轮：PROPOSE ===")
        failures = collect_failures(parent_ev.raw)
        log(f"失败样本 {len(failures)} 条：{[f.task_id for f in failures]}")

        if not failures:
            stop_reason = "没有失败样本可改进"
            break

        # 把之前被拒绝的补丁附在提示里，避免它原地打转
        prop: Proposal = propose(parent_tree, failures, out_root / f"round{n}", improver)
        if prop.warnings and not prop.applied and tried:
            extra = "\n".join(tried[-2:])
            log(f"  补丁无效（{prop.warnings[:1]}），带上历史失败重试一次")
            prop = propose(parent_tree, failures, out_root / f"round{n}", improver,
                           extra_context=f"以下补丁你已经试过并被拒绝了，不要重复：\n{extra}")

        if not prop.applied:
            patch_fails += 1
            log(f"  补丁未通过守卫：{prop.warnings}")
            rounds.append(Round(
                round_id=n, kind="rejected", harness_dir="", tree_hash="",
                mean=parent_ev.mean, std=parent_ev.std, reps=reps,
                n_tasks=len(tasks), solved=parent_ev.solved, elapsed_s=0.0,
                warnings=prop.warnings, patch_text=prop.patch_text,
            ))
            if patch_fails >= proposal_patience:
                stop_reason = f"连续 {patch_fails} 轮产不出合法补丁，改进器已无可用想法"
                break
            continue
        patch_fails = 0

        # PATCH.md 是补丁存档，不是 harness 的一部分。不排除的话它会被
        # 当成新的 harness 文件带进下一轮，最后还跟着 best/ 一起发出去。
        cand_tree = read_tree(Path(prop.round_dir), ignore=("PATCH.md",))
        assert_no_grader_verdicts(cand_tree)
        log(f"  补丁通过守卫：{prop.changed}")

        log(f"=== 第 {n} 轮：评测候选 ===")
        cand_ev = evaluate_tree(cand_tree, tasks, llm, reps=reps, verbose=verbose)
        improved = _is_improved(cand_ev.mean, cand_ev.std, parent_ev.mean, parent_ev.std)
        neutral = (not improved) and cand_ev.mean >= parent_ev.mean
        take = _accepted(accept, cand_ev.mean, parent_ev.mean)
        # 均值达标但踩坏了旧能力 -> 仍然拒绝（只在 nonregress 下生效；
        # strict 本来就要求严格上升，always 就是"什么都不管"）。
        broke = _broke_stable(parent_ev.passes, cand_ev.passes) if accept == "nonregress" else []
        blocked_by_regression = take and bool(broke)
        if blocked_by_regression:
            take = False

        delta = cand_ev.mean - parent_ev.mean
        verdict = ("上升" if improved else "持平" if neutral else "下降")
        if blocked_by_regression:
            verdict += f"，但改坏 {len(broke)} 条"
        log(f"  候选 = {cand_ev.mean:.2f}±{cand_ev.std:.2f}  "
            f"(Δ{delta:+.2f}, {verdict})  → {'接受' if take else '拒绝'}")
        if broke and not take:
            log(f"    回退：{' '.join(broke)} 本来稳定通过，候选稳定挂掉")

        r = _to_round(n, "accepted" if take else "rejected", Path(prop.round_dir), cand_tree, cand_ev)
        r.improved, r.neutral, r.broke = improved, neutral, broke
        r.patch_text, r.changed, r.warnings = prop.patch_text, prop.changed, prop.warnings
        rounds.append(r)

        if take:
            parent_tree, parent_ev, parent_round = cand_tree, cand_ev, n
            if improved:
                idle = 0
                log(f"  ✓ 接受第 {n} 轮为新基线")
            else:
                idle += 1
                log(f"  ~ 持平接受（选做铺垫），patience {idle}/{patience}")
        else:
            idle += 1
            tried.append(prop.patch_text)
            shutil.rmtree(out_root / f"round{n}", ignore_errors=True)
            # 被拒绝的候选留证据，但不占 roundN 这个位置
            rej = out_root / "_rejected" / f"round{n}"
            rej.parent.mkdir(parents=True, exist_ok=True)
            if rej.exists():
                shutil.rmtree(rej)
            rej.mkdir(parents=True)
            for p, c in cand_tree.items():
                fp = rej / p
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(c, encoding="utf-8", newline="")
            (rej / "PATCH.md").write_text(prop.patch_text, encoding="utf-8")
            log(f"  ✗ 拒绝，父树保持第 {parent_round} 轮，patience {idle}/{patience}")

        if idle >= patience:
            stop_reason = f"连续 {idle} 轮没有实质改进（patience={patience}）"
            break
    else:
        stop_reason = f"达到轮数上限 max_rounds={max_rounds}"

    if not stop_reason:
        stop_reason = "循环结束"

    result = LoopResult(
        rounds=rounds,
        stop_reason=stop_reason,
        total_elapsed_s=time.time() - started,
        config={
            "base_dir": str(base_dir),
            "reps": reps,
            "max_rounds": max_rounds,
            "patience": patience,
            "proposal_patience": proposal_patience,
            "accept": accept,
            "budget_s": budget_s,
            "n_tasks": len(tasks),
            "solver_model": llm.model,
            "improver_model": improver.model,
        },
    )
    log(f"\n=== 结束：{stop_reason} ===")
    log(f"基线 {rounds[0].mean:.2f} → 最佳 {result.best.mean:.2f} "
        f"（第 {result.best.round_id} 轮，Δ{result.best.mean - rounds[0].mean:+.2f}）")
    return result