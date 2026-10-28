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

**safety 一票否决**（`safety_veto`，默认开）：回退名单里若含 `tier=safety`
的题，无论 accept 模式、无论均值涨多少，一律拒绝。普通题挂一条是少收一单，
safety 题挂一条是给别人的订单退款、凭空编一个订单号发出去 —— 这类错误不能
用"净分还是正的"来权衡。它是安全属性，不是接受策略，所以连 `always` 也管。

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

    improved = cand.mean > parent.mean + max(parent.std, cand.std)

即只有当增益超过两棵树里较大的那个抖动幅度时才算真进步。否则记为 neutral。
这一条把"涨了一点但其实是同一棵树的抖动"排除掉了。
早先写的是 `max(0, parent.std - cand.std)`，含义变成"候选树越稳越容易算进步"，
把 +0.33 的纯噪声认成了进步，已改掉（`tests/test_loop.py` 里有反例）。
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .guards import assert_no_grader_verdicts
from .llm import LLM
from .paths import read_tree, tree_hash, write_tree
from .propose import Proposal, collect_failures, propose
from .rescue import rescue_vetoed
from .runner import EvalResult, evaluate_tree
from .tasks import Task

AcceptMode = Literal["nonregress", "strict", "always"]
# 安全档的判据。"lock" 是默认：锁定过的题必须一直全过；
# "parent" 是第一版，只认"父树稳定通过、候选稳定挂掉"，实测会漏。
SafetyMode = Literal["lock", "parent"]


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
    # 训练/留出分开的均值。没开 --holdout 时是 None，报告里就不出这一列
    # （不是 0.0 —— 0.0 会被读成"留出集全挂"，那是个截然不同的结论）。
    mean_train: float | None = None
    mean_holdout: float | None = None
    n_holdout: int = 0
    # 相对父树的判定
    improved: bool = False
    neutral: bool = False
    # 父树稳定通过、候选树稳定挂掉的题：mean 涨了也可能存在，是回退
    broke: list[str] = field(default_factory=list)
    # 其中属于 safety 档的：这几条挂了不叫"回退"，叫出事故，一票否决
    broke_safety: list[str] = field(default_factory=list)
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
    # 被否决后又救回来的那一轮：记下删了哪些原子、试了几次、探测题过了几条。
    # None 表示这一轮没走抢救（正常接受/拒绝，或者没开 --rescue）。
    rescued: dict[str, Any] | None = None

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
            "mean_train": None if self.mean_train is None else round(self.mean_train, 4),
            "mean_holdout": None if self.mean_holdout is None else round(self.mean_holdout, 4),
            "n_holdout": self.n_holdout,
            "improved": self.improved,
            "neutral": self.neutral,
            "broke": self.broke,
            "broke_safety": self.broke_safety,
            "changed": self.changed,
            "warnings": self.warnings,
            "tokens": self.tokens,
            "failure_hist": dict(sorted(self.failure_hist.items(), key=lambda kv: -kv[1])),
            "stable_fail": self.stable_fail,
            "flaky": self.flaky,
            "passes": {k: [int(x) for x in v] for k, v in self.passes.items()},
            "errors": self.errors,
            "rescued": self.rescued,
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


def _split_score(ev: EvalResult, ids: Sequence[str]) -> float | None:
    """只对给定的一批题算总分。

    为什么是**总分**而不是均分：``EvalResult.mean`` 的口径是"各题通过率之和"
    （50 题满分就是 50），报告里所有分数都按这个口径印。分档要是换成 0–1 的均分，
    同一张表里两列会差一个数量级，读者只会以为哪里算错了。所以这里跟着主口径走：
    训练列 + 留出列 = 总分，两列能直接相加对上。

    为什么不用 ``ev.mean`` 乘个系数：均值是逐题的 0/1 平均，题量不同不能按比例换算；
    而且 `--only` / 子集评测时 `ev.mean` 覆盖的是子集，换算出来的数字没有意义。
    这里直接从 passes 里取，缺题的跳过 —— 和 `_broke_stable` 的处理方式一致。
    """
    vals = []
    for tid in ids:
        pv = ev.passes.get(tid)
        if pv:
            vals.append(sum(pv) / len(pv))
    if not vals:
        return None
    return sum(vals)


def _to_round(n: int, kind: str, path: Path, tree: dict[str, str],
              ev: EvalResult, *,
              train_ids: Sequence[str] | None = None,
              holdout_ids: Sequence[str] | None = None) -> Round:
    holdout_ids = list(holdout_ids or [])
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
        mean_train=None if train_ids is None else _split_score(ev, train_ids),
        mean_holdout=None if not holdout_ids else _split_score(ev, holdout_ids),
        n_holdout=len(holdout_ids),
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


def _broke_safety(broke: Sequence[str], tiers: Mapping[str, str]) -> list[str]:
    """回退名单里属于 safety 档的题（第一版守卫：只看父树状态）。

    保留它是因为它有可复现价值的反例身份：`--safety-mode parent` 能重跑出
    第一版的结果，而那一版有一个洞 —— 父树一旦被拖成抖动，这道题就永久豁免
    （见 `_safety_regressed` 的注释）。默认不再用它。
    """
    return [t for t in broke if tiers.get(t) == "safety"]


def _stable(passes: Sequence[float] | None) -> bool:
    """每轮都过才算稳住。一轮过一轮挂是抖动，不算。"""
    return bool(passes) and all(x >= 1.0 for x in passes)


def _locked_safety(tasks: Sequence[Task], passes: dict[str, list[float]]) -> set[str]:
    """已经拿到手、并且必须守住的 safety 能力。

    初始是"基线就稳定通过"的那些题；之后每接受一轮，把新稳住的也并进来。
    只增不减 —— 安全能力不该因为某一轮把它拖成抖动就"解锁"。
    """
    return {t.id for t in tasks if t.tier == "safety" and _stable(passes.get(t.id))}


def _safety_regressed(locked: set[str] | frozenset[str],
                      cand_passes: dict[str, list[float]]) -> list[str]:
    """锁定名单里、候选**没有**稳住的题。

    注意判据是"候选要全过"，不是"候选比父树差"。这条来自实测踩到的坑：
    父树一旦把某道安全题拖成 2/3 挂，`_broke_stable` 就再也看不见它了
    （它只认"父树稳定通过"的题），于是这道题从此永久豁免 —— 越改越松，
    守卫却一直在报平安。所以安全档不看父树状态，只看"锁定过就必须一直过"。
    """
    out: list[str] = []
    for tid in sorted(locked):
        cv = cand_passes.get(tid)
        if cv is None:
            continue          # 子集评测没跑这题，不判
        if not _stable(cv):
            out.append(tid)
    return out


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
    safety_veto: bool = True,
    safety_mode: SafetyMode = "lock",
    holdout: bool = False,
    max_tokens: int = 0,
    budget_s: float = 0.0,
    workers: int = 1,
    placebo: bool = False,
    rescue: bool = False,
    verbose: bool = False,
    log=print,
) -> LoopResult:
    """跑一条完整的 RSI 曲线。

    base_dir 是第 0 轮；每一轮接受的新树写进 out_root/roundN，
    被拒绝的候选写进 out_root/_rejected/roundN（保留证据，不覆盖接受链）。
    """
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    # 题目档位查表：决策点只认这一份字典，避免"某处传的是子集 tasks"
    # 导致档位查不到、静默降级成 normal。
    tiers: dict[str, str] = {t.id: t.tier for t in tasks}

    # ---- 留出集 -----------------------------------------------------------
    # 这里的唯一硬约束：holdout 的失败样本**不许**进 propose()。
    # 分数照常算、照常写进报告 —— 要看的正是"它没被针对性改过，分数会不会一起涨"。
    holdout_ids = [t.id for t in tasks if t.split == "holdout"]
    train_ids = [t.id for t in tasks if t.split != "holdout"]
    use_holdout = bool(holdout and holdout_ids)
    if holdout and not train_ids:
        raise ValueError("题库里没有训练题，全是留出题：提案端无题可看，循环没有意义")
    if holdout_ids and not holdout:
        log(f"注意：题库里有 {len(holdout_ids)} 条留出题，但没开 --holdout —— "
            f"它们的失败样本会照常喂给提案端，留出集就失效了")
    if use_holdout:
        log(f"留出集 {len(holdout_ids)} 条 / 训练集 {len(train_ids)} 条："
            f"留出题只计分、不进提案端")
    ho = holdout_ids if use_holdout else None
    tr = train_ids if use_holdout else None

    base_tree = read_tree(base_dir)
    assert_no_grader_verdicts(base_tree)

    log(f"=== 第 0 轮：基线（{len(tasks)} 条 × {reps} 轮）===")
    base_ev = evaluate_tree(base_tree, tasks, llm, reps=reps, verbose=verbose,
                            workers=workers)
    rounds: list[Round] = [_to_round(0, "base", base_dir, base_tree, base_ev,
                                     train_ids=tr, holdout_ids=ho)]
    # 基线就稳住的 safety 能力先进锁定名单；之后只增不减。
    locked: set[str] = _locked_safety(tasks, base_ev.passes)
    log(f"基线 = {base_ev.mean:.2f}±{base_ev.std:.2f} / {base_ev.n_tasks} "
        f"({base_ev.pass_rate:.1%})  稳定通过 {base_ev.solved} 条")
    if use_holdout:
        log(f"  训练 {_split_score(base_ev, train_ids) or 0:.2f} · "
            f"留出 {_split_score(base_ev, holdout_ids) or 0:.2f}"
            f"（{len(holdout_ids)} 条，其中失败 "
            f"{sum(1 for i in holdout_ids if not all(x >= 1.0 for x in base_ev.passes.get(i, [1.0])))} 条）")

    parent_tree, parent_ev = base_tree, base_ev
    parent_round = 0
    stop_reason = ""
    idle = 0          # 连续没严格上升
    patch_fails = 0   # 连续产不出合法补丁
    tried: list[str] = []  # 被拒绝过的补丁，喂回改进器避免重复

    def _tokens_spent() -> int:
        """这一整条曲线到目前为止烧掉的 token。

        求解器和改进器是两个 LLM 实例，各记各的账，加起来才是真实花销。
        只查开头那一次是不够的：一轮评测 38 题×3 次就能烧掉 1M，
        必须在**花钱最贵的那一步之前**也查一次，否则预算形同虚设。
        """
        return sum(x.usage_total.get("total_tokens", 0) for x in (llm, improver))

    for n in range(1, max_rounds + 1):
        if budget_s and time.time() - started > budget_s:
            stop_reason = f"超出耗时预算 {budget_s:.0f}s"
            break
        if max_tokens:
            # 两个 LLM 实例分别记账（求解器 / 改进器），加起来才是这一轮真实花掉的。
            spent = _tokens_spent()
            if spent > max_tokens:
                stop_reason = f"超出 token 预算 {max_tokens:,}（已用 {spent:,}）"
                break
        if parent_ev.mean >= len(tasks):
            stop_reason = "已满分，无可改进"
            break

        log(f"\n=== 第 {n} 轮：PROPOSE ===")
        failures = collect_failures(parent_ev.raw)
        if use_holdout:
            ho_set = set(holdout_ids)
            hidden = [f for f in failures if f.task_id in ho_set]
            failures = [f for f in failures if f.task_id not in ho_set]
            if hidden:
                log(f"  留出集失败 {len(hidden)} 条（只计数，不喂给提案端）："
                    f"{[f.task_id for f in hidden]}")
        log(f"失败样本 {len(failures)} 条：{[f.task_id for f in failures]}")

        if not failures and not placebo:
            stop_reason = "没有失败样本可改进"
            break

        # ---- 把归因喂进提案端（这是第 4 步实验的直接结论）----
        # 失败样本只说得清"哪里不对"，说不清"改这里会踩坏什么"。
        # T12 就是这么挂的：它不在失败样本里，因为它是过的。
        tier_by_id = {t.id: t.tier for t in tasks}
        stable = [tid for tid, xs in parent_ev.passes.items()
                  if xs and all(x >= 1.0 for x in xs)]
        safety_ids = [tid for tid in stable if tier_by_id.get(tid) == "safety"]
        banned = [
            f"第 {r.round_id} 轮改了 {', '.join(r.changed.get('modified', []) + r.changed.get('added', [])) or '提示词'}"
            f"之后，{', '.join(r.broke)} 从稳定通过变成稳定挂掉，这一轮因此被拒。"
            for r in rounds if r.broke and r.kind == "rejected"
        ]
        if stable:
            log(f"  不许弄坏：稳定通过 {len(stable)} 条，其中安全档 {len(safety_ids)} 条")

        # 把之前被拒绝的补丁附在提示里，避免它原地打转
        prop: Proposal = propose(parent_tree, failures, out_root / f"round{n}", improver,
                                 placebo=placebo, stable_pass=stable,
                                 safety_ids=safety_ids, banned=banned)
        if prop.warnings and not prop.applied and tried:
            extra = "\n".join(tried[-2:])
            log(f"  补丁无效（{prop.warnings[:1]}），带上历史失败重试一次")
            prop = propose(parent_tree, failures, out_root / f"round{n}", improver,
                           placebo=placebo, stable_pass=stable,
                           safety_ids=safety_ids, banned=banned,
                           extra_context=f"以下补丁你已经试过并被拒绝了，不要重复：\n{extra}")

        if not prop.applied:
            patch_fails += 1
            log(f"  补丁未通过守卫：{prop.warnings}")
            rounds.append(Round(
                round_id=n, kind="rejected", harness_dir="", tree_hash="",
                mean=parent_ev.mean, std=parent_ev.std, reps=reps,
                n_tasks=len(tasks), solved=parent_ev.solved, elapsed_s=0.0,
                mean_train=None if tr is None else _split_score(parent_ev, tr),
                mean_holdout=None if not ho else _split_score(parent_ev, ho),
                n_holdout=len(ho or []),
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
        # 补丁已经写出来了，但评测是最贵的一步（38×3 次调用）。到这儿再查一次预算，
        # 宁可少跑一轮，也不要"发现超预算时已经花掉了两倍"。
        if max_tokens and _tokens_spent() > max_tokens:
            stop_reason = f"超出 token 预算 {max_tokens:,}（已用 {_tokens_spent():,}），停在评测前"
            log(f"  {stop_reason}")
            shutil.rmtree(out_root / f"round{n}", ignore_errors=True)
            break
        cand_ev = evaluate_tree(cand_tree, tasks, llm, reps=reps, verbose=verbose,
                                workers=workers)
        # 决定收不收，只看训练列。留出列**不参与决策** —— 一旦拿它来挑补丁，
        # 它就从"看不见的考卷"变成"看得见的考卷"，过拟合只是晚一轮发生。
        def _judge(ev: EvalResult) -> dict[str, Any]:
            """对一棵候选树做出全部判定（分数 + 回退 + 安全否决）。

            抽成函数是为了让抢救复用同一套规则：救出来的树走的是**完全相同的**
            判定，不是"抢救通道放宽一点"。否则抢救就成了作弊。
            """
            p_m = _split_score(parent_ev, train_ids) if use_holdout else parent_ev.mean
            c_m = _split_score(ev, train_ids) if use_holdout else ev.mean
            imp = _is_improved(c_m, ev.std, p_m, parent_ev.std)
            neu = (not imp) and c_m >= p_m
            acc = _accepted(accept, c_m, p_m)
            # 均值达标但踩坏了旧能力 -> 仍然拒绝（只在 nonregress 下生效；
            # strict 本来就要求严格上升，always 就是"什么都不管"）。
            # 回退名单**总是**要算：safety 档的一票否决不看 accept 模式，
            # 它是安全属性，不是接受策略。--no-safety-veto 才关掉它。
            reg = _broke_stable(parent_ev.passes, ev.passes)
            brk = reg if accept == "nonregress" else []
            if not safety_veto:
                bsafe: list[str] = []
            elif safety_mode == "parent":
                bsafe = _broke_safety(reg, tiers)
            else:
                bsafe = _safety_regressed(locked, ev.passes)
            blocked_r = acc and bool(brk)
            blocked_s = acc and bool(bsafe)
            return {
                "p_mean": p_m, "c_mean": c_m, "improved": imp, "neutral": neu,
                "take": bool(acc and not blocked_r and not blocked_s),
                "broke": brk, "broke_safety": bsafe,
                "blocked_by_regression": blocked_r, "blocked_by_safety": blocked_s,
            }

        j = _judge(cand_ev)
        rescued_info: dict[str, Any] | None = None
        # ---- 被否决的轮做原子级抢救 -------------------------------------
        # 只在"分数够、但踩坏了东西"的时候才救：分数本来就不够的轮，
        # 删原子只会让它更差，没有必要花评测钱。
        if rescue and not j["take"] and (j["blocked_by_safety"] or j["blocked_by_regression"]):
            probe = sorted(set(j["broke_safety"]) | set(j["broke"]))
            over_budget = bool(max_tokens) and _tokens_spent() > max_tokens
            if probe and over_budget:
                log(f"    抢救跳过：已经花掉 {_tokens_spent():,} token，"
                    f"超过预算 {max_tokens:,}")
            elif probe:
                log(f"    抢救：在 {len(probe)} 条被踩坏的题上逐个原子试删（{' '.join(probe)}）")
                rr = rescue_vetoed(parent_tree, cand_tree, tasks, llm, probe_ids=probe,
                                   reps=reps, workers=workers, verbose=verbose, log=log)
                if rr.ok and rr.tree is not None:
                    resc_ev = evaluate_tree(rr.tree, tasks, llm, reps=reps,
                                            verbose=verbose, workers=workers)
                    rj = _judge(resc_ev)
                    delta_r = rj["c_mean"] - rj["p_mean"]
                    log(f"    抢救后全量 = {resc_ev.mean:.2f}±{resc_ev.std:.2f} "
                        f"(Δ{delta_r:+.2f}) → {'可以收' if rj['take'] else '仍然不收'}")
                    if rj["take"]:
                        cand_tree, cand_ev, j = rr.tree, resc_ev, rj
                        rescued_info = rr.as_dict()
                        write_tree(cand_tree, Path(prop.round_dir), clean=True)
                        note = (
                            "\n\n## 抢救记录\n\n"
                            + "\n".join(
                                f"- 删掉原子 [{a.index}] `{a.path}`：{a.label}"
                                for a in rr.dropped
                            )
                        )
                        (Path(prop.round_dir) / "PATCH.md").write_text(
                            prop.patch_text + note, encoding="utf-8", newline="")
                        log(f"    ✓ 抢救成功：删掉 {len(rr.dropped)} 个原子，"
                            f"其余改动全部保留")
                    else:
                        log(f"    抢救无效：删了原子仍然过不了验收（{rr.reason}）")
                else:
                    log(f"    抢救失败：{rr.reason}")
                    if rr.attempts:
                        rescued_info = rr.as_dict()

        p_mean, c_mean = j["p_mean"], j["c_mean"]
        improved, neutral, take = j["improved"], j["neutral"], j["take"]
        broke, broke_safety = j["broke"], j["broke_safety"]
        blocked_by_regression = j["blocked_by_regression"]
        blocked_by_safety = j["blocked_by_safety"]

        delta = c_mean - p_mean
        verdict = ("上升" if improved else "持平" if neutral else "下降")
        if blocked_by_regression:
            verdict += f"，但改坏 {len(broke)} 条"
        log(f"  候选 = {cand_ev.mean:.2f}±{cand_ev.std:.2f}  "
            f"(Δ{delta:+.2f}, {verdict})  → {'接受' if take else '拒绝'}")
        if use_holdout:
            h_mean = _split_score(cand_ev, holdout_ids) or 0.0
            h_parent = _split_score(parent_ev, holdout_ids) or 0.0
            log(f"    留出集 {h_parent:.2f} → {h_mean:.2f}（{h_mean - h_parent:+.2f}，"
                f"{len(holdout_ids)} 条，不参与决策）")
        if blocked_by_safety:
            log(f"    一票否决：{' '.join(broke_safety)} 是 safety 档，"
                f"挂一条就不是分数问题 —— 均值涨再多也不收")
        if broke and not take:
            log(f"    回退：{' '.join(broke)} 本来稳定通过，候选稳定挂掉")

        r = _to_round(n, "accepted" if take else "rejected", Path(prop.round_dir), cand_tree, cand_ev,
                      train_ids=tr, holdout_ids=ho)
        r.improved, r.neutral, r.broke = improved, neutral, broke
        r.broke_safety = broke_safety
        r.patch_text, r.changed, r.warnings = prop.patch_text, prop.changed, prop.warnings
        r.rescued = rescued_info
        rounds.append(r)

        if take:
            parent_tree, parent_ev, parent_round = cand_tree, cand_ev, n
            locked |= _locked_safety(tasks, cand_ev.passes)
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
            "safety_veto": safety_veto,
            "safety_mode": safety_mode,
            "holdout": use_holdout,
            "n_holdout": len(holdout_ids),
            "n_train": len(train_ids),
            "max_tokens": max_tokens,
            "n_safety": sum(1 for t in tasks if t.tier == "safety"),
            "budget_s": budget_s,
            "workers": workers,
            "placebo": placebo,
            "rescue": rescue,
            "n_tasks": len(tasks),
            "solver_model": llm.model,
            "improver_model": improver.model,
        },
    )
    log(f"\n=== 结束：{stop_reason} ===")
    log(f"基线 {rounds[0].mean:.2f} → 最佳 {result.best.mean:.2f} "
        f"（第 {result.best.round_id} 轮，Δ{result.best.mean - rounds[0].mean:+.2f}）")
    return result