"""把几条曲线并排看：同一条改进器，换接受规则，结局差多少。

## 为什么单独一个模块

`loop.py` 生成的 `curve.json` 是"一条曲线的自述"；这里要做的是"几条曲线的
横向对照"。两者读者不同：前者回答"这一轮发生了什么"，后者回答"换个规则会怎样"。

## 对照能说明什么，不能说明什么

能说明：**接受规则本身是一个变量**。朴素 RSI（`always`）无条件替换，某轮补丁
把题改坏了，下一轮就在坏树上继续改，错误累积；`nonregress` 把父树钉住，
坏的那轮被丢弃。

不能说明：**单条轨迹之间的 1 分差距**。每条曲线只跑一次，模型没有固定 seed，
两次跑基线就差 1 分（实测 32.67 vs 31.67）。所以这张表只用来支持"结构性差异"
（比如 always 累积退化、nonregress 保住基线），不用来排名。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Curve", "load_curve", "render_comparison"]


@dataclass
class Curve:
    name: str
    accept: str
    baseline: float
    baseline_std: float
    baseline_solved: int
    best: float
    best_std: float
    best_solved: int
    best_round: int
    stop_reason: str
    elapsed_s: float
    rounds: list[dict[str, Any]] = field(default_factory=list)
    n_tasks: int = 0
    reps: int = 0

    @property
    def last(self) -> dict[str, Any]:
        """最后一轮（可能是被拒绝的候选，不代表交付的树）。"""
        return self.rounds[-1] if self.rounds else {}

    @property
    def last_accepted(self) -> dict[str, Any]:
        """最后一个**被接受**的轮次。这才是这条规则真正停下来的那棵树。

        全拒的曲线（比如被回退规则拦了一路）这里退回到基线 —— 因为它就是
        最终留下的东西。用 `rounds[-1]` 会把被拒绝的候选当成交付物。
        """
        acc = [r for r in self.rounds if r.get("kind") != "rejected"]
        return acc[-1] if acc else {}

    @property
    def last_accepted_mean(self) -> float:
        return float(self.last_accepted.get("mean", 0.0))

    def lost_stable(self) -> list[str]:
        """基线稳定通过、而**最终留下的树**稳定挂掉的题。

        和 `loop._broke_stable` 口径一致，但这里是对任何接受规则都能回头算的：
        看两棵树的 passes，不依赖循环当时的判定。`always` 档从来不写 `broke`，
        只能这样算。
        """
        if len(self.rounds) < 2:
            return []
        base = self.rounds[0].get("passes") or {}
        final = self.last_accepted.get("passes") or {}
        out = []
        for tid, vals in base.items():
            if not vals or not all(v >= 1 for v in vals):
                continue
            now = final.get(tid)
            if now is not None and not any(v >= 1 for v in now):
                out.append(tid)
        return sorted(out)

    def gained_stable(self) -> list[str]:
        """基线没稳定过、最终稳定过的题 —— 涨的分是从哪来的。"""
        if len(self.rounds) < 2:
            return []
        base = self.rounds[0].get("passes") or {}
        final = self.last_accepted.get("passes") or {}
        return sorted(tid for tid, vals in final.items()
                      if vals and all(v >= 1 for v in vals)
                      and not (base.get(tid) and all(v >= 1 for v in base[tid])))

    @property
    def delta(self) -> float:
        return self.best - self.baseline

    @property
    def accepted(self) -> int:
        return sum(1 for r in self.rounds if r["kind"] == "accepted")

    @property
    def rejected(self) -> int:
        return sum(1 for r in self.rounds if r["kind"] == "rejected")

    @property
    def broke_total(self) -> int:
        """被接受的那些轮里，累计踩坏了多少条原本稳定通过的题。"""
        return sum(len(r.get("broke") or []) for r in self.rounds
                   if r["kind"] == "accepted")

    @property
    def regressed(self) -> int:
        """最好那棵树比基线少了几条稳定通过（负数表示反而更少）。"""
        return self.best_solved - self.baseline_solved


def load_curve(path: Path) -> Curve:
    """读 curve.json，顺手把名字和 --accept 取出来。"""
    path = Path(path)
    if path.is_dir():
        path = path / "curve.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = data.get("config", {})
    base = data.get("baseline", {})
    final = data.get("final", {})
    rounds = list(data.get("rounds", []))
    # 早期曲线没把 n_tasks/reps 写进 baseline，退回到第 0 轮里取
    first = rounds[0] if rounds else {}
    n_tasks = int(base.get("n_tasks") or first.get("n_tasks") or 0)
    reps = int(base.get("reps") or first.get("reps") or 0)
    best_round = int(final.get("round_id", 0))
    best_row = next((r for r in rounds if r.get("round_id") == best_round), first)
    # 同样：早期曲线没写 final.solved / baseline.solved
    base_solved = int(base.get("solved") or first.get("solved") or 0)
    best_solved = int(final.get("solved") or best_row.get("solved") or 0)
    return Curve(
        name=Path(path).parent.name,
        accept=cfg.get("accept", "?"),
        baseline=float(base.get("mean", 0.0)),
        baseline_std=float(base.get("std", 0.0)),
        baseline_solved=base_solved,
        best=float(final.get("mean", 0.0)),
        best_std=float(final.get("std", 0.0)),
        best_solved=best_solved,
        best_round=best_round,
        stop_reason=str(data.get("stop_reason", "")),
        elapsed_s=float(data.get("total_elapsed_s", 0.0)),
        rounds=rounds,
        n_tasks=n_tasks,
        reps=reps,
    )


def _fmt_rules(curves: list[Curve]) -> str:
    mapping = {"strict": "只收严格上升", "nonregress": "不退化（含平持）",
               "always": "无条件替换（朴素 RSI）"}
    return "、".join(f"`{c.accept}`={mapping.get(c.accept, '?')}" for c in curves)


def render_comparison(curves: list[Curve]) -> str:
    if not curves:
        return "# 消融对照\n\n没有找到曲线。\n"
    out: list[str] = []
    n = curves[0].n_tasks or 0
    reps = curves[0].reps or 0

    out.append("# 消融对照：接受规则是不是一个变量\n")
    out.append(f"> {n} 条任务 × {reps} 次重复 · {_fmt_rules(curves)}\n")

    out.append("## 1. 结局\n")
    out.append("| 接受规则 | 基线 | 历史最佳 | 最后接受的那棵 | Δ | 稳定通过 | 丢失稳定题 | 收/拒 |")
    out.append("|---|---:|---:|---:|---:|---:|---|---:|")
    for c in curves:
        lost = "、".join(c.lost_stable()) or "—"
        out.append(
            f"| `{c.accept}` | {c.baseline:.2f}±{c.baseline_std:.2f} | "
            f"{c.best:.2f} | {c.last_accepted_mean:.2f} | "
            f"**{c.last_accepted_mean - c.baseline:+.2f}** | "
            f"{c.baseline_solved}→{c.last_accepted.get('solved', 0)} | {lost} | "
            f"{c.accepted}/{c.rejected} |"
        )
    out.append("")
    out.append("> \"历史最佳\"是曲线上分数最高的那一轮（`best/` 导出的就是它）；"
               "\"最后接受的那棵\"是这条规则**停下来时真正留下的树**。"
               "两者在 `always` 档会分开：它不挑，分数掉了也照样替换。"
               "全拒的曲线（`nonregress` 被回退规则拦了一路）最后留下的就是基线，"
               "被拦下的候选只出现在下面的逐轮轨迹里。\n")

    out.append("## 2. 逐轮轨迹\n")
    for c in curves:
        out.append(f"### `{c.accept}`\n")
        out.append("| 轮 | 判定 | 得分 | 稳定通过 | 踩坏 |")
        out.append("|---:|---|---:|---:|---|")
        for r in c.rounds:
            mark = {"base": "基线", "accepted": "接受", "rejected": "拒绝"}.get(
                r["kind"], r["kind"])
            if r["kind"] == "accepted" and r.get("improved"):
                mark = "接受（上升）"
            elif r["kind"] == "accepted" and r.get("neutral"):
                mark = "接受（持平）"
            elif r["kind"] == "rejected" and r.get("broke"):
                mark = "拒绝（回退）"
            out.append(
                f"| {r['round_id']} | {mark} | {r['mean']:.2f}±{r['std']:.2f} | "
                f"{r.get('solved', 0)}/{n} | {'、'.join(r.get('broke') or []) or '—'} |"
            )
        out.append("")

    out.append("## 3. 怎么读\n")
    by_last = max(curves, key=lambda c: c.last_accepted_mean)
    out.append(f"- 按**最后留下的那棵树**排，最好的是 `{by_last.accept}`"
               f"（{by_last.last_accepted_mean - by_last.baseline:+.2f}）。")
    lost_any = [c for c in curves if c.lost_stable()]
    if lost_any:
        for c in lost_any:
            out.append(f"- `{c.accept}` 最后留下的树丢了基线时稳定通过的 "
                       f"{'、'.join(c.lost_stable())}。这些题不会自己回来，"
                       "后面每一轮都带着它们跑。")
    else:
        out.append("- 三条曲线最后留下的树都没丢基线时稳定通过的题 —— 这一批补丁是"
                   "纯增益，所以\"回退规则拦下的到底是不是真回退\"这个争论没有出现。")
    for c in curves:
        if c.accept == "always" and len(c.rounds) > 1 and abs(c.best - c.last_accepted_mean) > 1e-9:
            out.append(f"- `always` 的最高分 {c.best:.2f} 出现在第 {c.best_round} 轮，"
                       f"但它停下来时留下的是 {c.last_accepted_mean:.2f} —— 中间掉分的那几轮"
                       "没人拦，这正是\"接受规则\"存在的理由。")
        if c.accept != "always" and c.rejected and c.best - c.last_accepted_mean < 1e-9:
            best_rej = max((r for r in c.rounds if r.get("kind") == "rejected"),
                           key=lambda r: r["mean"], default=None)
            if best_rej and best_rej["mean"] > c.baseline + 1:
                out.append(f"- `{c.accept}` 拒绝了第 {best_rej['round_id']} 轮的 "
                           f"{best_rej['mean']:.2f}（比基线高 "
                           f"{best_rej['mean'] - c.baseline:+.2f}），"
                           f"代价是丢了 {'、'.join(best_rej.get('broke') or []) or '若干题'}。"
                           "**这是这套规则最该被质疑的地方**：它宁可不要 +4 分也不接受"
                           "一条回退，而这个取舍是否划算，取决于丢的那条题有多重要。")
    out.append("- **每档只跑了一次**，模型没有固定 seed，两次跑同一档基线就能差 1 分。"
               "所以这张表只支持结构性结论（比如\"无条件替换会累积退化\"），"
               "**不支持**\"哪一档高 0.5 分所以更好\"这种排名。")
    out.append("- 要排名得跑多条轨迹取均值，那是下一层的成本。现在这张表的作用是"
               "证明\"接受规则是个需要论证的选择\"，而不是拍脑袋定一个。")
    out.append("")
    out.append("---")
    out.append("")
    out.append("*由 `harnessdrift compare` 生成。原始数据见各曲线目录的 `curve.json`。*")
    return "\n".join(out) + "\n"