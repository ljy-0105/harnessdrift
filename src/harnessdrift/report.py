"""第 3 步（续）：把 LoopResult 写成一份人能读的报告。

报告不是日志的堆砌。企业里看这份东西的人只想知道四件事，报告必须让他
在**一分钟内**拿到：

1. 曲线长什么样（起点、终点、中间有没有回退）
2. 每一轮具体改了什么（不是"优化了提示词"，而是改动了哪几行）
3. 失败分布怎么移动的（真缺陷变少了，还是只是把它藏起来了）
4. 什么时候停的、为什么停（patience 用尽 ≠ 跑够了轮数）

第 3 点是最容易被水掉的地方。一份只报总分的报告，读者无法判断涨分是
"能力提升"还是"把难题绕过去了"。所以这里同时输出：
- `stable_fail`：每轮都挂的题（真缺陷）
- `flaky`：时过时不过的题（噪声，涨它不是涨分）
- 失败项计数表：失败原因从哪一类迁移到哪一类

**图表用纯 ASCII。** 不是审美问题：这份报告要能在没有渲染的终端、
在 PR 评论、在邮件正文里都保持可读。依赖图片的报告在秋招场景下
有一半时间打不开。
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .loop import LoopResult, Round

BAR_WIDTH = 44


def sparkline(values: Sequence[float], lo: float, hi: float) -> str:
    """把一串分数画成横向条。lo==hi 时退化成满/空。"""
    if not values:
        return ""
    span = hi - lo
    out = []
    for v in values:
        frac = 0.0 if span <= 1e-9 else (v - lo) / span
        n = max(0, min(BAR_WIDTH, round(frac * BAR_WIDTH)))
        out.append(n)
    return out


def curve_block(res: LoopResult) -> str:
    """曲线表。同时给出 mean±std，因为只报均值会掩盖抖动。"""
    rounds = res.rounds
    base = rounds[0]
    n_tasks = base.n_tasks
    # 纵轴：基线往下留一点余量，顶部到满分
    lo = min(r.mean for r in rounds) - max(1.0, n_tasks * 0.05)
    hi = float(n_tasks)
    lo = max(0.0, lo)
    span = max(hi - lo, 1e-9)

    has_ho = any(r.mean_holdout is not None for r in rounds)
    # 没开留出集就不出这一列：一列全是「—」只会让读者以为哪里算丢了
    head = (f"{'round':>6}  {'kind':<9} {'score':>7} {'±':>5}  {'solved':>6}  "
            + ("holdout  " if has_ho else "") + "bar")
    lines = [
        "```",
        head,
        "-" * (6 + 2 + 9 + 1 + 7 + 1 + 5 + 2 + 6 + 2 + (9 if has_ho else 0) + BAR_WIDTH),
    ]
    for r in rounds:
        frac = (r.mean - lo) / span
        n = max(0, min(BAR_WIDTH, round(frac * BAR_WIDTH)))
        bar = "█" * n + "·" * (BAR_WIDTH - n)
        mark = " " if r.kind == "base" else ("+" if r.improved else "~" if r.kind == "accepted" else "x")
        ho = f"{r.mean_holdout:>7.2f}  " if (has_ho and r.mean_holdout is not None) else (
            f"{'—':>7}  " if has_ho else "")
        lines.append(
            f"{r.round_id:>6}  {r.kind:<9} {r.mean:>7.2f} {r.std:>5.2f}  "
            f"{r.solved:>4}/{n_tasks:<2} {ho}{mark} {bar}"
        )
    lines.append("```")
    lines.append("")
    if has_ho:
        m = n_tasks - (base.n_holdout or 0)
        lines.append(f"满分 {n_tasks}。`+` 表示相对父树有超过噪声的上升，`~` 持平接受，`x` 被拒绝。"
                     f"**接受决策只看训练列 {m} 条**；留出列 {base.n_holdout} 条照常评测计分，"
                     f"但不参与任何决定。")
        lines.append("`holdout` 列是留出集得分（口径与 score 一致，满分 = 条数）：它**不参与任何接受决策**，"
                     "只看它跟不跟着涨 —— 涨了叫泛化，不涨说明训练列的涨幅是背题换来的。")
    else:
        lines.append(f"满分 {n_tasks}。`+` 表示相对父树有超过噪声的上升，`~` 持平接受，`x` 被拒绝。")
    return "\n".join(lines)


def delta_block(res: LoopResult) -> str:
    base, best = res.rounds[0], res.best
    d = best.mean - base.mean
    pct = (best.mean / base.mean - 1) * 100 if base.mean else 0.0
    lines = [
        "| 指标 | 基线 | 最佳 | 变化 |",
        "|---|---|---|---|",
        f"| 得分（mean） | {base.mean:.2f} | {best.mean:.2f} | **{d:+.2f}** |",
        f"| 通过率 | {base.pass_rate:.1%} | {best.pass_rate:.1%} | {pct:+.1f}% |",
        f"| 稳定通过（每轮都过） | {base.solved} | {best.solved} | {best.solved - base.solved:+d} |",
        f"| 抖动（std） | {base.std:.2f} | {best.std:.2f} | {best.std - base.std:+.2f} |",
        f"| 最佳轮次 | — | 第 {best.round_id} 轮 | — |",
    ]
    if base.mean_holdout is not None:
        # 放在最后一行，并显式标注"不参与决策"：否则读者会以为它是第六个可以权衡的指标。
        lines.append(
            f"| 留出集（不参与决策） | {base.mean_holdout:.2f} | {best.mean_holdout or 0:.2f} | "
            f"**{(best.mean_holdout or 0) - base.mean_holdout:+.2f}** |"
        )
    return "\n".join(lines)


def rounds_block(res: LoopResult) -> str:
    """每轮改了什么。被拒绝的也要写出来——拒绝记录本身是实验证据。"""
    out: list[str] = []
    for r in res.rounds:
        if r.kind == "base":
            continue
        head = f"### 第 {r.round_id} 轮 · {r.kind}"
        if r.kind == "rejected" and not r.tree_hash:
            out.append(f"{head}\n\n补丁未通过守卫，未进入评测。\n")
            for w in r.warnings:
                out.append(f"- {w}")
            out.append("")
            continue

        verdict = "上升" if r.improved else ("持平" if r.neutral else "下降")
        out.append(f"{head}（{r.mean:.2f}±{r.std:.2f}，{verdict}）\n")
        if r.broke_safety:
            # 单独写一节：这不是"扣分"，是出事故，读者需要知道它和普通回退
            # 不是一个量级的东西。
            out.append(f"- **安全档一票否决**：{'、'.join(r.broke_safety)} 是 safety 题，"
                       f"本轮没有做到每次都通过（锁定名单要求 {r.reps} 次全过）。"
                       f"这类错误是给别人的订单退款、凭空编订单号，"
                       f"不能拿净分来权衡，因此不看均值直接拒绝。")
        if r.broke:
            out.append(f"- **被回退规则拦下**：{'、'.join(r.broke)} 在父树是稳定通过、"
                       f"在本轮稳定挂掉。均值涨了也不收 —— 拿旧能力换新能力不算改进。")
        if r.changed:
            for k, files in r.changed.items():
                # 空分类不写出来：报告里出现"- **added**："后面什么都没有，
                # 读者会以为渲染坏了，而不是"这一类没变"。
                if not files:
                    continue
                out.append(f"- **{k}**：{', '.join(f'`{f}`' for f in files)}")
        else:
            out.append("- （补丁未产生文件级改动）")
        if r.warnings:
            out.append(f"- 守卫告警：{'；'.join(r.warnings)}")
        if r.patch_text:
            snippet = r.patch_text.strip()
            if len(snippet) > 1200:
                snippet = snippet[:1200] + "\n…（补丁过长已截断，完整内容见 PATCH.md）"
            out.append("\n<details><summary>补丁全文</summary>\n")
            out.append("```diff")
            out.append(snippet)
            out.append("```")
            out.append("\n</details>")
        out.append("")
    return "\n".join(out)


def failure_block(res: LoopResult) -> str:
    """失败分布的变化。这是"涨分是否可信"的证据。"""
    base, best = res.rounds[0], res.best
    keys = sorted(
        set(base.failure_hist) | set(best.failure_hist),
        key=lambda k: -(base.failure_hist.get(k, 0) + best.failure_hist.get(k, 0)),
    )
    if not keys:
        return "（没有失败项）"

    lines = [
        "| 失败项 | 基线次数 | 最佳次数 | 变化 |",
        "|---|---:|---:|---:|",
    ]
    for k in keys:
        b = base.failure_hist.get(k, 0)
        f = best.failure_hist.get(k, 0)
        lines.append(f"| `{k}` | {b} | {f} | {f - b:+d} |")
    return "\n".join(lines)


def stability_block(res: LoopResult) -> str:
    """抖动题 vs 稳定失败题。

    这两类必须分开报：稳定失败是能力问题，抖动是噪声。
    只报总分的报告会把"3 条题从稳定失败变成随机失败"也写成涨分。
    """
    base, best = res.rounds[0], res.best
    lines: list[str] = []

    fixed = sorted(set(base.stable_fail) - set(best.stable_fail))
    broke = sorted(set(best.stable_fail) - set(base.stable_fail))
    still = sorted(set(base.stable_fail) & set(best.stable_fail))

    lines.append(f"- **修好（不再是稳定失败）**：{', '.join(fixed) if fixed else '无'}")
    lines.append(f"- **改坏（变成稳定失败）**：{', '.join(broke) if broke else '无'}")
    lines.append(f"- **始终没修**：{', '.join(still) if still else '无'}")

    if best.flaky:
        items = ", ".join(f"{k}({[int(x) for x in v]})" for k, v in sorted(best.flaky.items()))
        lines.append(f"- **抖动项（最佳轮）**：{items}")
        lines.append("")
        lines.append(
            "> 抖动项不算能力。如果某个指标的上升主要发生在抖动项上，"
            "这个上升应当被当作噪声，不能写进结论。"
        )
    else:
        lines.append("- **抖动项**：无（所有题的重复轮次结果一致，评测是确定的）")

    if best.errors:
        lines.append("")
        lines.append("**评测期异常（已按 0 分计）**：")
        for k, v in best.errors.items():
            lines.append(f"- `{k}`：{v}")
    return "\n".join(lines)


def stopping_block(res: LoopResult) -> str:
    cfg = res.config
    return "\n".join([
        f"- **停止原因**：{res.stop_reason}",
        f"- **轮数上限** `max_rounds={cfg.get('max_rounds')}`",
        f"- **patience** `{cfg.get('patience')}`（连续无实质改进即停）",
        f"- **补丁失败容忍** `{cfg.get('proposal_patience')}`（连续产不出合法补丁即停）",
        f"- **接受规则** `{cfg.get('accept')}`",
        f"- **重复次数** `reps={cfg.get('reps')}`（每轮每题跑几次，用来估抖动）",
        f"- **总耗时** {res.total_elapsed_s:.0f}s",
        (f"- **总 token** {tok:,}（prompt {res.total_tokens.get('prompt_tokens', 0):,}"
         f" / completion {res.total_tokens.get('completion_tokens', 0):,}）"
         if (tok := res.total_tokens.get("total_tokens", 0)) else
         "- **总 token** 未统计（该曲线跑的时候还没接用量采集）"),
        "",
        "> 停下来的原因决定了这条曲线能不能用："
        "`patience` 用尽说明改进器已经提不出有效补丁，此时曲线是**收敛的**；"
        "`max_rounds` 用尽说明预算不够，曲线还没收敛，不能声称已到上限。",
    ])


def holdout_block(res: LoopResult) -> str:
    """留出集单独一节。

    单独写而不是混在曲线表里，是因为它的**读法**和别的列不一样：
    训练列涨了可能是背题，留出列涨了才是泛化。两列一起看看不出区别，
    必须有一句话把"怎么读"写清楚，否则读者会默认它们是同一件事。
    """
    base, best = res.rounds[0], res.best
    if base.mean_holdout is None:
        return ""
    n = base.n_holdout
    db = (base.mean_holdout or 0.0)
    ds = (best.mean_holdout or 0.0)
    return "\n".join([
        f"留出集 {n} 条，**从头到尾没进过提案端的失败样本**（每轮照常评测、照常计分，"
        f"但提案端只看训练列的错法）。",
        "",
        "|  | 基线 | 最佳 | 变化 |",
        "|---|---|---|---|",
        f"| 训练列（{base.n_tasks - n} 条，参与决策） | {base.mean_train or 0:.2f} | "
        f"{best.mean_train or 0:.2f} | **{(best.mean_train or 0) - (base.mean_train or 0):+.2f}** |",
        f"| 留出列（{n} 条，不参与决策） | {db:.2f} | {ds:.2f} | "
        f"**{ds - db:+.2f}** |",
        "",
        "怎么读：**留出列不涨而训练列涨，说明涨的是背题不是能力**。"
        "留出列跟着涨，才能说这次改动真的泛化了；两列一起跌，那就是通用回退。",
    ])


def cost_block(res: LoopResult) -> str:
    """token 账。

    只报 token，不把 token 折算成钱：汇率（单价）是会变的外部事实，
    写死在报告里就成了一个明天就假的数字。报告给的是可核算的量
    （每轮 token、总 token），换算交给读者 —— 或者 `--max-tokens` 直接卡死上限。
    """
    rows = [r for r in res.rounds if (r.tokens or {}).get("total_tokens")]
    if not rows:
        return "（这次跑批没有 token 记录）"
    total = sum(int(r.tokens.get("total_tokens", 0)) for r in rows)
    lines = ["| 轮次 | prompt | completion | 合计 |", "|---|---|---|---|"]
    for r in rows:
        t = r.tokens or {}
        lines.append(f"| {r.round_id} {r.kind} | {int(t.get('prompt_tokens', 0)):,} | "
                     f"{int(t.get('completion_tokens', 0)):,} | {int(t.get('total_tokens', 0)):,} |")
    lines.append("| **合计** | | | **{:,}** |".format(total))
    lines += ["", "token 就是钱，所以它也是这套系统的成本上界："
                  "`--max-tokens N` 一旦触到就停，不会跑完才发现超支。"]
    return "\n".join(lines)


def render(res: LoopResult, *, title: str = "HarnessDrift 自改进曲线") -> str:
    cfg = res.config
    base, best = res.rounds[0], res.best
    parts = [
        f"# {title}",
        "",
        f"> 求解模型 `{cfg.get('solver_model')}` · 改进模型 `{cfg.get('improver_model')}` · "
        f"{cfg.get('n_tasks')} 条任务 × {cfg.get('reps')} 次重复",
        "",
        "## 1. 结论",
        "",
        delta_block(res),
        "",
        f"**{base.mean:.2f} → {best.mean:.2f}（{best.mean - base.mean:+.2f} 分，"
        f"相当于 {base.solved} → {best.solved} 条稳定通过）。"
        f"共 {len([r for r in res.rounds if r.kind != 'base'])} 轮尝试，"
        f"{len(res.accepted) - 1} 轮被接受。**",
        *([f"（上面这个总分 = 训练列 {base.mean_train:.2f}→{best.mean_train:.2f}，"
            f"留出列 {base.mean_holdout:.2f}→{best.mean_holdout:.2f}。"
            f"**接受决策只用了训练列**，留出列是从头到尾没被优化过的那把尺子。）"]
           if base.mean_holdout is not None else []),
        "",
        "## 2. 曲线",
        "",
        curve_block(res),
        "",
        "## 3. 每轮改动",
        "",
        rounds_block(res) or "（没有轮次记录）",
        "",
        "## 4. 失败分布迁移",
        "",
        failure_block(res),
        "",
        "## 5. 稳定性分解",
        "",
        stability_block(res),
        "",
        "## 6. 停止条件",
        "",
        stopping_block(res),
        "",
        *(["## 7. 留出集", "", holdout_block(res), ""] if base.mean_holdout is not None else []),
        "## 8. 成本",
        "",
        cost_block(res),
        "",
        "---",
        "",
        "*本报告由 `harnessdrift loop` 自动生成。逐轮数据见同目录 `curve.json`，"
        "最佳 harness 见 `best/`。*",
    ]
    return "\n".join(parts)


def write_report(res: LoopResult, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(res), encoding="utf-8")
    return out_path