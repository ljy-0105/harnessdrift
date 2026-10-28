"""命令行入口。

    # 单次基线（省 token 先用 3 条验证链路）
    ./.venv/Scripts/python.exe -m harnessdrift.cli run --harness harness/base --limit 3

    # 基线要报 ±，单轮数字不可信
    ./.venv/Scripts/python.exe -m harnessdrift.cli run --repeat 3 --out runs/base.json

    # 一条完整的 RSI 曲线
    ./.venv/Scripts/python.exe -m harnessdrift.cli loop --reps 3 --max-rounds 5 --patience 2

评测逻辑统一在 `runner.evaluate_tree` 里。`run` 和 `loop` 必须走同一份实现——
第 2 步就踩过这个坑：CLI 里的判分和 propose 拿到的证据不是同一批对象，
导致工具调用记录对不上，补丁改的是错的东西。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import load_harness_prompt
from .guards import assert_no_grader_verdicts
from .llm import LLM
from .paths import read_tree, tree_hash, write_tree
from .report import write_report
from .runner import evaluate_tree, format_progress
from .tasks import load_tasks


def _select_tasks(args: argparse.Namespace):
    tasks = load_tasks(args.tasks)
    if args.only:
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if t.id in wanted]
    return tasks[: args.limit] if args.limit else tasks


def cmd_run(args: argparse.Namespace) -> int:
    tasks = _select_tasks(args)
    tree = read_tree(args.harness)
    # 跑之前先确认树是干净的：提示词里不能有"评分/评测"这类元描述，
    # 否则模型会开始表演给考官看，而不是干活。
    assert_no_grader_verdicts(tree)
    system = load_harness_prompt(tree)
    th = tree_hash(tree)
    print(f"harness={args.harness} hash={th} tasks={len(tasks)} prompt_chars={len(system)}")
    if args.repeat > 1:
        print(f"repeat={args.repeat} —— 单次跑出来的基数是不可信的："
              f"同一棵树同一套题，模型偶尔会把该转人工的转成不转。")

    llm = LLM()
    print(f"model={llm.model} base_url={llm.base_url}")

    ev = evaluate_tree(
        tree, tasks, llm, reps=args.repeat, verbose=args.verbose,
        workers=getattr(args, "workers", 1),
        on_task=None if not args.verbose else (
            lambda tid, rep, total, v: print(f"[{tid}] r{rep + 1} {'PASS' if v >= 1 else 'FAIL'}")
        ),
    )

    for task, _res, score, _tk in ev.raw:
        print(f"\n[{task.id}] {task.message[:40]}…")
        for c in score.checks:
            print("  " + str(c))

    print("\n" + "=" * 60)
    print(f"pass_rate = {ev.mean:.1f}±{ev.std:.2f} / {ev.n_tasks}"
          f"   ({ev.pass_rate:.3f}   {args.repeat} 轮   共 {ev.elapsed_s:.1f}s)")

    if ev.flaky:
        print("不稳定任务（各轮结果不一致，是噪声不是能力）：")
        for tid, vals in ev.flaky.items():
            print(f"   {tid}: {'/'.join(str(int(v)) for v in vals)}")
    if ev.failure_hist:
        print("失败项分布（全部轮次累计）：")
        for k, v in list(sorted(ev.failure_hist.items(), key=lambda kv: -kv[1]))[:10]:
            print(f"  {v:>2}×  {k}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "when": datetime.now(timezone.utc).isoformat(),
            "harness": str(args.harness),
            "harness_hash": th,
            "model": llm.model,
            **ev.as_dict(),
            "tasks": [
                {"task_id": t.id,
                 "passes": ev.passes[t.id],
                 "mean": round(sum(ev.passes[t.id]) / len(ev.passes[t.id]), 4),
                 "last_failed": ev.last_failed.get(t.id, [])}
                for t in tasks
            ],
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {out}")
    return 0


def cmd_loop(args: argparse.Namespace) -> int:
    from .loop import run_loop

    tasks = _select_tasks(args)
    if not tasks:
        print("没有任务可跑", file=sys.stderr)
        return 2

    llm = LLM(model=args.model) if args.model else LLM()
    # 改进器和求解器默认同一个模型，但**输出预算必须分开**：
    # 求解器只回一段客服话，1200 够；改进器要吐整篇文件，1200 必然截断
    # （实测第一轮就撞上限，两轮作废）。输出 token 只为实际生成的部分付费，
    # 给大预算不会变贵，只会让"想改写整篇"这件事变得可能。
    improver = LLM(
        model=args.improver_model or (args.model or None),
        max_tokens=args.improver_max_tokens,
    )

    out_root = Path(args.out)
    res = run_loop(
        Path(args.harness), tasks, out_root,
        llm=llm, improver=improver,
        reps=args.reps,
        max_rounds=args.max_rounds,
        patience=args.patience,
        proposal_patience=args.proposal_patience,
        accept=args.accept,
        safety_veto=args.safety_veto,
        safety_mode=args.safety_mode,
        holdout=args.holdout,
        placebo=args.placebo,
        max_tokens=args.max_tokens,
        budget_s=args.budget_min * 60 if args.budget_min else 0.0,
        workers=args.workers,
        verbose=args.verbose,
    )

    # 最佳树单独拷一份，报告里的路径要能直接打开
    best_dir = out_root / "best"
    if best_dir.exists():
        import shutil
        shutil.rmtree(best_dir)
    write_tree(read_tree(Path(res.best.harness_dir), ignore=("PATCH.md",)), best_dir)

    (out_root / "curve.json").write_text(
        json.dumps(res.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    rpt = write_report(res, out_root / "REPORT.md")

    print(f"\n报告 {rpt}")
    print(f"曲线 {out_root / 'curve.json'}")
    print(f"最佳 harness {best_dir}")
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    from .rollback import apply_rollback, plan_rollback

    plan = plan_rollback(Path(args.run), args.to, Path(args.target))
    print(plan.describe())
    if plan.empty:
        print("目标树已经就是那个状态，不用回滚")
        return 0
    if args.dry_run:
        print("（--dry-run，未写入任何文件）")
        return 0
    backup = apply_rollback(plan)
    print(f"\n已回滚。原树备份在 {backup}" if backup else "\n已回滚（目标原本不存在，无备份）")
    print("下一步建议：`harnessdrift run --harness " + args.target + "` 复核一遍均值")
    return 0


def cmd_show_prompt(args: argparse.Namespace) -> int:
    tree = read_tree(args.harness)
    print(load_harness_prompt(tree))
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    """归因：把一轮补丁拆成单点，逐条单独评测。"""
    from .ablate import ablate, dump, write_attribution

    tasks = _select_tasks(args)
    if not tasks:
        print("没有任务可跑", file=sys.stderr)
        return 2

    parent = read_tree(Path(args.parent))
    cand = read_tree(Path(args.cand), ignore=("PATCH.md",))
    if parent == cand:
        print("两棵树完全一样，没有可归因的改动", file=sys.stderr)
        return 2

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=args.model) if args.model else LLM()

    res = ablate(parent, cand, tasks, llm, reps=args.reps, workers=args.workers,
                 log=lambda s: print(s, flush=True), verbose=args.verbose)
    res.parent_dir = str(Path(args.parent))
    res.cand_dir = str(Path(args.cand))

    # 每个单点合出来的树都留一份：报告里点名了哪一条，就得能打开看是哪一条
    for a in res.atoms:
        if a.verdict.startswith("跳过"):
            continue
        write_tree(a.atom.apply(parent, cand), out_root / f"atom{a.atom.index}")

    dump(res, out_root / "attribution.json")
    rpt = write_attribution(res, out_root / "ATTRIBUTION.md")
    print(f"\n归因报告 {rpt}")
    print(f"原始数据 {out_root / 'attribution.json'}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """把几条曲线并排看，产出消融对照报告。"""
    from .compare import load_curve, render_comparison

    dirs = [d for d in args.curves.split(",") if d.strip()]
    curves = [load_curve(Path(d.strip())) for d in dirs]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_comparison(curves), encoding="utf-8")
    print(f"对照报告 {out}")
    for c in curves:
        print(f"  {c.accept:10} 基线 {c.baseline:.2f} → 最佳 {c.best:.2f} "
              f"({c.delta:+.2f})  收 {c.accepted} 拒 {c.rejected}  踩坏 {c.broke_total}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """把曲线/补丁/归因/对照打成一个自包含的 HTML 看板。"""
    from .export import export

    compare_dirs = [Path(d.strip()) for d in args.compare.split(",") if d.strip()]
    for d in compare_dirs:
        if not (d / "curve.json").exists():
            print(f"对照曲线不存在：{d}", file=sys.stderr)
            return 2
    attr = Path(args.attr) if args.attr else None
    if attr is not None and not (attr / "attribution.json").exists():
        print(f"归因目录缺少 attribution.json：{attr}", file=sys.stderr)
        return 2

    payload = export(
        Path(args.run),
        Path(args.out),
        tasks_path=Path(args.tasks),
        attr_dir=attr,
        compare_dirs=compare_dirs,
    )
    rounds = [r for r in payload["rounds"] if r["kind"] != "base"]
    attr = payload.get("attribution") or {}
    if attr and attr.get("_match") is False:
        print(f"警告：归因目录 {attr.get('_dir')} 的父树分数不在 {args.run} 的轮次里，"
              f"看板上的归因数字和这条曲线对不上 —— 请确认 --attr 是否指错了目录。",
              file=sys.stderr)
    print(f"看板 {args.out}  （{len(payload['tasks'])} 题，{len(rounds)} 轮，"
          f"{len(payload['curves'])} 条对照，归因={'有' if payload['attribution'] else '无'}）")
    print(f"  双击打开：file:///{Path(args.out).resolve().as_posix()}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    src = read_tree(args.harness)
    dest = Path(args.dest)
    write_tree(src, dest)
    print(f"已复制 {len(src)} 个文件到 {dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="harnessdrift")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--harness", default="harness/base")
        sp.add_argument("--tasks", default="data/tasks_ecommerce_after_sales.json")
        sp.add_argument("--limit", type=int, default=0)
        sp.add_argument("--only", default="", help="逗号分隔的 task id")
        sp.add_argument("--verbose", action="store_true")

    r = sub.add_parser("run", help="跑一遍基线并打分")
    _common(r)
    r.add_argument("--out", default="")
    r.add_argument("--repeat", type=int, default=1,
                   help="同一棵树跑几轮。基线要报 ±，单轮数字不可信")
    r.add_argument("--workers", type=int, default=1, help="评测并发数")
    r.set_defaults(func=cmd_run)

    lp = sub.add_parser("loop", help="跑完整的自改进循环，产出曲线和报告")
    _common(lp)
    lp.add_argument("--out", default="runs/loop")
    lp.add_argument("--reps", type=int, default=3, help="每轮每题重复几次（估抖动）")
    lp.add_argument("--workers", type=int, default=1,
                    help="评测并发数（一条题自己就是若干次模型调用，串行纯等）")
    lp.add_argument("--max-rounds", type=int, default=5)
    lp.add_argument("--patience", type=int, default=2, help="连续几轮无实质改进就停")
    lp.add_argument("--proposal-patience", type=int, default=2,
                    help="连续几轮产不出合法补丁就停")
    lp.add_argument("--accept", default="nonregress",
                    choices=["nonregress", "strict", "always"],
                    help="接受规则；always 用来复现朴素 RSI 做对照")
    lp.add_argument("--budget-min", type=float, default=0.0, help="耗时预算（分钟）")
    lp.add_argument("--max-tokens", type=int, default=0,
                    help="token 预算（求解+改进累计），超出即停；0 表示不限。"
                         "每轮开始与评测前各查一次 —— 评测是最贵的一步")
    lp.add_argument("--placebo", action="store_true",
                    help="安慰剂对照组：失败样本一律不进提案端，只要求补一条"
                         "与业务无关的通用说明。配合 --accept always 用，"
                         "看的就是「没有改进信号时曲线会不会自己往上走」")
    lp.add_argument("--holdout", action="store_true",
                    help="启用留出集：split=holdout 的题照常评测计分，"
                         "但失败样本不喂给提案端，也不参与接受决策")
    # 默认开着。留开关不是为了给"关掉"找理由，是为了能复现加档位之前
    # 那几条曲线：runs/ablate_* 的旧数字都是无否决时跑出来的。
    lp.add_argument("--no-safety-veto", dest="safety_veto", action="store_false",
                    default=True,
                    help="关掉 safety 题一票否决（复现加档位之前的曲线用）")
    lp.add_argument("--safety-mode", default="lock", choices=["lock", "parent"],
                    help="安全档判据：lock=锁定过就必须一直全过（默认）；"
                         "parent=只认父树稳定通过（第一版，会漏）")
    lp.add_argument("--model", default="", help="求解模型（覆盖环境变量）")
    lp.add_argument("--improver-model", default="", help="改进模型，默认同求解模型")
    lp.add_argument("--improver-max-tokens", type=int, default=8000,
                    help="改进器的输出上限；要能装下整篇文件，默认 8000")
    lp.set_defaults(func=cmd_loop)

    ap = sub.add_parser("ablate", help="归因：把一轮补丁拆成单点，逐条单独评测")
    _common(ap)
    ap.add_argument("--parent", default="harness/base", help="父树")
    ap.add_argument("--cand", required=True, help="候选树（通常是 runs/*/roundN）")
    ap.add_argument("--out", default="runs/attr")
    ap.add_argument("--reps", type=int, default=3, help="每个单点跑几次")
    ap.add_argument("--workers", type=int, default=1, help="评测并发数")
    ap.add_argument("--model", default="", help="求解模型（覆盖环境变量）")
    ap.set_defaults(func=cmd_ablate)

    cp = sub.add_parser("compare", help="并排对照多条曲线（消融）")
    cp.add_argument("--curves", required=True, help="逗号分隔的曲线目录")
    cp.add_argument("--out", default="runs/ABLATION.md")
    cp.set_defaults(func=cmd_compare)

    ep = sub.add_parser("export", help="把一次跑批打成一个静态看板（单文件 HTML）")
    ep.add_argument("--run", default="runs/curve3", help="曲线目录（含 curve.json）")
    ep.add_argument("--tasks", default="data/tasks_ecommerce_after_sales.json")
    ep.add_argument("--attr", default="", help="归因目录（含 attribution.json），可空")
    ep.add_argument("--compare", default="", help="逗号分隔的对照曲线目录，可空")
    ep.add_argument("--out", default="dist/index.html")
    ep.set_defaults(func=cmd_export)

    s = sub.add_parser("show-prompt", help="打印某棵 harness 渲染出的系统提示")
    s.add_argument("--harness", default="harness/base")
    s.set_defaults(func=cmd_show_prompt)

    i = sub.add_parser("init", help="复制一棵 harness 树")
    i.add_argument("--harness", default="harness/base")
    i.add_argument("--dest", required=True)
    i.set_defaults(func=cmd_init)

    rb = sub.add_parser("rollback", help="把线上那棵树整体退回 best 或某个 roundN")
    rb.add_argument("--run", required=True, help="曲线目录（内含 best/ 或 roundN/）")
    rb.add_argument("--to", default="best", help="best（默认）或 round1/round2/...")
    rb.add_argument("--target", default="harness/live",
                    help="要覆盖的线上树；默认 harness/live，并会自动备份原树")
    rb.add_argument("--dry-run", action="store_true", help="只打印会改什么，不落盘")
    rb.set_defaults(func=cmd_rollback)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())