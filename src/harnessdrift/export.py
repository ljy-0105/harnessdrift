"""把一次跑批的产物打成一个静态看板（单文件 HTML，双击就开）。

## 为什么要这一步

`report.py` 出的是给人读的 markdown，`curve.json` 是给程序读的原始数据。
中间缺一层：**给不打算读代码的人看的东西**。面试官不会 clone 下来跑一遍，
但他会点开一个链接。

所以这里只做一件事：把 `curve.json` + 每轮的 `PATCH.md` + `attribution.json`
+ 若干条对照曲线，塞进一个 HTML 文件里。没有后端、没有构建、没有依赖 ——
数据内联成 `window.__DATA__`，因此 `file://` 直接打开也能跑（不用起 server，
也就不会被 CORS 挡住）。

## 不做什么

不做实时、不做交互式重跑、不接数据库。看板是**证据的投影**，不是新的数据源：
页面上每个数字都必须能在 `runs/` 里找到出处，否则就是在编。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .compare import load_curve
from .paths import read_tree  # noqa: F401  (保留给以后的树对比)

__all__ = ["build_payload", "render_html", "export"]

_TEMPLATE = Path(__file__).resolve().parents[2] / "web" / "template.html"

# 每轮的得分要和"父树"比，而不是和上一轮的候选比：被拒绝的候选不是父树。
# 这个口径和 loop.py 里的 _is_improved(gain = cand - parent) 保持一致。


def _read_patch(run_dir: Path, rnd: dict[str, Any]) -> str:
    """读这一轮的 PATCH.md。

    `harness_dir` 是**仓库根相对**的路径，而且在 Windows 上写作 `runs\\curve3\\round1`。
    直接 `Path(...)` 在 Linux 上会把整串当成一个文件名 —— 于是同一份 curve.json
    在 Windows 导出时有补丁、在 Linux CI 里补丁全空，产物还不报错。
    所以：先归一化分隔符，再按几个候选位置依次试。
    """
    hd = str(rnd.get("harness_dir") or "").replace("\\", "/")
    if not hd:
        return ""
    cands = [
        Path(hd) / "PATCH.md",                    # 相对 cwd（仓库根）
        run_dir.parent / hd / "PATCH.md",         # 相对 runs/
        run_dir / hd / "PATCH.md",
        run_dir / f"round{rnd.get('round_id')}" / "PATCH.md",   # 最后兜底：按轮号
    ]
    for p in cands:
        try:
            if p.exists():
                return p.read_text(encoding="utf-8")
        except OSError:
            continue
    return ""


def _rounds(curve: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    parent_mean = float(curve["baseline"]["mean"])
    parent_solved = int((curve["rounds"][0] or {}).get("solved") or 0)
    for r in curve.get("rounds", []):
        if r.get("kind") == "base":
            out.append({
                "round_id": 0,
                "kind": "base",
                "mean": r["mean"],
                "std": r["std"],
                "solved": r.get("solved", 0),
                "n_tasks": r.get("n_tasks", 0),
                "elapsed_s": r.get("elapsed_s", 0.0),
                "delta": None,
                "delta_solved": None,
                "changed": {},
                "broke": [],
                "broke_safety": [],
                "flaky": r.get("flaky") or {},
                "passes": r.get("passes") or {},
                "failure_hist": r.get("failure_hist") or {},
                "patch": "",
                "tokens": r.get("tokens") or {},
                "errors": r.get("errors") or {},
            })
            continue
        accepted = r.get("kind") == "accepted"
        out.append({
            "round_id": r.get("round_id"),
            "kind": r.get("kind"),
            "accepted": accepted,
            "improved": bool(r.get("improved")),
            "neutral": bool(r.get("neutral")),
            "mean": r["mean"],
            "std": r["std"],
            "solved": r.get("solved", 0),
            "n_tasks": r.get("n_tasks", 0),
            "elapsed_s": r.get("elapsed_s", 0.0),
            "delta": round(float(r["mean"]) - parent_mean, 2),
            "delta_solved": int(r.get("solved", 0)) - parent_solved,
            "changed": r.get("changed") or {},
            "broke": r.get("broke") or [],
            "broke_safety": r.get("broke_safety") or [],
            "flaky": r.get("flaky") or {},
            "passes": r.get("passes") or {},
            "failure_hist": r.get("failure_hist") or {},
            "patch": _read_patch(run_dir, r),
            "tokens": r.get("tokens") or {},
            "errors": r.get("errors") or {},
        })
        if accepted:
            parent_mean = float(r["mean"])
            parent_solved = int(r.get("solved", 0))
    return out


def _tasks(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "id": t["id"],
            "message": t.get("message", ""),
            "why": t.get("why", ""),
            "expect": t.get("expect", ""),
            "tier": t.get("tier", "normal"),
        }
        for t in raw
    ]


def _attribution(attr_dir: Path | None, run_dir: Path | None = None) -> dict[str, Any] | None:
    if not attr_dir:
        return None
    p = attr_dir / "attribution.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    d["_dir"] = attr_dir.name
    d["_match"] = _attr_matches_run(d, run_dir)
    return d


def _attr_matches_run(attr: dict[str, Any], run_dir: Path | None) -> bool | None:
    """这份归因是不是这条曲线里的某一轮。

    用出处（`ablate` 记下的 `cand_dir` / `parent_dir`）判，不用分数判 ——
    分数判不了：同一棵树重新评测一次，基线就能差一分多（32.00 → 33.33 是常事），
    拿分数比对会把合法的重测当成挂错目录。

    没有出处字段的旧文件返回 None（不知道），看板上就不提示。

    这条检查是被自己坑出来的：`--run runs/strict_veto --attr runs/attr_curve3`
    拼错一次，看板上就是"这条曲线第 1 轮改了什么"配着**另一条曲线**的归因数字，
    而且两边都自洽、都带小数，看不出破绽。
    """
    if run_dir is None or not attr.get("cand_dir"):
        return None
    run = Path(run_dir).resolve()
    try:
        cand = Path(attr["cand_dir"]).resolve()
        parent = Path(attr["parent_dir"]).resolve() if attr.get("parent_dir") else None
    except OSError:
        return None
    for p in (cand, parent):
        if p is not None and (p == run or run in p.parents or p in run.parents):
            return True
    return False


def _curves(dirs: list[Path]) -> list[dict[str, Any]]:
    out = []
    for d in dirs:
        c = load_curve(d)
        out.append({
            "name": d.name,
            "accept": c.accept,
            "baseline": c.baseline,
            "baseline_std": c.baseline_std,
            "baseline_solved": c.baseline_solved,
            "best": c.best,
            "best_std": c.best_std,
            "best_solved": c.best_solved,
            "best_round": c.best_round,
            "last_accepted_mean": c.last_accepted_mean,
            "delta": round(c.delta, 2),
            "accepted": c.accepted,
            "rejected": c.rejected,
            "broke_total": c.broke_total,
            "regressed": c.regressed,
            "lost_stable": c.lost_stable(),
            "gained_stable": c.gained_stable(),
            "stop_reason": c.stop_reason,
            "elapsed_s": c.elapsed_s,
            "n_tasks": c.n_tasks,
            "reps": c.reps,
            "rows": [
                {"round_id": r.get("round_id"), "kind": r.get("kind"),
                 "mean": r["mean"], "std": r["std"], "solved": r.get("solved", 0)}
                for r in c.rounds
            ],
        })
    return out


def build_payload(
    run_dir: Path,
    *,
    tasks_path: Path,
    attr_dir: Path | None = None,
    compare_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    curve = json.loads((run_dir / "curve.json").read_text(encoding="utf-8"))
    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "run": run_dir.name,
        "config": curve.get("config") or {},
        "stop_reason": curve.get("stop_reason", ""),
        "elapsed_s": curve.get("total_elapsed_s", 0.0),
        "baseline": curve.get("baseline") or {},
        "final": curve.get("final") or {},
        "delta": curve.get("delta", 0.0),
        "tokens": (curve.get("total_tokens") or {}),
        "tasks": _tasks(tasks_path),
        "rounds": _rounds(curve, run_dir),
        "attribution": _attribution(attr_dir, run_dir),
        "curves": _curves(compare_dirs or []),
    }


def render_html(payload: dict[str, Any], template: Path = _TEMPLATE) -> str:
    tpl = template.read_text(encoding="utf-8")
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # 内联 JSON 必须切断 </script>，否则任务文案里出现这串字符就把脚本闭合了
    blob = blob.replace("</", "<\\/")
    if tpl.count("/*__DATA__*/") != 1:
        # 占位符必须唯一：它曾经同时出现在头部注释里，于是数据被内联了两遍，
        # 页面还能打开，但体积翻倍、且注释里躺着一份 JSON。
        raise ValueError(f"模板里的 /*__DATA__*/ 必须恰好出现一次：{template}")
    return tpl.replace("/*__DATA__*/", "window.__DATA__ = " + blob + ";")


def export(
    run_dir: Path,
    out: Path,
    *,
    tasks_path: Path,
    attr_dir: Path | None = None,
    compare_dirs: list[Path] | None = None,
    template: Path = _TEMPLATE,
) -> dict[str, Any]:
    payload = build_payload(
        run_dir, tasks_path=tasks_path, attr_dir=attr_dir, compare_dirs=compare_dirs
    )
    html = render_html(payload, template)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return payload