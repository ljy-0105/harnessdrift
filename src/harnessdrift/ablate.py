"""归因：把一轮补丁拆成单点改动，逐条单独合进父树，分别评测。

## 为什么需要这一层

曲线只回答"这棵树比上一棵好"。它答不了面试官最自然的那个追问：

> 你怎么知道是这十几行里的哪一条起了作用？说不定是模型那天状态好。

没有归因，这个问题的唯一答案就是"我不知道，但总分确实涨了"。而总分涨了
并不能排除"某条改动有害、被另一条更强的改动盖住了"——被盖住的那条如果
留着，下一轮就会在它上面继续叠。

## 怎么拆

以**文件级 diff 的 hunk** 为单位，不是以"文件"为单位。一轮补丁可能只改两个
文件，但其中一个是整篇重写；按文件拆只会得到两个原子，什么也说明不了。
`difflib.SequenceMatcher` 的 opcodes 天然给出连续改动段，一段一个原子。

实测 `runs/curve3/round1` 的补丁拆出 5 个原子：SKILL.md 1 个、systemprompt.md 4 个。

## 三种效应

- **单点有效**：这个原子单独合进父树就能涨分 → 它是真因
- **单点无感 / 单点有害**：单独看没变化甚至掉分 → 它只有在组合里才有用，
  或者根本是噪声。这类原子留着就是负债
- **交互效应**：全量 Δ 减去"最大单点 Δ"，剩下的部分来自原子之间的配合。
  拆出来的原子单独都无效、合起来却有效，是完全可能的（先加规则、再加引用它的段落）

## 诚实边界

单点评测比整棵树评测算得准（只动一处，噪声源少），但 `reps` 一样时它的
std 也一样粗。所以下面的表里每个单点都带着自己的 std，`Δ < std` 的一律
不称为"有效"，只记为"测不出"。
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .paths import read_tree, write_tree
from .runner import EvalResult, evaluate_tree

__all__ = ["Atom", "AblationResult", "split_atoms", "ablate", "write_attribution"]


@dataclass
class Atom:
    """一个 hunk：把父树的若干行换成候选树的若干行（或增/删整个文件）。"""

    index: int
    path: str
    kind: str                      # modify / add / delete
    tag: str                       # difflib 的 opcode 标签
    parent_span: tuple[int, int]
    cand_span: tuple[int, int]
    label: str

    def apply(self, parent_tree: dict[str, str], cand_tree: dict[str, str]) -> dict[str, str]:
        """把这个原子单独合进父树，返回一棵新树（不改入参）。

        注意 `parent_span` 是在**原始父树**上量的。所以多个原子要合到一起时
        必须**从后往前**（见 `apply_all`），否则前一个原子插入的行会把
        后一个原子的行号全部推偏，合出来的树既不是父树也不是候选树。
        """
        tree = dict(parent_tree)
        if self.kind == "add":
            tree[self.path] = cand_tree[self.path]
            return tree
        if self.kind == "delete":
            tree.pop(self.path, None)
            return tree

        parent_lines = parent_tree[self.path].splitlines(keepends=True)
        cand_lines = cand_tree[self.path].splitlines(keepends=True)
        i1, i2 = self.parent_span
        j1, j2 = self.cand_span
        merged = parent_lines[:i1] + cand_lines[j1:j2] + parent_lines[i2:]
        tree[self.path] = "".join(merged)
        return tree


def _label_for(lines: list[str], start: int, span: int) -> str:
    """给 hunk 取一个能认出来的名字：第一行非空内容，截断到 40 字。"""
    for ln in lines[start:start + max(span, 1)]:
        s = ln.strip().lstrip("#").strip()
        if s:
            return s[:40]
    return "（空行改动）"


def split_atoms(parent_tree: dict[str, str],
                cand_tree: dict[str, str]) -> list[Atom]:
    """按 hunk 切分两棵树之间的差异。顺序稳定，便于复现。"""
    atoms: list[Atom] = []
    for path in sorted(set(parent_tree) | set(cand_tree)):
        p = parent_tree.get(path)
        c = cand_tree.get(path)
        if p is None:
            atoms.append(Atom(len(atoms), path, "add", "add", (0, 0), (0, 0),
                              f"新增文件 {path}"))
            continue
        if c is None:
            atoms.append(Atom(len(atoms), path, "delete", "delete", (0, 0), (0, 0),
                              f"删除文件 {path}"))
            continue
        if p == c:
            continue

        pl = p.splitlines(keepends=True)
        cl = c.splitlines(keepends=True)
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, pl, cl).get_opcodes():
            if tag == "equal":
                continue
            # 纯空白/编码差异拆出来没有意义，还会白烧一次评测
            if "".join(pl[i1:i2]).strip() == "".join(cl[j1:j2]).strip():
                continue
            atoms.append(Atom(
                index=len(atoms), path=path, kind="modify", tag=tag,
                parent_span=(i1, i2), cand_span=(j1, j2),
                label=_label_for(cl, j1, j2 - j1) if j2 > j1 else _label_for(pl, i1, i2 - i1),
            ))
    return atoms


def apply_all(parent_tree: dict[str, str], cand_tree: dict[str, str],
              atoms: list[Atom]) -> dict[str, str]:
    """把所有原子合回父树，应当逐字节等于候选树。

    必须**从后往前**合：原子的行号都是在原始父树上量的，先合前面的会把
    后面原子的位置整体推偏。这个函数的存在意义是给测试一个可以断言的不变量
    —— 拆错了的话，后面每一条 delta 都是假的，还看不出来。
    """
    tree = dict(parent_tree)
    for atom in sorted(atoms, key=lambda a: (a.path, a.parent_span), reverse=True):
        tree = atom.apply(tree, cand_tree)
    return tree


@dataclass
class AtomResult:
    atom: Atom
    mean: float = 0.0
    std: float = 0.0
    delta: float = 0.0
    verdict: str = "未测"        # 有效 / 有害 / 测不出

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.atom.index,
            "path": self.atom.path,
            "kind": self.atom.kind,
            "label": self.atom.label,
            "parent_span": list(self.atom.parent_span),
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "delta": round(self.delta, 4),
            "verdict": self.verdict,
        }


@dataclass
class AblationResult:
    base_mean: float = 0.0
    base_std: float = 0.0
    full_mean: float = 0.0
    full_std: float = 0.0
    reps: int = 1
    n_tasks: int = 0
    atoms: list[AtomResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    full_ok: bool = True
    parent_dir: str = ""
    cand_dir: str = ""

    @property
    def full_delta(self) -> float:
        return self.full_mean - self.base_mean

    @property
    def best_atom_delta(self) -> float:
        return max((a.delta for a in self.atoms), default=0.0)

    @property
    def interaction(self) -> float:
        """全量 Δ 里，光靠最大单点解释不了的那部分。"""
        return self.full_delta - self.best_atom_delta

    def as_dict(self) -> dict[str, Any]:
        return {
            "base": {"mean": round(self.base_mean, 4), "std": round(self.base_std, 4)},
            "full": {"mean": round(self.full_mean, 4), "std": round(self.full_std, 4),
                     "ok": self.full_ok},
            "reps": self.reps,
            "n_tasks": self.n_tasks,
            "full_delta": round(self.full_delta, 4),
            "best_atom_delta": round(self.best_atom_delta, 4),
            "interaction": round(self.interaction, 4),
            "atoms": [a.as_dict() for a in self.atoms],
            "notes": self.notes,
            # 出处：看板要靠它判断"这份归因是不是属于这条曲线"
            "parent_dir": self.parent_dir,
            "cand_dir": self.cand_dir,
        }


def _verdict(delta: float, base_std: float, atom_std: float) -> str:
    """判据和循环里一致：增益要超过两边的抖动才算数。

    单点的 std 往往比整棵树小（只动一处，别的都不变），所以这里用
    max(两者的 std) 依然是保守的：宁可选不出结论，也别把噪声当功劳。
    """
    noise = max(base_std, atom_std, 1e-9)
    if delta > noise:
        return "有效"
    if delta < -noise:
        return "有害"
    return "测不出"


def ablate(
    parent_tree: dict[str, str],
    cand_tree: dict[str, str],
    tasks: list[Any],
    llm: Any,
    reps: int = 1,
    out_root: Path | None = None,
    log: Callable[[str], None] = print,
    verbose: bool = False,
    workers: int = 1,
) -> AblationResult:
    """评测父树、每个单点、全量补丁三者，算出每个原子的贡献。"""
    res = AblationResult(reps=reps, n_tasks=len(tasks))
    atoms = split_atoms(parent_tree, cand_tree)

    log(f"=== 基线与全量（{len(tasks)} 条 × {reps} 轮）===")
    base_ev = evaluate_tree(parent_tree, tasks, llm, reps=reps, verbose=verbose, workers=workers)
    res.base_mean, res.base_std = base_ev.mean, base_ev.std
    log(f"  父树 = {base_ev.mean:.2f}±{base_ev.std:.2f}")

    full_ok, why = _tree_is_applicable(cand_tree)
    if not full_ok:
        # 全量树根本跑不了（比如补丁把 systemprompt.md 删空了）。这本身就是
        # 一条结论，不该假装评出了分数 —— 报 0 分会让 Δ 看起来像"掉了 38 分"。
        res.full_ok = False
        res.notes.append(f"全量补丁不可运行（{why}），未评测；单点仍逐条评测。")
        log(f"  全量 = 不可运行（{why}）")
    else:
        full_ev = evaluate_tree(cand_tree, tasks, llm, reps=reps, verbose=verbose, workers=workers)
        res.full_mean, res.full_std = full_ev.mean, full_ev.std
        log(f"  全量 = {full_ev.mean:.2f}±{full_ev.std:.2f}  (Δ{res.full_delta:+.2f})")

    if not atoms:
        res.notes.append("两棵树内容相同，没有可拆的改动。")
        return res

    log(f"=== 逐条评测 {len(atoms)} 个单点 ===")
    for atom in atoms:
        tree = atom.apply(parent_tree, cand_tree)
        ok, why = _tree_is_applicable(tree)
        if not ok:
            res.atoms.append(AtomResult(atom, verdict=f"跳过：{why}"))
            log(f"  [{atom.index}] {atom.label[:28]} —— 跳过（{why}）")
            continue
        ev = evaluate_tree(tree, tasks, llm, reps=reps, verbose=verbose, workers=workers)
        ar = AtomResult(atom, mean=ev.mean, std=ev.std, delta=ev.mean - base_ev.mean)
        ar.verdict = _verdict(ar.delta, base_ev.std, ev.std)
        res.atoms.append(ar)
        log(f"  [{atom.index}] {atom.label[:26]:<28} {ev.mean:6.2f}±{ev.std:.2f}"
            f"  Δ{ar.delta:+.2f}  {ar.verdict}")

    return res


def _tree_is_applicable(tree: dict[str, str]) -> tuple[bool, str]:
    """单点合出来的树还得是棵能跑的树。

    比如删除整个 systemprompt.md 这个原子，合出来没有提示词可跑，评测会
    一路报错，白烧 token。先在本地判掉，比让评测器去撞错便宜得多。
    """
    if not tree.get("systemprompt.md", "").strip():
        return False, "systemprompt.md 变空"
    return True, ""


def write_attribution(res: AblationResult, path: Path) -> Path:
    """写 ATTRIBUTION.md：一张表说清每个原子单独值多少分。"""
    lines: list[str] = []
    lines.append("# 归因报告：这轮补丁里是哪一条在起作用\n")
    full_txt = (f"{res.full_mean:.2f}±{res.full_std:.2f}" if res.full_ok else "不可运行")
    lines.append(f"> {res.n_tasks} 条任务 × {res.reps} 次重复 · "
                 f"父树 {res.base_mean:.2f}±{res.base_std:.2f} · "
                 f"全量 {full_txt}\n")

    lines.append("## 1. 结论\n")
    lines.append("| 量 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 父树得分 | {res.base_mean:.2f}±{res.base_std:.2f} |")
    lines.append(f"| 全量补丁得分 | {full_txt} |")
    lines.append(f"| 全量 Δ | **{res.full_delta:+.2f}** |" if res.full_ok
                 else "| 全量 Δ | 无法计算（全量树跑不了） |")
    lines.append(f"| 最大单点 Δ | {res.best_atom_delta:+.2f} |")
    lines.append(f"| 交互效应（全量 − 最大单点） | {res.interaction:+.2f} |")
    lines.append("")

    effective = [a for a in res.atoms if a.verdict == "有效"]
    harmful = [a for a in res.atoms if a.verdict == "有害"]
    unclear = [a for a in res.atoms if a.verdict.startswith(("测不出", "跳过"))]

    if effective:
        names = "、".join(f"`{a.atom.label[:24]}`" for a in effective[:3])
        lines.append(f"**{len(effective)} 个单点单独就能涨分**（{names}），"
                     f"它们是这轮增益的直接来源。")
    else:
        lines.append("**没有任何单点能单独涨分。** 这说明增益来自原子之间的配合，"
                     "或者落在噪声里 —— 这两种情况必须分开讲，不能都叫\"改进\"。")
    if harmful:
        lines.append(f"\n**{len(harmful)} 个单点单独看是掉分的**："
                     + "、".join(f"`{a.atom.label[:24]}`" for a in harmful)
                     + "。它们在全量里被别的改动盖住了，留着是负债。")
    if unclear:
        lines.append(f"\n{len(unclear)} 个单点测不出结论（|Δ| 没超过抖动），"
                     "不能说它有效，也不能说它有害。")
    lines.append("")

    lines.append("## 2. 逐条明细\n")
    lines.append("| # | 文件 | 改动 | 单独得分 | Δ | 结论 |")
    lines.append("|---:|---|---|---:|---:|---|")
    for a in res.atoms:
        span = ""
        if a.atom.kind == "modify":
            i1, i2 = a.atom.parent_span
            span = f"L{i1 + 1}" if i1 == i2 else f"L{i1 + 1}-{i2}"
        lines.append(
            f"| {a.atom.index} | `{a.atom.path}` {span} | {a.atom.label[:30]} | "
            f"{a.mean:.2f}±{a.std:.2f} | {a.delta:+.2f} | {a.verdict} |"
        )
    lines.append("")
    lines.append("> Δ 是与父树比。判据同循环：Δ 必须超过 `max(父树 std, 单点 std)`，"
                 "否则记为「测不出」。单点 std 常比整棵树小，所以这个判据偏保守。\n")

    lines.append("## 3. 怎么读这张表\n")
    lines.append("- **有效**：这条改动单独拿出来就能涨分，是真正的因果。面试可以直说这一条。")
    lines.append("- **测不出**：多数是这类补丁的常态 —— 一条规则单独加进去，模型没有"
                 "触发场景，自然看不出效果。这不代表它没用，只代表**单点评测证明不了它有用**。")
    lines.append("- **交互效应为正**：说明原子之间有配合（比如先加规则、再改引用它的段落）。"
                 "这部分归因只能到「组合」这一级，再细就拆不动了。")
    if res.notes:
        lines.append("")
        for n in res.notes:
            lines.append(f"- {n}")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*由 `harnessdrift ablate` 生成。原始数据见同目录 `attribution.json`。*")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def dump(res: AblationResult, path: Path) -> Path:
    path.write_text(json.dumps(res.as_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path