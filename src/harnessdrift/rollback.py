"""回滚：把线上那棵树退回某个已知状态。

为什么这个功能值得单独一个模块：自改进系统里最危险的一刻不是"没涨分"，
而是"涨了分但同时把某个能力改坏了，而没人发现"。有了回滚，试错的代价才有上界——
这也是把 RSI 拿去线上用之前必须有的那一件东西。

回滚是**整棵树替换**，不是打反向补丁。理由：反向补丁要求能精确反演，
而一轮补丁可能是 REPLACE 整篇 + 若干 DELETE，反演出来的东西未必等于父树；
替换则一定等于父树的快照（那份快照本来就在 runs/<run>/roundN 里存着）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence

from .guards import assert_no_grader_verdicts
from .paths import read_tree, write_tree

To = Literal["best"] | str  # "best" 或 "round1" / "round2" ...


@dataclass
class RollbackPlan:
    run_dir: Path
    source: Path
    target: Path
    tree: dict[str, str]
    added: list[str]
    changed: list[str]
    removed: list[str]

    @property
    def empty(self) -> bool:
        return not (self.added or self.changed or self.removed)

    def describe(self) -> str:
        rows = [f"回滚 {self.target} ← {self.source}"]
        rows.append(f"  新增 {len(self.added)} 个文件" + (f"：{'、'.join(self.added)}" if self.added else ""))
        rows.append(f"  改动 {len(self.changed)} 个文件" + (f"：{'、'.join(self.changed)}" if self.changed else ""))
        rows.append(f"  删除 {len(self.removed)} 个文件" + (f"：{'、'.join(self.removed)}" if self.removed else ""))
        return "\n".join(rows)


def _resolve_source(run_dir: Path, to: str) -> Path:
    """把 ``--to`` 解析成一个真实的树目录。

    只允许回滚到"曾经被接受过的"状态：``best`` 或某个 roundN。
    故意**不允许**回滚到 ``_rejected/roundN`` —— 那是被规则否掉的候选，
    把它变成线上状态等于绕过接受规则，回滚不该成为后门。
    """
    if to == "best":
        src = run_dir / "best"
        if not src.is_dir():
            raise FileNotFoundError(f"{run_dir} 下没有 best/（这条曲线可能没跑完）")
        return src
    if not to.startswith("round"):
        raise ValueError(f"--to 只接受 best 或 round<N>，收到 {to!r}")
    src = run_dir / to
    if not src.is_dir():
        raise FileNotFoundError(
            f"{run_dir} 下没有 {to}/。注意被拒绝的轮次在 _rejected/{to}，"
            "那是有意不让回滚进去的：被规则否掉的候选不该成为线上状态"
        )
    return src


def plan_rollback(run_dir: Path, to: str, target: Path,
                  ignore: Sequence[str] = ("PATCH.md",)) -> RollbackPlan:
    src = _resolve_source(Path(run_dir), to)
    new_tree = read_tree(src, ignore=ignore)
    assert_no_grader_verdicts(new_tree)
    old_tree = read_tree(target) if Path(target).is_dir() else {}
    added = sorted(set(new_tree) - set(old_tree))
    removed = sorted(set(old_tree) - set(new_tree))
    changed = sorted(p for p in set(new_tree) & set(old_tree) if new_tree[p] != old_tree[p])
    return RollbackPlan(
        run_dir=Path(run_dir), source=src, target=Path(target), tree=new_tree,
        added=added, changed=changed, removed=removed,
    )


def apply_rollback(plan: RollbackPlan, *, dry_run: bool = False) -> Path | None:
    """执行回滚，返回备份目录；dry_run 时什么都不写，返回 None。

    备份放在 ``<target>/../.rollback-backup-<时间戳>``：回滚本身也可能回错
    （例如回滚到了一条本身有问题的曲线），没有备份的"一键回滚"只是换一种方式丢东西。
    """
    if dry_run:
        return None
    backup: Path | None = None
    if plan.target.is_dir() and any(plan.target.iterdir()):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = plan.target.parent / f".rollback-backup-{stamp}"
        backup.mkdir(parents=True, exist_ok=True)
        for p, c in read_tree(plan.target).items():
            fp = backup / p
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(c, encoding="utf-8", newline="")
    # 先清掉旧文件再写：否则"回滚"会留下目标树里那些新树没有的文件，
    # 结果既不是旧树也不是新树。
    if plan.target.is_dir():
        for p in list(read_tree(plan.target)):
            if p not in plan.tree:
                (plan.target / p).unlink()
    write_tree(plan.tree, plan.target)
    return backup