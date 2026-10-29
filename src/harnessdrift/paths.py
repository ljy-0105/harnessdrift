"""Harness 树的路径归一化与物化。

这一层存在的理由，是 Harness? 在 Windows 上的一个坑：
``read_tree()`` 用 ``str(p.relative_to(d))`` 做 key，在 Windows 上得到的是
``scripts\\summarize.py``；而提案里写的是 ``scripts/summarize.py``。
两者不相等，补丁会**静默**不命中 —— 不报错，只是什么都没改。

所以这里的所有 key 在进入和离开模块时都被强制成 posix 形式。
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path

# 一棵 harness 树：相对路径(posix) -> 文件内容
Tree = dict[str, str]


class PathEscapeError(ValueError):
    """试图写到 harness 根目录之外。"""


def normalize_rel(rel: str) -> str:
    """把任意分隔符的相对路径规整成 posix，并拒绝逃逸。

    >>> normalize_rel(r"scripts\\\\summarize.py")
    'scripts/summarize.py'
    >>> normalize_rel("./systemprompt.md")
    'systemprompt.md'
    """
    if not rel or not str(rel).strip():
        raise PathEscapeError("empty path")
    r = str(rel).strip().replace("\\", "/")

    # 先判绝对路径：Windows 盘符（C:/）和 posix 根（/）都不允许
    if r.startswith("/") or (len(r) > 1 and r[1] == ":"):
        raise PathEscapeError(f"absolute path is not allowed in a harness tree: {rel!r}")

    parts = [p for p in r.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise PathEscapeError(f"path escapes the harness root: {rel!r}")
    if not parts:
        raise PathEscapeError(f"path resolves to the harness root itself: {rel!r}")
    return "/".join(parts)


def normalize_tree(tree: Mapping[str, str] | Iterable[tuple[str, str]]) -> Tree:
    """归一化整棵树；同一个 posix 路径出现两次视为错误（避免静默覆盖）。

    接受 list[pair] 是为了让"归一化后撞车"这件事可被测试 ——
    写成 dict 字面量的话 Python 早就把重复 key 合并了，永远撞不上。
    """
    items = tree.items() if isinstance(tree, Mapping) else tree
    out: Tree = {}
    for rel, body in items:
        key = normalize_rel(rel)
        if key in out:
            raise PathEscapeError(
                f"two entries normalize to the same path {key!r}; "
                "refusing to silently overwrite one with the other"
            )
        out[key] = body
    return dict(sorted(out.items()))


def read_tree(root: Path | str, *, ignore: Iterable[str] = ()) -> Tree:
    """把目录读成树。key 一律 posix。"""
    root = Path(root)
    if not root.is_dir():
        return {}
    skipped = {".git", "__pycache__", ".venv", *ignore}
    out: Tree = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part in skipped for part in rel.parts):
            continue
        if not p.is_file():
            continue
        out[normalize_rel(rel.as_posix())] = p.read_text(encoding="utf-8")
    return dict(sorted(out.items()))


def write_tree(tree: Mapping[str, str], dest: Path | str, *, clean: bool = True) -> Path:
    """把树精确物化成目录。

    ``clean=True`` 先清空目录 —— 这与 Harness? 的语义一致：surface 上的旧文件先删，
    避免上一轮的 skill/script 残留下来继续生效。
    不支持删除单个文件之外的花活，故意保持简单。
    """
    dest = Path(dest)
    if clean and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for rel, body in normalize_tree(tree).items():
        target = dest.joinpath(*rel.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        # newline="" 是刻意的：别让 Windows 把 \n 改成 \r\n，
        # 否则内容哈希和 diff 在跨平台上对不上。
        with open(target, "w", encoding="utf-8", newline="") as f:
            f.write(body)
    return dest


def tree_hash(tree: Mapping[str, str]) -> str:
    """内容哈希，用于日志与缓存键；与文件系统、路径分隔符无关。"""
    import hashlib

    h = hashlib.sha256()
    for rel, body in normalize_tree(tree).items():
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(body.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def replace_atomically(target: Path | str, body: str) -> None:
    """原子写单个文件 —— 用来替代 fcntl 文件锁。

    Harness? 用 ``fcntl.flock`` 保护 summary 的并发写；那个模块在 Windows 上不存在。
    单机、单进程场景下，真正需要的只是"读者永远不会看到半个文件"，
    ``os.replace`` 就够了。
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(body)
    os.replace(tmp, target)
