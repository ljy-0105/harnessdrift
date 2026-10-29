"""提案的 diff 契约：omitted is inherited（不写就是继承）。

Harness? 的编辑契约原文是 "a proposal emits only what it changes"。
这里把它落成一种可解析、可审计、可回滚的补丁格式 —— 不用 unified diff，
因为 LLM 写统一 diff 的上下文行经常对不齐，而这里要的是**声明式**改动。

格式（缩进 4 个空格的内容行可选，便于在聊天里直接贴）::

    ### ADD systemprompt.md
        <整份内容>

    ### REPLACE tool_descriptions/tool_guidance.md
        <整份内容>

    ### DELETE skills/old_one/SKILL.md

三条硬规则：

1. **没提到的路径保持原样。**这是 omitted-is-inherited 的全部含义。
2. **ADD 一个已存在的路径 = 错误。**否则"新增技能"和"改技能"分不清，
   日志里就无法判断提案到底想干什么。
3. **DELETE / REPLACE 一个不存在的路径 = 错误。**不做宽容处理 ——
   静默成功会让一次错的提案看起来像成功的提案。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .paths import PathEscapeError, Tree, normalize_rel, normalize_tree

Action = Literal["ADD", "REPLACE", "DELETE"]

_HEADER = re.compile(r"^###\s+(ADD|REPLACE|DELETE)\s+(.+?)\s*$")


class EditError(ValueError):
    """提案格式错误，或提案的语义前提不成立。"""


@dataclass(frozen=True)
class Edit:
    action: Action
    path: str  # 已归一化
    body: str | None = None

    def __post_init__(self) -> None:
        if self.action == "DELETE" and self.body is not None:
            raise EditError(f"DELETE {self.path!r} must not carry a body")
        if self.action in ("ADD", "REPLACE") and self.body is None:
            raise EditError(f"{self.action} {self.path!r} needs a body")


def parse_patch(text: str) -> list[Edit]:
    """把提案文本解析成 Edit 列表。任何格式问题都抛错，不做静默跳过。"""
    edits: list[Edit] = []
    current: tuple[Action, str] | None = None
    buf: list[str] = []
    seen: set[str] = set()

    def flush() -> None:
        nonlocal current, buf
        if current is None:
            return
        action, path = current
        # 在 flush 这里校验，而不是只在 Edit.__post_init__ 里校验：
        # DELETE 的 body 会被下面的三元表达式丢弃，永远不会传给构造函数，
        # 所以"DELETE 带了内容"这种手滑会被静默吞掉。必须在丢弃之前拦住。
        stray = [ln for ln in buf if ln.strip()]
        if action == "DELETE" and stray:
            raise EditError(
                f"DELETE {path!r} must not carry content; got {len(stray)} line(s) "
                "starting with " + repr(stray[0].strip()[:60])
            )
        body = None if action == "DELETE" else "\n".join(buf).rstrip() + "\n"
        edits.append(Edit(action=action, path=path, body=body))
        current, buf = None, []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        m = _HEADER.match(line)
        if m:
            flush()
            action = m.group(1)  # type: ignore[assignment]
            try:
                path = normalize_rel(m.group(2).strip())
            except PathEscapeError as e:
                raise EditError(f"line {lineno}: invalid edit path: {e}") from e
            if path in seen:
                raise EditError(f"line {lineno}: {path!r} appears twice in one proposal")
            seen.add(path)
            current = (action, path)  # type: ignore[assignment]
            continue
        if line.startswith("###"):
            raise EditError(
                f"line {lineno}: unrecognised header {line!r}; "
                "expected '### ADD|REPLACE|DELETE <path>'"
            )
        if current is None:
            if line.strip():
                raise EditError(
                    f"line {lineno}: content before the first header; "
                    "a proposal must start with '### ADD ...' etc."
                )
            continue
        # 正文必须整行缩进 4 个空格（空行豁免）。
        # 原来这里是 `line[4:] if startswith("    ") else line` —— 顶格的行被原样收下，
        # 于是 "### REPLACE x\nnot-indented" 会解析成功，把整个文件替换成一行垃圾。
        # **静默接受坏补丁比直接报错危险得多**：自改进会把它当成一次成功的改动记进曲线。
        if line.strip() and not line.startswith("    "):
            raise EditError(
                f"line {lineno}: body line is not indented by 4 spaces: {line.strip()[:60]!r}. "
                "A patch body must be uniformly indented so that prose around it cannot be "
                "mistaken for content."
            )
        buf.append(line[4:] if line.startswith("    ") else "")
    flush()

    if not edits:
        raise EditError("empty proposal: no '### ADD|REPLACE|DELETE <path>' header found")
    return edits


def apply_edits(tree: Tree, edits: list[Edit]) -> Tree:
    """把 edits 作用到树上，返回新树；原树不变。"""
    out = normalize_tree(tree)
    for e in edits:
        try:
            path = normalize_rel(e.path)
        except PathEscapeError as exc:
            raise EditError(f"invalid edit path {e.path!r}: {exc}") from exc
        exists = path in out
        if e.action == "ADD":
            if exists:
                raise EditError(f"ADD {path!r}: already present; use REPLACE to change it")
            assert e.body is not None
            out[path] = e.body
        elif e.action == "REPLACE":
            if not exists:
                raise EditError(f"REPLACE {path!r}: not present; use ADD to create it")
            assert e.body is not None
            out[path] = e.body
        else:  # DELETE
            if not exists:
                raise EditError(f"DELETE {path!r}: not present")
            del out[path]
    return dict(sorted(out.items()))


def diff_summary(before: Tree, after: Tree) -> dict[str, list[str]]:
    """两份树之间的增删改，按 posix 路径排序。

    这份摘要就是"这一轮到底改了什么"，要原样写进 runs/ 下的日志，
    也是里给面试官看的东西。
    """
    b, a = normalize_tree(before), normalize_tree(after)
    return {
        "added": sorted(set(a) - set(b)),
        "removed": sorted(set(b) - set(a)),
        "modified": sorted(k for k in set(a) & set(b) if a[k] != b[k]),
    }


def render_patch(before: Tree, after: Tree) -> str:
    """把一对树渲染回可重放的提案文本（用于日志回放与回归测试）。

    正文按 4 空格缩进渲染 —— 这是提案的规范形态，也让它能直接贴进聊天；
    parse_patch 会把这层缩进剥掉，所以 ``before -> render -> parse -> apply``
    必须精确还原 ``after``（测试里有断言）。

    副作用：正文里恰好以 4 个空格开头的行会被剥成 0 个空格。这是该格式的
    已知代价，换来的是"diff 里没有任何一行的开头的 ## 会被误当成新块头"。
    """
    b, a = normalize_tree(before), normalize_tree(after)
    d = diff_summary(b, a)
    blocks: list[str] = []

    def body_lines(p: str) -> list[str]:
        return ["    " + ln for ln in a[p].rstrip("\n").split("\n")]

    for p in d["added"]:
        blocks.append("\n".join([f"### ADD {p}", *body_lines(p)]))
    for p in d["modified"]:
        blocks.append("\n".join([f"### REPLACE {p}", *body_lines(p)]))
    for p in d["removed"]:
        blocks.append(f"### DELETE {p}")
    # 块之间只用一个换行：再叠一个空行会被 parse_patch 当成正文首行，
    # 正文会多出一个前导 \n，往返一轮就漂一次。
    return "\n".join(blocks) + "\n"
