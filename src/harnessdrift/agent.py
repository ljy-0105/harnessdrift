"""Agent 循环：harness 树 → 系统提示 → 工具循环 → 最终答复。

这里的 **harness 就是一个 dict**。第 0 步的 ``Tree`` 类型到这里第一次真的被用上：
``systemprompt.md`` 和 ``tool_descriptions/*.md`` 拼成系统提示，
``skills/*/SKILL.md`` 按需加载。

刻意没有的：反思、记忆、多 agent。那些是第 4、5 步的事，
第 1 步的基线必须足够简单，简单到你敢说"这个分数就是这棵树的能力"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .llm import LLM
from .paths import Tree, normalize_tree
from .tools import Toolkit


@dataclass
class Turn:
    role: str
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunResult:
    final_answer: str
    trace: list[dict[str, Any]]   # Toolkit.trace
    turns: int
    stopped_reason: str           # answered / max_turns / error
    prompt_violations: list[str] = field(default_factory=list)


_SKILL_HINT = re.compile(r"skills?/([A-Za-z0-9_\-]+)")


def load_harness_prompt(tree: Tree) -> str:
    """把 harness 树渲染成系统提示。可见性规则集中在这里，便于第 4 步对比。"""
    t = normalize_tree(tree)

    if "systemprompt.md" not in t:
        raise ValueError("harness tree has no systemprompt.md")

    # 工具说明：tool_descriptions/*.md 全部拼接，给人改的入口
    tool_notes = [
        body.strip()
        for rel, body in t.items()
        if rel.startswith("tool_descriptions/") and rel.endswith(".md")
    ]

    # 技能只在**被提到过**的时候注入正文；否则只给一行目录。
    # 这是"上下文预算"的第一次体现：全量注入会让树越大提示越长，不可持续。
    skill_index: list[str] = []
    skill_bodies: list[str] = []
    for rel, body in t.items():
        if rel.startswith("skills/") and rel.endswith("/SKILL.md"):
            name = rel.split("/")[1]
            first = next(
                (ln.strip("# ").strip() for ln in body.splitlines() if ln.strip()),
                name,
            )
            skill_index.append(f"- {name}: {first}")
            skill_bodies.append(f"### skill: {name}\n{body.strip()}")

    referenced = set(_SKILL_HINT.findall(t["systemprompt.md"]))
    active = [sb for sb in skill_bodies if re.search(r"### skill: ([\w\-]+)", sb).group(1) in referenced] \
        if skill_bodies else []

    parts = [t["systemprompt.md"].strip()]
    if tool_notes:
        parts.append("## 工具使用说明\n" + "\n\n".join(tool_notes))
    if skill_index:
        parts.append("## 可用技能\n" + "\n".join(skill_index))
    if active:
        parts.append("## 已加载技能正文\n" + "\n\n".join(active))
    return "\n\n".join(parts) + "\n"


def run_task(
    tree: Tree,
    task_message: str,
    toolkit: Toolkit,
    llm: LLM,
    *,
    max_turns: int = 8,
    verbose: bool = False,
) -> RunResult:
    system = load_harness_prompt(tree)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task_message},
    ]

    violations: list[str] = []
    for turn in range(max_turns):
        reply = llm.chat(messages, tools=Toolkit.TOOL_SPECS)

        if not reply.is_tool_call:
            return RunResult(
                final_answer=reply.text,
                trace=toolkit.trace,
                turns=turn + 1,
                stopped_reason="answered",
                prompt_violations=violations,
            )

        # 记下模型发起的每一次工具调用，模型不一定乖乖调对
        messages.append(
            {
                "role": "assistant",
                "content": reply.text or None,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["raw_arguments"]},
                    }
                    for c in reply.tool_calls
                ],
            }
        )
        for c in reply.tool_calls:
            args = c["arguments"]
            if "__malformed__" in args:
                result: dict[str, Any] = {
                    "ok": False,
                    "error": "BAD_JSON",
                    "message": "工具参数不是合法 JSON，请重新调用",
                }
                toolkit.trace.append({"tool": c["name"], "args": args, "result": result})
            else:
                result = toolkit.dispatch(c["name"], args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": c["id"],
                    "content": __import__("json").dumps(result, ensure_ascii=False),
                }
            )
            if verbose:
                print(f"  turn{turn} -> {c['name']}({args}) => {result.get('ok')}")

    return RunResult(
        final_answer="",
        trace=toolkit.trace,
        turns=max_turns,
        stopped_reason="max_turns",
        prompt_violations=violations,
    )