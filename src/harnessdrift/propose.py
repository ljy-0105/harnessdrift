"""第 2 步 PROPOSE：看失败轨迹，产出一份补丁，写进新的一轮目录。

设计上的三条硬约束（每一条都对应一个"如果不这样会怎样"）：

1. **改进器永远不改 `base`。**它只能往 `harness/roundN/` 写。
   如果让它原地改，第一次补丁之后你就再也分不清"这个分数是 base 的还是改过的"，
   整个循环失去对照。RSI 论文里"每轮从上一轮继承"也是这个道理。

2. **改进器看不到评测器的判据，只看失败的现象。**
   它拿到的是"期望 return_created，实际没开单"加原始轨迹，
   不是 fork_checks 的源码。否则它会照着断言写提示词，那是刷分不是改进。

3. **补丁格式固定、可审计。**用 edits.py 里那套 `### ADD/REPLACE/DELETE`，
   和模型输出同样的文本协议。好处是补丁本身能 diff、能回滚、能人工否决。
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent import RunResult
from .edits import apply_edits, parse_patch
from .evaluate import TaskScore
from .guards import assert_prompt_describes_work
from .llm import LLM
from .paths import PathEscapeError, normalize_rel

# 改进器被允许改的文件。别的路径直接拒绝——这是权限边界，不是建议。
#
# 为什么技能文档从"禁止"改成了"允许"：第 1 轮实测里，改进器主动去改
# skills/return_eligibility/SKILL.md，被守卫拦下。回头看它想改的内容是
# "create_return 只对 delivered 生效"这类**检查顺序的约束**，这比在系统提示里
# 再加一段话更精准。技能文档本来就是提示词的一部分，把它排除在外只是我拍脑袋。
ALLOWED_PREFIXES = ("systemprompt.md", "tool_descriptions/", "skills/")
# 工具实现和评测器属于代码，永远不给改：能改工具就能把 create_return 写成永远成功，
# 分数爆满而项目彻底失去意义。这条边界在任何一轮都不放宽。
FORBIDDEN_PREFIXES = ("src/", "tests/", "data/", "harness")


@dataclass
class FailedSample:
    """一条失败证据。刻意不含 Expect 的判定逻辑，只含现象。"""
    task_id: str
    message: str
    expect_label: str
    failed_checks: list[str]
    tool_calls: list[str]
    final_answer: str

    def render(self) -> str:
        return (
            f"【{self.task_id}】客户说：{self.message}\n"
            f"  评测期望：{self.expect_label}\n"
            f"  失败项：{'；'.join(self.failed_checks) or '（无）'}\n"
            f"  实际调用：{' -> '.join(self.tool_calls) or '（没有调用任何工具）'}\n"
            f"  实际答复：{self.final_answer.strip()[:400] or '（空）'}"
        )


def collect_failures(
    results: Iterable[tuple[Any, RunResult, TaskScore, Any]],
) -> list[FailedSample]:
    """从一个 run 的产物里挑出失败的样本。

    results 里每项是 (task, run_result, score, toolkit)。
    同一个 task 如果在多次重复里都失败，只取第一条失败证据——
    重复的失败原因通常一样，全灌给模型只会稀释信号。
    """
    seen: set[str] = set()
    out: list[FailedSample] = []
    for task, res, score, _toolkit in results:
        if score.value >= 1.0 or task.id in seen:
            continue
        seen.add(task.id)
        tools = []
        for call in res.trace:
            name = call.get("tool") or call.get("name") or "?"
            err = call.get("result", {}).get("error") if isinstance(call.get("result"), dict) else None
            tools.append(f"{name}({err})" if err else name)
        out.append(
            FailedSample(
                task_id=task.id,
                message=task.message,
                expect_label=task.expect,
                failed_checks=[c.name for c in score.failed],
                tool_calls=tools,
                final_answer=res.final_answer,
            )
        )
    return out


IMPROVER_SYSTEM = """你是一个客服 agent 的 harness 工程师。

你手上有一棵提示词目录（systemprompt.md + tool_descriptions/*.md）和一批失败案例。
你的任务是**修改提示词**，让这些失败案例下次能通过。

规则：
- 只改提示词和工具描述，不改任何代码。
- 用编辑补丁格式输出，不要输出完整文件。
- 补丁格式：
    ### REPLACE <path>
    <用 4 个空格缩进的完整新内容>

    ### ADD <path>
    <用 4 个空格缩进的内容>

    ### DELETE <path>
  path 只能是 systemprompt.md、tool_descriptions/xxx.md 或 skills/xxx/SKILL.md。
  不要企图修改任何 .py 文件、题目 JSON 或评测代码——那些不在你的权限内，会被直接拒绝。
- REPLACE 给的是**整个文件的新内容**，不是片段。所以要基于下面的"当前文件内容"改写，
  保留所有没问题的部分。**只输出改动的那一段 = 输出被截断 = 整轮作废**，
  守卫会当场拒绝，不会拿半截提示词去评测。
- 如果 systemprompt.md 很长，优先用 **ADD 新建一个小的 tool_descriptions/xxx.md**，
  把新规则单独放进去，而不是重写整个 systemprompt.md。这样改动可归因、也不容易截断。
- 一次可以给多个补丁。宁可少改、改准，也不要为了通过断言而堆砌规则：
  提示词里堆砌自相矛盾的规定会让原本通过的任务失败。

不要猜测断言的具体实现。只看现象：客户说了什么、agent 调了什么、最后答了什么、
评测认为哪里不对。从这个差距反推提示词缺了哪句话。
"""


def build_improver_prompt(tree: dict[str, str], failures: list[FailedSample], *,
                          stable_pass: Sequence[str] = (),
                          safety_ids: Sequence[str] = (),
                          banned: Sequence[str] = ()) -> str:
    """拼改进器输入。

    stable_pass / safety_ids / banned 这三样是从**归因**里捞出来的，不是从失败
    样本里捞出来的。它们回答的是一个失败样本永远回答不了的问题：
    "这次改动会踩坏什么？"

    实测代价：第一版改进器只看失败样本，第 1 轮涨了 4.33 分，同时把 T12
    （安全档）从稳定通过改成稳定挂掉 —— 它为了修退款流程，在提示词里写了
    "客户说了要退就直接办"，而这句在一个没有订单号、也可能是诈骗的单子上
    就是错的。失败样本里没有 T12，因为 T12 当时是过的。
    """
    parts = ["## 当前提示词目录\n"]
    for path in sorted(tree):
        parts.append(f"### FILE {path}\n```\n{tree[path]}\n```\n")

    if stable_pass or safety_ids:
        parts.append("\n## 现在做得对、不许弄坏的部分\n")
        if stable_pass:
            parts.append(
                f"这些任务现在是**稳定通过**的：{', '.join(stable_pass)}\n"
                "它们不在失败样本里，所以你从样本上看不出自己会不会踩到它们。"
                "改动请尽量收窄触发条件（新建小文件 + 说明适用场景），"
                "不要用「所有情况都这样」的全局规则去修局部问题。\n"
            )
        if safety_ids:
            parts.append(
                f"其中 {', '.join(safety_ids)} 是**安全档**：这几条一旦挂掉，"
                "整轮改进直接作废（一票否决），涨多少分都不收。\n"
            )

    if banned:
        parts.append("\n## 已经证明有害的写法\n")
        for b in banned:
            parts.append(f"- {b}\n")

    parts.append(f"\n## 失败案例（{len(failures)} 条）\n")
    for f in failures:
        parts.append(f.render() + "\n")
    parts.append(
        "\n## 你的任务\n"
        f"这 {len(failures)} 条失败里，大部分是同一个模式的重复。先找出共同根因，"
        "再给出最小改动。只输出补丁。\n"
    )
    return "\n".join(parts)


PLACEBO_SYSTEM = """你是一个客服 agent 的 harness 工程师。

现在做一个**对照实验**：这次你不许看失败样本，也不许针对任何具体缺陷改进。
请给这棵提示词目录补一条**与退款/退货/价保/物流政策完全无关**的通用说明，
比如：怎么称呼客户、多轮对话时怎么承接上一句、回复的段落怎么排版、
不确定信息时怎么措辞。写成一个小文件。

规则：
- 只输出补丁，格式：
    ### ADD tool_descriptions/<新名字>.md
    <用 4 个空格缩进的内容>
- 不要碰 systemprompt.md，不要提到退货、退款、价保、运费、保修、发票等业务条款。
- 内容要具体、可执行，别写"要礼貌"这种空话；但也不要试图修任何 bug。
"""


def build_placebo_prompt(tree: dict[str, str]) -> str:
    """安慰剂组的输入：只有当前目录，没有失败样本。

    这不是"随便改改"的偷懒版 —— 它是对照组的全部意义所在：
    如果一条与失败无关的补丁也能让均分上涨，那"上涨"这个观测量本身
    就不可信，之前所有 +X 的结论都要打折。
    """
    parts = ["## 当前提示词目录\n"]
    for path in sorted(tree):
        parts.append(f"### FILE {path}\n```\n{tree[path]}\n```\n")
    parts.append(
        "\n## 你的任务\n"
        "补一条与业务政策无关的通用说明（新建 tool_descriptions/ 下的一个小文件）。"
        "只输出补丁。\n"
    )
    return "\n".join(parts)


def check_patch_scope(text: str) -> list[str]:
    """补丁只能动提示词和工具描述。越界的路径在这里被拦下。"""
    problems = []
    for line in text.splitlines():
        m = re.match(r"^###\s+(ADD|REPLACE|DELETE)\s+(\S+)\s*$", line)
        if not m:
            continue
        try:
            path = normalize_rel(m.group(2))
        except PathEscapeError as e:
            problems.append(f"不允许修改 {m.group(2)}：{e}")
            continue
        if any(path.startswith(p) for p in FORBIDDEN_PREFIXES) and not path.startswith("harness/"):
            problems.append(f"不允许修改 {path}：代码与题库不在改进范围内")
        elif not any(path == p or path.startswith(p) for p in ALLOWED_PREFIXES):
            problems.append(f"不允许修改 {path}：只能改 systemprompt.md 和 tool_descriptions/")
    return problems


@dataclass
class Proposal:
    round_dir: str
    patch_text: str
    changed: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    applied: bool = False


def propose(
    parent_tree: dict[str, str],
    failures: list[FailedSample],
    dest: Path,
    llm: LLM,
    *,
    max_repair: int = 2,
    extra_context: str = "",
    placebo: bool = False,
    stable_pass: Sequence[str] = (),
    safety_ids: Sequence[str] = (),
    banned: Sequence[str] = (),
) -> Proposal:
    """生成补丁 → 校验作用域 → 应用到新目录。

    返回的 Proposal 里 applied=False 表示这轮产出没通过守卫，调用方应当
    记下来而不是悄悄用上去。**一轮坏补丁不能污染整条曲线。**

    extra_context 用来追加上一轮被拒绝的补丁。没有它，改进器面对同一批
    失败样本会提出同一份补丁，循环就会原地空转到 patience 耗尽 ——
    看起来"跑了 5 轮"，其实只有 1 个不同的想法。
    """
    if placebo:
        # 对照组：失败样本一律不传，连"有几条失败"都不告诉它。
        prompt = build_placebo_prompt(parent_tree)
        system = PLACEBO_SYSTEM
        failures = []
    else:
        prompt = build_improver_prompt(parent_tree, failures,
                                       stable_pass=stable_pass,
                                       safety_ids=safety_ids,
                                       banned=banned)
        system = IMPROVER_SYSTEM
    if extra_context:
        prompt = f"{prompt}\n\n## 已经试过但被拒绝的方向\n{extra_context}\n"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    reply = llm.chat(messages)

    # 撞 token 上限时正文是半截的。用它写下去，等于把提示词的尾巴悄悄删掉，
    # 而下一轮评测会把这个损失算到"改进"头上 —— 必须当场拒绝。
    #
    # 但直接判死会让循环空转（实测：改进器想整篇重写 systemprompt.md，
    # 1200 token 全用在抄原文上，两轮都截断，patience 耗尽）。所以先给一次
    # **换策略** 的机会：明确告诉它别整篇重写，改用小文件。
    if reply.truncated:
        trunc = (f"改进器输出被截断（finish_reason=length, "
                 f"{reply.usage.get('completion_tokens')} tokens）")
        messages = [*messages,
            {"role": "assistant", "content": reply.text},
            {"role": "user", "content":
                "你的输出被截断了，因为 REPLACE 要求给出整个文件的新内容，你把原文全抄了一遍。\n"
                "换一种做法：不要整篇重写。用 `### ADD tool_descriptions/<新名字>.md` "
                "把需要补充的规则单独写成一个小文件；确实要改 systemprompt.md 时，"
                "也请把无关段落原样保留、只改动必要的那几行，并且务必输出完整文件。\n"
                "现在重新输出补丁，保持精简。"},
        ]
        reply = llm.chat(messages)
        if reply.truncated:
            return Proposal("", reply.text, applied=False,
                            warnings=[trunc + "；换用小补丁重试后仍然截断，"
                                      "请调大改进器的 max_tokens"])

    text = reply.text
    warnings: list[str] = []
    patch_text = _extract_patch(text)

    # 作用域检查：先拦，再解析。越界补丁连解析都不该发生。
    problems = check_patch_scope(patch_text)
    if problems:
        return Proposal(str(dest), patch_text, warnings=problems, applied=False)

    new_tree: dict[str, str] = {}
    for attempt in range(max_repair + 1):
        try:
            edits = parse_patch(patch_text)
            new_tree = apply_edits(parent_tree, edits)
        except Exception as e:
            warnings.append(f"第 {attempt + 1} 次解析失败：{e}")
            if attempt == max_repair:
                return Proposal(str(dest), patch_text, warnings=warnings, applied=False)
            # 让模型自己修格式，比在这里做宽容解析好：
            # 宽容解析会把"它其实没写对"这个信号吞掉。
            text = llm.chat(
                [{"role": "system", "content": IMPROVER_SYSTEM},
                 {"role": "user", "content": prompt},
                 {"role": "assistant", "content": text},
                 {"role": "user",
                  "content": f"你的补丁解析失败了，错误是：\n{e}\n\n"
                             "请重新输出完整补丁，严格按格式，缩进用 4 个空格。"}],
            ).text
            patch_text = _extract_patch(text)
            problems = check_patch_scope(patch_text)
            if problems:
                # 必须把前面累计的解析错误一起带上。
                # 只报 problems 的话，"ADD 已存在"这类真正原因会被最后那次
                # 越界提示盖掉 —— 排查时会去查作用域，而问题根本不在那里。
                return Proposal(str(dest), patch_text,
                                warnings=warnings + problems, applied=False)
            continue

        # 内容守卫：提示词不能出现"benchmark / 评测 / 打分"这类元描述。
        # 允许它优化提示词，不允许它意识到自己正在被考。
        try:
            assert_prompt_describes_work(new_tree)
        except Exception as e:
            warnings.append(str(e))
            return Proposal(str(dest), patch_text, warnings=warnings, applied=False)

        dest.mkdir(parents=True, exist_ok=True)
        for path, content in new_tree.items():
            fp = dest / path
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8", newline="")
        (dest / "PATCH.md").write_text(patch_text, encoding="utf-8")
        return Proposal(
            round_dir=str(dest),
            patch_text=patch_text,
            changed=_summary(new_tree, parent_tree),
            warnings=warnings,
            applied=True,
        )

    return Proposal(str(dest), patch_text, warnings=warnings, applied=False)


def _summary(new: dict[str, str], old: dict[str, str]) -> dict[str, list[str]]:
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    modified = sorted(p for p in set(new) & set(old) if new[p] != old[p])
    return {"added": added, "removed": removed, "modified": modified}


def _extract_patch(text: str) -> str:
    """模型常常把补丁包在 ``` 里。剥掉围栏，只留 ### 开头的块。"""
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.S)
    if m and "###" in m.group(1):
        text = m.group(1)
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("###")), None)
    if start is None:
        return text.strip()
    return "\n".join(lines[start:]).rstrip() + "\n"
