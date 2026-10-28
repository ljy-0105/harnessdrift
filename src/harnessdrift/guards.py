"""反作弊三件套 —— 全部来自 Harness? 源码，不是想象出来的。

对应关系：

- ``assert_prompt_describes_work``  ~ ``core/spec.py`` 的 ``BenchPrompts.__post_init__``
      描述"工作本身"的提示词里出现 benchmark/evaluation/grader/rubric/score
      就直接抛 ValueError。它强制提示词只描述工作，不描述度量。
- ``assert_no_grader_verdicts``     ~ ``core/context.py`` 的
      ``assert_tree_has_no_grader_verdicts``。命中即拒，理由是多轮之后
      带评分字样的文件"永远不能再作为证据服务给后续任务"。
- ``is_leaked``                      ~ ``spec.leak_globs``。执行时把答案类文件挡在
      被评测的 agent 视野之外。

这三条在 HarnessDrift 里是**跑在每次提案前后**的断言，不是文档里的建议。
"""

from __future__ import annotations

import fnmatch
import re
from typing import Iterable, Mapping, Sequence

# 描述"工作"的文本里不许出现的度量词汇
MEASUREMENT_WORDS: tuple[str, ...] = (
    "benchmark",
    "evaluation",
    "grader",
    "rubric",
    "score",
)

# harness 文件里不许出现的评分标记（文件名和正文都会扫）
GRADER_VERDICT_MARKERS: tuple[str, ...] = (
    "grader",
    "rubric",
    "score.json",
    "raw_scores",
    "expected_answer",
    "reference_answer",
    "judge_verdict",
)

# 扫正文的文件后缀；二进制与模板不扫
_SCAN_SUFFIXES: tuple[str, ...] = (".md", ".txt", ".py", ".ts", ".js", ".json", ".yaml", ".yml")


class PromptLeakError(ValueError):
    """提示词在描述度量方式，而不是工作本身。"""


class LeakError(ValueError):
    """harness 树里出现了不该出现的东西。"""


def assert_prompt_describes_work(fields: Mapping[str, str]) -> None:
    """检查一组"面向 agent 的提示词字段"。

    ``fields`` 的 key 是字段名，value 是文本。整个字段都在检查范围内 ——
    Harness? 连字段名一起报错，这里也一样，方便一次定位。
    """
    for name, value in fields.items():
        low = value.lower()
        for bad in MEASUREMENT_WORDS:
            if bad in low:
                raise PromptLeakError(
                    f"{name} must describe the WORK, not its measurement "
                    f"(found {bad!r}). Say what the practitioner needs instead."
                )


def assert_no_grader_verdicts(
    tree: Mapping[str, str],
    markers: Sequence[str] = GRADER_VERDICT_MARKERS,
) -> None:
    """在树被物化 / 执行 / 存储**之前**调用。"""
    for rel, body in sorted(tree.items()):
        low_rel = rel.lower()
        for m in markers:
            if m in low_rel:
                raise LeakError(
                    f"harness file {rel!r} has grader-verdict marker {m!r} in its NAME; "
                    "rename it -- files carrying these strings cannot be served as "
                    "evidence to later tasks"
                )
        if not rel.lower().endswith(_SCAN_SUFFIXES):
            continue
        low = body.lower()
        for m in markers:
            if m in low:
                where = low.index(m)
                snippet = body[max(0, where - 60) : where + 120]
                raise LeakError(
                    f"harness file {rel!r} contains grader-verdict marker {m!r} "
                    f"at offset {where} ({snippet!r}); remove or rename that string"
                )


def is_leaked(rel: str, leak_globs: Iterable[str]) -> bool:
    """执行沙箱里的答案文件判定。

    用 fnmatch 而不是正则：glob 更好写、更好读，也更不容易写出灾难性回溯。
    同时按 basename 匹配，避免 ``answers/x.md`` 这种深层路径绕过。
    """
    base = rel.replace("\\", "/").rsplit("/", 1)[-1]
    for pattern in leak_globs:
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(base, pattern):
            return True
    return False


def redact(body: str, markers: Sequence[str] = GRADER_VERDICT_MARKERS) -> str:
    """把命中的标记替换成占位符，用于"必须记录但不能传播"的场景。

    只在日志里用；被 redact 过的内容**不能**再作为证据喂给改进器。
    """
    out = body
    for m in markers:
        out = re.sub(re.escape(m), f"[redacted:{m}]", out, flags=re.IGNORECASE)
    return out