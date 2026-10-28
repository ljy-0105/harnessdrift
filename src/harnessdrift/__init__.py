"""HarnessDrift — Propose / Probe / Compose 自改进骨架。

第 0 步只固化三条契约，不含任何业务逻辑、不调用任何模型：

1. ``paths``     —— 树里的 key 永远是 posix 反斜杠归一化后的相对路径
2. ``edits``     —— 提案只输出 diff（omitted is inherited），省略即继承
3. ``guards``    —— 反作弊三件套：禁用词 / 评分标记 / 答案文件屏蔽
"""

from .edits import EditError, apply_edits, parse_patch
from .guards import (
    LeakError,
    PromptLeakError,
    assert_no_grader_verdicts,
    assert_prompt_describes_work,
    is_leaked,
)
from .paths import PathEscapeError, normalize_rel, read_tree, write_tree

__all__ = [
    "EditError",
    "LeakError",
    "PathEscapeError",
    "PromptLeakError",
    "apply_edits",
    "assert_no_grader_verdicts",
    "assert_prompt_describes_work",
    "is_leaked",
    "normalize_rel",
    "parse_patch",
    "read_tree",
    "write_tree",
]

__version__ = "0.1.0"