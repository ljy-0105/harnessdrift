"""第 0 步验收测试：三条契约必须成立。

跑法：``.venv/Scripts/python.exe -m pytest tests -q``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harnessdrift.edits import (  # noqa: E402
    EditError,
    apply_edits,
    diff_summary,
    parse_patch,
    render_patch,
)
from harnessdrift.guards import (  # noqa: E402
    LeakError,
    PromptLeakError,
    assert_no_grader_verdicts,
    assert_prompt_describes_work,
    is_leaked,
)
from harnessdrift.paths import (  # noqa: E402
    PathEscapeError,
    normalize_rel,
    read_tree,
    replace_atomically,
    tree_hash,
    write_tree,
)

BASE = {
    "systemprompt.md": "You answer customer questions about orders.\n",
    "tool_descriptions/lookup_guidance.md": "Prefer order_id over free text.\n",
    "skills/none/SKILL.md": "placeholder\n",
}


# ---------------------------------------------------------------- 契约 1: 路径


def test_backslash_paths_normalize_to_posix():
    # 这就是 Harness? 在 Windows 上静默不命中的根因
    assert normalize_rel(r"scripts\summarize.py") == "scripts/summarize.py"
    assert normalize_rel("./tool_descriptions/a.md") == "tool_descriptions/a.md"
    assert normalize_rel("a//b/./c.md") == "a/b/c.md"


def test_traversal_and_absolute_paths_are_rejected():
    for bad in ("../evil.md", "a/../../evil.md", "/etc/passwd", "C:/Windows/x.md"):
        with pytest.raises(PathEscapeError):
            normalize_rel(bad)


def test_duplicate_after_normalization_is_an_error(tmp_path):
    # 用显式列表：字面量 {"a.md": ..., "a.md": ...} 在 Python 里本来就被合并了，
    # 那样的测试永远绿，什么也没验证。
    with pytest.raises(PathEscapeError):
        write_tree([("a.md", "1"), (r"a.md", "2")], tmp_path / "dup")


def test_write_tree_uses_lf_and_read_tree_roundtrips(tmp_path):
    target = tmp_path / "tree"
    write_tree(BASE, target)
    raw = (target / "systemprompt.md").read_bytes()
    assert b"\r\n" not in raw, "CRLF would break content hashing across platforms"
    assert read_tree(target) == BASE


def test_tree_hash_is_order_and_separator_independent():
    reordered = dict(reversed(list(BASE.items())))
    backslashed = {k.replace("/", "\\"): v for k, v in BASE.items()}
    assert tree_hash(BASE) == tree_hash(reordered) == tree_hash(backslashed)


def test_write_tree_clean_removes_stale_files(tmp_path):
    target = tmp_path / "tree"
    write_tree(BASE, target)
    write_tree({"systemprompt.md": BASE["systemprompt.md"]}, target)
    assert not (target / "skills").exists(), "stale skill would keep silently firing"


def test_replace_atomically_replaces_fcntl_locking(tmp_path):
    f = tmp_path / "summary.jsonl"
    replace_atomically(f, '{"round": 1}\n')
    replace_atomically(f, '{"round": 2}\n')
    assert f.read_text(encoding="utf-8") == '{"round": 2}\n'
    assert not list(tmp_path.glob("*.tmp*")), "temp file leaked"


# ------------------------------------------------------ 契约 2: diff / 继承


def test_omitted_is_inherited():
    edits = parse_patch("### REPLACE systemprompt.md\n    shorter.\n")
    out = apply_edits(BASE, edits)
    assert out["systemprompt.md"] == "shorter.\n"
    # 没提到的两个文件原样继承
    assert out["tool_descriptions/lookup_guidance.md"] == BASE["tool_descriptions/lookup_guidance.md"]
    assert out["skills/none/SKILL.md"] == BASE["skills/none/SKILL.md"]


def test_apply_edits_does_not_mutate_input():
    before = dict(BASE)
    apply_edits(BASE, parse_patch("### DELETE skills/none/SKILL.md\n"))
    assert BASE == before


def test_add_existing_replace_missing_delete_missing_all_raise():
    with pytest.raises(EditError):
        apply_edits(BASE, parse_patch("### ADD systemprompt.md\n    x\n"))
    with pytest.raises(EditError):
        apply_edits(BASE, parse_patch("### REPLACE nope.md\n    x\n"))
    with pytest.raises(EditError):
        apply_edits(BASE, parse_patch("### DELETE nope.md\n"))


def test_malformed_proposals_raise_instead_of_silently_skipping():
    for bad in (
        "",                                  # 空提案
        "hello\n### ADD a.md\n    x\n",      # 头之前有内容
        "### PATCH a.md\n    x\n",           # 未知动作
        "### ADD a.md\n    x\n### ADD a.md\n    y\n",  # 同一路径两次
        "### DELETE a.md\n    body\n",       # DELETE 带 body
    ):
        with pytest.raises(EditError):
            parse_patch(bad)


def test_unindented_body_is_rejected():
    """第 0 步这里测的是"缩进和顶格都接受、结果一致"。

    第 2 步发现那个宽松约定是个洞：`### REPLACE systemprompt.md\\nnot-indented`
    会解析成功并把整个文件替换成一行垃圾，而自改进会把这次替换记成"成功的改动"。
    **静默接受坏补丁比直接报错危险得多**，所以改成必须缩进。
    """
    a = parse_patch("### ADD a.md\n    line1\n    line2\n")
    assert a[0].body == "line1\nline2\n"

    with pytest.raises(EditError):
        parse_patch("### ADD a.md\nline1\nline2\n")


def test_blank_lines_inside_body_are_still_allowed():
    """空行不该被当成"没缩进"——提示词里到处是空行。"""
    a = parse_patch("### ADD a.md\n    line1\n\n    line2\n")
    assert a[0].body == "line1\n\nline2\n"


def test_diff_summary_and_render_patch_roundtrip():
    after = apply_edits(
        BASE,
        parse_patch(
            "### ADD skills/refund/SKILL.md\n    handle refunds\n"
            "### REPLACE systemprompt.md\n    new prompt\n"
            "### DELETE skills/none/SKILL.md\n"
        ),
    )
    d = diff_summary(BASE, after)
    assert d == {
        "added": ["skills/refund/SKILL.md"],
        "removed": ["skills/none/SKILL.md"],
        "modified": ["systemprompt.md"],
    }
    # 渲染出的补丁重放后必须得到同一棵树 —— 日志因此可回滚、可复现
    assert apply_edits(BASE, parse_patch(render_patch(BASE, after))) == after


# ---------------------------------------------------------------- 契约 3: 反作弊


def test_prompt_cannot_describe_its_own_measurement():
    assert_prompt_describes_work({"describe_work": "Resolve customer refund requests."})
    for word in ("benchmark", "Evaluation", "GRADER", "rubric", "score"):
        with pytest.raises(PromptLeakError):
            assert_prompt_describes_work({"describe_work": f"maximise the {word} here"})


def test_grader_verdicts_are_rejected_by_name_and_by_body():
    with pytest.raises(LeakError):
        assert_no_grader_verdicts({"notes/rubric_v2.md": "ok"})
    with pytest.raises(LeakError):
        assert_no_grader_verdicts({"notes/postmortem.md": "our grader gave 3/5"})
    # 干净的表应当通过
    assert_no_grader_verdicts(BASE)


def test_grader_verdict_scan_is_case_insensitive():
    with pytest.raises(LeakError):
        assert_no_grader_verdicts({"a.md": "Raw_Scores for round 2"})


def test_is_leaked_matches_glob_and_basename():
    globs = ("answers/*", "*.expected", "*solution*")
    assert is_leaked("answers/q01.md", globs)
    assert is_leaked("deep/nested/q01.expected", globs)
    assert is_leaked("docs/the_solution.md", globs)
    assert not is_leaked("skills/refund/SKILL.md", globs)