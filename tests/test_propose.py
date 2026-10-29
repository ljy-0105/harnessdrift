"""第 2 步的守卫测试。

这些测试的重点不是"补丁能不能用"，而是**坏补丁会不会被拦住**。
一个自改进系统，最贵的错误不是这轮没进步，而是悄悄用了一份坏补丁、
把分数抬高到无法归因。所以这里每一条都是"必须拒绝"的用例。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from harnessdrift.llm import LLMReply
from harnessdrift.propose import _extract_patch, check_patch_scope

BASE_TREE = {
    "systemprompt.md": "# 你是客服\n\n## 原则\n1. 先查后答。\n",
    "tool_descriptions/lookup_guidance.md": "查订单的说明。\n",
}


# ---------------- 作用域 ----------------

def test_code_files_are_forbidden():
    patch = "### REPLACE src/harnessdrift/tools.py\n    def x(): pass\n"
    assert check_patch_scope(patch), "改代码必须被拦"


def test_skill_docs_are_allowed():
    """第 1 轮实测后放宽：改进器主动想去改 SKILL.md，而且改得对。

    技能文档本来就是提示词的一部分，把它排除只是我拍脑袋的决定。
    """
    patch = "### REPLACE skills/return_eligibility/SKILL.md\n    x\n"
    assert check_patch_scope(patch) == []


def test_code_and_task_files_stay_forbidden():
    """代码和题库永不放宽：能改工具就能把 create_return 写成永远成功。"""
    for path in ("src/harnessdrift/tools.py", "data/tasks_ecommerce_after_sales.json",
                 "tests/test_contracts.py"):
        assert check_patch_scope(f"### REPLACE {path}\n    x\n"), f"{path} 必须被拦"


def test_prompt_and_tool_description_are_allowed():
    patch = (
        "### REPLACE systemprompt.md\n    # x\n"
        "### ADD tool_descriptions/new_rule.md\n    y\n"
    )
    assert check_patch_scope(patch) == []


def test_windows_separators_are_normalized_before_check():
    """模型可能吐出反斜杠路径。归一化要在判断之前做，否则能绕过。"""
    patch = "### REPLACE src\\harnessdrift\\tools.py\n    x\n"
    assert check_patch_scope(patch), "反斜杠路径也必须被识别为越界"


def test_traversal_is_rejected_before_writing():
    patch = "### ADD tool_descriptions/../../escaped.md\n    x\n"
    assert check_patch_scope(patch), ".. 逃逸必须在作用域检查阶段被拒绝"


# ---------------- 提取 ----------------

def test_extract_strips_markdown_fence():
    raw = "好的，补丁如下：\n```markdown\n### REPLACE systemprompt.md\n    # x\n```\n结束"
    out = _extract_patch(raw)
    assert out.startswith("### REPLACE")
    assert "```" not in out


def test_extract_drops_prose_before_first_header():
    raw = "我先分析一下：这里缺一条规则。\n### REPLACE systemprompt.md\n    # x\n"
    assert _extract_patch(raw).startswith("### REPLACE")


# ---------------- 守卫：元描述 ----------------

def test_prompt_that_mentions_benchmark_is_rejected(tmp_path: Path):
    """提示词里出现"评测/benchmark"说明它在对着考试写答案，不是在做工作。"""
    from harnessdrift.guards import assert_prompt_describes_work

    bad = {"systemprompt.md": "# 客服\n\n为通过 benchmark 评测，必须调用 create_return。\n"}
    with pytest.raises(Exception):
        assert_prompt_describes_work(bad)


# ---------------- 端到端：假的 LLM ----------------

class _FakeLLM:
    """只实现 propose 用到的那一小块。"""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, temperature=None, **kw):
        self.calls += 1
        text = self.replies.pop(0) if self.replies else ""
        return LLMReply(text=text, tool_calls=[], raw={}, usage={})


def _failures():
    from harnessdrift.propose import FailedSample

    return [
        FailedSample(
            task_id="T02", message="这条裤子想退掉",
            expect_label="return_created",
            failed_checks=["期望动作=return_created", "调用了 create_return"],
            tool_calls=["lookup_order", "get_return_policy"],
            final_answer="可以退，需要确认吗？",
        )
    ]


def test_propose_writes_new_round_and_never_touches_parent(tmp_path: Path):
    from harnessdrift.propose import propose

    new_prompt = "# 你是客服\n\n## 原则\n1. 先查后答。\n2. 客户说想退就是委托。\n"
    llm = _FakeLLM(["### REPLACE systemprompt.md\n" + "\n".join(
        "    " + ln for ln in new_prompt.splitlines()) + "\n"])
    dest = tmp_path / "round1"

    prop = propose(BASE_TREE, _failures(), dest, llm)

    assert prop.applied
    assert prop.warnings == []
    assert (dest / "systemprompt.md").read_text(encoding="utf-8") == new_prompt
    assert (dest / "PATCH.md").exists()
    # 父树（内存里的 base）一个字都没动
    assert BASE_TREE["systemprompt.md"] == "# 你是客服\n\n## 原则\n1. 先查后答。\n"
    # 没提到的文件被继承，不是丢失
    assert (dest / "tool_descriptions/lookup_guidance.md").exists()


def test_propose_rejects_out_of_scope_patch_without_writing(tmp_path: Path):
    from harnessdrift.propose import propose

    llm = _FakeLLM(["### REPLACE src/harnessdrift/tools.py\n    def hack(): pass\n"])
    dest = tmp_path / "round1"

    prop = propose(BASE_TREE, _failures(), dest, llm)

    assert not prop.applied
    assert any("不允许修改" in w for w in prop.warnings)
    assert not (dest / "systemprompt.md").exists(), "越界补丁不该留下任何产物"


def test_propose_rejects_traversal_without_escaping_destination(tmp_path: Path):
    from harnessdrift.propose import propose

    llm = _FakeLLM(["### ADD tool_descriptions/../../escaped.md\n    outside\n"])
    dest = tmp_path / "run" / "round1"

    prop = propose(BASE_TREE, _failures(), dest, llm)

    assert not prop.applied
    assert not (tmp_path / "run" / "escaped.md").exists()
    assert not (tmp_path / "escaped.md").exists()


def test_propose_keeps_base_untouched_even_on_bad_parse(tmp_path: Path):
    from harnessdrift.propose import propose

    # 第一次给一个解析不了的补丁，第二次重试还是坏的 -> 必须放弃而不是猜
    llm = _FakeLLM(["### REPLACE systemprompt.md\nnot-indented\n"] * 5)
    prop = propose(BASE_TREE, _failures(), tmp_path / "round1", llm)
    assert not prop.applied
    assert prop.warnings


def test_propose_retries_format_then_succeeds(tmp_path: Path):
    from harnessdrift.propose import propose

    good = "### REPLACE systemprompt.md\n" + "\n".join(
        "    " + ln for ln in ["# 你是客服", "", "好。", ""])
    llm = _FakeLLM(["### REPLACE systemprompt.md\n弄坏了\n", good])
    prop = propose(BASE_TREE, _failures(), tmp_path / "round1", llm)
    assert prop.applied
    assert llm.calls >= 2, "格式错误应当让模型自己修，而不是我们宽容解析"

def test_truncated_output_is_rejected_not_saved(tmp_path: Path):
    """改进器撞上 max_tokens 时，正文是半截的。

    如果照单全收，提示词的尾巴会被悄悄删掉，而下一轮评测会把损失算到"改进"头上。
    这个 bug 真的发生过：round1 的 systemprompt.md 最后一句是半句话。
    """
    from harnessdrift.propose import propose

    class _TruncatingLLM:
        def chat(self, messages, **kw):
            return LLMReply(
                text="### REPLACE systemprompt.md\n    # 半句话",
                tool_calls=[], raw={}, usage={"completion_tokens": 1200},
                finish_reason="length",
            )

    prop = propose(BASE_TREE, _failures(), tmp_path / "round1", _TruncatingLLM())
    assert not prop.applied
    assert any("截断" in w for w in prop.warnings)
    assert not (tmp_path / "round1" / "systemprompt.md").exists()


def test_truncation_retries_with_small_patch_before_giving_up(tmp_path: Path):
    """截断后必须给一次换策略的机会，否则循环会白跑。

    实测：改进器想整篇重写 systemprompt.md，1200 token 全花在抄原文上，
    连续两轮截断，patience 耗尽，曲线一动不动。光"拒绝"是不够的 ——
    拒绝之后要告诉它改用小补丁，它才动得起来。
    """
    from harnessdrift.propose import propose

    class _RetryLLM:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def chat(self, messages, **kw):
            self.calls.append(messages[-1]["content"])
            if len(self.calls) == 1:
                return LLMReply(text="### REPLACE systemprompt.md\n    # 半句话",
                                usage={"completion_tokens": 1200}, finish_reason="length")
            return LLMReply(
                text="### ADD tool_descriptions/refund_window.md\n    无理由退货窗口是 7 天。\n",
                finish_reason="stop",
            )

    llm = _RetryLLM()
    prop = propose(BASE_TREE, _failures(), tmp_path / "round1", llm)
    assert len(llm.calls) == 2, "截断后必须重试一次"
    assert "小文件" in llm.calls[1], "重试要明确要求换小补丁"
    assert prop.applied, "换成小补丁后应当通过"
    assert "tool_descriptions/refund_window.md" in prop.changed["added"]


def test_prompt_warns_about_what_must_not_break():
    """归因喂进提案端：稳定通过 + 安全档 + 历史踩雷，三样都要出现在提示里。

    这三样都来自"什么现在是对的"，而失败样本只说得出"什么现在是错的"。
    少了它们，改进器会拿全局规则去修局部问题 —— T12 就是这么坏的。
    """
    from harnessdrift.propose import FailedSample, build_improver_prompt

    tree = {"systemprompt.md": "# 你是客服\n"}
    fail = FailedSample(task_id="T02", message="要退货", expect_label="期望动作=return_created",
                        failed_checks=["调用了 create_return"], tool_calls=["lookup_order"],
                        final_answer="好的")
    p = build_improver_prompt(tree, [fail], stable_pass=["T01", "T12", "T27"],
                              safety_ids=["T12", "T27"],
                              banned=["第 1 轮改了 systemprompt.md 之后，T12 从稳定通过变成稳定挂掉，这一轮因此被拒。"])
    assert "不许弄坏" in p
    assert "T01, T12, T27" in p
    assert "T12, T27 是**安全档**" in p
    assert "一票否决" in p
    assert "第 1 轮改了 systemprompt.md" in p
    # 失败样本本身仍然要在，不然就变成另一种信息缺失
    assert "【T02】" in p


def test_placebo_prompt_has_no_such_warnings():
    """对照组不能偷偷享受这些提示 —— 否则安慰剂组不再是对照组。"""
    from harnessdrift.propose import build_placebo_prompt

    p = build_placebo_prompt({"systemprompt.md": "# 你是客服\n"})
    assert "不许弄坏" not in p and "安全档" not in p
