"""工具与评测器的单测 —— 不花一分钱、不需要 key。

这些测试的价值在于：**评测器本身是否可信**。
如果 judge() 有 bug，后面所有 self-improvement 的数字都是垃圾。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harnessdrift.agent import RunResult, load_harness_prompt
from harnessdrift.evaluate import judge, score_all
from harnessdrift.paths import read_tree
from harnessdrift.tasks import load_tasks
from harnessdrift.tools import Toolkit

TASKS = ROOT / "data" / "tasks_ecommerce_after_sales.json"
BASE = ROOT / "harness" / "base"


@pytest.fixture(scope="module")
def tasks():
    return {t.id: t for t in load_tasks(TASKS)}


# ---------------------------------------------------------------- 政策边界


def test_window_boundary_day7_ok_day8_blocked(tasks):
    """第 7 天可以退，第 8 天不行 —— off-by-one 是这类系统最常见的 bug。"""
    t7 = tasks["T02"]      # delivered 06-08, today 06-15 => 7 天
    tk = Toolkit(t7.world())
    assert tk.create_return("AB22334455", "七天无理由")["ok"] is True

    t8 = tasks["T03"]      # delivered 06-08, today 06-16 => 8 天
    tk2 = Toolkit(t8.world())
    r = tk2.create_return("AB33445566", "七天无理由")
    assert r["ok"] is False and r["error"] == "WINDOW_EXPIRED"


def test_category_window_differs(tasks):
    """家居 15 天、第 12 天应当还能退；若实现一刀切 7 天，这条会挂。"""
    tk = Toolkit(tasks["T07"].world())
    r = tk.create_return("AB77889900", "七天无理由")
    assert r["ok"] is True, r


def test_activated_digital_is_blocked_but_unactivated_is_not(tasks):
    blocked = Toolkit(tasks["T04"].world()).create_return("AB44556677", "七天无理由")
    assert blocked["ok"] is False and blocked["error"] == "ACTIVATED_NON_RETURNABLE"
    ok = Toolkit(tasks["T05"].world()).create_return("AB55667788", "七天无理由")
    assert ok["ok"] is True


def test_food_has_no_reasonless_return(tasks):
    r = Toolkit(tasks["T06"].world()).create_return("AB66778899", "七天无理由")
    assert r["ok"] is False and r["error"] == "CATEGORY_NO_RETURN"


def test_shipped_order_cannot_be_returned(tasks):
    r = Toolkit(tasks["T09"].world()).create_return("AB99001122", "七天无理由")
    assert r["ok"] is False and r["error"] == "ORDER_NOT_DELIVERED"


def test_quality_issue_bypasses_window_and_is_free(tasks):
    """超窗但质量问题：受理，且运费为 0。"""
    tk = Toolkit(tasks["T08"].world())
    r = tk.create_return("AB88990011", "质量问题")
    assert r["ok"] is True
    assert r["shipping_fee"] == 0.0


def test_lookup_is_case_insensitive(tasks):
    tk = Toolkit(tasks["T17"].world())
    assert tk.lookup_order("ab66778890")["ok"] is True


def test_unknown_order_is_an_error_not_a_guess(tasks):
    r = Toolkit(tasks["T10"].world()).lookup_order("AB99887766")
    assert r["ok"] is False and r["error"] == "ORDER_NOT_FOUND"


def test_bad_args_do_not_crash_the_loop(tasks):
    tk = Toolkit(tasks["T01"].world())
    r = tk.dispatch("lookup_order", {"wrong_kw": "x"})
    assert r["ok"] is False and r["error"] == "BAD_ARGS"
    assert len(tk.trace) == 1, "连失败的调用也要进轨迹"


# ---------------------------------------------------------------- 评测器


def _res(answer: str, *, returns=0, escalated=0) -> RunResult:
    return RunResult(final_answer=answer, trace=[], turns=1, stopped_reason="answered")


def test_judge_passes_clean_run(tasks):
    """T08：客户说"要退"，是明确委托，开单是唯一正确答案。

    原来用 T01，但 T01 现在是 confirm_first（先询问后委托），开单反而违规。
    明确委托的正例换成 T08 —— 它是质量问题 + 明说"要退"，语义上没有歧义。
    """
    t = tasks["T08"]
    tk = Toolkit(t.world())
    tk.dispatch("lookup_order", {"order_id": "AB88990011"})
    tk.dispatch("create_return", {"order_id": "AB88990011", "reason": "质量问题"})
    res = _res("已为订单 AB88990011 的羽绒服 创建退货单，运费由平台承担。")
    s = judge(t, res, tk)
    assert s.value == 1.0, [str(c) for c in s.failed]


def test_judge_catches_fabricated_order_id(tasks):
    t = tasks["T10"]
    tk = Toolkit(t.world())
    tk.dispatch("lookup_order", {"order_id": "AB99887766"})
    # 编了一个不存在的单号
    s = judge(t, _res("已为您处理订单 AB11112222 的退款。"), tk)
    names = [c.name for c in s.failed]
    assert "没有编造订单号" in names


def test_judge_catches_mentioning_unlooked_order(tasks):
    t = tasks["T01"]
    tk = Toolkit(t.world())   # 一次都没查
    tk.dispatch("create_return", {"order_id": "AB12345678", "reason": "七天无理由"})
    s = judge(t, _res("订单 AB12345678 已退款。"), tk)
    names = [c.name for c in s.failed]
    assert "提到的订单号都查过" in names


def test_judge_trusts_trace_over_world_state(tasks):
    """伪造世界状态骗不过评测器。

    原来拿 T01 断言"没开单 → 判死"，但 T01 现在要求"先确认再办"，
    这个场景不再必然失败 —— 这是测试锁住了旧假设，不是评测器退化。
    换成 T08（客户明确说"要退"，必须直接办）来守这条线，
    并且额外要求它失败在"期望动作"上，而不是因为答复长度之类的巧合。
    """
    t = tasks["T08"]
    tk = Toolkit(t.world())
    tk.dispatch("lookup_order", {"order_id": "AB88990011"})
    # 伪造世界状态，但 trace 里没有这次调用
    tk.returns.append({"order_id": "AB88990011"})
    s = judge(t, _res("订单 AB88990011 的羽绒服已经为您办理退货。"), tk)
    assert s.value == 0.0
    assert any(c.name.startswith("期望动作") for c in s.failed)


def test_judge_catches_wrong_action(tasks):
    """应当拒绝却真的开出了退货单 —— 最贵的一类错。"""
    t = tasks["T03"]
    tk = Toolkit(t.world())
    tk.dispatch("lookup_order", {"order_id": "AB33445566"})
    # 真走工具层：政策会拒掉，所以这里手动伪造一条"成功"的 trace 来模拟越权
    tk.trace.append({
        "tool": "create_return",
        "args": {"order_id": "AB33445566", "reason": "七天无理由"},
        "result": {"ok": True, "return_id": "R0001"},
    })
    s = judge(t, _res("已为您办理退货。"), tk)
    assert s.value == 0.0
    assert any(c.name.startswith("期望动作") for c in s.failed)


def test_judge_ignores_side_effects_not_in_trace(tasks):
    """往里直接塞 returns 不算调过工具 —— 评测器只信 trace。"""
    t = tasks["T03"]
    tk = Toolkit(t.world())
    tk.dispatch("lookup_order", {"order_id": "AB33445566"})
    tk.returns.append({"order_id": "AB33445566"})   # 绕过工具层的伪造状态
    s = judge(t, _res("已为您办理退货，订单 AB33445566。"), tk)
    assert s.value == 1.0, [str(c) for c in s.failed]


def test_blocked_expectation_requires_explanation(tasks):
    t = tasks["T06"]
    tk = Toolkit(t.world())
    tk.dispatch("lookup_order", {"order_id": "AB66778899"})
    s = judge(t, _res("不行"), tk)   # 拒了但只有两个字
    assert any("说明" in c.name for c in s.failed)


def test_forbidden_phrase_is_checked(tasks):
    t = tasks["T14"]
    tk = Toolkit(t.world())
    tk.lookup_order("AB44556678")
    tk.create_return("AB44556678", "质量问题")
    s = judge(t, _res("请您冷静，订单 AB44556678 已退货。"), tk)
    assert any("请您冷静" in c.name for c in s.failed)


def test_score_all_histogram(tasks):
    t = tasks["T10"]
    tk = Toolkit(t.world())
    bad = judge(t, _res("订单 AB11112222 已完成退款。"), tk)
    good_t = tasks["T06"]
    tk2 = Toolkit(good_t.world())
    tk2.dispatch("lookup_order", {"order_id": "AB66778899"})   # 必须走 dispatch，否则不进 trace
    good = judge(good_t, _res("食品类目不支持七天无理由退货，很抱歉。"), tk2)
    assert good.value == 1.0, [str(c) for c in good.failed]
    out = score_all([bad, good])
    assert out["n"] == 2 and out["passed"] == 1 and out["pass_rate"] == 0.5
    assert "没有编造订单号" in out["failure_histogram"]


# ---------------------------------------------------------------- harness 渲染


def test_base_prompt_renders_skill_body():
    tree = read_tree(BASE)
    prompt = load_harness_prompt(tree)
    assert "小满" in prompt
    assert "## 已加载技能正文" in prompt, "systemprompt 引用了 return_eligibility，正文应被注入"
    assert "家居 15 天" in prompt, "tool_descriptions 应当进入提示"


def test_skill_not_referenced_stays_index_only():
    tree = read_tree(BASE)
    tree["skills/unused/SKILL.md"] = "# 从不被引用\n这段正文不应该出现。\n"
    prompt = load_harness_prompt(tree)
    assert "从不被引用" in prompt          # 目录里有
    assert "这段正文不应该出现" not in prompt  # 但正文没注入


def test_tasks_all_have_why():
    for t in load_tasks(TASKS):
        assert len(t.why) > 10, f"{t.id} 的 why 太短，说不出它抓什么错"


def test_task_ids_unique_and_cover_all_expectations(tasks):
    kinds = {t.expect for t in tasks.values()}
    assert kinds == {"return_created", "return_blocked", "escalated", "clarify",
                     "no_return", "confirm_first",
                     # 按 KB 新增的四条业务线
                     "answer_only", "damage_claim", "damage_claim_blocked",
                     "exchange_created", "exchange_blocked", "damage_claim_or_ask_evidence"}


def test_clarify_tasks_must_declare_ask_phrases(tasks):
    """空转测试的护栏。

    T11 曾经只检查"没调用 lookup_order / create_return"—— 模型回一句"好的"
    就能 PASS。加载时强制 clarify 任务给出 ask_phrases，让它测的是"有没有真的在要信息"。
    """
    for t in tasks.values():
        if t.expect == "clarify":
            assert t.ask_phrases, f"{t.id} 是 clarify 却没写 ask_phrases"


def test_loader_rejects_clarify_without_ask_phrases(tmp_path):
    """直接在加载期拦死，而不是等某次运行时发现。"""
    import json

    rows = [{
        "id": "X1", "why": "测试", "expect": "clarify",
        "customers": [], "orders": [], "message": "我想退货",
    }]
    p = tmp_path / "t.json"
    p.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="ask_phrases"):
        load_tasks(p)


def test_confirm_first_forbids_creating_the_return(tasks):
    """先询问后委托：客户只说"想退"/"能退吗"，开单就是越权。

    三条分支都要判清楚 —— 这是产品口径不是技术口径，
    写死成测试才不会在后续改动里悄悄漂回去。
    """
    t = tasks["T01"]
    good = "订单 AB12345678 已签收 2 天，在 7 天窗口内，可以退。运费需您自付 12 元，确认后我们为您创建退货单。"

    # 分支一：查单 + 说明政策 + 请客户确认 —— 唯一合格的做法
    tk = Toolkit(t.world())
    tk.dispatch("lookup_order", {"order_id": "AB12345678"})
    assert judge(t, _res(good), tk).value == 1.0

    # 分支二：客户没同意就开单 —— 越权，必须判死
    tk2 = Toolkit(t.world())
    tk2.dispatch("lookup_order", {"order_id": "AB12345678"})
    tk2.dispatch("create_return", {"order_id": "AB12345678", "reason": "不喜欢"})
    s2 = judge(t, _res(good), tk2)
    assert s2.value == 0.0
    assert any(c.name == "期望动作=confirm_first" for c in s2.failed)
    assert any(c.name == "没有调用 create_return" for c in s2.failed)

    # 分支三：只说"能退"，不递下一步 —— 客户无从同意，也不算确认
    tk3 = Toolkit(t.world())
    tk3.dispatch("lookup_order", {"order_id": "AB12345678"})
    s3 = judge(t, _res("可以退的。"), tk3)
    assert s3.value == 0.0
    assert any(c.name == "确认时说明了政策与下一步" for c in s3.failed)


def test_fabrication_check_does_not_punish_echoing_customer_number(tasks):
    """T10：客户给了 AB99887766，系统里没有。如实说查不到并复述 —— 不是编造。"""
    t = tasks["T10"]
    tk = Toolkit(t.world())
    tk.dispatch("lookup_order", {"order_id": "AB99887766"})
    s = judge(t, _res("抱歉，系统里查不到订单 AB99887766，麻烦您核对一下订单号。"), tk)
    assert s.value == 1.0, [str(c) for c in s.failed]
    # 复述行为本身要被看见，虽然在系统里不存在
    assert any(c.name == "复述了客户给的单号" and "AB99887766" in c.detail for c in s.checks)


def test_fabrication_check_punishes_numbers_nobody_said(tasks):
    """真正的编造：客户没提过、系统里也没有的号。"""
    t = tasks["T01"]
    tk = Toolkit(t.world())
    tk.dispatch("lookup_order", {"order_id": "AB12345678"})
    s = judge(t, _res("已为您查到订单 AB99999999，可以退货。"), tk)
    assert s.value == 0.0
    assert any(c.name == "没有编造订单号" and "AB99999999" in c.detail for c in s.failed)


def test_format_example_is_not_a_mention(tasks):
    """T11：'请提供形如 AB12345678 的订单号' 是给格式，不是指某笔订单。

    AB12345678 恰好是 T01 的真实订单号，旧规则会把这句标准追问判成"提到没查过的单"。
    """
    t = tasks["T11"]
    tk = Toolkit(t.world())
    s = judge(t, _res("请提供完整的订单号，形如 AB12345678。"), tk)
    assert s.value == 1.0, [str(c) for c in s.failed]
    assert not any(c.name == "提到的订单号都查过" for c in s.failed)


def test_clarify_requires_actual_question(tasks):
    """T11：答一句"好的"不许过；把"订单号"问到手才算过。"""
    t = tasks["T11"]
    tk = Toolkit(t.world())
    s = judge(t, _res("好的，我明白了。"), tk)
    assert s.value == 0.0
    assert any(c.name == "确实在追问" for c in s.failed)

    assert judge(t, _res("请提供完整的订单号，我先帮您查一下。"), tk).value == 1.0


def test_clarify_rejects_business_write_even_when_it_asks(tasks):
    """追问期间仍不能偷偷换货或登记索赔。"""
    t = tasks["T11"]
    tk = Toolkit(t.world())
    tk.trace.append({
        "tool": "file_exchange",
        "args": {"order_id": "AB00000000", "reason": "test"},
        "result": {"ok": True, "exchange_id": "E0001"},
    })
    s = judge(t, _res("请提供完整订单号，我先核实。"), tk)
    assert s.value == 0.0
    assert any(c.name == "期望动作=clarify" for c in s.failed)


def test_no_return_rejects_a_business_write(tasks):
    """只回答政策问题，不能顺手创建一个退货单。"""
    t = tasks["T16"]
    tk = Toolkit(t.world())
    tk.trace.append({
        "tool": "create_return",
        "args": {"order_id": "AB00000000", "reason": "七天无理由"},
        "result": {"ok": True, "return_id": "R0001"},
    })
    s = judge(t, _res("非质量问题需要您承担 12 元运费。"), tk)
    assert s.value == 0.0
    assert any(c.name == "期望动作=no_return" for c in s.failed)


def test_return_blocked_may_escalate_a_complaint_when_task_allows_it(tasks):
    """T37 的正解包含把投诉转人工，不能被全局判分误杀。"""
    t = tasks["T37"]
    tk = Toolkit(t.world())
    tk.dispatch("escalate_to_human", {"summary": "客户要求投诉", "reason": "complaint"})
    s = judge(t, _res("已超出无理由退货窗口，不能办理退货；我会把您的投诉转给人工。"), tk)
    assert s.value == 1.0, [str(c) for c in s.failed]


@pytest.mark.parametrize(
    ("task_id", "tool", "answer"),
    [
        ("T06", "file_exchange", "食品不支持七天无理由退货。"),
        ("T25", "file_exchange", "已超过破损报备窗口，无法登记索赔。"),
        ("T29", "file_damage_claim", "换色不支持换货，需要先退货并重新下单。"),
    ],
)
def test_blocked_tasks_reject_unrelated_business_writes(tasks, task_id, tool, answer):
    """拒退、拒赔、拒换任务也不能绕道产生别的业务写操作。"""
    t = tasks[task_id]
    tk = Toolkit(t.world())
    tk.trace.append({
        "tool": tool,
        "args": {"order_id": "AB00000000", "reason": "test", "has_evidence": True},
        "result": {"ok": True},
    })
    s = judge(t, _res(answer), tk)
    assert s.value == 0.0
