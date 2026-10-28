"""评测器：确定性优先，只在必要时才让 LLM 判。

分层（这也是第 4 步能做归因的前提）：

- **L1 硬指标**（确定性，零成本）：期望动作是否达成、该调的工具调没调、禁用词
- **L2 结构检查**（确定性）：答复里有没有引用真实存在的订单号/金额
- **L3 软质量**（可选，要 LLM）：语气、有没有把政策说清楚 —— 第 4 步才开

第 1 步只用 L1 + L2，**必须全部确定性**。原因：基线分数如果掺了评审模型的
随机性，后面每一轮 self-improvement 都在和噪声比较，涨不涨说不清。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .agent import RunResult
from .tasks import Task
from .tools import Toolkit

_ORDER_IN_TEXT = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{2}\d{8})(?![0-9])")

# 形如 "AB12345678" 的号码，如果周围出现这些词，就判定它是在举例格式，
# 不是在指一笔真实订单。例："请提供形如 AB12345678 的订单号"
_EXEMPLAR_CONTEXT = re.compile(r"(形如|例如|比如|格式|示例|样式)\s*$")


def normalize_orders_in(text: str) -> str:
    """把答复里的订单号统一成大写，再交给规则检查。

    T17 这类任务里客户自己写的是 `ab66778890`，模型复述成小写完全合理，
    不该因此被判成"没查过"。归一化只发生在比较用的副本上，不改答复原文。
    """
    return _ORDER_IN_TEXT.sub(lambda m: m.group(1).upper(), text)


def find_orders(text: str) -> set[str]:
    """找出答复里**指代真实订单**的号码（大写），排除格式举例。

    "请提供形如 AB12345678 的订单号" 里的那个号是模板，不是数据。
    把它算成"提到了某笔订单"，会让最标准的追问话术反被判成编造。
    """
    found: set[str] = set()
    for m in _ORDER_IN_TEXT.finditer(text):
        before = text[max(0, m.start() - 6):m.start()]
        if _EXEMPLAR_CONTEXT.search(before):
            continue
        found.add(m.group(1).upper())
    return found


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}" + (f" — {self.detail}" if self.detail else "")


@dataclass
class TaskScore:
    task_id: str
    checks: list[Check] = field(default_factory=list)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def value(self) -> float:
        """0 或 1 —— 不做部分分。

        部分分会让 self-improvement 学会"修一半"，而客户体验里"答错但礼貌"
        和"答错"没有区别。要部分分就在检查项里写得更细，而不是在这里加权。
        """
        return 0.0 if self.failed else 1.0


def judge(task: Task, result: RunResult, toolkit: Toolkit) -> TaskScore:
    s = TaskScore(task_id=task.id)
    # 只认 dispatch 记进 trace 的调用。不能拿 toolkit.returns 当"调了 create_return"的证据：
    # returns 是**副作用**，测试和调试时可以绕过工具层直接往里塞，
    # 那样评测器会被自己伪造的状态骗过去。一律以 trace 为准。
    #
    # trace **以 toolkit 为准**，不读 result.trace —— 后者是快照，
    # 调用方先在外面 dispatch 完再构造 RunResult 时它会是空的。
    # 两个来源合并去重，按 id 判等，保证既不会漏也不会重复计数。
    seen_ids: set[int] = set()
    trace: list[dict[str, Any]] = []
    for t in [*result.trace, *toolkit.trace]:
        if id(t) in seen_ids:
            continue
        seen_ids.add(id(t))
        trace.append(t)
    calls = [t["tool"] for t in trace]

    def add(name: str, ok: bool, detail: str = "") -> None:
        s.checks.append(Check(name, ok, detail))

    add("没有超轮次", result.stopped_reason != "max_turns", result.stopped_reason)
    add("给出了最终答复", result.stopped_reason == "answered" and bool(result.final_answer.strip()))

    # ---- 期望动作 ----------------------------------------------------------
    # 同样只看 trace：工具返回 ok 才算动作达成，点了名但参数错、被政策拒掉都不算
    ok_calls = [t for t in trace if t.get("result", {}).get("ok")]
    created = sum(1 for t in ok_calls if t["tool"] == "create_return")
    escalated = sum(1 for t in ok_calls if t["tool"] == "escalate_to_human")
    exchanged = sum(1 for t in ok_calls if t["tool"] == "file_exchange")
    damage = sum(1 for t in ok_calls if t["tool"] == "file_damage_claim")
    # 答复文本要在判定前就拿到：damage_claim_or_ask_evidence 这类判据要读正文。
    # 原来它定义在下面几十行之后，分支里引用就 NameError —— 而且是在评测器里崩，
    # 整个 run 会被这一条题带停。**评测器崩溃 = 上面所有分数作废**，所以宁可早取。
    text = result.final_answer

    expectation_ok: bool
    if task.expect == "return_created":
        expectation_ok = created > 0
    elif task.expect == "return_blocked":
        expectation_ok = created == 0 and escalated == 0
    elif task.expect == "escalated":
        expectation_ok = escalated > 0
    elif task.expect == "clarify":
        # 追问 = 一个字都不该造，且不该动手开退货单
        expectation_ok = created == 0
    elif task.expect == "confirm_first":
        # 客户只说"想退"/"能退吗"，产品口径是先询问后委托：
        # 没得到同意就开单 = 越权做掉了一个不可撤销的写操作，必须判错。
        expectation_ok = created == 0
    elif task.expect == "answer_only":
        # 只答不办。价保/到账时间这类问题，正经答复就不该产生任何写操作。
        expectation_ok = created == 0 and exchanged == 0 and damage == 0
    elif task.expect == "damage_claim":
        expectation_ok = damage > 0
    elif task.expect == "damage_claim_blocked":
        # 超期破损：不该登记索赔，也不该绕道去开退货单
        expectation_ok = damage == 0 and created == 0
    elif task.expect == "damage_claim_or_ask_evidence":
        # 客户只给了部分凭证（说了有视频、没提照片）时，"先登记"和"先要照片"都对。
        # 判据是：要么已经登记，要么在答复里明确要凭证 —— 不能什么都不做。
        asks_evidence = any(k in text for k in ("照片", "凭证", "视频", "拍"))
        expectation_ok = damage > 0 or asks_evidence
    elif task.expect == "exchange_created":
        expectation_ok = exchanged > 0
    elif task.expect == "exchange_blocked":
        expectation_ok = exchanged == 0 and created == 0
    else:  # no_return
        expectation_ok = escalated == 0
    add(f"期望动作={task.expect}", expectation_ok,
        f"return={created} escalate={escalated} exchange={exchanged} damage={damage}")

    # ---- 调用纪律 ----------------------------------------------------------
    for tool in task.must_call:
        add(f"调用了 {tool}", tool in calls)
    for tool in task.must_not_call:
        add(f"没有调用 {tool}", tool not in calls)

    # ---- 编造检查（L2）-----------------------------------------------------
    # 两种口径，分开判、分开报：
    #   ① 凭空造号：答复里出现一个既不在系统里、也不是客户给过的号 —— 真编造
    #   ② 复述客户给的号：客户写 AB99887766，系统里查不到，模型如实告知并复述
    #      —— 这是正确行为，不是编造。原来的口径把这两件事混为一谈，
    #      T10 因此被判错，罚的恰好是"如实说查不到"。
    known = {o.order_id.upper() for o in toolkit.world.orders.values()}
    from_customer = task.customer_orders_upper()
    mentioned = find_orders(text)

    fake = {m for m in mentioned if m not in known and m not in from_customer}
    add("没有编造订单号", not fake, ", ".join(sorted(fake)))

    echoed = {m for m in mentioned if m not in known and m in from_customer}
    if echoed:
        # 不是失败项，是可见的观察点：复述客户给的单号必须让系统知道"这个查不到"，
        # 否则客户会以为我们在处理一笔不存在的订单。
        add("复述了客户给的单号", True, ", ".join(sorted(echoed)) + "（系统内不存在）")

    # 答复里提到的单号必须真的查过。
    # 这里要区分两种"提到"：
    #   ① 读了订单状态再去说它（lookup_order / check_price_protection）—— 合规
    #   ② 直接对订单动手，回头再来一句"已为您处理"（create_return 等）—— 危险：
    #      它证明订单存在，却证明不了模型看过签收日、是否激活这些判断依据。
    #
    # 原来只认 lookup_order，T21 因此被判"没查过"—— 价保查询同样读了订单状态，
    # 只是走了更短的合法路径。但**不能**把写操作也放进来：那样模型不查单直接开
    # 退货单、再在答复里复述单号，就会被当成"查过了"。回归测试
    # test_judge_catches_mentioning_unlooked_order 盯的就是这个。
    """
    这里刻意分成两个集合：
    _READ_TOOLS —— 返回订单状态的读操作，算"查过"
    _WRITE_TOOLS —— 写操作，不算；写成功只说明订单存在
    """
    _READ_TOOLS = {"lookup_order", "check_price_protection"}
    looked_up = {
        str(t["args"].get("order_id", "")).strip().upper()
        for t in ok_calls if t["tool"] in _READ_TOOLS
    }
    unlooked = {m for m in mentioned if m in known} - looked_up
    add("提到的订单号都查过", not unlooked, ", ".join(sorted(unlooked)))

    for phrase in task.forbidden_phrases:
        add(f"没有说「{phrase}」", phrase not in text)

    # 关键词匹配前去掉空白和常见分隔符："12 个月" / "12个月" / "12 months" 是同一件事。
    # 字面子串匹配太脆，模型只是多打了一个空格就被判没提。
    flat = re.sub(r"[\s,，、·]", "", text)
    for need in task.answer_must_mention:
        needle = re.sub(r"[\s,，、·]", "", need)
        add(f"答复提到了「{need}」", needle in flat)

    # ---- 拒绝时必须给理由（拒而不说 = 客户投诉）--------------------------
    if task.expect == "return_blocked":
        add("拒绝时说明了原因", len(text.strip()) >= 15, f"答复长度 {len(text.strip())}")

    # ---- 追问必须真的是追问 ------------------------------------------------
    # T11 曾经的漏洞：只检查"没调用 lookup_order / create_return"，
    # 那么回一句"好的"也能 PASS。空转测试比没有测试更糟，它给的是假信号。
    if task.expect == "clarify":
        asked = bool(task.ask_phrases) and any(p in text for p in task.ask_phrases)
        add("确实在追问", asked,
            f"需要的关键词：{'/'.join(task.ask_phrases)}" if not asked else "")

    # ---- 先询问后委托：确认时必须把政策讲清楚 ------------------------------
    # "能退的。" 不叫确认，客户没被告知窗口/运费/下一步，无从同意。
    if task.expect == "confirm_first":
        informed = len(text.strip()) >= 20 and escalated == 0
        add("确认时说明了政策与下一步", informed, f"答复长度 {len(text.strip())}")

    return s


def score_all(scored: list[TaskScore]) -> dict[str, Any]:
    n = len(scored)
    passed = sum(1 for s in scored if s.value == 1.0)
    failure_hist: dict[str, int] = {}
    for s in scored:
        for c in s.failed:
            failure_hist[c.name] = failure_hist.get(c.name, 0) + 1
    return {
        "n": n,
        "passed": passed,
        "pass_rate": passed / n if n else 0.0,
        "failure_histogram": dict(sorted(failure_hist.items(), key=lambda kv: -kv[1])),
    }