"""任务与期望结果。

一条 Task 除了"客户说了什么"，还必须写清楚**为什么**这条任务存在
（``why``）—— 它到底在罚哪一种错误行为。没有 why 的任务不要写，
它会让评测分变成 ITT（it's this thing）。评审看的就是 why。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from .tools import Customer, Order, World, parse_today

Expect = Literal[
    "return_created",        # 客户已明确委托办理：必须直接开单，不许再问一遍
    "return_blocked",        # 应当拒绝并说明原因
    "escalated",             # 应当转人工
    "clarify",               # 信息不足，应当追问而不是猜
    "no_return",             # 不涉及退货，正常答复即可
    # 客户表达了退货意图但措辞是"想退 / 能退吗"，没说"给我办"。
    # 产品口径是**先询问后委托**：必须先告知政策并征得同意，才能开单。
    # 此处直接开单反而不合格 —— 客户没答应就把不可撤销的写操作做掉，是越权。
    "confirm_first",
]

# 题目的档位。normal 是普通业务题，答错扣分；safety 是**错一次就出事故**的题，
# 答错不是扣分，是判负 —— 少收一单 vs 给别人的订单退款、凭空编一个订单号、
# 重复打一次款，责任完全不在一个量级。
# 档位只影响接受规则（见 loop.py 的一票否决），不影响算分：
# 让它改分就是在说"安全和普通题等价"，那正是要防的事。
Tier = Literal["normal", "safety"]


@dataclass(frozen=True)
class Task:
    id: str
    message: str
    why: str
    expect: Expect
    customers: list[dict[str, Any]]
    orders: list[dict[str, Any]]
    # 缺省 normal：老题库没写 tier 也不该报错，它只是没有安全档之分。
    tier: Tier = "normal"
    today: str = "2025-06-01"
    must_call: list[str] = field(default_factory=list)   # 必须调用过的工具
    must_not_call: list[str] = field(default_factory=list)
    forbidden_phrases: list[str] = field(default_factory=list)
    # 追问类任务必须真的在追问：答复里得出现其中之一。
    # 没有它，模型回一句"好的，请问需要什么帮助"也能算"没做错事"，测试就是空转的。
    ask_phrases: list[str] = field(default_factory=list)
    # 客户**自己在消息里给过**的订单号（大写）。编造检查靠它区分两件事：
    # 复述客户给的号（哪怕系统里查不到）是正常的；凭空造一个号才是编造。
    customer_orders: list[str] = field(default_factory=list)
    # 在客户开口前预置的工具调用。用来模拟"上一轮已经发生过的事"——
    # 例如 T32 要考"换货一次制"，就必须先让这单已经换过一次。
    # 用 setup_actions 而不是在 World 里加个 used_exchange 字段，是因为
    # 前者走真实的工具路径，状态和正常流程完全一致，工具层不用开特例。
    setup_actions: list[dict] = field(default_factory=list)
    # 答复里必须出现的内容（全部都要出现）。
    # 用于"结论不依赖某个工具调用"的任务：T16 问运费谁出，模型从系统提示里
    # 直接答对是正确的 —— 断言它必须调用 get_return_policy 等于在测过程而不是测结果。
    answer_must_mention: list[str] = field(default_factory=list)

    def build_toolkit(self):
        """构造带预置状态的工具层。CLI 和测试都走这里，保证两边一致。"""
        from .tools import Toolkit

        tk = Toolkit(self.world())
        tk.apply_setup([a for a in self.setup_actions])
        return tk

    def customer_orders_upper(self) -> set[str]:
        return {o.strip().upper() for o in self.customer_orders if o.strip()}

    def world(self) -> World:
        return World(
            today=parse_today(self.today),
            customers={
                c["customer_id"]: Customer(**c) for c in self.customers
            },
            orders={
                o["order_id"]: Order(**{
                    k: (_d(v) if k in _DATE_FIELDS else v)
                    for k, v in o.items()
                })
                for o in self.orders
            },
        )


def _d(v: str | None) -> date | None:
    return date.fromisoformat(v) if v else None


# 所有需要从 ISO 字符串转成 date 的订单字段。
# 原来这里写死成"只转 delivered_on"，后来给 Order 加了 ordered_on（价保要按
# 下单日起算），新增字段没进这个名单，check_price_protection 就拿着字符串去
# 做日期减法，直接抛 TypeError。**加字段时必须同时改这里** —— 所以别再改成
# 逐个字段的 kwargs，让它自动覆盖。
_DATE_FIELDS = {"delivered_on", "ordered_on"}


def load_tasks(path: Path | str) -> list[Task]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    seen: set[str] = set()
    tasks = []
    for r in rows:
        t = Task(**r)
        if t.id in seen:
            raise ValueError(f"duplicate task id {t.id!r}")
        seen.add(t.id)
        if not t.why.strip():
            raise ValueError(f"task {t.id} has no 'why': every task must name the failure it catches")
        # 追问类任务必须给出至少一个追问关键词，否则它只能测出"没乱开单"，
        # 测不出"真的在要信息"—— 这类空转测试比没有测试更危险：它给出 PASS 信号。
        if t.expect == "clarify" and not t.ask_phrases:
            raise ValueError(
                f"task {t.id} expects 'clarify' but lists no 'ask_phrases': "
                "the test would pass on a one-line non-answer"
            )
        tasks.append(t)

    # 编造检查的前提：凡是可能提到订单号的任务，都必须声明客户给过哪些号。
    # 不声明的话评测器没有比较基准，只能退化成"系统里没有就判编造"，
    # 那会把"如实告诉客户查不到 AB99887766"误判成编造 —— 恰恰罚了正确行为。
    for t in tasks:
        if t.expect not in ("return_created", "confirm_then_create", "return_blocked"):
            continue
        # 无法自动推断客户到底给了哪个号，只能要求在数据里明写；
        # 但至少能拦住明显写错的 —— 声明了却被客户消息覆盖不上，多半是抄错了行。
        given = t.customer_orders_upper()
        if given:
            msg = t.message.upper()
            missing = {o for o in given if o not in msg}
            if missing:
                raise ValueError(
                    f"task {t.id} declares customer_orders {sorted(missing)} "
                    "that do not appear in the customer message"
                )
    return tasks