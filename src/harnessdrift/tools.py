"""电商售后工具层 —— 确定性、可单测、不碰网络。

设计要点：

1. **所有工具都返回结构化 dict**，不返回自然语言。让模型自己组织话术，
   否则"工具写得好不好"和"模型会不会说话"两件事就混在一起，评测无法归因。
2. **没有 *查全部订单* 的工具**。只能按 order_id 精确查 —— 这是故意的，
   它逼着 agent 在信息不足时先追问，而不是瞎猜。
3. **政策条款集中在本文件顶部**，退货窗口、运费承担这些数字一处定义，
   改政策只动一个地方。

三条硬约定（抄自生产客服系统的做法，成本是注释，收益是可被追问）：

- **写操作永不重试。** ``create_return`` 超时不代表没执行成功，重试可能开两张退货单，
  而"重复执行"比"失败"更糟 —— 失败客户会再问一次，重复执行要人工冲销。
- **审计/留痕失败绝不反拦业务。** trace 只是记录，写不进去不该让一次正常的退货失败。
- **身份类参数不进模型可见的 schema。** 让模型有机会填 ``customer_id``，
  一句"我是客服主管"就可能把它改写掉。本工具层没有身份参数，
  但第 2 步自改进加参数时必须遵守这条。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, ClassVar

# ---------------------------------------------------------------- 政策常量
#
# 全部取自 data/kb/returns-policy.md（喵喵优选官方售后政策）。
# 这里只放**机器要判断的东西**：窗口天数、起算点、哪些原因免运费。
# 措辞层面的规定（"不要承诺几个工作日到账"）不放在常量里 —— 那不是可计算的规则，
# 放进系统提示当"禁令"更合适，硬塞成常量就成了假的可配置项。

RETURN_WINDOW_DAYS = {
    "服饰": 7,
    "鞋靴": 7,
    "家居": 15,     # 平台对家具有一定的宽松期，见 KB「适用范围」
    "数码": 7,      # 一旦激活不退，见 NON_RETURNABLE_IF_ACTIVATED
    "食品": 0,      # 不支持无理由退货
}

# KB: "定制类、生鲜类、已拆封的猫砂等消耗品" 不支持无理由退货。
# 注意这是**品类级**的：同是"家居"，猫砂（消耗品）不可无理由退，猫爬架可以。
NON_RETURNABLE_CATEGORIES = {"食品", "生鲜", "定制"}
NON_RETURNABLE_IF_ACTIVATED = {"数码"}

# KB: "因商品质量问题、发错货、少发漏发导致的退换货，运费由平台承担"
FREE_RETURN_SHIPPING = ("质量问题", "发错货", "少件漏发")

# KB 里没有承诺"几个工作日到账"，只有"平台不承诺具体天数"。
# 所以这里刻意不设 SLA 常量 —— 设了就等于给了模型一个可以背出来的数字，
# 而那正是政策明令禁止的。原来这里的 REFUND_SLA_DAYS = 3 与 KB 冲突，已删。

# 价保的 7 天从**下单**起算，不是签收 —— KB 专门用一句强调过这个区别。
PRICE_PROTECT_DAYS = 7
# 破损报备窗口，从签收起算。
DAMAGE_REPORT_HOURS = 48
# 电子类保修月数。
WARRANTY_MONTHS = 12


# ---------------------------------------------------------------- 数据模型


@dataclass(frozen=True)
class Order:
    order_id: str
    customer_id: str
    category: str
    item: str
    amount: float
    status: str           # paid / shipped / delivered / refunding / refunded / cancelled
    delivered_on: date | None
    activated: bool = False
    # 下单日期。价保的 7 天从这里起算而不是签收日 —— KB 专门强调过，
    # 少了这个字段价保题根本没法判。
    ordered_on: date | None = None
    # 价保参考价：下单时的成交价。客户看到降价时的"当前价"由任务直接给。
    paid_price: float = 0.0


@dataclass(frozen=True)
class Customer:
    customer_id: str
    name: str
    vip: bool = False


@dataclass
class World:
    """一次任务的全部世界状态。工具只能读它，不能写它。"""

    today: date
    customers: dict[str, Customer]
    orders: dict[str, Order]

    def get_order(self, order_id: str) -> Order | None:
        return self.orders.get(order_id.strip().upper())

    def orders_of(self, customer_id: str) -> list[Order]:
        return [o for o in self.orders.values() if o.customer_id == customer_id]


# ---------------------------------------------------------------- 工具实现


def _ok(**kw: Any) -> dict[str, Any]:
    return {"ok": True, **kw}


def _err(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": code, "message": message}


class Toolkit:
    """把 World 包成一组可调用工具，并记录调用轨迹。

    轨迹（tracess）是评测的一部分：不仅要答对，还要**先查后答**。
    """

    TOOL_SPECS: ClassVar[list[dict[str, Any]]] = [
        {
            "name": "lookup_order",
            "description": (
                "按订单号精确查询一笔订单。没有模糊搜索，没有列表全部订单的能力。"
                "订单号形如 AB12345678。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "完整订单号"},
                },
                "required": ["order_id"],
            },
        },
        {
            "name": "get_return_policy",
            "description": "查询某个商品品类当前适用的退换货政策条款。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "商品品类，如 服饰/鞋靴/家居/数码/食品"},
                },
                "required": ["category"],
            },
        },
        {
            "name": "create_return",
            "description": (
                "为已签收订单创建退货单。调用前必须已通过 lookup_order 确认订单，"
                "并已通过 get_return_policy 确认符合条件。"
                "这是不可撤销的写操作：客户只是**询问**能不能退、尚未表示要办时，"
                "先说明政策并征得同意；客户已明确表示要退时，直接创建，不要再问一遍。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string", "description": "退货原因，取值为 质量问题 / 发错货 / 少件漏发 / 七天无理由 / 不喜欢 之一"},
                },
                "required": ["order_id", "reason"],
            },
        },
        {
            "name": "check_price_protection",
            "description": (
                "查询一笔订单是否在价保期内（从**下单日**起 7 天，不是签收日）。"
                "客户看到降价、问能不能退差价时先调这个。"
            ),
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
        {
            "name": "file_exchange",
            "description": (
                "为已签收订单登记换货申请（同款换货，7 天内，每单一次）。"
                "客户明确说要换货时调用。不同型号/颜色需先退货再下单，不能走这个。"
                "这是写操作：客户已明确表示要换（如「我要换一件大码」）时直接登记，"
                "不要再问一遍「确认吗」；只有客户在询问（如「能换吗」）时才先说明政策。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string", "description": "换货原因"},
                },
                "required": ["order_id", "reason"],
            },
        },
        {
            "name": "file_damage_claim",
            "description": (
                "登记运输破损索赔。须在签收后 48 小时内、且客户提供开箱视频与照片。"
                "缺少凭证时不要调用，先向客户索要。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "has_evidence": {"type": "boolean", "description": "客户是否已提供开箱视频与照片"},
                },
                "required": ["order_id", "has_evidence"],
            },
        },
        {
            "name": "escalate_to_human",
            "description": "转人工。仅在本轮工具无法解决、信息自相矛盾、或客户明确要求转人工时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "给人工客服的一段话摘要"},
                    "reason": {"type": "string", "description": "升级原因代码"},
                },
                "required": ["summary", "reason"],
            },
        },
    ]

    def __init__(self, world: World) -> None:
        self.world = world
        self.trace: list[dict[str, Any]] = []
        self.returns: list[dict[str, Any]] = []
        self.escalations: list[dict[str, Any]] = []
        self.exchanges: list[dict[str, Any]] = []
        self.damage_claims: list[dict[str, Any]] = []

    # -- 工具本体 ----------------------------------------------------------

    def lookup_order(self, order_id: str) -> dict[str, Any]:
        o = self.world.get_order(order_id)
        if o is None:
            return _err("ORDER_NOT_FOUND", f"不存在订单 {order_id}")
        days = None
        if o.delivered_on is not None:
            days = (self.world.today - o.delivered_on).days
        return _ok(
            order_id=o.order_id,
            item=o.item,
            category=o.category,
            amount=o.amount,
            status=o.status,
            delivered_on=o.delivered_on.isoformat() if o.delivered_on else None,
            days_since_delivery=days,
            activated=o.activated,
        )

    def get_return_policy(self, category: str) -> dict[str, Any]:
        cat = category.strip()
        if cat not in RETURN_WINDOW_DAYS:
            return _err("CATEGORY_UNKNOWN", f"未知品类 {category}")
        return _ok(
            category=cat,
            return_window_days=RETURN_WINDOW_DAYS[cat],
            non_returnable_if_activated=cat in NON_RETURNABLE_IF_ACTIVATED,
            non_returnable_categories=sorted(NON_RETURNABLE_CATEGORIES),
            free_return_shipping_reasons=list(FREE_RETURN_SHIPPING),
            # 不再返回 refund_sla_days：KB 明确"平台不承诺具体到账天数"，
            # 工具返回一个天数字段，等于亲手递给模型一个可以编出来的数字。
            price_protect_window_days=PRICE_PROTECT_DAYS,
            price_protect_basis="from_order_date",
            damage_report_hours=DAMAGE_REPORT_HOURS,
            warranty_months=WARRANTY_MONTHS,
        )

    def create_return(self, order_id: str, reason: str) -> dict[str, Any]:
        # 写操作：调用方（agent）不重试，见模块 docstring。
        # 幂等性不在这里兜底 —— 重复开单要判"已退过"的业务状态，那是策略，
        # 归评测管（T13），不归工具管。
        o = self.world.get_order(order_id)
        if o is None:
            return _err("ORDER_NOT_FOUND", f"不存在订单 {order_id}")
        if o.status != "delivered":
            return _err("ORDER_NOT_DELIVERED", f"订单状态为 {o.status}，不能退货")

        policy = self.get_return_policy(o.category)
        assert policy["ok"]
        reason = reason.strip()
        # 官方写的"七天无理由"，用户常说"不喜欢"，这里做同义归一 ——
        # 归一化放在工具里而不是提示词里，是为了让换模型时不依赖模型记不记得
        if reason in ("不喜欢", "不想要", "不需要"):
            reason = "七天无理由"

        if reason == "七天无理由":
            window = policy["return_window_days"]
            if window == 0 or o.category in NON_RETURNABLE_CATEGORIES:
                return _err("CATEGORY_NO_RETURN", f"{o.category} 不支持无理由退货")
            assert o.delivered_on is not None
            if (self.world.today - o.delivered_on).days > window:
                return _err(
                    "WINDOW_EXPIRED",
                    f"已超出 {window} 天无理由退货窗口",
                )
            if o.activated and policy["non_returnable_if_activated"]:
                return _err("ACTIVATED_NON_RETURNABLE", "商品已激活，不支持无理由退货")

        shipping = 0.0 if reason in FREE_RETURN_SHIPPING else 12.0
        rec = {"order_id": o.order_id, "reason": reason, "shipping_fee": shipping}
        try:
            self.returns.append(rec)
        except Exception:
            pass
        return _ok(return_id=f"R{len(self.returns):04d}", **rec)

    def escalate_to_human(self, summary: str, reason: str) -> dict[str, Any]:
        self.escalations.append({"summary": summary, "reason": reason})
        return _ok(ticket_id=f"T{len(self.escalations):04d}")

    def check_price_protection(self, order_id: str) -> dict[str, Any]:
        """价保：7 天从**下单**起算。

        这是评测里最容易被模型整错的一条 —— KB 特意写了"价保是另一条规则，
        它的 7 天从下单时间起算"，就是因为退货的 7 天是从签收算的。
        工具只报事实（窗口内还是外、差几天），判断退不退差价交给模型。
        """
        o = self.world.get_order(order_id)
        if o is None:
            return _err("ORDER_NOT_FOUND", f"不存在订单 {order_id}")
        if o.ordered_on is None:
            return _err("NO_ORDER_DATE", "该订单缺少下单日期")
        elapsed_order = (self.world.today - o.ordered_on).days
        elapsed_delivery = (
            (self.world.today - o.delivered_on).days if o.delivered_on else None
        )
        return _ok(
            order_id=o.order_id,
            paid_price=o.paid_price or o.amount,
            ordered_on=o.ordered_on.isoformat(),
            delivered_on=o.delivered_on.isoformat() if o.delivered_on else None,
            days_since_order=elapsed_order,
            days_since_delivery=elapsed_delivery,
            price_protect_window_days=PRICE_PROTECT_DAYS,
            within_window=elapsed_order <= PRICE_PROTECT_DAYS,
            basis="days_since_order",
        )

    def file_exchange(self, order_id: str, reason: str) -> dict[str, Any]:
        o = self.world.get_order(order_id)
        if o is None:
            return _err("ORDER_NOT_FOUND", f"不存在订单 {order_id}")
        if o.status != "delivered":
            return _err("ORDER_NOT_DELIVERED", f"订单状态为 {o.status}，不能换货")
        if o.delivered_on is None:
            return _err("NO_DELIVERY_DATE", "缺少签收日期")
        days = (self.world.today - o.delivered_on).days
        if days > 7:
            return _err("WINDOW_EXPIRED", "已超过 7 天换货窗口")
        # KB: "可申请**一次**同款换货"
        used = sum(1 for e in self.exchanges if e["order_id"] == o.order_id)
        if used >= 1:
            return _err("EXCHANGE_ALREADY_USED", "该订单已申请过换货，仅支持一次")
        free = any(k in reason for k in FREE_RETURN_SHIPPING)
        rec = {"order_id": o.order_id, "reason": reason, "shipping": "平台" if free else "买家"}
        self.exchanges.append(rec)
        return _ok(exchange_id=f"E{len(self.exchanges):04d}", **rec)

    def file_damage_claim(self, order_id: str, has_evidence: bool) -> dict[str, Any]:
        o = self.world.get_order(order_id)
        if o is None:
            return _err("ORDER_NOT_FOUND", f"不存在订单 {order_id}")
        if o.delivered_on is None:
            return _err("NO_DELIVERY_DATE", "缺少签收日期")
        elapsed = (self.world.today - o.delivered_on)
        if not has_evidence:
            # 缺凭证不是拒绝，是"还不能登记"：先把要什么告诉模型，
            # 它才会去问客户要开箱视频，而不是直接开索赔。
            return _err("EVIDENCE_REQUIRED", "需客户提供开箱视频与照片后才能登记破损索赔")
        if elapsed > timedelta(hours=DAMAGE_REPORT_HOURS):
            return _err("WINDOW_EXPIRED", f"已超过签收后 {DAMAGE_REPORT_HOURS} 小时报备窗口")
        rec = {"order_id": o.order_id, "compensation": "补发或退款", "shipping_by": "平台"}
        self.damage_claims.append(rec)
        return _ok(claim_id=f"D{len(self.damage_claims):04d}", **rec)

    def apply_setup(self, actions: list[dict[str, Any]]) -> None:
        """执行任务预置动作，**不写 trace**。

        这是关键：预置的换货记录是"上一轮已经发生的事"，不是模型本轮的动作。
        如果写进 trace，判 must_not_call=["file_exchange"] 就永远不成立，
        题目会变成不可解。状态要走真实工具路径产生（不用特判字段），
        但痕迹必须和本轮调用分开。
        """
        for a in actions:
            name, args = a["tool"], a.get("args", {})
            impl = {
                "create_return": self.create_return,
                "file_exchange": self.file_exchange,
                "file_damage_claim": self.file_damage_claim,
                "escalate_to_human": self.escalate_to_human,
            }.get(name)
            if impl is None:
                raise ValueError(f"setup_actions 不支持的工具 {name}")
            impl(**args)

    # -- 调度 --------------------------------------------------------------

    _IMPL: ClassVar[dict[str, Callable[..., dict[str, Any]]]] = {}

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        impl = {
            "lookup_order": self.lookup_order,
            "get_return_policy": self.get_return_policy,
            "create_return": self.create_return,
            "check_price_protection": self.check_price_protection,
            "file_exchange": self.file_exchange,
            "file_damage_claim": self.file_damage_claim,
            "escalate_to_human": self.escalate_to_human,
        }.get(name)
        if impl is None:
            result = _err("UNKNOWN_TOOL", f"没有名为 {name} 的工具")
        else:
            try:
                result = impl(**args)
            except TypeError as e:
                result = _err("BAD_ARGS", f"参数不匹配：{e}")
            except Exception as e:
                # 工具内部异常一律包成错误结果回灌模型，绝不向上抛。
                # 抛出去 = 整条任务崩掉，评测拿不到分数，也说不清是模型的错还是我的错。
                result = _err("TOOL_CRASHED", f"工具内部错误：{type(e).__name__}")
        # 留痕失败不许反拦业务：trace 只是记录，写不进去不该让那次调用失败
        try:
            self.trace.append({"tool": name, "args": args, "result": result})
        except Exception:
            pass
        return result


# ---------------------------------------------------------------- 时间解析


_REL = re.compile(r"(\d+)\s*天前")


def parse_today(text: str) -> date:
    """把任务里的 today 写成 ISO 日期或 'N 天前' 均可。"""
    text = text.strip()
    m = _REL.fullmatch(text)
    if m:
        return date.today() - timedelta(days=int(m.group(1)))
    return datetime.strptime(text, "%Y-%m-%d").date()


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)
