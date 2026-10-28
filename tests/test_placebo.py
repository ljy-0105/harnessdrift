"""安慰剂对照组：不给失败样本时，曲线会不会自己往上走。

这是全项目唯一一个**故意让改进器看不见缺陷**的实验。它要回答的问题很朴素：
我看到的 +4.33，到底是"提示词改对了"，还是"这一版评测本来就会给高分"？

如果安慰剂轮也能涨，那说明评测本身在漂 —— 那么所有基于均值的结论都得打折。
这种实验丑，但它是唯一能把"药效"和"自愈"分开的办法。
"""
from __future__ import annotations

from pathlib import Path

from harnessdrift.llm import LLMReply
from harnessdrift.loop import run_loop
from harnessdrift.propose import PLACEBO_SYSTEM, build_placebo_prompt, propose
from harnessdrift.tasks import Task

MAGIC = "MAGIC_ALPHA"
PASSING_PROMPT = f"# 你是客服\n\n{MAGIC}\n"


class _Recorder:
    """记下收到的每一条消息；改进器按剧本吐一个合规的小补丁。"""

    def __init__(self, patch: str = "") -> None:
        self.model = "recorder"
        self.patch = patch or "### ADD tool_descriptions/tone.md\n    回复时先称呼客户。\n"
        self.improver_prompts: list[str] = []
        self.systems: list[str] = []
        self.usage_total: dict[str, int] = {"total_tokens": 0}

    def chat(self, messages, **kw):  # noqa: ANN001, ANN003
        system = messages[0]["content"] if messages else ""
        body = "\n".join(m.get("content", "") for m in messages)
        self.systems.append(system)
        if "harness 工程师" in system:
            self.improver_prompts.append(body)
            return LLMReply(text=self.patch, finish_reason="stop")
        return LLMReply(text=MAGIC, finish_reason="stop")


def _tree(tmp_path: Path) -> Path:
    d = tmp_path / "base"
    (d / "tool_descriptions").mkdir(parents=True)
    (d / "systemprompt.md").write_text(PASSING_PROMPT, encoding="utf-8")
    (d / "tool_descriptions" / "lookup.md").write_text("查订单。\n", encoding="utf-8")
    return d


def _tasks() -> list[Task]:
    """一条会过、一条必挂 —— 只有不是满分，循环才会去想"改进"。"""
    return [
        Task(id="T01", message="第 1 单要退", why="安慰剂测试，会过",
             expect="no_return", customers=[], orders=[]),
        Task(id="T02", message="第 2 单要退", why="安慰剂测试，必挂",
             expect="return_created", customers=[], orders=[],
             must_call=["create_return"]),
    ]


def test_placebo_prompt_carries_no_failure_information():
    prompt = build_placebo_prompt({"systemprompt.md": PASSING_PROMPT})
    assert "失败案例" not in prompt
    assert "T01" not in prompt
    assert "## 当前提示词目录" in prompt
    assert "无关" in PLACEBO_SYSTEM


def test_propose_placebo_ignores_the_failures_it_is_given(tmp_path: Path):
    """哪怕调用方把失败样本递进来，安慰剂模式也不许把它们写进提示词。"""
    from harnessdrift.propose import FailedSample

    llm = _Recorder()
    leak = FailedSample(task_id="T99", message="我要退款 —— 泄漏样本",
                        expect_label="期望动作=return_created", failed_checks=["调用了 create_return"],
                        tool_calls=["lookup_order"], final_answer="好的")
    prop = propose({"systemprompt.md": PASSING_PROMPT}, [leak], tmp_path / "out", llm,
                   placebo=True)
    assert prop.applied, prop.warnings
    sent = llm.improver_prompts[0]
    assert "T99" not in sent and "泄漏样本" not in sent
    assert PLACEBO_SYSTEM.strip()[:12] in llm.systems[0]


def test_placebo_keeps_looping_when_nothing_fails(tmp_path: Path):
    """没有失败样本时普通模式当场收工；安慰剂模式必须继续跑，否则对照不成立。"""
    tree = _tree(tmp_path)
    common = dict(reps=1, max_rounds=1, patience=9, accept="always",
                  safety_veto=False, budget_s=0.0, verbose=False)

    # 对照：这条失败样本在普通模式下会被喂给提案端
    plain_improver = _Recorder()
    plain = run_loop(tree, _tasks(), tmp_path / "plain", llm=_Recorder(),
                     improver=plain_improver, **common)
    assert len(plain.rounds) == 2, "普通模式应当提一次改进（基线 + 第 1 轮）"
    assert "失败案例" in plain_improver.improver_prompts[0]

    placebo_improver = _Recorder()
    placebo = run_loop(tree, _tasks(), tmp_path / "placebo", llm=_Recorder(),
                       improver=placebo_improver, placebo=True, **common)
    assert len(placebo.rounds) == 2, "安慰剂组也该提一次，且不看失败样本"
    assert placebo.config["placebo"] is True
    assert "失败案例" not in placebo_improver.improver_prompts[0]
    assert "T02" not in placebo_improver.improver_prompts[0]