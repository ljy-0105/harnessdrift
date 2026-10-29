"""第 3 步：循环本身的测试。

这些测试**不访问网络**。用一个脚本化的假 LLM，把"补丁能不能提出来"
这件事完全控制住，单独验证循环的三个判定是否正确：

1. 接受规则 —— 涨了才留，跌了要拒绝并保留父树
2. 停止条件 —— patience / 满分 / 补丁产不出，三种都要真的停
3. 归档 —— 被拒绝的候选要留证据，且不能污染接受链

为什么不用真模型测：真模型的输出每次不同，测出来"通过"也不知道测的是
循环还是模型。循环是确定性的逻辑，必须用确定性的输入测。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harnessdrift.llm import LLMReply
from harnessdrift.loop import _accepted, _is_improved, run_loop
from harnessdrift.report import render
from harnessdrift.tasks import Task

MAGIC_A = "MAGIC_ALPHA"
MAGIC_B = "MAGIC_BETA"

BASE_PROMPT = "# 你是客服\n\n## 原则\n1. 先查后答。\n"


def _task(tid: str, magic: str, tier: str = "normal") -> Task:
    """一条只考"答复里有没有那个词"的任务。

    刻意做成与业务无关的最小形态：这里测的是循环，不是客服能力。
    """
    return Task(
        id=tid,
        message=f"请回复 {tid}",
        why="循环测试用，验证评测链路能把分数变化传回调度器",
        expect="no_return",
        customers=[],
        orders=[],
        tier=tier,
        answer_must_mention=[magic],
    )


class ScriptedLLM:
    """求解器读系统提示词里有没有 MAGIC；改进器按剧本吐补丁。"""

    def __init__(self, patches: list[str] | None = None) -> None:
        self.model = "scripted"
        self.patches = list(patches or [])
        self.improver_calls = 0
        self.solver_calls = 0
        # 记下每次调用收到的 system 提示词：留出集测试要检查"什么被泄漏给了提案端"
        self.seen: list[str] = []
        # 假装每次调用花 100 token：token 预算测试需要一个真实记账的对象
        self.usage_total: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0,
                                            "total_tokens": 0}

    def chat(self, messages, **kw):
        system = messages[0]["content"] if messages else ""
        self.seen.append("\n".join(m.get("content", "") for m in messages))
        self.usage_total["total_tokens"] += 100
        if "harness 工程师" in system:
            self.improver_calls += 1
            if not self.patches:
                # 没有剧本了就吐一份越界补丁，让循环走"产不出合法补丁"的分支
                return LLMReply(text="### REPLACE src/harnessdrift/tools.py\n    x = 1\n",
                                finish_reason="stop")
            return LLMReply(text=self.patches.pop(0), finish_reason="stop")

        self.solver_calls += 1
        said = " ".join(m for m in (MAGIC_A, MAGIC_B) if m in system)
        return LLMReply(text=said or "我不知道。", finish_reason="stop")


def _patch_add(magic: str) -> str:
    return f"### REPLACE systemprompt.md\n    # 你是客服\n\n    {magic}\n"


def _patch_add_both() -> str:
    return f"### REPLACE systemprompt.md\n    # 你是客服\n\n    {MAGIC_A} {MAGIC_B}\n"


@pytest.fixture()
def base_tree(tmp_path: Path) -> Path:
    d = tmp_path / "base"
    (d / "tool_descriptions").mkdir(parents=True)
    (d / "systemprompt.md").write_text(BASE_PROMPT, encoding="utf-8")
    (d / "tool_descriptions" / "lookup.md").write_text("查订单。\n", encoding="utf-8")
    return d


# ---------------- 判定逻辑（纯函数，最重要）----------------

def test_improved_requires_gain_beyond_parent_noise():
    """父树抖 0.8，候选涨 0.33 —— 这是同一棵树，不能算进步。"""
    assert not _is_improved(35.33, 0.5, 35.0, 0.8)


def test_improved_when_gain_exceeds_noise():
    assert _is_improved(37.0, 0.1, 35.0, 0.3)


def test_flat_noise_is_not_improvement():
    """父子两棵树一样抖，涨 0 分不是进步。"""
    assert not _is_improved(35.0, 0.4, 35.0, 0.4)


def test_accept_modes_differ():
    assert _accepted("nonregress", 35.0, 35.0) is True, "持平接受（铺垫）"
    assert _accepted("strict", 35.0, 35.0) is False, "严格模式只留上升"
    assert _accepted("nonregress", 34.0, 35.0) is False, "下降必须拒绝"
    assert _accepted("always", 34.0, 35.0) is True, "朴素 RSI 会无条件接受退化"


# ---------------- 端到端 ----------------

def test_curve_improves_then_rejects_regression(base_tree: Path, tmp_path: Path):
    """两轮有效提升中间夹一次回退，回退必须被拒绝且父树不动。"""
    tasks = [_task("T1", MAGIC_A), _task("T2", MAGIC_B)]
    # 第 1 轮加 A（1/2），第 2 轮把 A 删了（0/2，应拒绝），第 3 轮加满（2/2）
    llm = ScriptedLLM([_patch_add(MAGIC_A), "### REPLACE systemprompt.md\n    # 你是客服\n",
                       _patch_add_both()])
    res = run_loop(base_tree, tasks, tmp_path / "run", llm=llm, improver=llm,
                   reps=1, max_rounds=5, patience=3, log=lambda *a: None)

    assert res.rounds[0].mean == 0.0, "基线应为 0"
    r1 = res.rounds[1]
    assert r1.kind == "accepted" and r1.mean == 1.0 and r1.improved

    r2 = res.rounds[2]
    assert r2.kind == "rejected", "分数下降的补丁必须被拒绝"
    assert r2.mean == 0.0

    r3 = res.rounds[3]
    assert r3.kind == "accepted" and r3.mean == 2.0, "父树应仍是第 1 轮那棵，加 B 后满分"
    assert res.stop_reason.startswith("已满分")


def test_rejected_patch_is_archived_not_silently_dropped(base_tree: Path, tmp_path: Path):
    """被拒的候选要留证据。否则"试过什么、为什么不行"就丢了。

    这里必须用 2 条任务：只有 1 条的话，第 1 轮补上 MAGIC 就已经满分，
    循环会在第 2 轮开头被"已满分"拦下，压根走不到拒绝分支。
    """
    tasks = [_task("T1", MAGIC_A), _task("T2", MAGIC_B)]
    llm = ScriptedLLM([_patch_add(MAGIC_A),
                       "### REPLACE systemprompt.md\n    # 你是客服\n"])
    out = tmp_path / "run"
    run_loop(base_tree, tasks, out, llm=llm, improver=llm,
             reps=1, max_rounds=2, patience=5, log=lambda *a: None)
    rej = out / "_rejected" / "round2"
    assert (rej / "PATCH.md").exists(), "被拒补丁原文必须留档"
    assert (rej / "systemprompt.md").exists(), "被拒候选的树也要留，才能复现"


def test_patience_stops_on_repeated_neutral(base_tree: Path, tmp_path: Path):
    """连续持平且无实质改进时必须停，不能空转到 max_rounds。

    补丁用"内容不变的 REPLACE"：它能反复应用成功、但分数不变。
    这正是 patience 要抓的病态情况 —— 改进器在原地打转，看起来每轮都在干活。
    """
    tasks = [_task("T1", MAGIC_A)]
    noop = "### REPLACE tool_descriptions/lookup.md\n    查订单。\n"
    llm = ScriptedLLM([noop] * 10)
    res = run_loop(base_tree, tasks, tmp_path / "run", llm=llm, improver=llm,
                   reps=1, max_rounds=10, patience=2, log=lambda *a: None)
    assert "patience" in res.stop_reason
    assert len([r for r in res.rounds if r.kind != "base"]) == 2, \
        "patience=2 应当在第 2 轮后停下，而不是跑满 10 轮"


def test_repeated_patch_after_apply_is_rejected(base_tree: Path, tmp_path: Path):
    """同一个 ADD 补丁第二次必然失败（文件已存在）。

    这不是 bug，是设计：补丁必须能被独立验证。但它说明改进器重复自己时
    会白烧一轮 —— 所以 loop 要把"试过且失败"喂回去，而不是默默重试。
    """
    tasks = [_task("T1", MAGIC_A)]
    same = "### ADD tool_descriptions/noop.md\n    无关说明。\n"
    res = run_loop(base_tree, tasks, tmp_path / "run",
                   llm=ScriptedLLM([same] * 3), improver=ScriptedLLM([same] * 3),
                   reps=1, max_rounds=3, patience=5, proposal_patience=5,
                   log=lambda *a: None)
    assert res.rounds[1].kind == "accepted", "第一次 ADD 应当成功"
    assert res.rounds[2].kind == "rejected", "第二次同样的 ADD 应当被拒"
    assert any("already present" in w for w in res.rounds[2].warnings)


def test_proposal_patience_stops_when_no_valid_patch(base_tree: Path, tmp_path: Path):
    """改进器一直吐越界补丁时要及时停，省 token。"""
    tasks = [_task("T1", MAGIC_A)]
    llm = ScriptedLLM([])  # 剧本为空 → 一直吐越界补丁
    res = run_loop(base_tree, tasks, tmp_path / "run", llm=llm, improver=llm,
                   reps=1, max_rounds=10, patience=5, proposal_patience=2,
                   log=lambda *a: None)
    assert "合法补丁" in res.stop_reason
    assert all(r.kind == "rejected" for r in res.rounds[1:])


def test_report_flags_safety_veto(tmp_path: Path):
    """报告里必须能一眼看出这轮是被安全档拦下的，而不是"分数不好看"。"""
    base = _safety_tree(tmp_path)
    tasks = [_task("T1", MAGIC_A, tier="safety"), _task("T2", MAGIC_B)]
    llm = ScriptedLLM([_patch_add(MAGIC_B)])
    res = run_loop(base, tasks, tmp_path / "run", llm=llm, improver=llm,
                   reps=1, max_rounds=1, patience=5,
                   log=lambda *a: None)
    md = render(res)
    assert "安全档一票否决" in md
    assert "T1" in md


def test_report_renders_without_crash(base_tree: Path, tmp_path: Path):
    tasks = [_task("T1", MAGIC_A)]
    llm = ScriptedLLM([_patch_add(MAGIC_A)])
    res = run_loop(base_tree, tasks, tmp_path / "run", llm=llm, improver=llm,
                   reps=2, max_rounds=3, patience=2, log=lambda *a: None)
    md = render(res)
    for section in ("## 1. 结论", "## 2. 曲线", "## 4. 失败分布迁移", "## 6. 停止条件"):
        assert section in md
    assert "±" in md, "报告必须给出离散度，不能只报均值"


def test_curve_json_is_serializable(base_tree: Path, tmp_path: Path):
    tasks = [_task("T1", MAGIC_A)]
    llm = ScriptedLLM([_patch_add(MAGIC_A)])
    res = run_loop(base_tree, tasks, tmp_path / "run", llm=llm, improver=llm,
                   reps=1, max_rounds=2, patience=2, log=lambda *a: None)
    json.dumps(res.as_dict(), ensure_ascii=False)  # 不抛异常即合格

def test_broke_stable_flags_parent_stable_to_cand_fail():
    """父树每轮都过、候选每轮都挂 —— 这就是回退，不管均值涨没涨。"""
    from harnessdrift.loop import _broke_stable

    parent = {"T01": [1, 1, 1], "T02": [0, 0, 1], "T03": [1, 0, 1]}
    cand = {"T01": [0, 0, 0], "T02": [1, 1, 1], "T03": [1, 1, 1]}
    assert _broke_stable(parent, cand) == ["T01"]


def test_broke_stable_ignores_parent_flaky():
    """父树自己就在抖的题，掉下去不能算候选的错。"""
    from harnessdrift.loop import _broke_stable

    parent = {"T03": [1, 0, 1]}
    cand = {"T03": [0, 0, 0]}
    assert _broke_stable(parent, cand) == []


def test_broke_stable_skips_tasks_absent_from_candidate():
    """--limit 跑子集时，候选没有的题不判回退，否则整条曲线都收不了。"""
    from harnessdrift.loop import _broke_stable

    parent = {"T01": [1, 1, 1], "T02": [1, 1, 1]}
    cand = {"T01": [1, 1, 1]}
    assert _broke_stable(parent, cand) == []


# ---------------- safety 档：一票否决 ----------------

def test_broke_safety_picks_only_safety_tier():
    from harnessdrift.loop import _broke_safety

    tiers = {"T01": "safety", "T02": "normal"}
    assert _broke_safety(["T01", "T02"], tiers) == ["T01"]
    assert _broke_safety(["T02"], tiers) == []
    assert _broke_safety(["T09"], tiers) == [], "查不到的档位按普通题算，不能因为缺档位就否决"


def test_safety_regressed_counts_flaky_as_regression():
    """2/3 挂也算回退。这是实测踩出来的：只认"全挂"会让题永久豁免。

    第一版守卫只看父树是否稳定通过。实测第 2 轮把 T12 拖成 [0,1,0]（三挂二），
    父树从那一刻起就不"稳定通过"了，于是这道题再也不会被任何一轮报出来 ——
    越改越松，守卫却一直报平安。所以锁定名单里的题，候选必须每轮都过。
    """
    from harnessdrift.loop import _safety_regressed

    assert _safety_regressed({"T01"}, {"T01": [0.0, 1.0, 0.0]}) == ["T01"], "抖动就是回退"
    assert _safety_regressed({"T01"}, {"T01": [1.0, 1.0, 1.0]}) == []
    assert _safety_regressed({"T01"}, {}) == [], "子集评测没跑这题，不判"


def test_locked_safety_only_takes_stable_tasks():
    """没稳住的题不该进锁定名单：它本来就没守住，锁它等于锁住噪音。"""
    from harnessdrift.loop import _locked_safety

    tasks = [_task("T1", MAGIC_A, tier="safety"), _task("T2", MAGIC_B, tier="safety"),
             _task("T3", MAGIC_A)]
    passes = {"T1": [1.0, 1.0], "T2": [1.0, 0.0], "T3": [1.0, 1.0]}
    assert _locked_safety(tasks, passes) == {"T1"}


def _safety_tree(tmp_path: Path) -> Path:
    """父树本来就答得出 MAGIC_A 的最小 harness。"""
    d = tmp_path / "base_safety"
    (d / "tool_descriptions").mkdir(parents=True)
    (d / "systemprompt.md").write_text(BASE_PROMPT + MAGIC_A + "\n", encoding="utf-8")
    (d / "tool_descriptions" / "lookup.md").write_text("查订单。\n", encoding="utf-8")
    return d


def test_safety_veto_blocks_even_accept_always(tmp_path: Path):
    """均值没跌（0.5 → 0.5），但挂掉的是 safety 题：连 always 也不收。

    这条是档位存在的全部理由。普通题挂一条是少收一单，净分还能是正的，
    可以用均值权衡；safety 题挂一条是给别人的订单退了款、凭空编了个订单号，
    没有"用别处多收几单补回来"这回事。
    """
    base = _safety_tree(tmp_path)
    tasks = [_task("T1", MAGIC_A, tier="safety"), _task("T2", MAGIC_B)]
    # 补丁把提示词整个换成只含 MAGIC_B：T2 修好、T1 挂掉，净分不动。
    llm = ScriptedLLM([_patch_add(MAGIC_B)])
    res = run_loop(base, tasks, tmp_path / "run", llm=llm, improver=llm,
                   reps=1, max_rounds=1, patience=5, accept="always",
                   log=lambda *a: None)

    r1 = res.rounds[1]
    assert r1.kind == "rejected", "always 也要被 safety 一票否决拦住"
    assert r1.broke_safety == ["T1"]
    assert r1.mean == res.rounds[0].mean, "净分没动 —— 拦它的不该是均值"


def test_no_safety_veto_restores_old_behaviour(tmp_path: Path):
    """关掉否决后同一份补丁会被接受：证明拦下它的确实是档位，不是别的规则。"""
    base = _safety_tree(tmp_path)
    tasks = [_task("T1", MAGIC_A, tier="safety"), _task("T2", MAGIC_B)]
    llm = ScriptedLLM([_patch_add(MAGIC_B)])
    res = run_loop(base, tasks, tmp_path / "run", llm=llm, improver=llm,
                   reps=1, max_rounds=1, patience=5, accept="always",
                   safety_veto=False, log=lambda *a: None)

    assert res.rounds[1].kind == "accepted"
    assert res.rounds[1].broke_safety == []


def test_safety_veto_does_not_fire_on_normal_task(tmp_path: Path):
    """普通题回退不触发否决（那由 nonregress 规则管），两条规则不能混成一条。"""
    base = _safety_tree(tmp_path)
    tasks = [_task("T1", MAGIC_A), _task("T2", MAGIC_B)]
    llm = ScriptedLLM([_patch_add(MAGIC_B)])
    res = run_loop(base, tasks, tmp_path / "run", llm=llm, improver=llm,
                   reps=1, max_rounds=1, patience=5, accept="always",
                   log=lambda *a: None)

    assert res.rounds[1].broke_safety == []
    assert res.rounds[1].kind == "accepted", "always 下普通题回退照旧接受"
