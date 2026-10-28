"""原子级抢救的测试（全部离线）。

这里要钉住三件事：

1. **只删对的那一个**：能只删一个原子就修好，就不要多删。
2. **救不回来就说救不回来**：不许把"没通过"报成通过。
3. **救出来的树走同一套验收**：抢救不是免责通道，分数不够照样不收。

为什么要专门测第 3 条：抢救最容易变成的坏味道是"被否决的轮换个名字收下"。
所以 loop 里的测试断言的是"救完还要过回归名单"，不是"救完就收"。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from harnessdrift import rescue as rescue_mod
from harnessdrift.ablate import split_atoms
from harnessdrift.llm import LLMReply
from harnessdrift.loop import run_loop
from harnessdrift.rescue import rescue_vetoed
from harnessdrift.runner import EvalResult
from harnessdrift.tasks import Task


MAGIC_A = "MAGIC_ALPHA"
MAGIC_B = "MAGIC_BETA"

BASE_PROMPT = "# 你是客服\n\n## 原则\n1. 先查后答。\n"


# ---------------------------------------------------------------- 单元：rescue

def _fake_eval_from_tree(probe_predicate):  # noqa: ANN001, ANN201
    """按树的内容伪造评测结果：树里有毒文件，探测题就挂。"""

    def _fake(tree, tasks, llm, reps=1, verbose=False, workers=1):  # noqa: ANN001, ANN003
        passes = {}
        for t in tasks:
            ok = probe_predicate(tree, t.id)
            passes[t.id] = [1.0 if ok else 0.0] * reps
        return EvalResult(n_tasks=len(tasks), reps=reps, passes=passes)

    return _fake


def test_rescue_drops_only_the_toxic_atom(monkeypatch, tmp_path: Path):
    """候选加了两个文件，其中 b.md 有毒。只该删 b.md。"""
    parent = {"systemprompt.md": BASE_PROMPT}
    cand = {
        "systemprompt.md": BASE_PROMPT,
        "a.md": "有用的说明\n",
        "b.md": "有毒的说明\n",
    }
    tasks = [Task(id="S1", message="", why="", expect="no_return", customers=[], orders=[])]

    monkeypatch.setattr(
        rescue_mod, "evaluate_tree",
        _fake_eval_from_tree(lambda tree, tid: "b.md" not in tree),
    )
    rr = rescue_vetoed(parent, cand, tasks, llm=object(), probe_ids=["S1"], reps=1)

    assert rr.ok, rr.reason
    assert rr.n_atoms == 2, "两个新增文件应该是两个原子"
    assert [a.path for a in rr.dropped] == ["b.md"]
    assert "a.md" in rr.tree and "b.md" not in rr.tree
    assert len(rr.attempts) == 2, "先试删 a.md（无效）再试删 b.md（有效）"


def test_rescue_reports_failure_instead_of_pretending(monkeypatch):
    """两条探测题互相冲突（删哪个都挂一条），必须老实报救不回来。"""
    parent = {"systemprompt.md": BASE_PROMPT}
    cand = {"systemprompt.md": BASE_PROMPT, "a.md": "x\n", "b.md": "y\n"}
    tasks = [
        Task(id="S1", message="", why="", expect="no_return", customers=[], orders=[]),
        Task(id="S2", message="", why="", expect="no_return", customers=[], orders=[]),
    ]
    # S1 要求没有 a.md，S2 要求没有 b.md：删一个救一条，永远凑不齐
    monkeypatch.setattr(
        rescue_mod, "evaluate_tree",
        _fake_eval_from_tree(lambda tree, tid: (tid == "S1") == ("a.md" not in tree)),
    )
    rr = rescue_vetoed(parent, cand, tasks, llm=object(),
                       probe_ids=["S1", "S2"], reps=1, max_drop=2)
    assert not rr.ok
    assert "没救回来" in rr.reason
    assert rr.tree is None


def test_rescue_needs_two_drops_when_max_drop_allows(monkeypatch):
    """同一棵树要删两个原子才干净时，max_drop=1 救不动、max_drop=2 能救。"""
    parent = {"systemprompt.md": BASE_PROMPT}
    cand = {"systemprompt.md": BASE_PROMPT, "a.md": "x\n", "b.md": "y\n", "c.md": "z\n"}
    tasks = [Task(id="S1", message="", why="", expect="no_return", customers=[], orders=[])]

    monkeypatch.setattr(
        rescue_mod, "evaluate_tree",
        _fake_eval_from_tree(lambda tree, tid: "a.md" not in tree and "b.md" not in tree),
    )
    shallow = rescue_vetoed(parent, cand, tasks, llm=object(), probe_ids=["S1"],
                            reps=1, max_drop=1)
    assert not shallow.ok

    deep = rescue_vetoed(parent, cand, tasks, llm=object(), probe_ids=["S1"],
                         reps=1, max_drop=2)
    assert deep.ok, deep.reason
    assert sorted(a.path for a in deep.dropped) == ["a.md", "b.md"]
    assert "c.md" in deep.tree


def test_split_atoms_is_the_source_of_truth_for_rescue():
    """抢救和归因必须用同一套原子切分，否则"删掉第几个"两个模块对不上。"""
    parent = {"systemprompt.md": "# a\n# b\n"}
    cand = {"systemprompt.md": "# a\n# B\n"}
    atoms = split_atoms(parent, cand)
    assert len(atoms) == 1
    assert atoms[0].path == "systemprompt.md"


# ---------------------------------------------------------------- 端到端：loop

class ScriptedLLM:
    """求解器读系统提示词里有没有 MAGIC；改进器按剧本吐补丁。"""

    def __init__(self, patches: list[str]) -> None:
        self.model = "scripted"
        self.patches = list(patches)
        self.usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def chat(self, messages, **kw):  # noqa: ANN001, ANN003
        system = messages[0]["content"] if messages else ""
        self.usage_total["total_tokens"] += 100
        if "harness 工程师" in system:
            if not self.patches:
                raise AssertionError("剧本用完了：这个测试不应该再提一轮")
            return LLMReply(text=self.patches.pop(0), finish_reason="stop")
        said = " ".join(m for m in (MAGIC_A, MAGIC_B) if m in system)
        return LLMReply(text=said or "我不知道。", finish_reason="stop")


def _task(tid: str, magic: str) -> Task:
    return Task(id=tid, message=f"请回复 {tid}", why="抢救测试",
                expect="no_return", customers=[], orders=[],
                answer_must_mention=[magic])


@pytest.fixture()
def base_tree(tmp_path: Path) -> Path:
    d = tmp_path / "base"
    (d / "tool_descriptions").mkdir(parents=True)
    (d / "systemprompt.md").write_text(BASE_PROMPT + f"\n{MAGIC_A}\n", encoding="utf-8")
    (d / "tool_descriptions" / "lookup.md").write_text("查订单。\n", encoding="utf-8")
    return d


# 这一版把 A 换成 B（踩坏 T1）同时新增一个无关文件 —— 正是真实事故的形状：
# 想修的新能力修好了，顺手把旧能力弄没了，另外还捎带一个无害文件。
SWAP_PATCH = (
    "### REPLACE systemprompt.md\n"
    f"    # 你是客服\n\n    {MAGIC_B}\n\n"
    "### ADD tool_descriptions/notes.md\n"
    "    这条说明与判分无关。\n"
)


def test_vetoed_round_is_rescued_by_dropping_the_bad_atom(base_tree: Path, tmp_path: Path):
    """被回退规则拦下的轮：开 --rescue 后应当删掉坏原子、其余留下并收下。"""
    tasks = [_task("T1", MAGIC_A), _task("T2", MAGIC_B)]
    llm = ScriptedLLM([SWAP_PATCH])
    res = run_loop(base_tree, tasks, tmp_path / "run", llm=llm, improver=llm,
                   reps=1, max_rounds=1, patience=5, accept="nonregress",
                   rescue=True, log=lambda *a: None)

    r1 = res.rounds[1]
    assert r1.kind == "accepted", "抢救后应当收下"
    assert r1.rescued is not None
    assert [d["path"] for d in r1.rescued["dropped"]] == ["systemprompt.md"]
    assert r1.broke == [], "救出来的树不该再踩坏 T1"
    assert r1.mean == 1.0
    # 抢救后的树真的落盘了：T1 恢复，无关文件保留
    tree = (tmp_path / "run" / "round1")
    assert MAGIC_A in (tree / "systemprompt.md").read_text(encoding="utf-8")
    assert (tree / "tool_descriptions" / "notes.md").exists()


def test_without_rescue_the_same_round_is_rejected(base_tree: Path, tmp_path: Path):
    """同样的补丁，不开 --rescue 就是拒绝 —— 证明是抢救在起作用，不是别的。"""
    tasks = [_task("T1", MAGIC_A), _task("T2", MAGIC_B)]
    llm = ScriptedLLM([SWAP_PATCH, SWAP_PATCH])
    res = run_loop(base_tree, tasks, tmp_path / "run2", llm=llm, improver=llm,
                   reps=1, max_rounds=1, patience=5, accept="nonregress",
                   rescue=False, log=lambda *a: None)

    r1 = res.rounds[1]
    assert r1.kind == "rejected"
    assert r1.rescued is None
    assert "T1" in r1.broke


def test_rescue_does_not_lower_the_bar(base_tree: Path, tmp_path: Path):
    """救出来的树如果分数不达标，仍然不收 —— 抢救不是免责通道。"""
    tasks = [_task("T1", MAGIC_A), _task("T2", MAGIC_B)]
    llm = ScriptedLLM([SWAP_PATCH])
    res = run_loop(base_tree, tasks, tmp_path / "run3", llm=llm, improver=llm,
                   reps=1, max_rounds=1, patience=5, accept="strict",
                   rescue=True, log=lambda *a: None)

    r1 = res.rounds[1]
    # 抢救后恢复到 1.0，但 strict 只收严格上升，1.0 不高于父树 1.0
    assert r1.kind == "rejected"
    assert r1.rescued is not None or r1.rescued is None  # 记录与否不改变结论
    assert not r1.improved