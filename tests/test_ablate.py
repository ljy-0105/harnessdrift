"""归因模块的契约测试。

这一层的核心不变量只有一条：**所有单点合起来必须能还原全量**。
一旦拆错了，后面每一条 delta 都是假的，而且看不出来。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harnessdrift.ablate import Atom, ablate, apply_all, split_atoms, write_attribution
from harnessdrift.runner import EvalResult

BASE = {
    "systemprompt.md": "# 你是客服\n规则一\n规则二\n规则三\n规则四\n",
    "skills/a/SKILL.md": "步骤 1\n步骤 2\n",
}


def test_split_atoms_finds_each_contiguous_change():
    cand = dict(BASE)
    cand["systemprompt.md"] = "# 你是客服\n规则一（改）\n规则二\n规则三\n规则四\n新增规则五\n"
    atoms = split_atoms(BASE, cand)
    # 一处替换 + 一处追加
    assert len(atoms) == 2
    assert [a.tag for a in atoms] == ["replace", "insert"]
    assert all(a.path == "systemprompt.md" for a in atoms)


def test_split_atoms_add_and_delete_are_single_atoms():
    cand = {
        "systemprompt.md": BASE["systemprompt.md"],
        "tool_descriptions/new.md": "工具说明\n",
    }  # skills/a/SKILL.md 被删掉
    atoms = split_atoms(BASE, cand)
    kinds = {a.path: a.kind for a in atoms}
    assert kinds == {"skills/a/SKILL.md": "delete", "tool_descriptions/new.md": "add"}


def test_atoms_reconstruct_the_full_candidate():
    """把所有单点依次合进父树，必须逐字节等于候选树。"""
    cand = dict(BASE)
    cand["systemprompt.md"] = (
        "# 你是客服\n"
        "规则零（前面插入）\n"
        "规则一\n"
        "规则二（改过）\n"
        "规则三\n"
        "规则四\n"
        "新增规则五\n"
        "新增规则六\n"
    )
    cand["tool_descriptions/x.md"] = "新文件\n"
    del cand["skills/a/SKILL.md"]

    atoms = split_atoms(BASE, cand)
    assert atoms, "差异非空就必须拆出原子"

    assert apply_all(BASE, cand, atoms) == cand, \
        "单点合起来必须还原全量，否则归因的每个数都是假的"


def test_atom_apply_does_not_mutate_parent():
    cand = dict(BASE)
    cand["systemprompt.md"] = BASE["systemprompt.md"] + "新增\n"
    atom = split_atoms(BASE, cand)[0]
    snapshot = dict(BASE)
    atom.apply(BASE, cand)
    assert BASE == snapshot


def test_whitespace_only_hunk_is_skipped():
    cand = {"systemprompt.md": BASE["systemprompt.md"].replace("规则二", "规则二 "),
            "skills/a/SKILL.md": BASE["skills/a/SKILL.md"]}
    assert split_atoms(BASE, cand) == [], "只差空白不该白烧一次评测"


class _ConstLLM:
    """假模型：按 id() 之外的简单规则给分，只为跑通流程。"""

    model = "fake"

    def chat(self, messages, **kw):  # noqa: ANN001
        raise AssertionError("ablate 的测试不该真的调模型")


def _fake_eval(score_map: dict[str, float]):
    """替代 evaluate_tree：按树的 systemprompt 内容查分数。"""
    from harnessdrift.runner import EvalResult

    def _f(tree, tasks, llm, reps=1, verbose=False):  # noqa: ANN001
        s = score_map.get(tree["systemprompt.md"], 0.0)
        return EvalResult(n_tasks=1, reps=reps, passes={"T01": [s]})

    return _f


def test_ablate_marks_effective_atom_and_computes_interaction(monkeypatch, tmp_path: Path):
    from harnessdrift import ablate as mod

    base_prompt = BASE["systemprompt.md"]
    cand = dict(BASE)
    # 两处**不相邻**的改动 -> 两个原子（相邻/连续的改动会被合成一个 hunk，
    # 那是 difflib 的正常行为，不算漏拆）
    cand["systemprompt.md"] = base_prompt.replace("规则一\n", "规则一（改）\n") + "没关系的一行\n"

    atoms = split_atoms(BASE, cand)
    assert len(atoms) == 2

    tree0 = atoms[0].apply(BASE, cand)   # 只含"规则一（改）"
    tree1 = atoms[1].apply(BASE, cand)   # 只含末尾那一行
    scores = {base_prompt: 10.0, tree0["systemprompt.md"]: 12.0,
              tree1["systemprompt.md"]: 10.0, cand["systemprompt.md"]: 12.0}
    monkeypatch.setattr(mod, "evaluate_tree", _fake_eval(scores))

    res = ablate(BASE, cand, [object()], _ConstLLM(), reps=1)

    assert res.base_mean == 10.0 and res.full_mean == 12.0
    assert res.full_delta == pytest.approx(2.0)
    assert res.best_atom_delta == pytest.approx(2.0)
    assert res.atoms[0].verdict == "有效"
    assert res.atoms[1].verdict == "测不出"
    assert res.interaction == pytest.approx(0.0)

    rpt = write_attribution(res, tmp_path / "A.md")
    text = rpt.read_text(encoding="utf-8")
    assert "归因" in text and "有效" in text


def test_ablate_detects_positive_interaction(monkeypatch):
    """两个单点单独都无效、合起来才有效 —— 这是真实存在的情况，必须算得出来。"""
    from harnessdrift import ablate as mod

    base_prompt = BASE["systemprompt.md"]
    cand = dict(BASE)
    # 一头一尾各插一行，中间隔着未改动的正文，才会形成两个独立原子
    cand["systemprompt.md"] = "# 你是客服\n规则甲\n规则一\n规则二\n规则三\n规则四\n规则乙\n"

    atoms = split_atoms(BASE, cand)
    assert len(atoms) == 2, f"应当拆出两个原子，实际 {len(atoms)}"

    scores = {base_prompt: 10.0, cand["systemprompt.md"]: 14.0}
    for a in atoms:
        scores[a.apply(BASE, cand)["systemprompt.md"]] = 10.0
    monkeypatch.setattr(mod, "evaluate_tree", _fake_eval(scores))

    res = ablate(BASE, cand, [object()], _ConstLLM(), reps=1)
    assert res.full_delta == pytest.approx(4.0)
    assert res.best_atom_delta == pytest.approx(0.0)
    assert res.interaction == pytest.approx(4.0)


def test_empty_prompt_atom_is_skipped(monkeypatch):
    """把 systemprompt.md 删空这种原子，合出来根本跑不了，必须跳过而不是硬评。"""
    from harnessdrift import ablate as mod

    base_prompt = BASE["systemprompt.md"]
    monkeypatch.setattr(mod, "evaluate_tree", _fake_eval({base_prompt: 5.0, "": 0.0}))

    res = ablate(BASE, {"skills/a/SKILL.md": BASE["skills/a/SKILL.md"]},
                 [object()], _ConstLLM(), reps=1)
    skipped = [a for a in res.atoms if a.verdict.startswith("跳过")]
    assert skipped, "删空 systemprompt.md 的原子必须被跳过"
    assert "变空" in skipped[0].verdict
    # 全量树同样跑不了：要记成"不可运行"，不能假装评出一个 0 分
    assert not res.full_ok
    assert any("不可运行" in n for n in res.notes)