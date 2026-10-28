"""留出集与回滚的测试。

这两件事是一对：留出集回答"这次涨分是不是真的"，回滚回答"如果是假的，怎么退回去"。
所以放在一个文件里——它们合起来才是"敢让它改线上"的最低条件。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harnessdrift.loop import run_loop, _split_score
from harnessdrift.rollback import apply_rollback, plan_rollback
from harnessdrift.tasks import Task

from test_loop import MAGIC_A, MAGIC_B, ScriptedLLM, _patch_add

BASE_PROMPT = "# 你是客服\n\n## 原则\n1. 先查后答。\n"


def _t(tid: str, magic: str, split: str = "train") -> Task:
    return Task(id=tid, message=f"请回复 {tid}", why="测试用", expect="no_return",
                customers=[], orders=[], split=split, answer_must_mention=[magic])


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    d = tmp_path / "base"
    (d / "tool_descriptions").mkdir(parents=True)
    (d / "systemprompt.md").write_text(BASE_PROMPT, encoding="utf-8")
    (d / "tool_descriptions" / "lookup.md").write_text("查订单。\n", encoding="utf-8")
    return d


# ---------------- 留出集 ----------------

def test_split_score_only_counts_its_own_ids():
    from harnessdrift.runner import EvalResult
    ev = EvalResult(n_tasks=3, reps=1)
    ev.passes = {"A": [1.0], "B": [0.0], "C": [1.0]}
    assert _split_score(ev, ["A", "C"]) == 2.0, "口径是总分：两题各 1.0，合计 2.0"
    assert _split_score(ev, ["B"]) == 0.0
    assert _split_score(ev, ["A", "B"]) == 1.0


def test_split_score_ignores_ids_missing_from_the_eval():
    """--limit 只跑了训练列时，留出列不能算成 0 分。"""
    from harnessdrift.runner import EvalResult
    ev = EvalResult(n_tasks=1, reps=1)
    ev.passes = {"A": [1.0]}
    assert _split_score(ev, ["A"]) == 1.0
    assert _split_score(ev, ["Z"]) is None


def test_holdout_failures_never_reach_the_improver(tree: Path, tmp_path: Path):
    """核心不变式：留出题的错误样本不许出现在提案端。

    这是留出集唯一的价值来源。一旦泄漏，留出列就等于训练列，
    那它测出来的"泛化"是自欺欺人 —— 而且这种错永远不会报错，只会悄悄失效。
    """
    tasks = [_t("T1", MAGIC_A), _t("H1", MAGIC_B, split="holdout")]
    llm = ScriptedLLM([_patch_add(MAGIC_A)])
    res = run_loop(tree, tasks, tmp_path / "run", llm=llm, improver=llm,
                   reps=1, max_rounds=1, patience=1, holdout=True, accept="nonregress")
    # 让 ScriptedLLM 把它收到的提示词记下来
    prompts = [c for c in llm.seen if "harness 工程师" in c]
    assert prompts, "改进器应该被调用过"
    assert "请回复 H1" not in prompts[0], "留出题的错误样本泄漏进了提案端"
    assert "请回复 T1" in prompts[0], "训练题的错误样本应该在里面"
    assert res.config["holdout"] is True
    assert res.config["n_holdout"] == 1


def test_holdout_score_is_reported_but_not_used_for_decision(tree: Path, tmp_path: Path):
    """留出列掉分也不影响接受 —— 它的角色是尺子，不是裁判。

    如果拿它做决策，它就被优化目标污染了，下次再想看泛化就没有干净数据了。
    """
    tasks = [_t("T1", MAGIC_A), _t("H1", MAGIC_B, split="holdout")]
    llm = ScriptedLLM([_patch_add(MAGIC_A)])
    res = run_loop(tree, tasks, tmp_path / "run", llm=llm, improver=llm,
                   reps=1, max_rounds=1, patience=1, holdout=True, accept="strict")
    r1 = res.rounds[1]
    assert r1.kind == "accepted"
    assert r1.mean_train == 1.0
    assert r1.mean_holdout == 0.0
    assert r1.n_holdout == 1


def test_without_holdout_flag_the_split_is_ignored(tree: Path, tmp_path: Path):
    tasks = [_t("T1", MAGIC_A), _t("H1", MAGIC_B, split="holdout")]
    llm = ScriptedLLM([_patch_add(MAGIC_A)])
    res = run_loop(tree, tasks, tmp_path / "run", llm=llm, improver=llm,
                   reps=1, max_rounds=1, patience=1, holdout=False, accept="nonregress")
    assert res.config["holdout"] is False
    assert res.rounds[1].mean_holdout is None
    # 不开 holdout 时两题都算训练题，patch 只加了 A 那一句：T1 过、H1 挂，总分 1.0
    assert res.rounds[1].mean == 1.0


# ---------------- token 预算 ----------------

def test_token_budget_stops_the_loop(tree: Path, tmp_path: Path):
    tasks = [_t("T1", MAGIC_A)]
    llm = ScriptedLLM([_patch_add(MAGIC_A), _patch_add(MAGIC_B)])
    res = run_loop(tree, tasks, tmp_path / "run", llm=llm, improver=llm,
                   reps=1, max_rounds=3, patience=3, accept="nonregress",
                   max_tokens=1)  # ScriptedLLM 不记账，但 usage_total 会累加
    assert "token 预算" in (res.stop_reason or "")


# ---------------- 回滚 ----------------

def _fake_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "best").mkdir(parents=True)
    (run / "best" / "systemprompt.md").write_text("# 好的\n", encoding="utf-8")
    (run / "best" / "notes.md").write_text("新文件\n", encoding="utf-8")
    (run / "_rejected" / "round2").mkdir(parents=True)
    (run / "_rejected" / "round2" / "systemprompt.md").write_text("# 坏的\n", encoding="utf-8")
    return run


def test_plan_reports_diffs_without_touching_anything(tmp_path: Path):
    run = _fake_run(tmp_path)
    live = tmp_path / "live"
    live.mkdir()
    (live / "systemprompt.md").write_text("# 现在的\n", encoding="utf-8")
    (live / "old.md").write_text("旧文件\n", encoding="utf-8")

    plan = plan_rollback(run, "best", live)
    assert plan.added == ["notes.md"]
    assert plan.changed == ["systemprompt.md"]
    assert plan.removed == ["old.md"]
    # 计划阶段不落盘
    assert (live / "systemprompt.md").read_text(encoding="utf-8") == "# 现在的\n"
    assert (live / "old.md").exists()


def test_apply_replaces_the_tree_and_backs_up(tmp_path: Path):
    run = _fake_run(tmp_path)
    live = tmp_path / "live"
    live.mkdir()
    (live / "systemprompt.md").write_text("# 现在的\n", encoding="utf-8")
    (live / "old.md").write_text("旧文件\n", encoding="utf-8")

    backup = apply_rollback(plan_rollback(run, "best", live))
    assert (live / "systemprompt.md").read_text(encoding="utf-8") == "# 好的\n"
    assert (live / "notes.md").exists()
    assert not (live / "old.md").exists(), "回滚必须删掉新树里没有的文件，否则既不是新树也不是旧树"
    assert backup is not None
    assert (backup / "systemprompt.md").read_text(encoding="utf-8") == "# 现在的\n"
    assert (backup / "old.md").exists()


def test_dry_run_writes_nothing(tmp_path: Path):
    run = _fake_run(tmp_path)
    live = tmp_path / "live"
    live.mkdir()
    (live / "systemprompt.md").write_text("# 现在的\n", encoding="utf-8")
    assert apply_rollback(plan_rollback(run, "best", live), dry_run=True) is None
    assert (live / "systemprompt.md").read_text(encoding="utf-8") == "# 现在的\n"


def test_rollback_refuses_a_rejected_round(tmp_path: Path):
    """被规则否掉的候选不能变成线上状态 —— 回滚不是绕过接受规则的后门。"""
    run = _fake_run(tmp_path)
    with pytest.raises(FileNotFoundError) as e:
        plan_rollback(run, "round2", tmp_path / "live")
    assert "_rejected" in str(e.value)


def test_rollback_rejects_nonsense_target(tmp_path: Path):
    with pytest.raises(ValueError):
        plan_rollback(_fake_run(tmp_path), "latest", tmp_path / "live")


def test_rollback_into_missing_target_creates_it(tmp_path: Path):
    run = _fake_run(tmp_path)
    live = tmp_path / "new_live"
    backup = apply_rollback(plan_rollback(run, "best", live))
    assert backup is None
    assert (live / "systemprompt.md").exists()