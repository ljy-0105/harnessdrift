"""报告里留出列的画法。

单独一个文件是因为这里的断言都是"读起来对不对"：
数字算对了但表头标错、留出列被当成决策依据，报告就是错的。
"""
from __future__ import annotations

import json

from harnessdrift.loop import LoopResult, Round
from harnessdrift.report import render


def _round(rid: int, kind: str, mean: float, *, train: float | None, hold: float | None,
           n_ho: int = 0, n_tasks: int = 10, solved: int = 0) -> Round:
    return Round(
        round_id=rid, kind=kind, harness_dir=f"/tmp/r{rid}", tree_hash="x",
        mean=mean, std=0.5, reps=3, n_tasks=n_tasks, solved=solved, elapsed_s=10.0,
        failure_hist={}, stable_fail=[], flaky={}, passes={}, errors={}, tokens={},
        mean_train=train, mean_holdout=hold, n_holdout=n_ho,
    )


def _res(*rounds: Round) -> LoopResult:
    return LoopResult(rounds=list(rounds), stop_reason="测试",
                      config={"solver_model": "s", "improver_model": "i",
                              "n_tasks": rounds[0].n_tasks, "reps": 3})


def test_report_prints_a_holdout_column_only_when_it_exists():
    plain = render(_res(_round(0, "base", 10.0, train=None, hold=None)))
    assert "holdout" not in plain, "没开留出集就不该出现这一列"

    with_ho = render(_res(_round(0, "base", 10.0, train=8.0, hold=2.0, n_ho=2)))
    assert "holdout" in with_ho


def test_holdout_section_states_the_rule():
    txt = render(_res(_round(0, "base", 10.0, train=8.0, hold=2.0, n_ho=2)))
    assert "不参与决策" in txt
    assert "不喂给提案端" in txt or "没进过提案端" in txt
    # 评分口径要说清楚：决策按训练列算
    assert "训练列 8" in txt


def test_holdout_delta_row_cannot_be_mistaken_for_a_decision_input():
    txt = render(_res(
        _round(0, "base", 10.0, train=8.0, hold=2.0, n_ho=2),
        _round(1, "accepted", 11.0, train=9.0, hold=2.0, n_ho=2),
    ))
    assert "留出集（不参与决策）" in txt
    assert "+1.00" in txt  # 训练列涨了 1 分


def test_curve_json_keeps_the_split_fields():
    """curve.json 是看板和别人的分析入口，字段丢了就再也算不出来。"""
    d = _res(_round(0, "base", 10.0, train=8.0, hold=2.0, n_ho=2)).as_dict()
    r0 = d["rounds"][0]
    assert r0["mean_train"] == 8.0
    assert r0["mean_holdout"] == 2.0
    assert r0["n_holdout"] == 2
    json.dumps(d)  # 必须可序列化
