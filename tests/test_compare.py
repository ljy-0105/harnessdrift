"""消融对照的渲染测试：只喂假 curve.json，不花钱。"""

from __future__ import annotations

import json
from pathlib import Path

from harnessdrift.compare import load_curve, render_comparison


def _write_curve(root: Path, accept: str, baseline: float, rounds: list[dict],
                 final: float, solved0: int, solved1: int) -> Path:
    d = root / f"ablate_{accept}"
    d.mkdir(parents=True)
    payload = {
        "config": {"accept": accept},
        "stop_reason": "patience",
        "total_elapsed_s": 100.0,
        "baseline": {"mean": baseline, "std": 0.5, "solved": solved0,
                     "n_tasks": 38, "reps": 3},
        "final": {"mean": final, "std": 0.4, "round_id": 1, "solved": solved1},
        "rounds": [{"round_id": 0, "kind": "base", "mean": baseline, "std": 0.5,
                    "solved": solved0, "improved": False, "neutral": False,
                    "broke": []}] + rounds,
    }
    (d / "curve.json").write_text(json.dumps(payload, ensure_ascii=False),
                                  encoding="utf-8")
    return d


def test_load_curve_reads_accept_and_metrics(tmp_path: Path):
    d = _write_curve(tmp_path, "strict", 30.0, [], 30.0, 28, 28)
    c = load_curve(d)
    assert c.accept == "strict"
    assert c.delta == 0.0
    assert c.accepted == 0 and c.rejected == 0
    assert c.n_tasks == 38


def test_comparison_flags_broke_and_counts(tmp_path: Path):
    rounds = [
        {"round_id": 1, "kind": "accepted", "mean": 33.0, "std": 0.6, "solved": 31,
         "improved": True, "neutral": False, "broke": []},
        {"round_id": 2, "kind": "rejected", "mean": 31.0, "std": 0.6, "solved": 29,
         "improved": False, "neutral": False, "broke": ["T07", "T34"]},
    ]
    d = _write_curve(tmp_path, "always", 32.0, rounds, 33.0, 30, 31)
    c = load_curve(d)
    assert c.accepted == 1 and c.rejected == 1
    assert c.broke_total == 0, "被拒绝的轮次不算踩坏（父树没被替换）"

    text = render_comparison([c])
    assert "消融对照" in text
    assert "无条件替换" in text, "always 档要讲清楚它是什么"
    assert "每档只跑了一次" in text, "必须写明这张表不能用来排名"


def test_comparison_accepts_multiple_curves(tmp_path: Path):
    a = load_curve(_write_curve(tmp_path, "strict", 31.0, [], 31.0, 29, 29))
    b = load_curve(_write_curve(tmp_path, "always", 31.0, [], 30.0, 29, 28))
    text = render_comparison([a, b])
    assert "`strict`" in text and "`always`" in text
    assert "按**最后留下的那棵树**排" in text


def test_empty_comparison_is_honest():
    assert "没有找到曲线" in render_comparison([])


def test_last_accepted_ignores_rejected_candidates(tmp_path: Path):
    """全拒的曲线：最后留下的就是基线，不能把被拒的候选当交付物。"""
    rounds = [
        {"round_id": 1, "kind": "rejected", "mean": 36.0, "std": 0.0, "solved": 35,
         "improved": False, "neutral": False, "broke": ["T12"]},
        {"round_id": 2, "kind": "rejected", "mean": 36.33, "std": 0.0, "solved": 35,
         "improved": False, "neutral": False, "broke": ["T12"]},
    ]
    c = load_curve(_write_curve(tmp_path, "nonregress", 32.0, rounds, 32.0, 30, 30))
    assert c.last["mean"] == 36.33, "last 是最后一行，可能是被拒的"
    assert c.last_accepted_mean == 32.0, "被拒的候选不是交付物"


def test_lost_stable_reads_passes_not_broke(tmp_path: Path):
    """always 档不写 broke 字段，丢失必须从 passes 反算。"""
    d = tmp_path / "ablate_always"
    d.mkdir(parents=True)
    base_passes = {"T01": [1, 1, 1], "T02": [1, 1, 1], "T03": [0, 1, 0]}
    round1 = {**base_passes, "T01": [0, 0, 0]}
    payload = {
        "config": {"accept": "always"},
        "stop_reason": "patience",
        "total_elapsed_s": 10.0,
        "baseline": {"mean": 2.0, "std": 0.0, "solved": 2},
        "final": {"mean": 1.0, "std": 0.0, "round_id": 1, "solved": 1},
        "rounds": [
            {"round_id": 0, "kind": "base", "mean": 2.0, "std": 0.0, "solved": 2,
             "passes": base_passes},
            {"round_id": 1, "kind": "accepted", "mean": 1.0, "std": 0.0, "solved": 1,
             "passes": round1, "broke": []},
        ],
    }
    (d / "curve.json").write_text(json.dumps(payload, ensure_ascii=False),
                                  encoding="utf-8")
    c = load_curve(d)
    assert c.lost_stable() == ["T01"], "T03 基线就不稳定，不算丢失"
    assert "T01" in render_comparison([c])