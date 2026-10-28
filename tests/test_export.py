"""看板导出的测试。

两件事必须钉住，否则看板会"看起来对但数字错"：
1. `Δ` 是相对**父树**算的 —— 被拒绝的候选不是父树，下一轮仍从父树出发；
2. 模板占位符只能出现一次 —— 它曾经同时出现在头部注释里，数据被内联两遍，
   页面照样能打开，但注释里躺着一份 JSON，体积翻倍。
"""

import json

import pytest

from harnessdrift.export import build_payload, render_html, export


def _curve(tmp_path, *, rounds, baseline_mean=30.0):
    run = tmp_path / "run1"
    run.mkdir()
    (run / "curve.json").write_text(json.dumps({
        "when": "2025-01-01T00:00:00",
        "config": {"reps": 3, "n_tasks": 2, "accept": "nonregress"},
        "stop_reason": "patience",
        "total_elapsed_s": 100.0,
        "baseline": {"mean": baseline_mean, "std": 0.5},
        "final": {"mean": rounds[-1]["mean"], "std": rounds[-1]["std"], "round_id": rounds[-1]["round_id"]},
        "delta": rounds[-1]["mean"] - baseline_mean,
        "rounds": [{"round_id": 0, "kind": "base", "mean": baseline_mean, "std": 0.5,
                    "solved": 1, "n_tasks": 2, "elapsed_s": 30.0,
                    "passes": {"T1": [1, 1, 1], "T2": [0, 0, 0]}, "flaky": {}}] + rounds,
    }, ensure_ascii=False), encoding="utf-8")
    return run


def _tasks(tmp_path):
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps([
        {"id": "T1", "message": "退一下", "why": "正例", "expect": "return_created"},
        {"id": "T2", "message": "能退吗", "why": "反例", "expect": "clarify"},
    ], ensure_ascii=False), encoding="utf-8")
    return p


def _rnd(rid, mean, kind, **kw):
    d = {"round_id": rid, "kind": kind, "mean": mean, "std": 0.0, "solved": 2,
         "n_tasks": 2, "elapsed_s": 10.0, "passes": {}, "flaky": {}}
    d.update(kw)
    return d


def test_delta_is_measured_against_the_parent_tree(tmp_path):
    """连续被拒时父树不动：两轮的 Δ 都是相对基线算的，不是相对上一轮候选。"""
    run = _curve(tmp_path, rounds=[
        _rnd(1, 28.0, "rejected"),
        _rnd(2, 32.0, "rejected"),
        _rnd(3, 33.0, "accepted"),
    ])
    p = build_payload(run, tasks_path=_tasks(tmp_path))
    rs = {r["round_id"]: r for r in p["rounds"]}
    assert rs[1]["delta"] == -2.0     # 28 - 30
    assert rs[2]["delta"] == 2.0      # 32 - 30（父树仍是基线）
    assert rs[3]["delta"] == 3.0      # 33 - 30
    assert rs[0]["delta"] is None


def test_accepted_round_becomes_the_next_parent(tmp_path):
    run = _curve(tmp_path, rounds=[
        _rnd(1, 33.0, "accepted"),
        _rnd(2, 35.0, "accepted"),
        _rnd(3, 34.0, "rejected"),
    ])
    p = build_payload(run, tasks_path=_tasks(tmp_path))
    rs = {r["round_id"]: r for r in p["rounds"]}
    assert rs[2]["delta"] == 2.0   # 35 - 33
    assert rs[3]["delta"] == -1.0  # 34 - 35（父树是第 2 轮那棵）


def test_broke_and_changed_survive_the_export(tmp_path):
    run = _curve(tmp_path, rounds=[
        _rnd(1, 26.0, "rejected", broke=["T1"], changed={"modified": ["systemprompt.md"]}),
    ])
    p = build_payload(run, tasks_path=_tasks(tmp_path))
    r1 = [r for r in p["rounds"] if r["round_id"] == 1][0]
    assert r1["broke"] == ["T1"]
    assert r1["changed"]["modified"] == ["systemprompt.md"]
    assert r1["accepted"] is False


def test_patch_markdown_is_read_from_the_round_tree(tmp_path):
    run = _curve(tmp_path, rounds=[_rnd(1, 33.0, "accepted", harness_dir=str(tmp_path / "run1" / "round1"))])
    d = tmp_path / "run1" / "round1"
    d.mkdir()
    (d / "PATCH.md").write_text("### REPLACE systemprompt.md\n    新提示\n", encoding="utf-8")
    p = build_payload(run, tasks_path=_tasks(tmp_path))
    r1 = [r for r in p["rounds"] if r["round_id"] == 1][0]
    assert "REPLACE systemprompt.md" in r1["patch"]


def test_patch_found_even_with_windows_style_harness_dir(tmp_path):
    """curve.json 里记的是 `runs\\curve3\\round1` 这种 Windows 路径。

    Linux 上把这串当文件名就找不到补丁，页面照样能开、只是补丁全空 ——
    一个不报错的功能失效。所以这里按轮号兜底。
    """
    run = _curve(tmp_path, rounds=[_rnd(1, 33.0, "accepted", harness_dir="runs\\x\\round1")])
    d = run / "round1"
    d.mkdir()
    (d / "PATCH.md").write_text("### ADD a.md\n    x\n", encoding="utf-8")
    p = build_payload(run, tasks_path=_tasks(tmp_path))
    r1 = [r for r in p["rounds"] if r["round_id"] == 1][0]
    assert "ADD a.md" in r1["patch"]


def test_render_inlines_data_exactly_once(tmp_path):
    run = _curve(tmp_path, rounds=[_rnd(1, 33.0, "accepted")])
    payload = build_payload(run, tasks_path=_tasks(tmp_path))
    html = render_html(payload)
    assert html.count("window.__DATA__ = ") == 1
    assert "/*__DATA__*/" not in html


def test_render_escapes_script_closer_in_task_text(tmp_path):
    """任务文案里出现 </script> 不能把脚本提前闭合。"""
    run = _curve(tmp_path, rounds=[_rnd(1, 33.0, "accepted")])
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([
        {"id": "T1", "message": "客户说 </script><b>x</b>", "why": "字面量", "expect": "clarify"},
    ], ensure_ascii=False), encoding="utf-8")
    html = render_html(build_payload(run, tasks_path=tasks))
    assert html.count("</script>") == 2          # 只有模板自带的那两个
    assert "<\\/script>" in html                 # 数据里那个被转义了


def test_template_must_contain_exactly_one_placeholder(tmp_path):
    tpl = tmp_path / "t.html"
    tpl.write_text("<p>没有占位符</p>", encoding="utf-8")
    with pytest.raises(ValueError, match="恰好出现一次"):
        render_html({"rounds": []}, template=tpl)


def test_export_writes_a_file(tmp_path):
    run = _curve(tmp_path, rounds=[_rnd(1, 33.0, "accepted")])
    out = tmp_path / "dist" / "index.html"
    export(run, out, tasks_path=_tasks(tmp_path))
    assert out.exists() and out.stat().st_size > 3000


def test_missing_attribution_is_none_not_an_error(tmp_path):
    run = _curve(tmp_path, rounds=[_rnd(1, 33.0, "accepted")])
    p = build_payload(run, tasks_path=_tasks(tmp_path), attr_dir=tmp_path / "nope")
    assert p["attribution"] is None


def _attr(tmp_path, *, cand_dir="", base_mean=30.0):
    d = tmp_path / "attr"
    d.mkdir(exist_ok=True)
    (d / "attribution.json").write_text(json.dumps({
        "base": {"mean": base_mean, "std": 0.0},
        "full": {"mean": 37.0, "std": 0.8},
        "full_delta": 4.0, "best_atom_delta": 2.3, "interaction": 1.7,
        "atoms": [{"index": 0, "path": "systemprompt.md", "label": "原子 0", "delta": 2.3}],
        "parent_dir": "harness/base", "cand_dir": cand_dir,
    }, ensure_ascii=False), encoding="utf-8")
    return d


def test_attribution_from_another_run_is_flagged(tmp_path):
    """归因指错目录时必须在看板上标出来。

    这条是踩过的坑：`--run runs/strict_veto --attr runs/attr_curve3` 拼错一次，
    看板就变成"这条曲线第 1 轮改了什么"配着另一次跑批的归因数字，
    两边都自洽、都带小数，肉眼看不出来。

    判据是**出处**不是分数：同一棵树重测一次基线就会差一分多（32.00 → 33.33），
    按分数判会把合法的重测误判成挂错目录。
    """
    run = _curve(tmp_path, rounds=[_rnd(1, 33.0, "accepted")])
    inside = build_payload(run, tasks_path=_tasks(tmp_path),
                           attr_dir=_attr(tmp_path, cand_dir=str(run / "round1")))
    assert inside["attribution"]["_match"] is True

    other = tmp_path / "attr2"
    other.mkdir()
    src = (tmp_path / "attr" / "attribution.json").read_text(encoding="utf-8")
    (other / "attribution.json").write_text(
        src.replace(f'"cand_dir": "{run / "round1"}"'.replace("\\", "\\\\"),
                    f'"cand_dir": "{(tmp_path / "other_run" / "round2").as_posix()}"'),
        encoding="utf-8")
    bad = build_payload(run, tasks_path=_tasks(tmp_path), attr_dir=other)
    assert bad["attribution"]["_match"] is False, "另一条跑批的归因不该认作本条曲线"


def test_attribution_without_provenance_is_unknown_not_wrong(tmp_path):
    """老文件没有出处字段：不提示，也不假装匹配。"""
    run = _curve(tmp_path, rounds=[_rnd(1, 33.0, "accepted")])
    d = _attr(tmp_path)
    p = d / "attribution.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data.pop("cand_dir")
    data.pop("parent_dir")
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert build_payload(run, tasks_path=_tasks(tmp_path),
                         attr_dir=d)["attribution"]["_match"] is None