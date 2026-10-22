"""scripts/modal_serve_cost.py: how a run's numbers become the verdicts METHOD.md sets for the serving kernel v5
(no Modal or GPU calls here)."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch, tmp_path):
    class Builder:
        def __getattr__(self, _):
            return lambda *a, **k: self

    class App:
        def __init__(self, *a, **k):
            pass

        def function(self, **_):
            return lambda f: f

        def local_entrypoint(self):
            return lambda f: f

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Image=SimpleNamespace(debian_slim=lambda **_: Builder()),
                                                              App=App))
    path = Path(__file__).resolve().parents[1] / "scripts" / "modal_serve_cost.py"
    spec = importlib.util.spec_from_file_location("modal_serve_cost_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "LOCAL", tmp_path)
    return module


def _row(v1=10.0, v3=4.0, t16=3.0, t64=5.0, clean=(1.0, 1.01), changed=(30.0, 30.2), block=(7, 7),
         counts=([1, 0], [1, 0]), fallbacks=0):
    """One layer at one row count; times added by each setting in microseconds."""
    r = {"linear_ms": 0.1}
    for kern, us in (("v1", v1), ("v3", v3), ("v5_t16", t16), ("v5_t64", t64)):
        r[f"{kern}_checked_ms"] = 0.1 + us / 1000
        r[f"{kern}_check_pct"] = us
        r[f"{kern}_fallbacks"] = fallbacks if kern.startswith("v5") else 0
    for case in ("clean", "one_pct", "large"):
        pz, fz = clean if case == "clean" else changed
        r[f"scores_{case}"] = {"plain": [pz, block[0], counts[0]]}
        for k in ("v1", "v3", "v5_t16", "v5_t64"):
            r[f"scores_{case}"][k] = [fz, block[1], counts[1]]
    return r


def _result(runner, **row_kw):
    def layers():
        return {name: {"by_rows": {str(m): _row(**row_kw) for m in runner.ROWS}} for name in ("a", "b", "c", "d")}
    good = {"compare_exit": 0, "compare_last": "ALL AGREE", "first_use": {"bfloat16": [True, "ok"],
                                                                         "float16": [True, "ok"]}}
    return {"agreement": {k: json.loads(json.dumps(good)) for k in ("v1", "v3", "v5")},
            "models": {"qwen1.5b": layers(), "qwen7b": layers()}, "estimated_cost_usd": 0.8,
            "v5_counters_clean": True}


def _set(res, m, kern, us, model="qwen7b"):
    for dd in res["models"][model].values():
        dd["by_rows"][str(m)][f"{kern}_checked_ms"] = 0.1 + us / 1000


def test_copies_keep_each_layer_out_of_l2_between_calls(runner):
    for k, n in ((3584, 4608), (1536, 1536), (3584, 37888), (8960, 1536)):
        c = runner.copies_for(k, n)
        weights_and_parts = n * k * 2 + k * 6 * -(-n // 128) * 2          # bf16 weights, three bf16 parts of E
        assert c >= 2 and c * weights_and_parts >= 2 * runner.L2_BYTES > (c - 1) * weights_and_parts or c == 2
    assert runner.copies_for(3584, 37888) == 2


def test_agreement_follows_the_tolerances_written_in_method(runner):
    assert runner._agrees(_row(), "v5_t16")
    assert not runner._agrees(_row(clean=(1.0, 1.2)), "v5_t16")                 # clean scores 20% apart
    assert not runner._agrees(_row(changed=(30.0, 5.0)), "v5_t16")              # alarm against no alarm
    assert not runner._agrees(_row(block=(7, 8)), "v5_t16")                     # alarms, but another block
    assert not runner._agrees(_row(counts=([1, 0], [1, 1])), "v5_t16")          # different counts
    assert runner._agrees(_row(changed=(3.0, 3.5), block=(7, 9)), "v5_t16")     # no alarm on either


def test_v5_is_used_only_when_all_three_criteria_are_met(runner):
    s = runner.summarize(_result(runner))
    assert s["v5_used"] and all(s["criteria"].values())
    assert s["added_us"]["qwen7b"]["1"]["v5_t16"] == pytest.approx(12.0)
    # 16-row tiles cheaper everywhere: no 64-row tiles; v5 (12) against v1 (40) below 256 and v3 (16) from 256 on
    assert s["v5_t64_from_rows"] is None and s["v5_up_to_rows"] == 4096 and s["v5_below_rows"] == 4097
    assert s["v5_vs_now"]["256"] == {"v5": 12.0, "now": 16.0, "ratio": 0.75}
    # dearer than v1 at 1 row: not used at all
    res = _result(runner)
    _set(res, 1, "v5_t16", 10.0)
    _set(res, 1, "v5_t64", 11.0)
    s = runner.summarize(res)
    assert not s["v5_used"] and s["v5_below_rows"] == 0
    assert not s["criteria"]["3 at 1 row, at most 95% of v1's cost over the four 7B layers"]
    # scores that disagree once, a fallback while timing, a counter left over at the end, a failed comparison set
    # or first use: not used
    for bad in (dict(block=(7, 8)), dict(fallbacks=1)):
        assert not runner.summarize(_result(runner, **bad))["v5_used"]
    res = _result(runner)
    res["v5_counters_clean"] = False
    assert not runner.summarize(res)["v5_used"]
    res = _result(runner)
    res["agreement"]["v5"]["compare_last"] = "DISAGREE"
    assert not runner.summarize(res)["criteria"]["1 first-use comparison and comparison set"]
    res = _result(runner)
    res["agreement"]["v5"]["first_use"]["float16"] = [False, "differs"]
    assert not runner.summarize(res)["v5_used"]
    res = _result(runner)
    res["models"]["qwen1.5b"]["a"]["by_rows"]["16"] = {"error": "RuntimeError: x"}
    assert not runner.summarize(res)["v5_used"]


def test_v5_stops_at_the_first_row_count_where_it_is_not_5_percent_cheaper(runner):
    res = _result(runner)
    _set(res, 64, "v5_t16", 9.6)              # 38.4 against v1's 40: 96%, not cheap enough
    s = runner.summarize(res)
    assert s["v5_up_to_rows"] == 48 and s["v5_below_rows"] == 49 and s["v5_used"]
    _set(res, 64, "v5_t16", 9.5)              # exactly 95%: enough
    assert runner.summarize(res)["v5_up_to_rows"] == 4096
    # cheaper again later doesn't count: the run of row counts starts at 1
    res = _result(runner)
    _set(res, 8, "v5_t16", 11.0)
    _set(res, 8, "v5_t64", 11.0)
    assert runner.summarize(res)["v5_up_to_rows"] == 4


def test_64_row_tiles_from_where_they_stay_cheaper(runner):
    res = _result(runner)
    for m in runner.ROWS:
        if m >= 96:
            _set(res, m, "v5_t64", 2.0)       # 64-row tiles cheaper from 96 rows on
    _set(res, 32, "v5_t64", 2.0)              # and at 32, but not at 48 and 64: doesn't count
    s = runner.summarize(res)
    assert s["v5_t64_from_rows"] == 96
    assert s["v5_vs_now"]["128"]["v5"] == 8.0 and s["v5_vs_now"]["32"]["v5"] == 12.0


def test_the_entrypoint_saves_the_run_and_its_verdicts(runner, tmp_path, capsys, monkeypatch):
    res = _result(runner)
    monkeypatch.setattr(runner, "h100", SimpleNamespace(remote=lambda expected: res))
    runner.main()
    saved = list((tmp_path / "results" / "serve_cost").glob("h100_*.json"))
    assert len(saved) == 1
    data = json.loads(saved[0].read_text())
    assert data["summary"]["v5_used"] is True and data["launcher_source"]["files"] >= 0
    out = capsys.readouterr().out
    assert "v5 used: True (below 4097 rows; 64-row tiles from None rows)" in out
    assert "met: 3 at 1 row, at most 95% of v1's cost over the four 7B layers" in out


def test_a_run_on_other_code_stops_before_anything_runs(runner, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "h100", SimpleNamespace(remote=lambda expected: {"stopped": "other code"}))
    runner.main()
    assert "Stopped: other code" in capsys.readouterr().out
