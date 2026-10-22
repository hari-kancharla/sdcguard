"""scripts/modal_serve_cost.py: how a run's numbers become the verdicts METHOD.md sets for the serving kernel v3
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


def _row(v1=10.0, v3=4.0, clean=(1.0, 1.01), changed=(30.0, 30.2), block=(7, 7), counts=([1, 0], [1, 0])):
    r = {"linear_ms": 0.1, "v1_checked_ms": 0.1 + v1 / 1000, "v3_checked_ms": 0.1 + v3 / 1000,
         "v3_p16_checked_ms": 0.1 + v3 / 1000, "v3_p64_checked_ms": 0.1 + v3 / 1000,
         "v1_check_pct": v1, "v3_check_pct": v3}
    for case in ("clean", "one_pct", "large"):
        pz, fz = clean if case == "clean" else changed
        r[f"scores_{case}"] = {"plain": [pz, block[0], counts[0]]}
        for k in ("v1", "v3", "v3_p16", "v3_p64"):
            r[f"scores_{case}"][k] = [fz, block[1], counts[1]]
    return r


def _result(runner, **row_kw):
    layers = {name: {"by_rows": {str(m): _row(**row_kw) for m in runner.ROWS}} for name in ("a", "b", "c", "d")}
    good = {"compare_exit": 0, "compare_last": "ALL AGREE", "first_use": {"bfloat16": [True, "ok"],
                                                                         "float16": [True, "ok"]}}
    return {"agreement": {"v1": dict(good), "v3": dict(good)}, "models": {"qwen1.5b": layers, "qwen7b": layers},
            "estimated_cost_usd": 0.8}


def test_agreement_follows_the_tolerances_written_in_method(runner):
    assert runner._agrees(_row(), "v3")
    assert not runner._agrees(_row(clean=(1.0, 1.2)), "v3")                 # clean scores 20% apart
    assert not runner._agrees(_row(changed=(30.0, 5.0)), "v3")              # alarm against no alarm
    assert not runner._agrees(_row(block=(7, 8)), "v3")                     # alarms, but another block
    assert not runner._agrees(_row(counts=([1, 0], [1, 1])), "v3")          # different counts
    assert runner._agrees(_row(changed=(3.0, 3.5), block=(7, 9)), "v3")     # no alarm on either: blocks don't matter


def test_v3_becomes_the_default_only_when_all_three_criteria_are_met(runner):
    s = runner.summarize(_result(runner))
    assert s["v3_becomes_default"] and all(s["criteria"].values())
    assert s["added_ms"]["qwen7b"]["256"]["v1"] == pytest.approx(40.0) and s["added_ms"]["qwen7b"]["256"]["v3"] == 16.0
    assert s["v3_from_rows"] == 1
    slower = runner.summarize(_result(runner, v1=4.0, v3=5.0))
    assert not slower["v3_becomes_default"] and slower["v3_from_rows"] is None
    assert not slower["criteria"]["3 cheaper than v1 over the four 7B layers from some row count of 256 or fewer on"]
    # dearer at 1 and 16 rows, cheaper from 64 on: used from 64 rows, v1 below
    res = _result(runner)
    for dd in res["models"]["qwen7b"].values():
        for m in ("1", "16"):
            dd["by_rows"][m]["v3_checked_ms"] = dd["by_rows"][m]["v1_checked_ms"] + 0.001
    mixed = runner.summarize(res)
    assert mixed["v3_from_rows"] == 64 and mixed["v3_becomes_default"]
    # cheaper only from 1,024 rows on: too late to help serving, so v1 stays
    for dd in res["models"]["qwen7b"].values():
        for m in ("64", "256"):
            dd["by_rows"][m]["v3_checked_ms"] = dd["by_rows"][m]["v1_checked_ms"] + 0.001
    late = runner.summarize(res)
    assert late["v3_from_rows"] == 1024 and not late["v3_becomes_default"]
    # a dearer row count above a cheaper one breaks the run of row counts it is cheaper at
    res = _result(runner)
    for dd in res["models"]["qwen7b"].values():
        dd["by_rows"]["4096"]["v3_checked_ms"] = dd["by_rows"]["4096"]["v1_checked_ms"] + 0.001
    assert runner.summarize(res)["v3_from_rows"] == 16384
    disagree = runner.summarize(_result(runner, block=(7, 8)))
    assert not disagree["criteria"]["2 scores agree with the plain check on every shape and row count"]
    res = _result(runner)
    res["agreement"]["v3"]["compare_last"] = "DISAGREE"
    assert not runner.summarize(res)["criteria"]["1 first-use comparison and comparison set"]
    res = _result(runner)
    res["agreement"]["v3"]["first_use"]["float16"] = [False, "differs"]
    assert not runner.summarize(res)["v3_becomes_default"]
    res = _result(runner)
    res["models"]["qwen7b"]["a"]["by_rows"]["16"] = {"error": "RuntimeError: x"}
    assert not runner.summarize(res)["v3_becomes_default"]


def test_another_blocks_per_program_setting_is_used_only_if_it_agrees_and_mostly_wins(runner):
    res = _result(runner)
    assert runner.summarize(res)["sums_p"] == 32                     # all three equal: the default stays
    for layers in res["models"].values():
        for dd in layers.values():
            for m in runner.ROWS:
                dd["by_rows"][str(m)]["v3_p64_checked_ms"] -= 0.001     # 64 cheaper everywhere
    assert runner.summarize(res)["sums_p"] == 64
    res["models"]["qwen1.5b"]["a"]["by_rows"]["1"]["scores_large"]["v3_p64"][1] = 99    # but disagrees once
    assert runner.summarize(res)["sums_p"] == 32


def test_the_entrypoint_saves_the_run_and_its_verdicts(runner, tmp_path, capsys, monkeypatch):
    res = _result(runner)
    monkeypatch.setattr(runner, "h100", SimpleNamespace(remote=lambda expected: res))
    runner.main()
    saved = list((tmp_path / "results" / "serve_cost").glob("h100_*.json"))
    assert len(saved) == 1
    data = json.loads(saved[0].read_text())
    assert data["summary"]["v3_becomes_default"] is True and data["launcher_source"]["files"] >= 0
    out = capsys.readouterr().out
    assert "v3 becomes the default: True (from 1 rows on" in out
    assert "met: 3 cheaper than v1 over the four 7B layers from some row count" in out


def test_a_run_on_other_code_stops_before_anything_runs(runner, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "h100", SimpleNamespace(remote=lambda expected: {"stopped": "other code"}))
    runner.main()
    assert "Stopped: other code" in capsys.readouterr().out
