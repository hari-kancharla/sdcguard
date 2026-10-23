"""scripts/modal_serve_cost.py: how a run's numbers become the summary METHOD.md uses for the serving kernels (no
Modal or GPU calls here)."""
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
    """One layer at one row count; times added by each design in microseconds."""
    r = {"linear_ms": 0.1}
    for kern, us in (("v1", v1), ("v3", v3)):
        r[f"{kern}_checked_ms"] = 0.1 + us / 1000
        r[f"{kern}_check_pct"] = us
    for case in ("clean", "one_pct", "large"):
        pz, fz = clean if case == "clean" else changed
        r[f"scores_{case}"] = {"plain": [pz, block[0], counts[0]]}
        for k in ("v1", "v3"):
            r[f"scores_{case}"][k] = [fz, block[1], counts[1]]
    return r


def _result(runner, **row_kw):
    def layers():
        return {name: {"by_rows": {str(m): _row(**row_kw) for m in runner.ROWS}} for name in ("a", "b", "c", "d")}
    good = {"compare_exit": 0, "compare_last": "ALL AGREE", "first_use": {"bfloat16": [True, "ok"],
                                                                         "float16": [True, "ok"]}}
    return {"agreement": {k: json.loads(json.dumps(good)) for k in ("v1", "v3")},
            "models": {"qwen1.5b": layers(), "qwen7b": layers()}, "estimated_cost_usd": 0.5}


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
    assert runner._agrees(_row(), "v3")
    assert not runner._agrees(_row(clean=(1.0, 1.2)), "v3")                 # clean scores 20% apart
    assert not runner._agrees(_row(changed=(30.0, 5.0)), "v3")              # alarm against no alarm
    assert not runner._agrees(_row(block=(7, 8)), "v3")                     # alarms, but another block
    assert not runner._agrees(_row(counts=([1, 0], [1, 1])), "v3")          # different counts
    assert runner._agrees(_row(changed=(3.0, 3.5), block=(7, 9)), "v3")     # no alarm on either


def test_v3_takes_over_from_the_row_count_it_stays_cheaper_from(runner):
    s = runner.summarize(_result(runner))
    assert s["agreement_ok"] == {"v1": True, "v3": True} and s["v3_from_rows"] == 1
    assert s["added_us"]["qwen7b"]["1"] == {"v1": 40.0, "v3": 16.0}
    res = _result(runner)
    for m in runner.ROWS:
        if m < 96:
            _set(res, m, "v3", 11.0)          # dearer below 96 rows
    _set(res, 160, "v3", 11.0)                # and at 160: the run of cheaper row counts starts above it
    assert runner.summarize(res)["v3_from_rows"] == 192
    res = _result(runner, v3=11.0)
    assert runner.summarize(res)["v3_from_rows"] is None
    assert not runner.summarize(_result(runner, block=(7, 8)))["agreement_ok"]["v3"]
    res = _result(runner)
    res["agreement"]["v1"]["compare_last"] = "DISAGREE"
    assert runner.summarize(res)["agreement_ok"] == {"v1": False, "v3": True}
    res = _result(runner)
    res["models"]["qwen7b"]["a"]["by_rows"]["16"] = {"error": "RuntimeError: x"}
    assert not any(runner.summarize(res)["agreement_ok"].values())


def test_the_entrypoint_saves_the_run_and_its_summary(runner, tmp_path, capsys, monkeypatch):
    res = _result(runner)
    monkeypatch.setattr(runner, "h100", SimpleNamespace(remote=lambda expected: res))
    runner.main()
    saved = list((tmp_path / "results" / "serve_cost").glob("h100_*.json"))
    assert len(saved) == 1
    data = json.loads(saved[0].read_text())
    assert data["summary"]["v3_from_rows"] == 1 and data["launcher_source"]["files"] >= 0
    out = capsys.readouterr().out
    assert "v3 is cheaper than v1 over the four 7B layers from 1 rows on" in out
    assert "v3: agrees with the plain check everywhere" in out


def test_a_run_on_other_code_stops_before_anything_runs(runner, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "h100", SimpleNamespace(remote=lambda expected: {"stopped": "other code"}))
    runner.main()
    assert "Stopped: other code" in capsys.readouterr().out
