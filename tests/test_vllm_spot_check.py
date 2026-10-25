"""experiments/vllm_spot_check.py without a GPU: its criteria (docs/METHOD.md, "Spot checks in serving") on made-up
results."""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import vllm_spot_check as vsc  # noqa: E402


def _stats(worst=0.8):
    return {"layers": vsc.LINEAR_LAYERS, "coverage": {"kernel": {"cuda:0 bfloat16": "Triton"}}, "worst": worst,
            "stats": {"outputs_judged": 40000, "active_calls": 5000}}


def passing():
    ident = ["a"] * vsc.IDENTITY
    speed = lambda v: {"16": {"median": {"output_tokens_per_s": v}},  # noqa: E731
                       "256": {"median": {"output_tokens_per_s": v * 4}}}
    tr = [{**t, "mark": {"layer": t["layer"], "cols": [0, 0], "width": 1536}, "unmark": 1, "detected": True,
           "rounds": 3, "verdicts": ["output_wrong"], "spot_alerts_other_layers": 0, "checksum_alerts": 0}
          for t in vsc.trials()]
    return {"gpu": "h100", "nospot": {"ready": True, "identity": ident, "speed": speed(1000.0), "clean_alerts": 0},
            "spot": {"ready": True, "identity": ident, "speed": speed(990.0), "clean_alerts": 0,
                     "spot_at_start": _stats(), "spot_clean": _stats()},
            "eager": {"ready": True, "identity": ident[:16], "clean_alerts": 0, "spot_at_start": _stats(),
                      "spot_clean": _stats(0.6), "trials": tr}}


def test_the_trials_are_fixed():
    t = vsc.trials()
    assert t == vsc.trials() and sum(x["criterion"] for x in t) == 6 and len(t) == 11
    assert all(x["layer"].startswith("model.layers.") for x in t)


def test_a_run_that_meets_every_criterion_passes():
    ok, fails, s = vsc.judge(passing())
    assert ok, fails
    assert s["cost"]["16"]["spot_cost"] == 0.01 and s["on_by_default"] is True


def test_each_criterion_fails_on_its_own():
    cases = []
    r = passing(); r["eager"]["ready"] = False; cases.append((r, "S1"))
    r = passing(); r["spot"]["spot_at_start"]["layers"] = 100; cases.append((r, "S1"))
    r = passing(); r["spot"]["spot_at_start"]["coverage"]["kernel"] = {"cuda:0 bfloat16": "PyTorch (x)"}
    cases.append((r, "S1"))
    r = passing(); r["spot"]["identity"] = ["b"] + r["spot"]["identity"][1:]; cases.append((r, "S2"))
    r = passing(); r["spot"]["clean_alerts"] = 2; cases.append((r, "S3"))
    r = passing(); r["eager"]["spot_clean"] = _stats(1.7); cases.append((r, "S3"))
    r = passing(); r["spot"]["spot_clean"]["stats"] = {"outputs_judged": 0}; cases.append((r, "S3"))
    r = passing(); r["eager"]["trials"][2]["detected"] = False; cases.append((r, "S4"))
    r = passing(); r["eager"]["trials"][4]["verdicts"] = ["output_agrees"]; cases.append((r, "S4"))
    r = passing(); r["eager"]["trials"][7]["spot_alerts_other_layers"] = 1; cases.append((r, "S4"))
    r = passing(); r["eager"]["trials"] = r["eager"]["trials"][:4]; cases.append((r, "S4"))
    for r, crit in cases:
        ok, fails, _ = vsc.judge(r)
        assert not ok and all(f.startswith(crit) for f in fails), (crit, fails)


def test_reported_trials_and_cost_dont_fail_the_run():
    r = passing()
    for t in r["eager"]["trials"]:
        if not t["criterion"]:
            t["detected"] = False
    r["spot"]["speed"]["16"]["median"]["output_tokens_per_s"] = 950.0       # 5%
    ok, fails, s = vsc.judge(r)
    assert ok and s["on_by_default"] is False


def test_the_l4_run_judges_the_spot_server_alone():
    r = {"gpu": "l4", "spot": copy.deepcopy(passing()["spot"])}
    ok, fails, s = vsc.judge(r)
    assert ok, fails
    assert s["on_by_default"] is False
    r["spot"]["spot_clean"] = _stats(2.5)
    ok, fails, _ = vsc.judge(r)
    assert not ok and fails[0].startswith("S3")


def test_a_trial_counts_a_late_alert_and_stops_at_the_deadline(monkeypatch):
    """An alert of the last round that arrives with the read after it counts; a trial stops sending rounds when the
    run's time is nearly up, and says so."""
    log = []
    state = {"marked": False, "rounds": 0}

    def rpc(port, method, args=None):
        if method == "sdcguard_test_scale_outputs":
            state["marked"] = True
            return {"layer": args[0], "cols": [0, 0], "width": 2048}
        if method == "sdcguard_test_unmark_rows":
            if state["marked"]:                          # the alert of the last round arrives after it
                log.append({"detector": "serve_spot", "detail": {"where": state["layer"],
                                                                 "recheck": {"verdict": "output_wrong"}}})
            state["marked"] = False
            return 1
        if method == "sdcguard_test_spot_stats":
            return {"turns": {state["layer"]: 3 * state["rounds"]}}
        return {}

    def try_(port, model, prompt, n):
        state["rounds"] += 1
        return {"text": "x"}
    monkeypatch.setattr(vsc.vn, "rpc", rpc)
    monkeypatch.setattr(vsc.vn, "_try", try_)
    monkeypatch.setattr(vsc.vn, "alerts", lambda path: list(log))
    monkeypatch.setattr(vsc.time, "sleep", lambda s: None)
    first = vsc.trials()[0]
    state["layer"] = first["layer"]
    monkeypatch.setattr(vsc, "trials", lambda: [first])
    rows = vsc.run_trials(8000, "x", ["p"] * 8, vsc.time.monotonic() + 3600)
    assert rows[0]["detected"] and rows[0]["rounds"] == vsc.ROUNDS and rows[0]["verdicts"] == ["output_wrong"]
    assert rows[0]["turns"] > 0 and rows[0]["cut_by_deadline"] is False
    log.clear()
    clock = iter([0.0, 0.0, 0.0] + [10.0] * 3 + [1000.0] * 1000)
    monkeypatch.setattr(vsc.time, "monotonic", lambda: next(clock))
    rows = vsc.run_trials(8000, "x", ["p"] * 8, 1030.0)
    assert rows[0]["cut_by_deadline"] is True and rows[0]["rounds_sent"] < vsc.ROUNDS


def test_band_odds_follow_the_column_order_exactly():
    from sdcguard import serve_spot as sp
    assert vsc.band_odds(2048, 5, 1, 1) == round(sp.SPOT_COLS / 2048, 4)       # one column: once in N / 8 turns
    assert vsc.band_odds(2048, 0, 2048, 1) == 1.0 and vsc.band_odds(2048, 7, 1, 256) == 1.0
    assert vsc.band_odds(17920, 100, 64, 0) is None
    t = vsc.trials()
    qkv = next(x for x in t if x["kind"] == "band" and "qkv" in x["layer"])
    assert vsc.band_odds(2048, int(qkv["band_frac"] * (2048 - vsc.BAND)), vsc.BAND, 11) == 1.0
    for x in (x for x in t if x["kind"] == "band" and "gate_up" in x["layer"]):
        assert 0.25 < vsc.band_odds(17920, int(x["band_frac"] * (17920 - vsc.BAND)), vsc.BAND, 11) < 0.4
