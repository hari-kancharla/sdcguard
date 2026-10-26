"""experiments/vllm_spot_check.py without a GPU: its criteria (docs/METHOD.md, "Spot checks in serving") on made-up
results."""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import vllm_spot_check as vsc  # noqa: E402


NAMES = [f"model.layers.{b}.{k}" for b in range(vsc.BLOCKS) for k in vsc.KINDS]


def _stats(worst=0.8, turns=1):
    return {"layers": vsc.LINEAR_LAYERS, "names": list(NAMES), "coverage": {"kernel": {"cuda:0 bfloat16": "Triton"}},
            "worst": worst, "stats": {"outputs_judged": 40000, "active_calls": 5000},
            "turns": {n: turns for n in NAMES}}


def passing():
    ident = ["a"] * vsc.IDENTITY
    speed = lambda v: {"16": {"median": {"output_tokens_per_s": v}, "complete": True},  # noqa: E731
                       "256": {"median": {"output_tokens_per_s": v * 4}, "complete": True}}
    tr = [{**t, "mark": {"layer": t["layer"], "cols": [0, 0], "width": 1536}, "unmark": 1, "detected": True,
           "rounds": 3, "verdicts": ["output_wrong"], "spot_alerts_other_layers": 0, "checksum_alerts": 0,
           "changed": [{"step": 7, "row": 0, "col": 3}] if t["kind"].startswith("one") else None}
          for t in vsc.trials()]
    layer = vsc.compiled_layer()
    return {"gpu": "h100", "nospot": {"ready": True, "identity": ident, "speed": speed(1000.0), "clean_alerts": 0},
            "spot": {"ready": True, "identity": ident, "speed": speed(990.0), "clean_alerts": 0,
                     "spot_at_start": _stats(), "spot_clean": _stats(turns=40)},
            "eager": {"ready": True, "identity": ident[:16], "clean_alerts": 0, "spot_at_start": _stats(),
                      "spot_clean": _stats(0.6, turns=5), "trials": tr},
            "faulted": {"ready": True, "spot_at_start": _stats(), "spot_after": _stats(turns=9),
                        "served": {"eager": "False", "cudagraph": "FULL_AND_PIECEWISE", "compilation": "VLLM_COMPILE"},
                        "compiled_fault": {"layer": layer, "factor": 1.02, "applied": True, "width": 2048},
                        "trial": {"layer": layer, "detected": True, "rounds": 2, "verdicts": ["output_wrong"],
                                  "spot_alerts_other_layers": 0, "checksum_alerts": 0, "turns": 1}}}


def test_the_trials_are_fixed():
    t = vsc.trials()
    assert t == vsc.trials() and sum(x["criterion"] for x in t) == 9 and len(t) == 17
    assert all(x["layer"].startswith("model.layers.") for x in t)
    # the criteria's trials run first (S4's 6, then S5's 3), then the reported ones; each group keeps its draws: the
    # October 25 trials (S4's and the first 5 reported) with seed 0, the new ones with seed 2
    other = vsc.trials(one_seed=99)
    assert t[:6] == other[:6] and t[9:14] == other[9:14]
    assert [x["kind"] for x in t] == ["all"] * 6 + ["one"] * 3 + ["all", "all", "band", "band", "band"] + \
        ["one_median"] * 3
    assert [x["criterion"] for x in t] == [True] * 9 + [False] * 8
    assert vsc.compiled_layer() in NAMES


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
    r = passing(); r["eager"]["spot_clean"] = _stats(1.7, turns=5); cases.append((r, "S3"))
    r = passing(); r["spot"]["spot_clean"]["stats"] = {"outputs_judged": 0}; cases.append((r, "S3"))
    r = passing(); r["eager"]["trials"][2]["detected"] = False; cases.append((r, "S4"))
    r = passing(); r["eager"]["trials"][4]["verdicts"] = ["output_agrees"]; cases.append((r, "S4"))
    r = passing(); r["eager"]["trials"][7]["spot_alerts_other_layers"] = 1; cases.append((r, "S4"))
    r = passing(); r["eager"]["trials"] = r["eager"]["trials"][:4]; cases.append((r, "S4"))
    # every layer's turns must go up during the measured traffic, and the kernel stay Triton to the end
    r = passing(); r["spot"]["spot_clean"]["turns"][NAMES[17]] = 1; cases.append((r, "S1"))
    r = passing(); del r["eager"]["spot_clean"]["turns"]; cases.append((r, "S1"))
    r = passing(); r["spot"]["spot_clean"]["coverage"]["kernel"] = {"cuda:0 bfloat16": "off (x)"}
    cases.append((r, "S1"))
    # the sampled single wrong values
    one = [i for i, t in enumerate(passing()["eager"]["trials"]) if t["kind"] == "one"]
    r = passing(); r["eager"]["trials"][one[0]]["detected"] = False; cases.append((r, "S5"))
    r = passing(); r["eager"]["trials"][one[1]]["changed"] = []; cases.append((r, "S5"))
    r = passing(); r["eager"]["trials"][one[2]]["verdicts"] = ["output_agrees"]; cases.append((r, "S5"))
    # the compiled positive control
    r = passing(); r["faulted"]["ready"] = False; cases.append((r, "S6"))
    r = passing(); r["faulted"]["compiled_fault"]["applied"] = False; cases.append((r, "S6"))
    r = passing(); r["faulted"]["trial"]["detected"] = False; cases.append((r, "S6"))
    r = passing(); r["faulted"]["trial"]["verdicts"] = []; cases.append((r, "S6"))
    r = passing(); r["faulted"]["trial"]["spot_alerts_other_layers"] = 2; cases.append((r, "S6"))
    r = passing(); r["faulted"]["spot_at_start"]["coverage"]["kernel"] = {"cuda:0 bfloat16": "off"}
    cases.append((r, "S6"))
    r = passing(); r["faulted"]["spot_after"]["coverage"]["kernel"] = {"cuda:0 bfloat16": "off (failed)"}
    cases.append((r, "S6"))
    r = passing(); r["faulted"]["served"] = {"eager": "True", "cudagraph": "NONE"}; cases.append((r, "S6"))
    r = passing(); r["faulted"]["served"] = {}; cases.append((r, "S6"))         # (no stack published)
    r = passing(); r["faulted"]["trial"]["turns"] = 0; cases.append((r, "S6"))
    r = passing(); r["eager"]["trials"][one[0]]["changed"] = [{"step": 7, "error": "IndexError: x"}]
    cases.append((r, "S5"))                                        # a hook error isn't a wrong value made
    for r, crit in cases:
        ok, fails, _ = vsc.judge(r)
        assert not ok and all(f.startswith(crit) for f in fails), (crit, fails)


def test_cost_counts_only_complete_speed_runs():
    r = passing()
    r["spot"]["speed"]["256"]["complete"] = False                  # a round with failed requests: no cost decision
    ok, fails, s = vsc.judge(r)
    assert ok and s["on_by_default"] is False and s["cost"]["256"]["complete"] is False


def test_reported_trials_and_cost_dont_fail_the_run():
    r = passing()
    for t in r["eager"]["trials"]:
        if not t["criterion"]:
            t["detected"] = False
    r["spot"]["speed"]["16"]["median"]["output_tokens_per_s"] = 950.0       # 5%
    ok, fails, s = vsc.judge(r)
    assert ok and s["on_by_default"] is False


def test_a_sampled_single_value_trial_records_what_it_changed(monkeypatch):
    log = []
    calls = []

    def rpc(port, method, args=None):
        calls.append(method)
        if method == "sdcguard_test_spot_one":
            assert args[2] == "largest"
            return {"layer": args[0], "width": 2048, "pick": args[2]}
        if method == "sdcguard_test_spot_one_log":
            log.append({"detector": "serve_spot", "detail": {"where": one["layer"],
                                                             "recheck": {"verdict": "output_wrong"}}})
            return [{"step": 9, "row": 0, "col": 5}]
        if method == "sdcguard_test_spot_stats":
            return {"turns": {one["layer"]: 1}}
        return 1
    one = next(t for t in vsc.trials() if t["kind"] == "one")
    monkeypatch.setattr(vsc.vn, "rpc", rpc)
    monkeypatch.setattr(vsc.vn, "_try", lambda *a: {"text": "x"})
    monkeypatch.setattr(vsc.vn, "alerts", lambda path: list(log))
    monkeypatch.setattr(vsc.time, "sleep", lambda s: None)
    monkeypatch.setattr(vsc, "trials", lambda: [one])
    rows = vsc.run_trials(8000, "x", ["p"] * 8, vsc.time.monotonic() + 3600)
    assert rows[0]["detected"] and rows[0]["changed"] == [{"step": 9, "row": 0, "col": 5}]
    assert "sdcguard_test_scale_outputs" not in calls and calls.index("sdcguard_test_spot_one_log") < \
        calls.index("sdcguard_test_unmark_rows")


def test_the_compiled_trial_stops_at_its_layers_alert(monkeypatch):
    layer = vsc.compiled_layer()
    log, sent = [], [0]

    def try_(port, model, prompt, n):
        sent[0] += 1
        if sent[0] == 3 * vsc.BURST:
            log.append({"detector": "serve_spot", "detail": {"where": layer, "recheck": {"verdict": "output_wrong"}}})
        return {"text": "x"}
    monkeypatch.setattr(vsc.vn, "rpc", lambda *a, **k: {"turns": {layer: sent[0]}})
    monkeypatch.setattr(vsc.vn, "_try", try_)
    monkeypatch.setattr(vsc.vn, "alerts", lambda path: list(log))
    monkeypatch.setattr(vsc.time, "sleep", lambda s: None)
    t = vsc.compiled_trial(8000, "x", ["p"] * 8, vsc.time.monotonic() + 3600, layer)
    assert t["detected"] and t["rounds"] == 3 and t["verdicts"] == ["output_wrong"] and t["turns"] > 0
    assert t["spot_alerts_other_layers"] == 0 and t["alerts_before_trial"] == 0
    # an alert from before the trial (warm-up: the fault is there from load) doesn't count as the trial's catch
    log[:] = [{"detector": "serve_spot", "detail": {"where": layer, "recheck": {"verdict": "output_wrong"}}}]
    sent[0] = 10 ** 6                                # (no new alert during these rounds)
    t = vsc.compiled_trial(8000, "x", ["p"] * 8, vsc.time.monotonic() + 3600, layer)
    assert not t["detected"] and t["alerts_before_trial"] == 1 and t["rounds_sent"] == vsc.ROUNDS


def test_compiled_served_reads_the_servers_own_stack():
    assert vsc.compiled_served({"eager": "False", "cudagraph": "PIECEWISE"})
    assert not vsc.compiled_served({"eager": "True", "cudagraph": "NONE"})
    assert not vsc.compiled_served({"eager": "False", "cudagraph": "NONE"}) and not vsc.compiled_served({})


def test_the_l4_run_judges_the_spot_server_alone():
    r = {"gpu": "l4", "spot": copy.deepcopy(passing()["spot"])}
    ok, fails, s = vsc.judge(r)
    assert ok, fails
    assert s["on_by_default"] is False
    r["spot"]["spot_clean"] = _stats(2.5, turns=40)
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


def _ext():
    import sdcguard_test_ext as ext
    return ext


class _Method:
    def _gemm_impl(self, layer, x, weight, bias=None):
        import torch
        return torch.nn.functional.linear(x, weight, bias)


def _model(n=48, k=96):
    import torch
    g = torch.Generator().manual_seed(1)
    model = torch.nn.Module()
    for name in ("a", "b"):
        lay = torch.nn.Module()
        lay.weight = torch.nn.Parameter((torch.randn(n, k, generator=g) * 0.05).to(torch.bfloat16),
                                        requires_grad=False)
        lay.quant_method = _Method()
        setattr(model, name, lay)
    return model


def test_the_sampled_single_value_fault_changes_one_sampled_output_at_the_layers_turns():
    import torch
    from sdcguard import serve_spot as sp
    model = _model()
    lay = model.a
    stepbuf = torch.zeros(1, dtype=torch.int64)
    st = torch.zeros(2, sp.COLS, dtype=torch.int64)
    caps = sp.new_caps("cpu", 96)
    lay._sdcguard_spot = (lay.weight, stepbuf, st, 1, 3, 1, *caps, sp.SPOT_Z)
    fi = _ext().FaultInjection()
    fi.get_model = lambda: model
    assert fi.sdcguard_test_spot_one("a", 1.02) == {"layer": "a", "width": 48, "pick": "largest"}
    assert fi.sdcguard_test_spot_one("b", 1.02) == {}                      # not spot-checked
    x = torch.randn(5, 96, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
    clean = torch.nn.functional.linear(x, lay.weight)
    for step, active in ((6, False), (7, True)):
        stepbuf.fill_(sp.pack_step(step, 5))
        sp._HOST[stepbuf.data_ptr()] = sp.pack_step(step, 5)
        y = lay.quant_method._gemm_impl(lay, x, lay.weight)
        diff = (y != clean).nonzero().tolist()
        if not active:
            assert diff == []
            continue
        rows, rmask, cols = sp.sampled(torch.tensor(step), 3, 5, 48)
        assert len(diff) == 1 and diff[0][0] in rows[rmask].tolist() and diff[0][1] in cols.tolist()
        sub = clean.index_select(0, rows[rmask]).index_select(1, cols).float().abs()
        assert float(clean[diff[0][0], diff[0][1]].float().abs()) == float(sub.max())     # the largest sampled
        sp.spot_check_torch(x, y, lay.weight, None, stepbuf, st, 1, 3, 1, *caps, sp.SPOT_Z)
        assert int(caps[0][0]) == 1 and [int(caps[0][1]), int(caps[0][2])] == diff[0]     # caught where it is
    log = fi.sdcguard_test_spot_one_log()
    assert len(log) == 1 and log[0]["step"] == 7 and [log[0]["row"], log[0]["col"]] == diff[0]
    assert fi.sdcguard_test_unmark_rows() == 1
    assert torch.equal(lay.quant_method._gemm_impl(lay, x, lay.weight), clean)
    sp._HOST.pop(stepbuf.data_ptr(), None)


def test_the_compiled_fault_scales_one_layers_outputs_from_load():
    import torch
    ext = _ext()
    model = _model()
    x = torch.randn(4, 96).to(torch.bfloat16)
    ya, yb = (torch.nn.functional.linear(x, m.weight) for m in (model.a, model.b))
    info = ext._compiled_fault(model, "a:1.02")
    assert info == {"layer": "a", "factor": 1.02, "applied": True, "width": 48}
    assert torch.equal(model.a.quant_method._gemm_impl(model.a, x, model.a.weight),
                       (ya.float() * 1.02).to(torch.bfloat16))
    assert torch.equal(model.b.quant_method._gemm_impl(model.b, x, model.b.weight), yb)
    assert ext._compiled_fault(model, "missing:1.02")["applied"] is False


def test_the_run_reads_its_baseline_after_a_read_and_holds_time_for_the_faulted_server(monkeypatch, tmp_path):
    """Each spot-checking server's baseline is read after a short request and a read of the results 2 s later; the
    eager trials stop FAULTED_RESERVE before the run's limit; the faulted server isn't started with under 7 minutes
    left; its compile cache is a temporary folder, removed after."""
    clock = [0.0]
    calls, trial_deadlines, caches = [], [], []

    class P:
        pass

    def server(role, out, port, metrics_port, cache=None):
        calls.append(("start", role))
        caches.append(cache)
        return P(), open(os.path.join(str(tmp_path), f"{role}.log"), "w")

    def try_(port, model, prompt, n=32):
        calls.append(("try", n))
        return {"text": "x"}

    def rpc(port, method, args=()):
        calls.append(("rpc", method))
        return {}

    def run_trials(port, log_path, prompts, deadline):
        trial_deadlines.append(deadline)
        clock[0] = deadline - 60                       # the trials use their time
        return []
    monkeypatch.setattr(vsc, "server", server)
    monkeypatch.setattr(vsc.vs, "wait_ready", lambda port, proc, timeout: True)
    monkeypatch.setattr(vsc.vs, "stop", lambda proc: {})
    monkeypatch.setattr(vsc.vn, "_try", try_)
    monkeypatch.setattr(vsc.vn, "rpc", rpc)
    monkeypatch.setattr(vsc.vn, "_speed", lambda port, model: {})
    monkeypatch.setattr(vsc.vn, "alerts", lambda path: [])
    monkeypatch.setattr(vsc, "run_trials", run_trials)
    monkeypatch.setattr(vsc, "compiled_trial", lambda *a: {"detected": True})
    monkeypatch.setattr(vsc.time, "sleep", lambda s: None)
    monkeypatch.setattr(vsc.time, "monotonic", lambda: clock[0])
    res = vsc.run(str(tmp_path), 8000, 9400, "h100")
    assert trial_deadlines == [vsc.RUN_S["h100"] - vsc.FAULTED_RESERVE]
    assert res["faulted"].get("ready") and res["faulted"]["trial"] == {"detected": True}
    i = calls.index(("start", "spot"))
    j = calls.index(("rpc", "sdcguard_test_spot_stats"), i)
    assert calls[i + 1:j] == [("try", 1), ("try", 1)]                    # two 1-token requests, then the baseline
    cache = caches[-1]
    assert cache and "sdcguard_faulted_" in cache and not os.path.exists(cache) and caches[:3] == [None] * 3
    clock[0] = 0.0
    calls.clear()
    monkeypatch.setattr(vsc, "run_trials", lambda port, log_path, prompts, deadline: (clock.__setitem__(
        0, vsc.RUN_S["h100"] - 6 * 60), [])[1])                          # only 6 minutes left after the trials
    res = vsc.run(str(tmp_path), 8000, 9400, "h100")
    assert "skipped" in res["faulted"] and ("start", "faulted") not in calls
    ok, fails, _ = vsc.judge(res)
    assert any(f.startswith("S6: faulted didn't start") for f in fails)
