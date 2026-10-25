"""experiments/vllm_weight_scan_check.py without a GPU: its criteria (docs/METHOD.md, "Scheduled weight scan in
serving") on made-up results."""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import vllm_weight_scan_check as vw  # noqa: E402


def passing():
    ident = ["a"] * vw.IDENTITY
    speed = {"16": {"on": [995.0, 994.0, 996.0, 995.0], "off": [1000.0] * 4},
             "256": {"on": [2000.0] * 4, "off": [2000.0] * 4}}
    rows = [{**t, "flip": {"layer": t["layer"], "before": 0.5, "after": 0.25},
             "back": {"layer": t["layer"], "before": 0.25, "after": 0.5}, "refreshed": 1,
             "detected": True, "alerted": True, "seconds_to_detect": 4.0 + i % 5, "rounds": 3, "kind": "unchecked",
             "scan_alerts_other_layers": 0, "checksum_alerts_on_layer": 0} for i, t in enumerate(vw.trials())]
    stats = {"entries": vw.LINEAR_LAYERS + 64,
             "coverage": {"entries": vw.LINEAR_LAYERS + 64, "by_kind": {"checked": 15, "unchecked": 97, "other": 64},
                          "not_scanned": {}, "never_compared": [], "kernel": {"cuda:0 2-byte": "Triton"},
                          "oldest_verification_age_seconds": 3.2},
             "stats": {"passes": 40, "last_pass_seconds": 6.5}}
    mstats = {"entries": 120, "coverage": {"entries": 120, "by_kind": {"checked": 12, "unchecked": 84, "other": 24},
                                           "not_scanned": {}, "never_compared": [],
                                           "kernel": {"cuda:0 2-byte": "Triton"}}, "stats": {"passes": 9}}
    uncovered = {"uncovered": [], "not_scanned": {}, "parameters": 340}

    def prow(t, i):
        entry = f"{t['prefix']}.{t['suffix']}[0:33554432]"
        before = [5, 6] if t["op"] == "swap" else [5]
        return {**t, "change": {"name": t["prefix"] + "." + t["suffix"], "index": 10, "entry": entry,
                                "before": before, "after": before[::-1] if t["op"] == "swap" else [7]},
                "back": {"after": before}, "entry": entry, "refreshed": 1, "detected": True, "alerted": True,
                "seconds_to_detect": 1.0 + i % 3, "rounds": 2, "undone": True, "scan_alerts_other_entries": 0}

    return {"noscan": {"ready": True, "identity": ident, "clean_alerts": 0},
            "scan": {"ready": True, "identity": ident, "speed": speed, "clean_alerts": 0,
                     "clean_scan_alerts": 0, "scan_at_clean": stats, "scan_at_end": stats, "trials": rows,
                     "uncovered": dict(uncovered),
                     "param_trials": [prow(t, i) for i, t in enumerate(vw.param_trials())]},
            "moe_scan": {"ready": True, "clean_alerts": 0, "clean_scan_alerts": 0, "scan_at_clean": mstats,
                         "scan_at_end": mstats, "uncovered": dict(uncovered),
                         "param_trials": [prow(t, i) for i, t in enumerate(vw.moe_trials())]}}


def test_the_trials_cover_every_bit_and_are_fixed():
    t = vw.trials()
    assert len(t) == 48 and sorted({x["bit"] for x in t}) == list(range(16))
    assert t == vw.trials() and all(x["layer"].startswith("model.layers.") for x in t)


def test_a_run_that_meets_every_criterion_passes():
    ok, fails, s = vw.judge(passing())
    assert ok, fails
    assert s["detected"] == 48 and s["cost"]["16"]["scan_cost"] == 0.005 and s["on_by_default"] is True
    assert s["seconds_to_detect"]["max"] == 8.0 and s["passes"] == 40
    assert s["params"]["trials"] == s["params"]["detected"] == 12 and s["moe"]["detected"] == 6
    assert s["oldest_verification_age_seconds"] == 3.2


def test_the_trials_outside_the_linear_layers_are_fixed():
    p, m = vw.param_trials(), vw.moe_trials()
    assert len(p) == 12 and p == vw.param_trials() and len(m) == 6 and m == vw.moe_trials()
    assert sum(t["op"] == "swap" for t in p) == 4 and sum(t["op"] == "swap" for t in m) == 2
    assert {t["prefix"] for t in p if t["op"] == "flip"} >= {"model.embed_tokens", "model.norm"}
    assert all(t["suffix"] in ("w13_weight", "w2_weight") for t in m)


def test_param_trials_change_undo_and_refresh_through_the_rpcs(monkeypatch, tmp_path):
    calls = []

    def rpc(port, method, args=()):
        calls.append((method, list(args)))
        if method == "sdcguard_test_change_param":
            undo = len(args) > 5
            return {"name": "p", "index": 42, "entry": "model.embed_tokens.weight[0:9]",
                    "before": [3] if not undo else [9], "after": [9] if not undo else [3]}
        return {"refreshed": 1}
    alerts = []
    monkeypatch.setattr(vw.vn, "rpc", rpc)
    monkeypatch.setattr(vw.vn, "alerts", lambda path: list(alerts))

    def try_(*a, **k):
        alerts.append({"detector": "serve_weights", "detail": {"where": "model.embed_tokens.weight[0:9]"},
                       "time": vw.time.time()})
        return {"text": "x"}
    monkeypatch.setattr(vw.vn, "_try", try_)
    monkeypatch.setattr(vw, "_empty_prefix_cache", lambda port: True)
    tr = vw.param_trials()[:1]
    rows = vw.run_param_trials(8000, str(tmp_path / "a.jsonl"), vw.MODEL, ["p"] * 8, tr, vw.time.monotonic() + 600,
                               str(tmp_path), "params")
    assert rows[0]["detected"] is True and rows[0]["undone"] is True and rows[0]["refreshed"] == 1
    assert calls[1] == ("sdcguard_test_change_param", [tr[0]["prefix"], "weight", tr[0]["frac"], "flip", 0, 42])
    assert calls[2] == ("sdcguard_test_scan_refresh_entry", ["model.embed_tokens.weight[0:9]"])
    assert (tmp_path / "PROGRESS_params.json").exists()


def test_each_criterion_fails_on_its_own():
    cases = []
    r = passing(); r["scan"]["ready"] = False; cases.append((r, "W1"))
    r = passing(); r["scan"]["scan_at_clean"]["coverage"] = {"entries": 100, "not_scanned": {"x": "moved"},
                                                           "kernel": {"cuda:0 2-byte": "Triton"}}
    cases.append((r, "W1"))
    r = passing(); r["noscan"]["clean_alerts"] = 2; cases.append((r, "W2"))
    r = passing(); r["scan"]["trials"][7]["detected"] = False; cases.append((r, "W3"))
    r = passing(); r["scan"]["trials"][3]["scan_alerts_other_layers"] = 1; cases.append((r, "W3"))
    r = passing(); r["scan"]["trials"][2]["back"] = {"after": 0.25}; cases.append((r, "W3"))
    r = passing(); r["scan"]["trials"] = r["scan"]["trials"][:40] + [{"stopped": "3 misses in a row before trial 40"}]
    cases.append((r, "W3"))
    r = passing(); r["scan"]["trials"][5]["flip"] = {"error": "HTTP 500"}; cases.append((r, "W3"))
    r = passing(); r["scan"]["trials"][6]["refreshed"] = 0; cases.append((r, "W3"))
    r = passing(); r["scan"]["scan_at_clean"]["coverage"]["never_compared"] = ["model.layers.3.mlp.down_proj"]
    cases.append((r, "W1"))
    r = passing(); r["scan"]["identity"] = ["b"] + r["scan"]["identity"][1:]; cases.append((r, "W4"))
    r = passing(); r["scan"]["uncovered"]["uncovered"] = ["model.embed_tokens.weight"]; cases.append((r, "W1"))
    r = passing(); r["scan"]["uncovered"] = "x"; cases.append((r, "W1"))
    r = passing(); r["scan"]["scan_at_clean"]["coverage"]["by_kind"] = {"checked": 15, "unchecked": 97}
    cases.append((r, "W1"))
    r = passing(); r["scan"]["param_trials"][3]["detected"] = False; cases.append((r, "W6"))
    r = passing(); r["scan"]["param_trials"][1]["entry"] = ""; cases.append((r, "W6"))
    r = passing(); r["scan"]["param_trials"][8]["undone"] = False; cases.append((r, "W6"))
    r = passing(); r["scan"]["param_trials"][9]["refreshed"] = 0; cases.append((r, "W6"))
    r = passing(); r["scan"]["param_trials"][0]["scan_alerts_other_entries"] = 2; cases.append((r, "W6"))
    r = passing(); r["scan"]["param_trials"] = r["scan"]["param_trials"][:5]; cases.append((r, "W6"))
    r = passing(); r["moe_scan"] = {"skipped": "the run's time was nearly up"}; cases.append((r, "W7"))
    r = passing(); r["moe_scan"]["ready"] = False; cases.append((r, "W7"))
    r = passing(); r["moe_scan"]["clean_alerts"] = 1; cases.append((r, "W7"))
    r = passing(); r["moe_scan"]["uncovered"]["not_scanned"] = {"x.w13_weight": "--enable-eplb"}
    cases.append((r, "W7"))
    r = passing(); r["moe_scan"]["param_trials"][5]["detected"] = False; cases.append((r, "W7"))
    r = passing(); r["moe_scan"]["scan_at_clean"]["coverage"]["by_kind"] = {"checked": 12}; cases.append((r, "W7"))
    r = passing(); r["moe_scan"]["scan_at_clean"]["coverage"]["kernel"] = {"cuda:0 2-byte": "PyTorch (x)"}
    cases.append((r, "W5"))
    r = passing(); r["scan"]["scan_at_end"] = copy.deepcopy(r["scan"]["scan_at_end"])
    r["scan"]["scan_at_end"]["coverage"]["kernel"] = {"cuda:0 2-byte": "PyTorch (a launch failed: RuntimeError)"}
    cases.append((r, "W5"))
    for r, crit in cases:
        ok, fails, _ = vw.judge(r)
        assert not ok and all(f.startswith(crit) for f in fails), (crit, fails)


def test_a_costly_scan_or_a_failed_run_isnt_on_by_default():
    r = passing()
    r["scan"]["speed"]["16"]["on"] = [980.0] * 4
    ok, fails, s = vw.judge(r)
    assert ok and s["on_by_default"] is False                       # the cost is reported, not a criterion
    r = passing()
    r["scan"]["trials"][0]["detected"] = False
    ok, fails, s = vw.judge(r)
    assert not ok and s["on_by_default"] is False                   # cheap, but a criterion failed


def test_the_cost_order_alternates_and_the_run_fits_its_phase():
    assert vw.ALTERNATIONS.count("on") == vw.ALTERNATIONS.count("off") == 4
    assert vw.RUN_S < 55 * 60 and vw.MAX_MISSES_IN_A_ROW == 3        # the Modal phase stops it at 55 minutes


def test_the_prefix_cache_reset_is_tried_again_while_vllm_says_no(monkeypatch):
    answers = [(200, {"success": False}), (200, {"success": False}), (200, {"success": True})]
    calls = []

    def post(port, path, body, timeout=600.0):
        calls.append(path)
        return answers.pop(0)
    monkeypatch.setattr(vw.vn, "_post", post)
    monkeypatch.setattr(vw.time, "sleep", lambda s: None)
    assert vw._empty_prefix_cache(8000) is True and calls == ["/reset_prefix_cache"] * 3
    monkeypatch.setattr(vw.vn, "_post", lambda *a, **k: (200, {"success": False}))
    assert vw._empty_prefix_cache(8000) is False
    monkeypatch.setattr(vw.vn, "_post", lambda *a, **k: (404, "not found"))
    assert vw._empty_prefix_cache(8000) is False
    r = passing()
    r["scan"]["trials"][0]["cache_emptied"] = [True, False]
    ok, fails, s = vw.judge(r)
    assert ok and s["prefix_cache_not_emptied"] == 1


def test_rpc_answers_are_read_as_vllm_returns_them(monkeypatch):
    """vLLM 0.32's /collective_rpc returns a worker's dict or list as is and anything else as a string: the scan RPCs
    answer with dicts, and a string where a count was expected isn't taken as one."""
    assert vw._refreshed({"refreshed": 1}) == 1 and vw._refreshed({"refreshed": -1}) == -1
    assert vw._refreshed("1") is None and vw._refreshed({"refreshed": "1"}) is None
    assert vw._refreshed({"refreshed": True}) is None and vw._refreshed({}) is None
    r = passing()
    r["scan"]["trials"][4]["refreshed"] = vw._refreshed("1")
    ok, fails, _ = vw.judge(r)
    assert not ok and any("taken again" in f for f in fails)
    answers = iter([{"scan": True, "paused": True}, "True", {"scan": False, "paused": None}])
    monkeypatch.setattr(vw.vn, "rpc", lambda port, method, args=(): next(answers))
    assert vw._pause(8000, True) is True and vw._pause(8000, True) is False and vw._pause(8000, False) is False


def test_a_pause_that_didnt_take_leaves_the_cost_unmeasured():
    r = passing()
    r["scan"]["speed"]["16"]["pause_failed"] = ["round 1 (off)"]
    ok, fails, s = vw.judge(r)
    assert ok and s["cost"]["16"]["measured"] is False and s["on_by_default"] is False
    r = passing()
    r["scan"]["speed"]["16"]["off"] = [1000.0] * 3                 # a round missing
    ok, fails, s = vw.judge(r)
    assert ok and s["cost"]["16"]["measured"] is False and s["on_by_default"] is False


def test_a_late_alert_or_a_server_error_fails_the_run():
    r = passing()
    r["scan"]["trials"][9].update(detected=False, alerted=True, seconds_to_detect=124.0)
    ok, fails, s = vw.judge(r)
    assert not ok and any(f.startswith("W3") and "within 120 s" in f for f in fails)
    assert s["alerted_after_the_limit"] == 1
    r = passing()
    r["scan"]["error"] = "ConnectionRefusedError: [Errno 111]"
    ok, fails, s = vw.judge(r)
    assert not ok and any(f.startswith("run: scan stopped with an error") for f in fails)


def test_trials_keep_what_was_measured_when_the_server_dies(monkeypatch, tmp_path):
    calls = {"n": 0}

    def rpc(port, method, args=()):
        calls["n"] += 1
        if calls["n"] > 3:
            raise ConnectionRefusedError("server gone")
        if method == "sdcguard_test_flip_bit":
            return {"layer": args[0], "before": 0.5, "after": 0.25}
        return {"refreshed": 1}
    monkeypatch.setattr(vw.vn, "rpc", rpc)
    monkeypatch.setattr(vw.vn, "alerts", lambda path: [])
    monkeypatch.setattr(vw.vn, "_try", lambda *a, **k: {"text": "x"})
    monkeypatch.setattr(vw, "_empty_prefix_cache", lambda port: True)
    monkeypatch.setattr(vw, "DETECT_S", 0.01)
    rows = vw.run_trials(8000, str(tmp_path / "a.jsonl"), ["p"] * 8, vw.time.monotonic() + 600, str(tmp_path))
    assert rows[0]["bit"] == 0 and rows[0]["refreshed"] == 1 and rows[0]["detected"] is False
    assert "error during trial 1" in rows[-1]["stopped"] and (tmp_path / "PROGRESS.json").exists()



def test_the_scan_goes_on_by_default_only_with_the_kernel_and_1pct_at_both_loads():
    """The amendment of October 25: W5 (the Triton fingerprint was used on the GPU) and a measured cost of at most 1%
    at 16 and at 256 requests at once, not 16 alone."""
    r = passing()
    r["scan"]["speed"]["256"]["on"] = [1970.0] * 4                 # 1.5% at 256, 0.5% at 16
    ok, fails, s = vw.judge(r)
    assert ok and s["on_by_default"] is False
    r = passing()
    r["scan"]["scan_at_clean"]["coverage"]["kernel"] = {"cuda:0 2-byte": "PyTorch (AssertionError: differs)"}
    ok, fails, s = vw.judge(r)
    assert not ok and any(f.startswith("W5") for f in fails) and s["on_by_default"] is False
    r = passing()
    r["scan"]["scan_at_clean"]["coverage"]["kernel"] = {}
    ok, fails, _ = vw.judge(r)
    assert not ok and any(f.startswith("W5") for f in fails)
    ok, fails, s = vw.judge(passing())
    assert ok and s["on_by_default"] is True and s["kernel"] == {"cuda:0 2-byte": "Triton"}
