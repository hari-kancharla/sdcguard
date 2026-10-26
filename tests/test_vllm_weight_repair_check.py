"""experiments/vllm_weight_repair_check.py without a GPU: its criteria (docs/METHOD.md, "Weight repair in serving") on
made-up results, and its trial loop against a stand-in server."""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import vllm_weight_repair_check as wr  # noqa: E402


def passing():
    tr = [{**t, "flip": {"layer": t["layer"]}, "alerted": True, "restored": True, "seconds": 2.0 + i * 0.1,
           "value_before": {"bits": 100 + i}, "value_after": {"bits": 100 + i}} for i, t in enumerate(wr.trials())]
    rec = [{"detector": "serve_weights", "detail": {"where": t["layer"], "finding": f, "values_changed": 1}}
           for t in tr for f in ("weights_changed", "weights_restored")]
    ident = [f"answer {i}" for i in range(wr.IDENTITY)]
    return {"ready": True, "trials": tr, "records": rec, "identity_before": ident, "identity_after": list(ident),
            "prefix_cache_emptied": True, "clean_scan_alerts": 0,
            "scan_at_end": {"coverage": {"changed": [], "repair": {"host_copy_bytes": 3 << 30, "repaired": 8,
                                                                   "failed": 0, "not_compared_since_repair": []}}}}


def test_the_trials_are_fixed_and_spread_over_bits():
    t = wr.trials()
    assert t == wr.trials() and len(t) == 8 and sorted({x["bit"] for x in t}) == [7, 10, 13, 14]
    assert all(x["layer"].startswith("model.layers.") for x in t)


def test_a_clean_repair_run_passes():
    ok, fails, s = wr.judge(passing())
    assert ok, fails
    assert s["restored"] == 8 and s["answers_identical_after"] and s["median_seconds"] <= wr.MEDIAN_S


def test_each_criterion_fails_on_its_own():
    cases = []
    r = passing(); r["trials"][0]["restored"] = False; cases.append((r, "R1"))
    r = passing(); r["trials"][1]["alerted"] = False; cases.append((r, "R1"))
    r = passing(); r["trials"][2]["value_after"] = {"bits": -1}; cases.append((r, "R1"))         # not its old value
    r = passing(); r["trials"][3] = {**r["trials"][3], "skipped": "time"}; cases.append((r, "R1"))
    r = passing(); r["records"].append({"detector": "serve_weights",
                                        "detail": {"where": "x", "finding": "weights_restore_failed"}})
    cases.append((r, "R2"))
    r = passing(); r["records"].append({"detector": "serve_weights",
                                        "detail": {"where": "model.layers.99.mlp.down_proj",
                                                   "finding": "weights_changed"}})
    cases.append((r, "R2"))
    r = passing(); r["clean_scan_alerts"] = 1; cases.append((r, "R2"))
    r = passing(); r["records"][1]["detail"]["values_changed"] = 7; cases.append((r, "R1"))  # more than the flip
    r = passing(); r["identity_after"][5] = "different"; cases.append((r, "R3"))
    r = passing(); r["identity_after"] = r["identity_after"][:10]; cases.append((r, "R3"))
    r = passing(); r["prefix_cache_emptied"] = False; cases.append((r, "R3"))
    r = passing(); r["scan_at_end"]["coverage"]["changed"] = ["model.layers.3.mlp.down_proj"]; cases.append((r, "R5"))
    r = passing(); r["scan_at_end"]["coverage"]["repair"]["not_compared_since_repair"] = ["x"]
    cases.append((r, "R5"))
    r = passing(); r["scan_at_end"] = None; cases.append((r, "R5"))
    r = passing()
    for t in r["trials"]:
        t["seconds"] = 9.0
    cases.append((r, "R4"))
    r = passing(); r["trials"][4]["seconds"] = 45.0; cases.append((r, "R4"))
    for r, tag in cases:
        ok, fails, _ = wr.judge(r)
        assert not ok and any(f.startswith(tag) for f in fails), (tag, fails)
    assert not wr.judge({"ready": False})[0]


def test_the_trial_loop_flips_waits_for_the_repair_and_reads_the_weight_back(monkeypatch):
    log, values, sent = [], {}, [0]

    def rpc(port, method, args=()):
        if method == "sdcguard_test_weight_value":
            return {"bits": values.get(args[0], 7)}
        if method == "sdcguard_test_flip_bit":
            values[args[0]] = 7 ^ (1 << args[3])
            log.append({"detector": "serve_weights", "detail": {"where": args[0], "finding": "weights_changed"}})
            return {"layer": args[0]}
        return {}

    def try_(port, model, prompt, n=32):
        sent[0] += 1
        for name in list(values):                      # the plugin restores the layer after some traffic
            if sent[0] % 8 == 0 and values[name] != 7:
                values[name] = 7
                log.append({"detector": "serve_weights", "detail": {"where": name, "finding": "weights_restored"}})
        return {"text": "x"}
    monkeypatch.setattr(wr.vn, "rpc", rpc)
    monkeypatch.setattr(wr.vn, "_try", try_)
    monkeypatch.setattr(wr.vn, "alerts", lambda path: copy.deepcopy(log))
    out = wr.run_trials(8000, "x", ["p"] * 4, wr.time.monotonic() + 3600)
    assert len(out) == 8 and all(t["alerted"] and t["restored"] for t in out)
    assert all(t["value_before"] == t["value_after"] == {"bits": 7} for t in out)
    assert all(isinstance(t["seconds"], float) for t in out)
    # two trials on one layer: the second waits for its own repair, not the first one's record
    monkeypatch.setattr(wr, "trials", lambda seed=3: [{"bit": 14, "layer": "L", "row_frac": 0.1, "col_frac": 0.2},
                                                      {"bit": 13, "layer": "L", "row_frac": 0.1, "col_frac": 0.2}])
    log.clear()
    values.clear()
    out = wr.run_trials(8000, "x", ["p"] * 4, wr.time.monotonic() + 3600)
    assert all(t["restored"] and t["alerted"] for t in out)
    assert sum(1 for a in log if a["detail"]["finding"] == "weights_restored") == 2


def test_the_repaired_layers_get_traffic_until_compared_again(monkeypatch):
    left = [3, 1, 0]
    sent = []
    monkeypatch.setattr(wr.vn, "rpc", lambda port, method, args=(): {"coverage": {"repair": {
        "not_compared_since_repair": ["x"] * left.pop(0)}}})
    monkeypatch.setattr(wr.vn, "_try", lambda port, model, prompt, n=32: sent.append(prompt) or {"text": "x"})
    r = wr.recheck_repaired(8000, ["p"] * 4, wr.time.monotonic() + 3600)
    assert r["left"] == 0 and r["rounds"] == 2 and len(sent) == 2 * wr.BURST
    monkeypatch.setattr(wr.vn, "rpc", lambda port, method, args=(): {"coverage": {"repair": {
        "not_compared_since_repair": ["x"]}}})
    r = wr.recheck_repaired(8000, ["p"] * 4, wr.time.monotonic() + 30)     # too close to the deadline: no traffic
    assert r["left"] == 1 and r["rounds"] == 0
