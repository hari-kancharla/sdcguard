"""experiments/vllm_change_decisions.py without a GPU: its decisions from files like the run writes, and its criteria
(docs/METHOD.md)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import vllm_change_decisions as cd  # noqa: E402

METRICS = ("sdcguard_layers_checked 112\nsdcguard_checks_executed_total 9000\nsdcguard_reads_total 50\n"
           "sdcguard_weights_checked_ratio 0.95\n")


def _quality(v="PASS"):
    return {"verdict": {"verdict": v, "why": "x"}, "summary": {"mean_kl": 1e-4, "top1_agreement": 0.999},
            "stack": {"vllm": "0.32.0"}, "ref_stack": {"vllm": "0.32.0"}}


def _perf(tps):
    return {"model": "model", "load": {"prompts": 128}, "complete": True,
            "rounds": [{"output_tokens_per_s": tps * f, "ttft_p99_ms": 100.0, "itl_p99_ms": 20.0}
                       for f in (0.99, 1.0, 1.01)],
            "median": {"output_tokens_per_s": tps, "ttft_p99_ms": 100.0, "itl_p99_ms": 20.0}}


def _write(out, role, quality="PASS", tps=1000.0, alerts=()):
    if role != "base":
        json.dump(_quality(quality), open(os.path.join(out, f"{role}_quality.json"), "w"))
    json.dump(_perf(tps), open(os.path.join(out, f"{role}_perf16.json"), "w"))
    open(os.path.join(out, f"{role}_metrics.txt"), "w").write(METRICS)
    with open(os.path.join(out, f"{role}_alerts.jsonl"), "w") as f:
        for a in alerts:
            f.write(json.dumps(a) + "\n")


FAULT_ALERT = {"step": 40, "detail": {"where": cd.FAULT_LAYER, "recheck": {"verdict": "weights_changed"}}}


def _run(tmp_path, **over):
    out = str(tmp_path)
    roles = {"base": {}, "plain": {}, "same": {}, "triton": {"tps": 900.0}, "awq": {"quality": "WARN", "tps": 1250.0},
             "fault": {"alerts": [FAULT_ALERT]}}
    roles.update(over)
    for r, kw in roles.items():
        _write(out, r, **kw)
    res = {r: {"ready": True} for r in cd.ROLES}
    res["decisions"] = cd.decide_all(out, list(cd.ROLES))
    return res, out


def test_the_expected_decisions_meet_the_criteria(tmp_path):
    res, out = _run(tmp_path)
    ok, fails, s = cd.judge(res)
    assert ok, fails
    d = s["decisions"]
    assert d["hold_same"]["decision"] == "ACCEPT" and d["hold_fault"]["decision"] == "REJECT"
    assert d["hold_triton"]["decision"] == "REJECT" and d["hold_triton"]["performance"] == "MISSES"
    assert d["speed_awq"]["decision"] == "ACCEPT" and d["hold_awq"]["decision"] == "REJECT"
    assert os.path.exists(os.path.join(out, "decision_hold_fault.txt"))


def test_a_fault_that_isnt_caught_fails_the_criteria(tmp_path):
    res, _ = _run(tmp_path, fault={})
    ok, fails, _ = cd.judge(res)
    assert not ok and any(f.startswith("2:") for f in fails) and any(f.startswith("3:") for f in fails)


def test_a_false_alarm_on_a_clean_role_fails_the_criteria(tmp_path):
    res, _ = _run(tmp_path, triton={"alerts": [dict(FAULT_ALERT, detail={"where": "x", "recheck":
                                                                          {"verdict": "output_wrong"}})]})
    ok, fails, _ = cd.judge(res)
    assert not ok and any(f.startswith("4:") for f in fails)


def test_a_server_that_didnt_start_fails_first():
    ok, fails, _ = cd.judge({"base": {"ready": True}, "same": {"ready": False}})
    assert not ok and fails[0].startswith("1:")


def test_cost_compares_base_with_plain_and_times_the_first_alert(tmp_path):
    res, out = _run(tmp_path)
    open(os.path.join(out, "base_server.log"), "w").write("INFO GPU KV cache size: 1,000,000 tokens\n")
    open(os.path.join(out, "plain_server.log"), "w").write("INFO GPU KV cache size: 1,050,000 tokens\n")
    res["fault"] = {"fault": {"injected_at": 100.0}}
    with open(os.path.join(out, "fault_alerts.jsonl"), "w") as f:
        f.write(json.dumps(dict(FAULT_ALERT, time=103.5)) + "\n")
    c = cd.cost(res, out)
    assert c["kv_cache_tokens"] == {"plain": 1050000, "base": 1000000} and c["fault_seconds_to_first_alert"] == 3.5
    assert c["perf16"]["output_tokens_per_s"]["change"] == 0.0
