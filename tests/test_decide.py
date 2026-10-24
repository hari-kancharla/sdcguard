"""sdcguard decide: the four questions kept apart, and the decision they add up to."""
import json

import pytest

from sdcguard import decide as dc
from sdcguard import kernels

K = "algoId=66 tile=MATMUL_TILE_64x16 stages=MATMUL_STAGES_64xAUTO customOption=1 clusterShape=CLUSTER_SHAPE_{c}"
STACK = {"gpu": "NVIDIA H100 80GB HBM3", "torch": "2.13.0", "cuda": "13.0", "vllm": "0.32.0", "env": {}}


def _line(m, n, k, c="1x1x1"):
    return (f"[cublasLt][9][Trace][cublasLtMatmul] A=0X1 Adesc=[type=R_16BF rows={k} cols={m} ld={k}] B=0X2 "
            f"Bdesc=[type=R_16BF rows={k} cols={n} ld={k}] C=0X3 Cdesc=[type=R_16BF rows={m} cols={n} ld={m}] D=0X3 "
            f"Ddesc=[type=R_16BF rows={m} cols={n} ld={m}] computeDesc=[computeType=COMPUTE_32F scaleType=R_32F "
            f"transa=OP_T] algo=[{K.format(c=c)}] workSpaceSizeInBytes=0")


def _quality(v="PASS", kl=2e-4, top1=0.998):
    return {"verdict": {"verdict": v, "why": f"mean KL {kl}"}, "summary": {"mean_kl": kl, "top1_agreement": top1},
            "stack": dict(STACK, vllm="0.32.1"), "ref_stack": STACK}


def _perf(tps, spread=0.01, load=None):
    rounds = [{"output_tokens_per_s": tps * f, "ttft_p99_ms": 100.0, "itl_p99_ms": 20.0}
              for f in (1 - spread, 1, 1 + spread)]
    med = {m: None for m in ("ttft_p50_ms", "itl_p50_ms", "latency_p50_ms", "latency_p99_ms")}
    med.update(output_tokens_per_s=tps, ttft_p99_ms=100.0, itl_p99_ms=20.0)
    return {"model": "m", "load": load or {"prompts": 128}, "complete": True, "rounds": rounds, "median": med}


METRICS = ("sdcguard_layers_checked 28\nsdcguard_checks_executed_total 5000\nsdcguard_reads_total 40\n"
           "sdcguard_weights_checked_ratio 0.97\n"
           'sdcguard_protection_info{UUID="GPU-1",share="0.125"} 1\n')
COV = dc.plugin_coverage([METRICS])


def test_a_clean_change_that_meets_its_target_is_accepted():
    r = dc.decide({"performance": {"required": True, "min_throughput_gain": 0.05}}, quality=_quality(),
                  alerts=[], gate={"c1": {"verdict": "PASSED"}}, perf_before=_perf(1000), perf_after=_perf(1200),
                  plugin=COV)
    assert r["decision"] == "ACCEPT" and r["arithmetic"]["verdict"] == "NO FAULT OBSERVED"
    assert r["approximation"]["verdict"] == "WITHIN" and r["performance"]["verdict"] == "MEETS"
    assert r["scope"]["stacks"]["candidate"]["vllm"] == "0.32.1"


def test_a_confirmed_wrong_output_rejects_whatever_the_quality_says():
    alerts = [{"step": 7, "detail": {"where": "model.layers.3.mlp.down_proj", "recheck": {"verdict": "output_wrong"}}}]
    r = dc.decide(quality=_quality(), alerts=alerts)
    assert r["decision"] == "REJECT" and r["arithmetic"]["verdict"] == "FAULT" and "down_proj" in r["why"][0]
    r = dc.decide(quality=_quality(), gate={"c1": {"verdict": "UNCHANGED", "flavor": "still wrong", "why": "x"}})
    assert r["decision"] == "REJECT"
    r = dc.decide(quality=_quality(), kernel_test={"counts": {"WRONG": 1}, "calls": [
        {"verdict": "WRONG", "runs": 3, "runs_wrong": 3, "call": "k | m=1"}]})
    assert r["decision"] == "REJECT"


def test_an_avoided_case_is_no_fault_on_the_candidate_but_is_noted():
    r = dc.decide(quality=_quality(), gate={"c1": {"verdict": "UNCHANGED", "flavor": "avoided", "why": "x"}})  # noqa
    assert r["arithmetic"]["verdict"] == "NO FAULT OBSERVED" and "avoided" in r["arithmetic"]["notes"][0]
    assert r["decision"] == "ACCEPT"


def test_an_unconfirmed_alert_or_no_source_is_not_decided():
    r = dc.decide(quality=_quality(), alerts=[{"step": 1, "detail": {"recheck": {"verdict": "unknown"}}}])
    assert r["decision"] == "NOT DECIDED" and r["arithmetic"]["verdict"] == "INCONCLUSIVE"
    assert dc.decide(quality=_quality())["decision"] == "NOT DECIDED"
    cleared = [{"step": 1, "detail": {"recheck": {"verdict": "output_agrees"}}}]
    assert dc.decide(quality=_quality(), alerts=cleared, plugin=COV)["arithmetic"]["verdict"] == "NO FAULT OBSERVED"


def test_quality_outside_the_budget_rejects_and_warn_needs_the_policy():
    assert dc.decide(quality=_quality("FAIL", kl=0.2), alerts=[], plugin=COV)["decision"] == "REJECT"
    assert dc.decide(quality=_quality("WARN", kl=4e-3), alerts=[], plugin=COV)["decision"] == "REJECT"
    r = dc.decide({"quality": {"accept_warn": True}}, quality=_quality("WARN", kl=4e-3), alerts=[], plugin=COV)
    assert r["decision"] == "ACCEPT"
    r = dc.decide({"quality": {"max_mean_kl": 1e-4}}, quality=_quality(kl=2e-4), alerts=[], plugin=COV)
    assert r["decision"] == "REJECT" and "policy" in r["approximation"]["why"]
    r = dc.decide(quality={"verdict": {"verdict": "INCONCLUSIVE", "why": "missing"}}, alerts=[], plugin=COV)
    assert r["decision"] == "NOT DECIDED" and r["approximation"]["verdict"] == "UNMEASURED"


def test_a_gain_inside_the_noise_or_below_target_misses():
    pol = {"performance": {"required": True, "min_throughput_gain": 0.05}}
    r = dc.decide(pol, quality=_quality(), alerts=[], plugin=COV, perf_before=_perf(1000, 0.1), perf_after=_perf(1060, 0.1))
    assert r["decision"] == "REJECT" and "spread" in r["why"][0]
    r = dc.decide(pol, quality=_quality(), alerts=[], plugin=COV, perf_before=_perf(1000), perf_after=_perf(1020))
    assert r["decision"] == "REJECT"
    r = dc.decide(pol, quality=_quality(), alerts=[], plugin=COV, perf_before=_perf(1000), perf_after=_perf(1200, load={"x": 1}))
    assert r["decision"] == "NOT DECIDED" and r["performance"]["verdict"] == "UNMEASURED"
    # not required: a miss is reported, and decides nothing
    r = dc.decide(quality=_quality(), alerts=[], plugin=COV, perf_before=_perf(1000), perf_after=_perf(900))
    assert r["performance"]["verdict"] == "MISSES" and r["decision"] == "ACCEPT"
    r = dc.decide({"performance": {"min_throughput_gain": None}}, quality=_quality(), alerts=[], plugin=COV,
                  perf_before=_perf(1000), perf_after=_perf(900))
    assert r["performance"]["verdict"] == "MEETS"


def test_scope_lists_added_kernels_with_known_issues_and_untested_ones():
    before = kernels.inventory([_line(1536, 16, 8960)])
    after = kernels.inventory([_line(1536, 16, 8960), _line(256, 16, 1536, c="4x1x1")])
    r = dc.decide(quality=_quality(), alerts=[], plugin=COV, kernels_before=before, kernels_after=after)
    s = r["scope"]
    assert len(s["kernels"]["added"]) == 1 and s["added_kernels_with_known_issues"][0]["issues"] == ["6862950"]
    assert len(s["added_kernels_not_tested"]) == 1
    text = dc.render(r)
    assert "matches known issue 6862950" in text and "DECISION: ACCEPT" in text


def test_a_decision_stops_applying_when_the_stack_changes():
    r = dc.decide(quality=_quality(), alerts=[], plugin=COV)
    assert dc.still_applies(r, dict(STACK, vllm="0.32.1"))["applies"]
    w = dc.still_applies(r, dict(STACK, vllm="0.33.0"))
    assert not w["applies"] and "vllm" in w["differences"][0]
    assert dc.still_applies({"scope": {}}, STACK)["applies"] is None


def test_the_policy_rejects_unknown_keys():
    with pytest.raises(ValueError):
        dc.policy_with_defaults({"quality": {"max_kl": 1}})


def test_the_command_writes_the_decision_and_its_text(tmp_path, capsys):
    q = tmp_path / "q.json"
    q.write_text(json.dumps(_quality()))
    al = tmp_path / "a.jsonl"
    al.write_text("")
    met = tmp_path / "w0.txt"
    met.write_text(METRICS)
    out = tmp_path / "d.json"
    assert dc.main(["--quality", str(q), "--alerts", str(al), "--plugin-metrics", str(met), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["decision"] == "ACCEPT" and (tmp_path / "d.txt").exists()
    assert "DECISION: ACCEPT" in capsys.readouterr().out


def test_no_alerts_mean_nothing_without_coverage():
    r = dc.decide(quality=_quality(), alerts=[])
    assert r["decision"] == "NOT DECIDED" and "coverage" in r["arithmetic"]["unresolved"][0]
    nothing = dc.plugin_coverage(["sdcguard_layers_checked 0\nsdcguard_checks_executed_total 0\n"])
    r = dc.decide(quality=_quality(), alerts=[], plugin=nothing)
    assert r["decision"] == "NOT DECIDED" and not nothing["checked"]
    # a confirmed wrong output, or a changed weight, rejects whatever the coverage says
    bad = [{"step": 3, "detail": {"recheck": {"verdict": "output_wrong"}}}]
    r = dc.decide(quality=_quality(), alerts=bad)
    assert r["decision"] == "REJECT" and r["arithmetic"]["verdict"] == "FAULT"
    bad = [{"step": 3, "detail": {"recheck": {"verdict": "weights_changed"}}}]
    r = dc.decide(quality=_quality(), alerts=bad)
    assert r["decision"] == "REJECT" and r["weights"]["verdict"] == "CHANGED" and r["arithmetic"]["verdict"] != "FAULT"
    assert COV["checked"] and COV["workers"][0]["share"] == "0.125"


def test_no_slower_tolerates_noise_and_a_gain_must_clear_it():
    pol = {"performance": {"required": True, "min_throughput_gain": 0.0}}
    r = dc.decide(pol, quality=_quality(), alerts=[], plugin=COV, perf_before=_perf(1000, 0.02),
                  perf_after=_perf(995, 0.02))
    assert r["performance"]["verdict"] == "MEETS"
    r = dc.decide(pol, quality=_quality(), alerts=[], plugin=COV, perf_before=_perf(1000, 0.01),
                  perf_after=_perf(900, 0.01))
    assert r["performance"]["verdict"] == "MISSES" and r["decision"] == "REJECT"


def test_a_plugin_that_checks_little_of_the_model_doesnt_stand_for_no_fault():
    lm_head_only = dc.plugin_coverage([METRICS.replace("0.97", "0.08")])
    r = dc.decide(quality=_quality(), alerts=[], plugin=lm_head_only)
    assert r["decision"] == "NOT DECIDED" and "8%" in r["arithmetic"]["unresolved"][0]
    r = dc.decide({"arithmetic": {"min_weights_checked": 0.05}}, quality=_quality(), alerts=[], plugin=lm_head_only)
    assert r["decision"] == "ACCEPT"


def test_changed_weights_are_a_fault_without_a_kept_row():
    ev = [{"step": None, "detail": {"where": "model.layers.10.mlp.down_proj", "weights_changed_since_load": True,
                                    "captured": None}}]
    r = dc.decide(quality=_quality(), alerts=ev, plugin=COV)
    assert r["decision"] == "REJECT" and r["why"][0].startswith("weights: ") and "step" not in r["why"][0]
    assert "model.layers.10.mlp.down_proj changed since load (1 alert)" in r["why"][0]
    # the GPU may multiply right with a wrong weight: the arithmetic stays NO FAULT OBSERVED, the weights CHANGED
    assert r["arithmetic"]["verdict"] == "NO FAULT OBSERVED" and r["weights"]["verdict"] == "CHANGED"
    assert r["arithmetic"]["sources"]["plugin_alerts"]["weights_changed"] == 1
    assert r["arithmetic"]["sources"]["plugin_alerts"]["confirmed_wrong"] == 0
    assert "1a. Weight integrity" in dc.render(r) and "CHANGED" in dc.render(r)
    ev = [{"step": None, "detail": {"where": "x", "weights_changed_since_load": False, "captured": {"step": 9}}}]
    r = dc.decide(quality=_quality(), alerts=ev, plugin=COV)
    assert r["decision"] == "NOT DECIDED" and "at step 9" in r["arithmetic"]["unresolved"][0]


def test_a_leak_between_requests_rejects_even_with_right_arithmetic():
    leak = {"verdict": "LEAK", "why": ["primer dependence: 88 of 114 probe answers changed"],
            "primer": {"pairs": 114, "pairs_differing": 88}, "canary": {"answered": 48, "leaks": []}}
    r = dc.decide(quality=_quality(), alerts=[], plugin=COV, isolation=leak)
    assert r["decision"] == "REJECT" and r["why"][0].startswith("isolation") and r["arithmetic"]["verdict"] == \
        "NO FAULT OBSERVED"
    assert "4. Isolation" in dc.render(r)
    clean = {"verdict": "NO LEAK SEEN", "why": ["ok"]}
    assert dc.decide(quality=_quality(), alerts=[], plugin=COV, isolation=clean)["decision"] == "ACCEPT"
    r = dc.decide({"isolation": {"required": True}}, quality=_quality(), alerts=[], plugin=COV)
    assert r["decision"] == "NOT DECIDED" and "isolation" in r["why"][0]
    r = dc.decide({"isolation": {"required": True}}, quality=_quality(), alerts=[], plugin=COV,
                  isolation={"verdict": "INCONCLUSIVE", "why": ["a control disagreed"]})
    assert r["decision"] == "NOT DECIDED"


def test_fp8_quantization_findings_are_their_own_section_and_reject():
    fp8 = {"detector": "serve_fp8", "step": None, "detail": {"where": "model.layers.3.mlp.experts (expert inputs)",
                                                             "finding": "scale_from_outside_rows"}}
    cov = {"workers": [{"layers_checked": 10, "checks_executed": 5, "fp8_calls_judged": 40}], "checked": True,
           "min_weights_checked": 0.9}
    r = dc.decide(None, quality=None, alerts=[fp8], plugin=cov)
    assert r["decision"] == "REJECT" and r["quantization"]["verdict"] == "WRONG"
    assert "outside the batch" in r["why"][0] and r["arithmetic"]["sources"]["plugin_alerts"]["alerts"] == 0
    assert r["arithmetic"]["verdict"] == "NO FAULT OBSERVED"       # the GEMMs' own checks found nothing
    clean = dc.decide(None, alerts=[], plugin=cov)
    assert clean["quantization"]["verdict"] == "RIGHT"
    none = dc.decide(None, alerts=[], plugin={**cov, "workers": [{**cov["workers"][0], "fp8_calls_judged": None}]})
    assert none["quantization"]["verdict"] == "UNMEASURED"
    assert "1b. FP8 quantization" in dc.render(r)
