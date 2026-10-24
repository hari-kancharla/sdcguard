"""experiments/vllm_norm_check.py without a GPU: the trials it draws and the criteria (docs/METHOD.md, "Norm checks
in serving") as its judge applies them, on made-up results."""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import vllm_norm_check as nc  # noqa: E402

NAMES = [f"model.layers.{i}.{k}" for i in range(28) for k in ("input_layernorm", "post_attention_layernorm")] + \
    ["model.norm"]
Q3NAMES = [f"model.layers.{i}.self_attn.{k}" for i in range(28) for k in ("q_norm", "k_norm")]


def _stats(n, worst=2.1):
    return {"norms": n, "kernel": {"cuda:0 bfloat16": [True, ""]}, "worst": worst, "names": NAMES}


def _trial(t, detected=True, before=0.75, after=1.5, other=0):
    return {**t, "flip": {"before": before, "after": after}, "back": {"after": before}, "detected": detected,
            "alerts": int(detected), "finite_alerts_other_layers": other, "other_layers": ["x"] * other,
            "answers_changed": 0, "relative_change": 1.0}


def passing():
    ident = ["a"] * nc.IDENTITY
    res = {"plain": {"ready": True, "identity": ident, "speed": {"16": {"median": {"output_tokens_per_s": 1000.0}}}},
           "guarded": {"ready": True, "identity": ident, "norms_at_start": _stats(57), "norms_clean": _stats(57),
                       "clean_alerts": 0, "trials": [_trial(t) for t in nc.trials(NAMES)],
                       "speed": {"16": {"median": {"output_tokens_per_s": 980.0}}}},
           "nonorms": {"ready": True, "speed": {"16": {"median": {"output_tokens_per_s": 1000.0}}}},
           "eager": {"ready": True, "norms_at_start": _stats(57), "norms_clean": _stats(57, 0.9), "clean_alerts": 0,
                     "exact": {"prompts_holding_marker_naturally": [],
                               "trials": [{"alerts": 1, "named": [[f"cmpl-{k}-0"]], "target": f"cmpl-{k}"}
                                          for k in range(nc.MARK_TRIALS)]}},
           "q3plain": {"ready": True, "identity": ident},
           "q3guarded": {"ready": True, "identity": ident, "norms_at_start": _stats(113), "norms_clean": _stats(113),
                         "clean_alerts": 0, "trials": [_trial(t) for t in nc.trials(Q3NAMES, (14,), 4, 1)]}}
    return res


def test_the_trials_cover_every_bit_as_written():
    t = nc.trials(NAMES)
    assert len(t) == 40 and t == nc.trials(NAMES)
    assert {b: sum(x["bit"] == b for x in t) for b in nc.PER_BIT} == nc.PER_BIT
    assert all(x["layer"] in NAMES and 0 <= x["col_frac"] < 1 for x in t)
    q = nc.trials(Q3NAMES, (14,), 4, 1)
    assert len(q) == 4 and all(x["bit"] == 14 and x["layer"].endswith(("q_norm", "k_norm")) for x in q)


def test_a_run_that_meets_every_criterion_passes():
    ok, fails, s = nc.judge(passing())
    assert ok, fails
    assert s["cost"]["16"]["norm_cost"] == 0.02 and s["on_by_default"] is True
    assert s["by_bit"][15]["detected"] == 3 and s["clean_worst"]["eager"] == 0.9


def test_each_criterion_fails_on_its_own():
    cases = []
    r = passing()
    r["q3guarded"]["ready"] = False
    cases.append((r, "N1: q3guarded didn't start"))
    r = passing()
    r["guarded"]["norms_at_start"]["norms"] = 55
    cases.append((r, "55 norm layers checked"))
    r = passing()
    r["eager"]["norms_at_start"]["kernel"] = {"cuda:0 bfloat16": [False, "kernel mismatch"]}
    cases.append((r, "Triton kernel isn't in use"))
    r = passing()
    r["guarded"]["identity"] = ["b"] + r["guarded"]["identity"][1:]
    cases.append((r, "N2: guarded"))
    r = passing()
    r["eager"]["clean_alerts"] = 2
    cases.append((r, "N3 (eager): 2 alerts"))
    r = passing()
    r["q3guarded"]["norms_clean"]["worst"] = 4.3
    cases.append((r, "above the bound"))
    r = passing()
    sure = next(i for i, t in enumerate(r["guarded"]["trials"]) if t["bit"] == 5)
    r["guarded"]["trials"][sure]["detected"] = False
    cases.append((r, "N4: 1 of 24 flips"))
    r = passing()
    r["guarded"]["trials"][-1]["finite_alerts_other_layers"] = 1
    r["guarded"]["trials"][-1]["other_layers"] = ["model.layers.3.mlp.down_proj"]
    cases.append((r, "finite-score alerts on other layers"))
    r = passing()
    r["guarded"]["trials"][0]["back"] = {"error": "HTTP 500"}
    cases.append((r, "weren't undone"))
    r = passing()
    r["eager"]["exact"]["trials"][2]["named"] = [["cmpl-9-0"]]
    cases.append((r, "N5: exact naming in 4 of 5"))
    r = passing()
    r["q3guarded"]["trials"][1]["detected"] = False
    cases.append((r, "N6"))
    for res, want in cases:
        ok, fails, _ = nc.judge(res)
        assert not ok and any(want in f for f in fails), (want, fails)


def test_misses_below_the_sure_bits_and_zero_weights_dont_fail_the_run():
    r = passing()
    for t in r["guarded"]["trials"]:
        if t["bit"] in nc.REPORT_BITS:
            t["detected"] = False
    zero = next(t for t in r["guarded"]["trials"] if t["bit"] == 7)
    zero["flip"] = {"before": 0.0, "after": 1e-38}
    zero["back"] = {"after": 0.0}
    zero["detected"] = False
    ok, fails, s = nc.judge(r)
    assert ok, fails
    assert s["by_bit"][0]["detected"] == 0 and s["by_bit"][7]["zero_weight"] == 1


def test_a_costly_check_is_reported_as_not_on_by_default():
    r = passing()
    r["guarded"]["speed"]["16"]["median"]["output_tokens_per_s"] = 900.0
    ok, fails, s = nc.judge(r)
    assert ok and s["on_by_default"] is False and s["cost"]["16"]["norm_cost"] == 0.1


def test_relative_change_is_left_out_for_non_finite_values():
    assert nc.relative_change(0.75, 1.5) == 1.0 and nc.relative_change(-2.0, 2.0) == 2.0
    assert nc.relative_change(1.0, "inf") is None and nc.relative_change(1.0, "nan") is None
    assert nc.relative_change(0.0, 1e-38) is None and nc.relative_change(None, 1.0) is None
