"""experiments/vllm_moe_check.py without a GPU: its criteria (docs/METHOD.md, "MoE checks in serving") on made-up
results."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import vllm_moe_check as vm  # noqa: E402

NAMES = [f"model.layers.{i}.block_sparse_moe.experts" for i in range(24)]


ROUTERS = [f"model.layers.{i}.block_sparse_moe.experts.router" for i in range(24)]


def _stats(worst=0.3):
    return {"layers": 24, "names": NAMES, "kernel": {"cuda:0 bfloat16 act 0": [True, ""]}, "worst": worst,
            "stats": {"active_calls": 900}, "backends": ["TRITON"], "experts": 32,
            "routing": {"routers": 24, "names": ROUTERS, "coverage": {"kernel": {"cuda:0 bfloat16": "Triton"}},
                        "stats": {"calls": 900}}}


def passing():
    ident = ["a"] * vm.IDENTITY
    speed = lambda v: {"16": {"median": {"output_tokens_per_s": v}}}  # noqa: E731
    return {"plain": {"ready": True, "identity": ident, "speed": speed(1000.0)},
            "guarded": {"ready": True, "identity": ident, "moe_at_start": _stats(), "moe_clean": _stats(),
                        "clean_alerts": 0, "speed": speed(985.0)},
            "nomoe": {"ready": True, "speed": speed(1000.0)},
            "eager": {"ready": True, "moe_at_start": _stats(), "moe_clean": _stats(0.2), "clean_alerts": 0,
                      "corrupt": [{"mode": m, "layer": NAMES[k], "mark": NAMES[k], "detected": True,
                                   "rounds_to_detect": 1} for k, m in enumerate(vm.MODES)],
                      "route_corrupt": [{"mode": m, "router": ROUTERS[k], "mark": ROUTERS[k], "detected": True,
                                         "rounds_to_detect": 1, "findings": [f], "verdicts": ["output_wrong"]}
                                        for k, (m, f) in enumerate((("experts", "wrong_experts"),
                                                                    ("weights", "wrong_weights")))]}}


def test_a_run_that_meets_every_criterion_passes():
    ok, fails, s = vm.judge(passing())
    assert ok, fails
    assert s["cost"]["16"]["moe_cost"] == 0.015 and s["on_by_default"] is True and s["backends"] == ["TRITON"]


def test_each_criterion_fails_on_its_own():
    cases = []
    r = passing()
    r["nomoe"]["ready"] = False
    cases.append((r, "M1: nomoe didn't start"))
    r = passing()
    r["guarded"]["moe_at_start"]["layers"] = 0
    r["guarded"]["moe_at_start"]["not_checked"] = {"x": "the FLASHINFER_CUTLASS MoE backend"}
    cases.append((r, "M1 (guarded): 0 MoE layers"))
    r = passing()
    r["eager"]["moe_at_start"]["kernel"] = {"cuda:0 bfloat16 act 0": [False, "mismatch"]}
    cases.append((r, "Triton kernels aren't in use"))
    r = passing()
    r["guarded"]["identity"] = ["b"] * vm.IDENTITY
    cases.append((r, "M2"))
    r = passing()
    r["guarded"]["clean_alerts"] = 1
    cases.append((r, "M3 (guarded): 1 alerts"))
    r = passing()
    r["eager"]["moe_clean"]["worst"] = 2.5
    cases.append((r, "above 2.0"))
    r = passing()
    r["guarded"]["moe_clean"]["stats"] = {"active_calls": 0}
    cases.append((r, "no MoE check ran"))
    r = passing()
    r["eager"]["corrupt"][2]["detected"] = False
    cases.append((r, "M4: wrong kernels caught in 5 of 6"))
    r = passing()
    r["guarded"]["moe_at_start"]["routing"]["routers"] = 0
    cases.append((r, "M1 (guarded): the routing of 0 routers"))
    r = passing()
    r["eager"]["moe_clean"]["routing"]["coverage"]["kernel"] = {"cuda:0 bfloat16": "PyTorch (mismatch)"}
    cases.append((r, "routing check's Triton kernel isn't in use"))
    r = passing()
    r["guarded"]["moe_clean"]["routing"]["stats"] = {"calls": 0}
    cases.append((r, "M3 (guarded): no routing check ran"))
    r = passing()
    r["eager"]["route_corrupt"][1]["detected"] = False
    cases.append((r, "M5: wrong routers caught in 1 of 2"))
    r = passing()
    r["eager"]["route_corrupt"][0]["findings"] = ["wrong_weights"]
    cases.append((r, "M5"))
    for res, want in cases:
        ok, fails, _ = vm.judge(res)
        assert not ok and any(want in f for f in fails), (want, fails)


def test_a_costly_check_isnt_on_by_default():
    r = passing()
    r["guarded"]["speed"]["16"]["median"]["output_tokens_per_s"] = 950.0
    ok, _, s = vm.judge(r)
    assert ok and s["on_by_default"] is False
