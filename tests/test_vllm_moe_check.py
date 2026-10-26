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
    speed = lambda v: {"16": {"median": {"output_tokens_per_s": v}, "complete": True}}  # noqa: E731
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
                                                                    ("weights", "wrong_weights")))],
                      "drop": [{"mode": "drop", "layer": NAMES[6], "mark": NAMES[6], "detected": True,
                                "rounds_to_detect": 1, "findings": ["real_rows_unrouted"],
                                "verdicts": ["output_wrong"]}]}}


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
    r["eager"]["drop"][0]["detected"] = False
    cases.append((r, "M6: real rows without experts caught in 0 of 1"))
    r = passing()
    r["eager"]["drop"][0]["findings"] = ["wrong_experts"]
    cases.append((r, "M6"))
    r = passing()
    r["eager"]["drop"] = []
    cases.append((r, "M6"))
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


def test_nomoe_answers_and_cublas_settings_say_why_answers_differ():
    res = passing()
    res["nomoe"]["identity"] = ["a"] * vm.IDENTITY
    res["guarded"]["identity"] = ["b"] + ["a"] * (vm.IDENTITY - 1)
    res["plain"]["reductions"] = [{"bf16": True, "fp16": True}]
    res["guarded"]["reductions"] = [{"bf16": False, "fp16": False}]
    ok, fails, s = vm.judge(res)
    assert not ok and any(f.startswith("M2") for f in fails)                   # M2 as written
    assert s["identity_differ"] == {"plain_vs_guarded": 1, "guarded_vs_nomoe": 1, "plain_vs_nomoe": 0}
    assert s["reductions"]["plain"] == [{"bf16": True, "fp16": True}] and s["reductions"]["nomoe"] is None


def test_an_incomplete_speed_run_decides_nothing():
    r = passing()
    r["guarded"]["speed"]["16"]["complete"] = False      # a round with failed requests
    ok, fails, s = vm.judge(r)
    assert s["on_by_default"] is None and s["cost"]["16"]["complete"] is False


def test_a_server_that_dies_or_starts_too_late_is_recorded_not_a_crash(monkeypatch, tmp_path):
    """The run bounds each startup by the time left and writes an error instead of crashing, so SUMMARY.json is
    always written and the judge says what happened."""
    import os
    clock = [0.0]

    class P:
        pass

    def server(role, out, port, metrics_port):
        return P(), open(os.path.join(str(tmp_path), f"{role}.log"), "w")

    waits = []

    def wait_ready(port, proc, timeout):
        waits.append(timeout)
        clock[0] += 20 * 60                        # each start takes 20 minutes
        return True

    def rpc(port, method, args=()):
        raise ConnectionError("worker gone")
    monkeypatch.setattr(vm, "server", server)
    monkeypatch.setattr(vm.vs, "wait_ready", wait_ready)
    monkeypatch.setattr(vm.vs, "stop", lambda proc: {})
    monkeypatch.setattr(vm.vn, "rpc", rpc)
    monkeypatch.setattr(vm.time, "monotonic", lambda: clock[0])
    res = vm.run(str(tmp_path), 8000, 9400)
    assert res["plain"]["error"].startswith("ConnectionError") and "skipped" in res["eager"]
    assert waits[0] == min(1500.0, vm.RUN_S - 240)
    ok, fails, _ = vm.judge(res)
    assert not ok and any("plain stopped with an error" in f for f in fails)
    assert any("eager didn't start (the run's time was nearly up" in f for f in fails)
