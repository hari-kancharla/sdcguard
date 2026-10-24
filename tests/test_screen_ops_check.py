"""experiments/screen_ops_check.py without a GPU: its criteria (docs/METHOD.md, "Exact operator tests for screening
GPUs") on made-up results, and on a real CPU run of the operator tests with the GPU-only parts left out."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import screen_ops_check as sc  # noqa: E402
from sdcguard import screen_ops as so  # noqa: E402

SMALL = dict(size=(64, 48), fma_n=1 << 12, sm_n=1 << 12, mem_share=0.001, mem_limit_gb=0.001)


def _results():
    clean = so.run(torch.device("cpu"), 0.0, None, seed=0, **SMALL)
    sims = {}
    for t in clean["tests"]:
        r = so.run(torch.device("cpu"), 0.0, None, seed=5, simulate=t, tests=[t], **SMALL)
        sims[t] = {"result": r["result"], "wrong": r["wrong"], "first_wrong": r["tests"][t]["first_wrong"]}
    return {"gpu": "cpu", "capability": [7, 0], "clean": clean, "simulated": sims}


def test_a_clean_run_and_caught_simulations_pass():
    res = _results()
    res["clean"]["skipped"] = {"gemm_fp8": "x", "gemm_tf32": "x", "gemm_bf16": "x"}   # as on a Turing GPU (7.0)
    ok, fails = sc.judge(res)
    assert ok, fails


def test_each_criterion_fails_on_its_own():
    base = _results()
    base["clean"]["skipped"] = {"gemm_fp8": "x", "gemm_tf32": "x", "gemm_bf16": "x"}
    r = {**base, "capability": [9, 0]}
    assert any("S1: tests skipped" in f for f in sc.judge(r)[1])          # an H100 must run them all
    r = {**base, "clean": {**base["clean"], "result": "FAIL", "wrong": 2}}
    assert any("S1: the clean run was FAIL" in f for f in sc.judge(r)[1])
    sims = {**base["simulated"], "fma_fp32": {"result": "PASS", "wrong": 0, "first_wrong": []}}
    assert any("S2: fma_fp32" in f for f in sc.judge({**base, "simulated": sims})[1])
    sims = {**base["simulated"], "memory": {"result": "FAIL", "wrong": 1,
                                            "first_wrong": [{"chunk": 1, "offset": 0, "element": 5}]}}
    assert any("S2: memory located" in f for f in sc.judge({**base, "simulated": sims})[1])
    assert sc.expected_skips((8, 9)) == set() and sc.expected_skips((8, 0)) == {"gemm_fp8"}
