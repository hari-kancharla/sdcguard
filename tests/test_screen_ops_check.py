"""experiments/screen_ops_check.py without a GPU: its criteria (docs/METHOD.md, "Exact operator tests for screening
GPUs") on made-up results, and on a real CPU run of the operator tests with the GPU-only parts left out."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import screen_ops_check as sc  # noqa: E402
from sdcguard import screen_ops as so  # noqa: E402

SMALL = dict(size=(64, 48), fma_n=1 << 12, sm_n=1 << 12, mem_share=0.001, mem_limit_gb=0.001,
             shapes=((None, 1.0), ((32, 64), 1.0), ((48, 32), 0.75), ((32, 48), 0.5)),
             real=((16, 256, 512), (48, 128, 1024)))
TURING = {"gemm_fp8": "x", "gemm_tf32": "x", "gemm_bf16": "x", "gemm_bf16_real": "x"}


@pytest.fixture(autouse=True)
def _small_real(monkeypatch):
    monkeypatch.setattr(so, "REAL", SMALL["real"])          # the judge reads the realistic-data shapes from so.REAL


def _results():
    clean = so.run(torch.device("cpu"), 0.0, None, seed=0, tests=sc.ALL, **SMALL)   # (the realistic-data ones too)
    sims = {}
    for t in sc.sims(clean["tests"]):
        base = t.split(".")[0]
        r = so.run(torch.device("cpu"), 0.0, None, seed=sc.SIM_SEED, simulate=t, tests=[base], **SMALL)
        sims[t] = {"result": r["result"], "wrong": r["wrong"], "first_wrong": r["tests"][base]["first_wrong"]}
    return {"gpu": "cpu", "capability": [7, 0], "clean": clean, "simulated": sims}


def test_a_clean_run_and_caught_simulations_pass():
    res = _results()
    assert {"gemm_bf16_real", "gemm_fp16_real"} <= set(res["clean"]["tests"]) <= set(res["simulated"])
    assert {"gemm_bf16_real.repeat", "gemm_fp16_real.repeat"} <= set(res["simulated"])
    assert res["simulated"]["gemm_fp16_real.repeat"]["first_wrong"][0]["check"] == "repeat"
    res["clean"]["skipped"] = dict(TURING)                                         # as on a Turing GPU (7.0)
    ok, fails = sc.judge(res)
    assert ok, fails


def test_each_criterion_fails_on_its_own():
    base = _results()
    base["clean"]["skipped"] = dict(TURING)
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
    tests = {**base["clean"]["tests"], "gemm_fp16_real": {**base["clean"]["tests"]["gemm_fp16_real"],
                                                          "worst_score": 1.6}}
    assert any("S3: gemm_fp16_real" in f for f in sc.judge({**base, "clean": {**base["clean"], "tests": tests}})[1])
    fw = dict(base["simulated"]["gemm_bf16_real"]["first_wrong"][0], col=-1)
    sims = {**base["simulated"], "gemm_bf16_real": {"result": "FAIL", "wrong": 1, "first_wrong": [fw]}}
    assert any("S2: gemm_bf16_real located" in f for f in sc.judge({**base, "simulated": sims})[1])
    fw = dict(base["simulated"]["gemm_bf16_real"]["first_wrong"][0], seed=1)        # found in another round's data
    sims = {**base["simulated"], "gemm_bf16_real": {"result": "FAIL", "wrong": 1, "first_wrong": [fw]}}
    assert any("S2: gemm_bf16_real found a value somewhere else" in f for f in sc.judge({**base, "simulated": sims})[1])
    fw = dict(base["simulated"]["gemm_fp16_real.repeat"]["first_wrong"][0], element=0)
    sims = {**base["simulated"], "gemm_fp16_real.repeat": {"result": "FAIL", "wrong": 1, "first_wrong": [fw]}}
    assert any("S2: gemm_fp16_real.repeat located" in f for f in sc.judge({**base, "simulated": sims})[1])
    sims = {k: v for k, v in base["simulated"].items() if k != "gemm_bf16_real.repeat"}
    assert any("S2: gemm_bf16_real.repeat" in f for f in sc.judge({**base, "simulated": sims})[1])
