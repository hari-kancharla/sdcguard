"""sdcguard.screen_ops on the CPU: every test's inputs keep its arithmetic exact, a clean run passes, a value made
wrong in each test is found and located, the memory test finds a changed value, and the report and exit code."""
import json

import pytest
import torch

from sdcguard import screen_ops as so

SMALL = dict(size=(64, 48), fma_n=1 << 12, sm_n=1 << 12, mem_share=0.001, mem_limit_gb=0.001)


def test_the_integer_ranges_keep_every_matmul_exact():
    """The largest product times the inner size stays under the integer range of the accumulation (2^53 for FP64,
    2^24 for FP32 and the 16-bit types' FP32 accumulation, 2^13 for FP8's fast accumulation) and of FP16's output; and
    the inputs are exact in their type (2^11 for TF32, 256 for BF16, 2048 for FP16, 16 for FP8 e4m3)."""
    limit = {"fp64": 2 ** 53, "fp32": 2 ** 24, "tf32": 2 ** 24, "bf16": 2 ** 24, "fp16": 65504, "fp8": 2 ** 13,
             "int8": 2 ** 31}
    exact_in = {"fp64": 2 ** 53, "fp32": 2 ** 24, "tf32": 2 ** 11, "bf16": 256, "fp16": 2048, "fp8": 16, "int8": 127}
    for name, p in so.GEMM.items():
        assert p["r"] ** 2 * p["k"] < limit[name], name
        assert p["r"] <= exact_in[name], name
    lo, hi = so.SCALES["fp32"]
    assert 2 ** 21 * 2.0 ** hi < 3e38 and 2.0 ** lo > 2 ** -126          # scaled results stay normal in fp32


@pytest.mark.parametrize("name", ["fp64", "fp32", "bf16", "fp16", "int8"])
def test_each_matmul_set_is_exact_scaled_and_transposed(name):
    gs = so.GemmSet(name, 3, so.GEMM[name]["k"] // 8, torch.device("cpu"), size=(32, 24))
    for rnd in range(3):
        ea, eb = so._scales(name, rnd, 0)
        for tr in (False, True):
            with so._Matmul(False):
                y, want = gs.run(ea, eb, tr)
            assert y.dtype == want.dtype and torch.equal(y, want), (name, rnd, tr)


def test_a_clean_run_passes_and_writes_its_report(tmp_path):
    r = so.run(torch.device("cpu"), 0.0, str(tmp_path), seed=1, **SMALL)
    assert r["result"] == "PASS" and r["wrong"] == 0 and r["values_checked"] > 0, r
    assert set(r["tests"]) == {"gemm_fp64", "gemm_fp32", "gemm_bf16", "gemm_fp16", "gemm_int8", "fma_fp32",
                               "fma_fp64", "sm_consistency", "memory"}
    assert set(r["skipped"]) == {"gemm_tf32", "gemm_fp8"} and r["rounds"] >= 1
    with open(r["path"]) as f:
        assert json.load(f)["result"] == "PASS"


@pytest.mark.parametrize("test", ["gemm_fp32", "gemm_bf16", "gemm_int8", "fma_fp32", "fma_fp64", "sm_consistency",
                                  "memory"])
def test_a_wrong_value_is_found_and_located(test):
    r = so.run(torch.device("cpu"), 0.0, None, seed=2, simulate=test, **SMALL)
    assert r["result"] == "FAIL" and r["wrong"] == 1, r["tests"]
    t = r["tests"][test]
    assert t["wrong"] == 1 and len(t["first_wrong"]) == 1
    w = t["first_wrong"][0]
    want_element = {"gemm_fp32": 777, "gemm_bf16": 777, "gemm_int8": 777, "fma_fp32": 5, "fma_fp64": 5,
                    "sm_consistency": 3}
    if test == "memory":
        assert w["chunk"] == 0 and w["offset"] + w["element"] == 12345 and w["complement"] is False
    else:
        assert w["element"] == want_element[test] and w["round"] == 0 and w["got"] != w["expected"]
    others = [n for n, x in r["tests"].items() if n != test and x["wrong"]]
    assert not others


def test_the_command_line_exit_codes(tmp_path, capsys):
    args = ["--device", "cpu", "--minutes", "0", "--out", str(tmp_path), "--tests", "gemm_fp32,fma_fp32"]
    assert so.main(args) == 0
    assert "PASS" in capsys.readouterr().out
    assert so.main(args + ["--simulate", "fma_fp32"]) == 1
    assert "SIMULATED" in capsys.readouterr().out
    assert so.main(["--device", "nonsense", "--minutes", "0"]) == 2
