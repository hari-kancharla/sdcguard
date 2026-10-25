"""sdcguard.screen_ops on the CPU: every test's inputs keep its arithmetic exact (every input set's shape too), a clean
run passes, a value made wrong in each test is found and located, the memory test finds a changed value, the
realistic-data matmuls are quiet when right and find a value off against the reference or between two runs, the
floors the module states, and the report and exit code."""
import json

import pytest
import torch

from sdcguard import screen_ops as so

SMALL = dict(size=(64, 48), fma_n=1 << 12, sm_n=1 << 12, mem_share=0.001, mem_limit_gb=0.001,
             shapes=((None, 1.0), ((32, 64), 1.0), ((48, 32), 0.75), ((32, 48), 0.5)),
             real=((16, 256, 512), (48, 128, 1024)))
REAL = ["gemm_bf16_real", "gemm_fp16_real"]


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
    # every input set: K no larger than the precision's (so still exact), and M, N, K multiples of 16 with M over 16
    # (torch._scaled_mm and torch._int_mm need them)
    assert len(so.SHAPES) == so.SETS and so.SHAPES[0][0] is None
    for name, p in so.GEMM.items():
        for shape, share in so.SHAPES:
            m, n = so.SIZE if shape is None else shape
            k = max(16, int(p["k"] * share) // 16 * 16)
            assert k <= p["k"] and m % 16 == 0 and n % 16 == 0 and k % 16 == 0 and m > 16, (name, shape, k)


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
    assert set(r["skipped"]) == {"gemm_tf32", "gemm_fp8", *REAL} and r["rounds"] >= 1          # (REAL: GPUs only)
    assert {t["rounds"] for n, t in r["tests"].items() if n.startswith("gemm_")} == {r["rounds"]}
    with open(r["path"]) as f:
        assert json.load(f)["result"] == "PASS"


@pytest.mark.parametrize("test", ["gemm_fp32", "gemm_bf16", "gemm_int8", "fma_fp32", "fma_fp64", "sm_consistency",
                                  "memory", "gemm_bf16_real", "gemm_fp16_real"])
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
    elif test.endswith("_real"):
        rows, cols = so.real_samples(w["seed"], w["shape"])
        assert (w["row"], w["col"]) == (int(rows[0]), int(cols[0])) and w["check"] == "reference"
        assert w["round"] == 0 and w["score"] > so.REAL_LINE and t["worst_score"] > so.REAL_LINE
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


def test_the_input_sets_turn_two_a_round(monkeypatch):
    seen = []
    real_add = so._Found.add

    def add(self, got, want, where):
        if self.name == "gemm_fp32":
            seen.append((where["round"], where["set"]))
        return real_add(self, got, want, where)
    monkeypatch.setattr(so._Found, "add", add)
    r = so.run(torch.device("cpu"), 0.0, None, seed=4, tests=["gemm_fp32"], min_rounds=3, **SMALL)
    assert r["rounds"] == 3 and r["result"] == "PASS"
    sets = sorted(set(seen))                             # each set twice a round: plain and transposed
    assert sets == [(0, 0), (0, 1), (1, 2), (1, 3), (2, 0), (2, 1)] and len(seen) == 12


@pytest.mark.parametrize("name", ["fp64", "fp32", "bf16", "fp16", "int8"])
def test_every_full_size_input_set_is_exact(name):
    """The sets run() builds on a GPU, at their full shapes and inner sizes, come out exact on the CPU (scaled and
    transposed): the shapes keep every product and partial sum exact, not only on paper."""
    p = so.GEMM[name]
    for s, (shape, share) in enumerate(so.SHAPES):
        k = max(16, int(p["k"] * share) // 16 * 16)
        gs = so.GemmSet(name, 1000 + s, k, torch.device("cpu"), so.SIZE if shape is None else shape)
        ea, eb = so._scales(name, 0, s)
        for tr in (False, True):
            with so._Matmul(False):
                y, want = gs.run(ea, eb, tr)
            assert torch.equal(y, want), (name, s, tr)


def test_the_realistic_data_matmuls_are_quiet_when_right():
    r = so.run(torch.device("cpu"), 0.0, None, seed=1, tests=REAL, **SMALL)
    assert r["result"] == "PASS" and not r["skipped"], r
    for n in REAL:
        t = r["tests"][n]
        assert t["wrong"] == 0 and t["worst_score"] <= 1.0 and t["rounds"] == r["rounds"]
        m, nn, _ = SMALL["real"][0]
        assert t["values_checked"] >= so.REAL_SAMPLES + m * nn


def test_the_simulated_value_is_always_over_the_line():
    """The demo's moved value (3% of it plus 0.01, away from zero) scores over the line whatever the value: over many
    seeds in both dtypes, at the realistic-data shapes the tests use."""
    for dt in (torch.bfloat16, torch.float16):
        for seed in range(60):
            rr = so.real_round(dt, SMALL["real"][seed % 2], seed, torch.device("cpu"), wrong="both")
            assert float(rr["scores"][0]) > 3 * so.REAL_LINE, (dt, seed, float(rr["scores"][0]))
            assert len(set(zip(rr["rows"].tolist(), rr["cols"].tolist()))) == so.REAL_SAMPLES   # no repeats


def test_matmul_settings_are_put_back_as_they_were():
    """Inside: TF32 as asked, reduced-precision reductions and fp16 accumulation off. After: every setting as it was,
    the reductions' split-K part too (setting the plain flag back would turn it on again)."""
    m = torch.backends.cuda.matmul
    saved = {n: so._Matmul._get(n) for n in so._Matmul.RED}
    acc16 = getattr(m, "allow_fp16_accumulation", None)
    pair = isinstance(saved[so._Matmul.RED[0]], tuple)
    try:
        if acc16 is not None:                                # (PyTorch 2.7+) on before: off inside, on again after
            m.allow_fp16_accumulation = True
            with so._Matmul(False):
                assert m.allow_fp16_accumulation is False
            assert m.allow_fp16_accumulation is True
            m.allow_fp16_accumulation = False
        # (PyTorch allows split-K off only with reduced precision off)
        for start, inside in (((True, True), (False, True)), ((False, False), (False, False))):
            for n in so._Matmul.RED:
                setattr(m, n, start if pair else start[0])
            with so._Matmul(True):
                assert all(so._Matmul._get(n) == (inside if pair else False) for n in so._Matmul.RED)
                assert getattr(m, "allow_fp16_accumulation", False) is False
            assert all(so._Matmul._get(n) == (start if pair else start[0]) for n in so._Matmul.RED)
    finally:
        for n, v in saved.items():
            setattr(m, n, v)
        if acc16 is not None:
            m.allow_fp16_accumulation = acc16


@pytest.mark.parametrize("transposed", [False, True])
def test_a_value_that_changes_between_two_runs_is_found_by_the_repeat(transposed):
    rr = so.real_round(torch.bfloat16, (16, 256, 512), 9, torch.device("cpu"), transposed, wrong="second")
    assert float(rr["scores"].max()) <= 1.0                  # the reference sees the first run, which is right
    f = so._Found("gemm_bf16_real")
    f.add(rr["again"], rr["y"], {"check": "repeat"})
    assert f.wrong == 1 and f.first[0]["element"] == int(rr["rows"][0]) * 256 + int(rr["cols"][0])
    rr = so.real_round(torch.bfloat16, (16, 256, 512), 9, torch.device("cpu"), transposed)
    assert torch.equal(rr["y"], rr["again"]) and float(rr["scores"].max()) <= 1.0


def test_the_floors_the_module_states():
    """'a median sampled output moved by more than about 1.3% to 1.9% in bf16 or 0.3% to 0.9% in fp16, by K, is
    always found': serve_spot.min_caught over Gaussian outputs of each realistic-data matmul's K."""
    from sdcguard import serve_spot as sp
    g = torch.Generator().manual_seed(0)
    med = {}
    for _, _, k in so.REAL:
        x = torch.randn(16, k, generator=g).double()
        w = torch.randn(64, k, generator=g).double() / k ** 0.5
        acc, norm = x @ w.t(), x.norm(dim=1)[:, None] * w.norm(dim=1)[None, :]
        for dt in (torch.bfloat16, torch.float16):
            rel = [sp.min_caught(dt, k, float(norm[i, j]), float(acc[i, j])) / abs(float(acc[i, j]))
                   for i in range(16) for j in range(64)]
            med[(dt, k)] = float(torch.tensor(rel).median())
    bf = [v for (dt, _), v in med.items() if dt == torch.bfloat16]
    fp = [v for (dt, _), v in med.items() if dt == torch.float16]
    assert 0.0125 <= min(bf) and max(bf) <= 0.0195 and 0.0025 <= min(fp) and max(fp) <= 0.0095, med
