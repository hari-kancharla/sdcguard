"""The math-mode check: clean results pass, small injected errors are caught at the size the card
claims, tensor-core rounding (emulated from a published model of it) doesn't set it off."""
import math

import pytest
import torch

from sdcguard import modes

CPU = torch.device("cpu")
SHAPE = (128, 256, 256)


def _gen(seed=5):
    return torch.Generator().manual_seed(seed)


def test_every_mode_runs_clean_on_the_cpu():
    r = modes.screen(CPU, 0, shape=SHAPE)
    assert r["ran"] == list(modes.MODES) and r["not_checked"] == {}
    assert r["wrong"] == 0 and r["matmuls"] == len(modes.MODES) * len(modes.VARIANTS)
    for m, rec in r["modes"].items():
        assert rec["matmuls"] == len(modes.VARIANTS) and rec["blocks_checked"] > 0
        if m == "int8":
            assert rec["worst_z"] is None and rec["typical_error_caught"] == "exact sums"
        else:
            assert 0 < rec["worst_z"] < 2 and 0 < rec["typical_error_caught"] < 1e-2


@pytest.mark.parametrize("mode", ["bf16", "fp16", "tf32", "fp32", "fp64"])
@pytest.mark.parametrize("variant", [0, 2, 4])
def test_errors_are_caught_at_the_size_the_card_claims(mode, variant):
    dist, scale = modes.VARIANTS[variant]
    clean = modes.check_once(mode, CPU, SHAPE, dist, scale, _gen())
    claim = clean["typical_caught"]
    assert clean["wrong_blocks"] == 0 and claim > 0
    big = modes.check_once(mode, CPU, SHAPE, dist, scale, _gen(), fault=3 * claim)
    small = modes.check_once(mode, CPU, SHAPE, dist, scale, _gen(), fault=claim / 3)
    assert big["wrong_blocks"] == 1 and small["wrong_blocks"] == 0


def test_int8_has_to_be_exact():
    assert modes.check_once("int8", CPU, SHAPE, "normal", 1.0, _gen())["wrong_blocks"] == 0
    assert modes.check_once("int8", CPU, SHAPE, "normal", 1.0, _gen(), fault=1)["wrong_blocks"] == 1


def test_a_non_finite_result_is_wrong():
    def run(mode, a, b):
        c = modes.gemm(mode, a, b)
        c[3, 7] = float("nan")
        return c
    r = modes.check_once("fp32", CPU, SHAPE, "normal", 1.0, _gen(), run=run)
    assert r["wrong_blocks"] == 1 and math.isinf(r["worst_z"])


def _tensor_core(nfma, frac_bits):
    """Tensor-core accumulation as modelled by Khattak and Mikaitis (arXiv 2512.07004): each step adds nfma
    exact products and the accumulator, every term truncated to frac_bits below the largest exponent, and the
    sum truncated to fp32."""
    def rtz32(x):
        y = x.float()
        return torch.where(y.double().abs() > x.abs(), torch.nextafter(y, torch.zeros_like(y)), y)

    def run(mode, a, b):
        a, b = a.double(), b.double()
        acc = torch.zeros(a.shape[0], b.shape[1], dtype=torch.float32)
        for s in range(0, a.shape[1], nfma):
            terms = torch.cat([a[:, s:s + nfma, None] * b[None, s:s + nfma, :], acc.double()[:, None, :]], dim=1)
            e = torch.frexp(terms.abs().amax(dim=1))[1].double()
            q = torch.pow(2.0, e - 1 - frac_bits)[:, None, :]
            acc = rtz32((torch.trunc(terms / q) * q).sum(dim=1))
        return acc
    return run


@pytest.mark.parametrize("mode,nfma,frac", [("tf32", 4, 24), ("bf16", 8, 24), ("fp16", 16, 25), ("tf32", 8, 25)])
def test_tensor_core_rounding_stays_far_from_the_alarm(mode, nfma, frac):
    run = _tensor_core(nfma, frac)
    shape = (64, 512, 256)
    for dist, scale in modes.VARIANTS[:3]:
        r = modes.check_once(mode, CPU, shape, dist, scale, _gen(), run=run, nfma=nfma)
        assert r["wrong_blocks"] == 0 and r["worst_z"] < 1.5, (dist, r)
    caught = modes.check_once(mode, CPU, shape, "normal", 1.0, _gen(), run=run, nfma=nfma,
                              fault=3 * r["typical_caught"])
    assert caught["wrong_blocks"] == 1


def test_tensor_core_steps_by_generation():
    assert modes.n_fma("bf16", (9, 0)) == 16 and modes.n_fma("tf32", (9, 0)) == 8      # Hopper
    assert modes.n_fma("fp16", (10, 0)) == 16 and modes.n_fma("tf32", (12, 0)) == 8    # Blackwell
    assert modes.n_fma("bf16", (8, 0)) == 8 and modes.n_fma("tf32", (8, 9)) == 4       # Ampere, Ada
    assert modes.n_fma("fp16", (7, 5)) == 4 and modes.n_fma("fp16", (13, 0)) == 4      # older, unknown
    assert modes.n_fma("fp32", (9, 0)) is None and modes.n_fma("int8", (9, 0)) is None


def test_modes_that_cant_run_are_listed_not_passed(monkeypatch):
    monkeypatch.setattr(modes, "available", lambda m, dev: "not here" if m in ("fp64", "int8") else None)
    r = modes.screen(CPU, 0, shape=SHAPE)
    assert r["ran"] == ["bf16", "fp16", "tf32", "fp32"] and r["not_checked"] == {"fp64": "not here", "int8": "not here"}


def test_a_mode_that_errors_is_listed_and_the_rest_still_run(monkeypatch):
    real = modes.check_once

    def flaky(mode, *a, **k):
        if mode == "int8":
            raise RuntimeError("shape not supported")
        return real(mode, *a, **k)
    monkeypatch.setattr(modes, "check_once", flaky)
    r = modes.screen(CPU, 0, shape=SHAPE)
    assert "int8" not in r["ran"] and "shape not supported" in r["not_checked"]["int8"]
    assert r["modes"]["fp32"]["matmuls"] == len(modes.VARIANTS)
