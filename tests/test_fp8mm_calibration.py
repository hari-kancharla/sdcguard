"""experiments/fp8mm_calibration.py's terms on the CPU: on exact arithmetic the block residual is the outputs' bf16
rounding (S); on outputs accumulated as Hopper's FP8 fast accumulation does (an emulator of the published model:
32 products a step, aligned to the largest, 13 fractional bits kept, truncated toward zero), the residual grows with
K and the accumulation term A covers it, as the noise model the check will be fitted with says."""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))
import fp8mm_calibration as fc  # noqa: E402


def _raw_exp(x):
    ax = x.abs()
    e = torch.floor(torch.log2(ax.clamp_min(1e-300)))
    e = torch.where(ax < 2.0 ** -6, torch.full_like(e, -6.0), e)
    return torch.where(ax == 0, torch.full_like(e, -1e9), e)


def fast_accum(a, b, g=32, F=13):
    """Hopper wgmma FP8 accumulation over all of K (no promotion), a: [M, K], b: [K, N] FP8 values in float64."""
    M, K = a.shape
    ea, eb = _raw_exp(a), _raw_exp(b)
    c = torch.zeros(M, b.shape[1], dtype=torch.float64)
    for k0 in range(0, K, g):
        p = a[:, k0:k0 + g, None] * b[None, k0:k0 + g, :]
        ep = torch.where(p == 0, torch.full_like(p, -1e9), ea[:, k0:k0 + g, None] + eb[None, k0:k0 + g, :])
        ec = torch.where(c == 0, torch.full_like(c, -1e9), torch.floor(torch.log2(c.abs().clamp_min(1e-300))))
        emax = torch.maximum(ep.max(1).values, ec)
        ulp = torch.where(emax < -1e8, torch.ones_like(emax), torch.pow(2.0, emax - F))
        s = (torch.trunc(p / ulp[:, None, :]) * ulp[:, None, :]).sum(1) + torch.trunc(c / ulp) * ulp
        es = torch.floor(torch.log2(s.abs().clamp_min(1e-300)))
        u2 = torch.pow(2.0, es - F)
        s = torch.where(s == 0, s, torch.trunc(s / u2) * u2)
        c = s.to(torch.float32).to(torch.float64)
    return c


def _operands(m, k, n, seed=0, dc=0.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=g) + dc
    w = torch.randn(k, n, generator=g) / k ** 0.5 + dc / k ** 0.5
    xs = (x.abs().amax(-1, keepdim=True) / 448.0)
    ws = (w.abs().amax() / 448.0).reshape(1)
    xq = (x / xs).to(torch.float8_e4m3fn)
    wq = (w / ws).to(torch.float8_e4m3fn)
    return xq, xs, wq, ws


@pytest.mark.parametrize("k", [256, 2048])
def test_exact_arithmetic_leaves_only_the_outputs_rounding(k):
    xq, xs, wq, ws = _operands(16, k, 256)
    exact = xs.double() * ws.double() * (xq.double() @ wq.double())
    y = exact.to(torch.bfloat16)
    r, S, A, B, E = fc.block_terms(y, xq, xs, wq, ws)
    assert r.shape == (16, 2) and (r.abs() / S).max() < 6
    assert torch.allclose(E, exact.reshape(16, 2, 128).sum(-1), rtol=1e-12, atol=0)


def test_fast_accumulation_grows_with_k_and_the_accumulation_term_covers_it():
    worst = {}
    for k in (256, 4096):
        xq, xs, wq, ws = _operands(8, k, 128, seed=k, dc=0.3)
        acc = fast_accum(xq.double(), wq.double())
        y = (xs.double() * ws.double() * acc).to(torch.bfloat16)
        r, S, A, B, E = fc.block_terms(y, xq, xs, wq, ws)
        worst[k] = float((r.abs() / S).max())
        assert float(fc.score(r, S, A, B, 1.0, 0.2).max()) < 6, k        # the full model allows it
    assert worst[4096] > worst[256]                                      # in bf16 units it grows with K


def test_the_experiments_emulator_and_epilogue_and_agreement():
    xq, xs, wq, ws = _operands(4, 512, 256, seed=3, dc=0.3)
    acc = fc.emulate_fast_accum(xq, wq, g=32, chunk=100)               # in column chunks
    assert torch.equal(acc, fast_accum(xq.double(), wq.double()))      # the same model as this test's
    y = fc.emulate_output(acc, xs, ws)
    a = fc.agreement(y, y.clone())
    assert a["bit_identical"] == 1.0 and a["max_steps"] == 0
    z = y.clone()
    z.view(torch.int16)[0, 0] += 3
    a = fc.agreement(z, y)
    assert a["max_steps"] == 3 and a["bit_identical"] == 1 - 1 / y.numel()
    neg = torch.tensor([-1.0, 1.0], dtype=torch.bfloat16)
    nxt = torch.tensor([-1.0078125, 1.0078125], dtype=torch.bfloat16)  # one step away from zero each
    assert fc.agreement(neg, nxt)["max_steps"] == 1


def test_the_whole_measurement_runs_and_saves_its_arrays(monkeypatch, tmp_path):
    """measure() end to end on the CPU, with vLLM's two FP8 ops and the model capture replaced: every layer, recipe
    and kernel gives its terms, its scores and (for CUTLASS) an emulation agreement, and the arrays are saved."""
    import types

    import numpy as np

    def scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=False):
        if use_per_token_if_dynamic:
            s = (x.float().abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-12)
        else:
            s = (x.float().abs().amax() / 448.0).reshape(1)
        return (x.float() / s).to(torch.float8_e4m3fn), s

    def cutlass_scaled_mm(a, b, sa, sb, out_dtype):            # the emulated fast accumulation and epilogue
        return fc.emulate_output(fc.emulate_fast_accum(a, b, g=32), sa, sb).to(out_dtype)

    ops = types.SimpleNamespace(scaled_fp8_quant=scaled_fp8_quant, cutlass_scaled_mm=cutlass_scaled_mm)
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(_custom_ops=ops))
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", ops)
    g = torch.Generator().manual_seed(1)
    layers = {f"model.layers.0.{k}": (torch.randn(40, 64, generator=g).to(torch.bfloat16),
                                      (torch.randn(n, 64, generator=g) / 8).to(torch.bfloat16))
              for k, n in (("mlp.gate_proj", 256), ("self_attn.k_proj", 64))}
    monkeypatch.setattr(fc, "capture", lambda model_id, blocks, device: (layers, list(range(40))))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda d=None: "CPU stand-in")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (9, 0))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(fc, "ROWS", 16)
    monkeypatch.setattr(fc, "EMU_ROWS", 4)
    res = fc.measure(["1.5b"], torch.device("cpu"), str(tmp_path))
    lr = res["models"]["1.5b"]["layers"]["model.layers.0.mlp.gate_proj"]
    assert lr["K"] == 64 and lr["N"] == 256 and lr["rows"] == 16
    for recipe in ("per_token", "per_tensor"):
        st = lr[f"{recipe}_cutlass"]
        assert st["emulation"]["bit_identical"] == 1.0 and st["emulation"]["g"] == 32   # the stand-in is the model
        assert st["bf16_units_max"] >= 0 and "score_max_alpha0.5_beta0.0" in st
        assert "error" in lr[f"{recipe}_torch_promoted"] or "bf16_units_max" in lr[f"{recipe}_torch_promoted"]
    z = np.load(tmp_path / "1.5b__model.layers.0.mlp.gate_proj.npz")
    assert z["per_token_cutlass_r"].shape == (16, 2) and z["prompt"].shape == (16,)
    assert z["per_tensor_cutlass_S"].dtype == np.float16 and z["per_token_cutlass_r"].dtype == np.float32
    k = res["models"]["1.5b"]["layers"]["model.layers.0.self_attn.k_proj"]
    assert k["N"] == 64 and k["per_token_cutlass"]["emulation"]["outputs"] == 4 * 64     # one partial block
