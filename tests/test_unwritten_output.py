"""A real silent failure, emulated: part of a matmul's output is never written.

Reported on NVIDIA's developer forum on Oct 2, 2026 (thread 384873; reproduction at
github.com/Yeosangho/mps-cublaslt-fp16-corruption-repro): on H200 under MPS with
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE below 100, some cuBLASLt kernels return CUBLAS_STATUS_SUCCESS with no
Xid, but part of the output is never written. In the reported 16x2048x1000 fp16/bf16 case (the last layer
of ResNet-50 at batch 16), columns 256..999 keep whatever the buffer held: zeros, stale values or NaN.
NVIDIA reproduced it on Oct 6, 2026. A software bug, not a broken GPU, but the same kind of silent wrong
result, so the checks that watch a model's own matmuls should name the layer the first time it happens.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sdcguard import SDCGuard, ServeGuard

M, K, N = 16, 2048, 1000          # the reported shape
FIRST_UNWRITTEN = 256


def _layer(dt):
    torch.manual_seed(0)
    return nn.Linear(K, N).to(dt)


def _leftovers(kind, dt, shape):
    """What an unwritten part of the buffer can hold."""
    if kind == "zeros":
        return torch.zeros(shape, dtype=dt)
    if kind == "nan":
        return torch.full(shape, float("nan"), dtype=dt)
    # stale: an earlier result of a similar size, left in memory the allocator handed out again
    g = torch.Generator().manual_seed(7)
    return (torch.randn(M, K, generator=g).to(dt) @ torch.randn(K, N, generator=g).to(dt) / K ** 0.5)[:, :shape[1]]


def _unwritten_linear(kind, real=F.linear):
    def linear(x, w, b=None):
        y = real(x, w, b)
        if tuple(w.shape) == (N, K):
            y = y.clone()
            y[..., FIRST_UNWRITTEN:] = _leftovers(kind, y.dtype, (y.shape[0], N - FIRST_UNWRITTEN))
        return y
    return linear


@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["zeros", "stale", "nan"])
def test_training_check_names_the_layer(dt, kind, monkeypatch):
    lin = _layer(dt)
    events = []
    g = SDCGuard(every=1, on_alert=events.append).protect(nn.Sequential(lin))
    x = torch.randn(M, K, generator=torch.Generator().manual_seed(1)).to(dt)
    monkeypatch.setattr(torch.nn.functional, "linear", _unwritten_linear(kind))
    with g.on_step(0):
        y = lin(x)
    assert events, "the unwritten output went unnoticed"
    fw = [e for e in events if e.detail["where"] == "0"]
    assert fw and all(e.detail["col_block"] >= FIRST_UNWRITTEN // 128 for e in fw)
    if kind == "nan":
        assert all(e.detail["nonfinite"] for e in fw)


@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["zeros", "stale", "nan"])
def test_serving_check_flags_it(dt, kind):
    lin = _layer(dt)
    sg = ServeGuard()
    slot = sg.add_layer(lin, "fc")
    x = torch.randn(M, K, generator=torch.Generator().manual_seed(2)).to(dt)
    y = _unwritten_linear(kind)(x, lin.weight, lin.bias).detach()
    sg.check(slot, x, y)
    z = sg.read_slot(slot)[0]
    assert z > 6.0                      # NaN outputs score inf


def test_the_correct_output_passes_both_checks():
    for dt in (torch.float16, torch.bfloat16):
        lin = _layer(dt)
        x = torch.randn(M, K, generator=torch.Generator().manual_seed(3)).to(dt)
        events = []
        g = SDCGuard(every=1, on_alert=events.append).protect(nn.Sequential(lin))
        with g.on_step(0):
            lin(x)
        sg = ServeGuard()
        slot = sg.add_layer(lin, "fc")
        sg.check(slot, x, lin(x).detach())
        assert not events and sg.read_slot(slot)[0] < 6.0


def test_the_gpu_experiment_measures_it(monkeypatch):
    """experiments/mps_unwritten.py runs this layer on a real H200 under MPS (scripts/modal_mps.py). Its own
    counting has to work: here, on the CPU, with the unwritten output emulated."""
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "mps_unwritten.py")
    spec = importlib.util.spec_from_file_location("mps_unwritten", path)
    mu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mu)
    clean = mu.one_dtype(torch.float16, torch.device("cpu"))
    assert clean["wrong_outputs"] == 0 and clean["training_check"]["alerts"] == 0
    assert clean["serving_check"] == {"layers": 1, "alerts": 0, "first": None}
    monkeypatch.setattr(torch.nn.functional, "linear", _unwritten_linear("stale"))
    bad = mu.one_dtype(torch.float16, torch.device("cpu"))
    assert bad["wrong_outputs"] > 10000 and bad["training_check"]["alerts"] >= 1
    assert bad["serving_check"]["alerts"] == 1 and bad["serving_check"]["first"]["col_block"] >= 2
