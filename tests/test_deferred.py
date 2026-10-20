"""The GPU-friendly paths: deferred verdicts, the fp32-output GEMM split, and odd outputs."""
import math

import pytest
import torch
import torch.nn as nn

from sdcguard import FaultInjector, SDCError, SDCGuard
from sdcguard import checksum as cs
from sdcguard.checksum import check_matmul, matmul_zscores, resolve


def _layer(m=256, k=384, n=640, seed=0, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(m, k, generator=g).to(dtype)
    B = (torch.randn(k, n, generator=g) / k**0.5).to(dtype)
    C = (A.float() @ B.float()).to(dtype)
    return A, B, C


def test_queued_checks_resolve_like_immediate_ones():
    A, B, C = _layer()
    bad = C.clone()
    bad[17, 256:288] = 0
    stats = [matmul_zscores(A, B, C), matmul_zscores(A, B, bad)]
    good, wrong = resolve(stats, z=6.0)
    assert good.ok and not wrong.ok
    assert wrong.row == 17 and wrong.col_block == 2         # columns 256..383 are block 2
    ref = check_matmul(A, B, bad)
    assert ref.row == wrong.row and ref.col_block == wrong.col_block
    assert math.isclose(ref.max_z, wrong.max_z, rel_tol=1e-6)


def test_remainder_block_location():
    A, B, C = _layer(n=300)                                  # blocks: 0-127, 128-255, rest 256-299
    bad = C.clone()
    bad[5, 260:290] = 0
    res = check_matmul(A, B, bad)
    assert not res.ok and res.row == 5 and res.col_block == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 1e30])
def test_nonfinite_and_huge_outputs_are_flagged(value):
    A, B, C = _layer()
    bad = C.clone()
    bad[3, 7] = value
    res = check_matmul(A, B, bad)
    assert not res.ok
    if value != 1e30:
        assert res.nonfinite


def _emulated_lowp_gemm(A3, Bs):
    """What cuBLAS computes for low-precision inputs with an fp32 output (fp32 accumulate)."""
    return torch.matmul(A3.float(), Bs.float())


def _emulated_c_gemm(flat, W):
    return flat.float() @ W.float()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_split_gemm_path_matches_fp32_path(monkeypatch, dtype):
    A, B, C = _layer(dtype=dtype)
    ref_ok = check_matmul(A, B, C)
    monkeypatch.setattr(cs, "_use_lowp_gemm", lambda A3: True)
    monkeypatch.setattr(cs, "_lowp_gemm", _emulated_lowp_gemm)
    monkeypatch.setattr(cs, "_c_gemm", _emulated_c_gemm)
    res = check_matmul(A, B, C)
    assert res.ok and res.max_z < 3
    assert abs(res.max_z - ref_ok.max_z) < 0.05
    bad = C.clone()
    bad[9, 0:32] = 0
    assert not check_matmul(A, B, bad).ok


def test_split_gemm_handles_fp16_values_beyond_fp16_range(monkeypatch):
    # dW = dY^T X where X has big activations: block sums of B exceed fp16's max (65504).
    g = torch.Generator().manual_seed(1)
    A = (torch.randn(128, 512, generator=g) * 0.01).half()
    B = (torch.randn(512, 256, generator=g) * 2000).half()
    C = (A.float() @ B.float()).half()
    monkeypatch.setattr(cs, "_use_lowp_gemm", lambda A3: True)
    monkeypatch.setattr(cs, "_lowp_gemm", _emulated_lowp_gemm)
    monkeypatch.setattr(cs, "_c_gemm", _emulated_c_gemm)
    assert check_matmul(A, B, C).ok


def test_split3_reconstructs_fp32():
    x = torch.randn(1000) * torch.logspace(-6, 6, 1000)
    parts = cs._split3(x.unsqueeze(-1), torch.bfloat16)
    back = parts.float().sum(-1)
    assert torch.allclose(back, x, rtol=2e-8, atol=0)


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))


def _step(model, x):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(x).float().pow(2).mean()
    loss.backward()


def test_alerts_arrive_when_the_with_block_exits():
    model, x = _model(), torch.randn(64, 256)
    inj = FaultInjector(rate=1.0, seed=3).arm_on([model[0]])
    guard = SDCGuard()
    with inj:
        with guard.at_step(4):
            _step(model, x)
            assert guard.stats["alerts"] == 0 and len(guard._pending) > 0   # still queued
    inj.remove_hooks()
    assert guard.stats["alerts"] >= 1 and not guard._pending
    assert guard.log.events[0].step == 4


def test_immediate_mode_raises_inside_the_op():
    model = _model()
    x = torch.randn(64, 256, generator=torch.Generator().manual_seed(1))
    inj = FaultInjector(rate=1.0, seed=8).arm_on([model[2]])      # a fault well above the bf16 check's floor
    guard = SDCGuard(immediate=True, raise_on_alert=True)
    with pytest.raises(SDCError):
        with inj, guard:
            _step(model, x)
    inj.remove_hooks()


def test_alerts_name_the_layer_and_pass():
    torch.manual_seed(0)
    model = nn.Sequential()
    model.add_module("fc1", nn.Linear(256, 512))
    model.add_module("act", nn.GELU())
    model.add_module("fc2", nn.Linear(512, 256))
    x = torch.randn(64, 256)
    inj = FaultInjector(rate=1.0, seed=7).arm_on([model.fc2])
    guard = SDCGuard().watch(model)
    with inj:
        with guard:
            _step(model, x)
    inj.remove_hooks()
    guard.unwatch()
    fwd = [e for e in guard.log.events if e.detail["pass"] == "forward"]
    assert fwd and fwd[0].detail["where"] == "fc2"


def test_injector_matches_checks_and_records_z():
    model, x = _model(), torch.randn(64, 256)
    inj = FaultInjector(rate=1.0, seed=11).arm_on([model[0]])
    guard = SDCGuard(on_check_start=lambda name, C: inj.match(C),
                     on_check=lambda name, tok, res: inj.resolve(tok, res.ok, res.max_z))
    with inj:
        with guard:
            _step(model, x)
    inj.remove_hooks()
    ev = inj.events[0]
    assert ev["checked"] and "z" in ev
    assert ev["caught"] == (ev["z"] == "inf" or ev["z"] == "nan" or ev["z"] > 6)


def test_start_step_delays_faults():
    model, x = _model(), torch.randn(64, 256)
    inj = FaultInjector(rate=1.0, seed=1, start_step=5).arm_on([model[0]])
    with inj:
        for step in range(8):
            inj.step = step
            _step(model, x)
    inj.remove_hooks()
    assert inj.events and min(e["step"] for e in inj.events) == 5


def test_changes_that_cancel_in_the_plain_sum_are_still_caught():
    A, B, C = _layer()
    s = cs._signs(128, C.device)
    i, j = 3, next(t for t in range(4, 128) if s[t] != s[3])     # same block, opposite signs
    bad = C.clone()
    d = float(C[5, 0:128].float().abs().max()) * 4
    bad[5, i] = (bad[5, i].float() + d).to(C.dtype)
    bad[5, j] = (bad[5, j].float() - d).to(C.dtype)
    res = check_matmul(A, B, bad)
    assert not res.ok and res.row == 5 and res.col_block == 0


def test_c_side_gemm_path_on_strided_blocks(monkeypatch):
    # remainder spans are not a plain view of C: the fast path must fall back cleanly
    A, B, C = _layer(n=300)
    monkeypatch.setattr(cs, "_use_lowp_gemm", lambda A3: True)
    monkeypatch.setattr(cs, "_lowp_gemm", _emulated_lowp_gemm)
    monkeypatch.setattr(cs, "_c_gemm", _emulated_c_gemm)
    assert check_matmul(A, B, C).ok
    bad = C.clone()
    bad[2, 270:290] = 0
    res = check_matmul(A, B, bad)
    assert not res.ok and res.col_block == 2


@pytest.mark.parametrize("dout", [1024, 1000])          # 1000: a remainder block of 104
def test_fast_path_on_linear_layers_matches_reference(monkeypatch, dout):
    # nn.Linear checks B = W^T (a transposed view) plus a bias: the NVIDIA path makes B
    # contiguous and sums blocks with a skinny GEMM; it must agree with the plain path.
    from sdcguard.checksum import check_linear
    g = torch.Generator().manual_seed(4)
    x = torch.randn(256, 512, generator=g).bfloat16()
    W = (torch.randn(dout, 512, generator=g) / 512**0.5).bfloat16()
    b = (torch.randn(dout, generator=g) * 0.02).bfloat16()
    y = torch.nn.functional.linear(x, W, b)
    ref = check_linear(x, W, b, y)
    monkeypatch.setattr(cs, "_use_lowp_gemm", lambda A3: True)
    monkeypatch.setattr(cs, "_lowp_gemm", _emulated_lowp_gemm)
    monkeypatch.setattr(cs, "_c_gemm", _emulated_c_gemm)
    fast = check_linear(x, W, b, y)
    assert ref.ok and fast.ok and abs(ref.max_z - fast.max_z) < 0.05
    for cols in ((64, 96), (980, 1000) if dout == 1000 else (992, 1024)):
        bad = y.clone()
        bad[11, cols[0]:cols[1]] = 0
        res = check_linear(x, W, b, bad)
        assert not res.ok and res.row == 11 and res.col_block == cols[0] // 128


def test_fast_path_on_bmm(monkeypatch):
    monkeypatch.setattr(cs, "_use_lowp_gemm", lambda A3: True)
    monkeypatch.setattr(cs, "_lowp_gemm", _emulated_lowp_gemm)
    monkeypatch.setattr(cs, "_c_gemm", _emulated_c_gemm)
    g = torch.Generator().manual_seed(1)
    A = torch.randn(4, 64, 128, generator=g).bfloat16()
    B = (torch.randn(4, 128, 300, generator=g) / 128**0.5).bfloat16()
    C = torch.bmm(A, B)
    assert check_matmul(A, B, C).ok
    C2 = C.clone()
    C2[3, 10, 260:290] = 0
    res = check_matmul(A, B, C2)
    assert not res.ok and res.batch == 3 and res.row == 10 and res.col_block == 2


def test_fp16_gpu_path_one_bad_column_does_not_spoil_the_others(monkeypatch):
    """fp16 on NVIDIA scales B's block sums into range per column: an Inf in one block of B (say
    dO after a loss-scale overflow) must leave the other blocks judgeable, with no false alarm."""
    monkeypatch.setattr(cs, "_use_lowp_gemm", lambda A3: True)
    monkeypatch.setattr(cs, "_lowp_gemm", _emulated_lowp_gemm)
    monkeypatch.setattr(cs, "_c_gemm", _emulated_c_gemm)
    g = torch.Generator().manual_seed(0)
    A = torch.randn(64, 96, generator=g).half().abs()
    B = torch.randn(96, 300, generator=g).half()
    B[7, 200] = float("inf")                       # only block 1 sees it (blocks 0 and 1 are scaled together)
    res = check_matmul(A, B, (A.float() @ B.float()).half())
    assert res.ok, res
    C = (A.float() @ B.float()).half()
    C[5, 10] = 1000.0                              # corruption in block 0 is still caught there
    res = check_matmul(A, B, C)
    assert not res.ok and (res.row, res.col_block) == (5, 0)


@pytest.mark.parametrize("ratio", [1e2, 1e4, 1e5])
def test_fp16_gpu_path_blocks_of_very_different_size(monkeypatch, ratio):
    monkeypatch.setattr(cs, "_use_lowp_gemm", lambda A3: True)
    monkeypatch.setattr(cs, "_lowp_gemm", _emulated_lowp_gemm)
    monkeypatch.setattr(cs, "_c_gemm", _emulated_c_gemm)
    g = torch.Generator().manual_seed(1)
    A = torch.randn(256, 512, generator=g).half()
    Bf = torch.randn(512, 256, generator=g) * 1e-3
    Bf[:, :128] *= ratio
    B = Bf.half()
    assert check_matmul(A, B, (A.float() @ B.float()).half()).ok
