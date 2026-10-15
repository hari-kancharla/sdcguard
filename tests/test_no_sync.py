"""A check must never make the CPU wait for the GPU (on a GPU that serializes the whole step).

Runs matmul_zscores on fake tensors (shapes only, no data). Any operation that needs actual
values on the CPU (.item(), indexing with a 0-d tensor, Python bool/float of a tensor) raises
DataDependentOutputException here. On check run #3 an x[gpu_index] slipped in and doubled the
cost of a checked step on an H100; this test catches that class of bug on a CPU-only machine.
"""
import pytest
import torch

from sdcguard import checksum as cs

fake_tensor = pytest.importorskip("torch._subclasses.fake_tensor")

SHAPES = [(64, 128, 256, False), (64, 128, 300, True), (32, 64, 50257, False)]   # 300, 50257: remainder blocks


def _run_fake(shapes, batched=False):
    mode = fake_tensor.FakeTensorMode()
    cs._SIGNS.clear()
    cs._PAIR_W.clear()
    try:
        with mode:
            for m, k, n, bias in shapes:
                if batched:
                    A = torch.empty(3, m, k, dtype=torch.bfloat16)
                    B = torch.empty(3, k, n, dtype=torch.bfloat16)
                    C = torch.empty(3, m, n, dtype=torch.bfloat16)
                    b = None
                else:
                    A = torch.empty(m, k, dtype=torch.bfloat16)
                    B = torch.empty(n, k, dtype=torch.bfloat16).t()      # like nn.Linear: W^T
                    C = torch.empty(m, n, dtype=torch.bfloat16)
                    b = torch.empty(n, dtype=torch.bfloat16) if bias else None
                cs.matmul_zscores(A, B, C, b)
                cs.matmul_zscores(A, B, C, b, alpha=2.0, beta=0.5)
    finally:
        cs._SIGNS.clear()            # don't leak fake constants into other tests
        cs._PAIR_W.clear()


def test_checks_never_read_gpu_values_cpu_path():
    _run_fake(SHAPES)
    _run_fake(SHAPES[:2], batched=True)


def test_checks_never_read_gpu_values_nvidia_path(monkeypatch):
    monkeypatch.setattr(cs, "_use_lowp_gemm", lambda X: True)
    monkeypatch.setattr(cs, "_lowp_gemm", lambda A3, Bs: torch.matmul(A3.float(), Bs.float()))
    monkeypatch.setattr(cs, "_c_gemm", lambda flat, W: flat.float() @ W.float())
    _run_fake(SHAPES)
    _run_fake(SHAPES[:2], batched=True)


def test_this_test_would_catch_the_old_bug():
    def leaky(z):
        t = torch.max(z.view(-1), 0)[1]
        return z.view(-1)[t]                 # the check run #3 bug: index with a 0-d tensor
    mode = fake_tensor.FakeTensorMode()
    with pytest.raises(Exception):
        with mode:
            leaky(torch.empty(4, 5))
