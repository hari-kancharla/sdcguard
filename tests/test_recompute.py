"""The recompute check (recompute.py): checkpointed blocks are compared with their recompute, through
fingerprints of their outputs' raw bits."""
from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn
import torch.utils.checkpoint as tuc

from sdcguard import SDCGuard
from sdcguard.recompute import RecomputeCheck, fingerprint


class Flaky(nn.Module):
    """A block whose output changes on chosen calls: a transient fault in some operation inside it."""

    def __init__(self, width=64, bad_calls=(), delta=1e-3, op="norm"):
        super().__init__()
        self.lin = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.calls, self.bad_calls, self.delta, self.op = 0, set(bad_calls), delta, op

    def forward(self, x):
        # The same operations run on every call (a hardware fault changes values, not the graph; checkpointing
        # also requires the recompute to save the same tensors as the forward)
        self.calls += 1
        h = self.norm(torch.nn.functional.gelu(self.lin(x)))
        out = x + h
        with torch.no_grad():
            corr = torch.zeros_like(out)
            if self.calls in self.bad_calls:
                if self.op == "norm":                # a small error in a non-matmul op: invisible to checksums
                    corr += self.delta * (h.abs() > 0.5).to(h.dtype)
                else:
                    # one element of the block's output, its lowest bit flipped (exactly: out + corr is the flipped
                    # value). Flipped in an intermediate value instead, the change could round away in the sum
                    # after it, depending on the platform's math library (it did on arm64): no fault left to see.
                    v = out.detach().reshape(-1)[3:4].clone()
                    flipped = (v.view(torch.int32) ^ 1).view(torch.float32)
                    corr.view(-1)[3] = (flipped - v)[0]
        return out + corr


class Net(nn.Module):
    def __init__(self, blocks, reentrant=False):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.reentrant = reentrant

    def forward(self, x):
        for b in self.blocks:
            x = tuc.checkpoint(b, x, use_reentrant=self.reentrant)
        return x


def _step(net, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(8, 64, requires_grad=True)
    net(x).pow(2).mean().backward()


@pytest.mark.parametrize("reentrant", [False, True])
def test_clean_blocks_match_exactly(reentrant):
    torch.manual_seed(0)
    net = Net([Flaky() for _ in range(3)], reentrant)
    with RecomputeCheck() as rc:
        _step(net)
        assert rc.flush() == []
    assert rc.stats["recomputes_compared"] == 3 and rc.stats["mismatches"] == 0
    assert all(b.calls == 2 for b in net.blocks)                  # forward + recompute, nothing extra


@pytest.mark.parametrize("reentrant", [False, True])
@pytest.mark.parametrize("bad_call,op", [(2, "norm"), (1, "norm"), (2, "bit")])
def test_a_transient_fault_in_one_run_is_caught_and_the_block_named(reentrant, bad_call, op):
    torch.manual_seed(0)
    net = Net([Flaky(), Flaky(bad_calls={bad_call}, op=op), Flaky()], reentrant)
    seen = []
    with RecomputeCheck(on_alert=seen.append) as rc:
        _step(net)
        events = rc.flush()
    assert len(events) == 1 and seen == events
    ev = events[0]
    assert ev.detector == "recompute" and ev.detail["where"] == "Flaky" and ev.detail["outputs_that_differ"] == [0]
    assert ev.detail["block_call"] == 2                           # the second block of the step
    assert rc.stats["recomputes_compared"] == 3


def test_a_fault_that_repeats_exactly_is_not_seen():
    """The documented limit: the same wrong bits in both runs look clean (matmul checksums and the float64
    `sdcguard check` cover that case)."""
    torch.manual_seed(0)
    net = Net([Flaky(bad_calls={1, 2})])
    with RecomputeCheck() as rc:
        _step(net)
        assert rc.flush() == []


def test_full_recompute_even_with_early_stop():
    """Non-reentrant checkpointing stops recomputing once it has the tensors it needs; the check turns that
    off for its blocks, so the whole block reruns and its outputs exist to compare."""
    torch.manual_seed(0)

    def two_outputs(x):
        a = torch.sin(x) * 2
        return a, a.exp()           # the second output is only needed for the result, not saved for backward

    x = torch.randn(16, requires_grad=True)
    with RecomputeCheck() as rc:
        a, b = tuc.checkpoint(two_outputs, x, use_reentrant=False)
        (a.sum() + b.sum()).backward()
        rc.flush()
    assert rc.stats["recomputes_compared"] == 1


def test_no_backward_means_nothing_to_compare_and_nothing_leaks():
    torch.manual_seed(0)
    net = Net([Flaky(), Flaky()])
    with RecomputeCheck() as rc:
        with torch.no_grad():
            net(torch.randn(4, 64))
        assert rc.flush() == []
    assert rc.stats["recomputes_compared"] == 0 and rc.stats["never_recomputed"] == 2 and not rc._open


def test_checkpoint_sequential_and_name_imported_references_are_routed():
    orig = tuc.checkpoint
    fake = types.ModuleType("fake_hf_modeling_utils")
    fake.checkpoint = orig                                       # like `from torch.utils.checkpoint import checkpoint`
    fake.ckpt_alias = orig                                       # like torch's checkpoint_wrapper alias
    sys.modules["fake_hf_modeling_utils"] = fake
    try:
        torch.manual_seed(0)
        seq = nn.Sequential(Flaky(), Flaky(bad_calls={2}), Flaky(), Flaky())
        with RecomputeCheck() as rc:
            assert fake.checkpoint is not orig and fake.ckpt_alias is not orig and tuc.checkpoint is not orig
            x = torch.randn(8, 64, requires_grad=True)
            tuc.checkpoint_sequential(seq, 2, x, use_reentrant=False).sum().backward()
            ev = rc.flush()
            y = fake.checkpoint(Flaky(bad_calls={1}), torch.randn(8, 64, requires_grad=True), use_reentrant=False)
            y.sum().backward()
            ev2 = rc.flush()
        assert len(ev) == 1 and len(ev2) == 1
        assert fake.checkpoint is orig and fake.ckpt_alias is orig and tuc.checkpoint is orig       # restored
    finally:
        del sys.modules["fake_hf_modeling_utils"]


def test_gradients_are_unchanged_by_the_check():
    def grads(install):
        torch.manual_seed(0)
        net = Net([Flaky() for _ in range(2)])
        torch.manual_seed(1)
        x = torch.randn(8, 64, requires_grad=True)
        if install:
            with RecomputeCheck() as rc:
                net(x).pow(2).mean().backward()
                rc.flush()
        else:
            net(x).pow(2).mean().backward()
        return [p.grad.clone() for p in net.parameters()] + [x.grad.clone()]
    for a, b in zip(grads(False), grads(True)):
        assert torch.equal(a, b)


def test_fingerprint_sees_any_single_change_and_swaps():
    torch.manual_seed(0)
    for dt in (torch.float32, torch.bfloat16, torch.float16, torch.float64):
        x = torch.randn(1000, 77).to(dt)
        f = fingerprint(x)
        for i in (0, 251, 252, 50000, x.numel() - 1):
            y = x.clone().view(-1)
            iv = y.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[y.element_size()])
            iv[i] ^= 1                                                  # the lowest bit of one element
            assert not torch.equal(fingerprint(y.view_as(x)), f), (dt, i)
        for i, j in ((0, 1), (0, 251), (5, 5 + 251 * 3), (10, 76990)):
            y = x.clone().view(-1)
            if torch.equal(y[i], y[j]):
                continue
            y[i], y[j] = x.view(-1)[j], x.view(-1)[i]
            assert not torch.equal(fingerprint(y.view_as(x)), f), (dt, i, j)
    assert torch.equal(fingerprint(torch.zeros(0)), torch.zeros(3, dtype=torch.int64))
    b = torch.tensor([True, False, True])
    assert not torch.equal(fingerprint(b), fingerprint(~b))
    assert fingerprint(torch.randn(3, 5)[:, ::2]).shape == (3,)              # non-contiguous input


def test_calibrate_reports_determinism():
    torch.manual_seed(0)
    net = Net([Flaky() for _ in range(2)])
    rc = RecomputeCheck()
    res = rc.calibrate(lambda: _step(net), steps=2)
    assert res["deterministic"] is True and res["recomputes_compared"] == 4 and res["mismatches"] == 0
    flaky = Net([Flaky(bad_calls={4})])
    assert RecomputeCheck().calibrate(lambda: _step(flaky), steps=3)["mismatches"] == 1
    assert RecomputeCheck().calibrate(lambda: None)["deterministic"] is False


def test_sdcguard_recompute_option_reports_on_checked_steps_only():
    torch.manual_seed(0)
    net = Net([Flaky(), Flaky(bad_calls={2, 6}), Flaky()])         # bad on step 0 (call 2) and step 2 (call 6)
    seen = []
    g = SDCGuard(recompute=True, every=2, on_alert=seen.append).protect(net)
    try:
        for step in range(3):
            with g.on_step(step):
                _step(net, seed=step)
    finally:
        g.unprotect()
    rec = [e for e in seen if e.detector == "recompute"]
    assert [e.step for e in rec] == [0, 2]                       # step 1 wasn't checked: no compare, no alert
    assert g.stats["recompute_compared"] == 6 and g.stats["recompute_mismatches"] == 2
    assert tuc.checkpoint.__module__ == "torch.utils.checkpoint"   # put back by unprotect()
    assert any(e.detector == "recompute" for e in g.log.events)


def test_sdcguard_default_mode_with_recompute():
    torch.manual_seed(0)
    net = Net([Flaky(bad_calls={1})])
    g = SDCGuard(recompute=True)
    with g.at_step(5):
        _step(net)
    assert g.stats["recompute_mismatches"] == 1 and g.log.events[-1].step == 5
    g._recompute.uninstall()


def test_raise_on_alert_raises_after_the_step():
    from sdcguard.guard import SDCError
    torch.manual_seed(0)
    net = Net([Flaky(bad_calls={2})])
    g = SDCGuard(recompute=True, raise_on_alert=True).protect(net)
    try:
        with pytest.raises(SDCError, match="recompute"):
            with g.on_step(0):
                _step(net)
    finally:
        g.unprotect()


def _raw(t):
    return t.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def test_fingerprint_sees_three_neighbouring_bf16_changes():
    """+1, -2, +1 in the last bits of three neighbours: the plain sum and any weights linear in the position
    (the earlier column and row weights) cancel; pseudo-random weights don't."""
    torch.manual_seed(0)
    x = torch.randn(1000).bfloat16()
    y = x.clone()
    r = _raw(y)
    r[10] += 1
    r[11] -= 2
    r[12] += 1
    assert not torch.equal(x, y)
    assert not torch.equal(fingerprint(x), fingerprint(y))


def test_fingerprint_sees_a_swap_of_two_fp32_values():
    """1.0 and 0.99609381, whose two 16-bit words add up to the same number, swapped 597 places apart."""
    x = torch.zeros(1000)
    _raw(x)[3] = 0x3F800000                       # 1.0
    _raw(x)[600] = 0x3F7F0001                     # 0.99609381
    y = x.clone()
    y[3], y[600] = x[600], x[3]
    assert not torch.equal(x, y)
    assert not torch.equal(fingerprint(x), fingerprint(y))


def test_fingerprint_sees_a_swap_far_apart():
    """A swap of two elements 251 x 65521 apart (the same column and row weight under the earlier weights)."""
    j = 251 * 65521
    x = torch.zeros(j + 251, dtype=torch.bfloat16)  # (33 MB)
    x[0] = 1.0
    y = x.clone()
    y[0], y[j] = x[j], x[0]
    assert not torch.equal(x, y)
    assert not torch.equal(fingerprint(x), fingerprint(y))


def test_fingerprint_sees_any_single_changed_fp64_element():
    """One fp64 element whose four 16-bit words change by +1, -2, +1 and 0."""
    x = torch.ones(100, dtype=torch.float64)
    _raw(x)[5] = 0x3FF0000000020000
    y = x.clone()
    _raw(y)[5] = 0x3FF0000100000001
    assert not torch.equal(x, y)
    assert not torch.equal(fingerprint(x), fingerprint(y))


class _ThreeBits(nn.Module):
    """A checkpointed block whose recompute differs from its forward in three neighbouring bf16 outputs."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(64, 64).bfloat16()
        self.calls = 0

    def forward(self, x):
        # the same operations on every call (checkpointing needs the recompute to save the same tensors)
        self.calls += 1
        h = self.lin(x)
        with torch.no_grad():
            corr = torch.zeros_like(h)
            if self.calls == 2:                    # the recompute: +1, -2, +1 in three neighbours' last bits
                r = h.detach().reshape(-1).view(torch.int16).clone()
                r[20] += 1
                r[21] -= 2
                r[22] += 1
                corr = (r.view(torch.bfloat16).view_as(h) - h.detach())    # exact: 1-2 ulp differences
        return h + corr


def test_recompute_check_reports_a_recompute_that_computed_different_bits():
    """End to end: the two runs of the block differ in 3 bits, so the check reports one mismatch."""
    torch.manual_seed(0)
    blk = _ThreeBits()
    x = torch.randn(4, 64).bfloat16().requires_grad_()
    seen = []
    with RecomputeCheck(on_alert=seen.append) as rc:
        tuc.checkpoint(blk, x, use_reentrant=False).float().sum().backward()
        rc.flush()
    assert blk.calls == 2 and rc.stats["recomputes_compared"] == 1
    assert len(seen) == 1, "the two runs gave different bits, but no mismatch was reported"


def test_fingerprint_weights_are_odd_and_differ_by_position():
    """Odd weights: a single changed word changes both weighted sums too, whatever the change."""
    from sdcguard.recompute import _fp_weights
    w1, w2 = _fp_weights(torch.arange(100_000, dtype=torch.int64))
    assert bool((w1 & 1).all()) and bool((w2 & 1).all())
    assert w1.unique().numel() == w2.unique().numel() == 100_000
