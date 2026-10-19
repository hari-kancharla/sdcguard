"""Precise mode: linear layers' fp32 accumulators are checked before they're rounded."""
import math

import pytest
import torch
import torch.nn as nn

from sdcguard import FaultInjector, SDCGuard
from sdcguard import checksum as cs
from sdcguard import guard as gmod


def _model(bias=True):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(256, 512, bias=bias), nn.GELU(), nn.Linear(512, 256, bias=bias))


def _x(seed=0, shape=(64, 256)):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _step(model, x, guard=None, step=0):
    ctx = guard.on_step(step) if guard is not None else None
    if ctx is not None:
        ctx.__enter__()
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(x).float().pow(2).mean()
        loss.backward()           # inside autocast on purpose: training loops often do this
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    return loss.detach(), grads


@pytest.mark.parametrize("bias", [True, False])
def test_every_check_is_an_fp32_check_and_clean(bias):
    model = _model(bias)
    zs = []
    g = SDCGuard(every=1, precise=True, on_result=lambda d, r: zs.append(r.max_z)).protect(model)
    for seed in range(3):
        _step(model, _x(seed), g, seed)
    assert g.stats["checked"] == g.stats["checked_fp32"] == 15          # forward x2, dX x1, dW x2, per step
    assert g.stats["alerts"] == 0 and g.stats["unjudged"] == 0
    assert 0 < max(zs) < 1.0


def test_three_dim_inputs():
    model = _model()
    g = SDCGuard(every=1, precise=True).protect(model)
    _step(model, _x(1, (4, 16, 256)), g)
    assert g.stats["checked_fp32"] == 5 and g.stats["alerts"] == 0


def test_outputs_and_gradients_match_a_plain_step():
    """Same fp32 result rounded once: the model sees what cuBLAS's own rounding would give (on CPUs the
    kernels differ, so allow a bf16 step)."""
    model = _model()
    x = _x(2)
    loss0, g0 = _step(model, x)
    g = SDCGuard(every=1, precise=True).protect(model)
    loss1, g1 = _step(model, x, g)
    assert torch.allclose(loss0, loss1, rtol=1e-3)
    for a, b in zip(g0, g1):
        assert (a - b).abs().max() <= 2.0 ** -6 * a.abs().max()


def test_unchecked_steps_are_untouched():
    model = _model()
    x = _x(3)
    loss0, g0 = _step(model, x)
    g = SDCGuard(every=10, precise=True).protect(model)
    loss1, g1 = _step(model, x, g, step=3)                              # not a checked step
    assert torch.equal(loss0, loss1) and all(torch.equal(a, b) for a, b in zip(g0, g1))
    assert g.stats["checked"] == 0


def test_injected_faults_reach_the_accumulator_and_are_caught():
    model = _model()
    inj = FaultInjector(rate=1.0, seed=8).arm_on([model[2]])
    g = SDCGuard(every=1, precise=True, on_accumulator=inj.corrupt_tensor).protect(model)
    _step(model, _x(0), g)
    inj.remove_hooks()
    assert len(inj.events) == 1 and inj.events[0]["op"] == "accumulator of 2"
    assert g.stats["alerts"] == 1 and "float32 accumulator" in g.log.events[0].detail["dtype"]
    assert g.log.events[0].detail["where"] == "2"


def test_a_tiny_error_is_caught_that_bf16_rounding_would_hide():
    """An error of 0.1% of a typical output in one value: the precise check flags it; the ordinary check of
    the same result rounded to bf16 can't see it."""
    model = _model()
    seen = {}

    def nudge(layer, pass_, c):
        if layer == "2" and pass_ == "forward":
            typical = float(c.norm()) / math.sqrt(c.numel())
            c[10, 100] += 1e-3 * typical
            seen["c"] = c.clone()

    g = SDCGuard(every=1, precise=True, on_accumulator=nudge).protect(model)
    _step(model, _x(4), g)
    assert g.stats["alerts"] == 1 and g.log.events[0].detail["where"] == "2"
    assert g.log.events[0].detail["row"] == 10 and g.log.events[0].detail["col_block"] == 0
    # the same nudge, seen through bf16 rounding
    torch.manual_seed(0)
    x = _x(4)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        h = model[1](model[0](x))
    w, b = model[2].weight.bfloat16(), model[2].bias.bfloat16()
    old = cs.USE_FAST
    cs.USE_FAST = False
    try:
        r = cs.check_matmul(h.bfloat16(), w.t(), seen["c"].bfloat16(), b)
    finally:
        cs.USE_FAST = old
    assert r.ok


def _tensor_core(nfma, frac_bits):
    """Tensor-core rounding (Khattak and Mikaitis, arXiv 2512.07004): nfma exact products plus the
    accumulator per step, each truncated to frac_bits below the largest exponent, the sum truncated to fp32."""
    def rtz32(v):
        y = v.float()
        return torch.where(y.double().abs() > v.abs(), torch.nextafter(y, torch.zeros_like(y)), y)

    def mm(a, b):
        a, b = a.double(), b.double()
        acc = torch.zeros(a.shape[0], b.shape[1], dtype=torch.float32)
        for s in range(0, a.shape[1], nfma):
            terms = torch.cat([a[:, s:s + nfma, None] * b[None, s:s + nfma, :], acc.double()[:, None, :]], dim=1)
            e = torch.frexp(terms.abs().amax(dim=1))[1].double()
            q = torch.pow(2.0, e - 1 - frac_bits)[:, None, :]
            acc = rtz32((torch.trunc(terms / q) * q).sum(dim=1))
        return acc
    return mm


@pytest.mark.parametrize("nfma,frac", [(8, 24), (16, 25)])
def test_tensor_core_rounding_stays_far_from_the_alarm(monkeypatch, nfma, frac):
    monkeypatch.setattr(gmod, "_acc_mm_raw", _tensor_core(nfma, frac))
    monkeypatch.setattr(SDCGuard, "_tensor_core_steps", lambda self, c, dt: nfma)
    model = _model()
    zs = []
    g = SDCGuard(every=1, precise=True, on_result=lambda d, r: zs.append(r.max_z)).protect(model)
    _step(model, _x(5), g)
    assert g.stats["checked_fp32"] == 5 and g.stats["alerts"] == 0 and max(zs) < 1.5


def test_nan_inputs_are_unjudged_not_clean():
    model = _model()
    g = SDCGuard(every=1, precise=True).protect(model)
    with g.on_step(0):
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            model(torch.full((4, 256), float("nan")))
    assert g.stats["checked_fp32"] == 2 and g.stats["alerts"] == 0 and g.stats["unjudged"] == 2
